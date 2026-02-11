"""FontSpace Scraper - Download handwriting fonts from FontSpace.com.

This module provides a web scraper for downloading fonts from FontSpace.com.
It supports both search-based and category-based font discovery, handles
pagination, and extracts TTF/OTF font files from ZIP downloads or direct
font file links.

The scraper fetches download URLs from individual font pages since FontSpace
requires visiting each font's page to obtain the actual download link.

Example:
    Basic usage with search query::

        $ python fontspace_scraper.py --output ./fonts --pages 10

    Using category-based scraping::

        $ python fontspace_scraper.py --output ./fonts --category --query handwriting

    Programmatic usage::

        scraper = FontSpaceScraper('./output', rate_limit=1.0)
        metadata = scraper.scrape_and_download(
            query='handwritten',
            max_pages=10,
            max_fonts=100,
            use_category=False
        )

Command-line Arguments:
    --output, -o: Output directory for downloaded fonts (default: ./fontspace_fonts)
    --query, -q: Search query or category name (default: handwritten)
    --pages, -p: Maximum pages to scrape (default: 10)
    --max-fonts, -m: Maximum fonts to download (default: unlimited)
    --category, -c: Use category browsing instead of search
    --rate-limit, -r: Seconds to wait between requests (default: 1.0)
"""

import argparse
import io
import json
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

import requests
from bs4 import BeautifulSoup


@dataclass
class FontInfo:
    """Information about a font from FontSpace.

    Attributes:
        name: The display name of the font.
        url: The URL to the font's detail page on FontSpace.
        download_url: The direct download URL for the font file or ZIP.
            May be empty initially and populated when visiting the font page.
        downloads: The number of downloads (may be 0 if not parsed).
    """
    name: str
    url: str
    download_url: str
    downloads: int = 0

    def to_dict(self):
        """Convert the FontInfo to a dictionary for JSON serialization.

        Returns:
            dict: A dictionary containing all font information fields.
        """
        return {
            'name': self.name,
            'url': self.url,
            'download_url': self.download_url,
            'downloads': self.downloads
        }


class FontSpaceScraper:
    """Scrape handwriting fonts from FontSpace.com.

    This class handles the complete workflow of discovering fonts from FontSpace
    search results or category pages, fetching download URLs from individual
    font pages, and downloading font files.

    Attributes:
        BASE_URL (str): The base URL for FontSpace.com.
        output_dir (Path): Directory where downloaded fonts are saved.
        rate_limit (float): Delay in seconds between HTTP requests.
        session (requests.Session): HTTP session for making requests.
        fonts_found (List[FontInfo]): List of all fonts discovered during scraping.
        downloaded (Set[str]): Set of font names that were successfully downloaded.
        failed (List[str]): List of font names that failed to download.
    """

    BASE_URL = "https://www.fontspace.com"

    def __init__(self, output_dir: str, rate_limit: float = 1.0):
        """Initialize the FontSpace scraper.

        Args:
            output_dir: Directory path where downloaded fonts will be saved.
                The directory will be created if it does not exist.
            rate_limit: Minimum seconds to wait between HTTP requests to avoid
                overwhelming the server. Defaults to 1.0 second.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        })

        self.fonts_found: list[FontInfo] = []
        self.downloaded: set[str] = set()
        self.failed: list[str] = []

    def scrape_search(self, query: str = "handwritten", max_pages: int = 10) -> list[FontInfo]:
        """Scrape fonts from FontSpace search results.

        Performs a search query and iterates through paginated results,
        extracting font information from each page.

        Args:
            query: The search query string to find fonts. Defaults to "handwritten".
            max_pages: Maximum number of result pages to scrape. Defaults to 10.

        Returns:
            A list of FontInfo objects representing all fonts found in the
            search results across all scraped pages.
        """
        print(f"\nSearching FontSpace for: {query}")

        fonts = []
        page = 1

        while page <= max_pages:
            url = f"{self.BASE_URL}/search?q={query}&p={page}"
            print(f"  Page {page}: {url}")

            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()

                page_fonts = self._parse_search_results(resp.text)

                if not page_fonts:
                    print(f"  No fonts found on page {page}, stopping.")
                    break

                fonts.extend(page_fonts)
                print(f"  Found {len(page_fonts)} fonts (total: {len(fonts)})")

                page += 1
                time.sleep(self.rate_limit)

            except requests.RequestException as e:
                print(f"  Error fetching page {page}: {e}")
                break

        return fonts

    def scrape_category(self, category: str = "handwriting", max_pages: int = 10) -> list[FontInfo]:
        """Scrape fonts from a FontSpace category.

        Browses a specific category and iterates through paginated results,
        extracting font information from each page.

        Args:
            category: The category slug to browse (e.g., "handwriting", "script").
                Defaults to "handwriting".
            max_pages: Maximum number of category pages to scrape. Defaults to 10.

        Returns:
            A list of FontInfo objects representing all fonts found in the
            category across all scraped pages.
        """
        print(f"\nScraping FontSpace category: {category}")

        fonts = []
        page = 1

        while page <= max_pages:
            url = f"{self.BASE_URL}/category/{category}?p={page}"
            print(f"  Page {page}: {url}")

            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()

                page_fonts = self._parse_search_results(resp.text)

                if not page_fonts:
                    print(f"  No fonts found on page {page}, stopping.")
                    break

                fonts.extend(page_fonts)
                print(f"  Found {len(page_fonts)} fonts (total: {len(fonts)})")

                page += 1
                time.sleep(self.rate_limit)

            except requests.RequestException as e:
                print(f"  Error fetching page {page}: {e}")
                break

        return fonts

    def _parse_search_results(self, html: str) -> list[FontInfo]:
        """Parse font list from search or category page HTML.

        Extracts font information by finding links to font pages in the HTML.
        Automatically deduplicates fonts by URL.

        Args:
            html: The raw HTML content of the search/category page.

        Returns:
            A list of unique FontInfo objects parsed from the page.
            Download URLs are left empty and fetched later per-font.
        """
        soup = BeautifulSoup(html, 'html.parser')
        fonts = []

        # FontSpace uses font cards with links
        # Look for font links - they typically have /font/ in the URL
        for link in soup.find_all('a', href=re.compile(r'/font/[^/]+$')):
            try:
                font_name = link.get_text(strip=True)
                font_url = link.get('href', '')

                if not font_name or len(font_name) < 2:
                    continue

                # Skip navigation/UI links
                if font_name.lower() in ['download', 'preview', 'share', 'embed']:
                    continue

                if font_url and not font_url.startswith('http'):
                    font_url = urljoin(self.BASE_URL, font_url)

                # Build download URL from font page URL
                # FontSpace download: /font/fontname -> click download button
                # We'll get the actual download URL when we visit the font page

                fonts.append(FontInfo(
                    name=font_name,
                    url=font_url,
                    download_url='',  # Will be fetched per-font
                    downloads=0
                ))

            except Exception:
                continue

        # Deduplicate by URL
        seen = set()
        unique_fonts = []
        for f in fonts:
            if f.url not in seen:
                seen.add(f.url)
                unique_fonts.append(f)

        return unique_fonts

    def _get_download_url(self, font_url: str) -> str:
        """Fetch the actual download URL from a font's detail page.

        Visits the font's page on FontSpace and extracts the download link
        from the page HTML.

        Args:
            font_url: The URL to the font's detail page on FontSpace.

        Returns:
            The direct download URL for the font file or ZIP, or an empty
            string if the download URL could not be found.
        """
        try:
            resp = self.session.get(font_url, timeout=30)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Look for download button/link
            dl_link = soup.find('a', href=re.compile(r'/download/'))
            if dl_link:
                url = dl_link.get('href', '')
                if url and not url.startswith('http'):
                    url = urljoin(self.BASE_URL, url)
                return url

            # Alternative: look for direct ZIP link
            zip_link = soup.find('a', href=re.compile(r'\.zip'))
            if zip_link:
                url = zip_link.get('href', '')
                if url and not url.startswith('http'):
                    url = urljoin(self.BASE_URL, url)
                return url

        except Exception as e:
            print(f"    Error getting download URL: {e}")

        return ''

    def download_font(self, font: FontInfo) -> bool:
        """Download a font and extract TTF/OTF files.

        Fetches the download URL if not already known, downloads the font
        file (either a ZIP or direct font file), and saves it to the output
        directory.

        Args:
            font: FontInfo object containing the font's URL and metadata.
                The download_url may be empty and will be fetched if needed.

        Returns:
            True if the font was successfully downloaded, False otherwise.
        """
        if font.name in self.downloaded:
            return True

        try:
            print(f"  Downloading: {font.name}")

            # Get download URL if we don't have it
            if not font.download_url:
                font.download_url = self._get_download_url(font.url)
                time.sleep(0.5)

            if not font.download_url:
                print("    Could not find download URL")
                self.failed.append(font.name)
                return False

            resp = self.session.get(font.download_url, timeout=60)
            resp.raise_for_status()

            # Check if it's a ZIP file
            if resp.content[:4] == b'PK\x03\x04':
                extracted = self._extract_zip(resp.content)
                if extracted > 0:
                    print(f"    Extracted {extracted} font file(s)")
                    self.downloaded.add(font.name)
                    return True
                else:
                    print("    No TTF/OTF files in ZIP")
                    return False
            else:
                # Might be a direct font file
                content_type = resp.headers.get('content-type', '')
                if 'font' in content_type or font.download_url.endswith(('.ttf', '.otf')):
                    ext = '.ttf' if '.ttf' in font.download_url else '.otf'
                    safe_name = self._safe_filename(font.name) + ext
                    out_path = self.output_dir / safe_name
                    out_path.write_bytes(resp.content)
                    print(f"    Saved {safe_name}")
                    self.downloaded.add(font.name)
                    return True

            print("    Unknown file format")
            return False

        except Exception as e:
            print(f"    Error: {e}")
            self.failed.append(font.name)
            return False

    def _extract_zip(self, content: bytes) -> int:
        """Extract font files from ZIP content.

        Extracts TTF and OTF files from a ZIP archive, saving them to the
        output directory with sanitized filenames.

        Args:
            content: The raw bytes of the ZIP file.

        Returns:
            The number of font files successfully extracted.
        """
        extracted = 0
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for name in zf.namelist():
                    lower_name = name.lower()
                    if lower_name.endswith(('.ttf', '.otf')):
                        safe_name = self._safe_filename(os.path.basename(name))
                        out_path = self.output_dir / safe_name
                        if not out_path.exists():
                            with zf.open(name) as src, open(out_path, 'wb') as dst:
                                dst.write(src.read())
                            extracted += 1
        except zipfile.BadZipFile as e:
            logger.debug("Bad zip file: %s", e)
        return extracted

    def _safe_filename(self, name: str) -> str:
        """Create a safe filename by removing problematic characters.

        Args:
            name: The original filename that may contain unsafe characters.

        Returns:
            A sanitized filename safe for use on most filesystems.
            Returns 'unnamed_font' if the result would be empty.
        """
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe = safe.strip('. ')
        return safe or 'unnamed_font'

    def scrape_and_download(
        self,
        query: str = "handwritten",
        max_pages: int = 10,
        max_fonts: int = None,
        use_category: bool = False
    ) -> dict:
        """Scrape and download fonts from FontSpace.

        This is the main entry point for the scraper. It coordinates the
        complete workflow of discovering fonts via search or category,
        and downloading font files.

        Args:
            query: The search query or category name to use for finding fonts.
                Defaults to "handwritten".
            max_pages: Maximum number of pages to scrape. Defaults to 10.
            max_fonts: Maximum number of fonts to download. If None, downloads
                all discovered fonts.
            use_category: If True, browse by category instead of searching.
                The query parameter is then interpreted as a category slug.
                Defaults to False.

        Returns:
            A dictionary containing scraping statistics and metadata:
                - source (str): Always "fontspace".
                - query (str): The search query or category used.
                - fonts_found (int): Total number of fonts discovered.
                - fonts_downloaded (int): Number of fonts successfully downloaded.
                - fonts_failed (int): Number of fonts that failed to download.
                - font_list (list): List of font dictionaries with metadata.
        """
        print("=" * 60)
        print("FontSpace Scraper")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Query/Category: {query}")
        print(f"Max pages: {max_pages}")

        # Scrape
        if use_category:
            fonts = self.scrape_category(query, max_pages)
        else:
            fonts = self.scrape_search(query, max_pages)

        self.fonts_found = fonts

        print(f"\nTotal fonts found: {len(fonts)}")

        # Limit if specified
        if max_fonts:
            fonts = fonts[:max_fonts]
            print(f"Limiting to {max_fonts} fonts")

        # Download
        print(f"\nDownloading {len(fonts)} fonts...")

        for i, font in enumerate(fonts):
            print(f"[{i+1}/{len(fonts)}]", end='')
            self.download_font(font)
            time.sleep(self.rate_limit)

        # Save metadata
        metadata = {
            'source': 'fontspace',
            'query': query,
            'fonts_found': len(self.fonts_found),
            'fonts_downloaded': len(self.downloaded),
            'fonts_failed': len(self.failed),
            'font_list': [f.to_dict() for f in fonts]
        }

        meta_path = self.output_dir / 'fontspace_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)
        print(f"Fonts found: {len(self.fonts_found)}")
        print(f"Fonts downloaded: {len(self.downloaded)}")
        print(f"Failed: {len(self.failed)}")

        return metadata


def main() -> None:
    """Parse command-line arguments and run the FontSpace scraper.

    This function serves as the entry point when the module is run as a script.
    It configures argument parsing and initiates the scraping process.
    """
    parser = argparse.ArgumentParser(description='Scrape fonts from FontSpace')
    parser.add_argument('--output', '-o', type=str, default='./fontspace_fonts',
                        help='Output directory')
    parser.add_argument('--query', '-q', type=str, default='handwritten',
                        help='Search query')
    parser.add_argument('--pages', '-p', type=int, default=10,
                        help='Max pages to scrape')
    parser.add_argument('--max-fonts', '-m', type=int, default=None,
                        help='Max fonts to download')
    parser.add_argument('--category', '-c', action='store_true',
                        help='Use category instead of search')
    parser.add_argument('--rate-limit', '-r', type=float, default=1.0,
                        help='Seconds between requests')

    args = parser.parse_args()

    scraper = FontSpaceScraper(args.output, rate_limit=args.rate_limit)
    scraper.scrape_and_download(
        query=args.query,
        max_pages=args.pages,
        max_fonts=args.max_fonts,
        use_category=args.category
    )


if __name__ == '__main__':
    main()
