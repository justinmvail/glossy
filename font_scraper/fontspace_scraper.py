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

from font_source import FontMetadata, FontSource, ScraperConfig


class FontSpaceScraper(FontSource):
    """Scrape handwriting fonts from FontSpace.com.

    This class handles the complete workflow of discovering fonts from FontSpace
    search results or category pages, fetching download URLs from individual
    font pages, and downloading font files.

    Inherits from FontSource to provide a consistent interface with other
    font scrapers (DaFont, Google Fonts).

    Attributes:
        SOURCE_NAME (str): Identifier for this source ('fontspace').
        BASE_URL (str): The base URL for FontSpace.com.
        output_dir (Path): Directory where downloaded fonts are saved.
        rate_limit (float): Delay in seconds between HTTP requests.
        session (requests.Session): HTTP session for making requests.
        fonts_found (list): List of all FontMetadata discovered during scraping.
        downloaded (Set[str]): Set of font names that were successfully downloaded.
        failed (List[str]): List of font names that failed to download.
    """

    SOURCE_NAME = "fontspace"
    BASE_URL = "https://www.fontspace.com"

    # HTTP timeout constants (seconds)
    PAGE_TIMEOUT = 30
    DOWNLOAD_TIMEOUT = 60

    def __init__(self, output_dir: str, rate_limit: float = 1.0):
        """Initialize the FontSpace scraper.

        Args:
            output_dir: Directory path where downloaded fonts will be saved.
                The directory will be created if it does not exist.
            rate_limit: Minimum seconds to wait between HTTP requests to avoid
                overwhelming the server. Defaults to 1.0 second.
        """
        super().__init__(output_dir, rate_limit)

    def scrape_search(self, query: str = "handwritten", max_pages: int = 10) -> list[FontMetadata]:
        """Scrape fonts from FontSpace search results.

        Performs a search query and iterates through paginated results,
        extracting font information from each page.

        Args:
            query: The search query string to find fonts. Defaults to "handwritten".
            max_pages: Maximum number of result pages to scrape. Defaults to 10.

        Returns:
            A list of FontMetadata objects representing all fonts found in the
            search results across all scraped pages.
        """
        logger.info("Searching FontSpace for: %s", query)

        fonts = []
        page = 1

        while page <= max_pages:
            url = f"{self.BASE_URL}/search?q={query}&p={page}"
            logger.debug("Page %d: %s", page, url)

            try:
                resp = self.get_with_retry(url, timeout=self.PAGE_TIMEOUT)
                resp.raise_for_status()

                page_fonts = self._parse_search_results(resp.text)

                if not page_fonts:
                    logger.debug("No fonts found on page %d, stopping.", page)
                    break

                fonts.extend(page_fonts)
                logger.debug("Found %d fonts (total: %d)", len(page_fonts), len(fonts))

                page += 1
                time.sleep(self.rate_limit)

            except requests.RequestException as e:
                logger.warning("Error fetching page %d: %s", page, e)
                break

        return fonts

    def scrape_category(self, category: str = "handwriting", max_pages: int = 10) -> list[FontMetadata]:
        """Scrape fonts from a FontSpace category.

        Browses a specific category and iterates through paginated results,
        extracting font information from each page.

        Args:
            category: The category slug to browse (e.g., "handwriting", "script").
                Defaults to "handwriting".
            max_pages: Maximum number of category pages to scrape. Defaults to 10.

        Returns:
            A list of FontMetadata objects representing all fonts found in the
            category across all scraped pages.
        """
        logger.info("Scraping FontSpace category: %s", category)

        fonts = []
        page = 1

        while page <= max_pages:
            url = f"{self.BASE_URL}/category/{category}?p={page}"
            logger.debug("Page %d: %s", page, url)

            try:
                resp = self.get_with_retry(url, timeout=self.PAGE_TIMEOUT)
                resp.raise_for_status()

                page_fonts = self._parse_search_results(resp.text)

                if not page_fonts:
                    logger.debug("No fonts found on page %d, stopping.", page)
                    break

                fonts.extend(page_fonts)
                logger.debug("Found %d fonts (total: %d)", len(page_fonts), len(fonts))

                page += 1
                time.sleep(self.rate_limit)

            except requests.RequestException as e:
                logger.warning("Error fetching page %d: %s", page, e)
                break

        return fonts

    def _parse_search_results(self, html: str) -> list[FontMetadata]:
        """Parse font list from search or category page HTML.

        Extracts font information by finding links to font pages in the HTML.
        Automatically deduplicates fonts by URL.

        Args:
            html: The raw HTML content of the search/category page.

        Returns:
            A list of unique FontMetadata objects parsed from the page.
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

                fonts.append(FontMetadata(
                    name=font_name,
                    url=font_url,
                    download_url='',  # Will be fetched per-font
                    source=self.SOURCE_NAME,
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
            resp = self.get_with_retry(font_url, timeout=self.PAGE_TIMEOUT)
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
            logger.warning("Error getting download URL for %s: %s", font_url, e)

        return ''

    def download_font(self, font: FontMetadata) -> bool:
        """Download a font and extract TTF/OTF files.

        Fetches the download URL if not already known, downloads the font
        file (either a ZIP or direct font file), and saves it to the output
        directory.

        Args:
            font: FontMetadata object containing the font's URL and metadata.
                The download_url may be empty and will be fetched if needed.

        Returns:
            True if the font was successfully downloaded, False otherwise.
        """
        if font.name in self.downloaded:
            return True

        try:
            logger.debug("Downloading: %s", font.name)

            # Get download URL if we don't have it
            if not font.download_url:
                font.download_url = self._get_download_url(font.url)
                time.sleep(0.5)

            if not font.download_url:
                logger.warning("Could not find download URL for %s", font.name)
                self.failed.append(font.name)
                return False

            resp = self.get_with_retry(font.download_url, timeout=self.DOWNLOAD_TIMEOUT)
            resp.raise_for_status()

            # Check if it's a ZIP file
            if resp.content[:4] == b'PK\x03\x04':
                extracted = self._extract_zip(resp.content)
                if extracted > 0:
                    logger.debug("Extracted %d font file(s) for %s", extracted, font.name)
                    self.downloaded.add(font.name)
                    return True
                else:
                    logger.warning("No TTF/OTF files in ZIP for %s", font.name)
                    return False
            else:
                # Might be a direct font file
                content_type = resp.headers.get('content-type', '')
                if 'font' in content_type or font.download_url.endswith(('.ttf', '.otf')):
                    ext = '.ttf' if '.ttf' in font.download_url else '.otf'
                    safe_name = self.safe_filename(font.name) + ext
                    out_path = self.output_dir / safe_name
                    out_path.write_bytes(resp.content)
                    logger.debug("Saved %s", safe_name)
                    self.downloaded.add(font.name)
                    return True

            logger.warning("Unknown file format for %s", font.name)
            return False

        except Exception as e:
            logger.error("Error downloading %s: %s", font.name, e)
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
                        safe_name = self.safe_filename(os.path.basename(name))
                        out_path = self.output_dir / safe_name
                        if not out_path.exists():
                            with zf.open(name) as src, open(out_path, 'wb') as dst:
                                dst.write(src.read())
                            extracted += 1
        except zipfile.BadZipFile as e:
            logger.debug("Bad zip file: %s", e)
        return extracted

    def scrape_fonts(self, config: ScraperConfig) -> list[FontMetadata]:
        """Discover fonts from FontSpace.

        Scrapes fonts via search or category browsing based on config.

        Args:
            config: ScraperConfig with options. Uses config.query for search,
                config.use_category to browse by category instead.

        Returns:
            List of FontMetadata objects for discovered fonts.
        """
        query = config.query or "handwritten"
        max_pages = config.max_pages

        if config.use_category:
            return self.scrape_category(query, max_pages)
        else:
            return self.scrape_search(query, max_pages)

    def scrape_and_download(
        self,
        config: ScraperConfig | None = None,
        query: str = "handwritten",
        max_pages: int = 10,
        max_fonts: int = None,
        use_category: bool = False
    ) -> dict:
        """Scrape and download fonts from FontSpace.

        This is the main entry point for the scraper. It coordinates the
        complete workflow of discovering fonts via search or category,
        and downloading font files.

        Supports both the new ScraperConfig interface and the legacy
        keyword arguments for backward compatibility.

        Args:
            config: ScraperConfig with options. If provided, other arguments
                are ignored.
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
                - fonts_downloaded/downloaded (int): Number of fonts successfully downloaded.
                - fonts_failed/failed (int): Number of fonts that failed to download.
                - font_list/fonts (list): List of font dictionaries with metadata.
        """
        # Support ScraperConfig
        if config is not None:
            return super().scrape_and_download(config)

        logger.info("=" * 60)
        logger.info("FontSpace Scraper")
        logger.info("=" * 60)
        logger.info("Output directory: %s", self.output_dir)
        logger.info("Query/Category: %s", query)
        logger.info("Max pages: %d", max_pages)

        # Scrape
        if use_category:
            fonts = self.scrape_category(query, max_pages)
        else:
            fonts = self.scrape_search(query, max_pages)

        self.fonts_found = fonts

        logger.info("Total fonts found: %d", len(fonts))

        # Limit if specified
        if max_fonts:
            fonts = fonts[:max_fonts]
            logger.info("Limiting to %d fonts", max_fonts)

        # Download
        logger.info("Downloading %d fonts...", len(fonts))

        for i, font in enumerate(fonts):
            logger.debug("[%d/%d] Processing %s", i + 1, len(fonts), font.name)
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

        logger.info("=" * 60)
        logger.info("COMPLETE")
        logger.info("=" * 60)
        logger.info("Fonts found: %d", len(self.fonts_found))
        logger.info("Fonts downloaded: %d", len(self.downloaded))
        logger.info("Failed: %d", len(self.failed))

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
