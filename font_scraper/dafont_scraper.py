"""DaFont Scraper - Download handwriting fonts from DaFont.com.

This module provides a web scraper for downloading handwriting and calligraphy
fonts from DaFont.com. It supports scraping multiple categories, handles
pagination, and extracts TTF/OTF font files from ZIP downloads.

The scraper respects rate limiting to avoid overwhelming the server and
maintains metadata about downloaded fonts for tracking purposes.

Categories:
    - cat=601: Calligraphy fonts
    - cat=603: Handwritten fonts

Example:
    Basic usage with default settings::

        $ python dafont_scraper.py --output ./fonts --pages 10

    Download from specific categories with a font limit::

        $ python dafont_scraper.py --output ./fonts --categories 601 603 --pages 20

    Programmatic usage::

        scraper = DaFontScraper('./output', rate_limit=1.0)
        metadata = scraper.scrape_and_download(
            categories=['601', '603'],
            max_pages=10,
            max_fonts=100
        )

Attributes:
    BASE_URL (str): The base URL for DaFont.com.
    CATEGORIES (dict): Mapping of category IDs to human-readable names.

Command-line Arguments:
    --output, -o: Output directory for downloaded fonts (default: ./dafont_fonts)
    --categories, -c: Category IDs to scrape (default: 601 603)
    --pages, -p: Maximum pages per category to scrape (default: 10)
    --max-fonts, -m: Maximum total fonts to download (default: unlimited)
    --rate-limit, -r: Seconds to wait between requests (default: 1.0)
"""

import argparse
import io
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


@dataclass
class FontInfo:
    """Information about a font from DaFont.

    Attributes:
        name: The display name of the font.
        url: The URL to the font's detail page on DaFont.
        download_url: The direct download URL for the font ZIP file.
        category: The category name this font belongs to.
        downloads: The number of downloads reported on DaFont (may be 0 if not parsed).
    """
    name: str
    url: str
    download_url: str
    category: str
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
            'category': self.category,
            'downloads': self.downloads
        }


class DaFontScraper:
    """Scrape handwriting fonts from DaFont.com.

    This class handles the complete workflow of discovering fonts from DaFont
    category pages, parsing font information, downloading ZIP files, and
    extracting TTF/OTF font files.

    Attributes:
        BASE_URL (str): The base URL for DaFont.com.
        CATEGORIES (dict): Mapping of category IDs to human-readable names.
            Keys are string category IDs (e.g., '601'), values are category
            names (e.g., 'Calligraphy').
        output_dir (Path): Directory where downloaded fonts are saved.
        rate_limit (float): Delay in seconds between HTTP requests.
        session (requests.Session): HTTP session for making requests.
        fonts_found (List[FontInfo]): List of all fonts discovered during scraping.
        downloaded (Set[str]): Set of font names that were successfully downloaded.
        failed (List[str]): List of font names that failed to download.
    """

    BASE_URL = "https://www.dafont.com"

    # Handwriting-related categories
    CATEGORIES = {
        '601': 'Calligraphy',
        '603': 'Handwritten',
    }

    def __init__(self, output_dir: str, rate_limit: float = 1.0):
        """Initialize the DaFont scraper.

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
            'Accept-Language': 'en-US,en;q=0.5',
        })

        self.fonts_found: list[FontInfo] = []
        self.downloaded: set[str] = set()
        self.failed: list[str] = []

    def scrape_category(self, category: str, max_pages: int = 10) -> list[FontInfo]:
        """Scrape fonts from a specific DaFont category.

        Iterates through paginated category listings, parsing font information
        from each page until no more fonts are found or max_pages is reached.

        Args:
            category: The DaFont category ID (e.g., '601' for Calligraphy,
                '603' for Handwritten).
            max_pages: Maximum number of pages to scrape from this category.
                Defaults to 10.

        Returns:
            A list of FontInfo objects representing all fonts found in the
            category across all scraped pages.
        """
        category_name = self.CATEGORIES.get(category, f'Unknown ({category})')
        print(f"\nScraping category: {category_name} (cat={category})")

        fonts = []
        page = 1

        while page <= max_pages:
            url = f"{self.BASE_URL}/theme.php?cat={category}&page={page}"
            print(f"  Page {page}: {url}")

            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()

                page_fonts = self._parse_font_list(resp.text, category_name)

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

    def _parse_font_list(self, html: str, category: str) -> list[FontInfo]:
        """Parse font list from category page HTML.

        Extracts font information from the HTML of a DaFont category page.
        Uses multiple parsing strategies to handle different page layouts.

        Args:
            html: The raw HTML content of the category page.
            category: The human-readable category name to assign to parsed fonts.

        Returns:
            A list of FontInfo objects parsed from the page. Returns an empty
            list if no fonts could be parsed.
        """
        soup = BeautifulSoup(html, 'html.parser')
        fonts = []

        # Method 1: Find font name links (end with .font)
        font_links = soup.find_all('a', href=re.compile(r'\.font$'))

        for font_link in font_links:
            try:
                font_name = font_link.get_text(strip=True)
                font_url = font_link.get('href', '')

                if not font_name or font_name.lower() in ['download', 'donate']:
                    continue

                if font_url and not font_url.startswith('http'):
                    font_url = urljoin(self.BASE_URL, font_url)

                # Build download URL from font slug
                # Font URL: https://www.dafont.com/font-name.font
                # Download URL: https://dl.dafont.com/dl/?f=font_name
                font_slug = font_url.replace('.font', '').split('/')[-1]
                # Convert hyphens to underscores for download URL
                slug_underscore = font_slug.replace('-', '_')
                download_url = f"https://dl.dafont.com/dl/?f={slug_underscore}"

                # Get download count from nearby text
                downloads = 0
                # Look for download count in sibling/parent text
                next_text = font_link.find_next(string=re.compile(r'[\d,]+\s*downloads'))
                if next_text:
                    dl_match = re.search(r'([\d,]+)\s*downloads', next_text)
                    if dl_match:
                        downloads = int(dl_match.group(1).replace(',', ''))

                fonts.append(FontInfo(
                    name=font_name,
                    url=font_url,
                    download_url=download_url,
                    category=category,
                    downloads=downloads
                ))

            except Exception:
                continue

        # Fallback: parse download links directly
        if not fonts:
            fonts = self._parse_font_list_alt(soup, category)

        return fonts

    def _parse_font_list_alt(self, soup: BeautifulSoup, category: str) -> list[FontInfo]:
        """Alternative parsing method using download URLs.

        This fallback method extracts font information by finding download
        links directly, used when the primary parsing method fails.

        Args:
            soup: BeautifulSoup object of the parsed HTML page.
            category: The human-readable category name to assign to parsed fonts.

        Returns:
            A list of FontInfo objects parsed from download links.
        """
        fonts = []
        seen = set()

        # Look for download links
        for link in soup.find_all('a', href=re.compile(r'dl\.dafont\.com|dl\.php\?f=')):
            try:
                download_url = link.get('href', '')

                # Extract font slug from download URL
                match = re.search(r'[?&]f=([^&]+)', download_url)
                if not match:
                    continue

                font_slug = match.group(1)
                if font_slug in seen:
                    continue
                seen.add(font_slug)

                # Convert slug to readable name
                font_name = font_slug.replace('_', ' ').title()

                if download_url.startswith('//'):
                    download_url = 'https:' + download_url
                elif not download_url.startswith('http'):
                    download_url = urljoin(self.BASE_URL, download_url)

                # Build font page URL from slug
                font_url = urljoin(self.BASE_URL, font_slug.replace('_', '-') + '.font')

                fonts.append(FontInfo(
                    name=font_name,
                    url=font_url,
                    download_url=download_url,
                    category=category,
                    downloads=0
                ))

            except Exception:
                continue

        return fonts

    def download_font(self, font: FontInfo) -> bool:
        """Download a font and extract TTF/OTF files from the ZIP archive.

        Downloads the font ZIP file from DaFont, extracts any TTF or OTF
        font files, and saves them to the output directory.

        Args:
            font: FontInfo object containing the font's download URL and metadata.

        Returns:
            True if the font was successfully downloaded and at least one
            font file was extracted, False otherwise.
        """
        if font.name in self.downloaded:
            return True

        try:
            print(f"  Downloading: {font.name}")

            resp = self.session.get(font.download_url, timeout=60)
            resp.raise_for_status()

            # DaFont serves ZIP files
            if resp.content[:4] != b'PK\x03\x04':
                print("    Warning: Not a ZIP file")
                return False

            # Extract fonts from ZIP
            extracted = 0
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for name in zf.namelist():
                    lower_name = name.lower()
                    if lower_name.endswith(('.ttf', '.otf')):
                        # Clean filename
                        safe_name = self._safe_filename(os.path.basename(name))
                        out_path = self.output_dir / safe_name

                        if not out_path.exists():
                            with zf.open(name) as src, open(out_path, 'wb') as dst:
                                dst.write(src.read())
                            extracted += 1

            if extracted > 0:
                print(f"    Extracted {extracted} font file(s)")
                self.downloaded.add(font.name)
                return True
            else:
                print("    No TTF/OTF files found in ZIP")
                return False

        except zipfile.BadZipFile:
            print("    Error: Invalid ZIP file")
            self.failed.append(font.name)
            return False
        except requests.RequestException as e:
            print(f"    Error downloading: {e}")
            self.failed.append(font.name)
            return False
        except Exception as e:
            print(f"    Error: {e}")
            self.failed.append(font.name)
            return False

    def _safe_filename(self, name: str) -> str:
        """Create a safe filename by removing problematic characters.

        Args:
            name: The original filename that may contain unsafe characters.

        Returns:
            A sanitized filename safe for use on most filesystems.
            Returns 'unnamed_font' if the result would be empty.
        """
        # Remove/replace problematic characters
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe = safe.strip('. ')
        return safe or 'unnamed_font'

    def scrape_and_download(
        self,
        categories: list[str] = None,
        max_pages: int = 10,
        max_fonts: int = None
    ) -> dict:
        """Scrape and download fonts from specified categories.

        This is the main entry point for the scraper. It coordinates the
        complete workflow of discovering fonts across categories, removing
        duplicates, and downloading font files.

        Args:
            categories: List of DaFont category IDs to scrape (e.g., ['601', '603']).
                If None, defaults to all categories defined in CATEGORIES.
            max_pages: Maximum number of pages to scrape per category.
                Defaults to 10.
            max_fonts: Maximum total number of fonts to download across all
                categories. If None, downloads all discovered fonts.
                Fonts are sorted by download count before limiting.

        Returns:
            A dictionary containing scraping statistics and metadata:
                - fonts_found (int): Total number of fonts discovered.
                - fonts_downloaded (int): Number of fonts successfully downloaded.
                - fonts_failed (int): Number of fonts that failed to download.
                - categories (list): List of category IDs that were scraped.
                - font_list (list): List of font dictionaries with metadata.
        """
        if categories is None:
            categories = list(self.CATEGORIES.keys())

        print("=" * 60)
        print("DaFont Scraper")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Categories: {[self.CATEGORIES.get(c, c) for c in categories]}")
        print(f"Max pages per category: {max_pages}")

        # Scrape all categories
        all_fonts = []
        for cat in categories:
            fonts = self.scrape_category(cat, max_pages)
            all_fonts.extend(fonts)
            self.fonts_found.extend(fonts)

        print(f"\nTotal fonts found: {len(all_fonts)}")

        # Remove duplicates by name
        unique_fonts = {f.name: f for f in all_fonts}
        fonts_to_download = list(unique_fonts.values())

        print(f"Unique fonts: {len(fonts_to_download)}")

        # Sort by downloads (most popular first) if we have that info
        fonts_to_download.sort(key=lambda x: x.downloads, reverse=True)

        # Limit if specified
        if max_fonts:
            fonts_to_download = fonts_to_download[:max_fonts]
            print(f"Limiting to {max_fonts} fonts")

        # Download fonts
        print(f"\nDownloading {len(fonts_to_download)} fonts...")

        for i, font in enumerate(fonts_to_download):
            print(f"[{i+1}/{len(fonts_to_download)}]", end='')
            self.download_font(font)
            time.sleep(self.rate_limit)

        # Save metadata
        metadata = {
            'fonts_found': len(all_fonts),
            'fonts_downloaded': len(self.downloaded),
            'fonts_failed': len(self.failed),
            'categories': categories,
            'font_list': [f.to_dict() for f in fonts_to_download]
        }

        meta_path = self.output_dir / 'dafont_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)
        print(f"Fonts found: {len(all_fonts)}")
        print(f"Fonts downloaded: {len(self.downloaded)}")
        print(f"Failed: {len(self.failed)}")
        print(f"Metadata saved: {meta_path}")

        return metadata


def main() -> None:
    """Parse command-line arguments and run the DaFont scraper.

    This function serves as the entry point when the module is run as a script.
    It configures argument parsing and initiates the scraping process.
    """
    parser = argparse.ArgumentParser(description='Scrape handwriting fonts from DaFont')
    parser.add_argument('--output', '-o', type=str, default='./dafont_fonts',
                        help='Output directory for fonts')
    parser.add_argument('--categories', '-c', nargs='+', default=['601', '603'],
                        help='Category IDs to scrape (601=Calligraphy, 603=Handwritten)')
    parser.add_argument('--pages', '-p', type=int, default=10,
                        help='Max pages per category')
    parser.add_argument('--max-fonts', '-m', type=int, default=None,
                        help='Max fonts to download (default: unlimited)')
    parser.add_argument('--rate-limit', '-r', type=float, default=1.0,
                        help='Seconds between requests')

    args = parser.parse_args()

    scraper = DaFontScraper(args.output, rate_limit=args.rate_limit)
    scraper.scrape_and_download(
        categories=args.categories,
        max_pages=args.pages,
        max_fonts=args.max_fonts
    )


if __name__ == '__main__':
    main()
