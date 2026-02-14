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
    --categories, -c: Category IDs to scrape (default: all categories)
    --pages, -p: Maximum pages per category to scrape (default: 10)
    --max-fonts, -m: Maximum total fonts to download (default: unlimited)
    --rate-limit, -r: Seconds to wait between requests (default: 1.0)
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from font_source import FontMetadata, FontSource, ScraperConfig, create_scrape_metadata

logger = logging.getLogger(__name__)


class DaFontScraper(FontSource):
    """Scrape handwriting fonts from DaFont.com.

    This class handles the complete workflow of discovering fonts from DaFont
    category pages, parsing font information, downloading ZIP files, and
    extracting TTF/OTF font files.

    Inherits from FontSource to provide a consistent interface with other
    font scrapers (FontSpace, Google Fonts).

    Attributes:
        SOURCE_NAME (str): Identifier for this source ('dafont').
        BASE_URL (str): The base URL for DaFont.com.
        CATEGORIES (dict): Mapping of category IDs to human-readable names.
            Keys are string category IDs (e.g., '601'), values are category
            names (e.g., 'Calligraphy').
        output_dir (Path): Directory where downloaded fonts are saved.
        rate_limit (float): Delay in seconds between HTTP requests.
        session (requests.Session): HTTP session for making requests.
        fonts_found (list): List of all FontMetadata discovered during scraping.
        downloaded (Set[str]): Set of font names that were successfully downloaded.
        failed (List[str]): List of font names that failed to download.
    """

    SOURCE_NAME = "dafont"
    BASE_URL = "https://www.dafont.com"

    # HTTP timeout constants (seconds)
    PAGE_TIMEOUT = 30
    DOWNLOAD_TIMEOUT = 60

    # All DaFont categories
    CATEGORIES = {
        # Fancy
        '101': 'Fancy - Curly',
        '102': 'Fancy - Western',
        '103': 'Fancy - Eroded',
        '104': 'Fancy - Distorted',
        '105': 'Fancy - Destroy',
        '106': 'Fancy - Horror',
        '107': 'Fancy - Fire-Ice',
        '108': 'Fancy - Old School',
        '109': 'Fancy - Various',
        '110': 'Fancy - Initials',
        '111': 'Fancy - Decorative',
        '112': 'Fancy - Groovy',
        '113': 'Fancy - Modern',
        '114': 'Fancy - Hearts',
        '115': 'Fancy - Cartoon',
        '116': 'Fancy - Rounded',
        '117': 'Fancy - Stencil Army',
        '118': 'Fancy - Retro',
        # Foreign Look
        '201': 'Foreign Look - Arabic',
        '202': 'Foreign Look - Chinese',
        '203': 'Foreign Look - Mexican',
        '204': 'Foreign Look - Roman Greek',
        '205': 'Foreign Look - Russian',
        '206': 'Foreign Look - Various',
        # Techno
        '301': 'Techno - LCD',
        '302': 'Techno - Sci-Fi',
        '303': 'Techno - Square',
        '304': 'Techno - Various',
        # Bitmap
        '401': 'Bitmap - Pixel',
        '402': 'Bitmap - Bitmap',
        # Gothic
        '501': 'Gothic - Medieval',
        '502': 'Gothic - Modern',
        '503': 'Gothic - Celtic',
        '504': 'Gothic - Initials',
        '505': 'Gothic - Various',
        # Script
        '601': 'Script - Calligraphy',
        '602': 'Script - School',
        '603': 'Script - Handwritten',
        '604': 'Script - Brush',
        '605': 'Script - Trash',
        '606': 'Script - Various',
        '607': 'Script - Graffiti',
        # Basic
        '801': 'Basic - Sans Serif',
        '802': 'Basic - Serif',
        '803': 'Basic - Fixed Width',
        '804': 'Basic - Various',
        # Dingbats (skip for now as they're symbols, not text)
        # '901': 'Dingbats - Various',
    }

    # Regex patterns for HTML parsing (compiled once for efficiency)
    FONT_LINK_PATTERN = re.compile(r'\.font$')
    DOWNLOAD_COUNT_PATTERN = re.compile(r'[\d,]+\s*downloads')
    DOWNLOAD_COUNT_EXTRACT = re.compile(r'([\d,]+)\s*downloads')
    DOWNLOAD_LINK_PATTERN = re.compile(r'dl\.dafont\.com|dl\.php\?f=')
    FONT_SLUG_PATTERN = re.compile(r'[?&]f=([^&]+)')

    def __init__(self, output_dir: str, rate_limit: float = 1.0):
        """Initialize the DaFont scraper.

        Args:
            output_dir: Directory path where downloaded fonts will be saved.
                The directory will be created if it does not exist.
            rate_limit: Minimum seconds to wait between HTTP requests to avoid
                overwhelming the server. Defaults to 1.0 second.
        """
        super().__init__(output_dir, rate_limit)

    def scrape_category(self, category: str, max_pages: int = 10) -> list[FontMetadata]:
        """Scrape fonts from a specific DaFont category.

        Iterates through paginated category listings, parsing font information
        from each page until no more fonts are found or max_pages is reached.

        Args:
            category: The DaFont category ID (e.g., '601' for Calligraphy,
                '603' for Handwritten).
            max_pages: Maximum number of pages to scrape from this category.
                Defaults to 10.

        Returns:
            A list of FontMetadata objects representing all fonts found in the
            category across all scraped pages.
        """
        category_name = self.CATEGORIES.get(category, f'Unknown ({category})')
        logger.info("Scraping category: %s (cat=%s)", category_name, category)

        # Emit category start event
        from font_source import ScraperEventType
        self._emit(ScraperEventType.SCRAPE_PAGE,
                   f"Scraping category: {category_name}",
                   category=category_name, category_id=category)

        url_template = f"{self.BASE_URL}/theme.php?cat={category}&page={{page}}"

        # Create a parser that includes the category name
        def parse_with_category(html: str) -> list[FontMetadata]:
            return self._parse_font_list(html, category_name)

        fonts = []
        for page_fonts in self.paginate(url_template, max_pages, parse_with_category, self.PAGE_TIMEOUT):
            fonts.extend(page_fonts)
            logger.debug("Total fonts so far: %d", len(fonts))

        # Emit category complete event with count
        if fonts:
            self._emit(ScraperEventType.SCRAPE_PAGE,
                       f"{category_name}: found {len(fonts)} fonts",
                       category=category_name, count=len(fonts))

        return fonts

    def _parse_font_list(self, html: str, category: str) -> list[FontMetadata]:
        """Parse font list from category page HTML.

        Extracts font information from the HTML of a DaFont category page.
        Uses multiple parsing strategies to handle different page layouts.

        Args:
            html: The raw HTML content of the category page.
            category: The human-readable category name to assign to parsed fonts.

        Returns:
            A list of FontMetadata objects parsed from the page. Returns an empty
            list if no fonts could be parsed.
        """
        soup = BeautifulSoup(html, 'html.parser')
        fonts = []

        # Method 1: Find font name links (end with .font)
        font_links = soup.find_all('a', href=self.FONT_LINK_PATTERN)

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
                next_text = font_link.find_next(string=self.DOWNLOAD_COUNT_PATTERN)
                if next_text:
                    dl_match = self.DOWNLOAD_COUNT_EXTRACT.search(next_text)
                    if dl_match:
                        downloads = int(dl_match.group(1).replace(',', ''))

                fonts.append(FontMetadata(
                    name=font_name,
                    url=font_url,
                    download_url=download_url,
                    source=self.SOURCE_NAME,
                    category=category,
                    downloads=downloads
                ))

            except Exception:
                continue

        # Fallback: parse download links directly
        if not fonts:
            fonts = self._parse_font_list_alt(soup, category)

        return fonts

    def _parse_font_list_alt(self, soup: BeautifulSoup, category: str) -> list[FontMetadata]:
        """Alternative parsing method using download URLs.

        This fallback method extracts font information by finding download
        links directly, used when the primary parsing method fails.

        Args:
            soup: BeautifulSoup object of the parsed HTML page.
            category: The human-readable category name to assign to parsed fonts.

        Returns:
            A list of FontMetadata objects parsed from download links.
        """
        fonts = []
        seen = set()

        # Look for download links
        for link in soup.find_all('a', href=self.DOWNLOAD_LINK_PATTERN):
            try:
                download_url = link.get('href', '')

                # Extract font slug from download URL
                match = self.FONT_SLUG_PATTERN.search(download_url)
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

                fonts.append(FontMetadata(
                    name=font_name,
                    url=font_url,
                    download_url=download_url,
                    source=self.SOURCE_NAME,
                    category=category,
                    downloads=0
                ))

            except Exception:
                continue

        return fonts

    def download_font(self, font: FontMetadata) -> bool:
        """Download a font and extract TTF/OTF files from the ZIP archive.

        Downloads the font ZIP file from DaFont, extracts any TTF or OTF
        font files, and saves them to the output directory.

        Args:
            font: FontMetadata object containing the font's download URL and metadata.

        Returns:
            True if the font was successfully downloaded and at least one
            font file was extracted, False otherwise.
        """
        if font.name in self.downloaded:
            return True

        # Use base class download method
        content = self.download_font_file(font.download_url, font.name, self.DOWNLOAD_TIMEOUT)
        if content is None:
            self.failed.append(font.name)
            return False

        # DaFont serves ZIP files
        if content[:4] != b'PK\x03\x04':
            logger.warning("Not a ZIP file for %s", font.name)
            return False

        # Use base class ZIP extraction
        extracted = self.extract_font_from_zip(content)

        if extracted > 0:
            logger.debug("Extracted %d font file(s) for %s", extracted, font.name)
            self.downloaded.add(font.name)
            return True
        else:
            logger.warning("No TTF/OTF files found in ZIP for %s", font.name)
            return False

    def scrape_fonts(self, config: ScraperConfig) -> list[FontMetadata]:
        """Discover fonts from DaFont categories.

        Scrapes specified categories and returns a list of FontMetadata objects.

        Args:
            config: ScraperConfig with options. Uses config.categories if provided,
                otherwise defaults to all categories in CATEGORIES.

        Returns:
            List of FontMetadata objects for discovered fonts.
        """
        categories = config.categories if config.categories else list(self.CATEGORIES.keys())
        max_pages = config.max_pages

        all_fonts = []
        for cat in categories:
            fonts = self.scrape_category(cat, max_pages)
            all_fonts.extend(fonts)

        # Remove duplicates by name
        unique_fonts = {f.name: f for f in all_fonts}
        fonts = list(unique_fonts.values())

        # Sort by downloads (most popular first)
        fonts.sort(key=lambda x: x.downloads, reverse=True)

        return fonts

    def scrape_and_download(
        self,
        config: ScraperConfig | None = None,
        categories: list[str] = None,
        max_pages: int = 10,
        max_fonts: int = None
    ) -> dict:
        """Scrape and download fonts from specified categories.

        This is the main entry point for the scraper. It coordinates the
        complete workflow of discovering fonts across categories, removing
        duplicates, and downloading font files.

        Supports both the new ScraperConfig interface and the legacy
        keyword arguments for backward compatibility.

        Args:
            config: ScraperConfig with options. If provided, other arguments
                are ignored.
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
                - fonts_downloaded/downloaded (int): Number of fonts successfully downloaded.
                - fonts_failed/failed (int): Number of fonts that failed to download.
                - categories (list): List of category IDs that were scraped.
                - font_list/fonts (list): List of font dictionaries with metadata.
        """
        # Support ScraperConfig
        if config is not None:
            return super().scrape_and_download(config)

        # Legacy interface
        if categories is None:
            categories = list(self.CATEGORIES.keys())

        logger.info("=" * 60)
        logger.info("DaFont Scraper")
        logger.info("=" * 60)
        logger.info("Output directory: %s", self.output_dir)
        logger.info("Categories: %s", [self.CATEGORIES.get(c, c) for c in categories])
        logger.info("Max pages per category: %d", max_pages)

        # Scrape all categories
        all_fonts = []
        for cat in categories:
            fonts = self.scrape_category(cat, max_pages)
            all_fonts.extend(fonts)
            self.fonts_found.extend(fonts)

        logger.info("Total fonts found: %d", len(all_fonts))

        # Remove duplicates by name
        unique_fonts = {f.name: f for f in all_fonts}
        fonts_to_download = list(unique_fonts.values())

        logger.info("Unique fonts: %d", len(fonts_to_download))

        # Sort by downloads (most popular first) if we have that info
        fonts_to_download.sort(key=lambda x: x.downloads, reverse=True)

        # Limit if specified
        if max_fonts:
            fonts_to_download = fonts_to_download[:max_fonts]
            logger.info("Limiting to %d fonts", max_fonts)

        # Download fonts
        logger.info("Downloading %d fonts...", len(fonts_to_download))

        for i, font in enumerate(fonts_to_download, 1):
            self.log_progress(i, len(fonts_to_download), font.name)
            self.download_font(font)
            time.sleep(self.rate_limit)

        # Build standardized metadata
        metadata = create_scrape_metadata(
            source=self.SOURCE_NAME,
            fonts_found=len(all_fonts),
            fonts_downloaded=len(self.downloaded),
            fonts_failed=len(self.failed),
            output_dir=str(self.output_dir),
            fonts=[f.to_dict() for f in fonts_to_download],
            categories=categories,
        )

        meta_path = self.output_dir / 'dafont_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 60)
        logger.info("COMPLETE")
        logger.info("=" * 60)
        logger.info("Fonts found: %d", len(all_fonts))
        logger.info("Fonts downloaded: %d", len(self.downloaded))
        logger.info("Failed: %d", len(self.failed))
        logger.info("Metadata saved: %s", meta_path)

        return metadata


def main() -> None:
    """Parse command-line arguments and run the DaFont scraper.

    This function serves as the entry point when the module is run as a script.
    It configures argument parsing and initiates the scraping process.
    """
    parser = argparse.ArgumentParser(description='Scrape handwriting fonts from DaFont')
    parser.add_argument('--output', '-o', type=str, default='./dafont_fonts',
                        help='Output directory for fonts')
    parser.add_argument('--categories', '-c', nargs='+', default=None,
                        help='Category IDs to scrape (default: all categories)')
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
