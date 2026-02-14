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

    # FontSpace categories (slug names for URL)
    CATEGORIES = [
        'handwriting',
        'script',
        'calligraphy',
        'brush',
        'signature',
        'graffiti',
        'kids',
        'cute',
        'marker',
        'pen',
    ]

    # Regex patterns for HTML parsing (compiled once for efficiency)
    # FontSpace changed URL structure: now /fontname-font-fXXXXX instead of /font/fontname
    FONT_PAGE_PATTERN = re.compile(r'/[a-z0-9-]+-font-f\d+$', re.IGNORECASE)
    # Download links now use /get/family/XXXX pattern
    DOWNLOAD_PATH_PATTERN = re.compile(r'/get/family/')
    ZIP_FILE_PATTERN = re.compile(r'\.zip')

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

        from font_source import ScraperEventType
        self._emit(ScraperEventType.SCRAPE_PAGE,
                   f"Searching for: {query}",
                   query=query)

        url_template = f"{self.BASE_URL}/search?q={query}&p={{page}}"

        fonts = []
        for page_fonts in self.paginate(url_template, max_pages, self._parse_search_results, self.PAGE_TIMEOUT):
            fonts.extend(page_fonts)
            logger.debug("Total fonts so far: %d", len(fonts))

        if fonts:
            self._emit(ScraperEventType.SCRAPE_COMPLETE,
                       f"Search '{query}': found {len(fonts)} fonts",
                       query=query, count=len(fonts))

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

        from font_source import ScraperEventType
        self._emit(ScraperEventType.SCRAPE_PAGE,
                   f"Browsing category: {category}",
                   category=category)

        url_template = f"{self.BASE_URL}/category/{category}?p={{page}}"

        fonts = []
        for page_fonts in self.paginate(url_template, max_pages, self._parse_search_results, self.PAGE_TIMEOUT):
            fonts.extend(page_fonts)
            logger.debug("Total fonts so far: %d", len(fonts))

        if fonts:
            self._emit(ScraperEventType.SCRAPE_COMPLETE,
                       f"Category {category}: found {len(fonts)} fonts",
                       category=category, count=len(fonts))

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
        for link in soup.find_all('a', href=self.FONT_PAGE_PATTERN):
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
            dl_link = soup.find('a', href=self.DOWNLOAD_PATH_PATTERN)
            if dl_link:
                url = dl_link.get('href', '')
                if url and not url.startswith('http'):
                    url = urljoin(self.BASE_URL, url)
                return url

            # Alternative: look for direct ZIP link
            zip_link = soup.find('a', href=self.ZIP_FILE_PATTERN)
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

        # Get download URL if we don't have it
        if not font.download_url:
            font.download_url = self._get_download_url(font.url)
            time.sleep(0.5)

        if not font.download_url:
            logger.warning("Could not find download URL for %s", font.name)
            self.failed.append(font.name)
            return False

        # Use base class download method
        content = self.download_font_file(font.download_url, font.name, self.DOWNLOAD_TIMEOUT)
        if content is None:
            self.failed.append(font.name)
            return False

        # Check if it's a ZIP file
        if content[:4] == b'PK\x03\x04':
            # Use base class ZIP extraction
            extracted = self.extract_font_from_zip(content)
            if extracted > 0:
                logger.debug("Extracted %d font file(s) for %s", extracted, font.name)
                self.downloaded.add(font.name)
                return True
            else:
                logger.warning("No TTF/OTF files in ZIP for %s", font.name)
                return False
        else:
            # Might be a direct font file
            if font.download_url.endswith(('.ttf', '.otf')):
                ext = '.ttf' if '.ttf' in font.download_url else '.otf'
                safe_name = self.safe_filename(font.name) + ext
                out_path = self.output_dir / safe_name
                out_path.write_bytes(content)
                logger.debug("Saved %s", safe_name)
                self.downloaded.add(font.name)
                return True

        logger.warning("Unknown file format for %s", font.name)
        return False

    def scrape_fonts(self, config: ScraperConfig) -> list[FontMetadata]:
        """Discover fonts from FontSpace.

        Scrapes fonts via search or category browsing based on config.
        By default, scrapes all categories in CATEGORIES.

        Args:
            config: ScraperConfig with options.
                - If config.categories is set, scrapes those categories.
                - If config.use_category is True with config.query, scrapes that single category.
                - If config.query is set without use_category, performs search.
                - Otherwise, scrapes all default CATEGORIES.

        Returns:
            List of FontMetadata objects for discovered fonts.
        """
        from font_source import ScraperEventType

        max_pages = config.max_pages

        # Determine which categories to scrape
        if config.categories:
            # Explicit category list provided
            categories = config.categories
        elif config.use_category and config.query:
            # Single category specified
            categories = [config.query]
        elif config.query and not config.use_category:
            # Search mode
            return self.scrape_search(config.query, max_pages)
        else:
            # Default: scrape all categories
            categories = self.CATEGORIES

        self._emit(ScraperEventType.SCRAPE_PAGE,
                   f"Scraping {len(categories)} categories",
                   categories=categories)

        # Scrape all categories
        all_fonts = []
        seen_urls = set()

        for category in categories:
            fonts = self.scrape_category(category, max_pages)
            # Deduplicate across categories
            for font in fonts:
                if font.url not in seen_urls:
                    seen_urls.add(font.url)
                    all_fonts.append(font)

        self._emit(ScraperEventType.SCRAPE_COMPLETE,
                   f"Found {len(all_fonts)} unique fonts across {len(categories)} categories",
                   count=len(all_fonts))

        return all_fonts

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

        for i, font in enumerate(fonts, 1):
            self.log_progress(i, len(fonts), font.name)
            self.download_font(font)
            time.sleep(self.rate_limit)

        # Build standardized metadata
        metadata = create_scrape_metadata(
            source=self.SOURCE_NAME,
            fonts_found=len(self.fonts_found),
            fonts_downloaded=len(self.downloaded),
            fonts_failed=len(self.failed),
            output_dir=str(self.output_dir),
            fonts=[f.to_dict() for f in fonts],
            query=query,
        )

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
