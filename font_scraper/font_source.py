"""Common interfaces and types for font scrapers.

This module provides the base class and common types for font scraper
implementations. All scrapers (DaFont, FontSpace, Google Fonts) inherit
from FontSource to ensure a consistent interface.

Design Pattern:
    Uses the Template Method pattern where FontSource defines the skeleton
    of the scraping algorithm, and subclasses implement the specific steps.

Example:
    Create a custom scraper::

        from font_source import FontSource, FontMetadata, ScraperConfig

        class MyFontScraper(FontSource):
            SOURCE_NAME = "myfontsite"

            def scrape_fonts(self, config: ScraperConfig) -> list[FontMetadata]:
                # Implement site-specific scraping logic
                return [FontMetadata(name="Font1", url="...", download_url="...")]

            def download_font(self, font: FontMetadata) -> bool:
                # Implement site-specific download logic
                return True

        # Use the scraper
        scraper = MyFontScraper('./output', rate_limit=1.0)
        metadata = scraper.scrape_and_download(
            ScraperConfig(max_pages=10, max_fonts=100)
        )

Attributes:
    FontMetadata: Dataclass containing font information.
    ScraperConfig: Dataclass for configuring scraper behavior.
    FontSource: Abstract base class for font scrapers.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, Protocol

import requests

logger = logging.getLogger(__name__)


class ScraperEventType(Enum):
    """Types of events emitted during scraping."""
    SCRAPE_START = "scrape_start"
    SCRAPE_PAGE = "scrape_page"
    SCRAPE_COMPLETE = "scrape_complete"
    DOWNLOAD_START = "download_start"
    DOWNLOAD_PROGRESS = "download_progress"
    DOWNLOAD_SUCCESS = "download_success"
    DOWNLOAD_FAIL = "download_fail"
    DOWNLOAD_COMPLETE = "download_complete"
    EXTRACT_FILE = "extract_file"
    ERROR = "error"


@dataclass
class ScraperEvent:
    """Event data emitted during scraping.

    Attributes:
        event_type: Type of event.
        source: Name of the scraper source.
        message: Human-readable message.
        data: Additional event-specific data.
    """
    event_type: ScraperEventType
    source: str
    message: str
    data: dict = field(default_factory=dict)


class ScraperObserver(Protocol):
    """Protocol for scraper event observers.

    Implement this protocol to receive events from scrapers.
    """

    def on_event(self, event: ScraperEvent) -> None:
        """Called when a scraper event occurs.

        Args:
            event: The scraper event with type, source, message, and data.
        """
        ...


def create_scrape_metadata(
    source: str,
    fonts_found: int,
    fonts_downloaded: int,
    fonts_failed: int,
    output_dir: str = "",
    fonts: list[dict] = None,
    categories: list[str] = None,
    query: str = "",
) -> dict:
    """Create a standardized metadata dictionary for scrape results.

    This function ensures all scrapers return consistent metadata keys.

    Args:
        source: Name of the source (e.g., 'dafont', 'fontspace', 'google').
        fonts_found: Total number of fonts discovered.
        fonts_downloaded: Number of fonts successfully downloaded.
        fonts_failed: Number of fonts that failed to download.
        output_dir: Directory where fonts were saved.
        fonts: List of font dictionaries with metadata.
        categories: List of category IDs (for scrapers that use categories).
        query: Search query used (for scrapers that use search).

    Returns:
        Dictionary with standardized keys:
            - source: Name of the source
            - fonts_found: Total fonts discovered
            - fonts_downloaded: Successfully downloaded count
            - fonts_failed: Failed download count
            - output_dir: Output directory path
            - fonts: List of font metadata dicts
            - categories: List of categories (if applicable)
            - query: Search query (if applicable)
    """
    result = {
        'source': source,
        'fonts_found': fonts_found,
        'fonts_downloaded': fonts_downloaded,
        'fonts_failed': fonts_failed,
    }
    if output_dir:
        result['output_dir'] = output_dir
    if fonts is not None:
        result['fonts'] = fonts
    if categories:
        result['categories'] = categories
    if query:
        result['query'] = query
    return result


@dataclass
class FontMetadata:
    """Common metadata for fonts from any source.

    This dataclass provides a unified structure for font information
    across all scraper implementations. Not all fields are required;
    scrapers populate relevant fields based on what their source provides.

    Attributes:
        name: Display name of the font.
        url: URL to the font's detail page on the source site.
        download_url: Direct URL to download the font file or archive.
        source: Name of the source (e.g., 'dafont', 'fontspace', 'google').
        category: Font category if available (e.g., 'handwriting', 'script').
        family: Font family name (for Google Fonts).
        variants: List of available variants (e.g., ['regular', 'bold']).
        downloads: Number of downloads reported by the source.
        extra: Additional source-specific metadata.
    """
    name: str
    url: str = ""
    download_url: str = ""
    source: str = ""
    category: str = ""
    family: str = ""
    variants: list[str] = field(default_factory=list)
    downloads: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary containing all non-empty fields.
        """
        result = {'name': self.name}
        if self.url:
            result['url'] = self.url
        if self.download_url:
            result['download_url'] = self.download_url
        if self.source:
            result['source'] = self.source
        if self.category:
            result['category'] = self.category
        if self.family:
            result['family'] = self.family
        if self.variants:
            result['variants'] = self.variants
        if self.downloads:
            result['downloads'] = self.downloads
        if self.extra:
            result['extra'] = self.extra
        return result


@dataclass
class ScraperConfig:
    """Configuration for scraper operations.

    Attributes:
        max_pages: Maximum pages to scrape (for paginated sources).
        max_fonts: Maximum total fonts to download.
        query: Search query string (for search-based scrapers).
        categories: List of category IDs to scrape.
        fonts: Specific font names to download.
        use_category: Use category browsing instead of search.
    """
    max_pages: int = 10000  # Effectively unlimited; pagination stops on duplicate detection
    max_fonts: int | None = None
    query: str = ""
    categories: list[str] = field(default_factory=list)
    fonts: list[str] = field(default_factory=list)
    use_category: bool = False


class FontSource(ABC):
    """Abstract base class for font scrapers.

    This class defines the common interface and shared functionality for
    all font scrapers. Subclasses must implement the abstract methods
    to provide source-specific scraping and download logic.

    The class provides:
        - Common initialization (output directory, rate limiting, session)
        - Tracking of discovered, downloaded, and failed fonts
        - Rate limiting between requests
        - Safe filename generation
        - Common scrape_and_download workflow

    Subclasses must define:
        - SOURCE_NAME: Class attribute identifying the source
        - scrape_fonts(): Method to discover fonts from the source
        - download_font(): Method to download a single font

    Attributes:
        SOURCE_NAME: Identifier for this font source (e.g., 'dafont').
        output_dir: Directory for downloaded fonts.
        rate_limit: Seconds to wait between requests.
        session: HTTP session for requests.
        fonts_found: List of discovered FontMetadata objects.
        downloaded: Set of successfully downloaded font names.
        failed: List of font names that failed to download.
    """

    SOURCE_NAME: str = "unknown"

    def __init__(self, output_dir: str, rate_limit: float = 1.0):
        """Initialize the font source.

        Args:
            output_dir: Directory path where downloaded fonts will be saved.
                The directory is created if it doesn't exist.
            rate_limit: Minimum seconds between HTTP requests.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;'
                      'q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

        self.fonts_found: list[FontMetadata] = []
        self.downloaded: set[str] = set()
        self.failed: list[str] = []
        self._observers: list[ScraperObserver] = []

    def add_observer(self, observer: ScraperObserver) -> None:
        """Add an observer to receive scraper events.

        Args:
            observer: Object implementing ScraperObserver protocol.
        """
        self._observers.append(observer)

    def remove_observer(self, observer: ScraperObserver) -> None:
        """Remove an observer.

        Args:
            observer: The observer to remove.
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def _emit(self, event_type: ScraperEventType, message: str, **data) -> None:
        """Emit an event to all observers.

        Args:
            event_type: Type of event.
            message: Human-readable message.
            **data: Additional event data.
        """
        event = ScraperEvent(
            event_type=event_type,
            source=self.SOURCE_NAME,
            message=message,
            data=data
        )
        for observer in self._observers:
            try:
                observer.on_event(event)
            except Exception as e:
                logger.warning("Observer error: %s", e)

    def wait_rate_limit(self) -> None:
        """Wait for the configured rate limit duration."""
        time.sleep(self.rate_limit)

    def request_with_retry(
        self,
        method: str,
        url: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs
    ) -> requests.Response:
        """Make an HTTP request with retry logic for transient failures.

        Handles 429 rate limit responses and connection errors with
        exponential backoff.

        Args:
            method: HTTP method ('get', 'post', etc.).
            url: The URL to request.
            max_retries: Maximum number of retry attempts. Defaults to 3.
            base_delay: Base delay in seconds for exponential backoff.
                Defaults to 1.0.
            **kwargs: Additional arguments passed to requests.Session.request().

        Returns:
            The requests.Response object.

        Raises:
            requests.RequestException: If all retries are exhausted.
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, **kwargs)

                # Handle rate limiting (429)
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        delay = float(retry_after)
                    else:
                        delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Rate limited (429) on %s, waiting %.1fs (attempt %d/%d)",
                        url, delay, attempt + 1, max_retries
                    )
                    time.sleep(delay)
                    continue

                # Retry on server errors (5xx)
                if response.status_code >= 500:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Server error (%d) on %s, waiting %.1fs (attempt %d/%d)",
                        response.status_code, url, delay, attempt + 1, max_retries
                    )
                    time.sleep(delay)
                    continue

                return response

            except (requests.ConnectionError, requests.Timeout) as e:
                last_exception = e
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Connection error on %s: %s, waiting %.1fs (attempt %d/%d)",
                    url, e, delay, attempt + 1, max_retries
                )
                time.sleep(delay)

        # All retries exhausted
        if last_exception:
            raise last_exception
        raise requests.RequestException(f"Max retries exceeded for {url}")

    def get_with_retry(self, url: str, **kwargs) -> requests.Response:
        """Convenience method for GET requests with retry logic.

        Args:
            url: The URL to fetch.
            **kwargs: Additional arguments passed to request_with_retry().

        Returns:
            The requests.Response object.
        """
        return self.request_with_retry('GET', url, **kwargs)

    def safe_filename(self, name: str) -> str:
        """Create a filesystem-safe filename.

        Args:
            name: Original filename that may contain unsafe characters.

        Returns:
            Sanitized filename safe for most filesystems.
        """
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe = safe.strip('. ')
        return safe or 'unnamed_font'

    @abstractmethod
    def scrape_fonts(self, config: ScraperConfig) -> list[FontMetadata]:
        """Discover fonts from the source.

        Subclasses implement this method to scrape font listings from
        their specific source website or API.

        Args:
            config: ScraperConfig with options for scraping.

        Returns:
            List of FontMetadata objects for discovered fonts.
        """
        pass

    @abstractmethod
    def download_font(self, font: FontMetadata) -> bool:
        """Download a single font.

        Subclasses implement this method to handle source-specific
        download logic (ZIP extraction, direct downloads, etc.).

        Args:
            font: FontMetadata with download URL and other info.

        Returns:
            True if download succeeded, False otherwise.
        """
        pass

    def log_progress(self, current: int, total: int, name: str) -> None:
        """Log download progress in a standardized format.

        Args:
            current: Current item number (1-indexed).
            total: Total number of items.
            name: Name of the item being processed.
        """
        logger.info("[%d/%d] %s", current, total, name)

    def scrape_and_download(self, config: ScraperConfig = None) -> dict:
        """Main entry point: discover and download fonts.

        This method coordinates the complete workflow:
        1. Scrape fonts using scrape_fonts()
        2. Download each font using download_font()
        3. Track successes and failures
        4. Return summary metadata

        Args:
            config: ScraperConfig with options. Defaults to ScraperConfig().

        Returns:
            Dictionary with standardized keys (via create_scrape_metadata):
                - 'source': Name of the source
                - 'fonts_found': Number of fonts discovered
                - 'fonts_downloaded': Number successfully downloaded
                - 'fonts_failed': Number that failed to download
                - 'output_dir': Path to output directory
                - 'fonts': List of FontMetadata dicts for downloaded fonts
        """
        if config is None:
            config = ScraperConfig()

        # Discover fonts
        logger.info("Scraping fonts from %s...", self.SOURCE_NAME)
        self._emit(ScraperEventType.SCRAPE_START,
                   f"Starting {self.SOURCE_NAME} scraper...")

        self.fonts_found = self.scrape_fonts(config)
        logger.info("Found %d fonts", len(self.fonts_found))

        self._emit(ScraperEventType.SCRAPE_COMPLETE,
                   f"Found {len(self.fonts_found)} fonts",
                   count=len(self.fonts_found))

        # Apply max_fonts limit
        fonts_to_download = self.fonts_found
        if config.max_fonts and len(fonts_to_download) > config.max_fonts:
            fonts_to_download = fonts_to_download[:config.max_fonts]
            logger.info("Limiting to %d fonts", config.max_fonts)

        # Download each font
        total = len(fonts_to_download)
        logger.info("Downloading %d fonts...", total)
        self._emit(ScraperEventType.DOWNLOAD_START,
                   f"Downloading {total} fonts...",
                   total=total)

        for i, font in enumerate(fonts_to_download, 1):
            self.log_progress(i, total, font.name)
            self._emit(ScraperEventType.DOWNLOAD_PROGRESS,
                       f"[{i}/{total}] Downloading {font.name}",
                       current=i, total=total, font_name=font.name)

            try:
                if self.download_font(font):
                    self.downloaded.add(font.name)
                    self._emit(ScraperEventType.DOWNLOAD_SUCCESS,
                               f"Downloaded {font.name}",
                               font_name=font.name, downloaded=len(self.downloaded))
                else:
                    # Track failure if download_font returns False but doesn't track
                    if font.name not in self.failed:
                        self.failed.append(font.name)
                    self._emit(ScraperEventType.DOWNLOAD_FAIL,
                               f"Failed to download {font.name}",
                               font_name=font.name)
            except Exception as e:
                logger.error("Error downloading %s: %s", font.name, e)
                if font.name not in self.failed:
                    self.failed.append(font.name)
                self._emit(ScraperEventType.ERROR,
                           f"Error downloading {font.name}: {e}",
                           font_name=font.name, error=str(e))

            self.wait_rate_limit()

        # Build result
        downloaded_fonts = [
            f for f in fonts_to_download if f.name in self.downloaded
        ]

        logger.info("Complete: %d downloaded, %d failed",
                    len(self.downloaded), len(self.failed))

        self._emit(ScraperEventType.DOWNLOAD_COMPLETE,
                   f"Complete: {len(self.downloaded)} downloaded, {len(self.failed)} failed",
                   downloaded=len(self.downloaded), failed=len(self.failed))

        return create_scrape_metadata(
            source=self.SOURCE_NAME,
            fonts_found=len(self.fonts_found),
            fonts_downloaded=len(self.downloaded),
            fonts_failed=len(self.failed),
            output_dir=str(self.output_dir),
            fonts=[f.to_dict() for f in downloaded_fonts],
        )

    def get_downloaded_fonts(self) -> list[FontMetadata]:
        """Get list of successfully downloaded fonts.

        Returns:
            List of FontMetadata for fonts that were downloaded.
        """
        return [f for f in self.fonts_found if f.name in self.downloaded]

    def get_failed_fonts(self) -> list[str]:
        """Get list of fonts that failed to download.

        Returns:
            List of font names that failed.
        """
        return list(self.failed)

    def reset(self) -> None:
        """Reset scraper state for a new run."""
        self.fonts_found = []
        self.downloaded = set()
        self.failed = []

    def download_font_file(self, url: str, filename: str, timeout: int = 60) -> bytes | None:
        """Download a font file from a URL.

        Common download logic used by all scrapers. Handles HTTP requests
        with retry logic and returns the raw content.

        Args:
            url: The URL to download from.
            filename: The font name (used for logging).
            timeout: Request timeout in seconds. Defaults to 60.

        Returns:
            The raw bytes of the downloaded file, or None if download failed.
        """
        try:
            logger.debug("Downloading: %s", filename)
            resp = self.get_with_retry(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            logger.error("Error downloading %s: %s", filename, e)
            return None

    def extract_font_from_zip(self, zip_content: bytes, output_dir: Path = None) -> int:
        """Extract TTF/OTF font files from a ZIP archive.

        Common ZIP extraction logic used by DaFont and FontSpace scrapers.

        Args:
            zip_content: The raw bytes of the ZIP file.
            output_dir: Directory to extract fonts to. Defaults to self.output_dir.

        Returns:
            The number of font files found (extracted + already existing),
            or 0 if extraction failed.
        """
        if output_dir is None:
            output_dir = self.output_dir

        found = 0
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                for name in zf.namelist():
                    lower_name = name.lower()
                    if lower_name.endswith(('.ttf', '.otf')):
                        # Clean filename
                        safe_name = self.safe_filename(os.path.basename(name))
                        out_path = output_dir / safe_name

                        if out_path.exists():
                            # File already exists - count it as found
                            found += 1
                        else:
                            with zf.open(name) as src, open(out_path, 'wb') as dst:
                                dst.write(src.read())
                            found += 1
                            self._emit(ScraperEventType.EXTRACT_FILE,
                                       f"Extracted {safe_name}",
                                       filename=safe_name)
        except zipfile.BadZipFile as e:
            logger.error("Invalid ZIP file: %s", e)
            self._emit(ScraperEventType.ERROR,
                       f"Invalid ZIP file: {e}",
                       error=str(e))
        return found

    def paginate(
        self,
        url_template: str,
        max_pages: int,
        page_parser: Callable[[str], list[FontMetadata]],
        timeout: int = 30
    ) -> Generator[list[FontMetadata], None, None]:
        """Iterate through paginated results.

        Common pagination pattern used by DaFont and FontSpace scrapers.
        Yields fonts from each page until no more results or max_pages reached.

        Args:
            url_template: URL template with {page} placeholder for page number.
            max_pages: Maximum number of pages to fetch.
            page_parser: Function that parses HTML and returns list of FontMetadata.
            timeout: Request timeout in seconds. Defaults to 30.

        Yields:
            List of FontMetadata objects for each page.

        Example:
            for page_fonts in self.paginate(
                "https://example.com/fonts?page={page}",
                max_pages=10,
                page_parser=self._parse_font_list
            ):
                all_fonts.extend(page_fonts)
        """
        page = 1
        seen_urls = set()  # Track seen fonts to detect pagination loops

        while page <= max_pages:
            url = url_template.format(page=page)
            logger.debug("Page %d: %s", page, url)

            try:
                resp = self.get_with_retry(url, timeout=timeout)
                resp.raise_for_status()

                page_fonts = page_parser(resp.text)

                if not page_fonts:
                    logger.debug("No fonts found on page %d, stopping.", page)
                    # Only emit if we got to page > 1 (indicates pagination)
                    if page > 1:
                        self._emit(ScraperEventType.SCRAPE_PAGE,
                                   f"Page {page}: no more fonts found",
                                   page=page, count=0)
                    break

                # Check for duplicate page (pagination loop detection)
                new_fonts = [f for f in page_fonts if f.url not in seen_urls]
                if not new_fonts:
                    logger.debug("Page %d has only duplicate fonts, stopping.", page)
                    self._emit(ScraperEventType.SCRAPE_PAGE,
                               f"Page {page}: no new fonts (pagination exhausted)",
                               page=page, count=0)
                    break

                # Track seen URLs
                for f in page_fonts:
                    seen_urls.add(f.url)

                yield new_fonts
                logger.debug("Found %d new fonts on page %d", len(new_fonts), page)

                # Only emit page events for page > 1 to reduce noise
                # (page 1 is typically covered by higher-level events)
                if page > 1:
                    self._emit(ScraperEventType.SCRAPE_PAGE,
                               f"Page {page}: found {len(new_fonts)} fonts",
                               page=page, count=len(new_fonts))

                page += 1
                time.sleep(self.rate_limit)

            except requests.RequestException as e:
                logger.warning("Error fetching page %d: %s", page, e)
                self._emit(ScraperEventType.ERROR,
                           f"Error fetching page {page}: {e}",
                           page=page, error=str(e))
                break
