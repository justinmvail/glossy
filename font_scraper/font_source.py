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

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)


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
    max_pages: int = 10
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
            Dictionary with:
                - 'source': Name of the source
                - 'fonts_found': Number of fonts discovered
                - 'downloaded': Number successfully downloaded
                - 'failed': Number that failed to download
                - 'output_dir': Path to output directory
                - 'fonts': List of FontMetadata dicts for downloaded fonts
        """
        if config is None:
            config = ScraperConfig()

        # Discover fonts
        print(f"Scraping fonts from {self.SOURCE_NAME}...")
        self.fonts_found = self.scrape_fonts(config)
        print(f"Found {len(self.fonts_found)} fonts")

        # Apply max_fonts limit
        fonts_to_download = self.fonts_found
        if config.max_fonts and len(fonts_to_download) > config.max_fonts:
            fonts_to_download = fonts_to_download[:config.max_fonts]
            print(f"Limiting to {config.max_fonts} fonts")

        # Download each font
        print(f"\nDownloading {len(fonts_to_download)} fonts...")
        for i, font in enumerate(fonts_to_download, 1):
            print(f"  [{i}/{len(fonts_to_download)}] {font.name}")

            try:
                if self.download_font(font):
                    self.downloaded.add(font.name)
                else:
                    self.failed.append(font.name)
            except Exception as e:
                print(f"    Error: {e}")
                self.failed.append(font.name)

            self.wait_rate_limit()

        # Build result
        downloaded_fonts = [
            f for f in fonts_to_download if f.name in self.downloaded
        ]

        print(f"\nComplete: {len(self.downloaded)} downloaded, "
              f"{len(self.failed)} failed")

        return {
            'source': self.SOURCE_NAME,
            'fonts_found': len(self.fonts_found),
            'downloaded': len(self.downloaded),
            'failed': len(self.failed),
            'output_dir': str(self.output_dir),
            'fonts': [f.to_dict() for f in downloaded_fonts],
        }

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
