"""Background scraper worker with incremental database storage.

This module provides a clean architecture for running font scrapers in
background threads with:
- Category-by-category processing (scrape -> download -> next category)
- Parallel downloads using ThreadPoolExecutor
- Incremental database storage (fonts available immediately)
- Persistent progress tracking in database
- Cancellation support

Architecture:
- ScraperWorker: Orchestrates the scraping workflow (single responsibility)
- ScraperAdapter: Protocol for scraper implementations
- ParallelDownloader: Handles concurrent downloads
- ScraperRegistry: Manages active workers (replaces global state)

Example:
    from scraper_worker import ScraperRegistry

    registry = ScraperRegistry()
    job_id = registry.start('dafont')

    # Check progress
    status = registry.get_status(job_id)

    # Cancel if needed
    registry.stop(job_id)
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols (Interfaces)
# =============================================================================

class ScraperAdapter(Protocol):
    """Protocol for font scraper implementations.

    Each scraper (DaFont, FontSpace, Google) implements this interface,
    allowing ScraperWorker to work with any scraper type without
    hardcoded conditionals.
    """

    @property
    def source_name(self) -> str:
        """Return the source identifier (e.g., 'dafont')."""
        ...

    def get_categories(self) -> list[str]:
        """Return list of category IDs to scrape."""
        ...

    def get_category_name(self, category_id: str) -> str:
        """Return human-readable category name."""
        ...

    def scrape_category(self, category_id: str) -> list[dict]:
        """Scrape fonts from a category.

        Returns list of dicts with: name, url, download_url, category
        """
        ...

    def download_font(self, font_data: dict) -> tuple[bool, str | None]:
        """Download a font.

        Args:
            font_data: Dict with name, url, download_url, category

        Returns:
            Tuple of (success, file_path or None)
        """
        ...


class JobRepository(Protocol):
    """Protocol for job persistence operations."""

    def create_job(self, source: str) -> int:
        ...

    def update_job(self, job_id: int, **kwargs) -> None:
        ...

    def get_job(self, job_id: int) -> dict | None:
        ...

    def add_fonts_batch(self, job_id: int, fonts: list[dict]) -> int:
        ...

    def get_pending_fonts(self, job_id: int, category: str) -> list[dict]:
        ...

    def update_font_status(self, font_id: int, status: str,
                           file_path: str = None, error: str = None) -> None:
        ...

    def register_font(self, name: str, file_path: str, source: str,
                      url: str = None, category: str = None) -> int:
        ...

    def font_exists(self, name: str, source: str) -> bool:
        ...


# =============================================================================
# Adapters - Wrap existing scrapers to implement ScraperAdapter
# =============================================================================

class DaFontAdapter:
    """Adapter for DaFontScraper implementing ScraperAdapter protocol."""

    def __init__(self, output_dir: Path):
        from dafont_scraper import DaFontScraper
        self._scraper = DaFontScraper(str(output_dir), rate_limit=0.5)
        self._progress_callback = None
        self._file_cache: set[str] | None = None  # Lazy-loaded, updated incrementally

    def set_progress_callback(self, callback: Callable[[str], None]) -> None:
        """Set a callback for progress updates during scraping."""
        self._progress_callback = callback

    @property
    def source_name(self) -> str:
        return 'dafont'

    def get_categories(self) -> list[str]:
        return list(self._scraper.CATEGORIES.keys())

    def get_category_name(self, category_id: str) -> str:
        return self._scraper.CATEGORIES.get(category_id, category_id)

    def scrape_category(self, category_id: str) -> list[dict]:
        """Scrape fonts with page-by-page progress logging."""
        category_name = self.get_category_name(category_id)
        fonts = []

        # Use pagination directly for progress reporting
        url_template = f"{self._scraper.BASE_URL}/theme.php?cat={category_id}&page={{page}}"

        def parse_with_category(html: str):
            return self._scraper._parse_font_list(html, category_name)

        page = 0
        for page_fonts in self._scraper.paginate(url_template, 100, parse_with_category, 30):
            page += 1
            fonts.extend(page_fonts)
            # Report progress every page
            if self._progress_callback:
                self._progress_callback(f"Page {page}: found {len(page_fonts)} fonts ({len(fonts)} total)")

        return [
            {
                'name': f.name,
                'url': f.url,
                'download_url': f.download_url,
                'category': f.category or category_name,
            }
            for f in fonts
        ]

    def download_font(self, font_data: dict) -> tuple[bool, str | None]:
        from font_source import FontMetadata
        font = FontMetadata(
            name=font_data['name'],
            url=font_data.get('url', ''),
            download_url=font_data.get('download_url', ''),
            source=self.source_name,
            category=font_data.get('category', '')
        )
        try:
            before = self._get_font_files()
            success = self._scraper.download_font(font)

            # Scan only for NEW files (cheap os.listdir diff against cache)
            new_files = self._get_new_files(before)

            if new_files:
                new_file = sorted(new_files)[0]
                # Update cache with new files
                self._file_cache.update(new_files)
                return True, f"fonts/{self.source_name}/{new_file}"

            if success:
                file_path = self._find_file_by_name(font_data['name'])
                if file_path:
                    return True, file_path
                return True, None

            file_path = self._find_file_by_name(font_data['name'])
            if file_path:
                return True, file_path

            return False, None
        except Exception as e:
            logger.error("Download failed for %s: %s", font_data['name'], e)
            return False, None

    def _get_font_files(self) -> set[str]:
        """Get cached set of font file names. Loads from disk once."""
        if self._file_cache is None:
            output_dir = self._scraper.output_dir
            self._file_cache = set()
            for ext in ['.ttf', '.otf', '.TTF', '.OTF']:
                for match in output_dir.glob(f"*{ext}"):
                    self._file_cache.add(match.name)
            logger.info("Font file cache loaded: %d files", len(self._file_cache))
        return self._file_cache

    def _get_new_files(self, before: set[str]) -> set[str]:
        """Fast scan for new files by diffing os.listdir against cache."""
        output_dir = self._scraper.output_dir
        current = set()
        for name in os.listdir(output_dir):
            if name.lower().endswith(('.ttf', '.otf')):
                current.add(name)
        return current - before

    def _find_file_by_name(self, font_name: str) -> str | None:
        """Search cached file list for a matching font name."""
        def normalize(s: str) -> str:
            return s.lower().replace(' ', '').replace('-', '').replace('_', '')

        search_norm = normalize(font_name)
        first_word = font_name.split()[0].lower() if ' ' in font_name else None
        cached = self._get_font_files()

        candidates = []
        for filename in cached:
            stem = filename.rsplit('.', 1)[0] if '.' in filename else filename
            file_norm = normalize(stem)

            if search_norm == file_norm:
                return f"fonts/{self.source_name}/{filename}"

            if search_norm in file_norm or file_norm in search_norm:
                candidates.append((filename, len(file_norm)))
                continue

            if first_word and first_word in stem.lower():
                candidates.append((filename, len(file_norm) + 100))

        if candidates:
            candidates.sort(key=lambda x: x[1])
            return f"fonts/{self.source_name}/{candidates[0][0]}"

        return None


class FontSpaceAdapter:
    """Adapter for FontSpaceScraper implementing ScraperAdapter protocol."""

    def __init__(self, output_dir: Path):
        from fontspace_scraper import FontSpaceScraper
        self._scraper = FontSpaceScraper(str(output_dir), rate_limit=0.5)
        self._file_cache: set[str] | None = None

    @property
    def source_name(self) -> str:
        return 'fontspace'

    def get_categories(self) -> list[str]:
        # FontSpace CATEGORIES is a list, not a dict
        return list(self._scraper.CATEGORIES)

    def get_category_name(self, category_id: str) -> str:
        # FontSpace categories are simple strings, capitalize for display
        return category_id.replace('-', ' ').title()

    def scrape_category(self, category_id: str) -> list[dict]:
        fonts = self._scraper.scrape_category(category_id, max_pages=100)
        return [
            {
                'name': f.name,
                'url': f.url,
                'download_url': f.download_url,
                'category': f.category or self.get_category_name(category_id),
            }
            for f in fonts
        ]

    def download_font(self, font_data: dict) -> tuple[bool, str | None]:
        from font_source import FontMetadata
        font = FontMetadata(
            name=font_data['name'],
            url=font_data.get('url', ''),
            download_url=font_data.get('download_url', ''),
            source=self.source_name,
            category=font_data.get('category', '')
        )
        try:
            before = self._get_font_files()
            success = self._scraper.download_font(font)
            if success:
                new_files = self._get_new_files(before)
                if new_files:
                    new_file = sorted(new_files)[0]
                    self._file_cache.update(new_files)
                    return True, f"fonts/{self.source_name}/{new_file}"
                file_path = self._find_file_by_name(font_data['name'])
                return True, file_path
            return False, None
        except Exception as e:
            logger.error("Download failed for %s: %s", font_data['name'], e)
            return False, None

    def _get_font_files(self) -> set[str]:
        """Get cached set of font file names. Loads from disk once."""
        if self._file_cache is None:
            output_dir = self._scraper.output_dir
            self._file_cache = set()
            for ext in ['.ttf', '.otf', '.TTF', '.OTF']:
                for match in output_dir.glob(f"*{ext}"):
                    self._file_cache.add(match.name)
            logger.info("Font file cache loaded: %d files", len(self._file_cache))
        return self._file_cache

    def _get_new_files(self, before: set[str]) -> set[str]:
        """Fast scan for new files by diffing os.listdir against cache."""
        output_dir = self._scraper.output_dir
        current = set()
        for name in os.listdir(output_dir):
            if name.lower().endswith(('.ttf', '.otf')):
                current.add(name)
        return current - before

    def _find_file_by_name(self, font_name: str) -> str | None:
        """Search cached file list for a matching font name."""
        search_terms = font_name.lower().replace(' ', '').replace('-', '').replace('_', '')
        for filename in self._get_font_files():
            stem = filename.rsplit('.', 1)[0] if '.' in filename else filename
            file_base = stem.lower().replace(' ', '').replace('-', '').replace('_', '')
            if search_terms in file_base or file_base in search_terms:
                return f"fonts/{self.source_name}/{filename}"
        return None


class GoogleFontsAdapter:
    """Adapter for GoogleFontsScraper implementing ScraperAdapter protocol."""

    # All Google Fonts categories
    ALL_CATEGORIES = ['Handwriting', 'Display', 'Sans Serif', 'Serif', 'Monospace']

    def __init__(self, output_dir: Path):
        from google_fonts_scraper import GoogleFontsScraper
        self._scraper = GoogleFontsScraper(str(output_dir), rate_limit=0.2)
        self._category_cache = {}  # Cache per category

    @property
    def source_name(self) -> str:
        return 'google'

    def get_categories(self) -> list[str]:
        # Return ALL Google Fonts categories
        return self.ALL_CATEGORIES.copy()

    def get_category_name(self, category_id: str) -> str:
        return category_id  # Category names are already human-readable

    def scrape_category(self, category_id: str) -> list[dict]:
        # Check cache first
        if category_id in self._category_cache:
            return self._category_cache[category_id]

        # Scrape this specific category
        fonts = self._scraper.scrape_category(category_id)
        result = [
            {
                'name': f.name,
                'url': f.url,
                'download_url': f.download_url,
                'category': category_id,
            }
            for f in fonts
        ]
        self._category_cache[category_id] = result
        return result

    def download_font(self, font_data: dict) -> tuple[bool, str | None]:
        from font_source import FontMetadata
        font = FontMetadata(
            name=font_data['name'],
            url=font_data.get('url', ''),
            download_url=font_data.get('download_url', ''),
            source=self.source_name,
            category=font_data.get('category', '')
        )
        try:
            success = self._scraper.download_font(font)
            if success:
                # Google Fonts uses predictable naming: google_{safe_name}{ext}
                file_path = self._construct_file_path(font_data['name'])
                return True, file_path
            return False, None
        except Exception as e:
            logger.error("Download failed for %s: %s", font_data['name'], e)
            return False, None

    def _construct_file_path(self, font_name: str) -> str | None:
        """Construct file path based on Google Fonts naming pattern."""
        safe_name = self._scraper.safe_filename(font_name.replace(' ', '_'))
        output_dir = self._scraper.output_dir

        # Check for the file with different extensions
        for ext in ['.woff2', '.woff', '.ttf']:
            expected_path = output_dir / f"google_{safe_name}{ext}"
            if expected_path.exists():
                return f"fonts/{self.source_name}/{expected_path.name}"

        return None


# =============================================================================
# Adapter Factory
# =============================================================================

def create_adapter(source: str, base_dir: Path = None) -> ScraperAdapter:
    """Factory function to create appropriate scraper adapter.

    Args:
        source: Scraper source name ('dafont', 'fontspace', 'google')
        base_dir: Base directory for output (defaults to module directory)

    Returns:
        ScraperAdapter implementation for the source.

    Raises:
        ValueError: If source is unknown.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent

    adapters = {
        'dafont': lambda: DaFontAdapter(base_dir / 'fonts' / 'dafont'),
        'fontspace': lambda: FontSpaceAdapter(base_dir / 'fonts' / 'fontspace'),
        'google': lambda: GoogleFontsAdapter(base_dir / 'fonts' / 'google'),
    }

    if source not in adapters:
        raise ValueError(f"Unknown source: {source}. Valid: {list(adapters.keys())}")

    return adapters[source]()


# =============================================================================
# SQLite Job Repository Implementation
# =============================================================================

class SQLiteJobRepository:
    """SQLite implementation of JobRepository protocol.

    Wraps database operations for scraper jobs and fonts.
    Uses connection-per-operation for thread safety.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path(__file__).parent / 'fonts.db')

    def _get_conn(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=30000')
        return conn

    def create_job(self, source: str) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "INSERT INTO scraper_jobs (source, status) VALUES (?, 'pending')",
                (source,)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def update_job(self, job_id: int, **kwargs) -> None:
        if not kwargs:
            return
        fields = [f"{k} = ?" for k in kwargs]
        values = list(kwargs.values()) + [job_id]

        conn = self._get_conn()
        try:
            conn.execute(
                f"UPDATE scraper_jobs SET {', '.join(fields)} WHERE id = ?",
                values
            )
            conn.commit()
        finally:
            conn.close()

    def get_job(self, job_id: int) -> dict | None:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM scraper_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def is_cancelled(self, job_id: int) -> bool:
        job = self.get_job(job_id)
        return job and job['status'] == 'cancelled'

    def add_fonts_batch(self, job_id: int, fonts: list[dict]) -> int:
        if not fonts:
            return 0

        conn = self._get_conn()
        try:
            # Get existing fonts to skip
            existing = {
                row[0] for row in conn.execute(
                    "SELECT name FROM fonts WHERE source = (SELECT source FROM scraper_jobs WHERE id = ?)",
                    (job_id,)
                ).fetchall()
            }
            existing |= {
                row[0] for row in conn.execute(
                    "SELECT name FROM scraper_fonts WHERE job_id = ?", (job_id,)
                ).fetchall()
            }

            # Filter new fonts
            new_fonts = [f for f in fonts if f['name'] not in existing]
            if not new_fonts:
                return 0

            conn.executemany(
                """INSERT INTO scraper_fonts (job_id, name, url, download_url, category, status)
                   VALUES (?, ?, ?, ?, ?, 'pending')""",
                [(job_id, f['name'], f.get('url'), f.get('download_url'), f.get('category'))
                 for f in new_fonts]
            )
            conn.commit()
            return len(new_fonts)
        finally:
            conn.close()

    def get_pending_fonts(self, job_id: int, category: str = None) -> list[dict]:
        conn = self._get_conn()
        try:
            if category:
                rows = conn.execute(
                    """SELECT id, name, url, download_url, category FROM scraper_fonts
                       WHERE job_id = ? AND category = ? AND status = 'pending'""",
                    (job_id, category)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, name, url, download_url, category FROM scraper_fonts
                       WHERE job_id = ? AND status = 'pending'""",
                    (job_id,)
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def update_font_status(self, font_id: int, status: str,
                           file_path: str = None, error: str = None) -> None:
        conn = self._get_conn()
        try:
            if file_path:
                conn.execute(
                    "UPDATE scraper_fonts SET status = ?, file_path = ? WHERE id = ?",
                    (status, file_path, font_id)
                )
            elif error:
                conn.execute(
                    "UPDATE scraper_fonts SET status = ?, error_message = ? WHERE id = ?",
                    (status, error, font_id)
                )
            else:
                conn.execute(
                    "UPDATE scraper_fonts SET status = ? WHERE id = ?",
                    (status, font_id)
                )
            conn.commit()
        finally:
            conn.close()

    def register_font(self, name: str, file_path: str, source: str,
                      url: str = None, category: str = None) -> int:
        conn = self._get_conn()
        try:
            # Check if font exists by name+source
            existing = conn.execute(
                "SELECT id FROM fonts WHERE name = ? AND source = ?",
                (name, source)
            ).fetchone()
            if existing:
                return existing[0]

            # Check if file_path already exists (UNIQUE constraint)
            existing_path = conn.execute(
                "SELECT id FROM fonts WHERE file_path = ?",
                (file_path,)
            ).fetchone()
            if existing_path:
                # File already registered under different name, skip
                return existing_path[0]

            cursor = conn.execute(
                """INSERT INTO fonts (name, file_path, source, url, category)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, file_path, source, url, category)
            )
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            # Handle race condition - another thread might have inserted
            if "UNIQUE constraint" in str(e):
                existing = conn.execute(
                    "SELECT id FROM fonts WHERE file_path = ?",
                    (file_path,)
                ).fetchone()
                return existing[0] if existing else -1
            raise
        finally:
            conn.close()

    def font_exists(self, name: str, source: str) -> bool:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM fonts WHERE name = ? AND source = ?",
                (name, source)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def add_log(self, job_id: int, message: str, level: str = 'info') -> None:
        """Add a log entry for a job."""
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO scraper_logs (job_id, level, message) VALUES (?, ?, ?)",
                (job_id, level, message)
            )
            conn.commit()
        except Exception:
            pass  # Don't fail on log errors
        finally:
            conn.close()

    def get_logs(self, job_id: int = None, limit: int = 100) -> list[dict]:
        """Get recent log entries."""
        conn = self._get_conn()
        try:
            if job_id:
                rows = conn.execute(
                    """SELECT id, job_id, level, message, created_at
                       FROM scraper_logs WHERE job_id = ?
                       ORDER BY id DESC LIMIT ?""",
                    (job_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, job_id, level, message, created_at
                       FROM scraper_logs
                       ORDER BY id DESC LIMIT ?""",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def clear_old_logs(self, keep_count: int = 500) -> None:
        """Keep only the most recent logs."""
        conn = self._get_conn()
        try:
            conn.execute(
                """DELETE FROM scraper_logs WHERE id NOT IN
                   (SELECT id FROM scraper_logs ORDER BY id DESC LIMIT ?)""",
                (keep_count,)
            )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()


# =============================================================================
# Parallel Downloader
# =============================================================================

@dataclass
class DownloadResult:
    """Result of a font download attempt."""
    font_id: int
    name: str
    success: bool
    file_path: str | None = None
    error: str | None = None


class ParallelDownloader:
    """Handles parallel font downloads with configurable concurrency."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def download_batch(
        self,
        fonts: list[dict],
        download_fn: Callable[[dict], tuple[bool, str | None]],
        on_complete: Callable[[DownloadResult], None] = None,
        is_cancelled: Callable[[], bool] = None
    ) -> tuple[int, int]:
        """Download fonts in parallel.

        Args:
            fonts: List of font dicts with id, name, url, download_url, category
            download_fn: Function to download a single font
            on_complete: Callback for each completed download
            is_cancelled: Function to check if download should stop

        Returns:
            Tuple of (downloaded_count, failed_count)
        """
        if not fonts:
            return 0, 0

        downloaded = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(download_fn, font): font
                for font in fonts
            }

            for future in as_completed(futures):
                if is_cancelled and is_cancelled():
                    for f in futures:
                        f.cancel()
                    break

                font = futures[future]
                try:
                    success, file_path = future.result()
                    result = DownloadResult(
                        font_id=font['id'],
                        name=font['name'],
                        success=success,
                        file_path=file_path
                    )
                    if success:
                        downloaded += 1
                    else:
                        failed += 1
                        result.error = "Download failed"

                except Exception as e:
                    result = DownloadResult(
                        font_id=font['id'],
                        name=font['name'],
                        success=False,
                        error=str(e)
                    )
                    failed += 1

                if on_complete:
                    on_complete(result)

        return downloaded, failed


# =============================================================================
# Scraper Worker - Single Responsibility: Orchestration
# =============================================================================

class ScraperWorker:
    """Orchestrates background scraping workflow.

    Single responsibility: coordinate the scraping process using
    injected dependencies for actual work.

    Attributes:
        adapter: ScraperAdapter for source-specific operations
        repository: JobRepository for persistence
        downloader: ParallelDownloader for concurrent downloads
        job_id: Database job ID
    """

    def __init__(
        self,
        adapter: ScraperAdapter,
        repository: JobRepository,
        job_id: int,
        downloader: ParallelDownloader = None
    ):
        self.adapter = adapter
        self.repository = repository
        self.job_id = job_id
        self.downloader = downloader or ParallelDownloader()

        self._thread = None
        self._cancelled = False
        self._lock = threading.Lock()

        # Statistics
        self.fonts_found = 0
        self.fonts_downloaded = 0
        self.fonts_failed = 0

    def _log(self, message: str, level: str = 'info') -> None:
        """Add a log entry for this job."""
        self.repository.add_log(self.job_id, message, level)

    def start(self) -> None:
        """Start the worker in a background thread."""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Worker already running")

        self._cancelled = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Signal the worker to cancel."""
        with self._lock:
            self._cancelled = True
        self.repository.update_job(self.job_id, status='cancelled')

    def is_running(self) -> bool:
        """Check if worker thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def _is_cancelled(self) -> bool:
        """Check cancellation from both local flag and database."""
        if self._cancelled:
            return True
        return self.repository.is_cancelled(self.job_id)

    def _run(self) -> None:
        """Main worker loop - orchestrates the scraping workflow."""
        source = self.adapter.source_name
        logger.info("Starting scraper worker for %s (job %d)", source, self.job_id)
        self._log(f"Starting {source} scraper", "info")

        try:
            # Update job to running
            self.repository.update_job(
                self.job_id,
                status='running',
                started_at=datetime.now().isoformat()
            )

            categories = self.adapter.get_categories()
            self.repository.update_job(self.job_id, categories_total=len(categories))
            self._log(f"Found {len(categories)} categories to scrape", "info")

            for i, category_id in enumerate(categories):
                if self._is_cancelled():
                    logger.info("Job %d cancelled", self.job_id)
                    self._log("Job cancelled by user", "warning")
                    return

                category_name = self.adapter.get_category_name(category_id)
                logger.info("Processing category %d/%d: %s",
                            i + 1, len(categories), category_name)
                self._log(f"Category {i+1}/{len(categories)}: {category_name}", "info")

                self.repository.update_job(
                    self.job_id,
                    current_category=category_name,
                    categories_done=i
                )

                # Step 1: Scrape category
                self._scrape_category(category_id, category_name)

                if self._is_cancelled():
                    self._log("Job cancelled by user", "warning")
                    return

                # Step 2: Download fonts for this category
                self._download_category(category_name)

                self.repository.update_job(
                    self.job_id,
                    fonts_downloaded=self.fonts_downloaded,
                    fonts_failed=self.fonts_failed,
                    categories_done=i + 1
                )

                self._log(f"Completed {category_name}: {self.fonts_downloaded} downloaded, {self.fonts_failed} failed", "success")

            # Complete
            self.repository.update_job(
                self.job_id,
                status='completed',
                completed_at=datetime.now().isoformat()
            )
            self._log(f"Scraping complete! {self.fonts_downloaded} fonts downloaded, {self.fonts_failed} failed", "success")
            logger.info("Job %d completed: %d downloaded", self.job_id, self.fonts_downloaded)

        except Exception as e:
            logger.exception("Job %d failed: %s", self.job_id, e)
            self.repository.update_job(
                self.job_id,
                status='failed',
                error_message=str(e),
                completed_at=datetime.now().isoformat()
            )

    def _scrape_category(self, category_id: str, category_name: str) -> None:
        """Scrape fonts from a category and add to staging."""
        try:
            self._log(f"Scraping {category_name}...", "info")

            # Set up progress callback if adapter supports it
            if hasattr(self.adapter, 'set_progress_callback'):
                self.adapter.set_progress_callback(
                    lambda msg: self._log(f"{category_name}: {msg}", "info")
                )

            fonts = self.adapter.scrape_category(category_id)
            added = self.repository.add_fonts_batch(self.job_id, fonts)
            self.fonts_found += added
            self.repository.update_job(self.job_id, fonts_found=self.fonts_found)
            if added > 0:
                self._log(f"Found {added} new fonts in {category_name}", "success")
            else:
                self._log(f"No new fonts in {category_name} (all already downloaded)", "info")
            logger.info("Found %d new fonts in %s", added, category_name)
        except Exception as e:
            self._log(f"Error scraping {category_name}: {e}", "error")
            logger.error("Error scraping %s: %s", category_name, e)

    def _download_category(self, category_name: str) -> None:
        """Download all pending fonts for a category."""
        pending = self.repository.get_pending_fonts(self.job_id, category_name)
        if not pending:
            self._log(f"No fonts to download in {category_name}", "info")
            return

        self._log(f"Downloading {len(pending)} fonts from {category_name}...", "info")

        # Build lookup dict for font data (needed for registration after download)
        font_lookup = {f['id']: f for f in pending}
        download_count = [0]  # Use list to allow mutation in closure
        log_interval = max(1, len(pending) // 10)  # Log every 10%

        def on_complete(result: DownloadResult):
            try:
                download_count[0] += 1
                if result.success:
                    self.repository.update_font_status(
                        result.font_id, 'completed', file_path=result.file_path
                    )
                    if result.file_path:
                        # Register font using cached data from before download
                        font_data = font_lookup.get(result.font_id)
                        if font_data:
                            self.repository.register_font(
                                font_data['name'], result.file_path, self.adapter.source_name,
                                font_data.get('url'), font_data.get('category')
                            )
                    # Log successful downloads
                    if download_count[0] % log_interval == 0 or download_count[0] <= 3:
                        self._log(f"Downloaded: {result.name}", "success")
                else:
                    self.repository.update_font_status(
                        result.font_id, 'failed', error=result.error
                    )
                    # Log first few failures
                    if self.fonts_failed < 5:
                        self._log(f"Failed: {result.name} - {result.error}", "error")

                # Progress update every 10%
                if download_count[0] % log_interval == 0:
                    pct = int(100 * download_count[0] / len(pending))
                    self._log(f"Progress: {download_count[0]}/{len(pending)} ({pct}%)", "info")

            except Exception as e:
                logger.error("Error in on_complete for %s: %s", result.name, e)

        downloaded, failed = self.downloader.download_batch(
            fonts=pending,
            download_fn=self.adapter.download_font,
            on_complete=on_complete,
            is_cancelled=self._is_cancelled
        )

        self.fonts_downloaded += downloaded
        self.fonts_failed += failed
        self._log(f"Category done: {downloaded} downloaded, {failed} failed", "info")
        logger.info("Category %s: %d downloaded, %d failed",
                    category_name, downloaded, failed)


# =============================================================================
# Scraper Registry - Manages Active Workers
# =============================================================================

class ScraperRegistry:
    """Manages active scraper workers.

    Thread-safe registry for starting, stopping, and tracking workers.
    Replaces global mutable state with encapsulated instance.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path(__file__).parent / 'fonts.db')
        self._workers: dict[int, ScraperWorker] = {}
        self._lock = threading.Lock()

    def start(self, source: str) -> int:
        """Start a scraper for the given source.

        Args:
            source: Scraper source ('dafont', 'fontspace', 'google')

        Returns:
            The job ID.
        """
        repository = SQLiteJobRepository(self.db_path)
        job_id = repository.create_job(source)

        adapter = create_adapter(source)
        worker = ScraperWorker(adapter, repository, job_id)
        worker.start()

        with self._lock:
            self._workers[job_id] = worker

        return job_id

    def stop(self, job_id: int) -> bool:
        """Stop a running scraper.

        Returns:
            True if cancelled, False if not found or already stopped.
        """
        with self._lock:
            worker = self._workers.get(job_id)
            if worker:
                worker.cancel()
                return True

        # Try database update even if worker not in registry
        repository = SQLiteJobRepository(self.db_path)
        job = repository.get_job(job_id)
        if job and job['status'] in ('pending', 'running'):
            repository.update_job(
                job_id,
                status='cancelled',
                completed_at=datetime.now().isoformat()
            )
            return True
        return False

    def get_status(self, job_id: int) -> dict | None:
        """Get status for a job."""
        repository = SQLiteJobRepository(self.db_path)
        return repository.get_job(job_id)

    def get_active_jobs(self) -> list[int]:
        """Get list of active job IDs."""
        with self._lock:
            return [jid for jid, w in self._workers.items() if w.is_running()]

    def cleanup(self) -> None:
        """Remove completed workers from registry."""
        with self._lock:
            completed = [jid for jid, w in self._workers.items() if not w.is_running()]
            for jid in completed:
                del self._workers[jid]


# =============================================================================
# Module-level convenience functions (backward compatible)
# =============================================================================

_default_registry: ScraperRegistry | None = None


def _get_registry() -> ScraperRegistry:
    """Get or create the default registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ScraperRegistry()
    return _default_registry


def start_scraper(source: str, db_path: str = None) -> int:
    """Start a scraper for the given source.

    Args:
        source: Scraper source ('dafont', 'fontspace', 'google')
        db_path: Optional database path (ignored, uses default)

    Returns:
        The job ID.
    """
    return _get_registry().start(source)


def stop_scraper(job_id: int) -> bool:
    """Stop a running scraper.

    Returns:
        True if cancelled, False if not found or already stopped.
    """
    return _get_registry().stop(job_id)


def get_active_workers() -> list[int]:
    """Get list of active job IDs."""
    return _get_registry().get_active_jobs()


def cleanup_workers() -> None:
    """Remove completed workers from registry."""
    _get_registry().cleanup()
