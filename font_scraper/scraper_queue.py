"""Queue-based scraper architecture with work-stealing pattern.

This module provides a resilient scraper architecture with:
- Discovery thread: Scrapes font listings and adds to queue
- Download workers: Pool of workers that pull from queue (work-stealing)
- Full recovery: Can resume from any failure point

Architecture:
    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │ DiscoveryWorker │────▶│ scraper_fonts    │◀────│ DownloadWorker  │
    │ (1 thread)      │     │ (queue table)    │     │ (N threads)     │
    └─────────────────┘     └──────────────────┘     └─────────────────┘

Usage:
    from scraper_queue import ScraperQueue

    queue = ScraperQueue()
    queue.start('dafont', num_workers=4)

    # Check status
    status = queue.get_status()

    # Stop gracefully
    queue.stop()
"""

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Database Repository
# =============================================================================

class QueueRepository:
    """Database operations for the queue-based scraper.

    Thread-safe with connection-per-operation pattern.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path(__file__).parent / 'fonts.db')
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=30000')
        return conn

    def _ensure_schema(self) -> None:
        """Ensure required columns exist (for migration from old schema)."""
        conn = self._get_conn()
        try:
            # Check if current_page column exists
            cursor = conn.execute("PRAGMA table_info(scraper_jobs)")
            columns = {row['name'] for row in cursor.fetchall()}

            if 'current_page' not in columns:
                conn.execute(
                    "ALTER TABLE scraper_jobs ADD COLUMN current_page INTEGER DEFAULT 0"
                )
                conn.commit()
                logger.info("Added current_page column to scraper_jobs table")
        except Exception as e:
            logger.warning("Schema migration check failed: %s", e)
        finally:
            conn.close()

    # --- Job Management ---

    def create_job(self, source: str) -> int:
        """Create a new scraper job."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO scraper_jobs
                   (source, status, current_page, categories_done)
                   VALUES (?, 'pending', 0, 0)""",
                (source,)
            )
            conn.commit()
            return cursor.lastrowid
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

    def get_active_job(self, source: str) -> dict | None:
        """Get the most recent active or paused job for a source."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT * FROM scraper_jobs
                   WHERE source = ? AND status IN ('pending', 'running', 'discovering', 'downloading', 'paused')
                   ORDER BY id DESC LIMIT 1""",
                (source,)
            ).fetchone()
            return dict(row) if row else None
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

    def get_recent_jobs(self, limit: int = 10) -> list:
        """Get recent jobs with actual counts from queue."""
        conn = self._get_conn()
        try:
            jobs = conn.execute(
                "SELECT * FROM scraper_jobs ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()

            result = []
            for job in jobs:
                job_dict = dict(job)
                job_id = job_dict['id']

                # Get real counts from scraper_fonts
                counts = conn.execute(
                    """SELECT status, COUNT(*) as cnt FROM scraper_fonts
                       WHERE job_id = ? GROUP BY status""",
                    (job_id,)
                ).fetchall()

                count_map = {row['status']: row['cnt'] for row in counts}
                job_dict['fonts_downloaded'] = count_map.get('completed', 0)
                job_dict['fonts_failed'] = count_map.get('failed', 0)
                job_dict['fonts_pending'] = count_map.get('pending', 0)
                job_dict['fonts_downloading'] = count_map.get('downloading', 0)

                result.append(job_dict)

            return result
        finally:
            conn.close()

    # --- Queue Operations ---

    def add_to_queue(self, job_id: int, fonts: list[dict]) -> int:
        """Add fonts to the download queue. Returns count of new fonts added."""
        if not fonts:
            return 0

        conn = self._get_conn()
        try:
            # Get existing font names for this job to avoid duplicates
            existing = set()
            for row in conn.execute(
                "SELECT name FROM scraper_fonts WHERE job_id = ?", (job_id,)
            ):
                existing.add(row[0])

            # Also check fonts already in the main fonts table
            source = conn.execute(
                "SELECT source FROM scraper_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if source:
                for row in conn.execute(
                    "SELECT name FROM fonts WHERE source = ?", (source[0],)
                ):
                    existing.add(row[0])

            new_fonts = [f for f in fonts if f['name'] not in existing]
            if not new_fonts:
                return 0

            conn.executemany(
                """INSERT INTO scraper_fonts
                   (job_id, name, url, download_url, category, status)
                   VALUES (?, ?, ?, ?, ?, 'pending')""",
                [(job_id, f['name'], f.get('url'), f.get('download_url'), f.get('category'))
                 for f in new_fonts]
            )
            conn.commit()
            return len(new_fonts)
        finally:
            conn.close()

    def claim_next_font(self, job_id: int) -> dict | None:
        """Atomically claim the next pending font for download.

        Uses UPDATE with LIMIT to ensure only one worker gets each font.
        Returns the claimed font or None if queue is empty.
        """
        conn = self._get_conn()
        try:
            # Find and claim in one transaction
            conn.execute("BEGIN IMMEDIATE")

            row = conn.execute(
                """SELECT id, name, url, download_url, category
                   FROM scraper_fonts
                   WHERE job_id = ? AND status = 'pending'
                   LIMIT 1""",
                (job_id,)
            ).fetchone()

            if not row:
                conn.execute("ROLLBACK")
                return None

            font_id = row['id']
            conn.execute(
                "UPDATE scraper_fonts SET status = 'downloading' WHERE id = ?",
                (font_id,)
            )
            conn.execute("COMMIT")

            return {
                'id': font_id,
                'name': row['name'],
                'url': row['url'],
                'download_url': row['download_url'],
                'category': row['category']
            }
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def complete_font(self, font_id: int, file_path: str = None) -> None:
        """Mark a font as successfully downloaded."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE scraper_fonts SET status = 'completed', file_path = ? WHERE id = ?",
                (file_path, font_id)
            )
            conn.commit()
        finally:
            conn.close()

    def fail_font(self, font_id: int, error: str = None) -> None:
        """Mark a font as failed."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE scraper_fonts SET status = 'failed', error_message = ? WHERE id = ?",
                (error, font_id)
            )
            conn.commit()
        finally:
            conn.close()

    def reset_downloading(self, job_id: int) -> int:
        """Reset any 'downloading' fonts back to 'pending' (recovery)."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "UPDATE scraper_fonts SET status = 'pending' WHERE job_id = ? AND status = 'downloading'",
                (job_id,)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_queue_stats(self, job_id: int) -> dict:
        """Get queue statistics for a job."""
        conn = self._get_conn()
        try:
            counts = conn.execute(
                """SELECT status, COUNT(*) as cnt FROM scraper_fonts
                   WHERE job_id = ? GROUP BY status""",
                (job_id,)
            ).fetchall()
            return {row['status']: row['cnt'] for row in counts}
        finally:
            conn.close()

    # --- Font Registration ---

    def register_font(self, name: str, file_path: str, source: str,
                      url: str = None, category: str = None) -> int | None:
        """Register a downloaded font in the main fonts table."""
        conn = self._get_conn()
        try:
            # Check if already exists
            existing = conn.execute(
                "SELECT id FROM fonts WHERE file_path = ?", (file_path,)
            ).fetchone()
            if existing:
                return existing[0]

            cursor = conn.execute(
                """INSERT INTO fonts (name, file_path, source, url, category)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, file_path, source, url, category)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Race condition - another thread inserted
            existing = conn.execute(
                "SELECT id FROM fonts WHERE file_path = ?", (file_path,)
            ).fetchone()
            return existing[0] if existing else None
        finally:
            conn.close()

    # --- Logging ---

    def add_log(self, job_id: int, message: str, level: str = 'info') -> None:
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

    def get_logs(self, job_id: int = None, limit: int = 100) -> list:
        """Get recent log entries, optionally filtered by job_id."""
        conn = self._get_conn()
        try:
            if job_id:
                rows = conn.execute(
                    """SELECT id, job_id, level, message, created_at
                       FROM scraper_logs
                       WHERE job_id = ?
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
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # --- Additional Job Queries ---

    def get_active_jobs(self) -> list:
        """Get all jobs with active or paused status."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM scraper_jobs
                   WHERE status IN ('pending', 'running', 'discovering', 'downloading', 'paused')
                   ORDER BY id DESC"""
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_font_counts(self, job_id: int) -> dict:
        """Get font status counts for a job."""
        conn = self._get_conn()
        try:
            counts = conn.execute(
                """SELECT status, COUNT(*) as cnt FROM scraper_fonts
                   WHERE job_id = ? GROUP BY status""",
                (job_id,)
            ).fetchall()
            return {row['status']: row['cnt'] for row in counts}
        finally:
            conn.close()


# =============================================================================
# Discovery Worker
# =============================================================================

class DiscoveryWorker:
    """Scrapes font listings and adds them to the download queue.

    Runs in a single thread. Tracks progress by category and page number
    to enable recovery from failures.
    """

    def __init__(self, source: str, job_id: int, repo: QueueRepository):
        self.source = source
        self.job_id = job_id
        self.repo = repo
        self._thread = None
        self._stop_event = threading.Event()
        self._adapter = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _get_adapter(self):
        """Lazy-load the scraper adapter."""
        if self._adapter:
            return self._adapter

        from scraper_worker import create_adapter
        self._adapter = create_adapter(self.source)
        return self._adapter

    def _run(self) -> None:
        logger.info("Discovery worker starting for %s (job %d)", self.source, self.job_id)
        self.repo.add_log(self.job_id, f"Discovery started for {self.source}", "info")

        try:
            self.repo.update_job(
                self.job_id,
                status='discovering',
                started_at=datetime.now().isoformat()
            )

            adapter = self._get_adapter()
            categories = adapter.get_categories()

            # Get resume point
            job = self.repo.get_job(self.job_id)
            start_category = job.get('categories_done', 0) or 0

            self.repo.update_job(self.job_id, categories_total=len(categories))

            for i, category_id in enumerate(categories):
                if i < start_category:
                    continue  # Skip already-completed categories

                if self._stop_event.is_set():
                    self.repo.add_log(self.job_id, "Discovery stopped by user", "warning")
                    return

                category_name = adapter.get_category_name(category_id)
                self.repo.update_job(
                    self.job_id,
                    current_category=category_name,
                    current_page=0
                )

                self.repo.add_log(
                    self.job_id,
                    f"Discovering category {i+1}/{len(categories)}: {category_name}",
                    "info"
                )

                # Scrape the category
                try:
                    fonts = adapter.scrape_category(category_id)
                    added = self.repo.add_to_queue(self.job_id, fonts)

                    if added > 0:
                        self.repo.add_log(
                            self.job_id,
                            f"Found {added} new fonts in {category_name}",
                            "success"
                        )
                        self.repo.update_job(
                            self.job_id,
                            fonts_found=(job.get('fonts_found', 0) or 0) + added
                        )
                        # Refresh job for next iteration
                        job = self.repo.get_job(self.job_id)
                    else:
                        self.repo.add_log(
                            self.job_id,
                            f"No new fonts in {category_name}",
                            "info"
                        )
                except Exception as e:
                    logger.error("Error scraping %s: %s", category_name, e)
                    self.repo.add_log(
                        self.job_id,
                        f"Error scraping {category_name}: {e}",
                        "error"
                    )

                # Mark category complete
                self.repo.update_job(self.job_id, categories_done=i + 1)

            # Discovery complete
            self.repo.update_job(self.job_id, status='downloading')
            self.repo.add_log(self.job_id, "Discovery complete", "success")
            logger.info("Discovery complete for job %d", self.job_id)

        except Exception as e:
            logger.exception("Discovery worker failed: %s", e)
            self.repo.update_job(
                self.job_id,
                status='failed',
                error_message=f"Discovery failed: {e}"
            )
            self.repo.add_log(self.job_id, f"Discovery failed: {e}", "error")


# =============================================================================
# Download Worker
# =============================================================================

class DownloadWorker:
    """Downloads fonts from the queue using work-stealing pattern.

    Multiple workers can run in parallel. Each worker claims fonts
    atomically from the queue and downloads them.
    """

    def __init__(self, worker_id: int, source: str, job_id: int,
                 repo: QueueRepository, on_complete: Callable = None):
        self.worker_id = worker_id
        self.source = source
        self.job_id = job_id
        self.repo = repo
        self.on_complete = on_complete
        self._thread = None
        self._stop_event = threading.Event()
        self._adapter = None

        # Stats
        self.downloaded = 0
        self.failed = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _get_adapter(self):
        if self._adapter:
            return self._adapter
        from scraper_worker import create_adapter
        self._adapter = create_adapter(self.source)
        return self._adapter

    def _run(self) -> None:
        logger.info("Download worker %d starting for job %d", self.worker_id, self.job_id)
        adapter = self._get_adapter()

        consecutive_empty = 0

        while not self._stop_event.is_set():
            # Try to claim a font from the queue
            font = self.repo.claim_next_font(self.job_id)

            if not font:
                consecutive_empty += 1

                # Check if discovery is still running or there's more work coming
                job = self.repo.get_job(self.job_id)
                if job and job['status'] in ('completed', 'failed', 'cancelled', 'paused'):
                    break

                # If discovery is done and queue is empty, we're done
                if job and job['status'] == 'downloading':
                    stats = self.repo.get_queue_stats(self.job_id)
                    if stats.get('pending', 0) == 0 and stats.get('downloading', 0) == 0:
                        break

                # Back off if queue is empty
                if consecutive_empty > 10:
                    time.sleep(1)
                else:
                    time.sleep(0.1)
                continue

            consecutive_empty = 0

            # Download the font
            try:
                success, file_path = adapter.download_font(font)

                if success:
                    self.repo.complete_font(font['id'], file_path)
                    self.downloaded += 1

                    # Register in fonts table
                    if file_path:
                        self.repo.register_font(
                            font['name'], file_path, self.source,
                            font.get('url'), font.get('category')
                        )

                    # Log periodically
                    if self.downloaded % 50 == 0:
                        self.repo.add_log(
                            self.job_id,
                            f"Worker {self.worker_id}: {self.downloaded} downloaded",
                            "info"
                        )
                else:
                    self.repo.fail_font(font['id'], "Download failed")
                    self.failed += 1

            except Exception as e:
                logger.error("Worker %d error downloading %s: %s",
                            self.worker_id, font['name'], e)
                self.repo.fail_font(font['id'], str(e))
                self.failed += 1

        logger.info("Download worker %d finished: %d downloaded, %d failed",
                   self.worker_id, self.downloaded, self.failed)

        if self.on_complete:
            self.on_complete(self.worker_id, self.downloaded, self.failed)


# =============================================================================
# Scraper Queue Coordinator
# =============================================================================

class ScraperQueue:
    """Coordinates discovery and download workers.

    Usage:
        queue = ScraperQueue()
        queue.start('dafont', num_workers=4)

        # Later...
        queue.stop()
    """

    def __init__(self, db_path: str = None):
        self.repo = QueueRepository(db_path)
        self._discovery_worker = None
        self._download_workers: list[DownloadWorker] = []
        self._job_id = None
        self._source = None
        self._lock = threading.Lock()
        self._completion_check_thread = None
        self._stop_event = threading.Event()

    def start(self, source: str, num_workers: int = 4, resume: bool = True) -> int:
        """Start scraping a source.

        Args:
            source: 'dafont', 'fontspace', or 'google'
            num_workers: Number of download workers
            resume: If True, resume existing job; if False, create new job

        Returns:
            The job ID
        """
        with self._lock:
            if self._discovery_worker and self._discovery_worker.is_running():
                raise RuntimeError("Scraper already running")

            # Find existing job or create new one
            if resume:
                job = self.repo.get_active_job(source)
                if job:
                    self._job_id = job['id']
                    logger.info("Resuming job %d for %s", self._job_id, source)
                    self.repo.add_log(self._job_id, f"Resuming job", "info")

                    # Reset any stuck downloads
                    reset = self.repo.reset_downloading(self._job_id)
                    if reset > 0:
                        self.repo.add_log(
                            self._job_id,
                            f"Reset {reset} stuck downloads",
                            "warning"
                        )

            if not self._job_id:
                self._job_id = self.repo.create_job(source)
                logger.info("Created new job %d for %s", self._job_id, source)

            self._source = source
            self._stop_event.clear()

            # Start discovery worker
            self._discovery_worker = DiscoveryWorker(source, self._job_id, self.repo)
            self._discovery_worker.start()

            # Start download workers
            self._download_workers = []
            for i in range(num_workers):
                worker = DownloadWorker(
                    i, source, self._job_id, self.repo,
                    on_complete=self._on_worker_complete
                )
                worker.start()
                self._download_workers.append(worker)

            # Start completion checker
            self._completion_check_thread = threading.Thread(
                target=self._check_completion, daemon=True
            )
            self._completion_check_thread.start()

            return self._job_id

    def stop(self, cancel: bool = True) -> None:
        """Stop all workers gracefully.

        Args:
            cancel: If True, mark job as cancelled. If False, mark as paused.
        """
        with self._lock:
            self._stop_event.set()

            if self._discovery_worker:
                self._discovery_worker.stop()

            for worker in self._download_workers:
                worker.stop()

            if self._job_id:
                if cancel:
                    self.repo.update_job(
                        self._job_id,
                        status='cancelled',
                        completed_at=datetime.now().isoformat()
                    )
                    self.repo.add_log(self._job_id, "Job cancelled by user", "warning")
                else:
                    self.repo.update_job(self._job_id, status='paused')
                    self.repo.add_log(self._job_id, "Job paused by user", "info")

            self._job_id = None
            self._discovery_worker = None
            self._download_workers = []

    def pause(self) -> bool:
        """Pause the current job (can be resumed later)."""
        if not self.is_running():
            return False
        self.stop(cancel=False)
        return True

    def resume_job(self, job_id: int, num_workers: int = 4) -> bool:
        """Resume a paused job.

        Args:
            job_id: The job ID to resume
            num_workers: Number of download workers

        Returns:
            True if job was resumed, False if not found or not paused
        """
        job = self.repo.get_job(job_id)
        if not job:
            return False

        if job['status'] not in ('paused', 'discovering', 'downloading'):
            return False

        with self._lock:
            if self.is_running():
                return False  # Already running something

            self._job_id = job_id
            self._source = job['source']
            self._stop_event.clear()

            # Reset any stuck downloads
            reset = self.repo.reset_downloading(job_id)
            if reset > 0:
                self.repo.add_log(job_id, f"Reset {reset} stuck downloads", "warning")

            self.repo.add_log(job_id, "Job resumed", "info")

            # Check if discovery needs to continue
            discovery_done = job['status'] == 'downloading' or (
                job.get('categories_done', 0) >= job.get('categories_total', 0) and
                job.get('categories_total', 0) > 0
            )

            if not discovery_done:
                self.repo.update_job(job_id, status='discovering')
                self._discovery_worker = DiscoveryWorker(self._source, job_id, self.repo)
                self._discovery_worker.start()
            else:
                self.repo.update_job(job_id, status='downloading')

            # Start download workers
            self._download_workers = []
            for i in range(num_workers):
                worker = DownloadWorker(
                    i, self._source, job_id, self.repo,
                    on_complete=self._on_worker_complete
                )
                worker.start()
                self._download_workers.append(worker)

            # Start completion checker
            self._completion_check_thread = threading.Thread(
                target=self._check_completion, daemon=True
            )
            self._completion_check_thread.start()

            return True

    def get_status(self) -> dict | None:
        """Get current job status."""
        if not self._job_id:
            return None
        return self.repo.get_job(self._job_id)

    def is_running(self) -> bool:
        """Check if any workers are running."""
        if self._discovery_worker and self._discovery_worker.is_running():
            return True
        return any(w.is_running() for w in self._download_workers)

    def _on_worker_complete(self, worker_id: int, downloaded: int, failed: int) -> None:
        """Called when a download worker finishes."""
        logger.info("Worker %d complete: %d downloaded, %d failed",
                   worker_id, downloaded, failed)

    def _check_completion(self) -> None:
        """Background thread to check for job completion."""
        while not self._stop_event.is_set():
            time.sleep(2)

            if not self._job_id:
                continue

            job = self.repo.get_job(self._job_id)
            if not job or job['status'] in ('completed', 'failed', 'cancelled', 'paused'):
                continue

            # Check if discovery is done
            discovery_done = (
                not self._discovery_worker or
                not self._discovery_worker.is_running()
            )

            # Check if all downloads are done
            if discovery_done:
                stats = self.repo.get_queue_stats(self._job_id)
                pending = stats.get('pending', 0)
                downloading = stats.get('downloading', 0)

                if pending == 0 and downloading == 0:
                    # All done
                    all_workers_done = all(
                        not w.is_running() for w in self._download_workers
                    )

                    if all_workers_done:
                        completed = stats.get('completed', 0)
                        failed = stats.get('failed', 0)

                        self.repo.update_job(
                            self._job_id,
                            status='completed',
                            completed_at=datetime.now().isoformat()
                        )
                        self.repo.add_log(
                            self._job_id,
                            f"Job complete: {completed} downloaded, {failed} failed",
                            "success"
                        )
                        logger.info("Job %d completed", self._job_id)


# =============================================================================
# Module-level interface (for API compatibility)
# =============================================================================

_default_queue: ScraperQueue | None = None


def get_queue() -> ScraperQueue:
    global _default_queue
    if _default_queue is None:
        _default_queue = ScraperQueue()
    return _default_queue


def start_scraper(source: str, num_workers: int = 4) -> int:
    """Start a scraper job."""
    return get_queue().start(source, num_workers=num_workers)


def stop_scraper(job_id: int = None) -> bool:
    """Stop the current scraper (cancels the job)."""
    queue = get_queue()
    if queue.is_running():
        queue.stop(cancel=True)
        return True
    return False


def pause_scraper() -> bool:
    """Pause the current scraper (can be resumed later).

    Works even if Flask reloaded and lost in-memory state.
    """
    queue = get_queue()

    # First try in-memory state
    if queue.is_running():
        queue.pause()
        return True

    # Fall back to database state - find any active jobs and pause them
    repo = QueueRepository()
    active_jobs = repo.get_active_jobs()
    paused_any = False

    for job in active_jobs:
        if job['status'] in ('discovering', 'downloading', 'running', 'pending'):
            repo.update_job(job['id'], status='paused')
            repo.add_log(job['id'], "Job paused by user", "info")
            paused_any = True

    return paused_any


def resume_scraper(job_id: int, num_workers: int = 4) -> bool:
    """Resume a paused or interrupted job."""
    return get_queue().resume_job(job_id, num_workers)


def get_scraper_status() -> dict | None:
    """Get current scraper status."""
    return get_queue().get_status()
