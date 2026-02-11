"""Flask application setup and database utilities for the Stroke Editor.

This module serves as the central configuration hub for the Stroke Editor Flask
application. It provides:

    - The Flask application instance shared across all route modules
    - Database connection utilities with context management
    - Global constants for rendering, optimization, and DiffVG parameters
    - Common validation and path resolution helpers
    - Template filters for Jinja2 rendering

The Stroke Editor is a web-based tool for viewing and editing InkSight stroke
data for font characters. This module is imported by the main entry point
(stroke_editor.py) and by all route handler modules.

Architecture:
    - stroke_flask.py: App instance, config, and utilities (this module)
    - stroke_editor.py: Main entry point that registers routes
    - stroke_routes_core.py: Core routes for character editing and rendering
    - stroke_routes_batch.py: Batch processing routes
    - stroke_routes_stream.py: Server-Sent Events (SSE) streaming routes

Example:
    Import the Flask app and database utilities::

        from stroke_flask import app, get_db, get_db_context

        @app.route('/my-route')
        def my_handler():
            with get_db_context() as db:
                result = db.execute("SELECT * FROM fonts").fetchall()
            return jsonify(result)

Attributes:
    BASE_DIR (str): Absolute path to the directory containing this module.
    DB_PATH (str): Absolute path to the SQLite database file (fonts.db).
    app (Flask): The Flask application instance.
    CHARS (str): Default character set for processing (A-Z, a-z, 0-9).
    STROKE_COLORS (list): RGB tuples for stroke visualization colors.
    DEFAULT_CANVAS_SIZE (int): Default canvas size in pixels (224).
    DEFAULT_FONT_SIZE (int): Default font rendering size (200).
    DEFAULT_STROKE_WIDTH (float): Default stroke width for rendering (8.0).
    OPTIMIZATION_TIME_BUDGET (float): Max optimization time in seconds (3600).
    CONVERGENCE_THRESHOLD (float): Threshold for optimization convergence (0.001).
    STALE_CYCLES_LIMIT (int): Max cycles without improvement before stopping (2).
    DIFFVG_ITERATIONS (int): Number of DiffVG optimization iterations (500).
    DIFFVG_TIMEOUT (int): DiffVG operation timeout in seconds (300).
"""

import logging
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from urllib.parse import quote as urlquote

from flask import Flask

# Base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'fonts.db')

# Module logger
logger = logging.getLogger(__name__)


def configure_logging(level: str = 'INFO', log_file: str | None = None) -> None:
    """Configure application-wide logging.

    Sets up structured logging with consistent format across all modules.
    Call this at application startup before importing route modules.

    Args:
        level: Log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to log file. If None, logs to stderr only.

    Example:
        Configure at startup::

            from stroke_flask import configure_logging
            configure_logging(level='DEBUG', log_file='stroke_editor.log')
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter with timestamp, level, module, and message
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    logger.info("Logging configured: level=%s, file=%s", level, log_file or 'stderr')

# Flask application
app = Flask(__name__)

# Default characters to process
CHARS = (
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
)

# Stroke colors for visualization
STROKE_COLORS = [
    (255, 80, 80), (80, 180, 255), (80, 220, 80), (255, 180, 40),
    (200, 100, 255), (255, 120, 200), (100, 220, 220), (180, 180, 80),
]


# --- Global constants ---
# Rendering (imported from stroke_rendering to avoid circular import)
from stroke_rendering import DEFAULT_CANVAS_SIZE, DEFAULT_FONT_SIZE, resolve_font_path
DEFAULT_STROKE_WIDTH = 8.0

# Optimization
OPTIMIZATION_TIME_BUDGET = 3600.0
CONVERGENCE_THRESHOLD = 0.001
STALE_CYCLES_LIMIT = 2

# DiffVG
DIFFVG_ITERATIONS = 500
DIFFVG_TIMEOUT = 300


@app.template_filter('urlencode')
def urlencode_filter(s: str) -> str:
    """URL encode a string for use in Jinja2 templates.

    This template filter safely encodes strings for inclusion in URLs,
    escaping all special characters including slashes and spaces.

    Args:
        s: The value to URL encode. Will be converted to string if not already.

    Returns:
        str: The URL-encoded string with all special characters escaped.

    Example:
        In a Jinja2 template::

            <a href="/search?q={{ query|urlencode }}">Search</a>
    """
    return urlquote(str(s), safe='')


def get_db() -> sqlite3.Connection:
    """Create and return a new database connection.

    Creates a connection to the SQLite database with Row factory enabled,
    allowing column access by name (e.g., row['column_name']).

    Note:
        The caller is responsible for closing the connection when done.
        For automatic connection management, use get_db_context() instead.

    Returns:
        sqlite3.Connection: A database connection with row_factory set to
            sqlite3.Row for dict-like row access.

    Example:
        Basic usage with manual cleanup::

            db = get_db()
            try:
                fonts = db.execute("SELECT * FROM fonts").fetchall()
                for font in fonts:
                    print(font['name'])
            finally:
                db.close()
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db_context() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections with automatic transaction handling.

    Provides a database connection that automatically commits on successful
    completion and rolls back on exception. The connection is always closed
    when exiting the context, regardless of success or failure.

    Yields:
        sqlite3.Connection: A database connection with row_factory set to
            sqlite3.Row for dict-like row access.

    Raises:
        Exception: Re-raises any exception that occurs within the context
            after performing a rollback.

    Example:
        Using the context manager for safe database operations::

            with get_db_context() as db:
                db.execute("INSERT INTO fonts (name) VALUES (?)", ("Arial",))
                db.execute("UPDATE fonts SET source = ? WHERE name = ?",
                          ("system", "Arial"))
            # Changes are committed automatically

        Handling exceptions::

            try:
                with get_db_context() as db:
                    db.execute("INSERT INTO fonts (name) VALUES (?)", ("Arial",))
                    raise ValueError("Something went wrong")
            except ValueError:
                # Changes are rolled back automatically
                pass
    """
    db = get_db()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def ensure_test_tables() -> None:
    """Create test-related database tables if they do not already exist.

    Initializes the database schema for storing test run results and associated
    images. This function is idempotent and safe to call multiple times.

    Tables created:
        test_runs:
            - id (INTEGER PRIMARY KEY): Unique identifier for the test run.
            - font_id (INTEGER NOT NULL): Foreign key to the fonts table.
            - run_date (TEXT): Timestamp of when the test was run.
            - test_chars (TEXT): Characters that were tested (JSON or comma-separated).
            - results (TEXT): Detailed test results (typically JSON).
            - summary (TEXT): Human-readable summary of the test results.

        test_run_images:
            - id (INTEGER PRIMARY KEY): Unique identifier for the image.
            - run_id (INTEGER NOT NULL): Foreign key to test_runs.id.
            - char (TEXT NOT NULL): The character this image represents.
            - image_data (BLOB): Binary image data (typically PNG).

    Example:
        Call at application startup to ensure tables exist::

            from stroke_flask import ensure_test_tables
            ensure_test_tables()
    """
    with get_db_context() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY,
                font_id INTEGER NOT NULL,
                run_date TEXT DEFAULT CURRENT_TIMESTAMP,
                test_chars TEXT,
                results TEXT,
                summary TEXT
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS test_run_images (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL,
                char TEXT NOT NULL,
                image_data BLOB
            )
        """)


def get_font(fid: int) -> sqlite3.Row | None:
    """Retrieve a font record from the database by its ID.

    Args:
        fid: The integer ID of the font to retrieve.

    Returns:
        sqlite3.Row: A row object with font data (id, name, source, file_path,
            etc.) if found, or None if no font exists with the given ID or
            if a database error occurs.

    Example:
        Fetching a font::

            font = get_font(42)
            if font:
                print(f"Font name: {font['name']}")
                print(f"File path: {font['file_path']}")
    """
    with get_db_context() as db:
        try:
            return db.execute("SELECT * FROM fonts WHERE id = ?", (fid,)).fetchone()
        except Exception:
            return None


def validate_char_param(char: str | None) -> tuple[bool, tuple | None]:
    """Validate a character parameter from a request query string.

    Ensures the character parameter is present and contains exactly one
    character. Returns a tuple indicating validity and an optional error
    response for invalid input.

    Args:
        char: The character value from request.args.get('c'), may be None
            or an empty string if not provided.

    Returns:
        tuple: A 2-tuple of (is_valid, error_response) where:
            - is_valid (bool): True if the character is valid, False otherwise.
            - error_response: None if valid, otherwise a tuple of
              (flask.Response, status_code) ready to be returned from a route.

    Example:
        Using in a route handler::

            @app.route('/api/char/<int:fid>')
            def api_get_char(fid):
                c = request.args.get('c')
                ok, err = validate_char_param(c)
                if not ok:
                    return err  # Returns (jsonify(error="..."), 400)
                # Process valid character...
    """
    from flask import jsonify

    if not char:
        return False, (jsonify(error="Missing ?c= parameter"), 400)
    if len(char) != 1:
        return False, (jsonify(error="Character must be a single character"), 400)
    return True, None


def get_font_or_error(fid: int, response_type: str = 'json'):
    """Get font by ID or return appropriate error response.

    Args:
        fid: Font ID to look up.
        response_type: 'json' for JSON error, 'text' for plain text,
            'sse' for Server-Sent Events format.

    Returns:
        tuple: (font, None) if found, or (None, error_response) if not found.
            The error_response is ready to return from a Flask route.

    Example:
        font, err = get_font_or_error(fid)
        if err:
            return err
        # Use font...
    """
    from flask import Response, jsonify
    font = get_font(fid)
    if font:
        return font, None

    if response_type == 'json':
        return None, (jsonify(error="Font not found"), 404)
    elif response_type == 'sse':
        import json
        return None, Response(f'data: {json.dumps({"error": "Font not found"})}\n\n',
                              mimetype='text/event-stream')
    else:
        return None, ("Font not found", 404)


def get_font_and_mask(fid: int, char: str, canvas_size: int = 224):
    """Get font and render glyph mask, with error handling.

    Combines font lookup and mask rendering into a single call with
    appropriate error responses for common failure cases.

    Args:
        fid: Font ID to look up.
        char: Character to render.
        canvas_size: Canvas size in pixels (default 224).

    Returns:
        tuple: (font, mask, None) on success, or
               (None, None, error_response) on failure.
            The error_response is ready to return from a Flask route.

    Example:
        font, mask, err = get_font_and_mask(fid, c)
        if err:
            return err
        # Use font and mask...
    """
    from flask import jsonify
    from stroke_rendering import render_glyph_mask

    font = get_font(fid)
    if not font:
        return None, None, (jsonify(error="Font not found"), 404)

    mask = render_glyph_mask(font['file_path'], char, canvas_size)
    if mask is None:
        return None, None, (jsonify(error="Could not render glyph"), 500)

    return font, mask, None


def send_pil_image_as_png(img):
    """Convert PIL Image to PNG and send as Flask response.

    Args:
        img: PIL Image object to send.

    Returns:
        flask.Response: Response with PNG data and correct mimetype.

    Example:
        img = Image.new('RGB', (100, 100), 'white')
        return send_pil_image_as_png(img)
    """
    import io
    from flask import send_file
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


class FontRepository:
    """Data access layer for font and character database operations.

    Encapsulates all database queries related to fonts and characters,
    providing a clean interface for route handlers and enabling easier
    testing through dependency injection.

    The repository uses the module-level get_db_context() for connection
    management, but can accept a custom connection factory for testing.

    Attributes:
        _connection_factory: Callable that returns a context manager
            yielding a database connection.

    Example:
        Using the default connection factory::

            repo = FontRepository()
            fonts = repo.list_fonts()
            font = repo.get_font_by_id(42)

        Using a custom connection factory for testing::

            def mock_db_context():
                return mock_connection

            repo = FontRepository(connection_factory=mock_db_context)
    """

    # Rejection reason ID for manually rejected fonts
    REJECTION_REASON_ID = 8
    # Duplicate reason ID
    DUPLICATE_REASON_ID = 2

    def __init__(self, connection_factory=None):
        """Initialize the repository with an optional connection factory.

        Args:
            connection_factory: Optional callable that returns a context
                manager yielding a database connection. Defaults to
                get_db_context.
        """
        self._connection_factory = connection_factory or get_db_context

    def list_fonts(self, show_rejected: bool = False) -> list:
        """List fonts with optional filtering for rejected fonts.

        Args:
            show_rejected: If True, show only rejected fonts.
                If False (default), show non-rejected, non-duplicate fonts.

        Returns:
            List of font records with id, name, source, file_path,
            char_count, and rejected flag.
        """
        with self._connection_factory() as db:
            if show_rejected:
                query = """
                    SELECT f.id, f.name, f.source, f.file_path,
                           COALESCE(cs.char_count, 0) as char_count,
                           1 as rejected
                    FROM fonts f
                    JOIN font_removals fr ON fr.font_id = f.id AND fr.reason_id = ?
                    LEFT JOIN (
                        SELECT font_id, COUNT(*) as char_count
                        FROM characters WHERE strokes_raw IS NOT NULL
                        GROUP BY font_id
                    ) cs ON cs.font_id = f.id
                    ORDER BY f.name
                """
                return db.execute(query, (self.REJECTION_REASON_ID,)).fetchall()
            else:
                query = """
                    SELECT f.id, f.name, f.source, f.file_path,
                           COALESCE(cs.char_count, 0) as char_count,
                           0 as rejected
                    FROM fonts f
                    LEFT JOIN font_removals rej ON rej.font_id = f.id AND rej.reason_id = ?
                    LEFT JOIN font_removals dup ON dup.font_id = f.id AND dup.reason_id = ?
                    LEFT JOIN (
                        SELECT font_id, COUNT(*) as char_count
                        FROM characters WHERE strokes_raw IS NOT NULL
                        GROUP BY font_id
                    ) cs ON cs.font_id = f.id
                    WHERE rej.id IS NULL AND dup.id IS NULL
                    ORDER BY f.name
                """
                return db.execute(query, (self.REJECTION_REASON_ID,
                                          self.DUPLICATE_REASON_ID)).fetchall()

    def get_font_by_id(self, fid: int):
        """Get a font record by ID.

        Args:
            fid: The font ID.

        Returns:
            Font record or None if not found.
        """
        with self._connection_factory() as db:
            return db.execute(
                "SELECT * FROM fonts WHERE id = ?", (fid,)
            ).fetchone()

    def get_font_characters(self, fid: int) -> list:
        """Get all characters with stroke data for a font.

        Args:
            fid: The font ID.

        Returns:
            List of character records with char, strokes_raw, point_count.
        """
        with self._connection_factory() as db:
            return db.execute(
                """SELECT char, strokes_raw, point_count
                   FROM characters
                   WHERE font_id = ? AND strokes_raw IS NOT NULL
                   ORDER BY char""",
                (fid,)
            ).fetchall()

    def get_character(self, fid: int, char: str):
        """Get a specific character's stroke data.

        Args:
            fid: The font ID.
            char: The character.

        Returns:
            Character record with strokes_raw and markers, or None.
        """
        with self._connection_factory() as db:
            return db.execute(
                "SELECT strokes_raw, markers FROM characters WHERE font_id = ? AND char = ?",
                (fid, char)
            ).fetchone()

    def save_character(self, fid: int, char: str, strokes_raw: str,
                       point_count: int, markers: str) -> None:
        """Save or update character stroke data.

        Args:
            fid: The font ID.
            char: The character.
            strokes_raw: JSON string of stroke data.
            point_count: Total number of points in strokes.
            markers: JSON string of marker data.
        """
        with self._connection_factory() as db:
            existing = db.execute(
                "SELECT id FROM characters WHERE font_id = ? AND char = ?",
                (fid, char)
            ).fetchone()
            if existing:
                db.execute(
                    """UPDATE characters
                       SET strokes_raw = ?, point_count = ?, markers = ?
                       WHERE font_id = ? AND char = ?""",
                    (strokes_raw, point_count, markers, fid, char)
                )
            else:
                db.execute(
                    """INSERT INTO characters (font_id, char, strokes_raw, point_count, markers)
                       VALUES (?, ?, ?, ?, ?)""",
                    (fid, char, strokes_raw, point_count, markers)
                )

    def get_character_strokes(self, fid: int, char: str):
        """Get just the strokes_raw for a character.

        Args:
            fid: The font ID.
            char: The character.

        Returns:
            Character record with strokes_raw, or None.
        """
        with self._connection_factory() as db:
            return db.execute(
                "SELECT strokes_raw FROM characters WHERE font_id = ? AND char = ?",
                (fid, char)
            ).fetchone()

    def list_fonts_for_scan(self) -> list:
        """List non-rejected fonts for batch scanning.

        Returns:
            List of font records with id and file_path.
        """
        with self._connection_factory() as db:
            return db.execute(
                """SELECT f.id, f.file_path FROM fonts f
                   LEFT JOIN font_removals fr ON fr.font_id = f.id AND fr.reason_id = ?
                   WHERE fr.id IS NULL""",
                (self.REJECTION_REASON_ID,)
            ).fetchall()

    def reject_font(self, fid: int, details: str) -> bool:
        """Mark a font as rejected.

        Args:
            fid: The font ID.
            details: Rejection details/reason.

        Returns:
            True if rejected, False if font not found or already rejected.
        """
        with self._connection_factory() as db:
            if not db.execute("SELECT id FROM fonts WHERE id = ?", (fid,)).fetchone():
                return False
            if db.execute(
                "SELECT id FROM font_removals WHERE font_id = ? AND reason_id = ?",
                (fid, self.REJECTION_REASON_ID)
            ).fetchone():
                return False
            db.execute(
                "INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, ?, ?)",
                (fid, self.REJECTION_REASON_ID, details)
            )
            return True

    def unreject_font(self, fid: int) -> int:
        """Remove rejection for a font.

        Args:
            fid: The font ID.

        Returns:
            Number of rows deleted (0 or 1).
        """
        with self._connection_factory() as db:
            cursor = db.execute(
                "DELETE FROM font_removals WHERE font_id = ? AND reason_id = ?",
                (fid, self.REJECTION_REASON_ID)
            )
            return cursor.rowcount

    def unreject_all_fonts(self) -> int:
        """Remove all font rejections.

        Returns:
            Number of fonts unrejected.
        """
        with self._connection_factory() as db:
            cursor = db.execute(
                "DELETE FROM font_removals WHERE reason_id = ?",
                (self.REJECTION_REASON_ID,)
            )
            return cursor.rowcount


# Default repository instance for convenience
font_repository = FontRepository()


class TestRunRepository:
    """Repository for test run data (test_runs table).

    Encapsulates database operations for storing and retrieving
    stroke generation test results.
    """

    def __init__(self, connection_factory=None):
        """Initialize the repository.

        Args:
            connection_factory: Callable that returns a DB context manager.
                Defaults to get_db_context.
        """
        self._connection_factory = connection_factory or get_db_context

    def save_run(self, font_id: int, run_date: str, chars_tested: int,
                 chars_ok: int, avg_score: float, avg_coverage: float,
                 avg_overshoot: float, avg_stroke_count: float,
                 avg_topology: float, results_json: str) -> int:
        """Save a test run result.

        Returns:
            The ID of the inserted run.
        """
        with self._connection_factory() as db:
            cursor = db.execute(
                """INSERT INTO test_runs
                   (font_id, run_date, chars_tested, chars_ok, avg_score,
                    avg_coverage, avg_overshoot, avg_stroke_count, avg_topology, results_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (font_id, run_date, chars_tested, chars_ok, avg_score,
                 avg_coverage, avg_overshoot, avg_stroke_count, avg_topology, results_json)
            )
            return cursor.lastrowid

    def get_history(self, font_id: int, limit: int = 10) -> list:
        """Get test run history for a font.

        Args:
            font_id: The font ID.
            limit: Maximum number of runs to return.

        Returns:
            List of test run records, most recent first.
        """
        with self._connection_factory() as db:
            return db.execute(
                """SELECT id, run_date, chars_tested, chars_ok, avg_score,
                          avg_coverage, avg_overshoot, avg_stroke_count, avg_topology
                   FROM test_runs WHERE font_id = ?
                   ORDER BY run_date DESC LIMIT ?""",
                (font_id, limit)
            ).fetchall()

    def get_run(self, run_id: int):
        """Get a specific test run by ID.

        Returns:
            Test run record or None.
        """
        with self._connection_factory() as db:
            return db.execute(
                "SELECT * FROM test_runs WHERE id = ?", (run_id,)
            ).fetchone()

    def get_recent_runs(self, font_id: int, count: int = 2) -> list:
        """Get the most recent N run IDs for a font.

        Returns:
            List of run IDs.
        """
        with self._connection_factory() as db:
            rows = db.execute(
                "SELECT id FROM test_runs WHERE font_id = ? ORDER BY run_date DESC LIMIT ?",
                (font_id, count)
            ).fetchall()
            return [r['id'] for r in rows]


# Default test run repository instance
test_run_repository = TestRunRepository()
