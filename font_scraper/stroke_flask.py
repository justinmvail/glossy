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
from contextlib import contextmanager
from urllib.parse import quote as urlquote

from flask import Flask

# Base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'fonts.db')

# Configure logging
logger = logging.getLogger(__name__)

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
# Rendering
DEFAULT_CANVAS_SIZE = 224
DEFAULT_FONT_SIZE = 200
DEFAULT_STROKE_WIDTH = 8.0

# Optimization
OPTIMIZATION_TIME_BUDGET = 3600.0
CONVERGENCE_THRESHOLD = 0.001
STALE_CYCLES_LIMIT = 2

# DiffVG
DIFFVG_ITERATIONS = 500
DIFFVG_TIMEOUT = 300


@app.template_filter('urlencode')
def urlencode_filter(s):
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


def get_db():
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
def get_db_context():
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


def ensure_test_tables():
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


def resolve_font_path(font_path):
    """Resolve a font file path to an absolute path.

    Converts relative font paths to absolute paths by joining them with
    BASE_DIR. Absolute paths are returned unchanged.

    Args:
        font_path: The font file path, either absolute or relative to BASE_DIR.

    Returns:
        str: The absolute path to the font file.

    Example:
        Resolving different path types::

            # Relative path gets BASE_DIR prepended
            resolve_font_path("fonts/Arial.ttf")
            # Returns: "/home/server/glossy/font_scraper/fonts/Arial.ttf"

            # Absolute path is returned unchanged
            resolve_font_path("/usr/share/fonts/Arial.ttf")
            # Returns: "/usr/share/fonts/Arial.ttf"
    """
    if os.path.isabs(font_path):
        return font_path
    return os.path.join(BASE_DIR, font_path)


def get_font(fid):
    """Retrieve a font record from the database by its ID.

    Args:
        fid: The integer ID of the font to retrieve.

    Returns:
        sqlite3.Row: A row object with font data (id, name, source, file_path,
            etc.) if found, or None if no font exists with the given ID.

    Example:
        Fetching a font::

            font = get_font(42)
            if font:
                print(f"Font name: {font['name']}")
                print(f"File path: {font['file_path']}")
    """
    db = get_db()
    f = db.execute("SELECT * FROM fonts WHERE id = ?", (fid,)).fetchone()
    db.close()
    return f


def validate_char_param(char):
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
