"""Flask application setup and database utilities.

This module contains the Flask app instance, database connection helpers,
and common Flask-related utilities used across the stroke editor.
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
    """URL encode filter for Jinja templates."""
    return urlquote(str(s), safe='')


def get_db():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db_context():
    """Context manager for database connections.

    Automatically commits on success and closes on exit.
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
    """Create test tables if they don't exist."""
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
    """Resolve font path to absolute path."""
    if os.path.isabs(font_path):
        return font_path
    return os.path.join(BASE_DIR, font_path)


def validate_char_param(char):
    """Validate character parameter.

    Returns (is_valid, error_response) tuple.
    """
    from flask import jsonify

    if not char:
        return False, (jsonify(error="Missing ?c= parameter"), 400)
    if len(char) != 1:
        return False, (jsonify(error="Character must be a single character"), 400)
    return True, None
