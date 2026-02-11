#!/usr/bin/env python3
"""Database setup script for the Stroke Editor.

Creates an empty fonts.db database with all required tables and seeds
the removal_reasons table with default values. Use this script to
initialize a fresh database for development or testing.

Example:
    Initialize a new database::

        $ python3 setup_database.py

    Force recreate (drops existing tables)::

        $ python3 setup_database.py --force

    Initialize with a sample font::

        $ python3 setup_database.py --sample /path/to/font.ttf
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# Database path
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'fonts.db'

# Table schemas
SCHEMA = """
-- Core tables
CREATE TABLE IF NOT EXISTS fonts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT,
    url TEXT,
    category TEXT,
    license TEXT,
    variant TEXT,
    file_path TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS removal_reasons (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS font_removals (
    id INTEGER PRIMARY KEY,
    font_id INTEGER REFERENCES fonts(id),
    reason_id INTEGER REFERENCES removal_reasons(id),
    details TEXT,
    removed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS font_checks (
    id INTEGER PRIMARY KEY,
    font_id INTEGER UNIQUE REFERENCES fonts(id),
    completeness_score REAL,
    missing_glyphs TEXT,
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_group_id INTEGER,
    keep_in_group BOOLEAN,
    connectivity_score REAL,
    contextual_score REAL,
    is_cursive BOOLEAN,
    prefilter_image_path TEXT,
    prefilter_ocr_text TEXT,
    prefilter_confidence REAL,
    prefilter_passed BOOLEAN,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY,
    font_id INTEGER REFERENCES fonts(id),
    char TEXT NOT NULL,
    image_path TEXT,
    strokes_raw TEXT,
    strokes_processed TEXT,
    point_count INTEGER,
    best_ocr_result TEXT,
    best_ocr_confidence REAL,
    best_ocr_match BOOLEAN,
    quality_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    markers TEXT,
    shape_params_cache TEXT,
    template_variant TEXT,
    UNIQUE(font_id, char)
);

CREATE TABLE IF NOT EXISTS ocr_runs (
    id INTEGER PRIMARY KEY,
    character_id INTEGER REFERENCES characters(id),
    stage TEXT NOT NULL,
    image_path TEXT,
    ocr_result TEXT,
    ocr_confidence REAL,
    ocr_match BOOLEAN,
    model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Test run tables
CREATE TABLE IF NOT EXISTS test_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    font_id INTEGER NOT NULL,
    run_date TEXT NOT NULL,
    chars_tested INTEGER NOT NULL,
    chars_ok INTEGER NOT NULL,
    avg_score REAL,
    avg_coverage REAL,
    avg_overshoot REAL,
    avg_stroke_count REAL,
    avg_topology REAL,
    results_json TEXT,
    FOREIGN KEY (font_id) REFERENCES fonts(id)
);

CREATE TABLE IF NOT EXISTS test_run_images (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    char TEXT NOT NULL,
    image_data BLOB
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_fonts_source ON fonts(source);
CREATE INDEX IF NOT EXISTS idx_fonts_name ON fonts(name);
CREATE INDEX IF NOT EXISTS idx_characters_font_id ON characters(font_id);
CREATE INDEX IF NOT EXISTS idx_characters_font_char ON characters(font_id, char);
CREATE INDEX IF NOT EXISTS idx_font_removals_font_id ON font_removals(font_id);
CREATE INDEX IF NOT EXISTS idx_font_removals_reason_id ON font_removals(reason_id);
CREATE INDEX IF NOT EXISTS idx_test_runs_font_id ON test_runs(font_id);
"""

# Known tables for validation (prevents SQL injection in print_summary)
KNOWN_TABLES = frozenset([
    'fonts',
    'removal_reasons',
    'font_removals',
    'font_checks',
    'characters',
    'ocr_runs',
    'test_runs',
    'test_run_images',
])

# Default removal reasons
REMOVAL_REASONS = [
    (1, 'incomplete', 'Missing required glyphs'),
    (2, 'duplicate', 'Duplicate of another font'),
    (3, 'cursive', 'Cursive/connected letterforms'),
    (4, 'contextual', 'Has contextual alternates'),
    (5, 'ocr_prefilter', 'Failed OCR prefilter'),
    (6, 'ocr_validation', 'Failed OCR validation after processing'),
    (7, 'low_quality', 'Quality score below threshold'),
    (8, 'manual', 'Manually rejected during review'),
    (9, 'load_error', 'Could not load font file'),
]


def create_database(force: bool = False) -> sqlite3.Connection:
    """Create the database and tables.

    Args:
        force: If True, drop existing tables before creating.

    Returns:
        Database connection.
    """
    if force and DB_PATH.exists():
        print(f"Removing existing database: {DB_PATH}")
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print(f"Creating database: {DB_PATH}")
    conn.executescript(SCHEMA)

    return conn


def seed_removal_reasons(conn: sqlite3.Connection) -> int:
    """Seed the removal_reasons table with default values.

    Args:
        conn: Database connection.

    Returns:
        Number of reasons inserted.
    """
    cursor = conn.cursor()
    inserted = 0

    for id_, code, description in REMOVAL_REASONS:
        try:
            cursor.execute(
                "INSERT INTO removal_reasons (id, code, description) VALUES (?, ?, ?)",
                (id_, code, description)
            )
            inserted += 1
        except sqlite3.IntegrityError:
            # Already exists
            pass

    conn.commit()
    return inserted


def add_sample_font(conn: sqlite3.Connection, font_path: str) -> int | None:
    """Add a sample font to the database.

    Args:
        conn: Database connection.
        font_path: Path to a TTF/OTF font file.

    Returns:
        Font ID if added, None if failed.
    """
    path = Path(font_path)
    if not path.exists():
        print(f"Error: Font file not found: {font_path}")
        return None

    name = path.stem
    # Make path relative to BASE_DIR if possible
    try:
        rel_path = path.relative_to(BASE_DIR)
        file_path = str(rel_path)
    except ValueError:
        file_path = str(path.absolute())

    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO fonts (name, source, file_path) VALUES (?, ?, ?)",
            (name, 'sample', file_path)
        )
        conn.commit()
        font_id = cursor.lastrowid
        print(f"Added sample font: {name} (id={font_id})")
        return font_id
    except sqlite3.IntegrityError:
        print(f"Font already exists: {file_path}")
        return None


def print_summary(conn: sqlite3.Connection) -> None:
    """Print database summary statistics."""
    cursor = conn.cursor()

    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()

    print("\nDatabase summary:")
    print("-" * 40)

    for (table_name,) in tables:
        # Validate table name against allowlist to prevent SQL injection
        if table_name not in KNOWN_TABLES:
            print(f"  {table_name}: (skipped - unknown table)")
            continue
        count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  {table_name}: {count} rows")

    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="Initialize the fonts.db database for Stroke Editor"
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force recreate database (drops existing tables)'
    )
    parser.add_argument(
        '--sample', '-s',
        metavar='FONT_PATH',
        help='Add a sample font file to the database'
    )
    args = parser.parse_args()

    # Create database and tables
    conn = create_database(force=args.force)
    print("Created tables successfully")

    # Seed removal reasons
    n_reasons = seed_removal_reasons(conn)
    print(f"Seeded {n_reasons} removal reasons")

    # Optionally add sample font
    if args.sample:
        add_sample_font(conn, args.sample)

    # Print summary
    print_summary(conn)

    conn.close()
    print("\nDatabase setup complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
