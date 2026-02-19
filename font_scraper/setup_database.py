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

from db_schema import SCHEMA, init_db

# Database path
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'fonts.db'


def create_database(force: bool = False) -> sqlite3.Connection:
    """Create the database and tables using the canonical schema from db_schema.py.

    Args:
        force: If True, drop existing tables before creating.

    Returns:
        Database connection.
    """
    if force and DB_PATH.exists():
        print(f"Removing existing database: {DB_PATH}")
        os.remove(DB_PATH)

    print(f"Creating database: {DB_PATH}")
    conn = init_db(str(DB_PATH))
    return conn


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
        count = cursor.execute(
            f"SELECT COUNT(*) FROM [{table_name}]"
        ).fetchone()[0]
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

    # Create database and tables (schema from db_schema.py)
    conn = create_database(force=args.force)
    print("Created tables successfully")

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
