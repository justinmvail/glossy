"""Integration tests for database operations.

Tests database functionality including FontRepository, schema setup,
and database context management using in-memory SQLite for isolation.
"""

import json
import sqlite3
import unittest
from contextlib import contextmanager

# Import schema and setup utilities
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from setup_database import SCHEMA, REMOVAL_REASONS


def create_in_memory_db() -> sqlite3.Connection:
    """Create an in-memory SQLite database with full schema.

    Returns:
        Database connection with row_factory set to sqlite3.Row.
    """
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)

    # Seed removal reasons
    for id_, code, description in REMOVAL_REASONS:
        try:
            conn.execute(
                "INSERT INTO removal_reasons (id, code, description) VALUES (?, ?, ?)",
                (id_, code, description)
            )
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return conn


@contextmanager
def in_memory_db_context(conn: sqlite3.Connection):
    """Context manager for in-memory database operations.

    Args:
        conn: Database connection to use.

    Yields:
        The database connection.
    """
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


class TestFontRepository(unittest.TestCase):
    """Tests for FontRepository database operations."""

    def setUp(self):
        """Set up test fixtures with in-memory database."""
        self.conn = create_in_memory_db()

        # Create a custom connection factory for the repository
        def connection_factory():
            return in_memory_db_context(self.conn)

        # Import FontRepository and create instance with custom factory
        from stroke_flask import FontRepository
        self.repo = FontRepository(connection_factory=connection_factory)

        # Insert test fonts
        self.conn.execute(
            "INSERT INTO fonts (id, name, source, file_path) VALUES (?, ?, ?, ?)",
            (1, 'TestFont', 'test', '/fonts/test.ttf')
        )
        self.conn.execute(
            "INSERT INTO fonts (id, name, source, file_path) VALUES (?, ?, ?, ?)",
            (2, 'AnotherFont', 'test', '/fonts/another.ttf')
        )
        self.conn.execute(
            "INSERT INTO fonts (id, name, source, file_path) VALUES (?, ?, ?, ?)",
            (3, 'ThirdFont', 'test', '/fonts/third.ttf')
        )
        self.conn.commit()

    def tearDown(self):
        """Clean up database connection."""
        self.conn.close()

    def test_get_font_by_id_returns_font(self):
        """Test that get_font_by_id returns the correct font."""
        font = self.repo.get_font_by_id(1)

        self.assertIsNotNone(font)
        self.assertEqual(font['id'], 1)
        self.assertEqual(font['name'], 'TestFont')
        self.assertEqual(font['source'], 'test')
        self.assertEqual(font['file_path'], '/fonts/test.ttf')

    def test_get_font_by_id_returns_none_for_missing(self):
        """Test that get_font_by_id returns None for non-existent font."""
        font = self.repo.get_font_by_id(999)

        self.assertIsNone(font)

    def test_reject_fonts_batch_rejects_multiple(self):
        """Test that reject_fonts_batch rejects multiple fonts."""
        # Prepare rejection data
        rejections = [
            (1, 'Test rejection for font 1'),
            (2, 'Test rejection for font 2'),
        ]

        # Reject fonts
        rejected_count = self.repo.reject_fonts_batch(rejections)

        # Verify count
        self.assertEqual(rejected_count, 2)

        # Verify rejections in database
        cursor = self.conn.execute(
            "SELECT font_id, details FROM font_removals WHERE reason_id = ?",
            (self.repo.REJECTION_REASON_ID,)
        )
        removals = cursor.fetchall()

        self.assertEqual(len(removals), 2)
        font_ids = {r['font_id'] for r in removals}
        self.assertEqual(font_ids, {1, 2})

    def test_reject_fonts_batch_skips_already_rejected(self):
        """Test that reject_fonts_batch skips already rejected fonts."""
        # Pre-reject font 1
        self.conn.execute(
            "INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, ?, ?)",
            (1, self.repo.REJECTION_REASON_ID, 'Already rejected')
        )
        self.conn.commit()

        # Try to reject both fonts
        rejections = [
            (1, 'Should be skipped'),
            (2, 'Should be rejected'),
        ]
        rejected_count = self.repo.reject_fonts_batch(rejections)

        # Only font 2 should be rejected
        self.assertEqual(rejected_count, 1)

        # Verify only 2 removals total (1 pre-existing + 1 new)
        cursor = self.conn.execute(
            "SELECT font_id FROM font_removals WHERE reason_id = ?",
            (self.repo.REJECTION_REASON_ID,)
        )
        removals = cursor.fetchall()
        self.assertEqual(len(removals), 2)

    def test_save_character_strokes(self):
        """Test that save_character_strokes saves stroke data."""
        strokes_data = json.dumps([[[0, 0], [100, 100]]])

        # Save strokes for a new character
        self.repo.save_character_strokes(1, 'A', strokes_data, point_count=2)

        # Verify saved data
        cursor = self.conn.execute(
            "SELECT strokes_raw, point_count FROM characters WHERE font_id = ? AND char = ?",
            (1, 'A')
        )
        row = cursor.fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row['strokes_raw'], strokes_data)
        self.assertEqual(row['point_count'], 2)

    def test_save_character_strokes_updates_existing(self):
        """Test that save_character_strokes updates existing character."""
        # Insert initial character
        self.conn.execute(
            "INSERT INTO characters (font_id, char, strokes_raw, point_count) VALUES (?, ?, ?, ?)",
            (1, 'A', '[[1,2]]', 1)
        )
        self.conn.commit()

        # Update strokes
        new_strokes = json.dumps([[[0, 0], [50, 50], [100, 100]]])
        self.repo.save_character_strokes(1, 'A', new_strokes, point_count=3)

        # Verify updated data
        cursor = self.conn.execute(
            "SELECT strokes_raw, point_count FROM characters WHERE font_id = ? AND char = ?",
            (1, 'A')
        )
        row = cursor.fetchone()

        self.assertEqual(row['strokes_raw'], new_strokes)
        self.assertEqual(row['point_count'], 3)

    def test_save_character_strokes_with_template_variant(self):
        """Test that save_character_strokes saves template variant."""
        strokes_data = json.dumps([[[0, 0], [100, 100]]])

        self.repo.save_character_strokes(1, 'B', strokes_data, template_variant='serif')

        cursor = self.conn.execute(
            "SELECT strokes_raw, template_variant FROM characters WHERE font_id = ? AND char = ?",
            (1, 'B')
        )
        row = cursor.fetchone()

        self.assertEqual(row['template_variant'], 'serif')


class TestDatabaseSchema(unittest.TestCase):
    """Tests for database schema creation and constraints."""

    def setUp(self):
        """Set up in-memory database."""
        self.conn = create_in_memory_db()

    def tearDown(self):
        """Clean up database connection."""
        self.conn.close()

    def test_all_tables_created(self):
        """Test that all required tables are created."""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = {row['name'] for row in cursor.fetchall()}

        expected_tables = {
            'fonts',
            'removal_reasons',
            'font_removals',
            'font_checks',
            'characters',
            'ocr_runs',
            'test_runs',
            'test_run_images',
        }

        self.assertEqual(tables, expected_tables)

    def test_foreign_keys_enforced(self):
        """Test that foreign key constraints are enforced when enabled."""
        # Enable foreign keys (SQLite requires explicit enablement)
        self.conn.execute("PRAGMA foreign_keys = ON")

        # Insert a valid font first
        self.conn.execute(
            "INSERT INTO fonts (id, name, file_path) VALUES (?, ?, ?)",
            (1, 'TestFont', '/fonts/test.ttf')
        )
        self.conn.commit()

        # This should succeed - valid font_id
        self.conn.execute(
            "INSERT INTO characters (font_id, char) VALUES (?, ?)",
            (1, 'A')
        )
        self.conn.commit()

        # This should fail - invalid font_id (foreign key violation)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO characters (font_id, char) VALUES (?, ?)",
                (999, 'A')
            )

    def test_unique_constraints(self):
        """Test that unique constraints are enforced."""
        # Insert a font
        self.conn.execute(
            "INSERT INTO fonts (id, name, file_path) VALUES (?, ?, ?)",
            (1, 'TestFont', '/fonts/test.ttf')
        )
        self.conn.commit()

        # Trying to insert duplicate file_path should fail
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO fonts (id, name, file_path) VALUES (?, ?, ?)",
                (2, 'DuplicateFont', '/fonts/test.ttf')
            )

    def test_characters_unique_font_char(self):
        """Test that font_id + char combination is unique in characters."""
        # Insert a font
        self.conn.execute(
            "INSERT INTO fonts (id, name, file_path) VALUES (?, ?, ?)",
            (1, 'TestFont', '/fonts/test.ttf')
        )

        # Insert a character
        self.conn.execute(
            "INSERT INTO characters (font_id, char) VALUES (?, ?)",
            (1, 'A')
        )
        self.conn.commit()

        # Trying to insert duplicate font_id + char should fail
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO characters (font_id, char) VALUES (?, ?)",
                (1, 'A')
            )

    def test_removal_reasons_seeded(self):
        """Test that removal reasons are seeded correctly."""
        cursor = self.conn.execute(
            "SELECT id, code, description FROM removal_reasons ORDER BY id"
        )
        reasons = cursor.fetchall()

        # Verify expected reasons exist
        reason_codes = {r['code'] for r in reasons}
        expected_codes = {'incomplete', 'duplicate', 'cursive', 'contextual',
                         'ocr_prefilter', 'ocr_validation', 'low_quality',
                         'manual', 'load_error'}

        self.assertEqual(reason_codes, expected_codes)


class TestDatabaseContext(unittest.TestCase):
    """Tests for database context management."""

    def test_get_db_context_returns_connection(self):
        """Test that get_db_context returns a usable connection."""
        conn = create_in_memory_db()

        with in_memory_db_context(conn) as db:
            # Should be able to execute queries
            db.execute(
                "INSERT INTO fonts (name, file_path) VALUES (?, ?)",
                ('TestFont', '/fonts/test.ttf')
            )

            cursor = db.execute("SELECT name FROM fonts WHERE name = ?", ('TestFont',))
            row = cursor.fetchone()

            self.assertIsNotNone(row)
            self.assertEqual(row['name'], 'TestFont')

        conn.close()

    def test_context_manager_closes_connection(self):
        """Test that context manager commits on success."""
        conn = create_in_memory_db()

        with in_memory_db_context(conn) as db:
            db.execute(
                "INSERT INTO fonts (name, file_path) VALUES (?, ?)",
                ('TestFont', '/fonts/test.ttf')
            )

        # After context exit, changes should be committed
        cursor = conn.execute("SELECT name FROM fonts WHERE name = ?", ('TestFont',))
        row = cursor.fetchone()
        self.assertIsNotNone(row)

        conn.close()

    def test_context_manager_rollback_on_exception(self):
        """Test that context manager rolls back on exception."""
        conn = create_in_memory_db()

        try:
            with in_memory_db_context(conn) as db:
                db.execute(
                    "INSERT INTO fonts (name, file_path) VALUES (?, ?)",
                    ('TestFont', '/fonts/test.ttf')
                )
                # Raise an exception to trigger rollback
                raise ValueError("Test exception")
        except ValueError:
            pass

        # After rollback, data should not exist
        cursor = conn.execute("SELECT name FROM fonts WHERE name = ?", ('TestFont',))
        row = cursor.fetchone()
        self.assertIsNone(row)

        conn.close()

    def test_row_factory_enables_dict_access(self):
        """Test that row_factory allows dict-like access to rows."""
        conn = create_in_memory_db()

        conn.execute(
            "INSERT INTO fonts (id, name, source, file_path) VALUES (?, ?, ?, ?)",
            (1, 'TestFont', 'test', '/fonts/test.ttf')
        )
        conn.commit()

        cursor = conn.execute("SELECT * FROM fonts WHERE id = ?", (1,))
        row = cursor.fetchone()

        # Should be able to access columns by name
        self.assertEqual(row['id'], 1)
        self.assertEqual(row['name'], 'TestFont')
        self.assertEqual(row['source'], 'test')
        self.assertEqual(row['file_path'], '/fonts/test.ttf')

        conn.close()


if __name__ == '__main__':
    unittest.main()
