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


class TestFontRepositoryExtended(unittest.TestCase):
    """Extended tests for FontRepository covering more methods."""

    def setUp(self):
        """Set up test fixtures with in-memory database."""
        self.conn = create_in_memory_db()

        def connection_factory():
            return in_memory_db_context(self.conn)

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
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def test_list_fonts_non_rejected(self):
        """Test list_fonts returns non-rejected fonts."""
        fonts = self.repo.list_fonts(show_rejected=False)
        self.assertEqual(len(fonts), 2)
        font_names = {f['name'] for f in fonts}
        self.assertEqual(font_names, {'TestFont', 'AnotherFont'})

    def test_list_fonts_shows_rejected(self):
        """Test list_fonts with show_rejected returns only rejected fonts."""
        # Reject font 1
        self.conn.execute(
            "INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, ?, ?)",
            (1, self.repo.REJECTION_REASON_ID, 'Test rejection')
        )
        self.conn.commit()

        fonts = self.repo.list_fonts(show_rejected=True)
        self.assertEqual(len(fonts), 1)
        self.assertEqual(fonts[0]['name'], 'TestFont')

    def test_list_fonts_excludes_duplicates(self):
        """Test list_fonts excludes fonts marked as duplicates."""
        # Mark font 2 as duplicate
        self.conn.execute(
            "INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, ?, ?)",
            (2, self.repo.DUPLICATE_REASON_ID, 'Duplicate of font 1')
        )
        self.conn.commit()

        fonts = self.repo.list_fonts(show_rejected=False)
        self.assertEqual(len(fonts), 1)
        self.assertEqual(fonts[0]['name'], 'TestFont')

    def test_get_font_characters(self):
        """Test get_font_characters returns characters with strokes."""
        # Add characters with strokes
        self.conn.execute(
            "INSERT INTO characters (font_id, char, strokes_raw, point_count) VALUES (?, ?, ?, ?)",
            (1, 'A', '[[[0,0],[100,100]]]', 2)
        )
        self.conn.execute(
            "INSERT INTO characters (font_id, char, strokes_raw, point_count) VALUES (?, ?, ?, ?)",
            (1, 'B', '[[[0,0],[50,50]]]', 2)
        )
        self.conn.commit()

        chars = self.repo.get_font_characters(1)
        self.assertEqual(len(chars), 2)
        char_names = {c['char'] for c in chars}
        self.assertEqual(char_names, {'A', 'B'})

    def test_get_character(self):
        """Test get_character returns character with strokes and markers."""
        self.conn.execute(
            "INSERT INTO characters (font_id, char, strokes_raw, markers) VALUES (?, ?, ?, ?)",
            (1, 'A', '[[[0,0]]]', '{"test": true}')
        )
        self.conn.commit()

        char = self.repo.get_character(1, 'A')
        self.assertIsNotNone(char)
        self.assertEqual(char['strokes_raw'], '[[[0,0]]]')
        self.assertEqual(char['markers'], '{"test": true}')

    def test_get_character_returns_none_for_missing(self):
        """Test get_character returns None for missing character."""
        char = self.repo.get_character(1, 'Z')
        self.assertIsNone(char)

    def test_save_character(self):
        """Test save_character saves new character."""
        self.repo.save_character(1, 'C', '[[[0,0]]]', 1, '{}')

        cursor = self.conn.execute(
            "SELECT strokes_raw, point_count, markers FROM characters WHERE font_id = ? AND char = ?",
            (1, 'C')
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row['strokes_raw'], '[[[0,0]]]')
        self.assertEqual(row['point_count'], 1)

    def test_save_character_updates_existing(self):
        """Test save_character updates existing character."""
        # Insert initial
        self.conn.execute(
            "INSERT INTO characters (font_id, char, strokes_raw, point_count, markers) VALUES (?, ?, ?, ?, ?)",
            (1, 'D', '[[[0,0]]]', 1, '{}')
        )
        self.conn.commit()

        # Update
        self.repo.save_character(1, 'D', '[[[1,1],[2,2]]]', 2, '{"updated": true}')

        cursor = self.conn.execute(
            "SELECT strokes_raw, point_count FROM characters WHERE font_id = ? AND char = ?",
            (1, 'D')
        )
        row = cursor.fetchone()
        self.assertEqual(row['point_count'], 2)

    def test_get_character_strokes(self):
        """Test get_character_strokes returns strokes_raw only."""
        self.conn.execute(
            "INSERT INTO characters (font_id, char, strokes_raw) VALUES (?, ?, ?)",
            (1, 'E', '[[[5,5]]]')
        )
        self.conn.commit()

        char = self.repo.get_character_strokes(1, 'E')
        self.assertIsNotNone(char)
        self.assertEqual(char['strokes_raw'], '[[[5,5]]]')

    def test_list_fonts_for_scan(self):
        """Test list_fonts_for_scan returns non-rejected fonts."""
        # Reject font 1
        self.conn.execute(
            "INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, ?, ?)",
            (1, self.repo.REJECTION_REASON_ID, 'Rejected')
        )
        self.conn.commit()

        fonts = self.repo.list_fonts_for_scan()
        self.assertEqual(len(fonts), 1)
        self.assertEqual(fonts[0]['file_path'], '/fonts/another.ttf')

    def test_reject_font(self):
        """Test reject_font marks font as rejected."""
        result = self.repo.reject_font(1, 'Test rejection')
        self.assertTrue(result)

        cursor = self.conn.execute(
            "SELECT details FROM font_removals WHERE font_id = ? AND reason_id = ?",
            (1, self.repo.REJECTION_REASON_ID)
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row['details'], 'Test rejection')

    def test_reject_font_returns_false_for_missing(self):
        """Test reject_font returns False for non-existent font."""
        result = self.repo.reject_font(999, 'Should fail')
        self.assertFalse(result)

    def test_reject_font_returns_false_if_already_rejected(self):
        """Test reject_font returns False if already rejected."""
        # Pre-reject
        self.conn.execute(
            "INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, ?, ?)",
            (1, self.repo.REJECTION_REASON_ID, 'Already')
        )
        self.conn.commit()

        result = self.repo.reject_font(1, 'Again')
        self.assertFalse(result)

    def test_reject_fonts_batch_empty(self):
        """Test reject_fonts_batch with empty list returns 0."""
        result = self.repo.reject_fonts_batch([])
        self.assertEqual(result, 0)

    def test_unreject_font(self):
        """Test unreject_font removes rejection."""
        # Reject first
        self.conn.execute(
            "INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, ?, ?)",
            (1, self.repo.REJECTION_REASON_ID, 'Rejected')
        )
        self.conn.commit()

        result = self.repo.unreject_font(1)
        self.assertEqual(result, 1)

        cursor = self.conn.execute(
            "SELECT id FROM font_removals WHERE font_id = ? AND reason_id = ?",
            (1, self.repo.REJECTION_REASON_ID)
        )
        self.assertIsNone(cursor.fetchone())

    def test_unreject_font_returns_zero_if_not_rejected(self):
        """Test unreject_font returns 0 if font not rejected."""
        result = self.repo.unreject_font(1)
        self.assertEqual(result, 0)

    def test_unreject_all_fonts(self):
        """Test unreject_all_fonts removes all rejections."""
        # Reject both fonts
        self.conn.execute(
            "INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, ?, ?)",
            (1, self.repo.REJECTION_REASON_ID, 'Rejected 1')
        )
        self.conn.execute(
            "INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, ?, ?)",
            (2, self.repo.REJECTION_REASON_ID, 'Rejected 2')
        )
        self.conn.commit()

        result = self.repo.unreject_all_fonts()
        self.assertEqual(result, 2)

    def test_clear_shape_cache_for_char(self):
        """Test clear_shape_cache clears cache for specific character."""
        # Insert character with cache
        self.conn.execute(
            "INSERT INTO characters (font_id, char, shape_params_cache) VALUES (?, ?, ?)",
            (1, 'F', '{"cached": true}')
        )
        self.conn.commit()

        result = self.repo.clear_shape_cache(1, 'F')
        self.assertEqual(result, 1)

        cursor = self.conn.execute(
            "SELECT shape_params_cache FROM characters WHERE font_id = ? AND char = ?",
            (1, 'F')
        )
        row = cursor.fetchone()
        self.assertIsNone(row['shape_params_cache'])

    def test_clear_shape_cache_for_all_chars(self):
        """Test clear_shape_cache clears cache for all characters."""
        # Insert characters with cache
        self.conn.execute(
            "INSERT INTO characters (font_id, char, shape_params_cache) VALUES (?, ?, ?)",
            (1, 'G', '{"cached": true}')
        )
        self.conn.execute(
            "INSERT INTO characters (font_id, char, shape_params_cache) VALUES (?, ?, ?)",
            (1, 'H', '{"cached": true}')
        )
        self.conn.commit()

        result = self.repo.clear_shape_cache(1)
        self.assertEqual(result, 2)

    def test_has_strokes_true(self):
        """Test has_strokes returns True when strokes exist."""
        self.conn.execute(
            "INSERT INTO characters (font_id, char, strokes_raw) VALUES (?, ?, ?)",
            (1, 'I', '[[[0,0]]]')
        )
        self.conn.commit()

        result = self.repo.has_strokes(1, 'I')
        self.assertTrue(result)

    def test_has_strokes_false(self):
        """Test has_strokes returns False when no strokes."""
        result = self.repo.has_strokes(1, 'Z')
        self.assertFalse(result)

    def test_save_character_strokes_without_variant_or_count(self):
        """Test save_character_strokes with only strokes_raw."""
        self.repo.save_character_strokes(1, 'J', '[[[0,0]]]')

        cursor = self.conn.execute(
            "SELECT strokes_raw FROM characters WHERE font_id = ? AND char = ?",
            (1, 'J')
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row['strokes_raw'], '[[[0,0]]]')

    def test_save_character_strokes_update_without_variant_or_count(self):
        """Test save_character_strokes update with only strokes_raw."""
        # Insert initial
        self.conn.execute(
            "INSERT INTO characters (font_id, char, strokes_raw) VALUES (?, ?, ?)",
            (1, 'K', '[[[0,0]]]')
        )
        self.conn.commit()

        self.repo.save_character_strokes(1, 'K', '[[[1,1]]]')

        cursor = self.conn.execute(
            "SELECT strokes_raw FROM characters WHERE font_id = ? AND char = ?",
            (1, 'K')
        )
        row = cursor.fetchone()
        self.assertEqual(row['strokes_raw'], '[[[1,1]]]')

    def test_ensure_template_variant_column(self):
        """Test ensure_template_variant_column handles existing column."""
        # Column already exists from SCHEMA
        result = self.repo.ensure_template_variant_column()
        self.assertFalse(result)


class TestTestRunRepository(unittest.TestCase):
    """Tests for TestRunRepository."""

    def setUp(self):
        """Set up test fixtures."""
        self.conn = create_in_memory_db()

        def connection_factory():
            return in_memory_db_context(self.conn)

        from stroke_flask import TestRunRepository
        self.repo = TestRunRepository(connection_factory=connection_factory)

        # Insert test font
        self.conn.execute(
            "INSERT INTO fonts (id, name, source, file_path) VALUES (?, ?, ?, ?)",
            (1, 'TestFont', 'test', '/fonts/test.ttf')
        )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def test_save_run(self):
        """Test save_run creates a test run record."""
        run_id = self.repo.save_run(
            font_id=1,
            run_date='2024-01-15 10:00:00',
            chars_tested=26,
            chars_ok=24,
            avg_score=0.85,
            avg_coverage=0.75,
            avg_overshoot=0.01,
            avg_stroke_count=1.5,
            avg_topology=0.90,
            results_json='{"A": {"score": 0.9}}'
        )

        self.assertIsNotNone(run_id)
        self.assertGreater(run_id, 0)

        cursor = self.conn.execute("SELECT * FROM test_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        self.assertEqual(row['chars_tested'], 26)
        self.assertEqual(row['avg_score'], 0.85)

    def test_get_history(self):
        """Test get_history returns test runs for a font."""
        # Insert multiple runs
        self.conn.execute(
            """INSERT INTO test_runs (font_id, run_date, chars_tested, chars_ok,
               avg_score, avg_coverage, avg_overshoot, avg_stroke_count, avg_topology)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (1, '2024-01-15 10:00:00', 26, 24, 0.85, 0.75, 0.01, 1.5, 0.9)
        )
        self.conn.execute(
            """INSERT INTO test_runs (font_id, run_date, chars_tested, chars_ok,
               avg_score, avg_coverage, avg_overshoot, avg_stroke_count, avg_topology)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (1, '2024-01-16 10:00:00', 26, 25, 0.90, 0.80, 0.01, 1.4, 0.95)
        )
        self.conn.commit()

        history = self.repo.get_history(1, limit=10)
        self.assertEqual(len(history), 2)
        # Most recent first
        self.assertEqual(history[0]['avg_score'], 0.90)

    def test_get_run(self):
        """Test get_run returns a specific run."""
        self.conn.execute(
            """INSERT INTO test_runs (id, font_id, run_date, chars_tested, chars_ok,
               avg_score, avg_coverage, avg_overshoot, avg_stroke_count, avg_topology)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (42, 1, '2024-01-15', 26, 24, 0.85, 0.75, 0.01, 1.5, 0.9)
        )
        self.conn.commit()

        run = self.repo.get_run(42)
        self.assertIsNotNone(run)
        self.assertEqual(run['id'], 42)
        self.assertEqual(run['chars_tested'], 26)

    def test_get_run_returns_none_for_missing(self):
        """Test get_run returns None for non-existent run."""
        run = self.repo.get_run(999)
        self.assertIsNone(run)

    def test_get_recent_runs(self):
        """Test get_recent_runs returns most recent run IDs."""
        # Insert runs
        self.conn.execute(
            """INSERT INTO test_runs (id, font_id, run_date, chars_tested, chars_ok,
               avg_score, avg_coverage, avg_overshoot, avg_stroke_count, avg_topology)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (1, 1, '2024-01-15', 26, 24, 0.85, 0.75, 0.01, 1.5, 0.9)
        )
        self.conn.execute(
            """INSERT INTO test_runs (id, font_id, run_date, chars_tested, chars_ok,
               avg_score, avg_coverage, avg_overshoot, avg_stroke_count, avg_topology)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (2, 1, '2024-01-16', 26, 25, 0.90, 0.80, 0.01, 1.4, 0.95)
        )
        self.conn.execute(
            """INSERT INTO test_runs (id, font_id, run_date, chars_tested, chars_ok,
               avg_score, avg_coverage, avg_overshoot, avg_stroke_count, avg_topology)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (3, 1, '2024-01-17', 26, 26, 0.95, 0.85, 0.00, 1.3, 0.98)
        )
        self.conn.commit()

        recent = self.repo.get_recent_runs(1, count=2)
        self.assertEqual(len(recent), 2)
        # Most recent IDs
        self.assertEqual(recent, [3, 2])


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
