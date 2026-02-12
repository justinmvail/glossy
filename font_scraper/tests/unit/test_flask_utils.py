"""Unit tests for stroke_flask utility functions.

Tests helper functions, response formatters, and Flask utilities
from stroke_flask.py.
"""

import io
import json
import logging
import os
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestConfigureLogging(unittest.TestCase):
    """Tests for configure_logging function."""

    def setUp(self):
        """Save original logging state."""
        self.root_logger = logging.getLogger()
        self.original_handlers = self.root_logger.handlers.copy()
        self.original_level = self.root_logger.level

    def tearDown(self):
        """Restore original logging state."""
        self.root_logger.handlers = self.original_handlers
        self.root_logger.setLevel(self.original_level)

    def test_configure_logging_sets_level(self):
        """Test that configure_logging sets the log level."""
        from stroke_flask import configure_logging

        configure_logging(level='DEBUG')

        self.assertEqual(self.root_logger.level, logging.DEBUG)

    def test_configure_logging_adds_console_handler(self):
        """Test that configure_logging adds a console handler."""
        from stroke_flask import configure_logging

        configure_logging(level='INFO')

        # Should have at least one StreamHandler
        stream_handlers = [
            h for h in self.root_logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        self.assertGreater(len(stream_handlers), 0)

    def test_configure_logging_with_file(self):
        """Test that configure_logging can add a file handler."""
        from stroke_flask import configure_logging

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name

        try:
            configure_logging(level='INFO', log_file=log_file)

            # Should have a FileHandler
            file_handlers = [
                h for h in self.root_logger.handlers
                if isinstance(h, logging.FileHandler)
            ]
            self.assertGreater(len(file_handlers), 0)
        finally:
            os.unlink(log_file)

    def test_configure_logging_clears_existing_handlers(self):
        """Test that configure_logging clears existing handlers."""
        from stroke_flask import configure_logging

        # Add some handlers first
        self.root_logger.addHandler(logging.StreamHandler())
        self.root_logger.addHandler(logging.StreamHandler())
        initial_count = len(self.root_logger.handlers)

        configure_logging(level='INFO')

        # Should have fewer or equal handlers (cleared and re-added)
        self.assertLessEqual(len(self.root_logger.handlers), initial_count)


class TestUrlencodeFilter(unittest.TestCase):
    """Tests for urlencode_filter template filter."""

    def test_encodes_spaces(self):
        """Test that spaces are encoded."""
        from stroke_flask import urlencode_filter

        result = urlencode_filter('hello world')
        self.assertEqual(result, 'hello%20world')

    def test_encodes_special_chars(self):
        """Test that special characters are encoded."""
        from stroke_flask import urlencode_filter

        result = urlencode_filter('a/b?c=d&e=f')
        self.assertIn('%2F', result)  # /
        self.assertIn('%3F', result)  # ?
        self.assertIn('%3D', result)  # =
        self.assertIn('%26', result)  # &

    def test_handles_non_string(self):
        """Test that non-strings are converted."""
        from stroke_flask import urlencode_filter

        result = urlencode_filter(123)
        self.assertEqual(result, '123')


class TestFormatSseEvent(unittest.TestCase):
    """Tests for format_sse_event function."""

    def test_formats_dict_as_sse(self):
        """Test that dict is formatted as SSE event."""
        from stroke_flask import format_sse_event

        result = format_sse_event({'phase': 'Init', 'score': 0.5})

        self.assertTrue(result.startswith('data: '))
        self.assertTrue(result.endswith('\n\n'))
        # Parse the JSON part
        json_part = result[6:-2]  # Strip 'data: ' and '\n\n'
        data = json.loads(json_part)
        self.assertEqual(data['phase'], 'Init')
        self.assertEqual(data['score'], 0.5)

    def test_formats_error_event(self):
        """Test that error events are formatted correctly."""
        from stroke_flask import format_sse_event

        result = format_sse_event({'error': 'Connection lost'})

        json_part = result[6:-2]
        data = json.loads(json_part)
        self.assertEqual(data['error'], 'Connection lost')

    def test_formats_done_event(self):
        """Test that done events are formatted correctly."""
        from stroke_flask import format_sse_event

        result = format_sse_event({'done': True, 'strokes': [[1, 2]]})

        json_part = result[6:-2]
        data = json.loads(json_part)
        self.assertTrue(data['done'])
        self.assertEqual(data['strokes'], [[1, 2]])


class TestResponseHelpers(unittest.TestCase):
    """Tests for response helper functions requiring Flask context."""

    def setUp(self):
        """Create Flask app for testing."""
        from stroke_flask import app
        self.app = app
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        """Clean up Flask context."""
        self.ctx.pop()

    def test_success_response_basic(self):
        """Test success_response returns ok=true."""
        from stroke_flask import success_response

        response = success_response()
        data = response.get_json()

        self.assertTrue(data['ok'])

    def test_success_response_with_extra_data(self):
        """Test success_response includes extra data."""
        from stroke_flask import success_response

        response = success_response(status='rejected', count=5)
        data = response.get_json()

        self.assertTrue(data['ok'])
        self.assertEqual(data['status'], 'rejected')
        self.assertEqual(data['count'], 5)

    def test_data_response(self):
        """Test data_response returns data without ok field."""
        from stroke_flask import data_response

        response = data_response(strokes=[[1, 2]], variant='default')
        data = response.get_json()

        self.assertNotIn('ok', data)
        self.assertEqual(data['strokes'], [[1, 2]])
        self.assertEqual(data['variant'], 'default')

    def test_error_response_json(self):
        """Test error_response with JSON format."""
        from stroke_flask import error_response

        response, status_code = error_response('Test error', 400, 'json')

        self.assertEqual(status_code, 400)
        data = response.get_json()
        self.assertEqual(data['error'], 'Test error')

    def test_error_response_text(self):
        """Test error_response with text format."""
        from stroke_flask import error_response

        response, status_code = error_response('Not found', 404, 'text')

        self.assertEqual(status_code, 404)
        self.assertEqual(response, 'Not found')

    def test_error_response_sse(self):
        """Test error_response with SSE format."""
        from stroke_flask import error_response

        response = error_response('Stream error', 500, 'sse')

        self.assertEqual(response.mimetype, 'text/event-stream')
        data = response.get_data(as_text=True)
        self.assertIn('data:', data)
        self.assertIn('Stream error', data)


class TestValidateCharParam(unittest.TestCase):
    """Tests for validate_char_param function."""

    def setUp(self):
        """Create Flask app for testing."""
        from stroke_flask import app
        self.app = app
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()

    def test_valid_single_char(self):
        """Test validation passes for single character."""
        from stroke_flask import validate_char_param

        ok, err = validate_char_param('A')

        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_missing_char(self):
        """Test validation fails for None."""
        from stroke_flask import validate_char_param

        ok, err = validate_char_param(None)

        self.assertFalse(ok)
        self.assertIsNotNone(err)
        response, status = err
        self.assertEqual(status, 400)

    def test_empty_char(self):
        """Test validation fails for empty string."""
        from stroke_flask import validate_char_param

        ok, err = validate_char_param('')

        self.assertFalse(ok)
        self.assertIsNotNone(err)

    def test_multiple_chars(self):
        """Test validation fails for multiple characters."""
        from stroke_flask import validate_char_param

        ok, err = validate_char_param('AB')

        self.assertFalse(ok)
        response, status = err
        self.assertEqual(status, 400)


class TestGetCharParamOrError(unittest.TestCase):
    """Tests for get_char_param_or_error function."""

    def setUp(self):
        """Create Flask test client."""
        from stroke_flask import app
        self.app = app
        self.client = self.app.test_client()

    def test_valid_char_param(self):
        """Test extraction of valid char param."""
        from stroke_flask import get_char_param_or_error

        with self.app.test_request_context('/?c=A'):
            char, err = get_char_param_or_error()

            self.assertEqual(char, 'A')
            self.assertIsNone(err)

    def test_missing_char_param_json(self):
        """Test error for missing param with JSON format."""
        from stroke_flask import get_char_param_or_error

        with self.app.test_request_context('/'):
            char, err = get_char_param_or_error(response_format='json')

            self.assertIsNone(char)
            self.assertIsNotNone(err)

    def test_missing_char_param_sse(self):
        """Test error for missing param with SSE format."""
        from stroke_flask import get_char_param_or_error

        with self.app.test_request_context('/'):
            char, err = get_char_param_or_error(response_format='sse')

            self.assertIsNone(char)
            self.assertIsNotNone(err)
            self.assertEqual(err.mimetype, 'text/event-stream')

    def test_multi_char_param(self):
        """Test error for multi-character param."""
        from stroke_flask import get_char_param_or_error

        with self.app.test_request_context('/?c=ABC'):
            char, err = get_char_param_or_error()

            self.assertIsNone(char)
            self.assertIsNotNone(err)


class TestGetFontOrError(unittest.TestCase):
    """Tests for get_font_or_error function."""

    def setUp(self):
        """Create Flask app for testing."""
        from stroke_flask import app
        self.app = app
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()

    @patch('stroke_flask.get_font')
    def test_returns_font_when_found(self, mock_get_font):
        """Test that font is returned when found."""
        from stroke_flask import get_font_or_error

        mock_font = {'id': 1, 'name': 'TestFont'}
        mock_get_font.return_value = mock_font

        font, err = get_font_or_error(1)

        self.assertEqual(font, mock_font)
        self.assertIsNone(err)

    @patch('stroke_flask.get_font')
    def test_returns_error_when_not_found_json(self, mock_get_font):
        """Test that JSON error is returned when font not found."""
        from stroke_flask import get_font_or_error

        mock_get_font.return_value = None

        font, err = get_font_or_error(999, response_format='json')

        self.assertIsNone(font)
        self.assertIsNotNone(err)
        response, status = err
        self.assertEqual(status, 404)

    @patch('stroke_flask.get_font')
    def test_returns_error_when_not_found_sse(self, mock_get_font):
        """Test that SSE error is returned when font not found."""
        from stroke_flask import get_font_or_error

        mock_get_font.return_value = None

        font, err = get_font_or_error(999, response_format='sse')

        self.assertIsNone(font)
        self.assertIsNotNone(err)
        self.assertEqual(err.mimetype, 'text/event-stream')


class TestGetFontAndMask(unittest.TestCase):
    """Tests for get_font_and_mask function."""

    def setUp(self):
        """Create fresh Flask app for testing."""
        from flask import Flask
        self.test_app = Flask(__name__)
        self.ctx = self.test_app.test_request_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()

    def test_returns_font_and_mask(self):
        """Test successful font and mask retrieval."""
        mock_font = {'id': 1, 'name': 'TestFont', 'file_path': '/fonts/test.ttf'}
        mock_mask = np.ones((64, 64), dtype=np.uint8)

        with patch('stroke_flask.get_font', return_value=mock_font):
            with patch('stroke_rendering.render_glyph_mask', return_value=mock_mask):
                from stroke_flask import get_font_and_mask
                font, mask, err = get_font_and_mask(1, 'A')

        self.assertEqual(font, mock_font)
        self.assertIsNotNone(mask)
        self.assertIsNone(err)

    def test_returns_error_when_font_not_found(self):
        """Test error when font not found."""
        with patch('stroke_flask.get_font', return_value=None):
            from stroke_flask import get_font_and_mask
            font, mask, err = get_font_and_mask(999, 'A')

        self.assertIsNone(font)
        self.assertIsNone(mask)
        self.assertIsNotNone(err)
        response, status = err
        self.assertEqual(status, 404)

    def test_returns_error_when_mask_fails(self):
        """Test error when mask rendering fails."""
        mock_font = {'id': 1, 'name': 'TestFont', 'file_path': '/fonts/test.ttf'}

        with patch('stroke_flask.get_font', return_value=mock_font):
            with patch('stroke_rendering.render_glyph_mask', return_value=None):
                from stroke_flask import get_font_and_mask
                font, mask, err = get_font_and_mask(1, 'A')

        self.assertIsNone(font)
        self.assertIsNone(mask)
        self.assertIsNotNone(err)
        response, status = err
        self.assertEqual(status, 500)


class TestSendPilImageAsPng(unittest.TestCase):
    """Tests for send_pil_image_as_png function."""

    def test_sends_png_image(self):
        """Test that PIL image is converted to PNG response."""
        # Test the underlying logic without Flask routing
        img = Image.new('RGB', (100, 100), color='red')

        # Save to buffer like the function does
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        data = buf.read()

        # Verify it's valid PNG data
        self.assertTrue(data.startswith(b'\x89PNG'))
        self.assertGreater(len(data), 100)  # Non-trivial size

    def test_sends_grayscale_image(self):
        """Test that grayscale image is converted to PNG."""
        img = Image.new('L', (50, 50), color=128)

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        data = buf.read()

        self.assertTrue(data.startswith(b'\x89PNG'))

    def test_function_returns_response(self):
        """Test that send_pil_image_as_png returns proper response object."""
        from flask import Flask
        from stroke_flask import send_pil_image_as_png

        # Create a fresh Flask app for testing
        test_app = Flask(__name__)

        with test_app.test_request_context():
            img = Image.new('RGB', (10, 10), color='blue')
            response = send_pil_image_as_png(img)

            self.assertEqual(response.mimetype, 'image/png')


class TestGetDb(unittest.TestCase):
    """Tests for get_db function."""

    @patch('stroke_flask.DB_PATH', ':memory:')
    def test_returns_connection(self):
        """Test that get_db returns a database connection."""
        from stroke_flask import get_db

        conn = get_db()

        try:
            self.assertIsInstance(conn, sqlite3.Connection)
            # Should be able to execute queries
            conn.execute("SELECT 1")
        finally:
            conn.close()

    @patch('stroke_flask.DB_PATH', ':memory:')
    def test_row_factory_set(self):
        """Test that row_factory is set to sqlite3.Row."""
        from stroke_flask import get_db

        conn = get_db()

        try:
            self.assertEqual(conn.row_factory, sqlite3.Row)
        finally:
            conn.close()


class TestGetDbContext(unittest.TestCase):
    """Tests for get_db_context context manager."""

    def test_yields_connection(self):
        """Test that context manager yields connection."""
        # Use a temp file for persistent database
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            with patch('stroke_flask.DB_PATH', db_path):
                from stroke_flask import get_db_context

                with get_db_context() as db:
                    self.assertIsInstance(db, sqlite3.Connection)
                    db.execute("CREATE TABLE test (id INTEGER)")
                    db.execute("INSERT INTO test VALUES (1)")
        finally:
            os.unlink(db_path)

    def test_commits_on_success(self):
        """Test that changes are committed on successful exit."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            with patch('stroke_flask.DB_PATH', db_path):
                from stroke_flask import get_db_context

                with get_db_context() as db:
                    db.execute("CREATE TABLE test (id INTEGER)")
                    db.execute("INSERT INTO test VALUES (1)")

                # Data should persist after commit
                with get_db_context() as db:
                    cursor = db.execute("SELECT * FROM test")
                    row = cursor.fetchone()
                    self.assertIsNotNone(row)
        finally:
            os.unlink(db_path)

    def test_rollback_on_exception(self):
        """Test that changes are rolled back on exception."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            with patch('stroke_flask.DB_PATH', db_path):
                from stroke_flask import get_db_context

                # First create the table
                with get_db_context() as db:
                    db.execute("CREATE TABLE test (id INTEGER)")

                # Try to insert and raise exception
                try:
                    with get_db_context() as db:
                        db.execute("INSERT INTO test VALUES (1)")
                        raise ValueError("Test exception")
                except ValueError:
                    pass

                # Data should not persist after rollback
                with get_db_context() as db:
                    cursor = db.execute("SELECT * FROM test")
                    row = cursor.fetchone()
                    self.assertIsNone(row)
        finally:
            os.unlink(db_path)


class TestGetFont(unittest.TestCase):
    """Tests for get_font function."""

    def setUp(self):
        """Create in-memory database with test data."""
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
            CREATE TABLE fonts (
                id INTEGER PRIMARY KEY,
                name TEXT,
                source TEXT,
                file_path TEXT
            )
        """)
        self.conn.execute(
            "INSERT INTO fonts VALUES (?, ?, ?, ?)",
            (1, 'TestFont', 'test', '/fonts/test.ttf')
        )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    @patch('stroke_flask.get_db_context')
    def test_returns_font(self, mock_context):
        """Test that get_font returns font when found."""
        from stroke_flask import get_font

        @contextmanager
        def mock_db_context():
            yield self.conn

        mock_context.side_effect = mock_db_context

        font = get_font(1)

        self.assertIsNotNone(font)
        self.assertEqual(font['name'], 'TestFont')

    @patch('stroke_flask.get_db_context')
    def test_returns_none_for_missing(self, mock_context):
        """Test that get_font returns None when not found."""
        from stroke_flask import get_font

        @contextmanager
        def mock_db_context():
            yield self.conn

        mock_context.side_effect = mock_db_context

        font = get_font(999)

        self.assertIsNone(font)

    def test_returns_none_on_exception(self):
        """Test that get_font returns None on database error."""
        from stroke_flask import get_font

        # Create a mock connection that raises on execute
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.Error("Database error")

        @contextmanager
        def mock_db_context():
            yield mock_conn

        with patch('stroke_flask.get_db_context', mock_db_context):
            font = get_font(1)

        self.assertIsNone(font)


class TestEnsureTestTables(unittest.TestCase):
    """Tests for ensure_test_tables function."""

    @patch('stroke_flask.get_db_context')
    def test_creates_tables(self, mock_context):
        """Test that ensure_test_tables creates required tables."""
        from stroke_flask import ensure_test_tables

        # Create in-memory database
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row

        @contextmanager
        def mock_db_context():
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        mock_context.side_effect = mock_db_context

        ensure_test_tables()

        # Check tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row['name'] for row in cursor.fetchall()}

        self.assertIn('test_runs', tables)
        self.assertIn('test_run_images', tables)

        conn.close()


if __name__ == '__main__':
    unittest.main()
