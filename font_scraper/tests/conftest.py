"""Shared pytest fixtures for font_scraper test suite.

This module provides common fixtures used across unit, integration, and regression
tests. Fixtures follow the patterns established in TESTING_STRATEGY.md.

Fixtures:
    test_font_path: Path to a valid test font file
    mock_glyph_mask: 64x64 binary numpy mask with a letter shape
    sample_strokes: List of numpy arrays representing sample strokes
    db_session: In-memory SQLite connection with schema setup
    flask_client: Flask test client for the stroke_editor app
    mock_bbox: Standard bounding box tuple (x, y, w, h)

Markers:
    slow: Mark test as slow-running (skip with -m "not slow")
    integration: Mark test as integration test
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Database schema from setup_database.py
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

# Default removal reasons (from setup_database.py)
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


# -----------------------------------------------------------------------------
# Pytest Markers
# -----------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )


# -----------------------------------------------------------------------------
# Font Path Fixture
# -----------------------------------------------------------------------------

@pytest.fixture
def test_font_path():
    """Return path to a test font file.

    Uses the first available .ttf file found in the fonts directory.
    Falls back to a known font path if directory listing fails.

    Returns:
        str: Absolute path to a valid TTF font file.

    Raises:
        pytest.skip: If no font files are available for testing.
    """
    base_dir = Path(__file__).parent.parent
    fonts_dir = base_dir / "fonts" / "dafont"

    # Try to find an available font
    if fonts_dir.exists():
        for font_file in fonts_dir.glob("*.ttf"):
            return str(font_file)

    # Fallback: check common test font locations
    fallback_paths = [
        base_dir / "fonts" / "dafont" / "Hello.ttf",
        base_dir / "fonts" / "dafont" / "Luna.ttf",
        base_dir / "fonts" / "test" / "TestFont.ttf",
    ]
    for path in fallback_paths:
        if path.exists():
            return str(path)

    pytest.skip("No font files available for testing")


# -----------------------------------------------------------------------------
# Mock Glyph Mask Fixture
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_glyph_mask():
    """Return a 64x64 binary numpy mask with a letter-like shape.

    Creates a mask resembling a simplified letter 'T' shape:
    - Horizontal bar at the top
    - Vertical stem in the center

    Returns:
        numpy.ndarray: 64x64 uint8 array where 255 = glyph, 0 = background.
    """
    mask = np.zeros((64, 64), dtype=np.uint8)

    # Horizontal bar at top (like top of 'T')
    mask[8:16, 12:52] = 255

    # Vertical stem in center
    mask[16:56, 26:38] = 255

    return mask


@pytest.fixture
def mock_glyph_mask_L():
    """Return a 64x64 binary numpy mask shaped like letter 'L'.

    Returns:
        numpy.ndarray: 64x64 uint8 array where 255 = glyph, 0 = background.
    """
    mask = np.zeros((64, 64), dtype=np.uint8)

    # Vertical stroke (left side)
    mask[8:56, 12:24] = 255

    # Horizontal stroke (bottom)
    mask[44:56, 12:52] = 255

    return mask


# -----------------------------------------------------------------------------
# Sample Strokes Fixture
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_strokes():
    """Return a list of numpy arrays representing sample stroke paths.

    Creates three sample strokes:
    1. Vertical line (10 points)
    2. Horizontal line (10 points)
    3. Diagonal line (10 points)

    Returns:
        list[numpy.ndarray]: List of Nx2 arrays with (x, y) coordinates.
    """
    # Vertical stroke
    vertical = np.array([
        [32, 10 + i * 4] for i in range(10)
    ], dtype=np.float32)

    # Horizontal stroke
    horizontal = np.array([
        [10 + i * 4, 32] for i in range(10)
    ], dtype=np.float32)

    # Diagonal stroke
    diagonal = np.array([
        [10 + i * 4, 10 + i * 4] for i in range(10)
    ], dtype=np.float32)

    return [vertical, horizontal, diagonal]


@pytest.fixture
def sample_strokes_T():
    """Return strokes that form a 'T' shape matching mock_glyph_mask.

    Returns:
        list[numpy.ndarray]: Two strokes - horizontal bar and vertical stem.
    """
    # Horizontal bar
    h_stroke = np.array([
        [12, 12], [22, 12], [32, 12], [42, 12], [52, 12]
    ], dtype=np.float32)

    # Vertical stem
    v_stroke = np.array([
        [32, 16], [32, 26], [32, 36], [32, 46], [32, 56]
    ], dtype=np.float32)

    return [h_stroke, v_stroke]


# -----------------------------------------------------------------------------
# Database Session Fixture
# -----------------------------------------------------------------------------

@pytest.fixture
def db_session():
    """Create an in-memory SQLite connection with full schema setup.

    Sets up the database with:
    - All tables from the schema
    - Default removal reasons
    - Row factory for dict-like access

    Yields:
        sqlite3.Connection: Configured database connection.

    Note:
        Connection is automatically closed after the test.
    """
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row

    # Create schema
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

    yield conn

    conn.close()


@pytest.fixture
def db_session_with_font(db_session):
    """Database session with a sample font pre-inserted.

    Extends db_session with a test font entry.

    Yields:
        tuple: (connection, font_id) where font_id is the test font's ID.
    """
    cursor = db_session.execute(
        "INSERT INTO fonts (name, source, file_path) VALUES (?, ?, ?)",
        ('TestFont', 'test', 'fonts/test/TestFont.ttf')
    )
    font_id = cursor.lastrowid
    db_session.commit()

    yield db_session, font_id


# -----------------------------------------------------------------------------
# Flask Client Fixture
# -----------------------------------------------------------------------------

@pytest.fixture
def flask_client():
    """Create a Flask test client for the stroke_editor app.

    Configures the Flask app for testing and returns a test client
    that can be used to make requests without running a server.

    Returns:
        flask.testing.FlaskClient: Test client for making requests.

    Example:
        def test_index(flask_client):
            response = flask_client.get('/')
            assert response.status_code == 200
    """
    from stroke_flask import app
    import stroke_routes_core  # noqa: F401 - registers routes
    import stroke_routes_batch  # noqa: F401 - registers routes
    import stroke_routes_stream  # noqa: F401 - registers routes

    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False

    with app.test_client() as client:
        yield client


@pytest.fixture
def flask_app():
    """Return the configured Flask app instance.

    Useful when you need direct access to the app (e.g., for app context).

    Returns:
        flask.Flask: The Flask application instance.
    """
    from stroke_flask import app
    import stroke_routes_core  # noqa: F401 - registers routes
    import stroke_routes_batch  # noqa: F401 - registers routes
    import stroke_routes_stream  # noqa: F401 - registers routes

    app.config['TESTING'] = True
    return app


# -----------------------------------------------------------------------------
# Bounding Box Fixture
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_bbox():
    """Return a standard bounding box tuple (x, y, w, h).

    Provides a typical bounding box for a character in a 64x64 canvas.
    Format matches what is commonly used in stroke processing.

    Returns:
        tuple: (x, y, width, height) = (10, 10, 44, 44)
    """
    return (10, 10, 44, 44)


@pytest.fixture
def mock_bbox_xyxy():
    """Return a bounding box in (x_min, y_min, x_max, y_max) format.

    Alternative format used by some functions.

    Returns:
        tuple: (x_min, y_min, x_max, y_max) = (10, 10, 54, 54)
    """
    return (10, 10, 54, 54)


# -----------------------------------------------------------------------------
# Additional Utility Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def canvas_size():
    """Return the default canvas size used in tests.

    Returns:
        int: Default canvas size (224 pixels).
    """
    return 224


@pytest.fixture
def temp_font_db(tmp_path, db_session):
    """Create a temporary database file with schema and test data.

    Useful for tests that need to read from an actual file.

    Args:
        tmp_path: pytest's tmp_path fixture
        db_session: Our db_session fixture (for schema)

    Yields:
        Path: Path to the temporary database file.
    """
    db_path = tmp_path / "test_fonts.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Copy schema
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
    conn.close()

    yield db_path


@pytest.fixture
def mock_scoring_context(mock_glyph_mask, mock_bbox):
    """Return a scoring context dict for use with CompositeScorer.

    Returns:
        dict: Context with mask, bbox, and common scoring parameters.
    """
    return {
        'mask': mock_glyph_mask,
        'bbox': mock_bbox,
        'canvas_size': 64,
        'stroke_width': 4,
    }
