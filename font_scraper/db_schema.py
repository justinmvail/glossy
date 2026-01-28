"""
Database schema for font training pipeline.

Usage:
    from db_schema import init_db, get_connection

    # Initialize database
    init_db('fonts.db')

    # Use in pipeline
    conn = get_connection('fonts.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO fonts (name, source, file_path) VALUES (?, ?, ?)",
                   (font_name, 'dafont', font_path))
    conn.commit()
"""

import sqlite3
from pathlib import Path
from typing import Optional


SCHEMA = """
-- Fonts table
CREATE TABLE IF NOT EXISTS fonts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT,  -- 'dafont', 'fontspace', 'google'
    url TEXT,
    category TEXT,
    license TEXT,
    variant TEXT,  -- 'Regular', 'Bold', 'Italic', etc.
    file_path TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Removal reasons lookup table
CREATE TABLE IF NOT EXISTS removal_reasons (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,  -- 'incomplete', 'duplicate', 'cursive', 'ocr_fail', 'manual', etc.
    description TEXT
);

-- Track removed fonts and why
CREATE TABLE IF NOT EXISTS font_removals (
    id INTEGER PRIMARY KEY,
    font_id INTEGER REFERENCES fonts(id),
    reason_id INTEGER REFERENCES removal_reasons(id),
    details TEXT,  -- Additional context (e.g., "duplicate of font_id 123")
    removed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Font quality checks
CREATE TABLE IF NOT EXISTS font_checks (
    id INTEGER PRIMARY KEY,
    font_id INTEGER UNIQUE REFERENCES fonts(id),

    -- Completeness
    completeness_score REAL,
    missing_glyphs TEXT,  -- JSON array

    -- Deduplication
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_group_id INTEGER,
    keep_in_group BOOLEAN,

    -- Cursive detection
    connectivity_score REAL,
    contextual_score REAL,
    is_cursive BOOLEAN,

    -- OCR prefilter
    prefilter_image_path TEXT,
    prefilter_ocr_text TEXT,
    prefilter_confidence REAL,
    prefilter_passed BOOLEAN,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual characters
CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY,
    font_id INTEGER REFERENCES fonts(id),
    char TEXT NOT NULL,

    -- Rendering
    image_path TEXT,

    -- Strokes
    strokes_raw TEXT,  -- JSON
    strokes_processed TEXT,  -- JSON
    point_count INTEGER,

    -- Best OCR result (denormalized for quick access)
    best_ocr_result TEXT,
    best_ocr_confidence REAL,
    best_ocr_match BOOLEAN,

    -- Quality
    quality_score REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(font_id, char)
);

-- OCR run history per character
CREATE TABLE IF NOT EXISTS ocr_runs (
    id INTEGER PRIMARY KEY,
    character_id INTEGER REFERENCES characters(id),

    -- What was OCR'd
    stage TEXT NOT NULL,  -- 'raw_strokes', 'processed_strokes', 'prefilter'
    image_path TEXT,

    -- Results
    ocr_result TEXT,
    ocr_confidence REAL,
    ocr_match BOOLEAN,

    -- Metadata
    model TEXT,  -- 'trocr', 'tesseract', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pre-populate removal reasons
INSERT OR IGNORE INTO removal_reasons (code, description) VALUES
    ('incomplete', 'Missing required glyphs'),
    ('duplicate', 'Duplicate of another font'),
    ('cursive', 'Cursive/connected letterforms'),
    ('contextual', 'Has contextual alternates'),
    ('ocr_prefilter', 'Failed OCR prefilter'),
    ('ocr_validation', 'Failed OCR validation after processing'),
    ('low_quality', 'Quality score below threshold'),
    ('manual', 'Manually rejected during review'),
    ('load_error', 'Could not load font file');

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_fonts_source ON fonts(source);
CREATE INDEX IF NOT EXISTS idx_fonts_category ON fonts(category);
CREATE INDEX IF NOT EXISTS idx_font_checks_font ON font_checks(font_id);
CREATE INDEX IF NOT EXISTS idx_font_checks_cursive ON font_checks(is_cursive);
CREATE INDEX IF NOT EXISTS idx_font_checks_passed ON font_checks(prefilter_passed);
CREATE INDEX IF NOT EXISTS idx_font_checks_duplicate ON font_checks(is_duplicate);
CREATE INDEX IF NOT EXISTS idx_characters_font ON characters(font_id);
CREATE INDEX IF NOT EXISTS idx_characters_quality ON characters(quality_score);
CREATE INDEX IF NOT EXISTS idx_characters_char ON characters(char);
CREATE INDEX IF NOT EXISTS idx_font_removals_font ON font_removals(font_id);
CREATE INDEX IF NOT EXISTS idx_font_removals_reason ON font_removals(reason_id);
CREATE INDEX IF NOT EXISTS idx_ocr_runs_character ON ocr_runs(character_id);
CREATE INDEX IF NOT EXISTS idx_ocr_runs_stage ON ocr_runs(stage);
"""


def init_db(db_path: str = 'fonts.db') -> sqlite3.Connection:
    """
    Initialize database with schema.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Database connection
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_connection(db_path: str = 'fonts.db') -> sqlite3.Connection:
    """Get a database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


class FontDB:
    """Helper class for common database operations."""

    def __init__(self, db_path: str = 'fonts.db'):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = get_connection(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def add_font(
        self,
        name: str,
        file_path: str,
        source: Optional[str] = None,
        url: Optional[str] = None,
        category: Optional[str] = None,
        license: Optional[str] = None,
        variant: Optional[str] = None
    ) -> int:
        """Add a font to the database. Returns font_id."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO fonts (name, file_path, source, url, category, license, variant)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, file_path, source, url, category, license, variant))
        self.conn.commit()

        # Get the id (either new or existing)
        cursor.execute("SELECT id FROM fonts WHERE file_path = ?", (file_path,))
        return cursor.fetchone()[0]

    def update_checks(
        self,
        font_id: int,
        completeness_score: Optional[float] = None,
        missing_glyphs: Optional[str] = None,
        is_duplicate: Optional[bool] = None,
        duplicate_group_id: Optional[int] = None,
        keep_in_group: Optional[bool] = None,
        connectivity_score: Optional[float] = None,
        is_cursive: Optional[bool] = None,
        prefilter_passed: Optional[bool] = None,
        prefilter_confidence: Optional[float] = None,
        prefilter_ocr_text: Optional[str] = None,
        prefilter_image_path: Optional[str] = None
    ):
        """Update font check results."""
        # Ensure row exists
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO font_checks (font_id) VALUES (?)",
            (font_id,)
        )

        # Build update query dynamically for non-None values
        updates = []
        values = []

        fields = [
            ('completeness_score', completeness_score),
            ('missing_glyphs', missing_glyphs),
            ('is_duplicate', is_duplicate),
            ('duplicate_group_id', duplicate_group_id),
            ('keep_in_group', keep_in_group),
            ('connectivity_score', connectivity_score),
            ('is_cursive', is_cursive),
            ('prefilter_passed', prefilter_passed),
            ('prefilter_confidence', prefilter_confidence),
            ('prefilter_ocr_text', prefilter_ocr_text),
            ('prefilter_image_path', prefilter_image_path),
        ]

        for field, value in fields:
            if value is not None:
                updates.append(f"{field} = ?")
                values.append(value)

        if updates:
            values.append(font_id)
            cursor.execute(
                f"UPDATE font_checks SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE font_id = ?",
                values
            )
            self.conn.commit()

    def add_character(
        self,
        font_id: int,
        char: str,
        image_path: Optional[str] = None,
        strokes_raw: Optional[str] = None,
        strokes_processed: Optional[str] = None,
        point_count: Optional[int] = None,
        ocr_result: Optional[str] = None,
        ocr_confidence: Optional[float] = None,
        ocr_match: Optional[bool] = None,
        quality_score: Optional[float] = None
    ) -> int:
        """Add or update a character record. Returns character_id."""
        cursor = self.conn.cursor()

        # Try insert first
        cursor.execute("""
            INSERT OR IGNORE INTO characters (font_id, char)
            VALUES (?, ?)
        """, (font_id, char))

        # Get id
        cursor.execute(
            "SELECT id FROM characters WHERE font_id = ? AND char = ?",
            (font_id, char)
        )
        char_id = cursor.fetchone()[0]

        # Update fields
        updates = []
        values = []

        fields = [
            ('image_path', image_path),
            ('strokes_raw', strokes_raw),
            ('strokes_processed', strokes_processed),
            ('point_count', point_count),
            ('ocr_result', ocr_result),
            ('ocr_confidence', ocr_confidence),
            ('ocr_match', ocr_match),
            ('quality_score', quality_score),
        ]

        for field, value in fields:
            if value is not None:
                updates.append(f"{field} = ?")
                values.append(value)

        if updates:
            values.append(char_id)
            cursor.execute(
                f"UPDATE characters SET {', '.join(updates)} WHERE id = ?",
                values
            )
            self.conn.commit()

        return char_id

    def get_fonts_needing_check(self, check_type: str) -> list:
        """Get fonts that haven't completed a specific check."""
        cursor = self.conn.cursor()

        if check_type == 'completeness':
            cursor.execute("""
                SELECT f.* FROM fonts f
                LEFT JOIN font_checks fc ON f.id = fc.font_id
                WHERE fc.completeness_score IS NULL OR fc.font_id IS NULL
            """)
        elif check_type == 'cursive':
            cursor.execute("""
                SELECT f.* FROM fonts f
                LEFT JOIN font_checks fc ON f.id = fc.font_id
                WHERE fc.is_cursive IS NULL OR fc.font_id IS NULL
            """)
        elif check_type == 'prefilter':
            cursor.execute("""
                SELECT f.* FROM fonts f
                JOIN font_checks fc ON f.id = fc.font_id
                WHERE fc.is_cursive = FALSE
                  AND fc.prefilter_passed IS NULL
            """)
        else:
            return []

        return [dict(row) for row in cursor.fetchall()]

    def get_fonts_for_rendering(self) -> list:
        """Get fonts that passed all checks and are ready for character rendering."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT f.* FROM fonts f
            JOIN font_checks fc ON f.id = fc.font_id
            WHERE fc.is_cursive = FALSE
              AND fc.is_duplicate = FALSE
              AND fc.prefilter_passed = TRUE
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_best_fonts(self, limit: int = 100) -> list:
        """Get fonts ranked by average character quality."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT f.name, f.source, f.category,
                   AVG(c.quality_score) as avg_quality,
                   COUNT(c.id) as char_count
            FROM fonts f
            JOIN font_checks fc ON f.id = fc.font_id
            JOIN characters c ON f.id = c.font_id
            WHERE fc.is_cursive = FALSE
              AND fc.prefilter_passed = TRUE
              AND fc.is_duplicate = FALSE
              AND c.quality_score IS NOT NULL
            GROUP BY f.id
            ORDER BY avg_quality DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def add_ocr_run(
        self,
        character_id: int,
        stage: str,
        ocr_result: Optional[str] = None,
        ocr_confidence: Optional[float] = None,
        ocr_match: Optional[bool] = None,
        image_path: Optional[str] = None,
        model: Optional[str] = None
    ) -> int:
        """
        Record an OCR run for a character.

        Args:
            character_id: Foreign key to characters table
            stage: 'raw_strokes', 'processed_strokes', or 'prefilter'
            ocr_result: The text OCR detected
            ocr_confidence: Confidence score 0-1
            ocr_match: Whether it matched the expected character
            image_path: Path to the image that was OCR'd
            model: OCR model used ('trocr', 'tesseract', etc.)

        Returns:
            ocr_run id
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO ocr_runs (character_id, stage, ocr_result, ocr_confidence,
                                  ocr_match, image_path, model)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (character_id, stage, ocr_result, ocr_confidence, ocr_match,
              image_path, model))
        self.conn.commit()

        # Update best result on character if this is better
        if ocr_confidence is not None:
            cursor.execute("""
                UPDATE characters
                SET best_ocr_result = ?,
                    best_ocr_confidence = ?,
                    best_ocr_match = ?
                WHERE id = ?
                  AND (best_ocr_confidence IS NULL OR best_ocr_confidence < ?)
            """, (ocr_result, ocr_confidence, ocr_match, character_id, ocr_confidence))
            self.conn.commit()

        return cursor.lastrowid

    def remove_font(
        self,
        font_id: int,
        reason_code: str,
        details: Optional[str] = None
    ):
        """
        Record that a font was removed from the pipeline.

        Args:
            font_id: The font being removed
            reason_code: One of the codes in removal_reasons table
            details: Additional context
        """
        cursor = self.conn.cursor()

        # Get reason_id
        cursor.execute(
            "SELECT id FROM removal_reasons WHERE code = ?",
            (reason_code,)
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Unknown removal reason: {reason_code}")
        reason_id = row[0]

        cursor.execute("""
            INSERT INTO font_removals (font_id, reason_id, details)
            VALUES (?, ?, ?)
        """, (font_id, reason_id, details))
        self.conn.commit()

    def get_removed_fonts(self, reason_code: Optional[str] = None) -> list:
        """Get list of removed fonts, optionally filtered by reason."""
        cursor = self.conn.cursor()

        if reason_code:
            cursor.execute("""
                SELECT f.*, rr.code as reason_code, rr.description as reason_desc,
                       fr.details, fr.removed_at
                FROM font_removals fr
                JOIN fonts f ON fr.font_id = f.id
                JOIN removal_reasons rr ON fr.reason_id = rr.id
                WHERE rr.code = ?
                ORDER BY fr.removed_at DESC
            """, (reason_code,))
        else:
            cursor.execute("""
                SELECT f.*, rr.code as reason_code, rr.description as reason_desc,
                       fr.details, fr.removed_at
                FROM font_removals fr
                JOIN fonts f ON fr.font_id = f.id
                JOIN removal_reasons rr ON fr.reason_id = rr.id
                ORDER BY fr.removed_at DESC
            """)

        return [dict(row) for row in cursor.fetchall()]

    def get_removal_stats(self) -> dict:
        """Get counts of removed fonts by reason."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT rr.code, rr.description, COUNT(fr.id) as count
            FROM removal_reasons rr
            LEFT JOIN font_removals fr ON rr.id = fr.reason_id
            GROUP BY rr.id
            ORDER BY count DESC
        """)
        return {row['code']: {'description': row['description'], 'count': row['count']}
                for row in cursor.fetchall()}

    def get_ocr_history(self, character_id: int) -> list:
        """Get all OCR runs for a character."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM ocr_runs
            WHERE character_id = ?
            ORDER BY created_at DESC
        """, (character_id,))
        return [dict(row) for row in cursor.fetchall()]


if __name__ == '__main__':
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else 'fonts.db'
    print(f"Initializing database: {db_path}")

    conn = init_db(db_path)
    print("Schema created successfully.")

    # Show tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables: {[t[0] for t in tables]}")

    conn.close()
