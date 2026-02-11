#!/usr/bin/env python3
"""Run all pre-AI pipeline steps on downloaded fonts.

This script processes a directory of font files through a series of filtering
and analysis steps to identify high-quality, usable fonts. It prepares fonts
for downstream AI-based evaluation by eliminating fonts that fail basic
quality checks.

Pipeline Steps:
    1. **Load fonts into database**: Scans the fonts directory, detects the
       source (dafont, fontspace, google), and adds each font to the SQLite
       database.

    2. **Completeness check**: Verifies that fonts contain all required
       glyphs (A-Z, a-z, 0-9, common punctuation). Fonts with less than
       90% completeness are flagged for removal.

    3. **Deduplication**: Uses perceptual hashing (pHash) to identify
       visually similar fonts. When duplicates are found, the highest
       scoring font is kept and others are marked as duplicates.

    4. **Cursive detection**: Identifies cursive/script fonts using two
       methods:
       - Connectivity analysis: Measures how connected letters are
       - Contextual analysis: Examines OpenType features and glyph shapes

Example:
    Process fonts in the default directory::

        $ python run_prefilters.py

    Specify custom database and fonts directory::

        $ python run_prefilters.py --db custom.db --fonts-dir /path/to/fonts/

Attributes:
    This script uses utilities from:
        - db_schema: Database initialization and FontDB context manager
        - font_utils: CompletenessChecker, FontDeduplicator, FontScorer,
          CursiveDetector

Note:
    This script should be run before run_ocr_prefilter.py, which performs
    OCR-based filtering on fonts that pass these initial checks.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from db_schema import FontDB, init_db
from font_utils import (
    CompletenessChecker,
    CursiveDetector,
    FontDeduplicator,
    FontScorer,
)


def find_fonts(fonts_dir: Path) -> list:
    """Find all font files in a directory recursively.

    Searches for font files with common extensions (.ttf, .otf, .woff,
    .woff2) in the specified directory and all subdirectories.

    Args:
        fonts_dir: Path object pointing to the root fonts directory.

    Returns:
        A sorted list of Path objects for each font file found.
        The list is sorted alphabetically by full path.
    """
    extensions = ['.ttf', '.otf', '.woff', '.woff2']
    fonts = []
    for ext in extensions:
        fonts.extend(fonts_dir.rglob(f'*{ext}'))
    return sorted(fonts)


def detect_source(font_path: Path) -> tuple:
    """Detect the font source based on its file path.

    Examines the font's directory path to determine which font repository
    or source it came from. This information is stored in the database
    for tracking and filtering purposes.

    Args:
        font_path: Path object for the font file.

    Returns:
        A tuple of (source_name, url) where:
            - source_name (str): One of 'dafont', 'fontspace', 'google',
              or 'unknown'
            - url (str or None): The source URL if available, otherwise None

    Example:
        >>> detect_source(Path('/fonts/dafont/Arial.ttf'))
        ('dafont', None)
        >>> detect_source(Path('/fonts/random/CustomFont.otf'))
        ('unknown', None)
    """
    path_str = str(font_path).lower()
    if '/dafont/' in path_str:
        return 'dafont', None
    elif '/fontspace/' in path_str:
        return 'fontspace', None
    elif '/google/' in path_str:
        return 'google', None
    return 'unknown', None


def _process_single_font(font_path: Path, db, completeness_checker,
                         cursive_detector, scorer) -> dict | None:
    """Process a single font through all checks.

    Args:
        font_path: Path to the font file.
        db: FontDB database connection.
        completeness_checker: CompletenessChecker instance.
        cursive_detector: CursiveDetector instance.
        scorer: FontScorer instance.

    Returns:
        Dict with font data for deduplication, or None if processing failed.
    """
    # Detect source and add to database
    source, url = detect_source(font_path)
    font_id = db.add_font(
        name=font_path.stem,
        file_path=str(font_path),
        source=source,
        url=url
    )

    # Run all checks
    completeness_score, missing = completeness_checker.check(str(font_path))
    cursive_result = cursive_detector.check_all(str(font_path))
    score_result = scorer.score_font(str(font_path))

    # Update database with check results
    db.update_checks(
        font_id,
        completeness_score=completeness_score,
        missing_glyphs=json.dumps(missing) if missing else None,
        connectivity_score=cursive_result['connectivity_score'],
        is_cursive=cursive_result['is_cursive']
    )

    # Store contextual score
    cursor = db.conn.cursor()
    cursor.execute(
        "UPDATE font_checks SET contextual_score = ? WHERE font_id = ?",
        (cursive_result['contextual_score'], font_id)
    )
    db.conn.commit()

    # Record removal if cursive
    if cursive_result['is_cursive']:
        methods = ', '.join(cursive_result['methods'])
        details = f"ctx={cursive_result['contextual_score']:.3f}, conn={cursive_result['connectivity_score']:.2f}, methods={methods}"
        reason = 'contextual' if 'contextual' in cursive_result['methods'] else 'cursive'
        db.remove_font(font_id, reason, details)

    # Record removal if incomplete (< 90%)
    if completeness_score < 0.9:
        db.remove_font(font_id, 'incomplete',
                       f"score={completeness_score:.1%}, missing={len(missing)}")

    # Return data for deduplication if phash available
    if score_result.get('phash'):
        return {
            'font_id': font_id,
            'path': str(font_path),
            'name': font_path.stem,
            'phash': score_result['phash'],
            'overall': score_result['overall'],
            'completeness': completeness_score,
            'connectivity': cursive_result['connectivity_score'],
            'contextual': cursive_result['contextual_score'],
            'is_cursive': cursive_result['is_cursive'],
        }
    return None


def _run_deduplication(db, font_scores: list, deduplicator) -> None:
    """Find and mark duplicate fonts in the database.

    Args:
        db: FontDB database connection.
        font_scores: List of font score dicts from processing.
        deduplicator: FontDeduplicator instance.
    """
    print(f"Finding duplicates among {len(font_scores)} fonts...")
    duplicate_groups = deduplicator.find_duplicates(font_scores)
    print(f"Found {len(duplicate_groups)} duplicate groups")

    keep_fonts, remove_fonts = deduplicator.select_best_from_groups(duplicate_groups)
    print(f"Keeping {len(keep_fonts)}, removing {len(remove_fonts)} duplicates")

    # Update database with dedup results
    for group_id, group in enumerate(duplicate_groups, start=1):
        sorted_group = sorted(group, key=lambda x: x.get('overall', 0), reverse=True)

        for i, font in enumerate(sorted_group):
            is_best = (i == 0)
            db.update_checks(
                font['font_id'],
                is_duplicate=not is_best,
                duplicate_group_id=group_id,
                keep_in_group=is_best
            )

            if not is_best:
                db.remove_font(font['font_id'], 'duplicate',
                               f"duplicate of {sorted_group[0]['name']} (group {group_id})")


def _print_summary_stats(db) -> None:
    """Print summary statistics for the pipeline run.

    Args:
        db: FontDB database connection.
    """
    cursor = db.conn.cursor()

    # Gather all stats
    cursor.execute("SELECT COUNT(*) FROM fonts")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM font_checks WHERE completeness_score >= 0.9")
    complete = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM font_checks WHERE completeness_score < 0.9")
    incomplete = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM font_checks WHERE is_cursive = 1")
    cursive = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM font_checks WHERE is_cursive = 0")
    not_cursive = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM font_checks WHERE is_duplicate = 1")
    duplicates = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM font_checks
        WHERE completeness_score >= 0.9
          AND is_cursive = 0
          AND (is_duplicate = 0 OR is_duplicate IS NULL)
    """)
    passing = cursor.fetchone()[0]

    # Print summary
    print(f"\nTotal fonts loaded: {total}")
    print(f"  Complete (≥90%): {complete}")
    print(f"  Incomplete (<90%): {incomplete}")
    print(f"  Cursive/contextual: {cursive}")
    print(f"  Not cursive: {not_cursive}")
    print(f"  Duplicates removed: {duplicates}")
    print(f"  ✓ Passing all checks: {passing}")

    # Removal breakdown
    print("\nRemoval breakdown:")
    stats = db.get_removal_stats()
    for code, info in stats.items():
        if info['count'] > 0:
            print(f"  {code}: {info['count']}")


def _print_cursive_report(db) -> None:
    """Print detailed cursive detection report.

    Args:
        db: FontDB database connection.
    """
    cursor = db.conn.cursor()

    # Contextual scores
    print("\n" + "-"*40)
    print("CURSIVE SCORES (contextual)")
    print("-"*40)

    cursor.execute("""
        SELECT f.name, fc.contextual_score, fc.connectivity_score, fc.is_cursive
        FROM fonts f
        JOIN font_checks fc ON f.id = fc.font_id
        WHERE fc.contextual_score IS NOT NULL
        ORDER BY fc.contextual_score DESC
        LIMIT 10
    """)
    print("\nHighest contextual scores (most likely cursive):")
    for row in cursor.fetchall():
        flag = "CURSIVE" if row['is_cursive'] else ""
        print(f"  {row['contextual_score']:.3f} | {row['name'][:40]:40} {flag}")

    cursor.execute("""
        SELECT f.name, fc.contextual_score, fc.connectivity_score, fc.is_cursive
        FROM fonts f
        JOIN font_checks fc ON f.id = fc.font_id
        WHERE fc.contextual_score IS NOT NULL
        ORDER BY fc.contextual_score ASC
        LIMIT 10
    """)
    print("\nLowest contextual scores (most likely print):")
    for row in cursor.fetchall():
        flag = "CURSIVE" if row['is_cursive'] else ""
        print(f"  {row['contextual_score']:.3f} | {row['name'][:40]:40} {flag}")

    # Connectivity scores
    print("\n" + "-"*40)
    print("CONNECTIVITY SCORES")
    print("-"*40)

    cursor.execute("""
        SELECT f.name, fc.connectivity_score, fc.contextual_score
        FROM fonts f
        JOIN font_checks fc ON f.id = fc.font_id
        WHERE fc.connectivity_score > 0
        ORDER BY fc.connectivity_score DESC
        LIMIT 10
    """)
    rows = cursor.fetchall()
    if rows:
        print("\nHighest connectivity scores:")
        for row in rows:
            print(f"  {row['connectivity_score']:.3f} | {row['name'][:40]}")
    else:
        print("\nNo fonts with connectivity > 0 (no truly connected cursive found)")


def run_pipeline(db_path: str, fonts_dir: str) -> None:
    """Run all pre-AI pipeline steps on fonts.

    This is the main orchestration function that executes the complete
    pre-filtering pipeline:

    1. Initializes the database schema
    2. Scans for font files in the specified directory
    3. For each font:
       - Adds to database with source detection
       - Runs completeness check
       - Runs cursive detection (connectivity + contextual)
       - Calculates font scoring for deduplication
    4. Performs deduplication across all fonts
    5. Updates database with all results
    6. Prints detailed statistics and reports

    Args:
        db_path: Path to the SQLite database file. Will be created if it
            doesn't exist.
        fonts_dir: Path to the directory containing font files. Will be
            searched recursively.

    Returns:
        None. Results are printed to stdout and written to the database.

    Note:
        Fonts failing any check are recorded in the font_removals table
        with appropriate reason codes ('incomplete', 'cursive',
        'contextual', 'duplicate').
    """
    fonts_dir = Path(fonts_dir)

    # Initialize database
    print(f"Initializing database: {db_path}")
    init_db(db_path)

    # Find all fonts
    print(f"\nScanning for fonts in: {fonts_dir}")
    font_files = find_fonts(fonts_dir)
    print(f"Found {len(font_files)} font files")

    if not font_files:
        print("No fonts found!")
        return

    # Initialize utilities
    completeness_checker = CompletenessChecker()
    scorer = FontScorer()
    cursive_detector = CursiveDetector()
    deduplicator = FontDeduplicator(threshold=8)

    # Track results for deduplication
    font_scores = []

    with FontDB(db_path) as db:
        # STEP 1-2-4: Load fonts, completeness check, cursive detection
        print("\n" + "="*60)
        print("STEP 1-2-4: Loading fonts, completeness, and cursive detection")
        print("="*60)

        for font_path in tqdm(font_files, desc="Processing fonts"):
            try:
                result = _process_single_font(
                    font_path, db, completeness_checker, cursive_detector, scorer
                )
                if result:
                    font_scores.append(result)
            except Exception as e:
                tqdm.write(f"Error processing {font_path.name}: {e}")
                continue

        # STEP 3: Deduplication
        print("\n" + "="*60)
        print("STEP 3: Deduplication")
        print("="*60)
        _run_deduplication(db, font_scores, deduplicator)

        # REPORT RESULTS
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        _print_summary_stats(db)
        _print_cursive_report(db)


def main() -> None:
    """Parse command-line arguments and run the pre-filter pipeline.

    Command-line Arguments:
        --db: Path to the SQLite database file. Defaults to 'fonts.db'.
            The database will be created if it doesn't exist.
        --fonts-dir: Path to the directory containing font files.
            Defaults to 'fonts/'. Will be searched recursively.

    Returns:
        None. Exits after running the pipeline.
    """
    parser = argparse.ArgumentParser(description='Run pre-AI pipeline steps')
    parser.add_argument('--db', default='fonts.db', help='Database path')
    parser.add_argument('--fonts-dir', default='fonts/', help='Fonts directory')
    args = parser.parse_args()

    run_pipeline(args.db, args.fonts_dir)


if __name__ == '__main__':
    main()
