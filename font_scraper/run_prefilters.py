#!/usr/bin/env python3
"""
Run all pre-AI pipeline steps on downloaded fonts.

Steps:
1. Load fonts into database
2. Completeness check
3. Deduplication (perceptual hash)
4. Cursive detection (connectivity + contextual)

Usage:
    python run_prefilters.py [--db fonts.db] [--fonts-dir fonts/]
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

from db_schema import init_db, FontDB
from font_utils import (
    CompletenessChecker,
    FontDeduplicator,
    FontScorer,
    CursiveDetector,
)


def find_fonts(fonts_dir: Path) -> list:
    """Find all font files in directory."""
    extensions = ['.ttf', '.otf', '.woff', '.woff2']
    fonts = []
    for ext in extensions:
        fonts.extend(fonts_dir.rglob(f'*{ext}'))
    return sorted(fonts)


def detect_source(font_path: Path) -> tuple:
    """Detect font source from path."""
    path_str = str(font_path).lower()
    if '/dafont/' in path_str:
        return 'dafont', None
    elif '/fontspace/' in path_str:
        return 'fontspace', None
    elif '/google/' in path_str:
        return 'google', None
    return 'unknown', None


def run_pipeline(db_path: str, fonts_dir: str):
    """Run all pre-AI pipeline steps."""
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
        # ========================================
        # STEP 1: Load fonts into database
        # STEP 2: Completeness check
        # STEP 4: Cursive detection
        # ========================================
        print("\n" + "="*60)
        print("STEP 1-2-4: Loading fonts, completeness, and cursive detection")
        print("="*60)

        for font_path in tqdm(font_files, desc="Processing fonts"):
            try:
                # Detect source
                source, url = detect_source(font_path)

                # Add to database
                font_id = db.add_font(
                    name=font_path.stem,
                    file_path=str(font_path),
                    source=source,
                    url=url
                )

                # Completeness check
                completeness_score, missing = completeness_checker.check(str(font_path))

                # Cursive detection (both methods)
                cursive_result = cursive_detector.check_all(str(font_path))

                # Font scoring (for dedup)
                score_result = scorer.score_font(str(font_path))

                # Update database
                db.update_checks(
                    font_id,
                    completeness_score=completeness_score,
                    missing_glyphs=json.dumps(missing) if missing else None,
                    connectivity_score=cursive_result['connectivity_score'],
                    is_cursive=cursive_result['is_cursive']
                )

                # Also store contextual score (need to add to update_checks or do raw SQL)
                cursor = db.conn.cursor()
                cursor.execute(
                    "UPDATE font_checks SET contextual_score = ? WHERE font_id = ?",
                    (cursive_result['contextual_score'], font_id)
                )
                db.conn.commit()

                # Track for deduplication
                if score_result.get('phash'):
                    font_scores.append({
                        'font_id': font_id,
                        'path': str(font_path),
                        'name': font_path.stem,
                        'phash': score_result['phash'],
                        'overall': score_result['overall'],
                        'completeness': completeness_score,
                        'connectivity': cursive_result['connectivity_score'],
                        'contextual': cursive_result['contextual_score'],
                        'is_cursive': cursive_result['is_cursive'],
                    })

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

            except Exception as e:
                tqdm.write(f"Error processing {font_path.name}: {e}")
                continue

        # ========================================
        # STEP 3: Deduplication
        # ========================================
        print("\n" + "="*60)
        print("STEP 3: Deduplication")
        print("="*60)

        print(f"Finding duplicates among {len(font_scores)} fonts...")
        duplicate_groups = deduplicator.find_duplicates(font_scores)
        print(f"Found {len(duplicate_groups)} duplicate groups")

        keep_fonts, remove_fonts = deduplicator.select_best_from_groups(duplicate_groups)
        print(f"Keeping {len(keep_fonts)}, removing {len(remove_fonts)} duplicates")

        # Update database with dedup results
        group_id = 0
        for group in duplicate_groups:
            group_id += 1
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

        # ========================================
        # REPORT RESULTS
        # ========================================
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)

        # Get stats
        cursor = db.conn.cursor()

        # Total fonts
        cursor.execute("SELECT COUNT(*) FROM fonts")
        total = cursor.fetchone()[0]

        # Completeness stats
        cursor.execute("SELECT COUNT(*) FROM font_checks WHERE completeness_score >= 0.9")
        complete = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM font_checks WHERE completeness_score < 0.9")
        incomplete = cursor.fetchone()[0]

        # Cursive stats
        cursor.execute("SELECT COUNT(*) FROM font_checks WHERE is_cursive = 1")
        cursive = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM font_checks WHERE is_cursive = 0")
        not_cursive = cursor.fetchone()[0]

        # Duplicate stats
        cursor.execute("SELECT COUNT(*) FROM font_checks WHERE is_duplicate = 1")
        duplicates = cursor.fetchone()[0]

        # Fonts passing all checks
        cursor.execute("""
            SELECT COUNT(*) FROM font_checks
            WHERE completeness_score >= 0.9
              AND is_cursive = 0
              AND (is_duplicate = 0 OR is_duplicate IS NULL)
        """)
        passing = cursor.fetchone()[0]

        print(f"\nTotal fonts loaded: {total}")
        print(f"  Complete (≥90%): {complete}")
        print(f"  Incomplete (<90%): {incomplete}")
        print(f"  Cursive/contextual: {cursive}")
        print(f"  Not cursive: {not_cursive}")
        print(f"  Duplicates removed: {duplicates}")
        print(f"  ✓ Passing all checks: {passing}")

        # Removal stats
        print("\nRemoval breakdown:")
        stats = db.get_removal_stats()
        for code, info in stats.items():
            if info['count'] > 0:
                print(f"  {code}: {info['count']}")

        # Cursive scores - highest and lowest
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

        # Also show connectivity scores
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


def main():
    parser = argparse.ArgumentParser(description='Run pre-AI pipeline steps')
    parser.add_argument('--db', default='fonts.db', help='Database path')
    parser.add_argument('--fonts-dir', default='fonts/', help='Fonts directory')
    args = parser.parse_args()

    run_pipeline(args.db, args.fonts_dir)


if __name__ == '__main__':
    main()
