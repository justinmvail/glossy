#!/usr/bin/env python3
"""Run OCR prefilter on all passing fonts using batch Docker processing.

This script filters fonts by rendering sample text and using TrOCR (Transformer
OCR) to verify that the text is readable. Fonts that produce unreadable or
heavily distorted text are filtered out, as they are likely decorative,
symbol-based, or otherwise unsuitable for standard text rendering.

The process involves:
    1. Query database for fonts that passed previous checks (completeness,
       non-cursive, non-duplicate)
    2. Render sample text with each font to a temporary image
    3. Run TrOCR in batch mode via Docker with GPU acceleration
    4. Calculate confidence scores using Levenshtein distance
    5. Update database with pass/fail status and confidence scores

The TrOCR model is loaded once and reused for all images, making batch
processing significantly faster than per-font processing.

Example:
    Default settings::

        $ python3 run_ocr_prefilter.py

    Custom database and threshold::

        $ python3 run_ocr_prefilter.py --db custom.db --threshold 0.8

Attributes:
    SAMPLE_TEXT (str): The text rendered and verified via OCR.

Note:
    Requires Docker with GPU support and the 'trocr:latest' image.
    The HuggingFace cache is mounted to avoid re-downloading the model.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from db_schema import FontDB

SAMPLE_TEXT = "Hello World 123"


def render_sample(font_path: str, output_path: str) -> bool:
    """Render sample text with a font and save as an image.

    Creates a white background image with black text rendered using the
    specified font. The image is sized to fit the text with padding.

    Args:
        font_path: Absolute path to the font file (.ttf, .otf, etc.).
        output_path: Path where the rendered PNG image will be saved.

    Returns:
        True if the image was successfully rendered and saved, False if
        the font could not be loaded or any error occurred.
    """
    font_size = 48
    padding = 20

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return False

    # Calculate size
    temp_img = Image.new('RGB', (1, 1), 'white')
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), SAMPLE_TEXT, font=font)
    width = bbox[2] - bbox[0] + padding * 2
    height = bbox[3] - bbox[1] + padding * 2

    # Create final image
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), SAMPLE_TEXT,
              font=font, fill='black')

    img.save(output_path)
    return True


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein edit distance between two strings.

    The Levenshtein distance is the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to change
    one string into the other.

    Args:
        s1: First string for comparison.
        s2: Second string for comparison.

    Returns:
        The minimum number of edits needed to transform s1 into s2.

    Example:
        >>> levenshtein_distance("hello", "hallo")
        1
        >>> levenshtein_distance("cat", "dog")
        3
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def calculate_confidence(expected: str, actual: str) -> float:
    """Calculate OCR confidence using normalized edit distance.

    Computes a confidence score between 0.0 and 1.0 based on how closely
    the OCR result matches the expected text. Uses case-insensitive
    comparison with leading/trailing whitespace stripped.

    Args:
        expected: The text that was rendered (ground truth).
        actual: The text recognized by OCR.

    Returns:
        A float between 0.0 and 1.0 where:
            - 1.0 means exact match (zero edits needed)
            - 0.0 means completely different (all characters wrong)

    Example:
        >>> calculate_confidence("Hello World", "Hello World")
        1.0
        >>> calculate_confidence("Hello World", "hello world")
        1.0
        >>> calculate_confidence("Hello World", "Hell World")
        0.909...
    """
    if not actual:
        return 0.0

    expected_lower = expected.lower().strip()
    actual_lower = actual.lower().strip()

    if not expected_lower:
        return 0.0

    distance = levenshtein_distance(expected_lower, actual_lower)
    max_len = max(len(expected_lower), len(actual_lower))

    # Normalize: 0 distance = 1.0 confidence, distance == max_len = 0.0
    confidence = 1.0 - (distance / max_len)
    return max(0.0, confidence)


def get_passing_fonts(db_path: str) -> list:
    """Get all fonts that passed pre-AI checks and need OCR prefiltering.

    Queries the database for fonts that meet the following criteria:
        - Completeness score >= 90%
        - Not flagged as cursive
        - Not flagged as duplicate (or duplicate status is NULL)
        - Prefilter has not yet been run (prefilter_passed is NULL)

    Args:
        db_path: Path to the SQLite fonts database.

    Returns:
        A list of dictionaries, each containing:
            - id (int): Font ID in the database
            - name (str): Font name
            - file_path (str): Absolute path to the font file
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT f.id, f.name, f.file_path FROM fonts f
        JOIN font_checks fc ON f.id = fc.font_id
        WHERE fc.completeness_score >= 0.9
          AND fc.is_cursive = 0
          AND (fc.is_duplicate = 0 OR fc.is_duplicate IS NULL)
          AND fc.prefilter_passed IS NULL
        ORDER BY f.id
    """)

    fonts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return fonts


def run_batch_ocr(image_dir: str, image_files: list) -> dict:
    """Run TrOCR on all images in batch mode via Docker.

    Executes TrOCR model inside a Docker container with GPU support.
    The model is loaded once and reused for all images, making this
    significantly faster than processing images individually.

    Args:
        image_dir: Absolute path to directory containing the images.
        image_files: List of image filenames (not full paths) to process.

    Returns:
        A dictionary mapping filename to result, where each result is a
        dict containing:
            - text (str or None): Recognized text, or None on error
            - error (str or None): Error message, or None on success

        Returns None if Docker execution failed entirely.

    Note:
        Requires Docker with GPU support (--gpus all) and the
        'trocr:latest' image. The HuggingFace model cache is mounted
        from ~/.cache/huggingface to avoid re-downloading.
    """

    # Create batch processing script
    batch_script = '''
import sys
import time
import json
import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

image_dir = sys.argv[1]
image_files = json.loads(sys.argv[2])

# Load model once
print("LOADING_MODEL", flush=True)
t0 = time.perf_counter()
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
model_load_time = time.perf_counter() - t0
print(f"MODEL_LOADED {model_load_time:.2f}s", flush=True)

# Move to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"DEVICE {device}", flush=True)

# Process each image
results = {}
for i, filename in enumerate(image_files):
    image_path = os.path.join(image_dir, filename)
    try:
        image = Image.open(image_path).convert('RGB')
        pixel_values = processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=64)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results[filename] = {'text': text, 'error': None}
    except Exception as e:
        results[filename] = {'text': None, 'error': str(e)}

    # Progress update every 10 fonts
    if (i + 1) % 10 == 0:
        print(f"PROGRESS {i + 1}/{len(image_files)}", flush=True)

print("RESULTS_START", flush=True)
print(json.dumps(results), flush=True)
print("RESULTS_END", flush=True)
'''

    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(batch_script)
        script_path = f.name

    try:
        # Prepare image list as JSON
        image_list_json = json.dumps(image_files)

        cmd = [
            'docker', 'run', '--rm', '--gpus', 'all',
            '-v', f'{image_dir}:/data',
            '-v', f'{script_path}:/app/batch_ocr.py',
            '-v', f'{os.path.expanduser("~")}/.cache/huggingface:/root/.cache/huggingface',
            'trocr:latest',
            'python3', '/app/batch_ocr.py', '/data', image_list_json
        ]

        # Run with realtime output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        results_json = None
        in_results = False
        results_lines = []

        for line in process.stdout:
            line = line.strip()
            if line.startswith('LOADING_MODEL'):
                print("  Loading TrOCR model...")
            elif line.startswith('MODEL_LOADED'):
                print(f"  Model loaded ({line.split()[1]})")
            elif line.startswith('DEVICE'):
                print(f"  Using device: {line.split()[1]}")
            elif line.startswith('PROGRESS'):
                parts = line.split()
                print(f"  Processing: {parts[1]}")
            elif line == 'RESULTS_START':
                in_results = True
            elif line == 'RESULTS_END':
                in_results = False
                results_json = ''.join(results_lines)
            elif in_results:
                results_lines.append(line)

        process.wait()

        if process.returncode != 0:
            stderr = process.stderr.read()
            print(f"  Docker error: {stderr[:500]}")
            return None

        if results_json:
            return json.loads(results_json)
        return None

    finally:
        os.unlink(script_path)


def _render_all_samples(fonts: list, temp_dir: str) -> tuple[dict, list]:
    """Render sample images for all fonts.

    Args:
        fonts: List of font dicts with 'id' and 'file_path' keys.
        temp_dir: Directory to save rendered images.

    Returns:
        Tuple of (font_to_file dict, render_errors list).
    """
    font_to_file = {}  # font_id -> filename
    render_errors = []

    for font in tqdm(fonts, desc="  Rendering"):
        filename = f"{font['id']}.png"
        output_path = os.path.join(temp_dir, filename)

        if os.path.exists(font['file_path']):
            if render_sample(font['file_path'], output_path):
                font_to_file[font['id']] = filename
            else:
                render_errors.append(font['id'])
        else:
            render_errors.append(font['id'])

    return font_to_file, render_errors


def _process_ocr_results(db_path: str, results: dict, font_to_file: dict,
                         threshold: float) -> dict:
    """Process OCR results and update database.

    Args:
        db_path: Path to SQLite database.
        results: OCR results dict from run_batch_ocr().
        font_to_file: Mapping of font_id to filename.
        threshold: Confidence threshold for passing.

    Returns:
        Dict with 'passed', 'failed', 'errors' counts.
    """
    passed = 0
    failed = 0
    errors = 0

    # Build reverse mapping
    file_to_font = {v: k for k, v in font_to_file.items()}

    with FontDB(db_path) as db:
        for filename, result in results.items():
            font_id = file_to_font.get(filename)
            if not font_id:
                continue

            if result['error']:
                errors += 1
                continue

            ocr_text = result['text']
            confidence = calculate_confidence(SAMPLE_TEXT, ocr_text)
            prefilter_passed = confidence >= threshold

            if prefilter_passed:
                passed += 1
            else:
                failed += 1

            # Update database
            db.update_checks(
                font_id,
                prefilter_passed=prefilter_passed,
                prefilter_confidence=confidence,
                prefilter_ocr_text=ocr_text
            )

            # Record removal if failed
            if not prefilter_passed:
                db.remove_font(
                    font_id,
                    'ocr_prefilter',
                    f"confidence={confidence:.1%}, expected='{SAMPLE_TEXT}', got='{ocr_text}'"
                )

    return {'passed': passed, 'failed': failed, 'errors': errors}


def _print_summary(fonts: list, counts: dict, render_errors: list,
                   timings: dict, threshold: float) -> None:
    """Print results and timing summary.

    Args:
        fonts: List of processed fonts.
        counts: Dict with 'passed', 'failed', 'errors' counts.
        render_errors: List of font IDs that failed to render.
        timings: Dict of timing measurements.
        threshold: Confidence threshold used.
    """
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total fonts processed: {len(fonts)}")
    print(f"  Passed (>={threshold:.0%}): {counts['passed']}")
    print(f"  Failed (<{threshold:.0%}): {counts['failed']}")
    print(f"  Errors: {counts['errors']}")
    print(f"  Render errors: {len(render_errors)}")

    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"  db_query:   {timings['db_query']:>7.2f}s")
    print(f"  render:     {timings['render']:>7.2f}s")
    print(f"  ocr:        {timings['ocr']:>7.2f}s")
    print(f"  db_update:  {timings['db_update']:>7.2f}s")
    print("  ─────────────────────────")
    print(f"  TOTAL:      {timings['total']:>7.2f}s ({timings['total']/60:.1f} min)")


def _print_sample_results(db_path: str, threshold: float) -> None:
    """Print sample OCR results from database.

    Args:
        db_path: Path to SQLite database.
        threshold: Confidence threshold for pass/fail display.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("SAMPLE RESULTS")
    print("=" * 60)

    print("\nHighest confidence (best OCR):")
    cursor.execute("""
        SELECT f.name, fc.prefilter_confidence, fc.prefilter_ocr_text
        FROM fonts f
        JOIN font_checks fc ON f.id = fc.font_id
        WHERE fc.prefilter_confidence IS NOT NULL
        ORDER BY fc.prefilter_confidence DESC
        LIMIT 5
    """)
    for row in cursor.fetchall():
        status = "PASS" if row['prefilter_confidence'] >= threshold else "FAIL"
        print(f"  {row['prefilter_confidence']:.1%} [{status}] {row['name'][:30]:30} -> '{row['prefilter_ocr_text']}'")

    print("\nLowest confidence (worst OCR):")
    cursor.execute("""
        SELECT f.name, fc.prefilter_confidence, fc.prefilter_ocr_text
        FROM fonts f
        JOIN font_checks fc ON f.id = fc.font_id
        WHERE fc.prefilter_confidence IS NOT NULL
        ORDER BY fc.prefilter_confidence ASC
        LIMIT 5
    """)
    for row in cursor.fetchall():
        status = "PASS" if row['prefilter_confidence'] >= threshold else "FAIL"
        print(f"  {row['prefilter_confidence']:.1%} [{status}] {row['name'][:30]:30} -> '{row['prefilter_ocr_text']}'")

    conn.close()


def run_prefilter(db_path: str, threshold: float = 0.7):
    """Run OCR prefilter on all passing fonts.

    This is the main orchestration function that:
        1. Queries the database for fonts to process
        2. Renders sample images for each font
        3. Runs batch OCR via Docker
        4. Calculates confidence scores
        5. Updates the database with results
        6. Prints summary statistics

    Args:
        db_path: Path to the SQLite fonts database.
        threshold: Minimum confidence score (0.0-1.0) for a font to pass.
            Defaults to 0.7 (70% match required).

    Returns:
        None. Results are printed to stdout and written to the database.

    Note:
        Fonts that fail the prefilter are recorded in the font_removals
        table with reason 'ocr_prefilter' and details including the
        expected text, recognized text, and confidence score.
    """
    import shutil

    print("=" * 60)
    print("OCR PREFILTER - BATCH MODE")
    print("=" * 60)

    total_start = time.perf_counter()
    timings = {}

    # Step 1: Get fonts
    print("\n[1] Getting fonts from database...")
    t0 = time.perf_counter()
    fonts = get_passing_fonts(db_path)
    timings['db_query'] = time.perf_counter() - t0
    print(f"  Found {len(fonts)} fonts to process ({timings['db_query']:.2f}s)")

    if not fonts:
        print("  No fonts to process!")
        return

    # Step 2: Render all samples
    print("\n[2] Rendering sample images...")
    t0 = time.perf_counter()
    temp_dir = tempfile.mkdtemp(prefix='ocr_prefilter_')
    print(f"  Temp dir: {temp_dir}")

    font_to_file, render_errors = _render_all_samples(fonts, temp_dir)

    timings['render'] = time.perf_counter() - t0
    print(f"  Rendered {len(font_to_file)} images ({timings['render']:.2f}s)")
    if render_errors:
        print(f"  Render errors: {len(render_errors)}")

    # Step 3: Run batch OCR
    print("\n[3] Running TrOCR (batch mode)...")
    t0 = time.perf_counter()

    image_files = list(font_to_file.values())
    results = run_batch_ocr(temp_dir, image_files)

    timings['ocr'] = time.perf_counter() - t0
    print(f"  OCR complete ({timings['ocr']:.2f}s)")

    if not results:
        print("  ERROR: OCR failed!")
        shutil.rmtree(temp_dir)
        return

    # Step 4: Process results and update database
    print("\n[4] Processing results and updating database...")
    t0 = time.perf_counter()

    counts = _process_ocr_results(db_path, results, font_to_file, threshold)

    timings['db_update'] = time.perf_counter() - t0
    print(f"  Database updated ({timings['db_update']:.2f}s)")

    # Cleanup temp directory
    shutil.rmtree(temp_dir)

    # Summary
    timings['total'] = time.perf_counter() - total_start
    _print_summary(fonts, counts, render_errors, timings, threshold)
    _print_sample_results(db_path, threshold)


def main():
    """Parse command-line arguments and run the OCR prefilter.

    Command-line Arguments:
        --db: Path to the SQLite fonts database. Defaults to 'fonts.db'.
        --threshold: Minimum confidence score (0.0-1.0) for a font to pass.
            Defaults to 0.7.

    Returns:
        None. Exits after running the prefilter.
    """
    parser = argparse.ArgumentParser(description='Run OCR prefilter on all passing fonts')
    parser.add_argument('--db', default='fonts.db', help='Database path')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold (default: 0.7)')
    args = parser.parse_args()

    run_prefilter(args.db, args.threshold)


if __name__ == '__main__':
    main()
