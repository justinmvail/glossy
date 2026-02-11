#!/usr/bin/env python3
"""Single-font OCR prefilter test with detailed timing analysis.

This script tests the OCR-based font prefiltering pipeline on a single font,
providing detailed timing breakdowns for each stage of processing. It uses
TrOCR (Transformer-based OCR) running in a Docker container with GPU support
to verify that fonts render legible text.

The OCR prefilter is used to identify fonts that produce readable output,
filtering out decorative or symbol fonts that would not be suitable for
handwriting synthesis.

Pipeline stages tested:
    1. Database query: Retrieve font metadata from SQLite
    2. Rendering: Generate sample text image using the font
    3. OCR inference: Run TrOCR model via Docker
    4. Confidence calculation: Compare OCR output to expected text

The script also provides time estimates for processing the full font database
in both batch mode (single container) and single mode (new container per font).

Example:
    Test a random passing font::

        $ python3 test_ocr_single.py

    Test a specific font by ID::

        $ python3 test_ocr_single.py --font-id 42

    Use a different database::

        $ python3 test_ocr_single.py --db custom_fonts.db

Attributes:
    DEFAULT_SAMPLE_TEXT: The text rendered and recognized ("Hello World 123").
    CONFIDENCE_THRESHOLD: Minimum OCR confidence to pass (0.7 = 70%).
"""

import argparse
import json
import os
import subprocess
import tempfile
import time

from PIL import Image, ImageDraw, ImageFont


def timed(name):
    """Decorator factory to time function execution.

    Creates a decorator that measures and prints the execution time of
    the wrapped function.

    Args:
        name: Label to display in timing output.

    Returns:
        callable: Decorator function that wraps the target function.

    Example:
        >>> @timed("my_operation")
        ... def slow_function():
        ...     time.sleep(1)
        ...     return "done"
        >>> result, elapsed = slow_function()
        [my_operation] 1.001s
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"  [{name}] {elapsed:.3f}s")
            return result, elapsed
        return wrapper
    return decorator


class OCRTester:
    """Orchestrates OCR testing for font prefiltering.

    This class manages the complete OCR testing workflow including database
    queries, font rendering, Docker-based OCR inference, and result analysis.

    Attributes:
        db_path: Path to the SQLite fonts database.
        sample_text: Text string to render and recognize.
        timings: Dictionary storing timing measurements for each stage.

    Example:
        >>> tester = OCRTester('fonts.db')
        >>> tester.run(font_id=42)
    """

    def __init__(self, db_path: str = 'fonts.db'):
        """Initialize the OCR tester.

        Args:
            db_path: Path to SQLite database containing font metadata.
                Defaults to 'fonts.db' in the current directory.
        """
        self.db_path = db_path
        self.sample_text = "Hello World 123"
        self.timings = {}

    def get_passing_font(self, font_id: int = None):
        """Retrieve a font that passed all pre-AI quality checks.

        Queries the database for fonts meeting the following criteria:
            - Completeness score >= 90% (has required glyphs)
            - Not cursive (is_cursive = 0)
            - Not a duplicate (is_duplicate = 0 or NULL)

        Args:
            font_id: Specific font ID to retrieve. If None, returns the
                first matching font. Defaults to None.

        Returns:
            dict: Font record with keys 'id', 'name', 'file_path', or None
                if no matching font is found.
        """
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if font_id:
            cursor.execute("""
                SELECT f.id, f.name, f.file_path FROM fonts f
                JOIN font_checks fc ON f.id = fc.font_id
                WHERE f.id = ?
                  AND fc.completeness_score >= 0.9
                  AND fc.is_cursive = 0
                  AND (fc.is_duplicate = 0 OR fc.is_duplicate IS NULL)
            """, (font_id,))
        else:
            cursor.execute("""
                SELECT f.id, f.name, f.file_path FROM fonts f
                JOIN font_checks fc ON f.id = fc.font_id
                WHERE fc.completeness_score >= 0.9
                  AND fc.is_cursive = 0
                  AND (fc.is_duplicate = 0 OR fc.is_duplicate IS NULL)
                LIMIT 1
            """)

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None
        return dict(row)

    def render_sample(self, font_path: str, output_path: str):
        """Render sample text as an image using the specified font.

        Creates a white background image with black text rendered using
        the target font. The image is sized to fit the rendered text
        with padding.

        Args:
            font_path: Path to the TrueType or OpenType font file.
            output_path: Path where the rendered PNG image will be saved.

        Returns:
            bool: True if rendering succeeded, False on error.
        """
        # Create image
        font_size = 48
        padding = 20

        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"  Error loading font: {e}")
            return False

        # Calculate size
        temp_img = Image.new('RGB', (1, 1), 'white')
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), self.sample_text, font=font)
        width = bbox[2] - bbox[0] + padding * 2
        height = bbox[3] - bbox[1] + padding * 2

        # Create final image
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((padding - bbox[0], padding - bbox[1]), self.sample_text,
                  font=font, fill='black')

        img.save(output_path)
        print(f"  Rendered: {width}x{height}px")
        return True

    def run_trocr_docker(self, image_path: str):
        """Execute TrOCR inference via Docker container.

        Runs the Microsoft TrOCR handwritten text recognition model inside
        a Docker container with GPU support. The model is loaded from the
        HuggingFace cache to avoid repeated downloads.

        Args:
            image_path: Path to the input image file to process.

        Returns:
            dict: OCR results containing:
                - 'text': Recognized text string
                - 'device': Compute device used ('cuda' or 'cpu')
                - 'timings': Dict with model_load, gpu_transfer, preprocess,
                    inference, and decode times in seconds
            Returns None if OCR fails or cannot parse output.
        """
        # Python script to run inside container
        ocr_script = '''
import sys
import time
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import json

image_path = sys.argv[1]

# Load model (this will be cached after first run)
t0 = time.perf_counter()
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
model_load_time = time.perf_counter() - t0

# Move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
t0 = time.perf_counter()
model = model.to(device)
gpu_transfer_time = time.perf_counter() - t0

# Load and process image
t0 = time.perf_counter()
image = Image.open(image_path).convert('RGB')
pixel_values = processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(device)
preprocess_time = time.perf_counter() - t0

# Generate
t0 = time.perf_counter()
with torch.no_grad():
    generated_ids = model.generate(pixel_values, max_length=64)
inference_time = time.perf_counter() - t0

# Decode
t0 = time.perf_counter()
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
decode_time = time.perf_counter() - t0

# Output result
result = {
    'text': generated_text,
    'device': device,
    'timings': {
        'model_load': model_load_time,
        'gpu_transfer': gpu_transfer_time,
        'preprocess': preprocess_time,
        'inference': inference_time,
        'decode': decode_time
    }
}
print(json.dumps(result))
'''

        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(ocr_script)
            script_path = f.name

        try:
            # Run Docker command
            abs_image_path = os.path.abspath(image_path)
            image_dir = os.path.dirname(abs_image_path)
            image_name = os.path.basename(abs_image_path)

            cmd = [
                'docker', 'run', '--rm', '--gpus', 'all',
                '-v', f'{image_dir}:/data',
                '-v', f'{script_path}:/app/ocr.py',
                '-v', f'{os.path.expanduser("~")}/.cache/huggingface:/root/.cache/huggingface',
                'trocr:latest',
                'python3', '/app/ocr.py', f'/data/{image_name}'
            ]

            print("  Running: docker run --gpus all trocr:latest ...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"  Docker stderr: {result.stderr}")
                return None

            # Parse JSON output
            output = result.stdout.strip()
            # Find the JSON line (last line should be JSON)
            for line in reversed(output.split('\n')):
                if line.startswith('{'):
                    return json.loads(line)

            print(f"  Could not parse output: {output[:200]}")
            return None

        finally:
            os.unlink(script_path)

    def calculate_confidence(self, expected: str, actual: str) -> float:
        """Calculate character-level match confidence between strings.

        Performs a simple character-by-character comparison between the
        expected and actual text, returning the fraction of matching
        characters.

        Args:
            expected: The ground truth text that was rendered.
            actual: The text recognized by OCR.

        Returns:
            float: Confidence score between 0.0 and 1.0, where 1.0 indicates
                perfect character-level match.
        """
        if not actual:
            return 0.0

        expected_lower = expected.lower().strip()
        actual_lower = actual.lower().strip()

        # Character-level match
        matches = sum(1 for e, a in zip(expected_lower, actual_lower) if e == a)
        max_len = max(len(expected_lower), len(actual_lower))

        if max_len == 0:
            return 0.0

        return matches / max_len

    def run(self, font_id: int = None):
        """Execute the complete OCR prefilter test on one font.

        Runs through all pipeline stages (database query, rendering, OCR,
        confidence calculation) with detailed timing output. Also provides
        estimates for processing the full font database.

        Args:
            font_id: Specific font ID to test. If None, uses the first
                font that passes pre-AI checks. Defaults to None.

        Returns:
            None. Results are printed to stdout.
        """
        print("=" * 60)
        print("OCR PREFILTER TEST - SINGLE FONT")
        print("=" * 60)

        total_start = time.perf_counter()

        # Step 1: Get font from database
        print("\n[1] Getting font from database...")
        t0 = time.perf_counter()
        font = self.get_passing_font(font_id)
        self.timings['db_query'] = time.perf_counter() - t0
        print(f"  [db_query] {self.timings['db_query']:.3f}s")

        if not font:
            print("  ERROR: No passing font found!")
            return

        print(f"  Font: {font['name']}")
        print(f"  Path: {font['file_path']}")

        # Check font exists
        if not os.path.exists(font['file_path']):
            print("  ERROR: Font file not found!")
            return

        # Step 2: Render sample text
        print("\n[2] Rendering sample text...")
        t0 = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            sample_path = f.name

        success = self.render_sample(font['file_path'], sample_path)
        self.timings['render'] = time.perf_counter() - t0
        print(f"  [render] {self.timings['render']:.3f}s")

        if not success:
            print("  ERROR: Failed to render!")
            os.unlink(sample_path)
            return

        # Step 3: Run TrOCR
        print("\n[3] Running TrOCR via Docker...")
        t0 = time.perf_counter()
        result = self.run_trocr_docker(sample_path)
        self.timings['docker_total'] = time.perf_counter() - t0
        print(f"  [docker_total] {self.timings['docker_total']:.3f}s")

        # Cleanup temp file
        os.unlink(sample_path)

        if not result:
            print("  ERROR: TrOCR failed!")
            return

        # Step 4: Process results
        print("\n[4] Processing results...")
        t0 = time.perf_counter()

        ocr_text = result['text']
        confidence = self.calculate_confidence(self.sample_text, ocr_text)
        passed = confidence >= 0.7

        self.timings['postprocess'] = time.perf_counter() - t0
        print(f"  [postprocess] {self.timings['postprocess']:.3f}s")

        # Results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Expected: '{self.sample_text}'")
        print(f"  OCR text: '{ocr_text}'")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Passed: {'YES' if passed else 'NO'}")
        print(f"  Device: {result['device']}")

        # Docker internal timings
        print("\n  Docker internal timings:")
        for name, t in result['timings'].items():
            print(f"    {name}: {t:.3f}s")

        # Total time
        total_time = time.perf_counter() - total_start
        self.timings['total'] = total_time

        # Summary
        print("\n" + "=" * 60)
        print("TIMING SUMMARY")
        print("=" * 60)
        print(f"  db_query:     {self.timings['db_query']:>7.3f}s")
        print(f"  render:       {self.timings['render']:>7.3f}s")
        print(f"  docker_total: {self.timings['docker_total']:>7.3f}s")
        print(f"    (model_load:  {result['timings']['model_load']:>7.3f}s)")
        print(f"    (gpu_transfer:{result['timings']['gpu_transfer']:>7.3f}s)")
        print(f"    (preprocess:  {result['timings']['preprocess']:>7.3f}s)")
        print(f"    (inference:   {result['timings']['inference']:>7.3f}s)")
        print(f"    (decode:      {result['timings']['decode']:>7.3f}s)")
        print(f"  postprocess:  {self.timings['postprocess']:>7.3f}s")
        print("  ─────────────────────────")
        print(f"  TOTAL:        {total_time:>7.3f}s")

        # Estimate for all fonts
        print("\n" + "=" * 60)
        print("ESTIMATE FOR 321 FONTS")
        print("=" * 60)

        # Per-font time (excluding model load - that's one-time)
        per_font_time = (
            self.timings['db_query'] +
            self.timings['render'] +
            result['timings']['preprocess'] +
            result['timings']['inference'] +
            result['timings']['decode'] +
            self.timings['postprocess']
        )

        # Docker overhead (startup/shutdown) per container
        docker_overhead = self.timings['docker_total'] - sum(result['timings'].values())

        print(f"  One-time model load: {result['timings']['model_load']:.1f}s")
        print(f"  Per-font processing: {per_font_time:.3f}s")
        print(f"  Docker overhead/run: {docker_overhead:.3f}s")
        print()

        # If we keep container running (batch mode)
        batch_total = result['timings']['model_load'] + (per_font_time * 321)
        print("  Batch mode (keep container alive):")
        print(f"    Estimated total: {batch_total/60:.1f} minutes")

        # If we restart container each time
        single_total = 321 * self.timings['docker_total']
        print("  Single mode (restart container each time):")
        print(f"    Estimated total: {single_total/60:.1f} minutes")

        print()
        print("  Recommendation: Use batch mode to avoid repeated model loading")


def main():
    """Main entry point for the OCR prefilter test script.

    Parses command-line arguments and runs the OCR test on the specified
    font or a default font from the database.
    """
    parser = argparse.ArgumentParser(description='Test OCR prefilter on single font')
    parser.add_argument('--font-id', type=int, help='Specific font ID to test')
    parser.add_argument('--db', default='fonts.db', help='Database path')
    args = parser.parse_args()

    tester = OCRTester(args.db)
    tester.run(args.font_id)


if __name__ == '__main__':
    main()
