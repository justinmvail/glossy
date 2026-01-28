#!/usr/bin/env python3
"""
Test OCR prefilter on a single font with detailed timing.

Usage:
    python test_ocr_single.py [--font-id ID] [--db fonts.db]
"""

import argparse
import subprocess
import tempfile
import time
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def timed(name):
    """Decorator to time a function."""
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
    def __init__(self, db_path: str = 'fonts.db'):
        self.db_path = db_path
        self.sample_text = "Hello World 123"
        self.timings = {}

    def get_passing_font(self, font_id: int = None):
        """Get a font that passed all pre-AI checks."""
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
        """Render sample text with the font."""
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
        """Run TrOCR via Docker container."""
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

            print(f"  Running: docker run --gpus all trocr:latest ...")
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
        """Calculate simple character-level confidence."""
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
        """Run the full OCR test on one font."""
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
            print(f"  ERROR: Font file not found!")
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
        print(f"  ─────────────────────────")
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
        print(f"  Batch mode (keep container alive):")
        print(f"    Estimated total: {batch_total/60:.1f} minutes")

        # If we restart container each time
        single_total = 321 * self.timings['docker_total']
        print(f"  Single mode (restart container each time):")
        print(f"    Estimated total: {single_total/60:.1f} minutes")

        print()
        print("  Recommendation: Use batch mode to avoid repeated model loading")


def main():
    parser = argparse.ArgumentParser(description='Test OCR prefilter on single font')
    parser.add_argument('--font-id', type=int, help='Specific font ID to test')
    parser.add_argument('--db', default='fonts.db', help='Database path')
    args = parser.parse_args()

    tester = OCRTester(args.db)
    tester.run(args.font_id)


if __name__ == '__main__':
    main()
