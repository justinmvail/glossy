#!/bin/bash
# OCR Prefilter - Shell version
# Usage: ./run_ocr_prefilter.sh

set -e

DB_PATH="${1:-fonts.db}"
THRESHOLD="${2:-0.7}"
TEMP_DIR=$(mktemp -d -t ocr_prefilter_XXXXXX)
SAMPLE_TEXT="Hello World 123"

echo "============================================================"
echo "OCR PREFILTER - SHELL VERSION"
echo "============================================================"
echo "Database: $DB_PATH"
echo "Threshold: $THRESHOLD"
echo "Temp dir: $TEMP_DIR"

# Step 1: Get passing fonts and render samples
echo ""
echo "[1] Getting fonts and rendering samples..."

TEMP_DIR="$TEMP_DIR" python3 - "$DB_PATH" << 'RENDER_SCRIPT'
import sqlite3
import sys
import os
from PIL import Image, ImageDraw, ImageFont

db_path = sys.argv[1] if len(sys.argv) > 1 else 'fonts.db'
temp_dir = os.environ.get('TEMP_DIR', '/tmp')
sample_text = "Hello World 123"

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

print(f"Found {len(fonts)} fonts to process")

# Write font list
with open(os.path.join(temp_dir, 'fonts.txt'), 'w') as f:
    for font in fonts:
        f.write(f"{font['id']}|{font['name']}|{font['file_path']}\n")

# Render samples
rendered = 0
for font in fonts:
    try:
        pil_font = ImageFont.truetype(font['file_path'], 48)
        temp_img = Image.new('RGB', (1, 1), 'white')
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), sample_text, font=pil_font)
        width = bbox[2] - bbox[0] + 40
        height = bbox[3] - bbox[1] + 40

        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((20 - bbox[0], 20 - bbox[1]), sample_text, font=pil_font, fill='black')
        img.save(os.path.join(temp_dir, f"{font['id']}.png"))
        rendered += 1
    except Exception as e:
        print(f"  Error rendering {font['name']}: {e}", file=sys.stderr)

print(f"Rendered {rendered} images")
RENDER_SCRIPT

# Check if we have fonts
if [ ! -f "$TEMP_DIR/fonts.txt" ]; then
    echo "ERROR: No fonts file created"
    exit 1
fi

FONT_COUNT=$(wc -l < "$TEMP_DIR/fonts.txt")
echo "  Font list: $TEMP_DIR/fonts.txt ($FONT_COUNT fonts)"

# Step 2: Create OCR script for Docker
echo ""
echo "[2] Running TrOCR via Docker..."

cat > "$TEMP_DIR/batch_ocr.py" << 'OCR_SCRIPT'
import sys
import json
import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

image_dir = sys.argv[1]

# Get list of images
images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
print(f"Found {len(images)} images", flush=True)

# Load model
print("Loading model...", flush=True)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"Using device: {device}", flush=True)

# Process images
results = {}
for i, filename in enumerate(images):
    try:
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        pixel_values = processor(images=image, return_tensors='pt').pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=64)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results[filename] = text
    except Exception as e:
        results[filename] = f"ERROR: {e}"

    if (i + 1) % 20 == 0:
        print(f"Progress: {i + 1}/{len(images)}", flush=True)

# Write results
with open(os.path.join(image_dir, 'results.json'), 'w') as f:
    json.dump(results, f)

print(f"Done! Processed {len(results)} images", flush=True)
OCR_SCRIPT

# Run Docker
echo "  Starting Docker container..."
sudo docker run --rm --gpus all \
    -v "$TEMP_DIR:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    trocr:latest \
    python3 /data/batch_ocr.py /data

# Check results
if [ ! -f "$TEMP_DIR/results.json" ]; then
    echo "ERROR: No results file created"
    exit 1
fi

echo "  Results saved to $TEMP_DIR/results.json"

# Step 3: Update database
echo ""
echo "[3] Updating database..."

TEMP_DIR="$TEMP_DIR" THRESHOLD="$THRESHOLD" python3 - "$DB_PATH" << 'UPDATE_SCRIPT'
import sqlite3
import json
import sys
import os

db_path = sys.argv[1] if len(sys.argv) > 1 else 'fonts.db'
temp_dir = os.environ.get('TEMP_DIR', '/tmp')
threshold = float(os.environ.get('THRESHOLD', '0.7'))
expected = "Hello World 123"

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]

def confidence(expected, actual):
    if not actual or actual.startswith('ERROR'):
        return 0.0
    e, a = expected.lower().strip(), actual.lower().strip()
    if not e:
        return 0.0
    dist = levenshtein(e, a)
    return max(0.0, 1.0 - dist / max(len(e), len(a)))

# Load results
with open(os.path.join(temp_dir, 'results.json')) as f:
    results = json.load(f)

# Load font list
fonts = {}
with open(os.path.join(temp_dir, 'fonts.txt')) as f:
    for line in f:
        parts = line.strip().split('|')
        fonts[f"{parts[0]}.png"] = {'id': int(parts[0]), 'name': parts[1]}

# Update database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

passed = failed = errors = 0

for filename, ocr_text in results.items():
    if filename not in fonts:
        continue

    font = fonts[filename]
    font_id = font['id']

    if ocr_text.startswith('ERROR'):
        errors += 1
        continue

    conf = confidence(expected, ocr_text)
    prefilter_passed = conf >= threshold

    if prefilter_passed:
        passed += 1
    else:
        failed += 1

    # Update font_checks
    cursor.execute("INSERT OR IGNORE INTO font_checks (font_id) VALUES (?)", (font_id,))
    cursor.execute("""
        UPDATE font_checks
        SET prefilter_passed = ?, prefilter_confidence = ?, prefilter_ocr_text = ?
        WHERE font_id = ?
    """, (prefilter_passed, conf, ocr_text, font_id))

    # Record removal if failed
    if not prefilter_passed:
        cursor.execute("SELECT id FROM removal_reasons WHERE code = 'ocr_prefilter'")
        reason_id = cursor.fetchone()[0]
        cursor.execute("""
            INSERT INTO font_removals (font_id, reason_id, details)
            VALUES (?, ?, ?)
        """, (font_id, reason_id, f"conf={conf:.1%}, got='{ocr_text}'"))

conn.commit()
conn.close()

print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Errors: {errors}")

# Show samples
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("\nTop 5 (highest confidence):")
cursor.execute("""
    SELECT f.name, fc.prefilter_confidence, fc.prefilter_ocr_text
    FROM fonts f JOIN font_checks fc ON f.id = fc.font_id
    WHERE fc.prefilter_confidence IS NOT NULL
    ORDER BY fc.prefilter_confidence DESC LIMIT 5
""")
for row in cursor.fetchall():
    status = "PASS" if row['prefilter_confidence'] >= threshold else "FAIL"
    print(f"  {row['prefilter_confidence']:.1%} [{status}] {row['name'][:30]:30} -> '{row['prefilter_ocr_text']}'")

print("\nBottom 5 (lowest confidence):")
cursor.execute("""
    SELECT f.name, fc.prefilter_confidence, fc.prefilter_ocr_text
    FROM fonts f JOIN font_checks fc ON f.id = fc.font_id
    WHERE fc.prefilter_confidence IS NOT NULL
    ORDER BY fc.prefilter_confidence ASC LIMIT 5
""")
for row in cursor.fetchall():
    status = "PASS" if row['prefilter_confidence'] >= threshold else "FAIL"
    print(f"  {row['prefilter_confidence']:.1%} [{status}] {row['name'][:30]:30} -> '{row['prefilter_ocr_text']}'")

conn.close()
UPDATE_SCRIPT

# Cleanup
echo ""
echo "[4] Cleanup..."
rm -rf "$TEMP_DIR"
echo "  Removed temp directory"

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
