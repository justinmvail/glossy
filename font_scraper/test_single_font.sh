#!/bin/bash
# Test InkSight on a single font with dynamic sizing
# Usage: ./test_single_font.sh <font_id>

set -e

FONT_ID=${1:-503}  # Default to Bubbles-Regular
DB_PATH="fonts.db"
WORKSPACE="/home/server/glossy/font_scraper"
MODEL_PATH="/home/server/inksight/model"
TARGET_HEIGHT=350

# Get font info and calculate optimal size
read FONT_NAME FONT_PATH FONT_SIZE < <(python3 - "$DB_PATH" "$FONT_ID" "$TARGET_HEIGHT" << 'GETFONT'
import sqlite3
import sys
import os
from PIL import ImageFont

db_path = sys.argv[1]
font_id = int(sys.argv[2])
target_height = int(sys.argv[3])

def get_optimal_font_size(font_path, target_h=350):
    try:
        base_size = 100
        font = ImageFont.truetype(font_path, base_size)
        bbox = font.getbbox('Hx')
        actual_height = bbox[3] - bbox[1]
        if actual_height == 0:
            return 200
        optimal = int(base_size * target_h / actual_height)
        return max(80, min(500, optimal))
    except:
        return 200

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name, file_path FROM fonts WHERE id = ?", (font_id,))
row = cursor.fetchone()
if row:
    abs_path = os.path.abspath(row[1])
    optimal_size = get_optimal_font_size(abs_path, target_height)
    print(f"{row[0]} {abs_path} {optimal_size}")
conn.close()
GETFONT
)

echo "============================================================"
echo "TEST: $FONT_NAME"
echo "============================================================"
echo "Font ID: $FONT_ID"
echo "Font path: $FONT_PATH"
echo "Dynamic font size: $FONT_SIZE"
echo ""

# Write ASCII chars (no space)
python3 -c "print(''.join(chr(i) for i in range(33, 127)), end='')" > /tmp/ascii_chars.txt

# Create batch script
cat > /tmp/inksight_test.py << 'DOCKERSCRIPT'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json

sys.path.insert(0, '/app/workspace')
from inksight_vectorizer import InkSightVectorizer

font_path = sys.argv[1]
font_size = int(sys.argv[2])

with open('/tmp/ascii_chars.txt') as f:
    chars = f.read()

print("LOADING_MODEL", flush=True)
v = InkSightVectorizer(model_path='/app/model')
v.load_model()
print(f"MODEL_READY font_size={font_size}", flush=True)

results = {}
for i, char in enumerate(chars):
    try:
        result = v.process(font_path, char, font_size=font_size)
        results[char] = {
            'strokes': [s.points.tolist() for s in result.strokes],
            'num_strokes': len(result.strokes),
            'total_points': sum(len(s.points) for s in result.strokes),
            'success': True
        }
    except Exception as e:
        results[char] = {'success': False, 'error': str(e)}

    if (i + 1) % 10 == 0:
        print(f"PROGRESS {i + 1}/{len(chars)}", flush=True)

print("RESULTS_START", flush=True)
print(json.dumps(results), flush=True)
print("RESULTS_END", flush=True)
DOCKERSCRIPT

# Copy font to safe location
cp "$FONT_PATH" /tmp/current_font.ttf

echo "Running InkSight..."
sudo docker run --rm --gpus all \
    -v "$MODEL_PATH:/app/model:ro" \
    -v "/tmp:/tmp:rw" \
    -v "$WORKSPACE:/app/workspace:ro" \
    inksight:latest \
    python3 /tmp/inksight_test.py "/tmp/current_font.ttf" "$FONT_SIZE" \
    2>&1 | tee /tmp/inksight_test_output.txt

# Extract and analyze results
if grep -q "RESULTS_START" /tmp/inksight_test_output.txt; then
    sed -n '/RESULTS_START/,/RESULTS_END/p' /tmp/inksight_test_output.txt | \
        grep -v "RESULTS_START\|RESULTS_END" > /tmp/inksight_test_results.json

    echo ""
    echo "============================================================"
    echo "RESULTS ANALYSIS"
    echo "============================================================"
    python3 - << 'ANALYZE'
import json

with open('/tmp/inksight_test_results.json') as f:
    results = json.load(f)

success = sum(1 for r in results.values() if r['success'])
failed = len(results) - success

points = [r['total_points'] for r in results.values() if r['success']]
low_pts = sum(1 for p in points if p < 10)

print(f"Success: {success}/{len(results)}")
print(f"Failed: {failed}")
print(f"Avg points: {sum(points)/len(points):.1f}")
print(f"Min points: {min(points)}")
print(f"Max points: {max(points)}")
print(f"Chars with <10 pts: {low_pts} ({100*low_pts/len(points):.1f}%)")

# Show worst characters
print("\nLowest point counts:")
sorted_chars = sorted([(c, r['total_points']) for c, r in results.items() if r['success']], key=lambda x: x[1])
for char, pts in sorted_chars[:10]:
    print(f"  '{char}': {pts} pts")

print("\nHighest point counts:")
for char, pts in sorted_chars[-5:]:
    print(f"  '{char}': {pts} pts")
ANALYZE
else
    echo "ERROR: No results from InkSight"
fi
