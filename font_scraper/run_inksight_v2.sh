#!/bin/bash
# Run InkSight vectorization with DYNAMIC FONT SIZING
# Each font gets optimal size to normalize character height
#
# Usage: ./run_inksight_v2.sh
#
# Improvements over v1:
# - Dynamic font_size per font (normalizes input to InkSight)
# - Skips space character (94 chars instead of 95)
# - Better progress reporting

set -e

DB_PATH="fonts.db"
WORKSPACE="/home/server/glossy/font_scraper"
MODEL_PATH="/home/server/inksight/model"
PROGRESS_FILE="inksight_progress_v2.txt"
TARGET_HEIGHT=350  # Target character height in pixels (on 512 canvas)

echo "============================================================"
echo "INKSIGHT VECTORIZATION v2 - DYNAMIC FONT SIZING"
echo "============================================================"
echo "Started at: $(date)"
echo "Target height: ${TARGET_HEIGHT}px"
echo ""

# Write ASCII printable characters to file (excluding space = 94 chars)
python3 -c "print(''.join(chr(i) for i in range(33, 127)), end='')" > /tmp/ascii_chars.txt

# Get list of fonts to process (reset all - start fresh)
python3 - "$DB_PATH" << 'GETFONTS' > /tmp/fonts_to_process.txt
import sqlite3
import sys
import os
from PIL import ImageFont

db_path = sys.argv[1]
target_height = 350

def get_optimal_font_size(font_path, target_h=350):
    """Calculate optimal font size to hit target height."""
    try:
        base_size = 100
        font = ImageFont.truetype(font_path, base_size)
        bbox = font.getbbox('Hx')
        actual_height = bbox[3] - bbox[1]
        if actual_height == 0:
            return 200
        optimal = int(base_size * target_h / actual_height)
        return max(80, min(500, optimal))  # Clamp to reasonable range
    except:
        return 200

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get fonts that passed prefilter
cursor.execute("""
    SELECT f.id, f.name, f.file_path FROM fonts f
    JOIN font_checks fc ON f.id = fc.font_id
    WHERE fc.prefilter_passed = 1
    ORDER BY f.id
""")

for row in cursor.fetchall():
    abs_path = os.path.abspath(row['file_path'])
    optimal_size = get_optimal_font_size(abs_path, target_height)
    print(f"{row['id']}|{row['name']}|{abs_path}|{optimal_size}")

conn.close()
GETFONTS

TOTAL_FONTS=$(wc -l < /tmp/fonts_to_process.txt)
echo "Fonts to process: $TOTAL_FONTS"
echo ""

if [ "$TOTAL_FONTS" -eq 0 ]; then
    echo "No fonts to process!"
    exit 0
fi

# Create the processing script that runs inside Docker
cat > /tmp/inksight_batch_v2.py << 'DOCKERSCRIPT'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json

sys.path.insert(0, '/app/workspace')
from inksight_vectorizer import InkSightVectorizer

# Parse args
font_path = sys.argv[1]
font_size = int(sys.argv[2])

# Read chars from file (no space)
with open('/tmp/ascii_chars.txt') as f:
    chars = f.read()

# Load model
print("LOADING_MODEL", flush=True)
v = InkSightVectorizer(model_path='/app/model')
v.load_model()
print(f"MODEL_READY font_size={font_size}", flush=True)

# Process each character
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

# Clear old v2 data
echo "Clearing old stroke data for fresh run..."
python3 - "$DB_PATH" << 'CLEARDATA'
import sqlite3
import sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("DELETE FROM characters")
conn.commit()
print(f"Cleared characters table")
conn.close()
CLEARDATA

# Process each font
FONT_NUM=0
while IFS='|' read -r FONT_ID FONT_NAME FONT_PATH FONT_SIZE; do
    FONT_NUM=$((FONT_NUM + 1))

    echo ""
    echo "============================================================"
    echo "[$FONT_NUM/$TOTAL_FONTS] $FONT_NAME"
    echo "============================================================"
    echo "Font ID: $FONT_ID"
    echo "Font size: $FONT_SIZE (dynamic)"
    echo "Started: $(date '+%H:%M:%S')"

    # Copy font to temp location with safe name
    SAFE_FONT="/tmp/current_font.ttf"
    cp "$FONT_PATH" "$SAFE_FONT"

    # Run Docker
    sudo docker run --rm --gpus all \
        -v "$MODEL_PATH:/app/model:ro" \
        -v "/tmp:/tmp:rw" \
        -v "$WORKSPACE:/app/workspace:ro" \
        inksight:latest \
        python3 /tmp/inksight_batch_v2.py "/tmp/current_font.ttf" "$FONT_SIZE" \
        2>&1 | tee /tmp/inksight_output.txt

    # Check if we got results
    if grep -q "RESULTS_START" /tmp/inksight_output.txt; then
        # Extract JSON results
        sed -n '/RESULTS_START/,/RESULTS_END/p' /tmp/inksight_output.txt | \
            grep -v "RESULTS_START\|RESULTS_END" > /tmp/inksight_results.json

        # Save to database
        python3 - "$DB_PATH" "$FONT_ID" "$FONT_SIZE" << 'SAVERESULTS'
import sqlite3
import json
import sys

db_path = sys.argv[1]
font_id = int(sys.argv[2])
font_size = int(sys.argv[3])

with open('/tmp/inksight_results.json') as f:
    results = json.load(f)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

saved = 0
errors = 0
for char, data in results.items():
    if data['success']:
        cursor.execute("""
            INSERT OR REPLACE INTO characters (font_id, char, strokes_raw, point_count)
            VALUES (?, ?, ?, ?)
        """, (font_id, char, json.dumps(data['strokes']), data['total_points']))
        saved += 1
    else:
        errors += 1

conn.commit()
conn.close()

print(f"  Saved: {saved} chars, Errors: {errors}, Font size: {font_size}")
SAVERESULTS

        echo "  Completed: $(date '+%H:%M:%S')"
    else
        echo "  ERROR: No results from InkSight"
    fi

    # Log progress
    echo "$FONT_ID|$FONT_NAME|$FONT_SIZE|$(date)" >> "$PROGRESS_FILE"

done < /tmp/fonts_to_process.txt

echo ""
echo "============================================================"
echo "COMPLETED"
echo "============================================================"
echo "Finished at: $(date)"

# Summary
python3 - "$DB_PATH" << 'SUMMARY'
import sqlite3
import sys

conn = sqlite3.connect(sys.argv[1])
cursor = conn.cursor()

cursor.execute("SELECT COUNT(DISTINCT font_id) FROM characters WHERE strokes_raw IS NOT NULL")
fonts_done = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM characters WHERE strokes_raw IS NOT NULL")
chars_done = cursor.fetchone()[0]

cursor.execute("SELECT AVG(point_count), MIN(point_count), MAX(point_count) FROM characters WHERE strokes_raw IS NOT NULL")
avg, min_p, max_p = cursor.fetchone()

print(f"Fonts with strokes: {fonts_done}")
print(f"Characters processed: {chars_done}")
print(f"Points - avg: {avg:.1f}, min: {min_p}, max: {max_p}")
conn.close()
SUMMARY
