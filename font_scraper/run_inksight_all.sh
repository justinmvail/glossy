#!/bin/bash
# Run InkSight vectorization on all passing fonts
# Saves progress incrementally to database after each font
#
# Usage: ./run_inksight_all.sh
#
# Progress is saved after each font, so you can Ctrl+C and resume later

set -e

DB_PATH="fonts.db"
WORKSPACE="/home/server/glossy/font_scraper"
MODEL_PATH="/home/server/inksight/model"
PROGRESS_FILE="inksight_progress.txt"

# Write ASCII printable characters to file (avoids shell escaping issues)
python3 -c "print(''.join(chr(i) for i in range(32, 127)), end='')" > /tmp/ascii_chars.txt

echo "============================================================"
echo "INKSIGHT VECTORIZATION - FULL RUN"
echo "============================================================"
echo "Started at: $(date)"
echo ""

# Get list of fonts to process
python3 - "$DB_PATH" << 'GETFONTS' > /tmp/fonts_to_process.txt
import sqlite3
import sys
import os

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get fonts that passed prefilter and haven't been fully processed yet
cursor.execute("""
    SELECT f.id, f.name, f.file_path FROM fonts f
    JOIN font_checks fc ON f.id = fc.font_id
    WHERE fc.prefilter_passed = 1
    ORDER BY f.id
""")

for row in cursor.fetchall():
    # Check if this font has all characters processed
    cursor.execute("""
        SELECT COUNT(*) FROM characters
        WHERE font_id = ? AND strokes_raw IS NOT NULL
    """, (row['id'],))
    processed = cursor.fetchone()[0]

    if processed < 95:  # Not all chars done
        abs_path = os.path.abspath(row['file_path'])
        print(f"{row['id']}|{row['name']}|{abs_path}|{processed}")

conn.close()
GETFONTS

TOTAL_FONTS=$(wc -l < /tmp/fonts_to_process.txt)
echo "Fonts to process: $TOTAL_FONTS"
echo ""

if [ "$TOTAL_FONTS" -eq 0 ]; then
    echo "All fonts already processed!"
    exit 0
fi

# Create the processing script that runs inside Docker
cat > /tmp/inksight_batch.py << 'DOCKERSCRIPT'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json

sys.path.insert(0, '/app/workspace')
from inksight_vectorizer import InkSightVectorizer

# Parse args
font_path = sys.argv[1]

# Read chars from file
with open('/tmp/ascii_chars.txt') as f:
    chars = f.read()

# Load model
print("LOADING_MODEL", flush=True)
v = InkSightVectorizer(model_path='/app/model')
v.load_model()
print("MODEL_READY", flush=True)

# Process each character
results = {}
for i, char in enumerate(chars):
    try:
        result = v.process(font_path, char)
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

# Process each font
FONT_NUM=0
while IFS='|' read -r FONT_ID FONT_NAME FONT_PATH ALREADY_DONE; do
    FONT_NUM=$((FONT_NUM + 1))

    echo ""
    echo "============================================================"
    echo "[$FONT_NUM/$TOTAL_FONTS] $FONT_NAME"
    echo "============================================================"
    echo "Font ID: $FONT_ID"
    echo "Path: $FONT_PATH"
    echo "Already processed: $ALREADY_DONE chars"
    echo "Started: $(date '+%H:%M:%S')"

    # Copy font to temp location with safe name (handles spaces/special chars)
    SAFE_FONT="/tmp/current_font.ttf"
    cp "$FONT_PATH" "$SAFE_FONT"

    # Run Docker
    sudo docker run --rm --gpus all \
        -v "$MODEL_PATH:/app/model:ro" \
        -v "/tmp:/tmp:rw" \
        -v "$WORKSPACE:/app/workspace:ro" \
        inksight:latest \
        python3 /tmp/inksight_batch.py "/tmp/current_font.ttf" \
        2>&1 | tee /tmp/inksight_output.txt

    # Check if we got results
    if grep -q "RESULTS_START" /tmp/inksight_output.txt; then
        # Extract JSON results
        sed -n '/RESULTS_START/,/RESULTS_END/p' /tmp/inksight_output.txt | \
            grep -v "RESULTS_START\|RESULTS_END" > /tmp/inksight_results.json

        # Save to database
        python3 - "$DB_PATH" "$FONT_ID" << 'SAVERESULTS'
import sqlite3
import json
import sys

db_path = sys.argv[1]
font_id = int(sys.argv[2])

with open('/tmp/inksight_results.json') as f:
    results = json.load(f)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

saved = 0
errors = 0
for char, data in results.items():
    if data['success']:
        # Insert or update character
        cursor.execute("""
            INSERT OR REPLACE INTO characters (font_id, char, strokes_raw, point_count)
            VALUES (?, ?, ?, ?)
        """, (font_id, char, json.dumps(data['strokes']), data['total_points']))
        saved += 1
    else:
        errors += 1

conn.commit()
conn.close()

print(f"  Saved: {saved} chars, Errors: {errors}")
SAVERESULTS

        echo "  Completed: $(date '+%H:%M:%S')"
    else
        echo "  ERROR: No results from InkSight"
    fi

    # Log progress
    echo "$FONT_ID|$FONT_NAME|$(date)" >> "$PROGRESS_FILE"

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

print(f"Fonts with strokes: {fonts_done}")
print(f"Characters processed: {chars_done}")
conn.close()
SUMMARY
