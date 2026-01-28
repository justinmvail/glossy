#!/usr/bin/env python3
"""Time InkSight on a single character to estimate full run."""

import sqlite3
import time
import subprocess
import json
import os

DB_PATH = 'fonts.db'

# Get one passing font
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("""
    SELECT f.id, f.name, f.file_path FROM fonts f
    JOIN font_checks fc ON f.id = fc.font_id
    WHERE fc.prefilter_passed = 1
""")
# Find a font without special characters in path
font = None
for row in cursor.fetchall():
    path = row['file_path']
    if "'" not in path and " " not in path:
        font = dict(row)
        break
if not font:
    print("No suitable font found!")
    exit(1)
conn.close()

# Convert to absolute path
font['file_path'] = os.path.abspath(font['file_path'])
print(f"Testing with: {font['name']}")
print(f"Path: {font['file_path']}")
print()

# Test single character
test_char = "A"

script = f'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import time

sys.path.insert(0, '/app/workspace')
from inksight_vectorizer import InkSightVectorizer

# Time model loading
t0 = time.perf_counter()
v = InkSightVectorizer(model_path='/app/model')
v.load_model()
model_load_time = time.perf_counter() - t0
print(f"MODEL_LOAD {{model_load_time:.2f}}", flush=True)

# Time single character
t0 = time.perf_counter()
result = v.process('/fonts/{os.path.basename(font["file_path"])}', '{test_char}')
char_time = time.perf_counter() - t0
print(f"CHAR_TIME {{char_time:.3f}}", flush=True)

# Time 10 characters
chars = "ABCDEFGHIJ"
t0 = time.perf_counter()
for c in chars:
    result = v.process('/fonts/{os.path.basename(font["file_path"])}', c)
batch_time = time.perf_counter() - t0
print(f"BATCH_10 {{batch_time:.3f}}", flush=True)

print("DONE", flush=True)
'''

font_dir = os.path.abspath(os.path.dirname(font["file_path"]))
workspace_dir = os.path.abspath(os.getcwd())

cmd = [
    'sudo', 'docker', 'run', '--rm', '--gpus', 'all',
    '-v', '/home/server/inksight/model:/app/model:ro',
    '-v', f'{font_dir}:/fonts:ro',
    '-v', f'{workspace_dir}:/app/workspace:ro',
    'inksight:latest', 'python3', '-c', script
]

print("Running InkSight timing test...")
print()

start = time.perf_counter()
result = subprocess.run(cmd, capture_output=True, text=True)
total = time.perf_counter() - start

if result.returncode != 0:
    print(f"Error: {result.stderr}")
else:
    print(result.stdout)

    # Parse timings
    lines = result.stdout.strip().split('\n')
    model_load = None
    char_time = None
    batch_time = None

    for line in lines:
        if line.startswith('MODEL_LOAD'):
            model_load = float(line.split()[1])
        elif line.startswith('CHAR_TIME'):
            char_time = float(line.split()[1])
        elif line.startswith('BATCH_10'):
            batch_time = float(line.split()[1])

    print()
    print("=" * 50)
    print("TIMING RESULTS")
    print("=" * 50)
    print(f"  Model load:        {model_load:.2f}s")
    print(f"  Single char:       {char_time:.3f}s")
    print(f"  10 chars:          {batch_time:.3f}s")
    print(f"  Per char (batch):  {batch_time/10:.3f}s")
    print(f"  Docker overhead:   {total - model_load - char_time - batch_time:.2f}s")
    print()
    print("=" * 50)
    print("ESTIMATES FOR FULL RUN")
    print("=" * 50)

    fonts = 254
    chars = 95
    total_chars = fonts * chars

    per_char = batch_time / 10

    # If we process all chars per font in one container
    per_font = model_load + (chars * per_char)
    total_single_container_per_font = fonts * per_font

    # If we keep container running (batch all fonts)
    total_batch = model_load + (total_chars * per_char)

    print(f"  Total characters: {total_chars:,}")
    print()
    print(f"  Single container per font:")
    print(f"    {total_single_container_per_font/60:.1f} minutes ({total_single_container_per_font/3600:.1f} hours)")
    print()
    print(f"  Batch mode (one container):")
    print(f"    {total_batch/60:.1f} minutes ({total_batch/3600:.1f} hours)")
