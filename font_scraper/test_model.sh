#!/bin/bash
# test_model.sh — Visual test of stroke model predictions.
#
# Picks sample fonts, runs inference on a set of characters, and generates
# a grid of glyph images with predicted strokes overlaid. Opens the result
# or saves to a file.
#
# Usage:
#   ./test_model.sh                          # Test with defaults (best_model.pt)
#   ./test_model.sh --checkpoint epoch10     # Use checkpoint_epoch10.pt
#   ./test_model.sh --chars "ABCabc123"      # Custom character set
#   ./test_model.sh --fonts 5               # Test on 5 random fonts
#   ./test_model.sh --threshold 0.3         # Lower existence threshold

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_DIR="$SCRIPT_DIR/docker/stroke_model/checkpoints"
OUTPUT_DIR="$SCRIPT_DIR/docker/stroke_model/test_results"

# Defaults
CHECKPOINT="best_model"
CHARS="ABCMRabehmr0235"
NUM_FONTS=3
THRESHOLD=0.3

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --chars) CHARS="$2"; shift 2 ;;
        --fonts) NUM_FONTS="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

CKPT_FILE="$CKPT_DIR/${CHECKPOINT}.pt"
if [ ! -f "$CKPT_FILE" ]; then
    echo "Checkpoint not found: $CKPT_FILE"
    echo ""
    echo "Available checkpoints:"
    ls -lh "$CKPT_DIR"/*.pt 2>/dev/null || echo "  (none)"
    exit 1
fi

CKPT_SIZE=$(du -h "$CKPT_FILE" | cut -f1)
echo "Using checkpoint: $CKPT_FILE ($CKPT_SIZE)"
echo "Characters: $CHARS"
echo "Fonts: $NUM_FONTS random"
echo "Threshold: $THRESHOLD"
echo ""

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/test_${CHECKPOINT}_${TIMESTAMP}.png"

docker run --rm --gpus all \
    -v "$SCRIPT_DIR/fonts:/fonts:ro" \
    -v "$SCRIPT_DIR/fonts.db:/data/fonts.db:ro" \
    -v "$CKPT_DIR:/app/checkpoints:ro" \
    -v "$OUTPUT_DIR:/app/output" \
    stroke-model:latest \
    python3 -c "
import json
import os
import random
import sqlite3
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from model import StrokePredictor, char_to_index, CANVAS_SIZE

# Config
CHARS = '$CHARS'
NUM_FONTS = $NUM_FONTS
THRESHOLD = $THRESHOLD
CKPT = '/app/checkpoints/${CHECKPOINT}.pt'
OUTPUT = '/app/output/test_${CHECKPOINT}_${TIMESTAMP}.png'

# Load fonts from DB
conn = sqlite3.connect('/data/fonts.db')
conn.row_factory = sqlite3.Row
rows = conn.execute('''
    SELECT f.id, f.name, f.file_path FROM fonts f
    LEFT JOIN font_removals fr ON f.id = fr.font_id
    WHERE fr.font_id IS NULL
    ORDER BY RANDOM()
    LIMIT ?
''', (NUM_FONTS * 3,)).fetchall()  # fetch extra in case some fail
conn.close()

# Filter to fonts whose files exist
fonts = []
for r in rows:
    fp = r['file_path']
    if not os.path.isabs(fp):
        fp = os.path.join('/', fp)
    if os.path.exists(fp):
        fonts.append({'id': r['id'], 'name': r['name'], 'path': fp})
    if len(fonts) >= NUM_FONTS:
        break

if not fonts:
    print('ERROR: No valid fonts found')
    sys.exit(1)

print(f'Testing {len(fonts)} fonts x {len(CHARS)} chars = {len(fonts) * len(CHARS)} predictions')

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StrokePredictor(feature_dim=256)
checkpoint = torch.load(CKPT, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

epoch = checkpoint.get('epoch', '?')
loss = checkpoint.get('loss', '?')
if isinstance(loss, float):
    loss = f'{loss:.4f}'
print(f'Model from epoch {epoch}, loss {loss}')

# Render glyph
def render_glyph(font_path, char, canvas_size=CANVAS_SIZE):
    img = Image.new('L', (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    for font_size in range(200, 20, -5):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            continue
        bbox = font.getbbox(char)
        if bbox is None:
            return None, None
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= canvas_size * 0.9 and th <= canvas_size * 0.9:
            x = (canvas_size - tw) / 2 - bbox[0]
            y = (canvas_size - th) / 2 - bbox[1]
            draw.text((x, y), char, fill=0, font=font)
            arr = np.array(img)
            return arr < 128, img
    return None, None

# Predict strokes
STROKE_COLORS = [
    (66, 133, 244),   # blue
    (234, 67, 53),    # red
    (52, 168, 83),    # green
    (251, 188, 4),    # yellow
    (171, 71, 188),   # purple
    (0, 172, 193),    # teal
    (255, 112, 67),   # orange
    (158, 158, 158),  # gray
]

# Build grid: rows=fonts, cols=chars
cell_size = CANVAS_SIZE
padding = 4
header_h = 24
n_cols = len(CHARS)
n_rows = len(fonts)

grid_w = padding + n_cols * (cell_size + padding)
grid_h = padding + header_h + n_rows * (cell_size + padding + header_h)

grid = Image.new('RGB', (grid_w, grid_h), (30, 30, 50))
grid_draw = ImageDraw.Draw(grid)

try:
    label_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
except:
    label_font = ImageFont.load_default()

# Column headers (characters)
for ci, char in enumerate(CHARS):
    x = padding + ci * (cell_size + padding) + cell_size // 2
    grid_draw.text((x - 4, 6), char, fill=(200, 200, 200), font=label_font)

total_strokes = 0
total_predictions = 0
t0 = time.time()

for fi, font_info in enumerate(fonts):
    y_base = padding + header_h + fi * (cell_size + padding + header_h)

    # Font name label
    name = font_info['name'][:40]
    grid_draw.text((padding + 2, y_base), name, fill=(150, 150, 180), font=label_font)

    for ci, char in enumerate(CHARS):
        x_base = padding + ci * (cell_size + padding)
        y_cell = y_base + header_h

        mask, glyph_img = render_glyph(font_info['path'], char)
        if mask is None:
            # Draw X for failed render
            grid_draw.text(
                (x_base + cell_size // 2 - 4, y_cell + cell_size // 2 - 4),
                '?', fill=(80, 80, 80), font=label_font,
            )
            continue

        # Create cell image: dark bg with glyph visible
        cell = Image.new('RGB', (cell_size, cell_size), (26, 26, 46))
        cell_arr = np.array(cell)
        cell_arr[mask] = [60, 60, 80]
        cell = Image.fromarray(cell_arr)
        cell_draw = ImageDraw.Draw(cell)

        # Run inference
        img_arr = 1.0 - mask.astype(np.float32)
        img_tensor = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0).to(device)
        char_idx = torch.tensor([char_to_index(char)], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(img_tensor, char_idx)

        existence = out['existence'][0]
        points = out['points'][0]
        point_count_logits = out['point_count_logits'][0]

        # Draw strokes
        n_drawn = 0
        for si in range(8):
            if existence[si].item() < THRESHOLD:
                continue
            n_points = torch.argmax(point_count_logits[si]).item() + 1
            n_points = max(2, min(n_points, 40))
            pts = points[si, :n_points] * cell_size
            stroke_pts = [(float(p[0]), float(p[1])) for p in pts]
            color = STROKE_COLORS[si % len(STROKE_COLORS)]
            if len(stroke_pts) >= 2:
                cell_draw.line(stroke_pts, fill=color, width=3)
                n_drawn += 1

        total_strokes += n_drawn
        total_predictions += 1

        # Paste cell into grid
        grid.paste(cell, (x_base, y_cell))

elapsed = time.time() - t0

# Add summary at bottom
summary = f'Epoch {epoch} | {total_predictions} chars | {total_strokes} strokes | {elapsed:.1f}s | threshold={THRESHOLD}'
grid_draw.text((padding, grid_h - 18), summary, fill=(120, 120, 140), font=label_font)

grid.save(OUTPUT)
print(f'Saved: {OUTPUT}')
print(f'Total: {total_predictions} predictions, {total_strokes} strokes, {elapsed:.1f}s')
print(f'Avg strokes/char: {total_strokes / max(total_predictions, 1):.1f}')
" 2>&1 | grep -v "^=" | grep -v NVIDIA | grep -v Container | grep -v license | grep -v governed | grep -v copy | grep -v pulling | grep -v "^$" | grep -v "CUDA Version"

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "Result saved: $OUTPUT_FILE"
    echo ""
    # Try to display info
    python3 -c "
from PIL import Image
img = Image.open('$OUTPUT_FILE')
print(f'Image: {img.size[0]}x{img.size[1]}px')
"
fi
