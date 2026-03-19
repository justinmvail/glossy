"""Render sample predictions as PNG images for visual inspection."""

import json
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw

from model import StrokePredictor, char_to_index, CANVAS_SIZE, CHARS_LIST
from predict import render_glyph_mask


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else '/app/checkpoints/best_model.pt'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/app/checkpoints/samples'
    db_path = sys.argv[3] if len(sys.argv) > 3 else '/data/fonts.db'

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = StrokePredictor(feature_dim=256)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('loss', '?'):.4f})")

    # Get a few font paths
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT f.file_path, f.name FROM fonts f
        LEFT JOIN font_checks fc ON f.id = fc.font_id
        LEFT JOIN font_removals fr ON f.id = fr.font_id
        WHERE fr.font_id IS NULL
        AND (fc.prefilter_passed = 1 OR fc.prefilter_passed IS NULL)
        AND (fc.is_cursive = 0 OR fc.is_cursive IS NULL)
        LIMIT 5
    """).fetchall()
    conn.close()

    chars = ['A', 'B', 'g', 'R', '5']

    for font_path_raw, font_name in rows:
        fp = font_path_raw if os.path.isabs(font_path_raw) else os.path.join('/', font_path_raw)
        if not os.path.exists(fp):
            continue

        for char in chars:
            mask = render_glyph_mask(fp, char, CANVAS_SIZE)
            if mask is None:
                continue

            img_arr = 1.0 - mask.astype(np.float32)
            img_tensor = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0).to(device)
            char_idx = torch.tensor([char_to_index(char)], dtype=torch.long, device=device)

            with torch.no_grad():
                strokes, avg_width = model.predict_strokes(
                    img_tensor, char_idx, CANVAS_SIZE, existence_threshold=0.3,
                )

            # Draw: glyph in light gray, strokes in red
            img = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            # Draw glyph mask
            for y in range(CANVAS_SIZE):
                for x in range(CANVAS_SIZE):
                    if mask[y, x]:
                        img.putpixel((x, y), (220, 220, 220))

            # Draw strokes
            colors = [(255, 0, 0), (0, 150, 0), (0, 0, 255), (255, 128, 0),
                      (128, 0, 255), (0, 200, 200), (200, 0, 128), (128, 128, 0)]
            for si, stroke in enumerate(strokes):
                color = colors[si % len(colors)]
                for i in range(len(stroke) - 1):
                    x1, y1 = stroke[i]
                    x2, y2 = stroke[i + 1]
                    draw.line([(x1, y1), (x2, y2)], fill=color, width=max(1, int(avg_width)))

            safe_name = font_name.replace(' ', '_').replace('/', '_')[:20]
            filename = f"{safe_name}_{char}.png"
            img.save(os.path.join(output_dir, filename))
            print(f"  {filename}: {len(strokes)} strokes")

    print(f"\nSaved samples to {output_dir}")


if __name__ == '__main__':
    main()
