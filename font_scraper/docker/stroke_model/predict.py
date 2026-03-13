"""Inference script for stroke prediction model.

Runs inside the Docker container. Takes a JSON config via stdin or file,
loads the trained model, predicts strokes for a given font+character,
and outputs JSON to stdout.

Usage:
    python3 predict.py /app/input.json

Input JSON:
    {
        "font_path": "/fonts/MyFont.ttf",
        "char": "A",
        "model_path": "/app/checkpoints/best_model.pt",
        "canvas_size": 224,
        "existence_threshold": 0.5
    }

Output JSON (stdout, last line):
    {
        "strokes": [[[x1, y1], [x2, y2], ...], ...],
        "score": 0.85,
        "n_strokes": 3,
        "elapsed": 0.12
    }
"""

import json
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from model import StrokePredictor, char_to_index, CANVAS_SIZE


def render_glyph_mask(font_path: str, char: str,
                      canvas_size: int = CANVAS_SIZE) -> np.ndarray | None:
    """Render a character as a binary mask. Same as in dataset.py."""
    img = Image.new('L', (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)

    for font_size in range(200, 20, -5):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            continue
        bbox = font.getbbox(char)
        if bbox is None:
            return None
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        max_dim = canvas_size * 0.9
        if tw <= max_dim and th <= max_dim:
            x = (canvas_size - tw) / 2 - bbox[0]
            y = (canvas_size - th) / 2 - bbox[1]
            draw.text((x, y), char, fill=0, font=font)
            arr = np.array(img)
            return arr < 128

    return None


def predict(config: dict) -> dict:
    """Run inference on a single font + character.

    Args:
        config: Dict with font_path, char, model_path, etc.

    Returns:
        Dict with strokes, score, n_strokes, elapsed.
    """
    t0 = time.time()

    font_path = config['font_path']
    char = config['char']
    model_path = config.get('model_path', '/app/checkpoints/best_model.pt')
    canvas_size = config.get('canvas_size', CANVAS_SIZE)
    existence_threshold = config.get('existence_threshold', 0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = StrokePredictor(feature_dim=config.get('feature_dim', 256))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Render glyph image
    mask = render_glyph_mask(font_path, char, canvas_size)
    if mask is None:
        return {'error': f'Cannot render character "{char}" from {font_path}'}

    # Prepare input tensor
    img_arr = 1.0 - mask.astype(np.float32)  # 1=white, 0=glyph
    img_tensor = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0).to(device)
    char_idx = torch.tensor([char_to_index(char)], dtype=torch.long, device=device)

    # Predict
    strokes, avg_width = model.predict_strokes(
        img_tensor, char_idx, canvas_size, existence_threshold,
    )

    # Compute score if we have strokes
    score = 0.0
    if strokes:
        try:
            import pydiffvg
            pydiffvg.set_use_gpu(device.type == 'cuda')
            from render_utils import render_strokes as render_fn, compute_score

            stroke_pts = [
                torch.tensor(s, dtype=torch.float32, device=device) for s in strokes
            ]
            stroke_wds = [
                torch.tensor(avg_width, dtype=torch.float32, device=device)
                for _ in strokes
            ]
            rendered = render_fn(stroke_pts, stroke_wds, canvas_size, device)
            glyph_mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)
            score = compute_score(rendered, glyph_mask_t)
        except Exception as e:
            # Score computation is optional, don't fail inference
            score = -1.0

    elapsed = time.time() - t0

    return {
        'strokes': strokes,
        'score': round(score, 4),
        'n_strokes': len(strokes),
        'elapsed': round(elapsed, 3),
    }


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else '/app/input.json'
    with open(config_path) as f:
        config = json.load(f)

    result = predict(config)
    print(json.dumps(result), flush=True)
