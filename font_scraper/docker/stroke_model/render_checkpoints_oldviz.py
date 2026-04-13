#!/usr/bin/env python3
"""Render multiple checkpoints in run 19's old-viz style (solid fill, no centerline).

Used to answer: which of these saved checkpoints actually has the best stroke
topology, now that we know total-loss-best != visually-best?

Saves into a new history dir so compare_runs.py picks it up automatically:
    history/<date>_oldviz_<commit>/tracking/epoch_NNN/<label>.png
"""

import os
import sys
import argparse
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from model import StrokePredictor, CANVAS_SIZE, char_to_index

FONTS_LOCAL = [
    ("/home/server/glossy/font_scraper/fonts/dafont/BrownFox.otf", "Brown_Fox"),
    ("/home/server/glossy/font_scraper/fonts/dafont/CoffeeMilkshake.otf", "Coffee_Milkshake"),
    ("/home/server/glossy/font_scraper/fonts/dafont/Buka Bird.ttf", "Buka_Bird"),
]
CHARS = ['A', 'R', 'g', '8']

COLORS = [
    (255, 0, 0), (0, 150, 0), (0, 0, 255), (255, 128, 0),
    (128, 0, 255), (0, 200, 200), (200, 0, 128), (128, 128, 0),
]


def build_samples(device):
    samples = []
    for fp, font_name in FONTS_LOCAL:
        if not os.path.exists(fp):
            print(f"  SKIP missing: {fp}")
            continue
        for char in CHARS:
            img = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 255)
            draw = ImageDraw.Draw(img)
            rendered = False
            for font_size in range(200, 20, -5):
                try:
                    font = ImageFont.truetype(fp, font_size)
                except Exception:
                    continue
                bbox = font.getbbox(char)
                if bbox is None:
                    break
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if tw <= CANVAS_SIZE * 0.9 and th <= CANVAS_SIZE * 0.9:
                    x = (CANVAS_SIZE - tw) / 2 - bbox[0]
                    y = (CANVAS_SIZE - th) / 2 - bbox[1]
                    draw.text((x, y), char, fill=0, font=font)
                    rendered = True
                    break
            if not rendered:
                continue

            mask = np.array(img) < 128
            img_arr = 1.0 - mask.astype(np.float32)
            img_tensor = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0).to(device)
            char_idx = torch.tensor([char_to_index(char)], dtype=torch.long).to(device)
            samples.append({
                'img_tensor': img_tensor,
                'char_idx': char_idx,
                'mask': mask,
                'label': f"{font_name}_{char}",
            })
    return samples


def render_oldviz(strokes, stroke_widths, mask):
    """Old viz: full-color solid fill + round caps, no centerline, no dots."""
    img = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    # Draw glyph background pixel by pixel (matches train.py's style)
    arr = np.array(img)
    arr[mask] = (220, 220, 220)
    img = Image.fromarray(arr)

    draw = ImageDraw.Draw(img)
    for si, stroke in enumerate(strokes):
        color = COLORS[si % len(COLORS)]
        sw = stroke_widths[si] if si < len(stroke_widths) else [2]
        for i in range(len(stroke) - 1):
            x1, y1 = stroke[i]
            x2, y2 = stroke[i + 1]
            if isinstance(sw, list) and len(sw) > i + 1:
                w = max(1, int((sw[i] + sw[i + 1]) / 2))
            elif isinstance(sw, list) and len(sw) > 0:
                w = max(1, int(sw[0]))
            else:
                w = max(1, int(sw))
            draw.line([(x1, y1), (x2, y2)], fill=color, width=w)
        for i, (px, py) in enumerate(stroke):
            if isinstance(sw, list) and len(sw) > i:
                r = max(1, int(sw[i])) // 2
            else:
                r = max(1, int(sw[0] if isinstance(sw, list) else sw)) // 2
            if r > 0:
                draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=color)
    return img


def render_checkpoint(ckpt_path, out_dir, samples, device):
    print(f"\n=== {os.path.basename(ckpt_path)} ===")
    model = StrokePredictor(feature_dim=256).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f"  loaded epoch {epoch}")

    os.makedirs(out_dir, exist_ok=True)
    for sample in samples:
        with torch.no_grad():
            strokes, stroke_widths = model.predict_strokes(
                sample['img_tensor'], sample['char_idx'],
                CANVAS_SIZE, existence_threshold=0.3,
            )
        img = render_oldviz(strokes, stroke_widths, sample['mask'])
        img.save(os.path.join(out_dir, f"{sample['label']}.png"))
    print(f"  saved {len(samples)} to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True,
                        help='Source run dir containing the checkpoints, e.g. '
                             'history/20260410_i34502855_b7a8ff4')
    parser.add_argument('--checkpoints', nargs='+',
                        default=['checkpoint_epoch14.pt', 'checkpoint_epoch19.pt',
                                 'checkpoint_epoch29.pt', 'checkpoint_epoch64.pt',
                                 'checkpoint_epoch84.pt', 'best_model.pt'],
                        help='Checkpoint filenames (relative to run dir)')
    parser.add_argument('--out-root', default=None,
                        help='Output history dir (defaults to <date>_oldviz_<commit>)')
    args = parser.parse_args()

    if args.out_root is None:
        commit = os.path.basename(args.run_dir).split('_')[-1]
        today = date.today().strftime('%Y%m%d')
        args.out_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'history', f'{today}_oldviz_{commit}',
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Output root: {args.out_root}")

    samples = build_samples(device)
    print(f"Built {len(samples)} tracking samples")
    if not samples:
        sys.exit("no samples — check font paths")

    for ckpt_name in args.checkpoints:
        ckpt_path = os.path.join(args.run_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"SKIP missing: {ckpt_path}")
            continue
        # Determine epoch label from checkpoint file or state dict
        if ckpt_name == 'best_model.pt':
            ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            label = f"best_ep{ck.get('epoch', '?'):03d}"
        else:
            # e.g. checkpoint_epoch84.pt -> epoch_084
            num = ''.join(c for c in ckpt_name if c.isdigit())
            label = f"epoch_{int(num):03d}"
        out_dir = os.path.join(args.out_root, 'tracking', label)
        render_checkpoint(ckpt_path, out_dir, samples, device)

    print(f"\nDone. History dir: {args.out_root}")


if __name__ == '__main__':
    main()
