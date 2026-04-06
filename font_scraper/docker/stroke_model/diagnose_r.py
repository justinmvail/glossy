#!/usr/bin/env python3
"""Diagnose why R has fewer strokes than A by inspecting model internals.

Loads the latest checkpoint, runs inference on Brown_Fox_R and Brown_Fox_A,
and prints per-stroke existence values + merge penalty breakdown per pair.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model import StrokePredictor, CANVAS_SIZE, char_to_index

CHECKPOINT = "/home/server/glossy/font_scraper/docker/stroke_model/history/latest/best_model.pt"
FONT_PATH = "/home/server/glossy/font_scraper/fonts/dafont/BrownFox.otf"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
model = StrokePredictor(feature_dim=256).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")


def render_glyph(char, font_path):
    img = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw = ImageDraw.Draw(img)
    for fs in range(200, 20, -5):
        try:
            font = ImageFont.truetype(font_path, fs)
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
            break
    mask = np.array(img) < 128
    img_arr = 1.0 - mask.astype(np.float32)
    img_tensor = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)
    return img_tensor, mask_tensor


def diagnose(char):
    print(f"\n{'='*70}\n  {char}  \n{'='*70}")
    img, mask = render_glyph(char, FONT_PATH)
    char_idx = torch.tensor([char_to_index(char)], dtype=torch.long).to(device)

    with torch.no_grad():
        out = model(img, char_idx, mask)

    existence = out['existence'][0]  # (8,)
    points = out['points'][0]          # (8, 40, 2)
    widths = out['widths'][0]          # (8, 40)
    n_points = out['point_count_logits'][0].argmax(dim=-1) + 1  # (8,)
    n_points = n_points.clamp(min=2, max=40)

    # Recover the raw logit: sigmoid^-1(existence)
    logits = torch.logit(existence.clamp(1e-6, 1 - 1e-6))

    print(f"\nPer-stroke existence (threshold 0.3 to be visible):")
    print(f"{'stroke':<8}{'logit':>10}{'sigmoid':>10}{'n_pts':>8}{'width':>10}")
    for s in range(8):
        visible = "VISIBLE" if existence[s].item() > 0.3 else "---"
        avg_w = widths[s, :n_points[s]].mean().item()
        print(f"{s:<8}{logits[s].item():>10.3f}{existence[s].item():>10.3f}"
              f"{n_points[s].item():>8}{avg_w:>10.2f}  {visible}")

    # Compute merge penalty contribution per pair (current implementation)
    print(f"\nMerge penalty breakdown (pairs with score > 0.1):")
    pts_scaled = points * CANVAS_SIZE  # (8, 40, 2)
    print(f"{'pair':<10}{'existence_i*j':>15}{'best_score':>12}{'combo':>12}{'contrib':>12}")

    total_merge = 0.0
    entries = []
    for i in range(8):
        for j in range(i + 1, 8):
            both_active = existence[i] * existence[j]
            last_i_idx = (n_points[i] - 1).clamp(min=0)
            last_i = pts_scaled[i, last_i_idx]
            first_i = pts_scaled[i, 0]
            prev_i_idx = (n_points[i] - 2).clamp(min=0)
            prev_i = pts_scaled[i, prev_i_idx]
            dir_end_i = last_i - prev_i
            dir_end_i = dir_end_i / (dir_end_i.norm().clamp(min=1e-6))
            dir_start_i = pts_scaled[i, 1] - first_i
            dir_start_i = dir_start_i / (dir_start_i.norm().clamp(min=1e-6))

            last_j_idx = (n_points[j] - 1).clamp(min=0)
            last_j = pts_scaled[j, last_j_idx]
            first_j = pts_scaled[j, 0]
            prev_j_idx = (n_points[j] - 2).clamp(min=0)
            prev_j = pts_scaled[j, prev_j_idx]
            dir_end_j = last_j - prev_j
            dir_end_j = dir_end_j / (dir_end_j.norm().clamp(min=1e-6))
            dir_start_j = pts_scaled[j, 1] - first_j
            dir_start_j = dir_start_j / (dir_start_j.norm().clamp(min=1e-6))

            w_end_i = widths[i, last_i_idx]
            w_start_i = widths[i, 0]
            w_end_j = widths[j, last_j_idx]
            w_start_j = widths[j, 0]

            combo_names = ['end→start', 'start→end', 'end→end  ', 'start→start']
            combos = [
                (last_i, first_j, dir_end_i, dir_start_j, w_end_i, w_start_j),
                (first_i, last_j, -dir_start_i, -dir_end_j, w_start_i, w_end_j),
                (last_i, last_j, dir_end_i, -dir_end_j, w_end_i, w_end_j),
                (first_i, first_j, -dir_start_i, dir_start_j, w_start_i, w_start_j),
            ]
            best_score = 0.0
            best_combo = ""
            for name, (pa, pb, da, db, wa, wb) in zip(combo_names, combos):
                gap = (pa - pb).norm()
                close = torch.sigmoid((15.0 - gap) * 0.5)
                aligned = (da * db).sum().clamp(min=0)
                w_compat = torch.sigmoid((5.0 - (wa - wb).abs()) * 0.5)
                score = (close * aligned * w_compat).item()
                if score > best_score:
                    best_score = score
                    best_combo = name
            contrib = best_score * both_active.item()
            total_merge += contrib
            if best_score > 0.1:
                entries.append((i, j, both_active.item(), best_score, best_combo, contrib))

    entries.sort(key=lambda x: -x[5])
    for i, j, ba, bs, bc, c in entries:
        print(f"({i},{j})     {ba:>15.3f}{bs:>12.3f}{bc:>12}{c:>12.4f}")

    loss_merge = total_merge / 28  # normalized by unique pairs
    print(f"\nTotal merge loss: {loss_merge:.4f}")
    print(f"Weighted contribution (weight=2.0): {2.0 * loss_merge:.4f}")


diagnose('R')
diagnose('A')
