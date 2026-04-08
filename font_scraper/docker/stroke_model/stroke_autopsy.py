#!/usr/bin/env python3
"""Stroke Autopsy — deep diagnostic tool for analyzing stroke model behavior.

Loads checkpoints, runs inference on specific characters, and produces
detailed per-stroke analysis including existence gradient decomposition,
coverage measurements, width profiles, and point positions.

Designed for comparing two runs (e.g., flat caps vs round caps) at the
same epoch to identify what causes training collapse.

Usage:
    # On Vast.ai instance (needs CUDA + Triton):
    python3 stroke_autopsy.py --checkpoint /path/to/model.pt --char R --font /path/to/font.ttf

    # Compare two checkpoints:
    python3 stroke_autopsy.py --compare /path/a.pt /path/b.pt --labels "round_caps" "flat_caps"

    # Full report on a single checkpoint:
    python3 stroke_autopsy.py --checkpoint /path/to/model.pt --full-report
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser(description='Stroke model diagnostic tool')
    p.add_argument('--checkpoint', type=str, help='Single checkpoint to analyze')
    p.add_argument('--compare', nargs=2, type=str, help='Two checkpoints to compare')
    p.add_argument('--labels', nargs=2, type=str, default=['A', 'B'], help='Labels for comparison')
    p.add_argument('--flat-caps', nargs=2, type=str, default=['auto', 'auto'],
                   help='Force flat_caps for each checkpoint: true/false/auto')
    p.add_argument('--char', type=str, default='R', help='Character to analyze')
    p.add_argument('--font', type=str, default='/data/fonts/dafont/BrownFox.otf', help='Font path')
    p.add_argument('--font-name', type=str, default='Brown_Fox', help='Font display name')
    p.add_argument('--output-dir', type=str, default='/workspace/autopsy', help='Output directory')
    p.add_argument('--full-report', action='store_true', help='Run all diagnostics on multiple chars')
    p.add_argument('--chars', type=str, default='A,R,g,8', help='Characters for full report (comma-sep)')
    return p.parse_args()


def render_glyph(char, font_path, canvas_size=224):
    """Render a character from a font file into a tensor."""
    img = Image.new('L', (canvas_size, canvas_size), 255)
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
        if tw <= canvas_size * 0.9 and th <= canvas_size * 0.9:
            x = (canvas_size - tw) / 2 - bbox[0]
            y = (canvas_size - th) / 2 - bbox[1]
            draw.text((x, y), char, fill=0, font=font)
            break
    mask = np.array(img) < 128
    img_arr = 1.0 - mask.astype(np.float32)
    img_tensor = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
    return img_tensor, mask_tensor


def load_model(checkpoint_path, device, force_flat_caps=None):
    """Load model from checkpoint, auto-detecting flat_caps and serifs.

    Args:
        checkpoint_path: path to .pt checkpoint
        device: torch device
        force_flat_caps: if not None, override flat_caps detection (True/False)
    """
    from model import StrokePredictor
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']

    # Detect features from state dict
    has_serifs = any('serif_head' in k for k in state.keys())

    if force_flat_caps is not None:
        flat_caps = force_flat_caps
    else:
        # Default: try to detect. Round-cap checkpoints (pre ff4707b) have no flat_caps.
        # Flat_caps doesn't affect weights, only rendering — both load fine.
        flat_caps = False  # conservative default

    model = StrokePredictor(feature_dim=256, flat_caps=flat_caps, serifs=has_serifs).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    loss = ckpt.get('loss', None)
    print(f"  Loaded epoch {epoch}, loss={loss}, flat_caps={flat_caps}, serifs={has_serifs}")
    return model, epoch


def analyze_stroke(model, img_tensor, mask_tensor, char, device, canvas_size=224):
    """Run full analysis on a single character. Returns detailed per-stroke metrics."""
    from model import CANVAS_SIZE, RENDER_SIZE, MAX_STROKES, MAX_POINTS, char_to_index
    from triton_render import render_single_stroke_triton

    char_idx = torch.tensor([char_to_index(char)], dtype=torch.long).to(device)
    img = img_tensor.to(device)
    mask = mask_tensor.to(device)

    # Forward with gradient tracking for existence decomposition
    model.eval()
    # Need gradients for existence analysis
    out = model(img, char_idx, mask)

    existence = out['existence'][0].detach()  # (S,)
    points = out['points'][0].detach()          # (S, N, 2)
    widths = out['widths'][0].detach()          # (S, N)
    pc_logits = out['point_count_logits'][0].detach()
    n_points = (pc_logits.argmax(dim=-1) + 1).clamp(min=2, max=MAX_POINTS)
    canvas_inv = out['canvas_inv'][0].detach()  # (R, R)
    target = out['target'][0].detach()          # (R, R)
    stroke_renders = out['stroke_renders'][0].detach()  # (S, R, R)

    R = RENDER_SIZE
    S = MAX_STROKES
    N = MAX_POINTS

    results = {
        'char': char,
        'flat_caps': model.flat_caps,
        'canvas_size': canvas_size,
        'render_size': R,
        'strokes': [],
        'canvas_coverage': float((1 - canvas_inv).mean()),
        'target_coverage': float(target.mean()),
        'canvas_mse': float(((1 - canvas_inv - target) ** 2 * 10.0).mean()),
    }

    # Glyph mask at full resolution for boundary analysis
    glyph_mask = mask[0]  # (224, 224)

    for s in range(S):
        exist = existence[s].item()
        n_pts = n_points[s].item()
        pts = points[s, :n_pts]  # (n_pts, 2)
        ws = widths[s, :n_pts]   # (n_pts,)
        sr = stroke_renders[s]   # (R, R)

        # Pixel coverage
        ink_pixels = (1 - sr).sum().item()  # number of inked pixels (soft)
        glyph_target_down = F.interpolate(target.unsqueeze(0).unsqueeze(0), size=(R, R),
                                           mode='bilinear', align_corners=False).squeeze()
        ink_on_glyph = ((1 - sr) * target).sum().item()
        ink_on_bg = ((1 - sr) * (1 - target)).sum().item()

        # Point positions relative to glyph
        pts_px = pts * canvas_size  # (n_pts, 2) in pixel coords
        pts_grid = (pts * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)  # (1, 1, n_pts, 2)
        glyph_4d = glyph_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 224, 224)
        on_glyph = F.grid_sample(glyph_4d, pts_grid, align_corners=False,
                                  mode='nearest', padding_mode='zeros')
        on_glyph = on_glyph.squeeze().cpu().numpy()  # (n_pts,)
        n_on_glyph = int(on_glyph.sum())
        n_off_glyph = n_pts - n_on_glyph

        # Width profile
        width_values = ws.cpu().numpy().tolist()
        width_mean = float(ws.mean())
        width_std = float(ws.std())
        width_start = float(ws[0])
        width_end = float(ws[-1])
        width_max = float(ws.max())
        width_min = float(ws.min())

        # Width taper detection: are endpoints significantly thinner than middle?
        if n_pts >= 4:
            mid_start = n_pts // 4
            mid_end = 3 * n_pts // 4
            width_mid_mean = float(ws[mid_start:mid_end].mean())
            width_endpoint_mean = float((ws[0] + ws[-1]) / 2)
            taper_ratio = width_endpoint_mean / max(width_mid_mean, 1e-6)
        else:
            taper_ratio = 1.0

        # Sinuosity
        seg = pts_px[1:] - pts_px[:-1]
        seg_lengths = seg.norm(dim=-1)
        path_length = float(seg_lengths.sum())
        endpoint_dist = float((pts_px[-1] - pts_px[0]).norm())
        sinuosity = path_length / max(endpoint_dist, 1e-6)

        # Self-overlap: check non-adjacent point distances
        if n_pts >= 5:
            dists = torch.cdist(pts_px.unsqueeze(0), pts_px.unsqueeze(0)).squeeze()
            idx = torch.arange(n_pts, device=pts_px.device)
            non_adj = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > 2
            w_half_max = torch.max(ws.unsqueeze(0).expand(n_pts, -1),
                                    ws.unsqueeze(1).expand(-1, n_pts)) * 0.5
            inside = (dists < w_half_max) & non_adj
            n_self_overlap = int(inside.any(dim=1).sum())
        else:
            n_self_overlap = 0

        # Inter-point spacing
        spacings = seg_lengths.cpu().numpy().tolist()
        min_spacing = float(seg_lengths.min()) if len(seg_lengths) > 0 else 0
        max_spacing = float(seg_lengths.max()) if len(seg_lengths) > 0 else 0

        # Point coordinates
        point_coords = pts_px.cpu().numpy().tolist()

        results['strokes'].append({
            'slot': s,
            'existence': exist,
            'existence_logit': float(torch.logit(torch.tensor(exist).clamp(1e-6, 1-1e-6))),
            'n_points': n_pts,
            'visible': exist > 0.3,
            # Coverage
            'ink_pixels': ink_pixels,
            'ink_on_glyph': ink_on_glyph,
            'ink_on_bg': ink_on_bg,
            'coverage_efficiency': ink_on_glyph / max(ink_pixels, 1e-6),
            # Points
            'n_on_glyph': n_on_glyph,
            'n_off_glyph': n_off_glyph,
            'off_glyph_fraction': n_off_glyph / n_pts,
            # Width
            'width_mean': width_mean,
            'width_std': width_std,
            'width_start': width_start,
            'width_end': width_end,
            'width_max': width_max,
            'width_min': width_min,
            'taper_ratio': taper_ratio,
            'width_profile': width_values,
            # Geometry
            'sinuosity': sinuosity,
            'path_length': path_length,
            'endpoint_dist': endpoint_dist,
            'n_self_overlap': n_self_overlap,
            'min_spacing': min_spacing,
            'max_spacing': max_spacing,
            # Raw data
            'point_coords': point_coords,
            'on_glyph_mask': on_glyph.tolist(),
        })

    return results


def compute_existence_gradients(model, img_tensor, mask_tensor, char, device):
    """Compute per-loss-term gradient decomposition on existence.

    Returns the gradient of each loss term w.r.t. each stroke's existence.
    This reveals which forces push existence UP vs DOWN.

    Positive gradient = gradient descent pushes existence DOWN (bad for survival).
    Negative gradient = gradient descent pushes existence UP (good for survival).
    """
    from model import CANVAS_SIZE, MAX_STROKES, char_to_index

    char_idx = torch.tensor([char_to_index(char)], dtype=torch.long).to(device)
    img = img_tensor.to(device)
    mask = mask_tensor.to(device)

    # Run in train mode to ensure gradients flow through the model
    model.train()
    out = model(img, char_idx, mask)
    model.eval()

    existence = out['existence']  # (1, S) — should have grad_fn from sigmoid
    if not existence.requires_grad:
        return {'error': 'existence does not require grad — check model mode'}

    loss_weights = {
        'canvas_mse': 1.0, 'merge': 2.0, 'sinuosity': 0.01, 'smoothness': 0.001,
        'width_smooth': 0.01, 'hires_mse': 1.0, 'overlap': 0.3, 'parallel': 1.0,
        'boundary': 0.1, 'self_overlap': 0.5, 'exist_reward': 0.3, 'exist_decay': 0.05,
    }

    # Loss terms that can affect existence (based on code analysis):
    # - canvas_mse: INDIRECT through compositing (LIVE)
    # - hires_mse: INDIRECT through DistanceFieldRender (LIVE)
    # - exist_decay: DIRECT (LIVE)
    # - merge: through existence[:, i] * existence[:, j] product (LIVE)
    # - overlap: through existence[:, :, None, None] multiplication (LIVE)
    # - parallel: through existence[:, i] * existence[:, j] product (LIVE)
    # All others use (existence > 0.3).float() hard threshold → BLOCKED

    loss_terms = {
        'canvas_mse': lambda: _compute_canvas_mse(out),
        'hires_mse': lambda: _compute_hires_mse(out, CANVAS_SIZE),
        'exist_decay': lambda: _compute_exist_decay(out),
        'exist_reward': lambda: _compute_exist_reward(out),
        'merge': lambda: _compute_merge(out, CANVAS_SIZE),
        'overlap': lambda: _compute_overlap(out),
    }

    grad_decomp = {}
    for name, loss_fn in loss_terms.items():
        try:
            loss_val = loss_fn()
            if not loss_val.requires_grad:
                grad_decomp[name] = {'note': 'no gradient (detached or constant)'}
                continue

            # Compute gradient of this loss w.r.t. existence
            grads = torch.autograd.grad(loss_val, existence, retain_graph=True,
                                         allow_unused=True)
            if grads[0] is not None:
                weight = loss_weights.get(name, 0.0)
                raw_grad = grads[0][0].detach()  # (S,)
                grad_decomp[name] = {
                    'raw_loss': loss_val.item(),
                    'weight': weight,
                    'weighted_loss': loss_val.item() * weight,
                    'grad_per_stroke': (raw_grad * weight).tolist(),
                    'raw_grad_per_stroke': raw_grad.tolist(),
                }
            else:
                grad_decomp[name] = {
                    'raw_loss': loss_val.item(),
                    'weight': loss_weights.get(name, 0.0),
                    'grad_per_stroke': [0.0] * MAX_STROKES,
                    'note': 'gradient is None (existence not in computation graph for this loss)',
                }
        except Exception as e:
            grad_decomp[name] = {'error': str(e)}

    return grad_decomp


def _compute_canvas_mse(out):
    ink = 1.0 - out['canvas_inv']
    target = out['target']
    sq_err = (ink - target) ** 2
    weight_map = target * 10.0 + (1.0 - target) * 10.0
    return (sq_err * weight_map).mean()


def _compute_hires_mse(out, canvas_size):
    from model import HIRES_RENDER_SIZE
    from triton_render import DistanceFieldRender
    HR = HIRES_RENDER_SIZE
    hires_target = F.interpolate(
        out['glyph_mask'].unsqueeze(1), size=(HR, HR),
        mode='bilinear', align_corners=False,
    ).squeeze(1)
    existence = out['existence']
    points = out['points']
    pc_logits = out['point_count_logits']
    n_points = (pc_logits.argmax(dim=-1) + 1).clamp(min=2, max=points.shape[2])
    flat_caps = out.get('flat_caps', False)
    hires_canvas_inv = DistanceFieldRender.apply(
        points, out['widths'], existence, n_points,
        canvas_size, HR, 4.0, 0.3, flat_caps,
    )
    hires_ink = 1.0 - hires_canvas_inv
    hires_sq_err = (hires_ink - hires_target) ** 2
    hires_wmap = hires_target * 10.0 + (1.0 - hires_target) * 10.0
    return (hires_sq_err * hires_wmap).mean()


def _compute_exist_decay(out):
    existence = out['existence']
    S = existence.shape[1]
    step_weights = torch.arange(S, device=existence.device, dtype=torch.float32)
    return (existence * step_weights.unsqueeze(0)).mean()


def _compute_exist_reward(out):
    """Coverage-based existence reward (the fix for zero canvas_mse gradient)."""
    existence = out['existence']  # (B, S)
    if 'stroke_renders' not in out:
        return torch.tensor(0.0, device=existence.device)
    stroke_renders = out['stroke_renders']  # (B, S, R, R)
    target = out['target']  # (B, R, R)
    per_stroke_ink = 1.0 - stroke_renders
    target_expanded = target.unsqueeze(1)
    glyph_pixel_count = target.sum(dim=(1, 2)).clamp(min=1)
    stroke_on_glyph = (per_stroke_ink * target_expanded).sum(dim=(2, 3))
    coverage_frac = stroke_on_glyph / glyph_pixel_count.unsqueeze(1)
    return -(existence * coverage_frac.detach()).mean()


def _compute_merge(out, canvas_size):
    points = out['points']
    existence = out['existence']
    pc_logits = out['point_count_logits']
    n_points = (pc_logits.argmax(dim=-1) + 1).clamp(min=2, max=points.shape[2])
    B, S, N, _ = points.shape
    pts_scaled = points * canvas_size
    merge_penalty = torch.tensor(0.0, device=points.device)
    for i in range(S):
        for j in range(i + 1, S):
            both = existence[:, i] * existence[:, j]
            last_i = (n_points[:, i] - 1).clamp(min=0)
            last_i_pt = pts_scaled[:, i].gather(1, last_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)
            first_j = pts_scaled[:, j, 0]
            gap = (last_i_pt - first_j).norm(dim=-1)
            close = torch.sigmoid((15.0 - gap) * 0.5)
            merge_penalty = merge_penalty + (close * both).mean()
    return merge_penalty / max(S * (S - 1) // 2, 1)


def _compute_overlap(out):
    if 'stroke_renders' not in out:
        return torch.tensor(0.0)
    existence = out['existence']
    sr = out['stroke_renders']
    target = out['target']
    per_stroke_ink = (1.0 - sr) * existence[:, :, None, None]
    total = per_stroke_ink.sum(dim=1)
    excess = (total - 1.0).clamp(min=0) ** 2
    return (excess * target).mean()


def print_report(results, label=""):
    """Print a human-readable report from analysis results."""
    prefix = f"[{label}] " if label else ""
    print(f"\n{'='*70}")
    print(f"{prefix}Character: {results['char']}  |  Flat caps: {results['flat_caps']}")
    print(f"{prefix}Canvas MSE: {results['canvas_mse']:.4f}  |  Coverage: {results['canvas_coverage']:.3f}")
    print(f"{'='*70}")

    print(f"\n{'slot':<5}{'exist':>8}{'logit':>8}{'n_pts':>6}{'vis':>5}"
          f"{'ink_px':>8}{'on_gly':>8}{'off_gl':>8}{'eff':>7}"
          f"{'w_mean':>8}{'taper':>7}{'sinu':>7}{'s_ovlp':>7}")
    print('-' * 95)

    for s in results['strokes']:
        vis = "YES" if s['visible'] else "---"
        print(f"{s['slot']:<5}{s['existence']:>8.3f}{s['existence_logit']:>8.2f}"
              f"{s['n_points']:>6}{vis:>5}"
              f"{s['ink_pixels']:>8.0f}{s['ink_on_glyph']:>8.0f}{s['n_off_glyph']:>8}"
              f"{s['coverage_efficiency']:>7.2f}"
              f"{s['width_mean']:>8.1f}{s['taper_ratio']:>7.2f}"
              f"{s['sinuosity']:>7.2f}{s['n_self_overlap']:>7}")


def print_gradient_report(grad_decomp, label=""):
    """Print existence gradient decomposition."""
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}Existence gradient decomposition (positive = pushes existence UP):")
    print(f"{'loss_term':<15}{'weight':>8}{'raw_loss':>10}{'contribution':>12}"
          f"  grad_per_stroke [0..7]")
    print('-' * 90)

    for name, info in grad_decomp.items():
        if 'error' in info:
            print(f"{name:<15} ERROR: {info['error']}")
            continue
        if 'note' in info and 'grad_per_stroke' not in info:
            print(f"{name:<15} {info['note']}")
            continue
        weight = info.get('weight', 0)
        raw = info.get('raw_loss', 0)
        grads = info.get('grad_per_stroke', [])
        grad_str = ' '.join(f"{g:>7.4f}" for g in grads[:8])
        print(f"{name:<15}{weight:>8.3f}{raw:>10.4f}{raw*weight:>12.4f}  {grad_str}")

    # Net force per stroke
    print(f"\n{prefix}Net force per stroke (sum of all gradients):")
    net = [0.0] * 8
    for name, info in grad_decomp.items():
        grads = info.get('grad_per_stroke', [0.0] * 8)
        for i in range(min(len(grads), 8)):
            net[i] += grads[i]
    for i, n in enumerate(net):
        direction = "↑ ALIVE" if n < 0 else "↓ DYING" if n > 0.001 else "~ neutral"
        print(f"  Stroke {i}: net gradient = {n:>+.5f}  {direction}")


def save_visualization(results, output_path):
    """Save a visualization image showing stroke positions, widths, coverage."""
    CS = results['canvas_size']
    img = Image.new('RGB', (CS * 2, CS), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    colors = [(255, 0, 0), (0, 150, 0), (0, 0, 255), (255, 128, 0),
              (128, 0, 255), (0, 200, 200), (200, 0, 128), (128, 128, 0)]

    # Left side: strokes on glyph
    # (would need glyph mask to draw, skip for now — just draw strokes)
    for s in results['strokes']:
        if not s['visible']:
            continue
        color = colors[s['slot'] % len(colors)]
        fill = tuple(min(255, c + 120) for c in color)
        pts = s['point_coords']
        ws = s['width_profile']

        # Draw fill
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            w = max(1, int((ws[i] + ws[min(i+1, len(ws)-1)]) / 2))
            draw.line([(x1, y1), (x2, y2)], fill=fill, width=w)

        # Draw centerline
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

        # On/off glyph markers
        for i, (px, py) in enumerate(pts):
            on = s['on_glyph_mask'][i] > 0.5
            marker_color = (0, 200, 0) if on else (255, 0, 0)
            draw.ellipse([(px-3, py-3), (px+3, py+3)], fill=marker_color, outline=color)

    # Right side: width profile visualization
    x_offset = CS
    for s in results['strokes']:
        if not s['visible']:
            continue
        color = colors[s['slot'] % len(colors)]
        ws = s['width_profile']
        n = len(ws)
        if n < 2:
            continue
        # Draw width profile as a small chart
        chart_y = 20 + s['slot'] * 25
        for i in range(n - 1):
            x1 = x_offset + 10 + int(i / n * (CS - 20))
            x2 = x_offset + 10 + int((i + 1) / n * (CS - 20))
            y1 = chart_y - int(ws[i] / 2)
            y2 = chart_y - int(ws[i + 1] / 2)
            draw.line([(x1, chart_y), (x2, chart_y)], fill=(200, 200, 200), width=1)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

    img.save(output_path)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.compare:
        # Compare two checkpoints
        chars = args.chars.split(',') if args.full_report else [args.char]

        fc_flags = []
        for fc in args.flat_caps:
            if fc.lower() == 'true':
                fc_flags.append(True)
            elif fc.lower() == 'false':
                fc_flags.append(False)
            else:
                fc_flags.append(None)

        for i, (ckpt_path, label) in enumerate(zip(args.compare, args.labels)):
            print(f"\n{'#'*70}")
            print(f"# Loading: {label} — {ckpt_path}")
            print(f"{'#'*70}")
            model, epoch = load_model(ckpt_path, device, force_flat_caps=fc_flags[i])
            print(f"Epoch: {epoch}, flat_caps: {model.flat_caps}")

            for char in chars:
                img_t, mask_t = render_glyph(char, args.font)
                results = analyze_stroke(model, img_t, mask_t, char, device)
                print_report(results, label=f"{label} ep{epoch}")

                # Existence gradient decomposition
                try:
                    grads = compute_existence_gradients(model, img_t, mask_t, char, device)
                    print_gradient_report(grads, label=f"{label} ep{epoch}")
                except Exception as e:
                    print(f"  Gradient computation failed: {e}")

                # Save visualization
                viz_path = os.path.join(args.output_dir, f"{label}_{char}_ep{epoch}.png")
                save_visualization(results, viz_path)
                print(f"  Saved: {viz_path}")

                # Save raw data
                json_path = os.path.join(args.output_dir, f"{label}_{char}_ep{epoch}.json")
                # Convert non-serializable types
                results_ser = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
                with open(json_path, 'w') as f:
                    json.dump(results_ser, f, indent=2)

    elif args.checkpoint:
        # Single checkpoint analysis
        model, epoch = load_model(args.checkpoint, device)
        print(f"Epoch: {epoch}, flat_caps: {model.flat_caps}")

        chars = args.chars.split(',') if args.full_report else [args.char]

        for char in chars:
            img_t, mask_t = render_glyph(char, args.font)
            results = analyze_stroke(model, img_t, mask_t, char, device)
            print_report(results, label=f"ep{epoch}")

            try:
                grads = compute_existence_gradients(model, img_t, mask_t, char, device)
                print_gradient_report(grads, label=f"ep{epoch}")
            except Exception as e:
                print(f"  Gradient computation failed: {e}")

            viz_path = os.path.join(args.output_dir, f"{char}_ep{epoch}.png")
            save_visualization(results, viz_path)
            print(f"  Saved: {viz_path}")
    else:
        print("Specify --checkpoint or --compare. Use --help for usage.")


if __name__ == '__main__':
    main()
