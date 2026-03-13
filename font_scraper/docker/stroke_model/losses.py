"""Loss functions for stroke prediction training.

All losses are differentiable. The rendering loss uses pydiffvg for differentiable
rasterization. Overlap and smoothness losses operate on coordinates directly.

Loss components:
    1. Coverage loss: rendered strokes vs glyph raster (MSE, glyph-weighted)
    2. Outside penalty: stroke pixels outside the glyph boundary
    3. Fractional overlap penalty: penalize duplicate paths, allow junction crossings
    4. Smoothness penalty: penalize high curvature via 2nd-order finite differences
    5. Existence loss: BCE on existence flags (learn stroke count)
"""

import torch
import torch.nn.functional as F

from model import MAX_STROKES, MAX_POINTS, CANVAS_SIZE
from render_utils import render_strokes, model_output_to_render_inputs


def coverage_loss(rendered: torch.Tensor, glyph_mask: torch.Tensor,
                  glyph_weight: float = 10.0) -> torch.Tensor:
    """MSE between rendered strokes and glyph mask, weighted heavier inside glyph.

    Args:
        rendered: (H, W) grayscale rendered image (1=white, 0=black strokes).
        glyph_mask: (H, W) binary mask (1=glyph, 0=background).
        glyph_weight: Extra weight for pixels inside the glyph.

    Returns:
        Scalar loss tensor.
    """
    # Target: 0 (black) where glyph, 1 (white) where background
    target = 1.0 - glyph_mask

    sq_err = (rendered - target) ** 2
    weight_map = glyph_mask * glyph_weight + (1.0 - glyph_mask)
    return (sq_err * weight_map).mean()


def outside_penalty(rendered: torch.Tensor,
                    glyph_mask: torch.Tensor) -> torch.Tensor:
    """Penalize stroke pixels that fall outside the glyph boundary.

    Args:
        rendered: (H, W) grayscale rendered image.
        glyph_mask: (H, W) binary mask (1=glyph, 0=background).

    Returns:
        Scalar loss tensor.
    """
    bg_mask = 1.0 - glyph_mask
    # Stroke intensity: 0=fully stroked, 1=no stroke -> invert
    stroke_pixels = (1.0 - rendered).clamp(min=0)
    outside = (stroke_pixels * bg_mask).sum()
    return outside / bg_mask.sum().clamp(min=1)


def fractional_overlap_penalty(stroke_points_list: list,
                               min_overlap_frac: float = 0.3,
                               proximity_threshold: float = 8.0) -> torch.Tensor:
    """Penalize duplicate/parallel strokes while allowing junction crossings.

    For each pair of strokes, compute what fraction of each stroke's length
    runs close and parallel to the other. Small fraction (junction crossing) is
    OK. Large fraction (duplicated path) is penalized.

    Args:
        stroke_points_list: List of (N_i, 2) tensors (pixel coordinates).
        min_overlap_frac: Fraction above which overlap is penalized.
        proximity_threshold: Distance in pixels to consider "close".

    Returns:
        Scalar loss tensor.
    """
    if len(stroke_points_list) < 2:
        return torch.tensor(0.0, device=stroke_points_list[0].device)

    device = stroke_points_list[0].device
    penalty = torch.tensor(0.0, device=device)

    for i in range(len(stroke_points_list)):
        for j in range(i + 1, len(stroke_points_list)):
            pts_a = stroke_points_list[i]  # (Na, 2)
            pts_b = stroke_points_list[j]  # (Nb, 2)

            # Pairwise distances: (Na, Nb)
            dists = torch.cdist(pts_a.unsqueeze(0), pts_b.unsqueeze(0))[0]

            # For each point in A, check if it's close to any point in B
            min_dists_a = dists.min(dim=1).values  # (Na,)
            close_a = (min_dists_a < proximity_threshold).float()
            frac_a = close_a.sum() / max(pts_a.shape[0], 1)

            # For each point in B, check if it's close to any point in A
            min_dists_b = dists.min(dim=0).values  # (Nb,)
            close_b = (min_dists_b < proximity_threshold).float()
            frac_b = close_b.sum() / max(pts_b.shape[0], 1)

            # Penalize if large fraction of either stroke overlaps
            overlap = torch.max(frac_a, frac_b)
            excess = (overlap - min_overlap_frac).clamp(min=0)
            penalty = penalty + excess ** 2

    return penalty


def smoothness_penalty(stroke_points_list: list) -> torch.Tensor:
    """Penalize high curvature via second-order finite differences.

    Prefers gentle arcs over zigzag paths.

    Args:
        stroke_points_list: List of (N_i, 2) tensors.

    Returns:
        Scalar loss tensor.
    """
    device = stroke_points_list[0].device
    penalty = torch.tensor(0.0, device=device)

    for points in stroke_points_list:
        if points.shape[0] >= 3:
            d1 = points[1:] - points[:-1]       # first differences
            d2 = d1[1:] - d1[:-1]               # second differences
            penalty = penalty + (d2 ** 2).mean()

    return penalty


def existence_loss(predicted_existence: torch.Tensor,
                   n_active: int = None,
                   target_existence: torch.Tensor = None) -> torch.Tensor:
    """BCE loss on existence flags.

    During training, we don't have ground truth stroke counts. Instead, we use
    a soft target that encourages the model to use a reasonable number of strokes.
    The coverage loss drives the model to use enough strokes; this loss regularizes
    against using too many.

    Args:
        predicted_existence: (B, MAX_STROKES) sigmoid probabilities.
        n_active: If provided, target is n_active strokes existing, rest not.
        target_existence: If provided, explicit target tensor (B, MAX_STROKES).

    Returns:
        Scalar loss tensor.
    """
    if target_existence is not None:
        return F.binary_cross_entropy(predicted_existence, target_existence)

    # Soft regularizer: encourage total existence to be moderate (3-5 strokes)
    # Penalize sum of existence probabilities being too high
    total_exist = predicted_existence.sum(dim=1)  # (B,)
    target_count = 4.0  # reasonable default for most glyphs
    return ((total_exist - target_count) ** 2).mean() * 0.01


def total_loss(model_output: dict, glyph_mask: torch.Tensor,
               canvas_size: int = CANVAS_SIZE,
               weights: dict = None) -> tuple:
    """Compute total training loss from model outputs and glyph mask.

    Args:
        model_output: Dict from StrokePredictor.forward() with keys:
            existence, points, widths, point_count_logits.
        glyph_mask: (B, H, W) binary glyph masks (1=glyph, 0=bg).
        canvas_size: Canvas size in pixels.
        weights: Dict of loss weights. Defaults to reasonable values.

    Returns:
        Tuple of (total_loss_tensor, loss_dict) where loss_dict contains
        individual loss components for logging.
    """
    if weights is None:
        weights = {
            'coverage': 1.0,
            'outside': 5.0,
            'overlap': 2.0,
            'smoothness': 0.001,
            'existence': 0.1,
        }

    B = glyph_mask.shape[0]
    device = glyph_mask.device

    loss_coverage = torch.tensor(0.0, device=device)
    loss_outside = torch.tensor(0.0, device=device)
    loss_overlap = torch.tensor(0.0, device=device)
    loss_smooth = torch.tensor(0.0, device=device)
    n_rendered = 0

    for b in range(B):
        pts = model_output['points'][b]           # (MAX_STROKES, MAX_POINTS, 2)
        wds = model_output['widths'][b]            # (MAX_STROKES,)
        exist = model_output['existence'][b]       # (MAX_STROKES,)
        pc_logits = model_output['point_count_logits'][b]  # (MAX_STROKES, MAX_POINTS)

        stroke_pts, stroke_wds, active = model_output_to_render_inputs(
            pts, wds, exist, pc_logits, canvas_size, existence_threshold=0.3,
        )

        if not stroke_pts:
            # No strokes predicted — penalize heavily via coverage
            loss_coverage = loss_coverage + 1.0
            continue

        # Render all strokes together
        rendered = render_strokes(stroke_pts, stroke_wds, canvas_size, device)

        mask_b = glyph_mask[b]
        loss_coverage = loss_coverage + coverage_loss(rendered, mask_b)
        loss_outside = loss_outside + outside_penalty(rendered, mask_b)
        loss_overlap = loss_overlap + fractional_overlap_penalty(stroke_pts)
        loss_smooth = loss_smooth + smoothness_penalty(stroke_pts)
        n_rendered += 1

    # Average over batch
    if n_rendered > 0:
        loss_coverage = loss_coverage / B
        loss_outside = loss_outside / B
        loss_overlap = loss_overlap / B
        loss_smooth = loss_smooth / B

    loss_exist = existence_loss(model_output['existence'])

    total = (weights['coverage'] * loss_coverage +
             weights['outside'] * loss_outside +
             weights['overlap'] * loss_overlap +
             weights['smoothness'] * loss_smooth +
             weights['existence'] * loss_exist)

    loss_dict = {
        'total': total.item(),
        'coverage': loss_coverage.item(),
        'outside': loss_outside.item(),
        'overlap': loss_overlap.item(),
        'smoothness': loss_smooth.item(),
        'existence': loss_exist.item(),
    }

    return total, loss_dict
