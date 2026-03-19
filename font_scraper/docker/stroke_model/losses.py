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
try:
    from render_utils import render_strokes, model_output_to_render_inputs
except ImportError:
    render_strokes = None
    model_output_to_render_inputs = None
from triton_render import render_strokes_triton


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


def unique_coverage_loss(points, widths, existence, n_points,
                         glyph_mask, canvas_size, render_size=56):
    """Unified coverage loss: each glyph pixel should be covered by exactly one stroke.

    Computes per-stroke soft coverage, sums to get total stroke intensity per pixel,
    then MSE against the glyph mask. This single loss replaces coverage + overlap +
    outside:
      - Uncovered glyph pixel (0 vs 1): drives strokes to cover the glyph
      - One stroke on glyph pixel (1 vs 1): perfect, zero loss
      - Multiple strokes on glyph pixel (3 vs 1): penalizes redundancy
      - Stroke on background (>0 vs 0): penalizes going outside

    Args:
        points: (B, S, N, 2) normalized coordinates.
        widths: (B, S) stroke widths in pixels.
        existence: (B, S) existence probabilities.
        n_points: (B, S) valid point counts.
        glyph_mask: (B, H, W) binary glyph masks.
        canvas_size: int.
        render_size: int, resolution to compute at (lower = faster).

    Returns:
        Scalar loss tensor.
    """
    B, S, N, _ = points.shape
    device = points.device

    # Scale points to render resolution
    scale = render_size / canvas_size
    pts = (points * canvas_size * scale).clamp(0, render_size - 1)  # (B, S, N, 2)
    w = widths * scale  # (B, S)

    # Downsample glyph mask to render resolution
    glyph_small = F.interpolate(
        glyph_mask.unsqueeze(1), size=(render_size, render_size),
        mode='bilinear', align_corners=False,
    ).squeeze(1)  # (B, render_size, render_size)

    # Pixel grid: (P, 2)
    P = render_size * render_size
    gy, gx = torch.meshgrid(
        torch.arange(render_size, device=device, dtype=torch.float32),
        torch.arange(render_size, device=device, dtype=torch.float32),
        indexing='ij',
    )
    pixels = torch.stack([gx, gy], dim=-1).reshape(P, 2)  # (P, 2)

    active = (existence > 0.3).float()  # (B, S)

    # Accumulate total stroke intensity per pixel
    total_intensity = torch.zeros(B, P, device=device)  # (B, P)

    for s in range(S):
        s_np = n_points[:, s]  # (B,)
        s_w = w[:, s]  # (B,)
        s_active = active[:, s]  # (B,)

        # Min distance from each pixel to any segment of this stroke
        min_dist = torch.full((B, P), 1e6, device=device)

        for seg in range(N - 1):
            valid = seg < (s_np - 1)  # (B,)
            if not valid.any():
                continue

            a = pts[:, s, seg]      # (B, 2)
            b = pts[:, s, seg + 1]  # (B, 2)

            ab = b - a  # (B, 2)
            ab_dot = (ab * ab).sum(dim=-1).clamp(min=1e-8)  # (B,)

            ap = pixels.unsqueeze(0) - a.unsqueeze(1)  # (B, P, 2)
            t = ((ap * ab.unsqueeze(1)).sum(dim=-1) / ab_dot.unsqueeze(1)).clamp(0, 1)  # (B, P)
            proj = a.unsqueeze(1) + t.unsqueeze(-1) * ab.unsqueeze(1)  # (B, P, 2)
            dist = (pixels.unsqueeze(0) - proj).norm(dim=-1)  # (B, P)

            # Update where valid and closer
            closer = dist < min_dist
            min_dist = torch.where(closer & valid.unsqueeze(1), dist, min_dist)

        # Soft coverage for this stroke: sigmoid(-(dist - half_width) * sharpness)
        coverage = torch.sigmoid(-(min_dist - s_w.unsqueeze(1) * 0.5) * 4.0)  # (B, P)
        total_intensity = total_intensity + coverage * s_active.unsqueeze(1)

    # Target: glyph_mask = 1.0 on glyph, 0.0 on background
    # Ideal: total_intensity = 1.0 on glyph (exactly one stroke), 0.0 on background
    target = glyph_small.reshape(B, P)  # (B, P)
    loss = ((total_intensity - target) ** 2).mean()

    return loss


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
    """Penalize duplicate/parallel strokes. Legacy per-sample version."""
    if len(stroke_points_list) < 2:
        return torch.tensor(0.0, device=stroke_points_list[0].device)

    device = stroke_points_list[0].device
    penalty = torch.tensor(0.0, device=device)

    for i in range(len(stroke_points_list)):
        for j in range(i + 1, len(stroke_points_list)):
            pts_a = stroke_points_list[i]
            pts_b = stroke_points_list[j]
            dists = torch.cdist(pts_a.unsqueeze(0), pts_b.unsqueeze(0))[0]
            min_dists_a = dists.min(dim=1).values
            close_a = (min_dists_a < proximity_threshold).float()
            frac_a = close_a.sum() / max(pts_a.shape[0], 1)
            min_dists_b = dists.min(dim=0).values
            close_b = (min_dists_b < proximity_threshold).float()
            frac_b = close_b.sum() / max(pts_b.shape[0], 1)
            overlap = torch.max(frac_a, frac_b)
            excess = (overlap - min_overlap_frac).clamp(min=0)
            penalty = penalty + excess ** 2

    return penalty


def smoothness_penalty(stroke_points_list: list) -> torch.Tensor:
    """Penalize high curvature. Legacy per-sample version."""
    device = stroke_points_list[0].device
    penalty = torch.tensor(0.0, device=device)
    for points in stroke_points_list:
        if points.shape[0] >= 3:
            d1 = points[1:] - points[:-1]
            d2 = d1[1:] - d1[:-1]
            penalty = penalty + (d2 ** 2).mean()
    return penalty


def overlap_penalty_batched(points, existence, n_points, canvas_size,
                            min_overlap_frac=0.3, proximity_threshold=8.0):
    """Batched overlap penalty. Loops over 28 stroke pairs, not B samples.

    Args:
        points: (B, S, N, 2) normalized coordinates.
        existence: (B, S) existence probabilities.
        n_points: (B, S) valid point counts.
        canvas_size: int.

    Returns:
        Scalar loss tensor averaged over batch.
    """
    B, S, N, _ = points.shape
    device = points.device
    pts = (points * canvas_size).clamp(0, canvas_size - 1)  # (B, S, N, 2)

    penalty = torch.zeros(B, device=device)

    for i in range(S):
        for j in range(i + 1, S):
            # Both strokes must exist
            both_exist = (existence[:, i] > 0.3) & (existence[:, j] > 0.3)  # (B,)
            if not both_exist.any():
                continue

            # Pairwise distances: (B, N, N)
            dists = torch.cdist(pts[:, i], pts[:, j])  # (B, N, N)

            # Mask invalid points with large distance
            valid_i = torch.arange(N, device=device) < n_points[:, i].unsqueeze(-1)  # (B, N)
            valid_j = torch.arange(N, device=device) < n_points[:, j].unsqueeze(-1)  # (B, N)
            dists = dists.masked_fill(~valid_i.unsqueeze(2), 1e6)
            dists = dists.masked_fill(~valid_j.unsqueeze(1), 1e6)

            # Fraction of stroke i close to stroke j
            min_dists_i = dists.min(dim=2).values  # (B, N)
            close_i = ((min_dists_i < proximity_threshold) & valid_i).float()
            frac_i = close_i.sum(dim=1) / n_points[:, i].float().clamp(min=1)  # (B,)

            # Fraction of stroke j close to stroke i
            min_dists_j = dists.min(dim=1).values  # (B, N)
            close_j = ((min_dists_j < proximity_threshold) & valid_j).float()
            frac_j = close_j.sum(dim=1) / n_points[:, j].float().clamp(min=1)  # (B,)

            overlap = torch.max(frac_i, frac_j)  # (B,)
            excess = (overlap - min_overlap_frac).clamp(min=0)
            penalty = penalty + (excess ** 2) * both_exist.float()

    return penalty.mean()


def reversal_penalty_batched(points, existence, n_points, canvas_size):
    """Penalize direction reversals (zigzag). Fully vectorized.

    Computes dot product between consecutive segments. When negative,
    the stroke reversed direction — this is the zigzag signature.

    Args:
        points: (B, S, N, 2) normalized coordinates.
        existence: (B, S) existence probabilities.
        n_points: (B, S) valid point counts.
        canvas_size: int.

    Returns:
        Scalar loss tensor averaged over batch.
    """
    B, S, N, _ = points.shape
    device = points.device
    pts = (points * canvas_size).clamp(0, canvas_size - 1)  # (B, S, N, 2)

    # Consecutive segment vectors
    d1 = pts[:, :, 1:] - pts[:, :, :-1]  # (B, S, N-1, 2)

    # Dot product between consecutive segments
    dots = (d1[:, :, :-1] * d1[:, :, 1:]).sum(dim=-1)  # (B, S, N-2)

    # Penalize negative dot products (direction reversals)
    reversal = (-dots).clamp(min=0)  # (B, S, N-2)

    # Normalize by segment lengths to make it scale-invariant
    len1 = d1[:, :, :-1].norm(dim=-1).clamp(min=1e-6)  # (B, S, N-2)
    len2 = d1[:, :, 1:].norm(dim=-1).clamp(min=1e-6)   # (B, S, N-2)
    reversal = reversal / (len1 * len2)  # cosine-based, 0 to 1

    # Mask: valid points and existing strokes
    valid = torch.arange(N - 2, device=device) < (n_points - 2).unsqueeze(-1)
    active = (existence > 0.3).unsqueeze(-1)
    mask = valid & active

    masked_rev = reversal * mask.float()
    per_stroke = masked_rev.sum(dim=2) / mask.float().sum(dim=2).clamp(min=1)
    per_sample = (per_stroke * (existence > 0.3).float()).sum(dim=1)
    return per_sample.mean()


def self_overlap_penalty_batched(points, existence, n_points, canvas_size,
                                 proximity_threshold=8.0, skip_neighbors=3):
    """Penalize strokes that loop back over themselves. Fully batched.

    For each stroke, compute pairwise distances between all points, masking
    out adjacent points (within skip_neighbors hops). Penalize points that
    are spatially close but far apart along the stroke.

    Args:
        points: (B, S, N, 2) normalized coordinates.
        existence: (B, S) existence probabilities.
        n_points: (B, S) valid point counts.
        canvas_size: int.
        proximity_threshold: distance in pixels below which points count as overlapping.
        skip_neighbors: ignore points within this many hops (they're naturally close).

    Returns:
        Scalar loss tensor averaged over batch.
    """
    B, S, N, _ = points.shape
    device = points.device
    pts = (points * canvas_size).clamp(0, canvas_size - 1)  # (B, S, N, 2)

    # Self-distance matrix for all strokes at once: (B, S, N, N)
    # Use reshape to batch S into B dimension for cdist
    pts_flat = pts.reshape(B * S, N, 2)  # (B*S, N, 2)
    self_dists = torch.cdist(pts_flat, pts_flat)  # (B*S, N, N)
    self_dists = self_dists.reshape(B, S, N, N)

    # Mask out adjacent points (diagonal band of width skip_neighbors)
    idx = torch.arange(N, device=device)
    neighbor_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= skip_neighbors  # (N, N)
    self_dists = self_dists.masked_fill(neighbor_mask.unsqueeze(0).unsqueeze(0), 1e6)

    # Mask invalid points
    valid = torch.arange(N, device=device) < n_points.unsqueeze(-1)  # (B, S, N)
    self_dists = self_dists.masked_fill(~valid.unsqueeze(3), 1e6)
    self_dists = self_dists.masked_fill(~valid.unsqueeze(2), 1e6)

    # For each point, how close is the nearest non-adjacent point?
    min_dists = self_dists.min(dim=3).values  # (B, S, N)

    # Soft penalty: points closer than threshold get penalized
    close = (proximity_threshold - min_dists).clamp(min=0) / proximity_threshold  # (B, S, N) in [0, 1]
    close = close * valid.float()

    # Average per stroke, weight by existence
    active = (existence > 0.3).float()  # (B, S)
    per_stroke = close.sum(dim=2) / valid.float().sum(dim=2).clamp(min=1)  # (B, S)
    per_sample = (per_stroke * active).sum(dim=1)  # (B,)

    return per_sample.mean()


def smoothness_penalty_batched(points, existence, n_points, canvas_size):
    """Batched smoothness penalty. Fully vectorized, no loops.

    Args:
        points: (B, S, N, 2) normalized coordinates.
        existence: (B, S) existence probabilities.
        n_points: (B, S) valid point counts.
        canvas_size: int.

    Returns:
        Scalar loss tensor averaged over batch.
    """
    B, S, N, _ = points.shape
    device = points.device
    pts = (points * canvas_size).clamp(0, canvas_size - 1)  # (B, S, N, 2)

    # Second-order finite differences: d2 = pts[:,2:] - 2*pts[:,1:-1] + pts[:,:-2]
    d1 = pts[:, :, 1:] - pts[:, :, :-1]       # (B, S, N-1, 2)
    d2 = d1[:, :, 1:] - d1[:, :, :-1]         # (B, S, N-2, 2)
    curvature = (d2 ** 2).sum(dim=-1)          # (B, S, N-2)

    # Mask: valid points and existing strokes
    valid = torch.arange(N - 2, device=device) < (n_points - 2).unsqueeze(-1)  # (B, S, N-2)
    active = (existence > 0.3).unsqueeze(-1)  # (B, S, 1)
    mask = valid & active                      # (B, S, N-2)

    masked_curv = curvature * mask.float()
    # Mean per stroke, then sum over strokes, mean over batch
    per_stroke = masked_curv.sum(dim=2) / mask.float().sum(dim=2).clamp(min=1)  # (B, S)
    per_sample = (per_stroke * (existence > 0.3).float()).sum(dim=1)  # (B,)
    return per_sample.mean()


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

    # Phase 1: Batched render via Triton (one kernel launch for all samples)
    rendered_all, all_stroke_pts = render_strokes_triton(
        model_output, glyph_mask, canvas_size,
    )

    # Phase 2: Batched coverage and outside losses (tensor ops, no loop)
    target = 1.0 - glyph_mask  # (B, H, W)
    sq_err = (rendered_all - target) ** 2
    weight_map = glyph_mask * 10.0 + (1.0 - glyph_mask)
    loss_coverage = (sq_err * weight_map).mean()

    bg_mask = 1.0 - glyph_mask
    stroke_pixels = (1.0 - rendered_all).clamp(min=0)
    outside_per_sample = (stroke_pixels * bg_mask).sum(dim=(1, 2))
    bg_total = bg_mask.sum(dim=(1, 2)).clamp(min=1)
    loss_outside = (outside_per_sample / bg_total).mean()

    # Phase 3: Batched overlap and smoothness (28 pair iterations, not B*28)
    points = model_output['points']
    existence = model_output['existence']
    pc_logits = model_output['point_count_logits']
    n_points = pc_logits.argmax(dim=-1) + 1
    n_points = n_points.clamp(min=2, max=points.shape[2])

    loss_overlap = overlap_penalty_batched(
        points, existence, n_points, canvas_size,
    )
    loss_smooth = smoothness_penalty_batched(
        points, existence, n_points, canvas_size,
    )
    loss_reversal = reversal_penalty_batched(
        points, existence, n_points, canvas_size,
    )
    loss_self_overlap = self_overlap_penalty_batched(
        points, existence, n_points, canvas_size,
    )

    if weights.get('unique_coverage', 0.0) > 0:
        loss_unique_cov = unique_coverage_loss(
            points, model_output['widths'], existence, n_points,
            glyph_mask, canvas_size,
        )
    else:
        loss_unique_cov = torch.tensor(0.0, device=device)

    loss_exist = existence_loss(model_output['existence'])

    total = (weights['coverage'] * loss_coverage +
             weights['outside'] * loss_outside +
             weights['overlap'] * loss_overlap +
             weights['smoothness'] * loss_smooth +
             weights.get('reversal', 0.0) * loss_reversal +
             weights.get('self_overlap', 0.0) * loss_self_overlap +
             weights.get('unique_coverage', 0.0) * loss_unique_cov +
             weights['existence'] * loss_exist)

    loss_dict = {
        'total': total.item(),
        'coverage': loss_coverage.item(),
        'outside': loss_outside.item(),
        'overlap': loss_overlap.item(),
        'smoothness': loss_smooth.item(),
        'reversal': loss_reversal.item(),
        'self_overlap': loss_self_overlap.item(),
        'unique_coverage': loss_unique_cov.item(),
        'existence': loss_exist.item(),
    }

    return total, loss_dict


def autoregressive_loss(model_output: dict, canvas_size: int = CANVAS_SIZE,
                        weights: dict = None) -> tuple:
    """Loss for autoregressive stroke prediction.

    Much simpler than parallel loss: just compare the final canvas to the target.
    The autoregressive structure naturally prevents overlap.

    Args:
        model_output: Dict from autoregressive StrokePredictor.forward() with:
            canvas_inv (B, R, R), target (B, R, R), plus per-step params.
        canvas_size: Canvas size in pixels.
        weights: Dict of loss weights.

    Returns:
        Tuple of (total_loss_tensor, loss_dict).
    """
    if weights is None:
        weights = {
            'canvas_mse': 1.0,
            'smoothness': 0.01,
            'reversal': 0.01,
            'exist_decay': 0.1,
        }

    canvas_inv = model_output['canvas_inv']  # (B, R, R), 1=blank, 0=ink
    target = model_output['target']          # (B, R, R), 1=glyph, 0=bg
    device = canvas_inv.device

    # 1. Canvas MSE: ink vs glyph target, weighted heavier on glyph pixels
    ink = 1.0 - canvas_inv  # 1=ink, 0=blank
    sq_err = (ink - target) ** 2
    weight_map = target * 10.0 + (1.0 - target) * 10.0
    loss_canvas = (sq_err * weight_map).mean()

    # 2. Smoothness and reversal on predicted strokes
    points = model_output['points']
    existence = model_output['existence']
    pc_logits = model_output['point_count_logits']
    n_points = pc_logits.argmax(dim=-1) + 1
    n_points = n_points.clamp(min=2, max=points.shape[2])

    loss_smooth = smoothness_penalty_batched(points, existence, n_points, canvas_size)
    loss_reversal = reversal_penalty_batched(points, existence, n_points, canvas_size)

    # 3. Total stroke length: sum of segment lengths across all strokes
    # Penalizes long, winding paths — encourages short, direct strokes
    B, S, N, _ = points.shape
    pts_scaled = points * canvas_size  # (B, S, N, 2)
    segments = pts_scaled[:, :, 1:] - pts_scaled[:, :, :-1]  # (B, S, N-1, 2)
    seg_lengths = segments.norm(dim=-1)  # (B, S, N-1)

    # Mask invalid segments and inactive strokes
    valid_seg = torch.arange(N - 1, device=device) < (n_points - 1).unsqueeze(-1)  # (B, S, N-1)
    active = (existence > 0.3).unsqueeze(-1)  # (B, S, 1)
    masked_lengths = seg_lengths * valid_seg.float() * active.float()

    # Total length per sample, averaged over batch
    loss_length = masked_lengths.sum(dim=(1, 2)).mean() / canvas_size  # normalize

    # 4. Existence decay: later strokes cost more to activate
    step_weights = torch.arange(existence.shape[1], device=device, dtype=torch.float32)
    loss_exist = (existence * step_weights.unsqueeze(0)).mean()

    total = (weights.get('canvas_mse', 1.0) * loss_canvas +
             weights.get('smoothness', 0.0) * loss_smooth +
             weights.get('reversal', 0.0) * loss_reversal +
             weights.get('stroke_length', 0.0) * loss_length +
             weights.get('exist_decay', 0.1) * loss_exist)

    loss_dict = {
        'total': total.item(),
        'canvas_mse': loss_canvas.item(),
        'smoothness': loss_smooth.item(),
        'reversal': loss_reversal.item(),
        'stroke_length': loss_length.item(),
        'exist_decay': loss_exist.item(),
    }

    return total, loss_dict
