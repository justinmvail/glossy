"""Triton-accelerated distance field stroke renderer.

Replaces pydiffvg for training. One GPU kernel launch renders ALL pixels
across ALL batch samples in parallel. No Python loops, no C++ bridge overhead.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from model import CANVAS_SIZE, MAX_STROKES, MAX_POINTS


@triton.jit
def _distance_field_kernel(
    # Inputs (all flattened)
    points_ptr,       # (B, S, N, 2) normalized coords
    widths_ptr,       # (B, S, N) per-point widths
    existence_ptr,    # (B, S)
    n_points_ptr,     # (B, S) int32
    # Outputs
    rendered_ptr,     # (B, P) float32
    closest_stroke_ptr,  # (B, P) int32 - for backward
    closest_seg_ptr,     # (B, P) int32 - for backward
    min_dist_ptr,        # (B, P) float32 - for backward
    # Dimensions
    total_pixels,     # B * P
    S: tl.constexpr,  # MAX_STROKES
    N: tl.constexpr,  # MAX_POINTS
    P,                # render_size * render_size
    render_size,
    canvas_size_float,
    sharpness,
    exist_thresh,
):
    pid = tl.program_id(0)
    if pid >= total_pixels:
        return

    batch_idx = pid // P
    pixel_idx = pid % P

    scale = canvas_size_float / render_size
    py = (pixel_idx // render_size).to(tl.float32) * scale + scale * 0.5
    px = (pixel_idx % render_size).to(tl.float32) * scale + scale * 0.5

    best_dist = tl.full([], 1e6, dtype=tl.float32)
    best_stroke = tl.full([], 0, dtype=tl.int32)
    best_seg = tl.full([], 0, dtype=tl.int32)

    for s in tl.static_range(S):
        exist_val = tl.load(existence_ptr + batch_idx * S + s)
        if exist_val >= exist_thresh:
            n_pts = tl.load(n_points_ptr + batch_idx * S + s)

            # Base pointers for this stroke
            pts_base = (batch_idx * S * N + s * N) * 2
            widths_base = batch_idx * S * N + s * N

            for seg in tl.static_range(N - 1):
                # Use mask instead of break
                if seg < n_pts - 1:
                    # Load segment endpoints (normalized) and scale
                    ax = tl.load(points_ptr + pts_base + seg * 2) * canvas_size_float
                    ay = tl.load(points_ptr + pts_base + seg * 2 + 1) * canvas_size_float
                    bx = tl.load(points_ptr + pts_base + (seg + 1) * 2) * canvas_size_float
                    by = tl.load(points_ptr + pts_base + (seg + 1) * 2 + 1) * canvas_size_float

                    # Clamp to canvas
                    ax = tl.minimum(tl.maximum(ax, 0.0), canvas_size_float - 1.0)
                    ay = tl.minimum(tl.maximum(ay, 0.0), canvas_size_float - 1.0)
                    bx = tl.minimum(tl.maximum(bx, 0.0), canvas_size_float - 1.0)
                    by = tl.minimum(tl.maximum(by, 0.0), canvas_size_float - 1.0)

                    # Point-to-segment distance
                    abx = bx - ax
                    aby = by - ay
                    apx = px - ax
                    apy = py - ay

                    ab_dot = abx * abx + aby * aby
                    t = (apx * abx + apy * aby) / tl.maximum(ab_dot, 1e-8)
                    t = tl.minimum(tl.maximum(t, 0.0), 1.0)

                    proj_x = ax + t * abx
                    proj_y = ay + t * aby

                    dx = px - proj_x
                    dy = py - proj_y
                    dist = tl.sqrt(dx * dx + dy * dy + 1e-6)
                    # Interpolate width between segment endpoints
                    w_a = tl.load(widths_ptr + widths_base + seg)
                    w_b = tl.load(widths_ptr + widths_base + seg + 1)
                    width_interp = w_a * (1.0 - t) + w_b * t
                    dist = dist - width_interp * 0.5

                    if dist < best_dist:
                        best_dist = dist
                        best_stroke = s
                        best_seg = seg

    # Clamp and sigmoid
    best_dist = tl.minimum(tl.maximum(best_dist, -50.0), 50.0)
    rendered = 1.0 / (1.0 + tl.exp(-sharpness * best_dist))

    out_idx = batch_idx * P + pixel_idx
    tl.store(rendered_ptr + out_idx, rendered)
    tl.store(min_dist_ptr + out_idx, best_dist)
    tl.store(closest_stroke_ptr + out_idx, best_stroke)
    tl.store(closest_seg_ptr + out_idx, best_seg)


class DistanceFieldRender(torch.autograd.Function):
    """Custom autograd function wrapping Triton forward + PyTorch backward."""

    @staticmethod
    def forward(ctx, points, widths, existence, n_points,
                canvas_size, render_size, sharpness, exist_thresh):
        """
        Args:
            points: (B, S, N, 2) normalized stroke coordinates.
            widths: (B, S) stroke widths in pixels.
            existence: (B, S) existence probabilities.
            n_points: (B, S) int32, valid point count per stroke.
            canvas_size: int, full canvas size.
            render_size: int, resolution to render at.
            sharpness: float, sigmoid sharpness.
            exist_thresh: float, existence threshold.

        Returns:
            rendered: (B, render_size, render_size) soft-rendered images.
        """
        B, S, N, _ = points.shape
        P = render_size * render_size
        device = points.device

        rendered = torch.empty(B, P, device=device, dtype=torch.float32)
        min_dist = torch.empty(B, P, device=device, dtype=torch.float32)
        closest_stroke = torch.empty(B, P, device=device, dtype=torch.int32)
        closest_seg = torch.empty(B, P, device=device, dtype=torch.int32)

        # Ensure contiguous
        points_c = points.contiguous()
        widths_c = widths.contiguous()
        existence_c = existence.contiguous()
        n_points_c = n_points.contiguous().to(torch.int32)

        total_pixels = B * P
        grid = (total_pixels,)

        _distance_field_kernel[grid](
            points_c, widths_c, existence_c, n_points_c,
            rendered, closest_stroke, closest_seg, min_dist,
            total_pixels, S, N, P, render_size,
            float(canvas_size), float(sharpness), float(exist_thresh),
        )

        ctx.save_for_backward(points_c, widths_c, existence_c, n_points_c,
                              min_dist, closest_stroke, closest_seg)
        ctx.canvas_size = canvas_size
        ctx.render_size = render_size
        ctx.sharpness = sharpness

        return rendered.reshape(B, render_size, render_size)

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradients w.r.t. points and widths using saved closest info."""
        (points, widths, existence, n_points,
         min_dist, closest_stroke, closest_seg) = ctx.saved_tensors
        canvas_size = ctx.canvas_size
        render_size = ctx.render_size
        sharpness = ctx.sharpness

        B, S, N, _ = points.shape
        P = render_size * render_size
        device = points.device

        grad_flat = grad_output.reshape(B, P)

        # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        sig = torch.sigmoid(sharpness * min_dist)
        grad_min_dist = grad_flat * sig * (1.0 - sig) * sharpness  # (B, P)

        # Pixel coordinates
        scale = canvas_size / render_size
        pixel_indices = torch.arange(P, device=device)
        px = (pixel_indices % render_size).float() * scale + scale * 0.5
        py = (pixel_indices // render_size).float() * scale + scale * 0.5
        # (P, 2) -> (1, P, 2)
        pixel_coords = torch.stack([px, py], dim=-1).unsqueeze(0).expand(B, -1, -1)

        # Get closest segment endpoints for each pixel
        cs = closest_stroke.long()  # (B, P)
        cseg = closest_seg.long()   # (B, P)
        cseg_b = (cseg + 1).clamp(max=N - 1)

        pts_scaled = (points * canvas_size).clamp(0, canvas_size - 1)  # (B, S, N, 2)

        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, P)  # (B, P)

        a = pts_scaled[b_idx, cs, cseg]       # (B, P, 2)
        b_pt = pts_scaled[b_idx, cs, cseg_b]  # (B, P, 2)

        # Recompute projection
        ab = b_pt - a                                    # (B, P, 2)
        ap = pixel_coords - a                            # (B, P, 2)
        ab_dot = (ab * ab).sum(-1).clamp(min=1e-8)       # (B, P)
        t = ((ap * ab).sum(-1) / ab_dot).clamp(0, 1)     # (B, P)

        proj = a + t.unsqueeze(-1) * ab                   # (B, P, 2)
        diff = pixel_coords - proj                         # (B, P, 2)
        dist = (diff * diff).sum(-1).add(1e-6).sqrt()     # (B, P)

        # d(dist)/d(proj) = -(pixel - proj) / dist
        # d(proj)/d(a) = (1-t) * I  (when t is clamped to [0,1])
        # d(proj)/d(b) = t * I
        # d(min_dist)/d(a) = d(dist)/d(a) = -(1-t) * (pixel-proj)/dist
        # d(min_dist)/d(b) = d(dist)/d(b) = -t * (pixel-proj)/dist

        grad_dist = grad_min_dist / dist  # (B, P), chain rule factor
        grad_diff = grad_dist.unsqueeze(-1) * diff  # (B, P, 2) -> d(dist)/d(diff) * upstream

        # grad w.r.t. proj (negative because diff = pixel - proj)
        grad_proj = -grad_diff  # (B, P, 2)

        grad_a_local = grad_proj * (1 - t).unsqueeze(-1)     # (B, P, 2)
        grad_b_local = grad_proj * t.unsqueeze(-1)            # (B, P, 2)

        # Scale gradients from canvas coords back to normalized coords
        grad_a_local = grad_a_local * canvas_size
        grad_b_local = grad_b_local * canvas_size

        # Scatter gradients back to points tensor using index_put_
        grad_points = torch.zeros_like(points)  # (B, S, N, 2)

        batch_indices = b_idx.reshape(-1)
        stroke_indices = cs.reshape(-1)
        seg_a_indices = cseg.reshape(-1)
        seg_b_indices = (cseg + 1).clamp(max=N - 1).reshape(-1)

        grad_a_flat = grad_a_local.reshape(-1, 2)
        grad_b_flat = grad_b_local.reshape(-1, 2)

        # Accumulate gradients for endpoint a and b
        grad_points.index_put_(
            (batch_indices, stroke_indices, seg_a_indices),
            grad_a_flat, accumulate=True,
        )
        grad_points.index_put_(
            (batch_indices, stroke_indices, seg_b_indices),
            grad_b_flat, accumulate=True,
        )

        # Width gradient: d(min_dist)/d(width_interp) = -0.5
        # width_interp = w_a * (1-t) + w_b * t
        # d(min_dist)/d(w_a) = -0.5 * (1-t)
        # d(min_dist)/d(w_b) = -0.5 * t
        grad_widths = torch.zeros_like(widths)  # (B, S, N)
        grad_w_a = grad_min_dist * (-0.5) * (1 - t)  # (B, P)
        grad_w_b = grad_min_dist * (-0.5) * t          # (B, P)

        # Scatter to per-point widths: (B, S, N) indexed by (batch, stroke, point)
        # Flatten to (B, S*N) for scatter_add_
        grad_widths_flat = grad_widths.reshape(B, -1)  # (B, S*N)
        scatter_idx_a = (cs * N + cseg).clamp(max=S * N - 1)          # (B, P)
        scatter_idx_b = (cs * N + cseg_b).clamp(max=S * N - 1)        # (B, P)
        grad_widths_flat.scatter_add_(1, scatter_idx_a, grad_w_a)
        grad_widths_flat.scatter_add_(1, scatter_idx_b, grad_w_b)
        grad_widths = grad_widths_flat.reshape(B, S, N)

        return grad_points, grad_widths, None, None, None, None, None, None


def render_single_stroke_triton(points, widths, n_points, canvas_size=CANVAS_SIZE,
                                render_size=56, sharpness=4.0):
    """Render a single stroke per batch item using the Triton kernel.

    Thin wrapper that reshapes single-stroke inputs to match the kernel's
    expected (B, S, ...) layout with S=1.

    Args:
        points: (B, N, 2) normalized coordinates for one stroke.
        widths: (B, N) per-point widths in pixels.
        n_points: (B,) valid point counts.
        canvas_size: int.
        render_size: int.
        sharpness: float.

    Returns:
        rendered: (B, render_size, render_size) where 1=bg, 0=ink.
    """
    B = points.shape[0]
    device = points.device

    # Reshape to (B, 1, N, 2) / (B, 1, N) for the kernel
    points_4d = points.unsqueeze(1)          # (B, 1, N, 2)
    widths_3d = widths.unsqueeze(1)          # (B, 1, N)
    existence = torch.ones(B, 1, device=device)
    n_pts_2d = n_points.unsqueeze(1) if n_points.dim() == 1 else n_points

    return DistanceFieldRender.apply(
        points_4d, widths_3d, existence, n_pts_2d,
        canvas_size, render_size, sharpness, 0.0,  # exist_thresh=0, always render
    )


def render_strokes_triton(model_output, glyph_masks, canvas_size=CANVAS_SIZE,
                          render_size=56, sharpness=4.0, exist_thresh=0.3):
    """Render batch using Triton kernel. Drop-in replacement for the pydiffvg loop.

    Args:
        model_output: Dict from StrokePredictor.forward().
        glyph_masks: (B, H, W) binary glyph masks.
        canvas_size: Canvas size in pixels.
        render_size: Resolution to render at.
        sharpness: Sigmoid sharpness.
        exist_thresh: Existence threshold.

    Returns:
        rendered: (B, H, W) soft-rendered canvases.
        all_stroke_pts: List of lists of point tensors (for overlap/smoothness).
    """
    points = model_output['points']                # (B, S, N, 2)
    widths = model_output['widths']                 # (B, S)
    existence = model_output['existence']           # (B, S)
    pc_logits = model_output['point_count_logits']  # (B, S, N)

    B = points.shape[0]
    device = points.device

    # Compute point counts
    n_points = pc_logits.argmax(dim=-1) + 1  # (B, S)
    n_points = n_points.clamp(min=2, max=MAX_POINTS)

    # Triton forward
    rendered_small = DistanceFieldRender.apply(
        points, widths, existence, n_points,
        canvas_size, render_size, sharpness, exist_thresh,
    )

    # Upsample to full canvas size
    if render_size != canvas_size:
        rendered = F.interpolate(
            rendered_small.unsqueeze(1),
            size=(canvas_size, canvas_size),
            mode='bilinear', align_corners=False,
        ).squeeze(1)
    else:
        rendered = rendered_small

    # Extract stroke point lists for overlap/smoothness losses
    all_stroke_pts = []
    pts_scaled = (points * canvas_size).clamp(0, canvas_size - 1)
    for b in range(B):
        sample_pts = []
        for s in range(points.shape[1]):
            if existence[b, s].item() >= exist_thresh:
                n = int(n_points[b, s].item())
                sample_pts.append(pts_scaled[b, s, :n])
        all_stroke_pts.append(sample_pts)

    return rendered, all_stroke_pts
