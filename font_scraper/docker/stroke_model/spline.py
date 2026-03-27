"""Catmull-Rom spline interpolation with per-point sharpness for stroke rendering.

Expands N control points into M = (N-1)*subdivisions + 1 smooth interpolated points.
Sharpness per point controls corner sharpness: 0=smooth curve, 1=sharp corner.
"""

import torch


def catmull_rom_segment(p0, p1, p2, p3, t):
    """Catmull-Rom interpolation between p1 and p2.

    Args:
        p0, p1, p2, p3: (B, 2) control points.
        t: (S,) parameter values in [0, 1].

    Returns:
        (B, S, 2) interpolated positions.
    """
    # t: (S,) -> (1, S, 1) for broadcasting with (B, 1, 2)
    t = t.to(p0.device).unsqueeze(0).unsqueeze(-1)  # (1, S, 1)
    t2 = t * t
    t3 = t2 * t

    # p's: (B, 2) -> (B, 1, 2)
    p0 = p0.unsqueeze(1)
    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)
    p3 = p3.unsqueeze(1)

    # q(t) = 0.5 * [(2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t² + (-p0+3*p1-3*p2+p3)*t³]
    result = 0.5 * (
        2 * p1 +
        (-p0 + p2) * t +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
        (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )  # (B, S, 2)
    return result


def interpolate_stroke(points, widths, sharpness, subdivisions=4):
    """Expand control points into smooth spline with per-point sharpness.

    Args:
        points: (B, N, 2) normalized control points.
        widths: (B, N) per-point widths.
        sharpness: (B, N) per-point sharpness in [0, 1].
            0 = full smooth Catmull-Rom curve
            1 = sharp corner (linear interpolation, current behavior)
        subdivisions: int, points to insert between each pair of control points.

    Returns:
        interp_points: (B, M, 2) where M = (N-1)*subdivisions + 1.
        interp_widths: (B, M) interpolated widths.
    """
    B, N, _ = points.shape
    device = points.device

    # t values: exclude 0 (to avoid duplicates), include 1
    t_vals = torch.linspace(1.0 / subdivisions, 1.0, subdivisions, device=device)

    # Phantom control points for endpoints via reflection
    p_start = 2 * points[:, 0:1] - points[:, 1:2]   # (B, 1, 2)
    p_end = 2 * points[:, -1:] - points[:, -2:-1]    # (B, 1, 2)
    ext_points = torch.cat([p_start, points, p_end], dim=1)  # (B, N+2, 2)

    # Extended sharpness (endpoints default to sharp)
    s_start = torch.ones(B, 1, device=device)
    s_end = torch.ones(B, 1, device=device)
    ext_sharp = torch.cat([s_start, sharpness, s_end], dim=1)  # (B, N+2)

    all_segments = []

    for i in range(N - 1):
        # Four control points (indices shifted by 1 due to phantom start point)
        p0_raw = ext_points[:, i]      # (B, 2)
        p1 = ext_points[:, i + 1]      # (B, 2)
        p2 = ext_points[:, i + 2]      # (B, 2)
        p3_raw = ext_points[:, i + 3]  # (B, 2)

        # Apply sharpness: lerp phantom points toward their adjacent control point
        s1 = ext_sharp[:, i + 1].unsqueeze(-1)  # (B, 1)
        s2 = ext_sharp[:, i + 2].unsqueeze(-1)  # (B, 1)
        p0_eff = torch.lerp(p0_raw, p1, s1)     # (B, 2)
        p3_eff = torch.lerp(p3_raw, p2, s2)     # (B, 2)

        seg_pts = catmull_rom_segment(p0_eff, p1, p2, p3_eff, t_vals)  # (B, sub, 2)
        all_segments.append(seg_pts)

    # Prepend the first control point
    first_pt = points[:, 0:1]  # (B, 1, 2)
    interp_points = torch.cat([first_pt] + all_segments, dim=1)  # (B, M, 2)

    # Interpolate widths linearly
    M = interp_points.shape[1]
    # Vectorized: build all widths at once
    w_segments = []
    for i in range(N - 1):
        w_start = widths[:, i:i + 1]    # (B, 1)
        w_end = widths[:, i + 1:i + 2]  # (B, 1)
        t_row = t_vals.unsqueeze(0)      # (1, sub)
        seg_w = w_start * (1 - t_row) + w_end * t_row  # (B, sub)
        w_segments.append(seg_w)

    interp_widths = torch.cat([widths[:, 0:1]] + w_segments, dim=1)  # (B, M)
    return interp_points, interp_widths


def interpolate_stroke_batch(points, widths, sharpness, n_points, subdivisions=4):
    """Batch interpolation over all strokes.

    Args:
        points: (B, S, N, 2) control points.
        widths: (B, S, N) per-point widths.
        sharpness: (B, S, N) per-point sharpness.
        n_points: (B, S) valid point counts.
        subdivisions: int.

    Returns:
        interp_points: (B, S, M, 2) where M = (N-1)*subdivisions + 1.
        interp_widths: (B, S, M).
        interp_n_points: (B, S) new valid point counts.
    """
    B, S, N, _ = points.shape

    flat_pts = points.reshape(B * S, N, 2)
    flat_w = widths.reshape(B * S, N)
    flat_s = sharpness.reshape(B * S, N)

    interp_pts, interp_w = interpolate_stroke(flat_pts, flat_w, flat_s, subdivisions)

    M = interp_pts.shape[1]
    interp_pts = interp_pts.reshape(B, S, M, 2)
    interp_w = interp_w.reshape(B, S, M)
    interp_n = (n_points - 1) * subdivisions + 1

    return interp_pts, interp_w, interp_n
