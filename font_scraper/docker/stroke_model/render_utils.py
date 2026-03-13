"""pydiffvg rendering helpers for differentiable stroke rendering.

Provides functions to render predicted strokes using pydiffvg and compute
coverage scores against glyph masks. Reuses patterns from optimize_strokes.py.
"""

import torch
import pydiffvg

from model import CANVAS_SIZE


def render_strokes(stroke_points_list: list, stroke_widths: list,
                   canvas_size: int = CANVAS_SIZE,
                   device: torch.device = None) -> torch.Tensor:
    """Render all strokes into a single grayscale image using pydiffvg.

    Args:
        stroke_points_list: List of (N_i, 2) tensors, each a polyline stroke.
        stroke_widths: List of scalar tensors, one width per stroke.
        canvas_size: Size of the square canvas in pixels.
        device: Torch device.

    Returns:
        (canvas_size, canvas_size) grayscale tensor. White=1.0 background,
        black=0.0 where strokes render.
    """
    if device is None:
        device = stroke_points_list[0].device

    shapes = []
    shape_groups = []

    for i, (points, width) in enumerate(zip(stroke_points_list, stroke_widths)):
        n_pts = points.shape[0]
        num_control_points = torch.zeros(n_pts - 1, dtype=torch.int32, device=device)

        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=width,
            is_closed=False,
        )
        shapes.append(path)

        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=None,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        shape_groups.append(path_group)

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_size, canvas_size, shapes, shape_groups,
    )
    render = pydiffvg.RenderFunction.apply

    # White background (required — otherwise canvas is all black)
    bg = torch.ones(canvas_size, canvas_size, 4, device=device)
    img = render(
        canvas_size, canvas_size,
        2, 2,  # num_samples_x, num_samples_y (anti-aliasing)
        0,     # seed
        bg,
        *scene_args,
    )

    # Convert RGBA to grayscale: mean of RGB channels
    return img[:, :, :3].mean(dim=2)


def compute_score(rendered: torch.Tensor, glyph_mask: torch.Tensor) -> float:
    """Compute coverage score: what fraction of the glyph is covered by strokes.

    Args:
        rendered: (H, W) grayscale rendered image (1.0=white, 0.0=black).
        glyph_mask: (H, W) binary mask (1.0=glyph, 0.0=background).

    Returns:
        Score in [0, 1]. Higher is better. Penalizes both uncovered glyph
        pixels and stroke pixels outside the glyph.
    """
    stroke_mask = (rendered < 0.5).float()

    # Coverage: fraction of glyph pixels covered by strokes
    glyph_pixels = glyph_mask.sum().clamp(min=1)
    covered = (stroke_mask * glyph_mask).sum()
    coverage = covered / glyph_pixels

    # Overshoot: fraction of stroke pixels outside glyph
    bg_mask = 1.0 - glyph_mask
    stroke_pixels = stroke_mask.sum().clamp(min=1)
    outside = (stroke_mask * bg_mask).sum()
    overshoot = outside / stroke_pixels

    # Score: coverage minus overshoot penalty
    score = (coverage - 0.5 * overshoot).clamp(0, 1)
    return float(score.item())


def model_output_to_render_inputs(points: torch.Tensor, widths: torch.Tensor,
                                  existence: torch.Tensor,
                                  point_count_logits: torch.Tensor,
                                  canvas_size: int = CANVAS_SIZE,
                                  existence_threshold: float = 0.3):
    """Convert model outputs to pydiffvg render inputs.

    Selects existing strokes and trims to valid point counts, scaling
    coordinates from [0, 1] to canvas pixel space.

    Args:
        points: (MAX_STROKES, MAX_POINTS, 2) normalized coordinates.
        widths: (MAX_STROKES,) stroke widths.
        existence: (MAX_STROKES,) existence probabilities.
        point_count_logits: (MAX_STROKES, MAX_POINTS) point count logits.
        canvas_size: Canvas size in pixels.
        existence_threshold: Minimum probability to include a stroke.

    Returns:
        Tuple of (stroke_points_list, stroke_widths_list, active_mask):
            - stroke_points_list: List of (N_i, 2) tensors in pixel coords.
            - stroke_widths_list: List of scalar tensors.
            - active_mask: (MAX_STROKES,) boolean tensor of active strokes.
    """
    device = points.device
    stroke_points_list = []
    stroke_widths_list = []
    active_indices = []

    for i in range(points.shape[0]):
        if existence[i].item() < existence_threshold:
            continue

        # Point count from softmax argmax
        n_points = torch.argmax(point_count_logits[i]).item() + 1
        n_points = max(2, min(n_points, points.shape[1]))

        # Scale to canvas and clamp
        pts = points[i, :n_points] * canvas_size
        pts = pts.clamp(0, canvas_size - 1)

        stroke_points_list.append(pts)
        stroke_widths_list.append(widths[i])
        active_indices.append(i)

    active_mask = torch.zeros(points.shape[0], dtype=torch.bool, device=device)
    if active_indices:
        active_mask[active_indices] = True

    return stroke_points_list, stroke_widths_list, active_mask
