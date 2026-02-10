#!/usr/bin/env python3
"""DiffVG-based differentiable stroke optimizer.

This module implements gradient-based optimization of polyline strokes to match
font glyph masks using pydiffvg, a differentiable vector graphics rasterizer.
It is designed to run inside a Docker container with GPU support.

The optimizer uses Adam gradient descent to adjust stroke point positions and
widths, minimizing a multi-component loss function that balances glyph coverage,
stroke smoothness, and topology preservation.

Pipeline:
    1. Load font and render target glyph as binary mask
    2. Initialize learnable stroke parameters (points, widths)
    3. Iteratively:
       a. Render strokes using pydiffvg
       b. Compute loss against target mask
       c. Backpropagate gradients and update parameters
    4. Output optimized strokes as JSON

Docker Integration:
    This script is called by the DiffVG Docker container with a JSON config file:

    Input JSON format::

        {
            "font_path": "/fonts/MyFont.ttf",
            "char": "A",
            "canvas_size": 224,
            "num_iterations": 500,
            "stroke_width": 8.0,
            "initial_strokes": [[[x1, y1], [x2, y2], ...], ...],
            "thin_iterations": 0
        }

    Output JSON format (printed to stdout)::

        {
            "strokes": [[[x1, y1], [x2, y2], ...], ...],
            "score": 0.85,
            "iterations": 500,
            "final_loss": 0.123,
            "elapsed": 45.2
        }

Usage:
    python3 optimize_strokes.py /app/input.json

Dependencies:
    - pydiffvg (with PyTorch backend)
    - PyTorch with CUDA support (optional, falls back to CPU)
    - Pillow, NumPy, scikit-image
"""

import json
import sys
import time

import numpy as np
import pydiffvg
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import thin


def render_glyph_mask(font_path, char, canvas_size=224, thin_iterations=0):
    """Render a single character glyph as a binary mask.

    Renders the specified character using the given font, automatically sizing
    the font to fit within 90% of the canvas while maintaining aspect ratio.
    The result is a boolean mask where True indicates glyph pixels.

    Args:
        font_path: Path to the font file (.ttf, .otf).
        char: Single character to render.
        canvas_size: Size of the square output canvas in pixels.
        thin_iterations: Number of morphological thinning iterations to apply.
            Thinning reduces thick strokes to single-pixel skeletons while
            preserving topology. Useful for thick fonts where junctions create
            large blobs. Default 0 (no thinning).

    Returns:
        A 2D boolean numpy array of shape (canvas_size, canvas_size) where
        True indicates glyph pixels and False indicates background.
        Returns None if the glyph cannot be rendered (e.g., character not
        in font, font file not found).

    Example:
        >>> mask = render_glyph_mask('/fonts/Arial.ttf', 'A', canvas_size=224)
        >>> print(f"Glyph covers {mask.sum()} pixels")
    """
    img = Image.new('L', (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)

    # Try font sizes from large to small until it fits
    for font_size in range(200, 20, -5):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            continue
        bbox = font.getbbox(char)
        if bbox is None:
            continue
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= 0 or th <= 0:
            continue
        # Check if it fits within 90% of canvas
        max_dim = canvas_size * 0.9
        if tw <= max_dim and th <= max_dim:
            x = (canvas_size - tw) / 2 - bbox[0]
            y = (canvas_size - th) / 2 - bbox[1]
            draw.text((x, y), char, fill=0, font=font)
            arr = np.array(img)
            mask = arr < 128  # Boolean mask

            # Apply topology-preserving thinning
            if thin_iterations > 0:
                mask = thin(mask, max_num_iter=thin_iterations)

            return mask
    return None


def build_scene(stroke_points_list, stroke_widths, canvas_size, device):
    """Build a pydiffvg scene from stroke point tensors.

    Constructs pydiffvg Path and ShapeGroup objects from learnable stroke
    parameters. Each stroke is rendered as a polyline (no bezier curves)
    with black color on a white background.

    Args:
        stroke_points_list: List of (N, 2) PyTorch tensors containing learnable
            point positions for each stroke. These tensors should have
            requires_grad=True for optimization.
        stroke_widths: List of scalar PyTorch tensors for learnable stroke
            widths. One width per stroke.
        canvas_size: Integer size of the square canvas.
        device: PyTorch device (cuda or cpu) for tensor operations.

    Returns:
        A tuple of (shapes, shape_groups) suitable for pydiffvg rendering:
        - shapes: List of pydiffvg.Path objects
        - shape_groups: List of pydiffvg.ShapeGroup objects with black stroke

    Example:
        >>> points = [torch.tensor([[10., 20.], [30., 40.]], requires_grad=True)]
        >>> widths = [torch.tensor(5.0, requires_grad=True)]
        >>> shapes, groups = build_scene(points, widths, 224, 'cuda')
    """
    shapes = []
    shape_groups = []

    for i, (points, width) in enumerate(zip(stroke_points_list, stroke_widths)):
        n_pts = points.shape[0]
        # num_control_points = 0 for each segment means straight lines (polyline)
        num_control_points = torch.zeros(n_pts - 1, dtype=torch.int32, device=device)
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=width,
            is_closed=False,
        )
        shapes.append(path)

        # Black stroke, no fill
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=None,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        shape_groups.append(path_group)

    return shapes, shape_groups


def render_scene(shapes, shape_groups, canvas_size, device):
    """Render pydiffvg shapes to a grayscale canvas.

    Uses pydiffvg's differentiable rendering pipeline to rasterize vector
    shapes onto a canvas. The rendering is differentiable, allowing gradients
    to flow back through the rendering process to update stroke parameters.

    Args:
        shapes: List of pydiffvg.Path objects from build_scene().
        shape_groups: List of pydiffvg.ShapeGroup objects from build_scene().
        canvas_size: Integer size of the square output canvas.
        device: PyTorch device for the output tensor.

    Returns:
        A 2D PyTorch tensor of shape (canvas_size, canvas_size) with float
        values in [0, 1]. Values near 0 indicate stroke pixels (black),
        values near 1 indicate background (white).

    Note:
        The background must be explicitly set to white (ones), otherwise
        pydiffvg renders on a black (zeros) background, which inverts the
        expected semantics.
    """
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_size, canvas_size, shapes, shape_groups,
    )
    render = pydiffvg.RenderFunction.apply
    # White background is essential — without it the canvas is all black (zeros)
    bg = torch.ones(canvas_size, canvas_size, 4, device=device)
    img = render(
        canvas_size,  # width
        canvas_size,  # height
        2,            # num_samples_x
        2,            # num_samples_y
        0,            # seed
        bg,           # background_image (white)
        *scene_args,
    )
    # img shape: (canvas_size, canvas_size, 4) RGBA
    # Convert to grayscale: take the mean of RGB channels
    return img[:, :, :3].mean(dim=2)


def compute_loss(rendered, target, stroke_points_list, stroke_widths=None,
                 initial_points_list=None, glyph_weight=10.0, outside_weight=5.0,
                 smoothness_weight=0.001, anchor_weight=2.0, coverage_weight=5.0):
    """Compute multi-component optimization loss.

    The loss function balances several objectives:
    1. MSE loss: Match rendered strokes to target glyph (weighted higher inside glyph)
    2. Outside penalty: Discourage strokes from rendering outside the glyph
    3. Coverage penalty: Encourage strokes to cover all glyph pixels
    4. Smoothness: Penalize high curvature to produce smooth strokes
    5. Anchor loss: Preserve topology by limiting deviation from initial positions

    Args:
        rendered: (H, W) float tensor from render_scene(). Values in [0, 1]
            where 0 = stroke (black), 1 = background (white).
        target: (H, W) float tensor target mask. Values in [0, 1] where
            0 = glyph (black), 1 = background (white).
        stroke_points_list: List of (N, 2) point tensors for smoothness
            regularization.
        stroke_widths: List of scalar width tensors (currently unused in loss,
            reserved for future width regularization).
        initial_points_list: List of (N, 2) tensors with original point
            positions. Used for anchor loss to preserve stroke topology.
            If None, anchor loss is zero.
        glyph_weight: Extra weight multiplier for MSE loss inside the glyph
            region. Higher values prioritize coverage over precision.
        outside_weight: Weight for the outside penalty term. Higher values
            more strongly discourage strokes from extending beyond the glyph.
        smoothness_weight: Weight for the curvature penalty. Higher values
            produce smoother strokes but may reduce accuracy.
        anchor_weight: Weight for topology preservation. Higher values keep
            strokes closer to their initial positions.
        coverage_weight: Weight for the uncovered pixel penalty. Higher values
            prioritize complete coverage.

    Returns:
        A scalar PyTorch tensor representing the total loss. This tensor is
        differentiable and can be used with loss.backward() for optimization.

    Example:
        >>> loss = compute_loss(rendered, target, stroke_points,
        ...                     initial_points_list=initial_points)
        >>> loss.backward()
        >>> optimizer.step()
    """
    glyph_mask = (target < 0.5).float()   # 1 where glyph, 0 where bg
    bg_mask = 1.0 - glyph_mask
    stroke_mask = (rendered < 0.5).float()  # 1 where stroke renders

    # Per-pixel squared error
    sq_err = (rendered - target) ** 2

    # Weighted MSE: glyph pixels weighted much higher
    weighted_err = sq_err * (glyph_mask * glyph_weight + bg_mask)
    loss_mse = weighted_err.mean()

    # Outside penalty: strokes rendering in background area
    stroke_pixels = (1.0 - rendered).clamp(min=0)  # how dark each pixel is
    outside_penalty = (stroke_pixels * bg_mask).sum() / bg_mask.sum().clamp(min=1)

    # Coverage penalty: glyph pixels not covered by strokes
    uncovered = glyph_mask * (1.0 - stroke_mask)  # glyph pixels without strokes
    coverage_penalty = uncovered.sum() / glyph_mask.sum().clamp(min=1)

    # Smoothness: penalize high curvature in polylines
    smoothness = torch.tensor(0.0, device=rendered.device)
    for points in stroke_points_list:
        if points.shape[0] >= 3:
            d1 = points[1:] - points[:-1]
            d2 = d1[1:] - d1[:-1]
            smoothness = smoothness + (d2 ** 2).mean()

    # Anchor: penalize deviation from initial positions to preserve topology
    anchor_loss = torch.tensor(0.0, device=rendered.device)
    if initial_points_list is not None:
        for points, initial in zip(stroke_points_list, initial_points_list):
            anchor_loss = anchor_loss + ((points - initial) ** 2).mean()

    return (loss_mse + outside_weight * outside_penalty +
            coverage_weight * coverage_penalty +
            smoothness_weight * smoothness + anchor_weight * anchor_loss)


def compute_score(rendered, target):
    """Compute a coverage score for evaluation.

    Calculates how well the rendered strokes cover the target glyph, with a
    penalty for strokes that extend outside the glyph boundary. This score
    is used for evaluation and early stopping, not for gradient optimization.

    The score formula is: coverage - 0.5 * overshoot
    - coverage: Fraction of glyph pixels covered by strokes
    - overshoot: Fraction of stroke pixels that are outside the glyph

    Args:
        rendered: (H, W) float tensor from render_scene(). Values in [0, 1]
            where 0 = stroke, 1 = background.
        target: (H, W) float tensor target mask. Values in [0, 1] where
            0 = glyph, 1 = background.

    Returns:
        A float score in [0, 1] where higher is better. Returns 0.0 if the
        target glyph has no pixels.

    Example:
        >>> score = compute_score(rendered.detach(), target)
        >>> print(f"Coverage score: {score:.3f}")
    """
    # Fraction of glyph pixels that are covered by strokes
    glyph_mask = target < 0.5       # True where glyph is
    stroke_mask = rendered < 0.5    # True where strokes render

    if glyph_mask.sum() == 0:
        return 0.0

    covered = (glyph_mask & stroke_mask).sum().float()
    total_glyph = glyph_mask.sum().float()
    coverage = (covered / total_glyph).item()

    # Penalty for strokes outside glyph
    outside = (stroke_mask & ~glyph_mask).sum().float()
    total_stroke = stroke_mask.sum().float().clamp(min=1)
    overshoot = (outside / total_stroke).item()

    return max(0.0, coverage - 0.5 * overshoot)


def downsample_stroke(points, max_points=40):
    """Downsample a polyline to reduce point count while preserving shape.

    Uses uniform sampling along the polyline, always keeping the first and
    last points to preserve stroke endpoints. This prevents excessive memory
    usage and computation time for strokes with many points.

    Args:
        points: List of [x, y] coordinate pairs representing the polyline.
        max_points: Maximum number of points to keep. If the input has fewer
            points, it is returned unchanged.

    Returns:
        A list of [x, y] pairs with at most max_points elements. The first
        and last points are always preserved.

    Example:
        >>> stroke = [[i, i*2] for i in range(100)]
        >>> downsampled = downsample_stroke(stroke, max_points=20)
        >>> len(downsampled)
        20
    """
    n = len(points)
    if n <= max_points:
        return points
    # Evenly spaced indices, always including first and last
    indices = np.round(np.linspace(0, n - 1, max_points)).astype(int)
    return [points[i] for i in indices]


def optimize(config):
    """Run the DiffVG optimization loop.

    Main optimization function that coordinates the entire gradient descent
    process. Sets up the target glyph mask, initializes learnable parameters,
    runs the optimization loop with early stopping, and returns the optimized
    strokes.

    The optimization uses two separate Adam optimizers:
    - Point optimizer (lr=1.0): Updates stroke point positions
    - Width optimizer (lr=0.5): Updates stroke widths

    Early stopping is triggered after 6 consecutive evaluation windows (300
    iterations) with no loss improvement.

    Args:
        config: Dictionary containing optimization configuration:
            - font_path (str): Path to font file (required)
            - char (str): Character to optimize (required)
            - canvas_size (int): Canvas size in pixels (default: 224)
            - num_iterations (int): Maximum iterations (default: 500)
            - stroke_width (float): Initial stroke width (default: 8.0)
            - lr (float): Learning rate for point optimizer (default: 1.0)
            - max_points_per_stroke (int): Max points per stroke (default: 40)
            - thin_iterations (int): Glyph thinning iterations (default: 0)
            - initial_strokes (list): Required list of initial stroke points
              as [[[x, y], ...], ...]

    Returns:
        A dictionary containing:
        - 'strokes': List of optimized stroke coordinates [[[x, y], ...], ...]
        - 'score': Final coverage score (0-1, higher is better)
        - 'iterations': Number of iterations configured
        - 'final_loss': Best loss value achieved
        - 'elapsed': Total optimization time in seconds

        On error, returns {'error': str} with an error message.

    Example:
        >>> config = {
        ...     'font_path': '/fonts/Arial.ttf',
        ...     'char': 'A',
        ...     'initial_strokes': [[[50, 200], [112, 20], [174, 200]]],
        ...     'num_iterations': 300
        ... }
        >>> result = optimize(config)
        >>> print(f"Score: {result['score']:.3f}")
    """
    t0 = time.monotonic()

    font_path = config['font_path']
    char = config['char']
    canvas_size = config.get('canvas_size', 224)
    num_iterations = config.get('num_iterations', 500)
    initial_width = config.get('stroke_width', 8.0)
    lr = config.get('lr', 1.0)
    max_pts_per_stroke = config.get('max_points_per_stroke', 40)
    thin_iterations = config.get('thin_iterations', 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    # Render target glyph mask (optionally thinned to reduce junction blobs)
    mask = render_glyph_mask(font_path, char, canvas_size, thin_iterations)
    if mask is None:
        return {'error': f'Could not render glyph for {char}'}

    # Target: 0.0 where glyph is, 1.0 for background
    target = torch.tensor(
        (~mask).astype(np.float32), device=device,
    )

    # Initialize stroke points
    initial_strokes = config.get('initial_strokes')
    if not initial_strokes:
        return {'error': 'No initial strokes provided'}

    stroke_points_list = []
    stroke_widths = []

    total_pts = 0
    for stroke in initial_strokes:
        # Downsample strokes with too many points
        stroke = downsample_stroke(stroke, max_pts_per_stroke)
        total_pts += len(stroke)
        pts = torch.tensor(stroke, dtype=torch.float32, device=device,
                           requires_grad=True)
        stroke_points_list.append(pts)

        width = torch.tensor(initial_width, dtype=torch.float32, device=device,
                             requires_grad=True)
        stroke_widths.append(width)

    print(f'Strokes: {len(stroke_points_list)}, total points: {total_pts}, '
          f'width: {initial_width}, lr: {lr}, iters: {num_iterations}',
          file=sys.stderr, flush=True)

    # Save initial positions for anchor regularization (preserves topology)
    initial_points_list = [pts.detach().clone() for pts in stroke_points_list]

    # Optimizers: separate LRs for points and widths
    optimizer_pts = torch.optim.Adam(stroke_points_list, lr=lr)
    optimizer_w = torch.optim.Adam(stroke_widths, lr=0.5)

    best_loss = float('inf')
    best_points = None
    best_widths = None
    best_score = 0.0
    stagnant_count = 0
    prev_best_loss = float('inf')

    for it in range(num_iterations):
        optimizer_pts.zero_grad()
        optimizer_w.zero_grad()

        shapes, shape_groups = build_scene(
            stroke_points_list, stroke_widths, canvas_size, device,
        )
        rendered = render_scene(shapes, shape_groups, canvas_size, device)

        loss = compute_loss(rendered, target, stroke_points_list, stroke_widths,
                            initial_points_list)
        loss.backward()

        optimizer_pts.step()
        optimizer_w.step()

        loss_val = loss.item()

        # Clamp points to canvas and widths to reasonable range every 10 iters
        if it % 10 == 0:
            with torch.no_grad():
                for pts in stroke_points_list:
                    pts.clamp_(0, canvas_size - 1)
                for w in stroke_widths:
                    w.clamp_(2.0, 25.0)

        # Track best
        if loss_val < best_loss:
            best_loss = loss_val
            best_points = [pts.detach().clone() for pts in stroke_points_list]
            best_widths = [w.detach().clone() for w in stroke_widths]

        # Log progress and compute score
        if it % 50 == 0:
            with torch.no_grad():
                score = compute_score(rendered.detach(), target)
                best_score = max(best_score, score)
                widths_str = ', '.join(f'{w.item():.1f}' for w in stroke_widths)
            print(f'iter {it:4d}  loss={loss_val:.4f}  score={score:.3f}  '
                  f'widths=[{widths_str}]',
                  file=sys.stderr, flush=True)

            # Early stopping on stagnation
            if it > 0:
                improvement = prev_best_loss - best_loss
                if improvement < 1e-5:
                    stagnant_count += 1
                else:
                    stagnant_count = 0
                if stagnant_count >= 6:  # 300 iters with no progress
                    print(f'Converged at iter {it}', file=sys.stderr, flush=True)
                    break
                prev_best_loss = best_loss

    # Final score computation with best points and widths
    if best_points is not None:
        with torch.no_grad():
            for orig, best in zip(stroke_points_list, best_points):
                orig.data.copy_(best)
            if best_widths is not None:
                for orig, best in zip(stroke_widths, best_widths):
                    orig.data.copy_(best)
            shapes, shape_groups = build_scene(
                stroke_points_list, stroke_widths, canvas_size, device,
            )
            rendered = render_scene(shapes, shape_groups, canvas_size, device)
            best_score = compute_score(rendered, target)

    elapsed = time.monotonic() - t0

    # Convert best points to output format
    result_strokes = []
    for pts in best_points or stroke_points_list:
        stroke = [[round(float(p[0]), 1), round(float(p[1]), 1)]
                  for p in pts.cpu().numpy()]
        result_strokes.append(stroke)

    return {
        'strokes': result_strokes,
        'score': round(best_score, 4),
        'iterations': num_iterations,
        'final_loss': round(best_loss, 6),
        'elapsed': round(elapsed, 1),
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 optimize_strokes.py <input.json>', file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    result = optimize(config)
    print(json.dumps(result), flush=True)
