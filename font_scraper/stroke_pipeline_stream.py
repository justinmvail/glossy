"""Streaming version of MinimalStrokePipeline for frame-by-frame debugging.

This module provides a generator-based streaming interface to the stroke extraction
pipeline, enabling real-time visualization of each processing step. It wraps the
MinimalStrokePipeline class to ensure identical results to the non-streaming
endpoint while yielding intermediate frames for debugging and visualization purposes.

The streaming approach allows developers and users to observe:
    - Glyph mask rendering
    - Skeleton analysis (endpoints, junctions, skeleton pixels)
    - Template variant evaluation with waypoint resolution
    - Stroke path tracing
    - Score computation and winner selection

Typical usage example:

    >>> from stroke_pipeline_stream import stream_minimal_strokes
    >>> for frame in stream_minimal_strokes('path/to/font.ttf', '5'):
    ...     print(f"Phase: {frame['phase']}, Step: {frame['step']}")
    ...     if frame.get('done'):
    ...         final_strokes = frame['strokes']
    ...         break

Note:
    This module is designed for debugging and visualization purposes. For
    production use cases where only the final result is needed, use the
    MinimalStrokePipeline directly from stroke_pipeline module.
"""

from collections.abc import Generator
from typing import Any

from stroke_core import _merge_to_expected_count, skel_strokes
from stroke_dataclasses import parse_stroke_template
from stroke_flask import resolve_font_path
from stroke_pipeline import MinimalStrokePipeline
from stroke_rendering import render_glyph_mask
from stroke_scoring import quick_stroke_score
from stroke_templates import NUMPAD_TEMPLATE_VARIANTS

# Maximum skeleton pixels to visualize (for performance)
MAX_SKELETON_MARKERS = 500


def _create_pipeline(font_path: str, char: str, canvas_size: int) -> MinimalStrokePipeline:
    """Create a MinimalStrokePipeline with all required callbacks.

    Uses PipelineFactory to ensure consistent configuration with other
    pipeline users (stroke_core.py, etc.).

    Args:
        font_path: Resolved path to the font file.
        char: Character to process.
        canvas_size: Canvas size for rendering.

    Returns:
        Configured MinimalStrokePipeline instance.
    """
    from stroke_pipeline import PipelineConfig, PipelineFactory

    config = PipelineConfig(
        canvas_size=canvas_size,
        apply_stroke_template_fn=_merge_to_expected_count,
    )
    return PipelineFactory.create_with_config(font_path, char, config)


def _consume_subgenerator(gen: Generator) -> Generator[dict, None, Any]:
    """Consume a sub-generator, yielding all frames and returning its result.

    Args:
        gen: Generator that yields dicts and returns a result via StopIteration.

    Yields:
        Each frame dict from the generator.

    Returns:
        The generator's return value.
    """
    try:
        while True:
            yield next(gen)
    except StopIteration as e:
        return e.value


def _stream_template_variants(pipe: MinimalStrokePipeline, char: str,
                               bbox: tuple) -> Generator[dict, None, tuple]:
    """Evaluate all template variants and stream progress frames.

    Args:
        pipe: The MinimalStrokePipeline instance.
        char: Character being processed.
        bbox: Bounding box of the glyph.

    Yields:
        Progress frames for each variant evaluation.

    Returns:
        Tuple of (best_strokes, best_score, best_variant).
    """
    variants = NUMPAD_TEMPLATE_VARIANTS.get(char, {})
    best_strokes = None
    best_score = -1
    best_variant = None

    if not variants:
        return best_strokes, best_score, best_variant

    yield {
        'phase': 'Templates',
        'step': f'Trying {len(variants)} template variants: {", ".join(variants.keys())}',
        'strokes': [],
        'markers': []
    }

    for var_name, variant_template in variants.items():
        yield {
            'phase': 'Template',
            'step': f'Trying variant: {var_name}',
            'strokes': [],
            'markers': []
        }

        # Stream variant evaluation
        var_gen = _stream_variant_strokes(pipe, var_name, variant_template, bbox, quick_stroke_score)
        strokes, score, _ = yield from _consume_subgenerator(var_gen)

        if strokes and score > best_score:
            best_score = score
            best_strokes = strokes
            best_variant = var_name

    return best_strokes, best_score, best_variant


def _create_skeleton_markers(info: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Create visualization markers for skeleton features.

    Args:
        info: Skeleton analysis dict with 'skel_set', 'endpoints', 'junction_pixels'.

    Returns:
        Tuple of (skeleton_markers, endpoint_markers, junction_markers).
    """
    skel_markers = [{'x': float(p[0]), 'y': float(p[1]), 'type': 'skeleton'}
                    for p in list(info['skel_set'])[:MAX_SKELETON_MARKERS]]
    ep_markers = [{'x': float(p[0]), 'y': float(p[1]), 'type': 'endpoint'}
                  for p in info['endpoints']]
    jp_markers = [{'x': float(p[0]), 'y': float(p[1]), 'type': 'junction'}
                  for p in info['junction_pixels']]
    return skel_markers, ep_markers, jp_markers


def _resolve_waypoint_marker(pipe: MinimalStrokePipeline, wp: Any, wp_idx: int,
                              segment_configs: list, waypoints: list,
                              mid_x: float, mid_y: float,
                              top_bound: float, bot_bound: float,
                              waist_margin: float) -> dict:
    """Resolve a waypoint and create a visualization marker.

    Args:
        pipe: The MinimalStrokePipeline instance.
        wp: Waypoint object to resolve.
        wp_idx: Index of this waypoint in the list.
        segment_configs: List of segment configurations.
        waypoints: Full list of waypoints.
        mid_x, mid_y: Center coordinates of bounding box.
        top_bound, bot_bound: Y-coordinate bounds for middle third.
        waist_margin: Margin for waist calculations.

    Returns:
        Dict with marker data including position, type, and label.
    """
    next_dir = segment_configs[wp_idx].direction if wp_idx < len(segment_configs) else None
    if next_dir is None and wp_idx + 1 < len(waypoints):
        next_dir = infer_direction_from_regions(wp.region, waypoints[wp_idx + 1].region)

    resolved_wp = pipe.resolve_waypoint(wp, next_dir, mid_x, mid_y,
                                        top_bound, bot_bound, waist_margin)

    wp_type = 'terminal'
    if wp.is_intersection:
        wp_type = 'intersection'
    elif wp.is_vertex:
        wp_type = 'vertex'
    elif wp.is_curve:
        wp_type = 'curve'

    return {
        'x': float(resolved_wp.position[0]),
        'y': float(resolved_wp.position[1]),
        'type': 'waypoint',
        'label': f'WP{wp_idx + 1}',
        'wp_type': wp_type,
        '_resolved': resolved_wp,
        '_template_pos': pipe.numpad_to_pixel(wp.region)
    }


def _stream_variant_strokes(pipe: MinimalStrokePipeline, var_name: str,
                             variant_template: list, bbox: tuple,
                             quick_stroke_score_fn) -> Generator[dict, None, tuple]:
    """Stream frames while processing a single template variant.

    Args:
        pipe: The MinimalStrokePipeline instance.
        var_name: Name of this variant.
        variant_template: List of stroke templates for this variant.
        bbox: Bounding box tuple (x_min, y_min, x_max, y_max).
        quick_stroke_score_fn: Function to score strokes.

    Yields:
        Frame dicts for each processing step.

    Returns:
        Tuple of (strokes, score, all_waypoint_markers).
    """
    strokes = []
    all_waypoint_markers = []
    global_traced = set()

    mid_x = (bbox[0] + bbox[2]) / 2
    mid_y = (bbox[1] + bbox[3]) / 2
    h = bbox[3] - bbox[1]
    third_h = h / 3
    top_bound = bbox[1] + third_h
    bot_bound = bbox[1] + 2 * third_h
    waist_margin = h * 0.05

    for stroke_idx, stroke_template in enumerate(variant_template):
        yield {
            'phase': 'Template',
            'step': f'{var_name}: Processing stroke {stroke_idx + 1}/{len(variant_template)}',
            'strokes': strokes,
            'markers': all_waypoint_markers
        }

        waypoints, segment_configs = parse_stroke_template(stroke_template)
        stroke_waypoint_markers = []

        for wp_idx, wp in enumerate(waypoints):
            marker = _resolve_waypoint_marker(
                pipe, wp, wp_idx, segment_configs, waypoints,
                mid_x, mid_y, top_bound, bot_bound, waist_margin
            )
            stroke_waypoint_markers.append(marker)

            template_pos = marker['_template_pos']
            resolved_pos = (marker['x'], marker['y'])
            dist = ((resolved_pos[0] - template_pos[0])**2 +
                    (resolved_pos[1] - template_pos[1])**2)**0.5

            yield {
                'phase': 'Waypoint',
                'step': f'{var_name} S{stroke_idx + 1}: WP{wp_idx + 1} [{marker["wp_type"]}] region {wp.region} -> ({resolved_pos[0]:.0f},{resolved_pos[1]:.0f}) dist={dist:.1f}',
                'strokes': strokes,
                'markers': all_waypoint_markers + stroke_waypoint_markers
            }

        result = pipe.process_stroke_template(stroke_template, global_traced, trace_paths=True)
        if result is None:
            continue
        stroke_points, global_traced = result

        if stroke_points and len(stroke_points) >= 2:
            strokes.append(stroke_points)
            all_waypoint_markers.extend(stroke_waypoint_markers)

            yield {
                'phase': 'Stroke',
                'step': f'{var_name}: Completed stroke {stroke_idx + 1}/{len(variant_template)} ({len(stroke_points)} points)',
                'strokes': strokes,
                'markers': all_waypoint_markers
            }

    score = 0.0
    if strokes:
        score = quick_stroke_score_fn(strokes, pipe.mask)
        yield {
            'phase': 'Score',
            'step': f'{var_name}: Score = {score:.3f}',
            'strokes': strokes,
            'markers': [],
            'score': score
        }

    return strokes, score, all_waypoint_markers


def _apply_stroke_count_penalty(skel_score: float, stroke_count: int,
                                 variants: dict) -> tuple[float, dict | None]:
    """Apply penalty for stroke count mismatch.

    Args:
        skel_score: Original skeleton score.
        stroke_count: Number of strokes extracted.
        variants: Dict of template variants.

    Returns:
        Tuple of (adjusted_score, penalty_frame or None).
    """
    if not variants:
        return skel_score, None

    expected_counts = [len(t) for t in variants.values()]
    expected_count = min(expected_counts) if expected_counts else stroke_count

    if stroke_count != expected_count:
        penalty = 0.3 * abs(stroke_count - expected_count)
        adjusted_score = skel_score - penalty
        frame = {
            'phase': 'Skeleton Penalty',
            'step': f'Stroke count {stroke_count} != expected {expected_count}, penalty {penalty:.2f} -> adjusted score {adjusted_score:.3f}',
            'markers': [],
            'score': adjusted_score
        }
        return adjusted_score, frame

    return skel_score, None


def _make_selection_frame(skel_score: float, best_score: float,
                          best_variant: str, skel_wins: bool) -> dict:
    """Create a selection comparison frame.

    Args:
        skel_score: Skeleton method score.
        best_score: Best template score.
        best_variant: Best template variant name.
        skel_wins: True if skeleton method wins.

    Returns:
        Frame dict for the selection step.
    """
    if skel_wins:
        if skel_score == best_score:
            step = f'Skeleton ({skel_score:.3f}) ties template {best_variant} ({best_score:.3f}) - preferring skeleton'
        else:
            step = f'Skeleton ({skel_score:.3f}) beats template {best_variant} ({best_score:.3f})'
    else:
        step = f'Template {best_variant} ({best_score:.3f}) beats skeleton ({skel_score:.3f})'

    return {
        'phase': 'Selection',
        'step': step,
        'markers': [],
        'score': skel_score if skel_wins else best_score
    }


def _stream_render_phase(font_path: str, char: str,
                         canvas_size: int) -> Generator[dict, None, Any]:
    """Stream the render phase and return the mask.

    Args:
        font_path: Resolved path to font file.
        char: Character to render.
        canvas_size: Canvas size in pixels.

    Yields:
        Progress and error frames.

    Returns:
        The rendered mask, or None if failed.
    """
    yield {'phase': 'Render', 'step': 'Rendering glyph mask...', 'strokes': [], 'markers': []}

    mask = render_glyph_mask(font_path, char, canvas_size)
    if mask is None:
        yield {'phase': 'Error', 'step': 'Failed to render mask', 'strokes': [], 'markers': [], 'done': True}
    return mask


def _stream_skeleton_phase(pipe: MinimalStrokePipeline) -> Generator[dict, None, tuple | None]:
    """Stream the skeleton analysis phase.

    Args:
        pipe: The MinimalStrokePipeline instance.

    Yields:
        Progress and error frames with skeleton markers.

    Returns:
        Tuple of (analysis, info, bbox, markers) or None if failed.
    """
    yield {'phase': 'Skeleton', 'step': 'Analyzing skeleton...', 'strokes': [], 'markers': []}

    analysis = pipe.analysis
    if analysis is None:
        yield {'phase': 'Error', 'step': 'Failed to analyze skeleton', 'strokes': [], 'markers': [], 'done': True}
        return None

    info = analysis.info
    bbox = analysis.bbox

    skel_markers, ep_markers, jp_markers = _create_skeleton_markers(info)
    all_markers = skel_markers + ep_markers + jp_markers

    yield {
        'phase': 'Skeleton',
        'step': f'Found {len(info["skel_set"])} skeleton pixels, {len(info["endpoints"])} endpoints, {len(info["junction_pixels"])} junctions',
        'strokes': [],
        'markers': all_markers
    }

    yield {
        'phase': 'Segments',
        'step': f'Found {len(analysis.segments)} skeleton segments',
        'strokes': [],
        'markers': ep_markers + jp_markers
    }

    return analysis, info, bbox, (ep_markers, jp_markers)


def _stream_skeleton_evaluation(pipe: MinimalStrokePipeline, variants: dict,
                                 best_strokes: list, best_score: float,
                                 best_variant: str, mask,
                                 quick_stroke_score_fn) -> Generator[dict, None, tuple]:
    """Stream frames while evaluating skeleton method.

    Args:
        pipe: The MinimalStrokePipeline instance.
        variants: Dict of template variants for this character.
        best_strokes: Current best strokes from template evaluation.
        best_score: Current best score.
        best_variant: Name of current best variant.
        mask: Glyph mask array.
        quick_stroke_score_fn: Function to score strokes.

    Yields:
        Frame dicts for skeleton evaluation steps.

    Returns:
        Tuple of (final_strokes, final_score, final_variant).
    """
    yield {
        'phase': 'Skeleton Method',
        'step': 'Trying pure skeleton extraction...',
        'strokes': best_strokes or [],
        'markers': []
    }

    skel_result = pipe.try_skeleton_method()

    if skel_result.strokes:
        raw_stroke_count = len(skel_result.strokes)
        yield {
            'phase': 'Skeleton Method',
            'step': f'Extracted {raw_stroke_count} strokes from skeleton (after merge/adjustment)',
            'strokes': skel_result.strokes,
            'markers': []
        }

        skel_score = skel_result.score
        yield {
            'phase': 'Skeleton Method',
            'step': f'Skeleton score = {skel_score:.3f}',
            'strokes': skel_result.strokes,
            'markers': [],
            'score': skel_score
        }

        # Apply penalty for stroke count mismatch
        skel_score, penalty_frame = _apply_stroke_count_penalty(
            skel_score, raw_stroke_count, variants
        )
        if penalty_frame:
            penalty_frame['strokes'] = skel_result.strokes
            yield penalty_frame

        # Compare with best template
        skel_wins = skel_score >= best_score
        sel_frame = _make_selection_frame(skel_score, best_score, best_variant, skel_wins)
        sel_frame['strokes'] = skel_result.strokes if skel_wins else (best_strokes or [])
        yield sel_frame

        if skel_wins:
            return skel_result.strokes, skel_score, 'skeleton'
        else:
            return best_strokes, best_score, best_variant

    elif not variants:
        # No variants and skeleton failed - try raw skeleton
        raw_skel = skel_strokes(mask, min_len=5)
        if raw_skel:
            score = quick_stroke_score_fn(raw_skel, mask)
            yield {
                'phase': 'Skeleton Method',
                'step': f'Using raw skeleton: {len(raw_skel)} strokes, score = {score:.3f}',
                'strokes': raw_skel,
                'markers': [],
                'score': score
            }
            return raw_skel, score, 'skeleton'

    return best_strokes, best_score, best_variant


def stream_minimal_strokes(font_path: str, char: str, canvas_size: int = 224) -> Generator[dict, None, None]:
    """Generate frames during minimal stroke extraction for step-by-step visualization.

    This generator function wraps the MinimalStrokePipeline to provide a streaming
    interface that yields intermediate results at each processing phase. It enables
    real-time debugging and visualization of the stroke extraction process.

    The function processes through several phases:
        1. Render: Creates the glyph mask from the font
        2. Skeleton: Analyzes the skeleton to find endpoints and junctions
        3. Segments: Identifies skeleton segments
        4. Templates: Evaluates template variants (if available for the character)
        5. Skeleton Method: Tries pure skeleton extraction as an alternative
        6. Complete: Returns the winning method with final strokes

    Args:
        font_path: Path to the font file (TTF, OTF, etc.). Can be a relative path
            or font name that will be resolved via resolve_font_path().
        char: Single character to extract strokes from. Typically a digit (0-9)
            for numpad template matching.
        canvas_size: Size of the square canvas in pixels for rendering the glyph.
            Defaults to 224. Larger values provide more detail but require more
            processing time.

    Yields:
        dict: A frame dictionary containing the current processing state with keys:
            - phase (str): Current phase name (e.g., 'Render', 'Skeleton', 'Template',
                'Waypoint', 'Stroke', 'Score', 'Selection', 'Complete', or 'Error').
            - step (str): Human-readable description of the current step.
            - strokes (list): List of stroke paths. Each stroke is a list of (x, y)
                coordinate tuples. May be partial during processing.
            - markers (list): Debug markers for visualization. Each marker is a dict
                with 'x', 'y', 'type', and optionally 'label' and 'wp_type' keys.
                Marker types include 'skeleton', 'endpoint', 'junction', 'waypoint'.
            - score (float, optional): Current coverage score when applicable.
                Present during Score, Selection, and Complete phases.
            - variant (str, optional): Name of the winning variant. Only present
                in the final 'Complete' frame.
            - done (bool, optional): True only on the final frame, indicating
                processing is complete.

    Notes:
        The generator uses the actual MinimalStrokePipeline internally to ensure
        that the final strokes are identical to what would be produced by the
        non-streaming min_strokes() endpoint.

        For characters with defined template variants (in NUMPAD_TEMPLATE_VARIANTS),
        each variant is evaluated and scored. The skeleton method is also tried
        as an alternative. The method with the highest score wins, with skeleton
        preferred in case of ties.

        If an error occurs (e.g., failed mask rendering or skeleton analysis),
        the generator yields an Error frame with done=True and returns early.

    Example:
        Collecting all frames for later analysis::

            frames = list(stream_minimal_strokes('/fonts/Arial.ttf', '8'))
            final_frame = frames[-1]
            if final_frame.get('done') and final_frame['phase'] != 'Error':
                strokes = final_frame['strokes']
                print(f"Extracted {len(strokes)} strokes with score {final_frame['score']:.3f}")

        Streaming frames for real-time display::

            for frame in stream_minimal_strokes('Roboto', '3'):
                update_visualization(frame['strokes'], frame['markers'])
                update_status(f"{frame['phase']}: {frame['step']}")
                if frame.get('done'):
                    break
    """
    font_path = resolve_font_path(font_path)

    # Phase 1: Render mask
    render_gen = _stream_render_phase(font_path, char, canvas_size)
    mask = yield from _consume_subgenerator(render_gen)
    if mask is None:
        return

    # Phase 2: Create pipeline and analyze skeleton
    pipe = _create_pipeline(font_path, char, canvas_size)
    skel_gen = _stream_skeleton_phase(pipe)
    skel_result = yield from _consume_subgenerator(skel_gen)
    if skel_result is None:
        return

    analysis, info, bbox, _ = skel_result

    # Phase 3: Evaluate template variants
    template_gen = _stream_template_variants(pipe, char, bbox)
    best_strokes, best_score, best_variant = yield from _consume_subgenerator(template_gen)

    # Phase 4: Evaluate skeleton method
    eval_gen = _stream_skeleton_evaluation(pipe, NUMPAD_TEMPLATE_VARIANTS.get(char, {}),
                                           best_strokes, best_score, best_variant,
                                           mask, quick_stroke_score)
    best_strokes, best_score, best_variant = yield from _consume_subgenerator(eval_gen)

    # Final result
    yield {
        'phase': 'Complete',
        'step': f'Winner: {best_variant} with score {best_score:.3f}',
        'strokes': best_strokes or [],
        'markers': [],
        'score': best_score,
        'variant': best_variant,
        'done': True
    }
