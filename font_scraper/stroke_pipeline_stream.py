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

from stroke_core import _merge_to_expected_count, _skel as analyze_skeleton, skel_strokes
from stroke_dataclasses import parse_stroke_template
from stroke_flask import resolve_font_path
from stroke_pipeline import MinimalStrokePipeline
from stroke_rendering import render_glyph_mask
from stroke_scoring import quick_stroke_score
from stroke_skeleton import (
    find_skeleton_segments,
    generate_straight_line,
    resample_path,
    trace_segment,
    trace_to_region,
)
from stroke_templates import NUMPAD_TEMPLATE_VARIANTS
from stroke_utils import point_in_region


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
    yield {'phase': 'Render', 'step': 'Rendering glyph mask...', 'strokes': [], 'markers': []}

    mask = render_glyph_mask(font_path, char, canvas_size)
    if mask is None:
        yield {'phase': 'Error', 'step': 'Failed to render mask', 'strokes': [], 'markers': [], 'done': True}
        return

    # Phase 2: Create pipeline (this does skeleton analysis lazily)
    yield {'phase': 'Skeleton', 'step': 'Analyzing skeleton...', 'strokes': [], 'markers': []}

    # Create the same pipeline as min_strokes() does
    pipe = MinimalStrokePipeline(
        font_path, char, canvas_size,
        resolve_font_path_fn=lambda x: x,  # Already resolved
        render_glyph_mask_fn=render_glyph_mask,
        analyze_skeleton_fn=analyze_skeleton,
        find_skeleton_segments_fn=find_skeleton_segments,
        point_in_region_fn=point_in_region,
        trace_segment_fn=trace_segment,
        trace_to_region_fn=trace_to_region,
        generate_straight_line_fn=generate_straight_line,
        resample_path_fn=resample_path,
        skeleton_to_strokes_fn=skel_strokes,
        apply_stroke_template_fn=_merge_to_expected_count,
        adjust_stroke_paths_fn=lambda st, c, m: st,  # No-op
        quick_stroke_score_fn=quick_stroke_score,
    )

    # Trigger skeleton analysis
    analysis = pipe.analysis
    if analysis is None:
        yield {'phase': 'Error', 'step': 'Failed to analyze skeleton', 'strokes': [], 'markers': [], 'done': True}
        return

    info = analysis.info
    bbox = analysis.bbox

    # Show skeleton points as markers
    skel_markers = [{'x': float(p[0]), 'y': float(p[1]), 'type': 'skeleton'}
                    for p in list(info['skel_set'])[:500]]  # Limit for performance

    # Show endpoints
    ep_markers = [{'x': float(p[0]), 'y': float(p[1]), 'type': 'endpoint'}
                  for p in info['endpoints']]

    # Show junctions
    jp_markers = [{'x': float(p[0]), 'y': float(p[1]), 'type': 'junction'}
                  for p in info['junction_pixels']]

    yield {
        'phase': 'Skeleton',
        'step': f'Found {len(info["skel_set"])} skeleton pixels, {len(info["endpoints"])} endpoints, {len(info["junction_pixels"])} junctions',
        'strokes': [],
        'markers': skel_markers + ep_markers + jp_markers
    }

    # Phase 3: Find segments
    yield {
        'phase': 'Segments',
        'step': f'Found {len(analysis.segments)} skeleton segments',
        'strokes': [],
        'markers': ep_markers + jp_markers
    }

    # Phase 4: Evaluate template variants
    variants = NUMPAD_TEMPLATE_VARIANTS.get(char, {})

    best_strokes = None
    best_score = -1
    best_variant = None

    if variants:
        yield {
            'phase': 'Templates',
            'step': f'Trying {len(variants)} template variants: {", ".join(variants.keys())}',
            'strokes': [],
            'markers': []
        }

        # Try each variant using the actual pipeline
        for var_name, variant_template in variants.items():
            yield {
                'phase': 'Template',
                'step': f'Trying variant: {var_name}',
                'strokes': [],
                'markers': []
            }

            # Use the pipeline's run() method, but we'll also visualize intermediate steps
            strokes = []
            all_waypoint_markers = []
            global_traced = set()

            # Process parameters matching pipeline
            mid_x = (bbox[0] + bbox[2]) / 2
            mid_y = (bbox[1] + bbox[3]) / 2
            h = bbox[3] - bbox[1]
            third_h = h / 3
            top_bound = bbox[1] + third_h
            bot_bound = bbox[1] + 2 * third_h
            waist_margin = h * 0.05

            # Process each stroke in the template
            for stroke_idx, stroke_template in enumerate(variant_template):
                yield {
                    'phase': 'Template',
                    'step': f'{var_name}: Processing stroke {stroke_idx + 1}/{len(variant_template)}',
                    'strokes': strokes,
                    'markers': all_waypoint_markers
                }

                # First, resolve waypoints for visualization (before drawing stroke)
                waypoints, segment_configs = parse_stroke_template(stroke_template)
                stroke_waypoint_markers = []

                for wp_idx, wp in enumerate(waypoints):
                    # Resolve using pipeline's method to get actual positions
                    next_dir = segment_configs[wp_idx].direction if wp_idx < len(segment_configs) else None

                    # Infer direction from next waypoint if not specified (matching pipeline behavior)
                    if next_dir is None and wp_idx + 1 < len(waypoints):
                        next_dir = pipe._infer_direction(wp.region, waypoints[wp_idx + 1].region)

                    resolved_wp = pipe.resolve_waypoint(wp, next_dir, mid_x, mid_y,
                                                       top_bound, bot_bound, waist_margin)

                    # Determine waypoint type for display
                    wp_type = 'terminal'
                    if wp.is_intersection:
                        wp_type = 'intersection'
                    elif wp.is_vertex:
                        wp_type = 'vertex'
                    elif wp.is_curve:
                        wp_type = 'curve'

                    template_pos = pipe.numpad_to_pixel(wp.region)
                    dist = ((resolved_wp.position[0] - template_pos[0])**2 +
                            (resolved_wp.position[1] - template_pos[1])**2)**0.5

                    # Add waypoint marker
                    wp_marker = {
                        'x': float(resolved_wp.position[0]),
                        'y': float(resolved_wp.position[1]),
                        'type': 'waypoint',
                        'label': f'WP{wp_idx + 1}',
                        'wp_type': wp_type
                    }
                    stroke_waypoint_markers.append(wp_marker)

                    yield {
                        'phase': 'Waypoint',
                        'step': f'{var_name} S{stroke_idx + 1}: WP{wp_idx + 1} [{wp_type}] region {wp.region} -> ({resolved_wp.position[0]:.0f},{resolved_wp.position[1]:.0f}) dist={dist:.1f}',
                        'strokes': strokes,
                        'markers': all_waypoint_markers + stroke_waypoint_markers
                    }

                # Now use the pipeline's process_stroke_template method to get actual stroke
                stroke_points = pipe.process_stroke_template(stroke_template, global_traced, trace_paths=True)

                if stroke_points and len(stroke_points) >= 2:
                    strokes.append(stroke_points)
                    all_waypoint_markers.extend(stroke_waypoint_markers)

                    yield {
                        'phase': 'Stroke',
                        'step': f'{var_name}: Completed stroke {stroke_idx + 1}/{len(variant_template)} ({len(stroke_points)} points)',
                        'strokes': strokes,
                        'markers': all_waypoint_markers
                    }

            # Score this variant
            if strokes:
                score = quick_stroke_score(strokes, mask)
                yield {
                    'phase': 'Score',
                    'step': f'{var_name}: Score = {score:.3f}',
                    'strokes': strokes,
                    'markers': [],
                    'score': score
                }

                if score > best_score:
                    best_score = score
                    best_strokes = strokes
                    best_variant = var_name

    # Phase 5: Try skeleton method using pipeline's method
    yield {
        'phase': 'Skeleton Method',
        'step': 'Trying pure skeleton extraction...',
        'strokes': best_strokes or [],
        'markers': []
    }

    # Use the pipeline's try_skeleton_method which applies all the proper
    # merging and adjustments
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

        # Apply the same penalty logic as evaluate_all_variants
        if variants:
            expected_counts = [len(t) for t in variants.values()]
            expected_count = min(expected_counts) if expected_counts else raw_stroke_count

            if raw_stroke_count != expected_count:
                penalty = 0.3 * abs(raw_stroke_count - expected_count)
                adjusted_score = skel_score - penalty
                yield {
                    'phase': 'Skeleton Penalty',
                    'step': f'Stroke count {raw_stroke_count} != expected {expected_count}, penalty {penalty:.2f} -> adjusted score {adjusted_score:.3f}',
                    'strokes': skel_result.strokes,
                    'markers': [],
                    'score': adjusted_score
                }
                skel_score = adjusted_score

        # Compare with best template (prefer skeleton when scores are tied)
        if skel_score >= best_score:
            if skel_score == best_score:
                yield {
                    'phase': 'Selection',
                    'step': f'Skeleton ({skel_score:.3f}) ties template {best_variant} ({best_score:.3f}) - preferring skeleton',
                    'strokes': skel_result.strokes,
                    'markers': [],
                    'score': skel_score
                }
            else:
                yield {
                    'phase': 'Selection',
                    'step': f'Skeleton ({skel_score:.3f}) beats template {best_variant} ({best_score:.3f})',
                    'strokes': skel_result.strokes,
                    'markers': [],
                    'score': skel_score
                }
            best_score = skel_score
            best_strokes = skel_result.strokes
            best_variant = 'skeleton'
        else:
            yield {
                'phase': 'Selection',
                'step': f'Template {best_variant} ({best_score:.3f}) beats skeleton ({skel_score:.3f})',
                'strokes': best_strokes or [],
                'markers': [],
                'score': best_score
            }

    # If no variants existed, skeleton is the only option
    elif not variants:
        # Get raw skeleton strokes as fallback
        raw_skel = skel_strokes(mask, min_len=5)
        if raw_skel:
            score = quick_stroke_score(raw_skel, mask)
            yield {
                'phase': 'Skeleton Method',
                'step': f'Using raw skeleton: {len(raw_skel)} strokes, score = {score:.3f}',
                'strokes': raw_skel,
                'markers': [],
                'score': score
            }
            best_strokes = raw_skel
            best_score = score
            best_variant = 'skeleton'

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
