"""Variant evaluation for stroke pipeline.

This module provides variant evaluation functionality extracted from
MinimalStrokePipeline to improve code organization and maintainability.

The VariantEvaluator class handles:
- Evaluating template variants for a character
- Comparing skeleton-based vs template-based results
- Scoring and selecting the best variant
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from stroke_dataclasses import VariantResult
from stroke_templates import NUMPAD_TEMPLATE_VARIANTS

if TYPE_CHECKING:
    import numpy as np


class VariantEvaluator:
    """Evaluates stroke variants to find the best representation.

    This class encapsulates the variant evaluation logic, comparing
    template-based and skeleton-based stroke extraction methods to
    find the highest quality result.

    Attributes:
        char: The character being processed.
        run_fn: Callback to run the pipeline with a template.
        try_skeleton_fn: Callback to try skeleton-based extraction.
        quick_score_fn: Callback to score stroke quality.
        get_mask_fn: Callback to get the glyph mask.
    """

    def __init__(
        self,
        char: str,
        run_fn: Callable,
        try_skeleton_fn: Callable,
        quick_score_fn: Callable,
        get_mask_fn: Callable,
    ):
        """Initialize the variant evaluator.

        Args:
            char: Character being processed.
            run_fn: Function to run pipeline with a template.
            try_skeleton_fn: Function to try skeleton method.
            quick_score_fn: Function to score strokes.
            get_mask_fn: Function to get the glyph mask.
        """
        self.char = char
        self._run_fn = run_fn
        self._try_skeleton_fn = try_skeleton_fn
        self._quick_score_fn = quick_score_fn
        self._get_mask_fn = get_mask_fn

    def evaluate_all_variants(self) -> VariantResult:
        """Evaluate all template variants and skeleton, return best result.

        Tries all predefined template variants for the character (if any exist)
        as well as the pure skeleton method. Returns the variant that produces
        the highest quality score.

        Returns:
            VariantResult containing:
                - strokes: Best stroke paths found, or None if all methods failed
                - score: Quality score of the best result
                - variant_name: Name of the winning variant ('skeleton' or template name)

        Note:
            Skeleton results are penalized if their stroke count differs from
            the expected count (based on template definitions). This helps
            prefer template-based results when they produce the correct
            number of strokes.

            When scores are tied, skeleton results are preferred since they
            are derived from the actual glyph structure rather than templates.
        """
        variants = NUMPAD_TEMPLATE_VARIANTS.get(self.char)

        if not variants:
            return self._try_skeleton_fn()

        mask = self._get_mask_fn()
        if mask is None:
            return VariantResult(strokes=None, score=-1, variant_name=None)

        best = VariantResult(strokes=None, score=-1, variant_name=None)

        for var_name, variant_template in variants.items():
            strokes = self._run_fn(variant_template, trace_paths=True)
            if strokes:
                score = self._quick_score_fn(strokes, mask)
                if score > best.score:
                    best = VariantResult(strokes=strokes, score=score, variant_name=var_name)

        skel_result = self._try_skeleton_fn()
        if skel_result.strokes:
            expected_counts = [len(t) for t in variants.values()]
            if expected_counts:
                expected_count = min(expected_counts)
                if len(skel_result.strokes) != expected_count:
                    skel_result.score -= 0.3 * abs(len(skel_result.strokes) - expected_count)

            # Prefer skeleton when scores are tied (skeleton is derived from actual glyph)
            if skel_result.score >= best.score:
                best = skel_result

        return best


def try_skeleton_method(
    analysis,
    char: str,
    skeleton_to_strokes_fn: Callable,
    apply_stroke_template_fn: Callable,
    adjust_stroke_paths_fn: Callable,
    quick_stroke_score_fn: Callable,
) -> VariantResult:
    """Try pure skeleton method and return result with score.

    Attempts to extract strokes directly from the skeleton structure without
    using templates. This method is useful for characters without predefined
    templates or as a fallback when template-based methods fail.

    Args:
        analysis: SkeletonAnalysis object with mask and skeleton data.
        char: Character being processed.
        skeleton_to_strokes_fn: Function to extract strokes from skeleton.
        apply_stroke_template_fn: Function to apply stroke templates.
        adjust_stroke_paths_fn: Function to adjust stroke paths.
        quick_stroke_score_fn: Function to score strokes.

    Returns:
        VariantResult containing:
            - strokes: List of stroke paths, or None if extraction failed
            - score: Quality score (higher is better), -1 on failure
            - variant_name: Always 'skeleton' for this method

    Note:
        The method compares raw skeleton strokes against merged/adjusted
        strokes and keeps the version with the better score. This prevents
        over-merging from degrading stroke quality.
    """
    if analysis is None:
        return VariantResult(strokes=None, score=-1, variant_name='skeleton')

    mask = analysis.mask
    skel_strokes = skeleton_to_strokes_fn(mask, min_stroke_len=5)

    if not skel_strokes:
        return VariantResult(strokes=None, score=-1, variant_name='skeleton')

    # Score raw strokes before any merging
    raw_score = quick_stroke_score_fn(skel_strokes, mask)
    raw_strokes = skel_strokes

    # Apply merge/template adjustments
    skel_strokes = apply_stroke_template_fn(skel_strokes, char)
    skel_strokes = adjust_stroke_paths_fn(skel_strokes, char, mask)

    if not skel_strokes:
        return VariantResult(strokes=None, score=-1, variant_name='skeleton')

    merged_score = quick_stroke_score_fn(skel_strokes, mask)

    # Keep raw strokes if merge degraded the score
    if merged_score < raw_score - 0.01:
        return VariantResult(strokes=raw_strokes, score=raw_score, variant_name='skeleton')

    return VariantResult(strokes=skel_strokes, score=merged_score, variant_name='skeleton')
