"""Stroke merging and absorption functions.

This module provides functions for cleaning up and simplifying stroke paths by
merging strokes at junctions and absorbing short stub segments. These operations
are essential for producing clean, coherent strokes from raw tracing output.

Design Patterns:
    The module implements the Strategy Pattern for merge operations. Each merge
    type is encapsulated in a MergeStrategy subclass, and MergePipeline allows
    combining strategies into configurable pipelines. This enables:
    - Easy addition of new merge strategies
    - Configurable merge pipelines per character type
    - Easy testing of strategies in isolation
    - Reordering or skipping merge steps as needed

Algorithm Overview:
    The stroke cleanup process typically follows this sequence:

    1. Junction-Based Merging (DirectionMergeStrategy):
        - Identifies strokes that meet at junction clusters
        - Merges pairs of strokes that have aligned directions (continuation)
        - Uses angle threshold to determine valid merges

    2. T-Junction Handling (TJunctionMergeStrategy):
        - Special case for 3+ strokes meeting at a junction
        - Identifies the main through-stroke and cross-branch
        - Merges main branches while potentially removing short cross-strokes

    3. Stub Absorption (StubAbsorptionStrategy):
        - Convergence stubs: Short strokes ending at apex points (like top of 'A')
        - Junction stubs: Short strokes at junction clusters
        - Proximity stubs: Short strokes near longer stroke endpoints

    4. Orphan Removal (OrphanRemovalStrategy):
        - Removes isolated short strokes with no neighboring strokes at their
          junction clusters

Key Concepts:
    - Junction cluster: A set of pixel coordinates where multiple strokes meet.
      Clusters are pre-computed from skeleton analysis.
    - Assigned clusters: Junction clusters that have been assigned to specific
      junction points during stroke tracing.
    - Stroke direction: Computed from endpoint samples to determine alignment.
    - Stub: A short stroke segment, typically a tracing artifact or minor branch.

Typical usage:
    # Using the Strategy Pattern:
    pipeline = MergePipeline([
        DirectionMergeStrategy(max_angle=np.pi/4),
        TJunctionMergeStrategy(),
        StubAbsorptionStrategy(conv_threshold=18, stub_threshold=20),
        OrphanRemovalStrategy(stub_threshold=20),
    ])
    strokes = pipeline.run(strokes, junction_clusters, assigned_clusters)

    # Or using legacy functions directly:
    strokes = run_merge_pass(strokes, assigned_clusters, max_angle=np.pi/4)
    strokes = merge_t_junctions(strokes, junction_clusters, assigned_clusters)
    strokes = absorb_convergence_stubs(strokes, junction_clusters, assigned_clusters)
    strokes = absorb_junction_stubs(strokes, assigned_clusters)
    strokes = absorb_proximity_stubs(strokes)
    strokes = remove_orphan_stubs(strokes, assigned_clusters)
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Merge Strategy Pattern
# ---------------------------------------------------------------------------

@dataclass
class MergeContext:
    """Context object for merge operations.

    Bundles the data needed by merge strategies to avoid passing
    multiple arguments through the pipeline.

    Attributes:
        strokes: List of stroke paths to merge (modified in place).
        junction_clusters: Original junction cluster sets from skeleton.
        assigned: Assigned junction cluster sets.
    """
    strokes: list[list[tuple]]
    junction_clusters: list[set] = field(default_factory=list)
    assigned: list[set] = field(default_factory=list)


class MergeStrategy(ABC):
    """Base class for stroke merge strategies.

    Each strategy implements a specific type of merge operation.
    Strategies can be combined using MergePipeline.

    Subclasses must implement the merge() method.

    Example:
        >>> class CustomMergeStrategy(MergeStrategy):
        ...     def merge(self, ctx: MergeContext) -> list[list[tuple]]:
        ...         # Custom merge logic
        ...         return ctx.strokes
    """

    @abstractmethod
    def merge(self, ctx: MergeContext) -> list[list[tuple]]:
        """Apply merge strategy to strokes.

        Args:
            ctx: MergeContext with strokes and cluster data.

        Returns:
            Modified strokes list (also modifies ctx.strokes in place).
        """
        pass

    @property
    def name(self) -> str:
        """Return the strategy name for logging/debugging."""
        return self.__class__.__name__


class DirectionMergeStrategy(MergeStrategy):
    """Merge strokes based on direction alignment at junctions.

    Merges pairs of strokes that meet at a junction cluster and have
    aligned directions (one stroke appears to continue into the other).

    Attributes:
        max_angle: Maximum angle (radians) for valid merge. Default pi/4.
        min_len: Minimum stroke length to consider. Default 0.
        max_ratio: Maximum length ratio for merging. Default 0 (no limit).
    """

    def __init__(self, max_angle: float = np.pi/4, min_len: int = 0,
                 max_ratio: float = 0):
        self.max_angle = max_angle
        self.min_len = min_len
        self.max_ratio = max_ratio

    def merge(self, ctx: MergeContext) -> list[list[tuple]]:
        """Apply direction-based merging."""
        return run_merge_pass(
            ctx.strokes, ctx.assigned,
            min_len=self.min_len,
            max_angle=self.max_angle,
            max_ratio=self.max_ratio
        )


class TJunctionMergeStrategy(MergeStrategy):
    """Merge strokes at T-junctions.

    Handles junctions with 3+ strokes by merging the two longest
    strokes that form the main "through" path.
    """

    def merge(self, ctx: MergeContext) -> list[list[tuple]]:
        """Apply T-junction merging."""
        return merge_t_junctions(
            ctx.strokes, ctx.junction_clusters, ctx.assigned
        )


class StubAbsorptionStrategy(MergeStrategy):
    """Absorb short stub strokes into longer neighbors.

    Combines convergence, junction, and proximity stub absorption
    into a single strategy for convenience.

    Attributes:
        conv_threshold: Max length for convergence stubs. Default 18.
        stub_threshold: Max length for junction/proximity stubs. Default 20.
        prox_threshold: Max distance for proximity merging. Default 20.
        absorb_convergence: Whether to absorb convergence stubs. Default True.
        absorb_junction: Whether to absorb junction stubs. Default True.
        absorb_proximity: Whether to absorb proximity stubs. Default True.
    """

    def __init__(self, conv_threshold: int = 18, stub_threshold: int = 20,
                 prox_threshold: int = 20, absorb_convergence: bool = True,
                 absorb_junction: bool = True, absorb_proximity: bool = True):
        self.conv_threshold = conv_threshold
        self.stub_threshold = stub_threshold
        self.prox_threshold = prox_threshold
        self.absorb_convergence = absorb_convergence
        self.absorb_junction = absorb_junction
        self.absorb_proximity = absorb_proximity

    def merge(self, ctx: MergeContext) -> list[list[tuple]]:
        """Apply stub absorption."""
        if self.absorb_convergence:
            ctx.strokes = absorb_convergence_stubs(
                ctx.strokes, ctx.junction_clusters, ctx.assigned,
                conv_threshold=self.conv_threshold
            )
        if self.absorb_junction:
            ctx.strokes = absorb_junction_stubs(
                ctx.strokes, ctx.assigned,
                stub_threshold=self.stub_threshold
            )
        if self.absorb_proximity:
            ctx.strokes = absorb_proximity_stubs(
                ctx.strokes,
                stub_threshold=self.stub_threshold,
                prox_threshold=self.prox_threshold
            )
        return ctx.strokes


class OrphanRemovalStrategy(MergeStrategy):
    """Remove orphaned short stubs with no neighbors.

    Removes short strokes at junction clusters where no other
    strokes have endpoints.

    Attributes:
        stub_threshold: Max length for a stroke to be considered. Default 20.
    """

    def __init__(self, stub_threshold: int = 20):
        self.stub_threshold = stub_threshold

    def merge(self, ctx: MergeContext) -> list[list[tuple]]:
        """Apply orphan removal."""
        return remove_orphan_stubs(
            ctx.strokes, ctx.assigned,
            stub_threshold=self.stub_threshold
        )


class MergePipeline:
    """Run multiple merge strategies in sequence.

    Chains merge strategies together into a configurable pipeline.
    Each strategy's output becomes the input to the next.

    Example:
        >>> pipeline = MergePipeline([
        ...     DirectionMergeStrategy(max_angle=np.pi/4),
        ...     TJunctionMergeStrategy(),
        ...     StubAbsorptionStrategy(),
        ...     OrphanRemovalStrategy(),
        ... ])
        >>> strokes = pipeline.run(strokes, junction_clusters, assigned)

        # Or create a default pipeline:
        >>> pipeline = MergePipeline.create_default()
        >>> strokes = pipeline.run(strokes, junction_clusters, assigned)
    """

    def __init__(self, strategies: list[MergeStrategy]):
        """Initialize with list of strategies.

        Args:
            strategies: List of MergeStrategy instances to run in order.
        """
        self.strategies = strategies

    @classmethod
    def create_default(cls) -> 'MergePipeline':
        """Create the default merge pipeline.

        Returns:
            MergePipeline with standard strategy sequence.
        """
        return cls([
            DirectionMergeStrategy(max_angle=np.pi/4),
            TJunctionMergeStrategy(),
            StubAbsorptionStrategy(conv_threshold=18, stub_threshold=20),
            OrphanRemovalStrategy(stub_threshold=20),
        ])

    @classmethod
    def create_aggressive(cls) -> 'MergePipeline':
        """Create an aggressive merge pipeline.

        Uses wider angle tolerance and lower thresholds for more merging.

        Returns:
            MergePipeline with aggressive settings.
        """
        return cls([
            DirectionMergeStrategy(max_angle=np.pi/3),  # 60 degrees
            TJunctionMergeStrategy(),
            StubAbsorptionStrategy(conv_threshold=25, stub_threshold=25,
                                   prox_threshold=25),
            OrphanRemovalStrategy(stub_threshold=25),
        ])

    @classmethod
    def create_conservative(cls) -> 'MergePipeline':
        """Create a conservative merge pipeline.

        Uses stricter angle tolerance and higher thresholds for less merging.

        Returns:
            MergePipeline with conservative settings.
        """
        return cls([
            DirectionMergeStrategy(max_angle=np.pi/6),  # 30 degrees
            TJunctionMergeStrategy(),
            StubAbsorptionStrategy(conv_threshold=12, stub_threshold=15,
                                   prox_threshold=15),
            OrphanRemovalStrategy(stub_threshold=15),
        ])

    def run(self, strokes: list[list[tuple]],
            junction_clusters: list[set] = None,
            assigned: list[set] = None) -> list[list[tuple]]:
        """Run all strategies in sequence.

        Args:
            strokes: List of stroke paths to merge (modified in place).
            junction_clusters: Original junction cluster sets.
            assigned: Assigned junction cluster sets.

        Returns:
            Modified strokes list.
        """
        ctx = MergeContext(
            strokes=strokes,
            junction_clusters=junction_clusters or [],
            assigned=assigned or [],
        )

        initial_count = len(ctx.strokes)
        logger.debug("MergePipeline starting with %d strokes", initial_count)

        for strategy in self.strategies:
            before = len(ctx.strokes)
            ctx.strokes = strategy.merge(ctx)
            after = len(ctx.strokes)
            if before != after:
                logger.debug("%s: %d -> %d strokes",
                            strategy.__class__.__name__, before, after)

        logger.debug("MergePipeline complete: %d -> %d strokes",
                    initial_count, len(ctx.strokes))
        return ctx.strokes

    def add_strategy(self, strategy: MergeStrategy,
                     position: int = None) -> None:
        """Add a strategy to the pipeline.

        Args:
            strategy: MergeStrategy instance to add.
            position: Index to insert at. If None, appends to end.
        """
        if position is None:
            self.strategies.append(strategy)
        else:
            self.strategies.insert(position, strategy)

    def remove_strategy_by_type(self, strategy_type: type) -> bool:
        """Remove all strategies of a specific type.

        Args:
            strategy_type: Type of strategy to remove.

        Returns:
            True if any strategies were removed.
        """
        original_len = len(self.strategies)
        self.strategies = [s for s in self.strategies
                          if not isinstance(s, strategy_type)]
        return len(self.strategies) < original_len


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def seg_dir(stroke: list[tuple], from_end: bool = False, n_samples: int = 5) -> tuple[float, float]:
    """Compute the direction vector at one end of a stroke.

    The direction is averaged over multiple points near the endpoint to reduce
    noise from local irregularities in the stroke path.

    Args:
        stroke: List of (x, y) coordinate tuples representing the stroke path.
        from_end: If True, compute direction at the end of the stroke (pointing
            outward from end). If False, compute at the start (pointing outward
            from start). Defaults to False.
        n_samples: Number of points to use for averaging the direction.
            Defaults to 5.

    Returns:
        A normalized direction vector (dx, dy) with unit length.
        Returns (1.0, 0.0) as a fallback if the stroke has fewer than 2 points
        or the computed direction has zero length.

    Note:
        The direction always points "outward" from the endpoint, meaning:
        - For from_end=True: direction points from interior toward the end
        - For from_end=False: direction points from start toward interior

        This convention makes it easy to check if two strokes meeting at a
        junction are aligned (their outward directions should be opposite,
        i.e., angle between them should be close to pi).
    """
    if len(stroke) < 2:
        return (1.0, 0.0)

    pts = stroke[-n_samples:] if from_end else stroke[:n_samples]
    if len(pts) < 2:
        pts = stroke[-2:] if from_end else stroke[:2]

    if from_end:
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
    else:
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]

    length = (dx*dx + dy*dy)**0.5
    if length < 1e-6:
        return (1.0, 0.0)
    return (dx/length, dy/length)


def angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Compute the angle between two direction vectors.

    Args:
        v1: First direction vector (dx, dy). Should be normalized for accurate
            results, though non-normalized vectors will work.
        v2: Second direction vector (dx, dy).

    Returns:
        Angle between the vectors in radians, in the range [0, pi].
        Returns 0 for parallel vectors pointing the same direction,
        pi for antiparallel vectors.

    Note:
        Uses the dot product formula: cos(angle) = v1 . v2 / (|v1| |v2|).
        The dot product is clamped to [-1, 1] to handle floating-point errors
        that might produce values slightly outside this range.
    """
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    dot = max(-1.0, min(1.0, dot))
    return np.arccos(dot)


def endpoint_cluster(stroke: list[tuple], from_end: bool, assigned: list[set]) -> int:
    """Find which junction cluster contains a stroke endpoint.

    Checks if the specified endpoint of a stroke lies within any of the
    assigned junction clusters, with a small tolerance for nearby pixels.

    Args:
        stroke: List of (x, y) coordinate tuples.
        from_end: If True, check the last point; if False, check the first.
        assigned: List of sets, where each set contains (x, y) integer tuples
            representing pixels in a junction cluster.

    Returns:
        The index of the cluster containing the endpoint (0-indexed), or -1
        if the endpoint is not in any cluster.

    Note:
        The function checks the exact pixel position and all 8 neighboring
        pixels (3x3 neighborhood) to handle slight misalignments between
        stroke endpoints and cluster centers.
    """
    if not stroke:
        return -1

    pt = stroke[-1] if from_end else stroke[0]
    if isinstance(pt, (list, tuple)):
        pt_int = (int(round(pt[0])), int(round(pt[1])))
    else:
        pt_int = pt

    for i, cluster in enumerate(assigned):
        if pt_int in cluster:
            return i
        # Check nearby
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (pt_int[0] + dx, pt_int[1] + dy) in cluster:
                    return i
    return -1


def _build_cluster_endpoint_map(strokes: list[list[tuple]],
                                 assigned: list[set],
                                 cluster_cache: dict = None) -> dict[int, list[tuple]]:
    """Build a mapping from cluster indices to stroke endpoints.

    Args:
        strokes: List of stroke paths.
        assigned: List of junction cluster sets.
        cluster_cache: Optional dict to populate with (stroke_idx, from_end) -> cluster_id.
            If provided, will be filled during the map building to avoid redundant lookups.

    Returns:
        Dict mapping cluster index to list of (stroke_index, 'start'/'end') tuples.
    """
    cluster_map = defaultdict(list)
    for si, s in enumerate(strokes):
        sc = endpoint_cluster(s, False, assigned)
        ec = endpoint_cluster(s, True, assigned)
        if cluster_cache is not None:
            cluster_cache[(si, False)] = sc
            cluster_cache[(si, True)] = ec
        if sc >= 0:
            cluster_map[sc].append((si, 'start'))
        if ec >= 0:
            cluster_map[ec].append((si, 'end'))
    return cluster_map


def _is_loop_stroke_cached(stroke_idx: int, cluster_cache: dict) -> bool:
    """Check if a stroke has both endpoints in the same cluster using cache.

    Args:
        stroke_idx: Index of the stroke in strokes list.
        cluster_cache: Dict mapping (stroke_idx, from_end) -> cluster_id.

    Returns:
        True if both endpoints are in the same cluster.
    """
    sc = cluster_cache.get((stroke_idx, False), -1)
    ec = cluster_cache.get((stroke_idx, True), -1)
    return sc >= 0 and sc == ec


def _is_loop_stroke(stroke: list[tuple], assigned: list[set]) -> bool:
    """Check if a stroke has both endpoints in the same cluster.

    Args:
        stroke: Stroke path.
        assigned: List of junction cluster sets.

    Returns:
        True if both endpoints are in the same cluster.
    """
    sc = endpoint_cluster(stroke, False, assigned)
    ec = endpoint_cluster(stroke, True, assigned)
    return sc >= 0 and sc == ec


def _find_best_merge_pair(strokes: list[list[tuple]], cluster_map: dict,
                          assigned: list[set], min_len: int, max_angle: float,
                          max_ratio: float, cluster_cache: dict = None) -> tuple | None:
    """Find the best pair of strokes to merge.

    Args:
        strokes: List of stroke paths.
        cluster_map: Cluster to endpoints mapping.
        assigned: List of junction cluster sets.
        min_len: Minimum stroke length for merging.
        max_angle: Maximum angle deviation for merging.
        max_ratio: Maximum length ratio for merging.
        cluster_cache: Optional dict mapping (stroke_idx, from_end) -> cluster_id
            for cached lookups. Avoids repeated endpoint_cluster() calls.

    Returns:
        Tuple (si, side_i, sj, side_j) for best merge, or None if no valid merge.
    """
    best_score = float('inf')
    best_merge = None

    for _cluster_id, entries in cluster_map.items():
        if len(entries) < 2:
            continue
        for ai in range(len(entries)):
            si, side_i = entries[ai]
            dir_i = seg_dir(strokes[si], from_end=(side_i == 'end'))

            for bi in range(ai + 1, len(entries)):
                sj, side_j = entries[bi]
                if sj == si:
                    continue

                # Check length constraints
                li, lj = len(strokes[si]), len(strokes[sj])
                if min(li, lj) < min_len:
                    continue
                if max_ratio > 0 and max(li, lj) / max(min(li, lj), 1) > max_ratio:
                    continue

                # Skip loop strokes (use cache if available)
                if cluster_cache is not None:
                    if _is_loop_stroke_cached(si, cluster_cache):
                        continue
                    if _is_loop_stroke_cached(sj, cluster_cache):
                        continue
                else:
                    if _is_loop_stroke(strokes[si], assigned):
                        continue
                    if _is_loop_stroke(strokes[sj], assigned):
                        continue

                # Check angle alignment
                dir_j = seg_dir(strokes[sj], from_end=(side_j == 'end'))
                angle = np.pi - angle_between(dir_i, dir_j)
                if angle < max_angle and angle < best_score:
                    best_score = angle
                    best_merge = (si, side_i, sj, side_j)

    return best_merge


def _execute_merge(strokes: list[list[tuple]], si: int, side_i: str,
                   sj: int, side_j: str) -> None:
    """Execute a merge of two strokes.

    Args:
        strokes: List of stroke paths (modified in place).
        si: Index of first stroke.
        side_i: Which end of first stroke ('start' or 'end').
        sj: Index of second stroke.
        side_j: Which end of second stroke ('start' or 'end').
    """
    seg_i = strokes[si] if side_i == 'end' else list(reversed(strokes[si]))
    seg_j = strokes[sj] if side_j == 'start' else list(reversed(strokes[sj]))
    merged_stroke = seg_i + seg_j[1:]
    hi, lo = max(si, sj), min(si, sj)
    strokes.pop(hi)
    strokes.pop(lo)
    strokes.append(merged_stroke)


def run_merge_pass(strokes: list[list[tuple]], assigned: list[set],
                   min_len: int = 0, max_angle: float = np.pi/4,
                   max_ratio: float = 0) -> list[list[tuple]]:
    """Merge strokes through junction clusters based on direction alignment.

    This function iteratively merges pairs of strokes that meet at a junction
    cluster and have aligned directions (i.e., one stroke appears to continue
    into the other). It processes one merge at a time, always choosing the
    best-aligned pair.

    Algorithm:
        1. Build a map of which stroke endpoints are at which clusters
        2. For each cluster with 2+ stroke endpoints, evaluate all pairs
        3. Compute the angle between outward directions of each pair
        4. Select the pair with smallest angle (best alignment)
        5. Merge the pair and repeat until no valid merges remain

    Args:
        strokes: List of stroke paths, each a list of (x, y) tuples.
            Modified in place.
        assigned: List of junction cluster sets from skeleton analysis.
        min_len: Minimum stroke length (in points) to consider for merging.
            Strokes shorter than this are skipped. Defaults to 0.
        max_angle: Maximum angle (radians) between directions for a valid merge.
            Strokes with larger angle difference are not merged.
            Defaults to pi/4 (45 degrees).
        max_ratio: If > 0, reject merges where max(len1, len2) / min(len1, len2)
            exceeds this ratio. Prevents merging very different length strokes.
            Defaults to 0 (no ratio check).

    Returns:
        The modified strokes list (also modified in place).

    Note:
        The angle check uses `pi - angle_between(dir1, dir2)` because outward
        directions of continuing strokes should point in opposite directions
        (angle close to pi), so we measure deviation from perfect continuation.

        Loop strokes (both endpoints in same cluster) are excluded from merging
        to preserve intentional closed paths.
    """
    changed = True
    while changed:
        changed = False
        # Build cluster map and cache endpoint lookups to avoid O(n²) repeated calls
        cluster_cache = {}
        cluster_map = _build_cluster_endpoint_map(strokes, assigned, cluster_cache)
        best_merge = _find_best_merge_pair(
            strokes, cluster_map, assigned, min_len, max_angle, max_ratio, cluster_cache
        )

        if best_merge:
            si, side_i, sj, side_j = best_merge
            _execute_merge(strokes, si, side_i, sj, side_j)
            changed = True

    return strokes


def _find_t_junction_candidate(strokes: list[list[tuple]], cluster_map: dict,
                                assigned: list[set],
                                endpoint_cache: dict[tuple[int, bool], int] = None) -> tuple | None:
    """Find a T-junction candidate for merging.

    Args:
        strokes: List of stroke paths.
        cluster_map: Cluster to endpoints mapping.
        assigned: List of junction cluster sets.
        endpoint_cache: Optional pre-built cache from _build_endpoint_cache().

    Returns:
        Tuple (cid, si, side_i, sj, side_j, second_longest_len) if found, else None.
    """
    T_JUNCTION_MAX_ANGLE = 2 * np.pi / 3  # 120 degrees

    for cid, entries in cluster_map.items():
        if len(entries) < 3:
            continue

        entries_sorted = sorted(entries, key=lambda e: len(strokes[e[0]]), reverse=True)
        shortest_idx, _shortest_side = entries_sorted[-1]
        second_longest_len = len(strokes[entries_sorted[1][0]])

        # Check shortest stroke has valid junctions (use cache if available)
        if endpoint_cache:
            s_sc = _get_cached_cluster(endpoint_cache, shortest_idx, False)
            s_ec = _get_cached_cluster(endpoint_cache, shortest_idx, True)
        else:
            s_sc = endpoint_cluster(strokes[shortest_idx], False, assigned)
            s_ec = endpoint_cluster(strokes[shortest_idx], True, assigned)
        if s_sc < 0 or s_ec < 0:
            continue
        if len(strokes[shortest_idx]) >= second_longest_len * 0.4:
            continue

        # Get the two longest strokes
        si, side_i = entries_sorted[0]
        sj, side_j = entries_sorted[1]
        if si == sj:
            continue

        # Check they don't form a loop (use cache if available)
        if endpoint_cache:
            far_i = _get_cached_cluster(endpoint_cache, si, side_i != 'end')
            far_j = _get_cached_cluster(endpoint_cache, sj, side_j != 'end')
        else:
            far_i = endpoint_cluster(strokes[si], from_end=(side_i != 'end'), assigned=assigned)
            far_j = endpoint_cluster(strokes[sj], from_end=(side_j != 'end'), assigned=assigned)
        if far_i >= 0 and far_i == far_j:
            continue

        # Check angle alignment
        dir_i = seg_dir(strokes[si], from_end=(side_i == 'end'))
        dir_j = seg_dir(strokes[sj], from_end=(side_j == 'end'))
        angle = np.pi - angle_between(dir_i, dir_j)

        if angle < T_JUNCTION_MAX_ANGLE:
            return (cid, si, side_i, sj, side_j, second_longest_len)

    return None


def _remove_short_cross_strokes(strokes: list[list[tuple]], cid: int,
                                 threshold_len: float, assigned: list[set],
                                 endpoint_cache: dict[tuple[int, bool], int] = None) -> bool:
    """Remove short cross-strokes at a junction.

    Args:
        strokes: List of stroke paths (modified in place).
        cid: Cluster ID to check.
        threshold_len: Length threshold (40% of second longest).
        assigned: List of junction cluster sets.
        endpoint_cache: Optional pre-built cache from _build_endpoint_cache().

    Returns:
        True if a stroke was removed.
    """
    for sk in range(len(strokes)):
        s = strokes[sk]
        if endpoint_cache:
            s_sc = _get_cached_cluster(endpoint_cache, sk, False)
            s_ec = _get_cached_cluster(endpoint_cache, sk, True)
        else:
            s_sc = endpoint_cluster(s, False, assigned)
            s_ec = endpoint_cluster(s, True, assigned)
        if s_sc >= 0 and s_ec >= 0 and len(s) < threshold_len and (s_sc == cid or s_ec == cid):
            strokes.pop(sk)
            return True
    return False


def merge_t_junctions(strokes: list[list[tuple]], junction_clusters: list[set],
                      assigned: list[set]) -> list[list[tuple]]:
    """Merge strokes at T-junctions by connecting the main through-stroke.

    T-junctions occur where three or more strokes meet, with two forming a
    main "through" path and others being branches. This function identifies
    the shortest stroke at such junctions and, if it's significantly shorter
    than the main branches, merges the two longest strokes while optionally
    removing the short cross-branch.

    Algorithm:
        1. Find junction clusters with 3+ stroke endpoints
        2. Sort strokes at the junction by length
        3. If shortest is < 40% of second-longest, treat as T-junction
        4. Merge the two longest strokes if their angle is acceptable
        5. Remove any remaining short cross-strokes at that junction

    Args:
        strokes: List of stroke paths. Modified in place.
        junction_clusters: Original list of junction cluster sets.
        assigned: List of assigned junction cluster sets.

    Returns:
        The modified strokes list.

    Note:
        The angle threshold for T-junction merges is relaxed to 2*pi/3 (120
        degrees) compared to normal merges, because the main stroke may bend
        at the junction.

        Strokes that would create a loop (same cluster at both ends) are
        excluded from merging.
    """
    changed = True
    while changed:
        changed = False
        # Build caches once per iteration (O(n) instead of O(n²) repeated calls)
        cluster_map = _build_cluster_endpoint_map(strokes, assigned)
        endpoint_cache = _build_endpoint_cache(strokes, assigned)
        candidate = _find_t_junction_candidate(strokes, cluster_map, assigned, endpoint_cache)

        if candidate:
            cid, si, side_i, sj, side_j, second_longest_len = candidate
            _execute_merge(strokes, si, side_i, sj, side_j)
            # Rebuild cache after merge since stroke indices changed
            endpoint_cache = _build_endpoint_cache(strokes, assigned)
            _remove_short_cross_strokes(strokes, cid, second_longest_len * 0.4, assigned, endpoint_cache)
            changed = True

    return strokes


def get_stroke_tail(stroke: list[tuple], at_end: bool, cluster: set) -> tuple[list[tuple], tuple]:
    """Extract the tail portion of a stroke before entering a junction cluster.

    The "tail" is the segment of the stroke leading up to (but not inside) the
    junction cluster, used to determine the stroke's approach direction and
    for extending strokes toward stub tips.

    Args:
        stroke: List of (x, y) coordinate tuples.
        at_end: If True, get the tail at the end of the stroke; if False,
            at the start.
        cluster: Set of (x, y) integer tuples representing the junction cluster.

    Returns:
        A tuple of (tail_points, leg_end_point) where:
            - tail_points: List of up to 8 points from the stroke leading to
                the junction, ordered from interior toward the junction.
            - leg_end_point: The endpoint of the stroke (inside or at the
                cluster boundary).

    Note:
        Points inside the cluster are excluded from the tail (except the first
        point, which establishes the connection). This ensures the tail
        represents the stroke's approach direction without cluster noise.
    """
    if at_end:
        tail = []
        for k in range(len(stroke) - 1, -1, -1):
            pt = tuple(stroke[k]) if isinstance(stroke[k], (list, tuple)) else stroke[k]
            if len(tail) >= 8:
                break
            if (int(round(pt[0])), int(round(pt[1]))) not in cluster or not tail:
                tail.insert(0, pt)
        return tail, stroke[-1]
    else:
        tail = []
        for k in range(len(stroke)):
            pt = tuple(stroke[k]) if isinstance(stroke[k], (list, tuple)) else stroke[k]
            if len(tail) >= 8:
                break
            if (int(round(pt[0])), int(round(pt[1]))) not in cluster or not tail:
                tail.append(pt)
        return list(reversed(tail)), stroke[0]


def extend_stroke_to_tip(stroke: list[tuple], at_end: bool, tail: list[tuple],
                         leg_end: tuple, stub_tip: tuple) -> None:
    """Extend a stroke from its junction end toward a stub tip point.

    When absorbing a convergence stub, the neighboring strokes are extended
    toward the stub's tip to preserve the overall stroke coverage. This
    function adds interpolated points that curve from the stroke's natural
    direction toward the target tip.

    The extension uses a blending approach: early points follow the stroke's
    existing direction, while later points curve toward the stub tip.

    Args:
        stroke: List of (x, y) tuples. Modified in place by appending or
            prepending extension points.
        at_end: If True, extend from the end of the stroke; if False, from
            the start.
        tail: Tail points from get_stroke_tail(), used to compute the stroke's
            approach direction.
        leg_end: The endpoint of the stroke being extended.
        stub_tip: The target point to extend toward (tip of the absorbed stub).

    Returns:
        None. The stroke is modified in place.

    Note:
        The number of extension points is proportional to the distance from
        leg_end to stub_tip. The blending parameter t interpolates between
        continuing straight (t=0) and reaching the tip (t=1).
    """
    if len(tail) >= 2:
        dx = tail[-1][0] - tail[0][0]
        dy = tail[-1][1] - tail[0][1]
        leg_len = (dx * dx + dy * dy) ** 0.5
    else:
        dx, dy = 0, 0
        leg_len = 0

    tip_dx = stub_tip[0] - leg_end[0]
    tip_dy = stub_tip[1] - leg_end[1]
    tip_dist = (tip_dx * tip_dx + tip_dy * tip_dy) ** 0.5
    steps = max(1, int(round(tip_dist)))

    if leg_len > 0.01:
        ux, uy = dx / leg_len, dy / leg_len
        ext_pts = []
        for k in range(1, steps + 1):
            t = k / steps
            ex = leg_end[0] + ux * k
            ey = leg_end[1] + uy * k
            px = ex * (1 - t) + stub_tip[0] * t
            py = ey * (1 - t) + stub_tip[1] * t
            ext_pts.append((px, py))
    else:
        ext_pts = []
        for k in range(1, steps + 1):
            t = k / steps
            ext_pts.append((leg_end[0] + tip_dx * t, leg_end[1] + tip_dy * t))

    if at_end:
        stroke.extend(ext_pts)
    else:
        for p in reversed(ext_pts):
            stroke.insert(0, p)


def absorb_convergence_stubs(strokes: list[list[tuple]], junction_clusters: list[set],
                             assigned: list[set], conv_threshold: int = 18) -> list[list[tuple]]:
    """Absorb short convergence stubs into longer strokes at junction clusters.

    A convergence stub is a short stroke with one endpoint at a junction cluster
    and the other end free (not at any junction). These often appear at apex
    points of letters like 'A', 'V', 'W' where multiple strokes converge.

    When a convergence stub is absorbed, neighboring strokes at the same junction
    are extended toward the stub's free tip, and the stub itself is removed.

    Args:
        strokes: List of stroke paths. Modified in place.
        junction_clusters: List of original junction cluster sets.
        assigned: List of assigned junction cluster sets.
        conv_threshold: Maximum length (in points) for a stroke to be considered
            a convergence stub. Defaults to 18.

    Returns:
        The modified strokes list.

    Note:
        The function requires at least 2 other strokes at the junction cluster
        to perform absorption. This prevents removing stubs that are actually
        the only stroke coverage in a region.

        Special handling exists for "loop stubs" where both endpoints are in
        the same cluster - in this case, the stub tip is determined by which
        endpoint is farther from the cluster center.
    """
    changed = True
    while changed:
        changed = False
        # Build detailed cluster index once per iteration (O(n) instead of O(n²))
        detailed_index = _build_detailed_cluster_index(strokes, assigned)

        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) < 2 or len(s) >= conv_threshold:
                continue
            sc = endpoint_cluster(s, False, assigned)
            ec = endpoint_cluster(s, True, assigned)

            if sc >= 0 and ec < 0:
                cluster_id = sc
                stub_path = list(s)
            elif ec >= 0 and sc < 0:
                cluster_id = ec
                stub_path = list(reversed(s))
            elif sc >= 0 and ec >= 0 and sc == ec:
                cluster_id = sc
                cluster = junction_clusters[sc]
                cx = sum(p[0] for p in cluster) / len(cluster)
                cy = sum(p[1] for p in cluster) / len(cluster)
                d_start = ((s[0][0] - cx) ** 2 + (s[0][1] - cy) ** 2) ** 0.5
                d_end = ((s[-1][0] - cx) ** 2 + (s[-1][1] - cy) ** 2) ** 0.5
                stub_path = list(reversed(s)) if d_start > d_end else list(s)
            else:
                continue

            # Count other endpoints at cluster using index (O(k) instead of O(n))
            endpoints_at_cluster = detailed_index.get(cluster_id, [])
            others_at_cluster = sum(1 for (idx, _) in endpoints_at_cluster if idx != si)
            if others_at_cluster < 2:
                continue

            stub_tip = stub_path[-1]
            cluster = junction_clusters[cluster_id]

            # Extend only strokes at this cluster using index (O(k) instead of O(n))
            strokes_extended = set()
            for sj, is_end in endpoints_at_cluster:
                if sj == si or sj in strokes_extended:
                    continue
                strokes_extended.add(sj)
                s2 = strokes[sj]
                at_end = is_end
                # Check if the other endpoint is also at this cluster
                if not at_end:
                    other_end_cid = endpoint_cluster(s2, True, assigned)
                    if other_end_cid == cluster_id:
                        at_end = True  # Prefer extending from the end
                        at_start = False
                    else:
                        at_start = True
                else:
                    at_start = False

                tail, leg_end = get_stroke_tail(s2, at_end, cluster)
                extend_stroke_to_tip(s2, at_end, tail, leg_end, stub_tip)

            strokes.pop(si)
            changed = True
            break
    return strokes


def absorb_junction_stubs(strokes: list[list[tuple]], assigned: list[set],
                          stub_threshold: int = 20) -> list[list[tuple]]:
    """Absorb short stubs into neighboring strokes at junction clusters.

    Unlike convergence stubs, junction stubs may have both endpoints at
    junction clusters. This function merges such stubs into the longest
    neighboring stroke at one of their junction clusters.

    Args:
        strokes: List of stroke paths. Modified in place.
        assigned: List of assigned junction cluster sets.
        stub_threshold: Maximum length (in points) for a stroke to be
            considered a stub. Defaults to 20.

    Returns:
        The modified strokes list.

    Note:
        The stub is appended to or prepended from the target stroke's endpoint
        that shares the junction cluster. The merging preserves point order
        to maintain a continuous path.
    """
    changed = True
    while changed:
        changed = False
        # Build detailed cluster index once per iteration (O(n) instead of O(n²))
        detailed_index = _build_detailed_cluster_index(strokes, assigned)

        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            sc = endpoint_cluster(s, False, assigned)
            ec = endpoint_cluster(s, True, assigned)
            clusters_touching = set()
            if sc >= 0:
                clusters_touching.add(sc)
            if ec >= 0:
                clusters_touching.add(ec)
            if not clusters_touching:
                continue

            best_target = -1
            best_len = 0
            best_target_side = None
            best_stub_side = None

            # Use index to find strokes at each cluster (O(k) instead of O(n))
            for cid in clusters_touching:
                endpoints_at_cluster = detailed_index.get(cid, [])
                for sj, is_end in endpoints_at_cluster:
                    if sj == si:
                        continue
                    s2 = strokes[sj]
                    if len(s2) > best_len:
                        best_target = sj
                        best_len = len(s2)
                        best_target_side = 'end' if is_end else 'start'
                        best_stub_side = 'start' if sc == cid else 'end'

            if best_target >= 0:
                stub = s if best_stub_side == 'end' else list(reversed(s))
                target = strokes[best_target]
                if best_target_side == 'end':
                    strokes[best_target] = target + stub[1:]
                else:
                    strokes[best_target] = list(reversed(stub[1:])) + target
                strokes.pop(si)
                changed = True
                break
    return strokes


def _find_proximity_merge_target(stub: list[tuple], stub_idx: int, strokes: list[list[tuple]],
                                  stub_threshold: int, prox_threshold: int) -> tuple:
    """Find best merge target for a stub based on endpoint proximity.

    Returns:
        Tuple of (target_idx, target_side, stub_side, distance) or (-1, None, None, inf) if none found.
    """
    best_dist = prox_threshold
    best_target = -1
    best_target_side = None
    best_stub_side = None

    for stub_end in [False, True]:
        sp = stub[-1] if stub_end else stub[0]
        for sj, target in enumerate(strokes):
            if sj == stub_idx or len(target) < stub_threshold:
                continue
            for target_end in [False, True]:
                tp = target[-1] if target_end else target[0]
                d = ((sp[0] - tp[0]) ** 2 + (sp[1] - tp[1]) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_target = sj
                    best_target_side = 'end' if target_end else 'start'
                    best_stub_side = 'end' if stub_end else 'start'

    return best_target, best_target_side, best_stub_side, best_dist


def absorb_proximity_stubs(strokes: list[list[tuple]], stub_threshold: int = 20,
                           prox_threshold: int = 20) -> list[list[tuple]]:
    """Absorb short stubs by proximity to longer stroke endpoints.

    This function handles stubs that may not be at recognized junction clusters
    but are close to endpoints of longer strokes. It's a fallback for cleanup
    after junction-based merging.

    Args:
        strokes: List of stroke paths. Modified in place.
        stub_threshold: Maximum length for a stroke to be considered a stub.
            Defaults to 20.
        prox_threshold: Maximum distance (in pixels) between stub endpoint and
            target endpoint for proximity merging. Defaults to 20.

    Returns:
        The modified strokes list.

    Note:
        Only considers merging with strokes that are at least stub_threshold
        in length, preventing chain reactions of small strokes merging.
    """
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue

            best_target, best_target_side, best_stub_side, _ = _find_proximity_merge_target(
                s, si, strokes, stub_threshold, prox_threshold)

            if best_target < 0:
                continue

            stub = s if best_stub_side == 'end' else list(reversed(s))
            target = strokes[best_target]
            if best_target_side == 'end':
                strokes[best_target] = target + stub[1:]
            else:
                strokes[best_target] = list(reversed(stub[1:])) + target
            strokes.pop(si)
            changed = True
            break
    return strokes


def _build_cluster_index(strokes: list[list[tuple]], assigned: list[set]) -> dict[int, set[int]]:
    """Build an index mapping cluster IDs to stroke indices with endpoints there.

    Returns:
        Dict mapping cluster_id -> set of stroke indices.
    """
    index = {}
    for si, s in enumerate(strokes):
        for is_end in [False, True]:
            cid = endpoint_cluster(s, is_end, assigned)
            if cid >= 0:
                if cid not in index:
                    index[cid] = set()
                index[cid].add(si)
    return index


def _build_detailed_cluster_index(strokes: list[list[tuple]], assigned: list[set]) -> dict[int, list[tuple[int, bool]]]:
    """Build an index mapping cluster IDs to (stroke_index, is_end) pairs.

    This provides more detail than _build_cluster_index by tracking which
    endpoint (start or end) of each stroke is at each cluster.

    Args:
        strokes: List of stroke paths.
        assigned: List of assigned junction cluster sets.

    Returns:
        Dict mapping cluster_id -> list of (stroke_index, is_end) tuples.
    """
    index = {}
    for si, s in enumerate(strokes):
        for is_end in [False, True]:
            cid = endpoint_cluster(s, is_end, assigned)
            if cid >= 0:
                if cid not in index:
                    index[cid] = []
                index[cid].append((si, is_end))
    return index


def _build_endpoint_cache(strokes: list[list[tuple]], assigned: list[set]) -> dict[tuple[int, bool], int]:
    """Build a cache mapping (stroke_index, is_end) to cluster_id.

    This is the reverse of _build_detailed_cluster_index and provides O(1)
    lookup of which cluster a stroke endpoint belongs to, avoiding repeated
    calls to endpoint_cluster().

    Args:
        strokes: List of stroke paths.
        assigned: List of assigned junction cluster sets.

    Returns:
        Dict mapping (stroke_index, is_end) -> cluster_id.
        Missing keys indicate the endpoint is not at any cluster (cluster_id = -1).
    """
    cache = {}
    for si, s in enumerate(strokes):
        for is_end in [False, True]:
            cid = endpoint_cluster(s, is_end, assigned)
            cache[(si, is_end)] = cid
    return cache


def _get_cached_cluster(cache: dict[tuple[int, bool], int], stroke_idx: int, is_end: bool) -> int:
    """Get cluster ID from cache, returning -1 if not found."""
    return cache.get((stroke_idx, is_end), -1)


def remove_orphan_stubs(strokes: list[list[tuple]], assigned: list[set],
                        stub_threshold: int = 20) -> list[list[tuple]]:
    """Remove orphaned short stubs with no neighbors at their junction clusters.

    An orphan stub is a short stroke at a junction cluster where no other
    strokes have endpoints. These are typically tracing artifacts that should
    be removed.

    Args:
        strokes: List of stroke paths. Modified in place.
        assigned: List of assigned junction cluster sets.
        stub_threshold: Maximum length for a stroke to be considered for
            removal. Defaults to 20.

    Returns:
        The modified strokes list.

    Note:
        A stub is only removed if at least one of its junction clusters has
        no other stroke endpoints. This preserves stubs that may be bridges
        between otherwise disconnected regions.
    """
    changed = True
    while changed:
        changed = False
        # Build caches once per iteration (O(n) instead of O(n²))
        cluster_index = _build_cluster_index(strokes, assigned)
        endpoint_cache = _build_endpoint_cache(strokes, assigned)

        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            sc = _get_cached_cluster(endpoint_cache, si, False)
            ec = _get_cached_cluster(endpoint_cache, si, True)

            orphan = False
            for cid in [sc, ec]:
                if cid < 0:
                    continue
                # Count neighbors using index (O(1) lookup instead of O(n))
                neighbors_at_cluster = cluster_index.get(cid, set())
                # Subtract 1 for self (this stroke is in the set)
                if len(neighbors_at_cluster) <= 1:
                    orphan = True
                    break

            if orphan:
                strokes.pop(si)
                changed = True
                break
    return strokes
