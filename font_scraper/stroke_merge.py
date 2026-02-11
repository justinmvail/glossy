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
    from stroke_merge import (
        DEFAULT_CONV_THRESHOLD, DEFAULT_STUB_THRESHOLD
    )
    pipeline = MergePipeline([
        DirectionMergeStrategy(max_angle=np.pi/4),
        TJunctionMergeStrategy(),
        StubAbsorptionStrategy(
            conv_threshold=DEFAULT_CONV_THRESHOLD,
            stub_threshold=DEFAULT_STUB_THRESHOLD
        ),
        OrphanRemovalStrategy(stub_threshold=DEFAULT_STUB_THRESHOLD),
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
from dataclasses import dataclass, field

import numpy as np

# Import utilities and strategy implementations from extracted modules
from stroke_merge_utils import (
    seg_dir,
    angle_between,
    endpoint_cluster,
    _build_cluster_endpoint_map,
    _build_cluster_index,
    _build_detailed_cluster_index,
    _build_endpoint_cache,
    _get_cached_cluster,
    _is_loop_stroke_cached,
    _is_loop_stroke,
    get_stroke_tail,
    extend_stroke_to_tip,
)
from stroke_merge_strategies import (
    _find_best_merge_pair,
    _execute_merge,
    run_merge_pass,
    _find_t_junction_candidate,
    _remove_short_cross_strokes,
    merge_t_junctions,
    absorb_convergence_stubs,
    absorb_junction_stubs,
    _find_proximity_merge_target,
    absorb_proximity_stubs,
    remove_orphan_stubs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Merge Threshold Constants
# ---------------------------------------------------------------------------

# Default thresholds for merge operations
# These control when strokes are considered for merging or absorption

# Convergence stub threshold: Maximum length for a stroke to be absorbed
# at a convergence point (where multiple strokes meet at an apex, like top of 'A')
DEFAULT_CONV_THRESHOLD = 18

# Junction stub threshold: Maximum length for a stroke to be absorbed
# at a junction cluster (where strokes meet)
DEFAULT_STUB_THRESHOLD = 20

# Proximity threshold: Maximum distance for proximity-based stub absorption
DEFAULT_PROX_THRESHOLD = 20

# Conservative thresholds (less merging, preserves more strokes)
CONSERVATIVE_CONV_THRESHOLD = 12
CONSERVATIVE_STUB_THRESHOLD = 15
CONSERVATIVE_PROX_THRESHOLD = 15

# Aggressive thresholds (more merging, simplifies more)
AGGRESSIVE_CONV_THRESHOLD = 25
AGGRESSIVE_STUB_THRESHOLD = 25
AGGRESSIVE_PROX_THRESHOLD = 25


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
        _cluster_maps_valid: Flag indicating if cached cluster maps are valid.
        _cluster_endpoint_map: Cached cluster-to-endpoint mapping.
        _endpoint_cache: Cached endpoint-to-cluster mapping.
        _detailed_cluster_index: Cached detailed cluster index.
        _cluster_index: Cached cluster index (cluster_id -> stroke indices).
    """
    strokes: list[list[tuple]]
    junction_clusters: list[set] = field(default_factory=list)
    assigned: list[set] = field(default_factory=list)
    # Cached cluster maps - invalidated when strokes change
    _cluster_maps_valid: bool = field(default=False, repr=False)
    _cluster_endpoint_map: dict = field(default_factory=dict, repr=False)
    _endpoint_cache: dict = field(default_factory=dict, repr=False)
    _detailed_cluster_index: dict = field(default_factory=dict, repr=False)
    _cluster_index: dict = field(default_factory=dict, repr=False)

    def invalidate_caches(self) -> None:
        """Mark all cached cluster maps as invalid.

        Call this after any operation that modifies the strokes list.
        """
        self._cluster_maps_valid = False

    def get_cluster_endpoint_map(self) -> dict:
        """Get cached cluster-to-endpoint map, building if needed."""
        self._ensure_caches_valid()
        return self._cluster_endpoint_map

    def get_endpoint_cache(self) -> dict:
        """Get cached endpoint-to-cluster map, building if needed."""
        self._ensure_caches_valid()
        return self._endpoint_cache

    def get_detailed_cluster_index(self) -> dict:
        """Get cached detailed cluster index, building if needed."""
        self._ensure_caches_valid()
        return self._detailed_cluster_index

    def get_cluster_index(self) -> dict:
        """Get cached cluster index, building if needed."""
        self._ensure_caches_valid()
        return self._cluster_index

    def _ensure_caches_valid(self) -> None:
        """Rebuild all caches if they are invalid."""
        if self._cluster_maps_valid:
            return

        # Build all caches in a single pass
        self._endpoint_cache = _build_endpoint_cache(self.strokes, self.assigned)
        self._cluster_endpoint_map = _build_cluster_endpoint_map(
            self.strokes, self.assigned, self._endpoint_cache
        )
        self._detailed_cluster_index = _build_detailed_cluster_index(
            self.strokes, self.assigned
        )
        self._cluster_index = _build_cluster_index(self.strokes, self.assigned)
        self._cluster_maps_valid = True


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
        conv_threshold: Max length for convergence stubs.
            Default DEFAULT_CONV_THRESHOLD (18).
        stub_threshold: Max length for junction/proximity stubs.
            Default DEFAULT_STUB_THRESHOLD (20).
        prox_threshold: Max distance for proximity merging.
            Default DEFAULT_PROX_THRESHOLD (20).
        absorb_convergence: Whether to absorb convergence stubs. Default True.
        absorb_junction: Whether to absorb junction stubs. Default True.
        absorb_proximity: Whether to absorb proximity stubs. Default True.
    """

    def __init__(self, conv_threshold: int = DEFAULT_CONV_THRESHOLD,
                 stub_threshold: int = DEFAULT_STUB_THRESHOLD,
                 prox_threshold: int = DEFAULT_PROX_THRESHOLD,
                 absorb_convergence: bool = True,
                 absorb_junction: bool = True, absorb_proximity: bool = True):
        self.conv_threshold = conv_threshold
        self.stub_threshold = stub_threshold
        self.prox_threshold = prox_threshold
        self.absorb_convergence = absorb_convergence
        self.absorb_junction = absorb_junction
        self.absorb_proximity = absorb_proximity

    def merge(self, ctx: MergeContext) -> list[list[tuple]]:
        """Apply stub absorption.

        Uses cached cluster maps from context when available to avoid
        rebuilding maps multiple times across sub-operations.
        """
        # Get initial cached data from context (built once, shared across sub-ops)
        detailed_index = ctx.get_detailed_cluster_index()
        endpoint_cache = ctx.get_endpoint_cache()

        if self.absorb_convergence:
            ctx.strokes = absorb_convergence_stubs(
                ctx.strokes, ctx.junction_clusters, ctx.assigned,
                conv_threshold=self.conv_threshold,
                detailed_index=detailed_index,
                endpoint_cache=endpoint_cache,
            )
            # Caches may be stale after absorption, let functions rebuild as needed
            detailed_index = None
            endpoint_cache = None
            ctx.invalidate_caches()

        if self.absorb_junction:
            ctx.strokes = absorb_junction_stubs(
                ctx.strokes, ctx.assigned,
                stub_threshold=self.stub_threshold,
                detailed_index=detailed_index,
                endpoint_cache=endpoint_cache,
            )
            detailed_index = None
            endpoint_cache = None
            ctx.invalidate_caches()

        if self.absorb_proximity:
            ctx.strokes = absorb_proximity_stubs(
                ctx.strokes,
                stub_threshold=self.stub_threshold,
                prox_threshold=self.prox_threshold
            )
            ctx.invalidate_caches()

        return ctx.strokes


class OrphanRemovalStrategy(MergeStrategy):
    """Remove orphaned short stubs with no neighbors.

    Removes short strokes at junction clusters where no other
    strokes have endpoints.

    Attributes:
        stub_threshold: Max length for a stroke to be considered.
            Default DEFAULT_STUB_THRESHOLD (20).
    """

    def __init__(self, stub_threshold: int = DEFAULT_STUB_THRESHOLD):
        self.stub_threshold = stub_threshold

    def merge(self, ctx: MergeContext) -> list[list[tuple]]:
        """Apply orphan removal.

        Uses cached cluster maps from context when available.
        """
        return remove_orphan_stubs(
            ctx.strokes, ctx.assigned,
            stub_threshold=self.stub_threshold,
            cluster_index=ctx.get_cluster_index(),
            endpoint_cache=ctx.get_endpoint_cache(),
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
            MergePipeline with standard strategy sequence using
            DEFAULT_CONV_THRESHOLD, DEFAULT_STUB_THRESHOLD.
        """
        return cls([
            DirectionMergeStrategy(max_angle=np.pi/4),
            TJunctionMergeStrategy(),
            StubAbsorptionStrategy(
                conv_threshold=DEFAULT_CONV_THRESHOLD,
                stub_threshold=DEFAULT_STUB_THRESHOLD,
            ),
            OrphanRemovalStrategy(stub_threshold=DEFAULT_STUB_THRESHOLD),
        ])

    @classmethod
    def create_aggressive(cls) -> 'MergePipeline':
        """Create an aggressive merge pipeline.

        Uses wider angle tolerance and higher thresholds for more merging.

        Returns:
            MergePipeline with aggressive settings using
            AGGRESSIVE_CONV_THRESHOLD, AGGRESSIVE_STUB_THRESHOLD.
        """
        return cls([
            DirectionMergeStrategy(max_angle=np.pi/3),  # 60 degrees
            TJunctionMergeStrategy(),
            StubAbsorptionStrategy(
                conv_threshold=AGGRESSIVE_CONV_THRESHOLD,
                stub_threshold=AGGRESSIVE_STUB_THRESHOLD,
                prox_threshold=AGGRESSIVE_PROX_THRESHOLD,
            ),
            OrphanRemovalStrategy(stub_threshold=AGGRESSIVE_STUB_THRESHOLD),
        ])

    @classmethod
    def create_conservative(cls) -> 'MergePipeline':
        """Create a conservative merge pipeline.

        Uses stricter angle tolerance and lower thresholds for less merging.

        Returns:
            MergePipeline with conservative settings using
            CONSERVATIVE_CONV_THRESHOLD, CONSERVATIVE_STUB_THRESHOLD.
        """
        return cls([
            DirectionMergeStrategy(max_angle=np.pi/6),  # 30 degrees
            TJunctionMergeStrategy(),
            StubAbsorptionStrategy(
                conv_threshold=CONSERVATIVE_CONV_THRESHOLD,
                stub_threshold=CONSERVATIVE_STUB_THRESHOLD,
                prox_threshold=CONSERVATIVE_PROX_THRESHOLD,
            ),
            OrphanRemovalStrategy(stub_threshold=CONSERVATIVE_STUB_THRESHOLD),
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
                # Invalidate caches when stroke count changes
                ctx.invalidate_caches()

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
