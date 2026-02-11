"""Skeleton segment classification.

This module provides the SegmentClassifier class for analyzing and classifying
skeleton segments. It traces segments through the skeleton graph, calculates
their geometric properties (angle, length), and provides methods for filtering
segments by direction (vertical, horizontal) and finding optimal segment chains.

The classifier works with SkeletonInfo objects produced by the SkeletonAnalyzer
and outputs Segment objects with calculated metadata.

Example usage:
    Basic segment classification::

        from stroke_lib.analysis.segments import SegmentClassifier
        from stroke_lib.analysis.skeleton import SkeletonAnalyzer

        analyzer = SkeletonAnalyzer()
        info = analyzer.analyze(mask)

        classifier = SegmentClassifier()
        segments = classifier.classify(info)

        for seg in segments:
            print(f"Angle: {seg.angle:.1f}, Length: {seg.length:.1f}")

    Filter by direction::

        vertical = classifier.find_vertical_segments(segments)
        horizontal = classifier.find_horizontal_segments(segments)
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

from ..domain.geometry import Point, Segment
from ..domain.skeleton import SkeletonInfo


class SegmentClassifier:
    """Classifies skeleton segments by direction and properties.

    This class traces skeleton segments from endpoints and junction pixels,
    calculates their geometric properties, and provides methods for filtering
    and finding optimal segment chains.

    The classification process:
        1. Starts from endpoints and junction pixels in the skeleton
        2. Traces connected paths through the adjacency graph
        3. Calculates angle, length, and junction connectivity for each segment
        4. Creates Segment objects with all computed metadata

    Attributes:
        None (stateless classifier)

    Example:
        >>> classifier = SegmentClassifier()
        >>> segments = classifier.classify(skeleton_info)
        >>> vertical = classifier.find_vertical_segments(segments)
    """

    def classify(self, info: SkeletonInfo) -> list[Segment]:
        """Find and classify all skeleton segments.

        Traces segments starting from endpoints and junction pixels in the
        skeleton graph. Each segment is traced until it reaches another
        endpoint or junction pixel, and its geometric properties are calculated.

        Args:
            info: SkeletonInfo object containing the skeleton adjacency graph,
                endpoints, junction pixels, and cluster assignments.

        Returns:
            List of Segment objects with the following properties calculated:
                - start: Starting Point of the segment
                - end: Ending Point of the segment
                - angle: Direction angle in degrees (0 = horizontal right,
                    90 = vertical down)
                - length: Euclidean distance from start to end
                - start_junction: Index of junction cluster at start (-1 if endpoint)
                - end_junction: Index of junction cluster at end (-1 if endpoint)
        """
        segments = []
        visited_edges = set()

        # Trace segments from endpoints and junctions
        start_points = info.endpoints | info.junction_pixels

        for start in start_points:
            for neighbor in info.adj.get(start, []):
                seg = self._trace_segment(
                    start, neighbor, info, visited_edges
                )
                if seg and len(seg) >= 2:
                    # Calculate segment properties
                    start_pt = Point(seg[0][0], seg[0][1])
                    end_pt = Point(seg[-1][0], seg[-1][1])

                    dx = end_pt.x - start_pt.x
                    dy = end_pt.y - start_pt.y
                    length = math.sqrt(dx * dx + dy * dy)

                    # Angle in degrees (0 = horizontal, 90 = vertical down)
                    if length > 0.01:
                        angle = math.degrees(math.atan2(dy, dx))
                    else:
                        angle = 0

                    # Find junction indices
                    start_junc = info.assigned.get(seg[0], -1)
                    end_junc = info.assigned.get(seg[-1], -1)

                    segments.append(Segment(
                        start=start_pt,
                        end=end_pt,
                        angle=angle,
                        length=length,
                        start_junction=start_junc,
                        end_junction=end_junc,
                    ))

        return segments

    def find_vertical_segments(self, segments: list[Segment]) -> list[Segment]:
        """Filter to only vertical segments (60-120 degree angle).

        Vertical segments are those where the absolute angle is between
        60 and 120 degrees, indicating a primarily vertical orientation.

        Args:
            segments: List of Segment objects to filter.

        Returns:
            List of Segment objects that are classified as vertical.
        """
        return [s for s in segments if s.is_vertical]

    def find_horizontal_segments(self, segments: list[Segment]) -> list[Segment]:
        """Filter to only horizontal segments.

        Horizontal segments are those where the absolute angle is less than
        30 degrees or greater than 150 degrees.

        Args:
            segments: List of Segment objects to filter.

        Returns:
            List of Segment objects that are classified as horizontal.
        """
        return [s for s in segments if s.is_horizontal]

    def _build_junction_connectivity(
        self,
        vertical: list[Segment]
    ) -> dict[int, list[int]]:
        """Build a mapping from junction indices to segment indices.

        Args:
            vertical: List of vertical Segment objects.

        Returns:
            Dict mapping junction index to list of segment indices connected to it.
        """
        junction_to_segs = defaultdict(list)
        for i, seg in enumerate(vertical):
            if seg.start_junction >= 0:
                junction_to_segs[seg.start_junction].append(i)
            if seg.end_junction >= 0:
                junction_to_segs[seg.end_junction].append(i)
        return junction_to_segs

    def _find_connected_chains(
        self,
        vertical: list[Segment],
        junction_to_segs: dict[int, list[int]]
    ) -> list[list[int]]:
        """Find connected chains of segments via BFS.

        Args:
            vertical: List of vertical Segment objects.
            junction_to_segs: Junction to segment index mapping.

        Returns:
            List of chains, where each chain is a list of segment indices.
        """
        visited = set()
        chains = []

        for i in range(len(vertical)):
            if i in visited:
                continue

            chain = []
            queue = deque([i])

            while queue:
                seg_idx = queue.popleft()
                if seg_idx in visited:
                    continue
                visited.add(seg_idx)
                chain.append(seg_idx)

                seg = vertical[seg_idx]
                for junc in [seg.start_junction, seg.end_junction]:
                    if junc >= 0:
                        for other_idx in junction_to_segs[junc]:
                            if other_idx not in visited:
                                queue.append(other_idx)

            if chain:
                chains.append(chain)

        return chains

    def _score_chain(
        self,
        chain: list[int],
        vertical: list[Segment],
        template_start: Point,
        template_end: Point
    ) -> tuple[float, tuple[Point, Point] | None]:
        """Score a chain by distance to template positions.

        Args:
            chain: List of segment indices in the chain.
            vertical: List of vertical Segment objects.
            template_start: Target start position.
            template_end: Target end position.

        Returns:
            Tuple of (score, (top_point, bottom_point)) or (inf, None) if empty.
        """
        points = []
        for seg_idx in chain:
            seg = vertical[seg_idx]
            points.append(seg.start)
            points.append(seg.end)

        if not points:
            return float('inf'), None

        # Find extremes by y-coordinate
        points.sort(key=lambda p: p.y)
        top = points[0]
        bottom = points[-1]

        # Score by distance to template (try both orientations)
        d1 = top.distance_to(template_start) + bottom.distance_to(template_end)
        d2 = bottom.distance_to(template_start) + top.distance_to(template_end)
        score = min(d1, d2)

        if d1 <= d2:
            return score, (top, bottom)
        else:
            return score, (bottom, top)

    def find_best_vertical_chain(
        self,
        segments: list[Segment],
        template_start: Point,
        template_end: Point
    ) -> tuple[Point, Point] | None:
        """Find vertical segment chain best matching template positions.

        Searches for connected chains of vertical segments and scores them
        based on their proximity to the template start and end positions.
        The best-matching chain is returned as a pair of extreme points.

        The algorithm:
            1. Filters segments to those with angles between 75-105 degrees
            2. Builds a connectivity graph based on shared junction clusters
            3. Uses BFS to find connected chains of segments
            4. Scores each chain by distance to template positions
            5. Returns the top and bottom points of the best-scoring chain

        Args:
            segments: List of Segment objects to search through.
            template_start: Target start position to match against.
            template_end: Target end position to match against.

        Returns:
            Tuple of (top_point, bottom_point) representing the endpoints
            of the best matching vertical chain, with points ordered to
            minimize total distance to template positions. Returns None
            if no vertical segments are found.
        """
        # Filter to vertical segments
        vertical = [s for s in segments if 75 <= abs(s.angle) <= 105]
        if not vertical:
            vertical = self.find_vertical_segments(segments)
        if not vertical:
            return None

        # Build connectivity and find chains
        junction_to_segs = self._build_junction_connectivity(vertical)
        chains = self._find_connected_chains(vertical, junction_to_segs)

        # Score chains and select best
        best = None
        best_score = float('inf')

        for chain in chains:
            score, result = self._score_chain(chain, vertical, template_start, template_end)
            if score < best_score:
                best_score = score
                best = result

        return best

    def _trace_segment(
        self,
        start: tuple[int, int],
        neighbor: tuple[int, int],
        info: SkeletonInfo,
        visited_edges: set[tuple]
    ) -> list[tuple[int, int]] | None:
        """Trace a single segment from start through neighbor.

        Follows the skeleton path from the start point through the initial
        neighbor, continuing until reaching an endpoint or junction pixel.
        Uses a straightness heuristic to choose the best continuation when
        multiple paths are available.

        Args:
            start: Starting pixel coordinates as (x, y) tuple.
            neighbor: Initial neighbor pixel to trace through.
            info: SkeletonInfo containing the adjacency graph and stop points.
            visited_edges: Set of already-visited edges to avoid retracing.
                Edges are stored as tuples of (min_point, max_point).

        Returns:
            List of (x, y) tuples representing the traced path, or None if
            the initial edge was already visited.
        """
        edge = (min(start, neighbor), max(start, neighbor))
        if edge in visited_edges:
            return None
        visited_edges.add(edge)

        stop_set = info.endpoints | info.junction_pixels
        path = [start, neighbor]
        current, prev = neighbor, start

        while True:
            if current in stop_set and len(path) > 2:
                break

            neighbors = [n for n in info.adj.get(current, []) if n != prev]
            candidates = []
            for n in neighbors:
                e = (min(current, n), max(current, n))
                if e not in visited_edges:
                    candidates.append((n, e))

            if not candidates:
                break

            # Pick straightest continuation
            if len(candidates) == 1:
                next_pt, next_edge = candidates[0]
            else:
                n_look = min(4, len(path))
                dx_in = current[0] - path[-n_look][0]
                dy_in = current[1] - path[-n_look][1]
                len_in = (dx_in * dx_in + dy_in * dy_in) ** 0.5
                if len_in > 0.01:
                    dx_in /= len_in
                    dy_in /= len_in

                best_dot = -2
                next_pt, next_edge = candidates[0]
                for n, e in candidates:
                    dx_out = n[0] - current[0]
                    dy_out = n[1] - current[1]
                    len_out = (dx_out * dx_out + dy_out * dy_out) ** 0.5
                    if len_out > 0.01:
                        dot = (dx_in * dx_out + dy_in * dy_out) / len_out
                    else:
                        dot = 0
                    if dot > best_dot:
                        best_dot = dot
                        next_pt, next_edge = n, e

            visited_edges.add(next_edge)
            path.append(next_pt)
            prev, current = current, next_pt

        return path
