"""Skeleton segment classification."""

from __future__ import annotations
import math
from typing import List, Dict, Set, Tuple
from collections import defaultdict

from ..domain.geometry import Segment, Point
from ..domain.skeleton import SkeletonInfo


class SegmentClassifier:
    """Classifies skeleton segments by direction and properties."""

    def classify(self, info: SkeletonInfo) -> List[Segment]:
        """Find and classify all skeleton segments.

        Args:
            info: SkeletonInfo from skeleton analysis

        Returns:
            List of Segment objects with direction classification
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

    def find_vertical_segments(self, segments: List[Segment]) -> List[Segment]:
        """Filter to only vertical segments (60-120 degree angle)."""
        return [s for s in segments if s.is_vertical]

    def find_horizontal_segments(self, segments: List[Segment]) -> List[Segment]:
        """Filter to only horizontal segments."""
        return [s for s in segments if s.is_horizontal]

    def find_best_vertical_chain(
        self,
        segments: List[Segment],
        template_start: Point,
        template_end: Point
    ) -> Tuple[Point, Point] | None:
        """Find vertical segment chain best matching template positions.

        Args:
            segments: List of segments to search
            template_start: Target start position
            template_end: Target end position

        Returns:
            (top_point, bottom_point) of best matching chain, or None
        """
        vertical = [s for s in segments if 75 <= abs(s.angle) <= 105]
        if not vertical:
            vertical = self.find_vertical_segments(segments)
        if not vertical:
            return None

        # Build junction connectivity graph
        junction_to_segs = defaultdict(list)
        for i, seg in enumerate(vertical):
            if seg.start_junction >= 0:
                junction_to_segs[seg.start_junction].append(i)
            if seg.end_junction >= 0:
                junction_to_segs[seg.end_junction].append(i)

        # Find connected chains
        visited = set()
        chains = []

        for i in range(len(vertical)):
            if i in visited:
                continue

            chain = []
            queue = [i]

            while queue:
                seg_idx = queue.pop(0)
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

        # Score each chain
        best = None
        best_score = float('inf')

        for chain in chains:
            points = []
            for seg_idx in chain:
                seg = vertical[seg_idx]
                points.append(seg.start)
                points.append(seg.end)

            if not points:
                continue

            # Find extremes
            points.sort(key=lambda p: p.y)
            top = points[0]
            bottom = points[-1]

            # Score by distance to template
            d1 = top.distance_to(template_start) + bottom.distance_to(template_end)
            d2 = bottom.distance_to(template_start) + top.distance_to(template_end)
            score = min(d1, d2)

            if score < best_score:
                best_score = score
                if d1 <= d2:
                    best = (top, bottom)
                else:
                    best = (bottom, top)

        return best

    def _trace_segment(
        self,
        start: Tuple[int, int],
        neighbor: Tuple[int, int],
        info: SkeletonInfo,
        visited_edges: Set[Tuple]
    ) -> List[Tuple[int, int]] | None:
        """Trace a single segment from start through neighbor."""
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
