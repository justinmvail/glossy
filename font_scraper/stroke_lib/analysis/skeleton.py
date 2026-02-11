"""Skeleton analysis facade.

This module provides the SkeletonAnalyzer class, which serves as the main
interface for skeleton analysis operations. It encapsulates the complexity
of skeleton extraction, marker detection, and stroke tracing into a clean,
high-level API.

The analyzer uses scikit-image for skeletonization and provides methods for:
    - Extracting and analyzing skeleton structure from binary masks
    - Detecting vertex, intersection, and termination markers
    - Tracing and extracting stroke paths from the skeleton

Example usage:
    Analyze a glyph skeleton::

        from stroke_lib.analysis.skeleton import SkeletonAnalyzer
        import numpy as np

        # mask is a binary numpy array where True = glyph pixels
        analyzer = SkeletonAnalyzer(merge_distance=12)
        info = analyzer.analyze(mask)

        if info:
            print(f"Skeleton has {len(info.skel_set)} pixels")
            print(f"Found {len(info.endpoints)} endpoints")
            print(f"Found {len(info.junction_clusters)} junction clusters")

    Detect markers::

        markers = analyzer.detect_markers(mask)
        for m in markers:
            print(f"{m.marker_type.value} at ({m.position.x}, {m.position.y})")

    Extract strokes::

        strokes = analyzer.to_strokes(mask, min_stroke_len=5)
        for stroke in strokes:
            print(f"Stroke with {len(stroke)} points")
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Set, Dict, Tuple
from collections import defaultdict
from scipy.spatial import cKDTree

from ..domain.skeleton import SkeletonInfo, Marker, MarkerType
from ..domain.geometry import Point, Stroke

# Distance threshold for suppressing termination markers near vertices
NEAR_VERTEX_DISTANCE = 5

# Minimum stroke length to avoid being classified as a stub
STUB_THRESHOLD = 20


class SkeletonAnalyzer:
    """Facade for skeleton analysis operations.

    Provides a clean interface to skeleton extraction, marker detection,
    and stroke tracing without exposing internal complexity. This class
    coordinates multiple analysis steps and maintains consistent parameters
    across operations.

    The analyzer performs the following types of analysis:
        - Skeleton extraction using morphological skeletonization
        - Adjacency graph construction for 8-connected skeleton pixels
        - Endpoint and junction pixel detection based on pixel degree
        - Junction pixel clustering and merging
        - Marker detection for vertices, intersections, and terminations
        - Stroke path tracing and cleanup

    Attributes:
        merge_distance: Distance threshold in pixels for merging nearby
            junction clusters. Clusters closer than this distance are
            combined into a single junction.

    Example:
        >>> analyzer = SkeletonAnalyzer(merge_distance=12)
        >>> info = analyzer.analyze(binary_mask)
        >>> markers = analyzer.detect_markers(binary_mask)
        >>> strokes = analyzer.to_strokes(binary_mask)
    """

    def __init__(self, merge_distance: int = 12):
        """Initialize analyzer.

        Args:
            merge_distance: Distance threshold for merging nearby junction
                clusters. Junction clusters with centroids closer than this
                distance will be merged into a single cluster. Default is 12
                pixels.
        """
        self.merge_distance = merge_distance

    def analyze(self, mask: np.ndarray) -> Optional[SkeletonInfo]:
        """Analyze skeleton of a binary mask.

        Extracts the morphological skeleton from the mask and builds a
        complete analysis including the adjacency graph, endpoints,
        junction pixels, and clustered junctions.

        The analysis process:
            1. Skeletonize the binary mask using scikit-image
            2. Build an 8-connected adjacency graph
            3. Identify endpoints (degree 1) and junctions (degree 3+)
            4. Cluster nearby junction pixels
            5. Merge clusters that are closer than merge_distance

        Args:
            mask: Binary numpy array of shape (H, W) where non-zero values
                represent the glyph pixels. The mask should have the glyph
                as foreground (True/non-zero) and background as False/zero.

        Returns:
            SkeletonInfo object containing:
                - skel_set: Set of (x, y) tuples for skeleton pixels
                - adj: Adjacency dictionary mapping pixels to neighbors
                - endpoints: Set of endpoint pixels (degree 1)
                - junction_pixels: Set of junction pixels (degree 3+)
                - junction_clusters: List of clustered junction pixel sets
                - assigned: Dict mapping pixels to their cluster index
            Returns None if mask is empty, has no foreground pixels, or
            if skeletonization fails (e.g., scikit-image not installed).
        """
        try:
            from skimage.morphology import skeletonize
        except ImportError:
            return None

        if mask is None or mask.sum() == 0:
            return None

        # Extract skeleton
        skel = skeletonize(mask > 0)
        skel_set = set(zip(*np.where(skel.T)))  # (x, y) format

        if not skel_set:
            return None

        # Build 8-connected adjacency graph
        adj = self._build_adjacency(skel_set)

        # Find endpoints (degree 1) and junction pixels (degree 3+)
        endpoints = set()
        junction_pixels = set()

        for pixel, neighbors in adj.items():
            degree = len(neighbors)
            if degree == 1:
                endpoints.add(pixel)
            elif degree >= 3:
                junction_pixels.add(pixel)

        # Cluster nearby junction pixels
        junction_clusters, assigned = self._cluster_junctions(
            junction_pixels, adj, skel_set
        )

        return SkeletonInfo(
            skel_set=skel_set,
            adj=adj,
            endpoints=endpoints,
            junction_pixels=junction_pixels,
            junction_clusters=junction_clusters,
            assigned=assigned,
        )

    def detect_markers(self, mask: np.ndarray) -> List[Marker]:
        """Detect vertex and termination markers from skeleton.

        Analyzes the skeleton to identify three types of markers:
            - VERTEX: Junction points where strokes meet at sharp corners
            - INTERSECTION: Junction points where strokes cross
            - TERMINATION: Endpoints of strokes (not at junctions)

        The detection algorithm:
            1. Extracts skeleton and identifies junction clusters
            2. Computes centroids for each junction cluster
            3. Merges nearby centroids
            4. Classifies junctions as vertex vs intersection based on
               endpoint connectivity patterns
            5. Identifies termination points from skeleton endpoints

        Args:
            mask: Binary numpy array of shape (H, W) representing the glyph.

        Returns:
            List of Marker objects, each containing:
                - position: Point with x, y coordinates
                - marker_type: MarkerType enum (VERTEX, INTERSECTION, or
                    TERMINATION)
            Returns empty list if analysis fails.
        """
        info = self.analyze(mask)
        if not info:
            return []

        markers = []

        # Vertices from junction cluster centroids
        vertices = []
        for cluster in info.junction_clusters:
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            vertices.append([cx, cy])

        # Merge nearby vertices
        self._merge_nearby_points(vertices, self.merge_distance)

        # Classify junctions as vertex vs intersection
        is_vertex = self._classify_junctions(
            info.endpoints, info.junction_clusters, info.junction_pixels,
            info.adj, info.assigned, vertices
        )

        # Create vertex/intersection markers
        for i, v in enumerate(vertices):
            marker_type = MarkerType.VERTEX if is_vertex[i] else MarkerType.INTERSECTION
            markers.append(Marker(Point(v[0], v[1]), marker_type))

        # Termination markers from endpoints
        absorbed = self._find_absorbed_endpoints(
            info.endpoints, info.junction_clusters, info.adj, info.assigned
        )

        for (x, y) in info.endpoints:
            if (x, y) in info.junction_pixels:
                continue
            if (x, y) in absorbed:
                continue

            too_close = False
            for v in vertices:
                dx = v[0] - x
                dy = v[1] - y
                if (dx * dx + dy * dy) ** 0.5 < NEAR_VERTEX_DISTANCE:
                    too_close = True
                    break

            if not too_close:
                markers.append(Marker(Point(float(x), float(y)), MarkerType.TERMINATION))

        return markers

    def to_strokes(self, mask: np.ndarray, min_stroke_len: int = 5) -> List[Stroke]:
        """Extract stroke paths from skeleton.

        Traces connected paths through the skeleton graph and converts them
        to Stroke objects. Includes post-processing to merge connected
        strokes and absorb short stubs.

        The extraction process:
            1. Trace all paths from endpoints and junction pixels
            2. Filter out strokes shorter than min_stroke_len
            3. Merge strokes that pass through the same junction cluster
            4. Absorb short stub strokes into longer neighbors

        Args:
            mask: Binary numpy array of shape (H, W) representing the glyph.
            min_stroke_len: Minimum number of points required for a valid
                stroke. Strokes with fewer points are filtered out.
                Default is 5.

        Returns:
            List of Stroke objects, each containing a sequence of Point
            objects representing the stroke path. Returns empty list if
            analysis fails.
        """
        info = self.analyze(mask)
        if not info:
            return []

        # Trace raw strokes
        raw_strokes = self._trace_all_strokes(info)

        # Filter tiny stubs
        strokes = [s for s in raw_strokes if len(s) >= min_stroke_len]

        # Merge and clean up strokes
        strokes = self._merge_strokes(strokes, info)
        strokes = self._absorb_stubs(strokes, info)

        # Convert to Stroke objects
        return [
            Stroke([Point(float(x), float(y)) for x, y in s])
            for s in strokes
        ]

    def _build_adjacency(self, skel_set: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        """Build 8-connected adjacency graph.

        Creates a dictionary mapping each skeleton pixel to its set of
        8-connected neighbors (including diagonals).

        Args:
            skel_set: Set of (x, y) tuples representing skeleton pixels.

        Returns:
            Dictionary mapping each pixel to a set of its neighboring
            pixels that are also in the skeleton.
        """
        adj = defaultdict(set)
        for x, y in skel_set:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (x + dx, y + dy)
                    if neighbor in skel_set:
                        adj[(x, y)].add(neighbor)
        return dict(adj)

    def _cluster_junctions(
        self,
        junction_pixels: Set[Tuple[int, int]],
        adj: Dict,
        skel_set: Set[Tuple[int, int]]
    ) -> Tuple[List[Set[Tuple[int, int]]], Dict[Tuple[int, int], int]]:
        """Cluster nearby junction pixels and merge close clusters.

        Groups connected junction pixels into clusters using BFS, then
        merges clusters whose centroids are closer than merge_distance.

        Args:
            junction_pixels: Set of (x, y) tuples for junction pixels.
            adj: Adjacency dictionary from _build_adjacency.
            skel_set: Set of all skeleton pixels (unused but kept for
                API consistency).

        Returns:
            Tuple containing:
                - List of sets, where each set contains the (x, y) tuples
                  of pixels belonging to that cluster
                - Dictionary mapping each junction pixel to its cluster index
        """
        if not junction_pixels:
            return [], {}

        # Initial clustering via BFS
        clusters = []
        assigned = {}
        visited = set()

        for start in junction_pixels:
            if start in visited:
                continue

            cluster = set()
            queue = [start]

            while queue:
                pixel = queue.pop(0)
                if pixel in visited:
                    continue
                visited.add(pixel)

                if pixel in junction_pixels:
                    cluster.add(pixel)
                    assigned[pixel] = len(clusters)

                    for neighbor in adj.get(pixel, []):
                        if neighbor not in visited and neighbor in junction_pixels:
                            queue.append(neighbor)

            if cluster:
                clusters.append(cluster)

        # Merge nearby clusters
        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                if not clusters[i]:
                    continue
                ci_x = sum(p[0] for p in clusters[i]) / len(clusters[i])
                ci_y = sum(p[1] for p in clusters[i]) / len(clusters[i])

                for j in range(i + 1, len(clusters)):
                    if not clusters[j]:
                        continue
                    cj_x = sum(p[0] for p in clusters[j]) / len(clusters[j])
                    cj_y = sum(p[1] for p in clusters[j]) / len(clusters[j])

                    dist = ((ci_x - cj_x) ** 2 + (ci_y - cj_y) ** 2) ** 0.5
                    if dist < self.merge_distance:
                        clusters[i] = clusters[i] | clusters[j]
                        for p in clusters[j]:
                            assigned[p] = i
                        clusters[j] = set()
                        merged = True
                        break
                if merged:
                    break

        # Remove empty clusters and reindex
        final_clusters = [c for c in clusters if c]
        final_assigned = {}
        for idx, cluster in enumerate(final_clusters):
            for pixel in cluster:
                final_assigned[pixel] = idx

        return final_clusters, final_assigned

    def _merge_nearby_points(self, points: List[List[float]], threshold: float) -> None:
        """Merge points that are closer than threshold (modifies in place).

        Iteratively merges the closest pair of points until no pairs are
        closer than the threshold. Merged points are replaced by their
        midpoint.

        Args:
            points: List of [x, y] coordinate lists. Modified in place.
            threshold: Distance threshold for merging. Points closer than
                this distance will be merged.
        """
        merged = True
        while merged:
            merged = False
            for i in range(len(points)):
                if points[i] is None:
                    continue
                for j in range(i + 1, len(points)):
                    if points[j] is None:
                        continue
                    dx = points[i][0] - points[j][0]
                    dy = points[i][1] - points[j][1]
                    if (dx * dx + dy * dy) ** 0.5 < threshold:
                        points[i][0] = (points[i][0] + points[j][0]) / 2
                        points[i][1] = (points[i][1] + points[j][1]) / 2
                        points[j] = None
                        merged = True
                        break
                if merged:
                    break

        # Remove None entries
        points[:] = [p for p in points if p is not None]

    def _classify_junctions(
        self,
        endpoints: Set[Tuple[int, int]],
        junction_clusters: List[Set[Tuple[int, int]]],
        junction_pixels: Set[Tuple[int, int]],
        adj: Dict,
        assigned: Dict,
        vertices: List[List[float]]
    ) -> List[bool]:
        """Classify junction clusters as vertex (True) or intersection (False).

        A junction is classified as a vertex if it has short paths leading
        to endpoints, suggesting stroke terminations converging at the
        junction. Intersections are crossing points without such convergence.

        Args:
            endpoints: Set of endpoint pixel coordinates.
            junction_clusters: List of junction cluster pixel sets.
            junction_pixels: Set of all junction pixels.
            adj: Adjacency dictionary.
            assigned: Pixel to cluster index mapping.
            vertices: List of vertex centroid coordinates.

        Returns:
            List of booleans, one per junction cluster. True indicates
            the cluster is a vertex, False indicates an intersection.
        """
        is_vertex = [False] * len(junction_clusters)

        # Check for convergence patterns
        for endpoint in endpoints:
            if endpoint in junction_pixels:
                continue

            # Trace from endpoint to find if it reaches a junction
            current = endpoint
            path = [current]
            max_steps = 20

            for _ in range(max_steps):
                neighbors = [n for n in adj.get(current, []) if n not in path]
                if not neighbors:
                    break

                current = neighbors[0]
                path.append(current)

                if current in junction_pixels:
                    cluster_idx = assigned.get(current, -1)
                    if cluster_idx >= 0 and len(path) < 15:
                        # Short path from endpoint to junction suggests vertex
                        is_vertex[cluster_idx] = True
                    break

        return is_vertex

    def _find_absorbed_endpoints(
        self,
        endpoints: Set[Tuple[int, int]],
        junction_clusters: List[Set[Tuple[int, int]]],
        adj: Dict,
        assigned: Dict
    ) -> Set[Tuple[int, int]]:
        """Find endpoints that should be absorbed into junction clusters.

        Identifies endpoints that are very close to junction clusters and
        should not be reported as separate termination markers.

        Args:
            endpoints: Set of endpoint pixel coordinates.
            junction_clusters: List of junction cluster pixel sets.
            adj: Adjacency dictionary.
            assigned: Pixel to cluster index mapping.

        Returns:
            Set of endpoint pixel coordinates that should be absorbed
            (not reported as termination markers).
        """
        absorbed = set()

        for endpoint in endpoints:
            current = endpoint
            path = [current]

            for _ in range(20):
                neighbors = [n for n in adj.get(current, []) if n not in path]
                if not neighbors:
                    break

                current = neighbors[0]
                path.append(current)

                cluster_idx = assigned.get(current, -1)
                if cluster_idx >= 0:
                    if len(path) < 15:
                        absorbed.add(endpoint)
                    break

        return absorbed

    @staticmethod
    def _trace_single_path(start: Tuple[int, int], neighbor: Tuple[int, int],
                           info: SkeletonInfo, stop_set: Set,
                           visited_edges: Set) -> Optional[List[Tuple[int, int]]]:
        """Trace a single stroke path from start through neighbor.

        Follows the skeleton graph from start, preferring straight paths
        at junctions. Stops when reaching a stop point (endpoint or junction)
        or when no unvisited edges remain.

        Args:
            start: Starting pixel position.
            neighbor: First neighbor to traverse to.
            info: SkeletonInfo with adjacency data.
            stop_set: Set of endpoints and junction pixels to stop at.
            visited_edges: Set of already-visited edges (modified in place).

        Returns:
            List of (x, y) tuples forming the path, or None if edge already visited.
        """
        edge = (min(start, neighbor), max(start, neighbor))
        if edge in visited_edges:
            return None
        visited_edges.add(edge)

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

            if len(candidates) == 1:
                next_pt, next_edge = candidates[0]
            else:
                # Pick straightest path
                next_pt, next_edge = SkeletonAnalyzer._pick_straightest_candidate(
                    current, path, candidates
                )

            visited_edges.add(next_edge)
            path.append(next_pt)
            prev, current = current, next_pt

        return path

    @staticmethod
    def _pick_straightest_candidate(current: Tuple[int, int], path: List,
                                     candidates: List) -> Tuple:
        """Pick the candidate that continues most straight from current direction.

        Args:
            current: Current position.
            path: Path so far (used to compute incoming direction).
            candidates: List of (neighbor, edge) tuples.

        Returns:
            Tuple of (next_pt, next_edge) for the straightest continuation.
        """
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

        return next_pt, next_edge

    def _trace_all_strokes(self, info: SkeletonInfo) -> List[List[Tuple[int, int]]]:
        """Trace all stroke paths from the skeleton.

        Traces paths starting from endpoints first, then from junction
        pixels, to extract all stroke segments from the skeleton graph.

        Args:
            info: SkeletonInfo object with skeleton analysis data.

        Returns:
            List of stroke paths, where each path is a list of (x, y)
            pixel coordinate tuples.
        """
        stop_set = info.endpoints | info.junction_pixels
        visited_edges: Set = set()
        strokes = []

        # Trace from endpoints first
        for start in sorted(info.endpoints):
            for neighbor in info.adj.get(start, []):
                p = self._trace_single_path(start, neighbor, info, stop_set, visited_edges)
                if p and len(p) >= 2:
                    strokes.append(p)

        # Then from junction pixels
        for start in sorted(info.junction_pixels):
            for neighbor in info.adj.get(start, []):
                p = self._trace_single_path(start, neighbor, info, stop_set, visited_edges)
                if p and len(p) >= 2:
                    strokes.append(p)

        return strokes

    def _merge_strokes(self, strokes: List[List[Tuple[int, int]]], info: SkeletonInfo) -> List[List[Tuple[int, int]]]:
        """Merge strokes that pass through junction clusters.

        Combines strokes that share a junction cluster endpoint and have
        compatible directions (nearly collinear). Uses direction-based
        scoring to find the best merge candidates.

        Args:
            strokes: List of stroke paths to merge.
            info: SkeletonInfo object with cluster assignments.

        Returns:
            List of merged stroke paths.
        """
        import math

        def endpoint_cluster(stroke, from_end):
            pt = tuple(stroke[-1]) if from_end else tuple(stroke[0])
            return info.assigned.get(pt, -1)

        def seg_dir(seg, from_end, n=8):
            if from_end:
                pts = seg[-min(n, len(seg)):]
            else:
                pts = seg[:min(n, len(seg))][::-1]
            dx = pts[-1][0] - pts[0][0]
            dy = pts[-1][1] - pts[0][1]
            length = (dx * dx + dy * dy) ** 0.5
            return (dx / length, dy / length) if length > 0.01 else (0, 0)

        def angle(d1, d2):
            dot = d1[0] * d2[0] + d1[1] * d2[1]
            return math.acos(max(-1.0, min(1.0, dot)))

        # Direction-based merge pass
        changed = True
        while changed:
            changed = False
            cluster_map = defaultdict(list)
            for si, s in enumerate(strokes):
                sc = endpoint_cluster(s, False)
                if sc >= 0:
                    cluster_map[sc].append((si, 'start'))
                ec = endpoint_cluster(s, True)
                if ec >= 0:
                    cluster_map[ec].append((si, 'end'))

            best_score = float('inf')
            best_merge = None

            for cid, entries in cluster_map.items():
                if len(entries) < 2:
                    continue
                for ai in range(len(entries)):
                    si, side_i = entries[ai]
                    dir_i = seg_dir(strokes[si], from_end=(side_i == 'end'))
                    for bi in range(ai + 1, len(entries)):
                        sj, side_j = entries[bi]
                        if sj == si:
                            continue
                        dir_j = seg_dir(strokes[sj], from_end=(side_j == 'end'))
                        ang = math.pi - angle(dir_i, dir_j)
                        if ang < math.pi / 4 and ang < best_score:
                            best_score = ang
                            best_merge = (si, side_i, sj, side_j)

            if best_merge:
                si, side_i, sj, side_j = best_merge
                seg_i = strokes[si] if side_i == 'end' else list(reversed(strokes[si]))
                seg_j = strokes[sj] if side_j == 'start' else list(reversed(strokes[sj]))
                merged = seg_i + seg_j[1:]
                hi, lo = max(si, sj), min(si, sj)
                strokes.pop(hi)
                strokes.pop(lo)
                strokes.append(merged)
                changed = True

        return strokes

    def _absorb_stubs(self, strokes: List[List[Tuple[int, int]]], info: SkeletonInfo) -> List[List[Tuple[int, int]]]:
        """Absorb short stub strokes into longer neighbors.

        Identifies short strokes (stubs) that connect to junction clusters
        and merges them into longer strokes touching the same cluster.

        Args:
            strokes: List of stroke paths to process.
            info: SkeletonInfo object with cluster assignments.

        Returns:
            List of stroke paths with stubs absorbed.
        """
        def endpoint_cluster(stroke, from_end):
            pt = tuple(stroke[-1]) if from_end else tuple(stroke[0])
            return info.assigned.get(pt, -1)

        # Absorb stubs touching junction clusters
        changed = True
        while changed:
            changed = False
            for si in range(len(strokes)):
                s = strokes[si]
                if len(s) >= STUB_THRESHOLD:
                    continue

                sc = endpoint_cluster(s, False)
                ec = endpoint_cluster(s, True)
                clusters_touching = set()
                if sc >= 0:
                    clusters_touching.add(sc)
                if ec >= 0:
                    clusters_touching.add(ec)

                if not clusters_touching:
                    continue

                # Find longest neighbor
                best_target = -1
                best_len = 0
                best_target_side = None
                best_stub_side = None

                for cid in clusters_touching:
                    for sj in range(len(strokes)):
                        if sj == si:
                            continue
                        s2 = strokes[sj]
                        tc_start = endpoint_cluster(s2, False)
                        tc_end = endpoint_cluster(s2, True)

                        if tc_start == cid and len(s2) > best_len:
                            best_target = sj
                            best_len = len(s2)
                            best_target_side = 'start'
                            best_stub_side = 'start' if sc == cid else 'end'
                        if tc_end == cid and len(s2) > best_len:
                            best_target = sj
                            best_len = len(s2)
                            best_target_side = 'end'
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
