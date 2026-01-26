"""
Improved Skeleton Tracer

Traces skeleton pixels into continuous strokes without fragmenting at junctions.
Uses a greedy path-following approach that prioritizes stroke continuity.
"""

import numpy as np
from scipy import ndimage
from typing import List, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class Point:
    x: float
    y: float
    t: float = 0.0


@dataclass
class Stroke:
    points: List[Point] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        return np.array([[p.x, p.y] for p in self.points])

    @classmethod
    def from_coords(cls, coords: List[Tuple[int, int]]) -> 'Stroke':
        """Create stroke from (row, col) coordinates, converting to (x, y)."""
        return cls([Point(float(c), float(r)) for r, c in coords])

    def length(self) -> float:
        """Calculate total path length."""
        if len(self.points) < 2:
            return 0
        arr = self.to_array()
        return np.sum(np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=1)))


def prune_spurs(skeleton: np.ndarray, min_length: int = 10) -> np.ndarray:
    """
    Remove short branches (spurs) from skeleton.

    Iteratively removes endpoint pixels that are part of short branches.
    """
    result = skeleton.copy()

    for _ in range(min_length):
        # Find current endpoints
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = ndimage.convolve(result.astype(int), kernel, mode='constant')
        endpoints = (neighbor_count == 1) & result

        # Remove endpoints (this shortens spurs by 1 pixel each iteration)
        result = result & ~endpoints

        if not endpoints.any():
            break

    return result


class SkeletonTracer:
    """
    Trace skeleton into continuous strokes.

    Key improvements over naive BFS:
    1. Doesn't break at junctions - follows the most aligned direction
    2. Traces longest paths first (from endpoints)
    3. Merges short fragments into longer strokes
    """

    def __init__(self, min_stroke_length: int = 5):
        self.min_stroke_length = min_stroke_length
        # 8-connectivity neighbor offsets (ordered for direction tracking)
        self.neighbors = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

    def trace(self, skeleton: np.ndarray, prune: bool = True) -> List[Stroke]:
        """Trace skeleton into strokes."""
        if not skeleton.any():
            return []

        # Prune short spurs first
        if prune:
            skeleton = prune_spurs(skeleton, min_length=8)

        if not skeleton.any():
            return []

        # Find special points
        neighbor_count = self._count_neighbors(skeleton)
        endpoints = set(zip(*np.where((neighbor_count == 1) & skeleton)))
        junctions = set(zip(*np.where((neighbor_count >= 3) & skeleton)))

        visited = np.zeros_like(skeleton, dtype=bool)
        strokes = []

        # Strategy 1: Trace from endpoints (produces cleaner long strokes)
        # Sort endpoints by their position to get consistent ordering
        sorted_endpoints = sorted(endpoints, key=lambda p: (p[0], p[1]))

        for start in sorted_endpoints:
            if visited[start]:
                continue

            path = self._trace_path(skeleton, visited, start, junctions)
            if len(path) >= self.min_stroke_length:
                strokes.append(Stroke.from_coords(path))

        # Strategy 2: Trace remaining unvisited regions (loops, missed segments)
        remaining = np.argwhere(skeleton & ~visited)
        for start in remaining:
            start = tuple(start)
            if visited[start]:
                continue

            path = self._trace_path(skeleton, visited, start, junctions)
            if len(path) >= self.min_stroke_length:
                strokes.append(Stroke.from_coords(path))

        # Aggressively merge nearby stroke endpoints
        strokes = self._merge_strokes(strokes, max_gap=5.0)

        # Filter out very short strokes (likely artifacts)
        strokes = [s for s in strokes if s.length() > 15]

        return strokes

    def _count_neighbors(self, skeleton: np.ndarray) -> np.ndarray:
        """Count 8-connected neighbors for each pixel."""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        counts = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        return counts * skeleton

    def _trace_path(
        self,
        skeleton: np.ndarray,
        visited: np.ndarray,
        start: Tuple[int, int],
        junctions: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Trace a continuous path from start point.

        At junctions, follows the most aligned direction instead of stopping.
        """
        path = [start]
        visited[start] = True
        current = start
        prev_dir = None  # Track direction for junction decisions

        while True:
            y, x = current

            # Find all unvisited neighbors on skeleton
            candidates = []
            for i, (dy, dx) in enumerate(self.neighbors):
                ny, nx = y + dy, x + dx
                if (0 <= ny < skeleton.shape[0] and
                    0 <= nx < skeleton.shape[1] and
                    skeleton[ny, nx] and
                    not visited[ny, nx]):
                    candidates.append((ny, nx, i))

            if not candidates:
                # Dead end - try to continue through junction if we just crossed one
                break

            # Pick next point
            if len(candidates) == 1:
                # Only one choice
                ny, nx, dir_idx = candidates[0]
            else:
                # Multiple choices (junction) - pick most aligned with current direction
                if prev_dir is not None:
                    # Score by alignment with previous direction
                    best_score = -2
                    best_candidate = candidates[0]

                    for ny, nx, dir_idx in candidates:
                        # Direction alignment: dot product of direction vectors
                        # prev_dir and dir_idx are indices into self.neighbors
                        prev_dy, prev_dx = self.neighbors[prev_dir]
                        curr_dy, curr_dx = self.neighbors[dir_idx]

                        # Dot product (higher = more aligned)
                        score = prev_dy * curr_dy + prev_dx * curr_dx

                        # Slight preference for continuing straight over sharp turns
                        if score > best_score:
                            best_score = score
                            best_candidate = (ny, nx, dir_idx)

                    ny, nx, dir_idx = best_candidate
                else:
                    # No previous direction - just pick first
                    ny, nx, dir_idx = candidates[0]

            # Move to next point
            visited[ny, nx] = True
            path.append((ny, nx))
            prev_dir = dir_idx
            current = (ny, nx)

        return path

    def _merge_strokes(self, strokes: List[Stroke], max_gap: float = 3.0) -> List[Stroke]:
        """Merge strokes whose endpoints are close together."""
        if len(strokes) < 2:
            return strokes

        merged = []
        used = set()

        for i, stroke1 in enumerate(strokes):
            if i in used:
                continue

            # Try to extend this stroke by finding matching endpoints
            current_points = list(stroke1.points)
            used.add(i)
            changed = True

            while changed:
                changed = False
                start = (current_points[0].x, current_points[0].y)
                end = (current_points[-1].x, current_points[-1].y)

                for j, stroke2 in enumerate(strokes):
                    if j in used:
                        continue

                    s2_start = (stroke2.points[0].x, stroke2.points[0].y)
                    s2_end = (stroke2.points[-1].x, stroke2.points[-1].y)

                    # Check all endpoint combinations
                    dist_end_start = np.sqrt((end[0]-s2_start[0])**2 + (end[1]-s2_start[1])**2)
                    dist_end_end = np.sqrt((end[0]-s2_end[0])**2 + (end[1]-s2_end[1])**2)
                    dist_start_start = np.sqrt((start[0]-s2_start[0])**2 + (start[1]-s2_start[1])**2)
                    dist_start_end = np.sqrt((start[0]-s2_end[0])**2 + (start[1]-s2_end[1])**2)

                    if dist_end_start <= max_gap:
                        # Append stroke2 to end
                        current_points.extend(stroke2.points)
                        used.add(j)
                        changed = True
                    elif dist_end_end <= max_gap:
                        # Append reversed stroke2 to end
                        current_points.extend(reversed(stroke2.points))
                        used.add(j)
                        changed = True
                    elif dist_start_start <= max_gap:
                        # Prepend reversed stroke2 to start
                        current_points = list(reversed(stroke2.points)) + current_points
                        used.add(j)
                        changed = True
                    elif dist_start_end <= max_gap:
                        # Prepend stroke2 to start
                        current_points = list(stroke2.points) + current_points
                        used.add(j)
                        changed = True

            if len(current_points) >= self.min_stroke_length:
                merged.append(Stroke(current_points))

        return merged


def trace_skeleton(skeleton: np.ndarray, min_length: int = 5) -> List[Stroke]:
    """Convenience function to trace a skeleton."""
    tracer = SkeletonTracer(min_stroke_length=min_length)
    return tracer.trace(skeleton)


# Test
if __name__ == '__main__':
    from PIL import Image, ImageDraw, ImageFont
    from skimage.morphology import skeletonize
    import matplotlib.pyplot as plt

    # Render test text
    font_path = "/home/server/glossy/font_scraper/fonts/dafont/Angelta Script.ttf"
    font_size = 200
    text = "Hello"

    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)
    padding = 30
    width = bbox[2] - bbox[0] + 2 * padding
    height = bbox[3] - bbox[1] + 2 * padding

    img = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=0)

    binary = np.array(img) < 128
    skeleton = skeletonize(binary)
    pruned = prune_spurs(skeleton, min_length=8)

    # Trace with new algorithm
    strokes = trace_skeleton(skeleton)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(binary, cmap='gray')
    axes[0, 0].set_title('1. Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(skeleton, cmap='gray')
    axes[0, 1].set_title('2. Raw Skeleton')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(pruned, cmap='gray')
    axes[1, 0].set_title('3. Pruned (spurs removed)')
    axes[1, 0].axis('off')

    axes[1, 1].set_xlim(0, width)
    axes[1, 1].set_ylim(height, 0)
    axes[1, 1].set_aspect('equal')

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(strokes), 1)))
    for i, stroke in enumerate(strokes):
        pts = stroke.to_array()
        axes[1, 1].plot(pts[:, 0], pts[:, 1], '-', color=colors[i % 10], linewidth=2.5)
        # Mark endpoints
        axes[1, 1].plot(pts[0, 0], pts[0, 1], 'o', color=colors[i % 10], markersize=6)
        axes[1, 1].plot(pts[-1, 0], pts[-1, 1], 's', color=colors[i % 10], markersize=6)

    axes[1, 1].set_title(f'4. Traced: {len(strokes)} strokes')
    axes[1, 1].set_facecolor('white')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('/home/server/glossy/font_scraper/improved_trace.png', dpi=150, facecolor='white')
    print(f"Traced {len(strokes)} strokes")
    for i, s in enumerate(strokes):
        print(f"  Stroke {i+1}: {len(s.points)} points, length={s.length():.1f}")
    print("Saved improved_trace.png")
