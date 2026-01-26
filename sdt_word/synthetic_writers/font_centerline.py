"""
Improved Centerline Extraction for Fonts

Uses medial axis transform + distance field for better centerline extraction.
Produces smooth, connected strokes suitable for pen plotting and SDT training.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import medial_axis, binary_dilation, disk
from skimage.graph import route_through_array
from scipy import ndimage
from scipy.interpolate import splprep, splev
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Stroke:
    points: List[Point] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        return np.array([[p.x, p.y] for p in self.points])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Stroke':
        return cls([Point(x, y) for x, y in arr])

    def smooth_spline(self, smoothing: float = 1.0, num_points: int = None) -> 'Stroke':
        """Smooth stroke using B-spline interpolation."""
        if len(self.points) < 4:
            return self

        pts = self.to_array()

        # Remove duplicate consecutive points
        diffs = np.diff(pts, axis=0)
        mask = np.concatenate([[True], np.any(diffs != 0, axis=1)])
        pts = pts[mask]

        if len(pts) < 4:
            return self

        try:
            # Fit B-spline
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=smoothing, k=min(3, len(pts)-1))

            # Evaluate at more points for smoother curve
            if num_points is None:
                num_points = max(len(pts), int(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) / 2))

            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)

            return Stroke([Point(x, y) for x, y in zip(x_new, y_new)])
        except Exception:
            return self


class FontCenterline:
    """Extract centerlines from outline fonts using medial axis transform."""

    def __init__(
        self,
        font_path: str,
        font_size: int = 300,  # Larger for better detail
        padding: int = 30,
        min_stroke_length: int = 8,
        smoothing: float = 2.0
    ):
        self.font_path = font_path
        self.font_size = font_size
        self.padding = padding
        self.min_stroke_length = min_stroke_length
        self.smoothing = smoothing

        self.font = ImageFont.truetype(font_path, font_size)

    def extract_char(self, char: str) -> Tuple[List[Stroke], Tuple[int, int]]:
        """Extract centerline strokes from a character."""

        # 1. Render character at high resolution
        img, bbox = self._render_char(char)
        if img is None:
            return [], (0, 0)

        # 2. Get medial axis with distance transform
        medial, distances = medial_axis(img, return_distance=True)

        # 3. Weight by distance (prefer center of thick strokes)
        weighted = medial * distances

        # 4. Extract ordered strokes by tracing
        strokes = self._trace_medial_axis(medial, distances)

        # 5. Smooth strokes
        smoothed = []
        for stroke in strokes:
            if len(stroke.points) >= 4:
                stroke = stroke.smooth_spline(self.smoothing)
            if len(stroke.points) >= 2:
                smoothed.append(stroke)

        width = bbox[2] - bbox[0] if bbox else self.font_size
        height = bbox[3] - bbox[1] if bbox else self.font_size

        return smoothed, (width, height)

    def _render_char(self, char: str) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        """Render character to binary bitmap."""
        bbox = self.font.getbbox(char)
        if not bbox:
            return None, None

        left, top, right, bottom = bbox
        width = right - left + 2 * self.padding
        height = bottom - top + 2 * self.padding

        if width <= 0 or height <= 0:
            return None, None

        # Render at 2x for antialiasing then threshold
        scale = 2
        img = Image.new('L', (width * scale, height * scale), 255)
        draw = ImageDraw.Draw(img)

        # Use scaled font
        font_scaled = ImageFont.truetype(self.font_path, self.font_size * scale)
        draw.text(
            ((self.padding - left) * scale, (self.padding - top) * scale),
            char, font=font_scaled, fill=0
        )

        # Downsample and threshold
        img = img.resize((width, height), Image.LANCZOS)
        arr = np.array(img)
        binary = arr < 128

        return binary, bbox

    def _trace_medial_axis(self, medial: np.ndarray, distances: np.ndarray) -> List[Stroke]:
        """Trace medial axis into ordered strokes."""
        if not medial.any():
            return []

        # Find endpoints (1 neighbor) and junctions (3+ neighbors)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = ndimage.convolve(medial.astype(int), kernel, mode='constant')
        neighbor_count = neighbor_count * medial

        endpoints = set(zip(*np.where((neighbor_count == 1) & medial)))
        junctions = set(zip(*np.where((neighbor_count >= 3) & medial)))

        strokes = []
        visited = np.zeros_like(medial, dtype=bool)

        # Trace from endpoints first (produces cleaner strokes)
        for start in sorted(endpoints, key=lambda p: distances[p], reverse=True):
            if visited[start]:
                continue
            stroke_points = self._trace_path(medial, visited, start, endpoints, junctions)
            if len(stroke_points) >= self.min_stroke_length:
                strokes.append(Stroke.from_array(np.array(stroke_points)))

        # Then trace from junctions
        for start in junctions:
            if visited[start]:
                continue
            # Find unvisited neighbors and trace from each
            for ny, nx in self._get_neighbors(start, medial, visited):
                if visited[ny, nx]:
                    continue
                stroke_points = self._trace_path(medial, visited, (ny, nx), endpoints, junctions)
                if len(stroke_points) >= self.min_stroke_length:
                    # Prepend junction point
                    stroke_points = [(start[1], start[0])] + stroke_points
                    strokes.append(Stroke.from_array(np.array(stroke_points)))

        # Handle remaining (loops)
        remaining = np.argwhere(medial & ~visited)
        for start in remaining:
            start = tuple(start)
            if visited[start]:
                continue
            stroke_points = self._trace_path(medial, visited, start, endpoints, junctions)
            if len(stroke_points) >= self.min_stroke_length:
                strokes.append(Stroke.from_array(np.array(stroke_points)))

        return strokes

    def _trace_path(
        self,
        medial: np.ndarray,
        visited: np.ndarray,
        start: Tuple[int, int],
        endpoints: set,
        junctions: set
    ) -> List[Tuple[float, float]]:
        """Trace a single path from start point."""
        path = []
        current = start

        while current is not None:
            y, x = current
            if visited[y, x]:
                break

            visited[y, x] = True
            path.append((float(x), float(y)))  # Convert to (x, y)

            # Stop at junctions (but include the junction point)
            if current in junctions and len(path) > 1:
                break

            # Find next unvisited neighbor
            neighbors = self._get_neighbors(current, medial, visited)

            if not neighbors:
                current = None
            elif len(neighbors) == 1:
                current = neighbors[0]
            else:
                # Multiple neighbors - pick the one most aligned with current direction
                if len(path) >= 2:
                    # Direction from previous point
                    dy = path[-1][1] - path[-2][1]
                    dx = path[-1][0] - path[-2][0]

                    # Pick neighbor most aligned
                    best = None
                    best_score = -2
                    for ny, nx in neighbors:
                        ndy = ny - y
                        ndx = nx - x
                        # Dot product (normalized)
                        score = (dx * ndx + dy * ndy) / (np.sqrt(dx*dx + dy*dy + 1e-6) * np.sqrt(ndx*ndx + ndy*ndy + 1e-6))
                        if score > best_score:
                            best_score = score
                            best = (ny, nx)
                    current = best
                else:
                    current = neighbors[0]

        return path

    def _get_neighbors(
        self,
        point: Tuple[int, int],
        medial: np.ndarray,
        visited: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Get unvisited neighbors of a point."""
        y, x = point
        neighbors = []

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (0 <= ny < medial.shape[0] and
                    0 <= nx < medial.shape[1] and
                    medial[ny, nx] and
                    not visited[ny, nx]):
                    neighbors.append((ny, nx))

        return neighbors

    def convert_text(self, text: str) -> Tuple[List[Stroke], List[Tuple[float, float]]]:
        """Convert text string to strokes with character positions."""
        all_strokes = []
        char_positions = []

        x_offset = 0

        for char in text:
            if char == ' ':
                x_offset += self.font_size * 0.3
                continue

            strokes, (width, height) = self.extract_char(char)

            # Offset strokes by current position
            for stroke in strokes:
                offset_points = [Point(p.x + x_offset, p.y) for p in stroke.points]
                all_strokes.append(Stroke(offset_points))

            char_positions.append((x_offset, 0))
            x_offset += width + self.font_size * 0.05  # Small kerning

        return all_strokes, char_positions

    def to_svg(self, strokes: List[Stroke], width: int, height: int) -> str:
        """Convert strokes to SVG string."""
        paths = []
        for stroke in strokes:
            if len(stroke.points) < 2:
                continue

            d = f"M {stroke.points[0].x:.1f} {stroke.points[0].y:.1f}"
            for p in stroke.points[1:]:
                d += f" L {p.x:.1f} {p.y:.1f}"

            paths.append(f'  <path d="{d}" fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')

        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
{chr(10).join(paths)}
</svg>'''


def compare_methods(font_path: str, text: str = "Hello", output_path: str = "centerline_comparison.png"):
    """Compare old skeletonize vs new medial axis method."""
    from PIL import Image, ImageDraw, ImageFont
    from skimage.morphology import skeletonize, medial_axis
    import matplotlib.pyplot as plt

    font_size = 200
    font = ImageFont.truetype(font_path, font_size)

    # Render text
    bbox = font.getbbox(text)
    padding = 30
    width = bbox[2] - bbox[0] + 2 * padding
    height = bbox[3] - bbox[1] + 2 * padding

    img = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=0)

    binary = np.array(img) < 128

    # Method 1: Basic skeletonize
    skeleton = skeletonize(binary)

    # Method 2: Medial axis
    medial, distances = medial_axis(binary, return_distance=True)

    # Method 3: Our improved centerline
    extractor = FontCenterline(font_path, font_size=font_size)
    strokes, _ = extractor.convert_text(text)

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original
    axes[0, 0].imshow(binary, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=14)
    axes[0, 0].axis('off')

    # Basic skeleton
    axes[0, 1].imshow(skeleton, cmap='gray')
    axes[0, 1].set_title('Basic Skeletonize (old method)', fontsize=14)
    axes[0, 1].axis('off')

    # Medial axis raw
    axes[1, 0].imshow(medial * distances, cmap='hot')
    axes[1, 0].set_title('Medial Axis + Distance', fontsize=14)
    axes[1, 0].axis('off')

    # Our smoothed strokes
    axes[1, 1].set_xlim(0, width)
    axes[1, 1].set_ylim(height, 0)  # Flip Y
    axes[1, 1].set_aspect('equal')
    for stroke in strokes:
        pts = stroke.to_array()
        axes[1, 1].plot(pts[:, 0], pts[:, 1], 'b-', linewidth=2)
    axes[1, 1].set_title('Smoothed Centerline Strokes (new method)', fontsize=14)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Comparison saved to {output_path}")
    return output_path


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python font_centerline.py <font.ttf> [text] [output.png]")
        sys.exit(1)

    font_path = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else "Hello"
    output = sys.argv[3] if len(sys.argv) > 3 else "centerline_comparison.png"

    compare_methods(font_path, text, output)
