"""
Convert Outline Fonts to Single-Line Fonts

Takes TTF/OTF handwriting fonts and converts them to single-line stroke data
suitable for pen plotters and SDT training.

Process:
1. Render glyph to high-res bitmap
2. Skeletonize (morphological thinning)
3. Trace skeleton to vector paths
4. Clean up and smooth

Requirements:
    pip install fonttools pillow scikit-image numpy scipy

Usage:
    python font_to_single_line.py --font input.ttf --output output_dir
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json
import numpy as np

# Image processing
from PIL import Image, ImageDraw, ImageFont

# Skeletonization
from skimage.morphology import skeletonize, remove_small_objects
from skimage import img_as_bool

# Path tracing
from scipy import ndimage
from collections import deque


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Stroke:
    points: List[Point] = field(default_factory=list)
    
    def simplify(self, tolerance: float = 1.0) -> 'Stroke':
        """Simplify stroke using Ramer-Douglas-Peucker algorithm."""
        if len(self.points) < 3:
            return self
        
        points = np.array([[p.x, p.y] for p in self.points])
        simplified = rdp_simplify(points, tolerance)
        return Stroke([Point(x, y) for x, y in simplified])
    
    def smooth(self, window: int = 3) -> 'Stroke':
        """Smooth stroke with moving average."""
        if len(self.points) < window:
            return self
        
        points = np.array([[p.x, p.y] for p in self.points])
        
        # Pad for edge handling
        padded = np.pad(points, ((window//2, window//2), (0, 0)), mode='edge')
        
        # Moving average
        kernel = np.ones(window) / window
        smoothed_x = np.convolve(padded[:, 0], kernel, mode='valid')
        smoothed_y = np.convolve(padded[:, 1], kernel, mode='valid')
        
        return Stroke([Point(x, y) for x, y in zip(smoothed_x, smoothed_y)])


@dataclass
class SingleLineGlyph:
    char: str
    strokes: List[Stroke] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0


def rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker line simplification."""
    if len(points) < 3:
        return points
    
    # Find point with max distance from line between first and last
    start, end = points[0], points[-1]
    
    # Line vector
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    if line_len == 0:
        return np.array([start, end])
    
    line_unit = line_vec / line_len
    
    # Perpendicular distances
    vecs = points - start
    proj_lengths = np.dot(vecs, line_unit)
    proj_points = start + np.outer(proj_lengths, line_unit)
    distances = np.linalg.norm(points - proj_points, axis=1)
    
    max_idx = np.argmax(distances)
    max_dist = distances[max_idx]
    
    if max_dist > epsilon:
        # Recursively simplify
        left = rdp_simplify(points[:max_idx + 1], epsilon)
        right = rdp_simplify(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


class FontToSingleLine:
    """Convert outline fonts to single-line strokes."""
    
    def __init__(
        self,
        font_path: str,
        font_size: int = 200,
        padding: int = 20,
        min_stroke_length: int = 5,
        simplify_tolerance: float = 1.5,
        smooth_window: int = 3
    ):
        self.font_path = font_path
        self.font_size = font_size
        self.padding = padding
        self.min_stroke_length = min_stroke_length
        self.simplify_tolerance = simplify_tolerance
        self.smooth_window = smooth_window
        
        # Load font
        self.font = ImageFont.truetype(font_path, font_size)
    
    def convert_char(self, char: str) -> Optional[SingleLineGlyph]:
        """Convert a single character to single-line strokes."""
        
        # 1. Render to bitmap
        bitmap, bbox = self._render_char(char)
        if bitmap is None:
            return None
        
        # 2. Skeletonize
        skeleton = self._skeletonize(bitmap)
        
        # 3. Trace to strokes
        strokes = self._trace_skeleton(skeleton)
        
        # 4. Clean up
        strokes = self._cleanup_strokes(strokes)
        
        if not strokes:
            return None
        
        # Calculate dimensions
        width = bbox[2] - bbox[0] if bbox else self.font_size
        height = bbox[3] - bbox[1] if bbox else self.font_size
        
        return SingleLineGlyph(
            char=char,
            strokes=strokes,
            width=width,
            height=height
        )
    
    def _render_char(self, char: str) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        """Render character to binary bitmap."""
        # Get bounding box
        bbox = self.font.getbbox(char)
        if not bbox:
            return None, None
        
        left, top, right, bottom = bbox
        width = right - left + 2 * self.padding
        height = bottom - top + 2 * self.padding
        
        if width <= 0 or height <= 0:
            return None, None
        
        # Create image
        img = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(img)
        
        # Draw character
        draw.text((self.padding - left, self.padding - top), char, font=self.font, fill=0)
        
        # Convert to binary numpy array (True = ink)
        arr = np.array(img)
        binary = arr < 128
        
        return binary, bbox
    
    def _skeletonize(self, bitmap: np.ndarray) -> np.ndarray:
        """Skeletonize bitmap to 1-pixel-wide centerline."""
        # Remove small noise
        cleaned = remove_small_objects(bitmap, min_size=10)
        
        # Skeletonize
        skeleton = skeletonize(cleaned)
        
        return skeleton
    
    def _trace_skeleton(self, skeleton: np.ndarray) -> List[Stroke]:
        """Trace skeleton pixels into connected strokes."""
        if not skeleton.any():
            return []
        
        # Find all skeleton pixels
        coords = np.argwhere(skeleton)
        if len(coords) == 0:
            return []
        
        # Build adjacency - find connected components and trace paths
        strokes = []
        visited = np.zeros_like(skeleton, dtype=bool)
        
        # Find endpoints and junction points
        endpoints, junctions = self._find_special_points(skeleton)
        
        # Start tracing from endpoints first, then any unvisited points
        start_points = list(endpoints) + list(junctions)
        
        for start in start_points:
            if visited[start[0], start[1]]:
                continue
            
            # Trace from this point
            traced_strokes = self._trace_from_point(skeleton, visited, start)
            strokes.extend(traced_strokes)
        
        # Handle any remaining unvisited pixels (closed loops)
        remaining = np.argwhere(skeleton & ~visited)
        for start in remaining:
            if visited[start[0], start[1]]:
                continue
            traced_strokes = self._trace_from_point(skeleton, visited, tuple(start))
            strokes.extend(traced_strokes)
        
        return strokes
    
    def _find_special_points(self, skeleton: np.ndarray) -> Tuple[List[Tuple], List[Tuple]]:
        """Find endpoints (1 neighbor) and junctions (3+ neighbors)."""
        # Count neighbors for each pixel
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        neighbor_count = neighbor_count * skeleton  # Only count skeleton pixels
        
        endpoints = list(zip(*np.where((neighbor_count == 1) & skeleton)))
        junctions = list(zip(*np.where((neighbor_count >= 3) & skeleton)))
        
        return endpoints, junctions
    
    def _trace_from_point(
        self, 
        skeleton: np.ndarray, 
        visited: np.ndarray, 
        start: Tuple[int, int]
    ) -> List[Stroke]:
        """Trace connected path from a starting point."""
        strokes = []
        
        # 8-connectivity neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),          (0, 1),
                     (1, -1),  (1, 0), (1, 1)]
        
        # BFS to find all connected points, building strokes
        queue = deque([(start, [start])])
        visited[start[0], start[1]] = True
        
        while queue:
            (y, x), path = queue.popleft()
            
            # Find unvisited neighbors
            unvisited_neighbors = []
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if (0 <= ny < skeleton.shape[0] and 
                    0 <= nx < skeleton.shape[1] and
                    skeleton[ny, nx] and 
                    not visited[ny, nx]):
                    unvisited_neighbors.append((ny, nx))
            
            if len(unvisited_neighbors) == 0:
                # End of path - save stroke
                if len(path) >= self.min_stroke_length:
                    # Convert from (row, col) to (x, y)
                    points = [Point(float(p[1]), float(p[0])) for p in path]
                    strokes.append(Stroke(points))
            
            elif len(unvisited_neighbors) == 1:
                # Continue path
                ny, nx = unvisited_neighbors[0]
                visited[ny, nx] = True
                queue.append(((ny, nx), path + [(ny, nx)]))
            
            else:
                # Junction - save current stroke and start new ones
                if len(path) >= self.min_stroke_length:
                    points = [Point(float(p[1]), float(p[0])) for p in path]
                    strokes.append(Stroke(points))
                
                for ny, nx in unvisited_neighbors:
                    visited[ny, nx] = True
                    queue.append(((ny, nx), [(y, x), (ny, nx)]))
        
        return strokes
    
    def _cleanup_strokes(self, strokes: List[Stroke]) -> List[Stroke]:
        """Clean up strokes: simplify, smooth, remove short ones."""
        cleaned = []
        
        for stroke in strokes:
            if len(stroke.points) < self.min_stroke_length:
                continue
            
            # Simplify
            stroke = stroke.simplify(self.simplify_tolerance)
            
            # Smooth
            if self.smooth_window > 1:
                stroke = stroke.smooth(self.smooth_window)
            
            if len(stroke.points) >= 2:
                cleaned.append(stroke)
        
        return cleaned
    
    def convert_charset(self, chars: str) -> Dict[str, SingleLineGlyph]:
        """Convert multiple characters."""
        glyphs = {}
        
        for char in chars:
            glyph = self.convert_char(char)
            if glyph:
                glyphs[char] = glyph
        
        return glyphs
    
    def export_svg(self, glyphs: Dict[str, SingleLineGlyph], output_dir: str):
        """Export glyphs as individual SVG files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for char, glyph in glyphs.items():
            # Create safe filename
            if char.isalnum():
                filename = f"{char}.svg"
            else:
                filename = f"char_{ord(char):04x}.svg"
            
            svg = self._glyph_to_svg(glyph)
            (output_path / filename).write_text(svg)
        
        # Also create combined SVG font
        combined = self._glyphs_to_svg_font(glyphs)
        (output_path / "font.svg").write_text(combined)
    
    def _glyph_to_svg(self, glyph: SingleLineGlyph) -> str:
        """Convert single glyph to SVG."""
        width = glyph.width + 2 * self.padding
        height = glyph.height + 2 * self.padding
        
        paths = []
        for stroke in glyph.strokes:
            if len(stroke.points) < 2:
                continue
            
            d = f"M {stroke.points[0].x:.2f} {stroke.points[0].y:.2f}"
            for p in stroke.points[1:]:
                d += f" L {p.x:.2f} {p.y:.2f}"
            
            paths.append(f'  <path d="{d}" fill="none" stroke="black" stroke-width="1"/>')
        
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}">
{chr(10).join(paths)}
</svg>'''
    
    def _glyphs_to_svg_font(self, glyphs: Dict[str, SingleLineGlyph]) -> str:
        """Create SVG font file with all glyphs."""
        glyph_elements = []
        
        for char, glyph in glyphs.items():
            # Build path data
            d_parts = []
            for stroke in glyph.strokes:
                if len(stroke.points) < 2:
                    continue
                
                d = f"M {stroke.points[0].x:.2f} {stroke.points[0].y:.2f}"
                for p in stroke.points[1:]:
                    d += f" L {p.x:.2f} {p.y:.2f}"
                d_parts.append(d)
            
            if d_parts:
                full_d = " ".join(d_parts)
                # Escape special chars for XML
                escaped_char = char
                if char == '"':
                    escaped_char = "&quot;"
                elif char == '&':
                    escaped_char = "&amp;"
                elif char == '<':
                    escaped_char = "&lt;"
                elif char == '>':
                    escaped_char = "&gt;"
                
                glyph_elements.append(
                    f'  <glyph unicode="{escaped_char}" d="{full_d}" horiz-adv-x="{glyph.width:.0f}"/>'
                )
        
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
<font id="converted-font">
  <font-face font-family="ConvertedFont" units-per-em="{self.font_size}"/>
{chr(10).join(glyph_elements)}
</font>
</svg>'''


def main():
    parser = argparse.ArgumentParser(description='Convert outline fonts to single-line')
    parser.add_argument('--font', type=str, required=True, help='Input TTF/OTF font file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--size', type=int, default=200, help='Render size (default: 200)')
    parser.add_argument('--chars', type=str, 
                       default='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
                       help='Characters to convert')
    parser.add_argument('--simplify', type=float, default=1.5, help='Simplification tolerance')
    parser.add_argument('--smooth', type=int, default=3, help='Smoothing window size')
    
    args = parser.parse_args()
    
    print(f"Converting {args.font}...")
    
    converter = FontToSingleLine(
        font_path=args.font,
        font_size=args.size,
        simplify_tolerance=args.simplify,
        smooth_window=args.smooth
    )
    
    print(f"Converting {len(args.chars)} characters...")
    glyphs = converter.convert_charset(args.chars)
    
    print(f"Successfully converted {len(glyphs)} characters")
    
    print(f"Exporting to {args.output}...")
    converter.export_svg(glyphs, args.output)
    
    # Also save stroke data as JSON for direct use
    stroke_data = {}
    for char, glyph in glyphs.items():
        stroke_data[char] = {
            'strokes': [
                [[p.x, p.y] for p in stroke.points]
                for stroke in glyph.strokes
            ],
            'width': glyph.width,
            'height': glyph.height
        }
    
    output_path = Path(args.output)
    with open(output_path / 'strokes.json', 'w') as f:
        json.dump(stroke_data, f, indent=2)
    
    print("Done!")
    print(f"  SVG files: {args.output}/*.svg")
    print(f"  Combined font: {args.output}/font.svg")
    print(f"  Stroke data: {args.output}/strokes.json")


if __name__ == '__main__':
    main()
