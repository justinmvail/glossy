"""
SVG Font Parser for Single-Line Fonts

Parses SVG fonts and extracts stroke/path data as coordinate sequences
suitable for SDT training.

Single-line fonts typically use:
- <path d="M x y L x y ..."> elements
- <polyline points="x,y x,y ..."> elements
- Sometimes nested in <glyph> or <symbol> elements
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class Point:
    """A single point in a stroke."""
    x: float
    y: float
    t: float = 0.0  # parameter/timestamp
    
    def __repr__(self):
        return f"Point({self.x:.2f}, {self.y:.2f})"


@dataclass
class Stroke:
    """A continuous pen stroke (pen-down to pen-up)."""
    points: List[Point] = field(default_factory=list)
    
    def translate(self, dx: float, dy: float) -> 'Stroke':
        return Stroke([Point(p.x + dx, p.y + dy, p.t) for p in self.points])
    
    def scale(self, sx: float, sy: float, cx: float = 0, cy: float = 0) -> 'Stroke':
        return Stroke([
            Point(cx + (p.x - cx) * sx, cy + (p.y - cy) * sy, p.t)
            for p in self.points
        ])
    
    def flip_y(self, y_offset: float = 0) -> 'Stroke':
        """Flip Y coordinates (SVG has Y pointing down, we want Y pointing up)."""
        return Stroke([Point(p.x, y_offset - p.y, p.t) for p in self.points])
    
    def to_array(self) -> np.ndarray:
        return np.array([[p.x, p.y, p.t] for p in self.points])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Stroke':
        return cls([Point(x, y, t) for x, y, t in arr])
    
    def resample(self, num_points: int) -> 'Stroke':
        """Resample stroke to fixed number of points."""
        if len(self.points) < 2:
            return self
        
        arr = self.to_array()
        
        # Compute cumulative arc length
        diffs = np.diff(arr[:, :2], axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cum_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cum_length[-1]
        
        if total_length == 0:
            return self
        
        # New sample positions
        new_positions = np.linspace(0, total_length, num_points)
        
        # Interpolate
        new_x = np.interp(new_positions, cum_length, arr[:, 0])
        new_y = np.interp(new_positions, cum_length, arr[:, 1])
        new_t = np.linspace(0, 1, num_points)
        
        return Stroke([Point(x, y, t) for x, y, t in zip(new_x, new_y, new_t)])


@dataclass
class Glyph:
    """A character glyph composed of strokes."""
    char: str
    unicode: int
    strokes: List[Stroke] = field(default_factory=list)
    width: float = 0.0  # advance width
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, min_y, max_x, max_y)."""
        all_points = [p for s in self.strokes for p in s.points]
        if not all_points:
            return (0, 0, 0, 0)
        xs = [p.x for p in all_points]
        ys = [p.y for p in all_points]
        return (min(xs), min(ys), max(xs), max(ys))
    
    def get_center(self) -> Tuple[float, float]:
        min_x, min_y, max_x, max_y = self.get_bounds()
        return ((min_x + max_x) / 2, (min_y + max_y) / 2)
    
    def normalize(self, target_height: float = 100.0) -> 'Glyph':
        """Normalize glyph to standard height, centered at origin."""
        min_x, min_y, max_x, max_y = self.get_bounds()
        height = max_y - min_y
        
        if height == 0:
            return self
        
        scale = target_height / height
        cx, cy = self.get_center()
        
        new_strokes = []
        for stroke in self.strokes:
            # Center, then scale
            centered = stroke.translate(-cx, -cy)
            scaled = centered.scale(scale, scale)
            new_strokes.append(scaled)
        
        return Glyph(
            char=self.char,
            unicode=self.unicode,
            strokes=new_strokes,
            width=self.width * scale
        )
    
    def to_sdt_format(self) -> List[List[float]]:
        """Convert to SDT training format: [[x, y, pen_state], ...]"""
        result = []
        for stroke_idx, stroke in enumerate(self.strokes):
            for point_idx, point in enumerate(stroke.points):
                # pen_state: 0 = drawing, 1 = stroke end (pen up next)
                is_last = (point_idx == len(stroke.points) - 1)
                pen_state = 1 if is_last else 0
                result.append([point.x, point.y, pen_state])
        return result


class SVGPathParser:
    """Parse SVG path 'd' attribute into strokes."""
    
    # SVG path command regex
    COMMAND_RE = re.compile(r'([MmZzLlHhVvCcSsQqTtAa])|(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)')
    
    def __init__(self):
        self.current_x = 0.0
        self.current_y = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
    
    def parse(self, d: str) -> List[Stroke]:
        """Parse SVG path data string into list of strokes."""
        tokens = self._tokenize(d)
        strokes = []
        current_stroke = Stroke()
        
        i = 0
        current_command = None
        
        while i < len(tokens):
            token = tokens[i]
            
            if token.isalpha():
                current_command = token
                i += 1
            elif current_command:
                # Process command with its arguments
                stroke_result, i = self._process_command(
                    current_command, tokens, i, current_stroke
                )
                
                if stroke_result is not None:
                    # Command produced a new stroke (M command)
                    if current_stroke.points:
                        strokes.append(current_stroke)
                    current_stroke = stroke_result
            else:
                i += 1
        
        # Add final stroke
        if current_stroke.points:
            strokes.append(current_stroke)
        
        return strokes
    
    def _tokenize(self, d: str) -> List[str]:
        """Tokenize path data into commands and numbers."""
        tokens = []
        for match in self.COMMAND_RE.finditer(d):
            tokens.append(match.group())
        return tokens
    
    def _process_command(
        self, 
        cmd: str, 
        tokens: List[str], 
        i: int,
        current_stroke: Stroke
    ) -> Tuple[Optional[Stroke], int]:
        """Process a single SVG path command."""
        
        is_relative = cmd.islower()
        cmd_upper = cmd.upper()
        
        if cmd_upper == 'M':  # MoveTo
            x, y = float(tokens[i]), float(tokens[i + 1])
            if is_relative:
                x += self.current_x
                y += self.current_y
            
            self.current_x, self.current_y = x, y
            self.start_x, self.start_y = x, y
            
            # Start new stroke
            new_stroke = Stroke([Point(x, y)])
            return new_stroke, i + 2
        
        elif cmd_upper == 'L':  # LineTo
            x, y = float(tokens[i]), float(tokens[i + 1])
            if is_relative:
                x += self.current_x
                y += self.current_y
            
            # Add intermediate points for smoother line
            current_stroke.points.append(Point(x, y))
            self.current_x, self.current_y = x, y
            return None, i + 2
        
        elif cmd_upper == 'H':  # Horizontal LineTo
            x = float(tokens[i])
            if is_relative:
                x += self.current_x
            
            current_stroke.points.append(Point(x, self.current_y))
            self.current_x = x
            return None, i + 1
        
        elif cmd_upper == 'V':  # Vertical LineTo
            y = float(tokens[i])
            if is_relative:
                y += self.current_y
            
            current_stroke.points.append(Point(self.current_x, y))
            self.current_y = y
            return None, i + 1
        
        elif cmd_upper == 'C':  # Cubic Bezier
            x1, y1 = float(tokens[i]), float(tokens[i + 1])
            x2, y2 = float(tokens[i + 2]), float(tokens[i + 3])
            x, y = float(tokens[i + 4]), float(tokens[i + 5])
            
            if is_relative:
                x1 += self.current_x
                y1 += self.current_y
                x2 += self.current_x
                y2 += self.current_y
                x += self.current_x
                y += self.current_y
            
            # Sample cubic bezier
            points = self._sample_cubic_bezier(
                self.current_x, self.current_y,
                x1, y1, x2, y2, x, y,
                num_samples=10
            )
            current_stroke.points.extend(points[1:])  # Skip first (duplicate)
            
            self.current_x, self.current_y = x, y
            return None, i + 6
        
        elif cmd_upper == 'Q':  # Quadratic Bezier
            x1, y1 = float(tokens[i]), float(tokens[i + 1])
            x, y = float(tokens[i + 2]), float(tokens[i + 3])
            
            if is_relative:
                x1 += self.current_x
                y1 += self.current_y
                x += self.current_x
                y += self.current_y
            
            # Sample quadratic bezier
            points = self._sample_quadratic_bezier(
                self.current_x, self.current_y,
                x1, y1, x, y,
                num_samples=8
            )
            current_stroke.points.extend(points[1:])
            
            self.current_x, self.current_y = x, y
            return None, i + 4
        
        elif cmd_upper == 'Z':  # ClosePath
            # Draw line back to start
            if (self.current_x, self.current_y) != (self.start_x, self.start_y):
                current_stroke.points.append(Point(self.start_x, self.start_y))
            self.current_x, self.current_y = self.start_x, self.start_y
            return None, i
        
        elif cmd_upper == 'A':  # Arc - simplified handling
            # rx, ry, x-axis-rotation, large-arc-flag, sweep-flag, x, y
            rx = float(tokens[i])
            ry = float(tokens[i + 1])
            rotation = float(tokens[i + 2])
            large_arc = int(float(tokens[i + 3]))
            sweep = int(float(tokens[i + 4]))
            x, y = float(tokens[i + 5]), float(tokens[i + 6])
            
            if is_relative:
                x += self.current_x
                y += self.current_y
            
            # Approximate arc with line segments
            points = self._sample_arc(
                self.current_x, self.current_y,
                rx, ry, rotation, large_arc, sweep, x, y,
                num_samples=16
            )
            current_stroke.points.extend(points[1:])
            
            self.current_x, self.current_y = x, y
            return None, i + 7
        
        # Unhandled command, skip
        return None, i + 1
    
    def _sample_cubic_bezier(
        self, x0, y0, x1, y1, x2, y2, x3, y3, num_samples: int = 10
    ) -> List[Point]:
        """Sample points along a cubic bezier curve."""
        points = []
        for i in range(num_samples + 1):
            t = i / num_samples
            t2, t3 = t * t, t * t * t
            mt, mt2, mt3 = 1 - t, (1 - t) ** 2, (1 - t) ** 3
            
            x = mt3 * x0 + 3 * mt2 * t * x1 + 3 * mt * t2 * x2 + t3 * x3
            y = mt3 * y0 + 3 * mt2 * t * y1 + 3 * mt * t2 * y2 + t3 * y3
            points.append(Point(x, y, t))
        
        return points
    
    def _sample_quadratic_bezier(
        self, x0, y0, x1, y1, x2, y2, num_samples: int = 8
    ) -> List[Point]:
        """Sample points along a quadratic bezier curve."""
        points = []
        for i in range(num_samples + 1):
            t = i / num_samples
            mt = 1 - t
            
            x = mt * mt * x0 + 2 * mt * t * x1 + t * t * x2
            y = mt * mt * y0 + 2 * mt * t * y1 + t * t * y2
            points.append(Point(x, y, t))
        
        return points
    
    def _sample_arc(
        self, x0, y0, rx, ry, rotation, large_arc, sweep, x, y, num_samples: int = 16
    ) -> List[Point]:
        """Approximate SVG arc with line segments (simplified)."""
        # Simple linear interpolation fallback
        points = []
        for i in range(num_samples + 1):
            t = i / num_samples
            px = x0 + t * (x - x0)
            py = y0 + t * (y - y0)
            points.append(Point(px, py, t))
        return points


class SVGFontParser:
    """Parse SVG font files into Glyph objects."""
    
    def __init__(self):
        self.path_parser = SVGPathParser()
    
    def parse_file(self, filepath: str) -> Dict[str, Glyph]:
        """Parse SVG file and extract all glyphs."""
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Handle namespace
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        glyphs = {}
        
        # Try different SVG structures
        
        # 1. SVG Font format (<font> with <glyph> elements)
        for glyph_elem in root.iter('{http://www.w3.org/2000/svg}glyph'):
            glyph = self._parse_glyph_element(glyph_elem)
            if glyph:
                glyphs[glyph.char] = glyph
        
        # Also try without namespace
        for glyph_elem in root.iter('glyph'):
            glyph = self._parse_glyph_element(glyph_elem)
            if glyph:
                glyphs[glyph.char] = glyph
        
        # 2. Regular SVG with paths (common for single-character files)
        if not glyphs:
            paths = root.findall('.//{http://www.w3.org/2000/svg}path')
            paths.extend(root.findall('.//path'))
            
            for path_elem in paths:
                d = path_elem.get('d', '')
                if d:
                    strokes = self.path_parser.parse(d)
                    # Try to determine character from filename or id
                    char = path_elem.get('id', '?')
                    glyph = Glyph(
                        char=char,
                        unicode=ord(char[0]) if char else 0,
                        strokes=strokes
                    )
                    glyphs[char] = glyph
        
        return glyphs
    
    def _parse_glyph_element(self, elem) -> Optional[Glyph]:
        """Parse a <glyph> element from SVG font."""
        # Get character/unicode
        char = elem.get('unicode', '')
        glyph_name = elem.get('glyph-name', '')
        
        if not char and glyph_name:
            char = glyph_name
        
        if not char:
            return None
        
        # Get path data
        d = elem.get('d', '')
        if not d:
            return None
        
        # Parse strokes
        self.path_parser = SVGPathParser()  # Reset state
        strokes = self.path_parser.parse(d)
        
        # Get advance width
        width = float(elem.get('horiz-adv-x', 0))
        
        # Determine unicode value
        unicode_val = ord(char[0]) if len(char) == 1 else 0
        
        return Glyph(
            char=char,
            unicode=unicode_val,
            strokes=strokes,
            width=width
        )
    
    def parse_directory(self, dirpath: str) -> Dict[str, Dict[str, Glyph]]:
        """Parse all SVG files in a directory."""
        fonts = {}
        
        dir_path = Path(dirpath)
        for svg_file in dir_path.glob('**/*.svg'):
            font_name = svg_file.stem
            try:
                glyphs = self.parse_file(str(svg_file))
                if glyphs:
                    fonts[font_name] = glyphs
            except Exception as e:
                print(f"Error parsing {svg_file}: {e}")
        
        return fonts


def parse_svg_font(filepath: str) -> Dict[str, Glyph]:
    """Convenience function to parse a single SVG font file."""
    parser = SVGFontParser()
    return parser.parse_file(filepath)


# CLI for testing
if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python svg_parser.py <svg_file_or_directory>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    parser = SVGFontParser()
    
    if path.is_file():
        glyphs = parser.parse_file(str(path))
        print(f"Parsed {len(glyphs)} glyphs from {path}")
        for char, glyph in list(glyphs.items())[:5]:
            print(f"  '{char}': {len(glyph.strokes)} strokes, {sum(len(s.points) for s in glyph.strokes)} points")
    else:
        fonts = parser.parse_directory(str(path))
        print(f"Parsed {len(fonts)} fonts from {path}")
        for font_name, glyphs in list(fonts.items())[:3]:
            print(f"  {font_name}: {len(glyphs)} glyphs")
