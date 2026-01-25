"""
Synthetic Writer Generation Pipeline for SDT Training

Transforms single-line fonts into diverse synthetic "writers" by applying
programmatic variations that preserve internal style consistency.

Usage:
    python generate.py --fonts_dir ./fonts --output_dir ./synthetic_data --num_writers 1000
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import json
import random
@dataclass
class WriterStyle:
    """Global style parameters that define a synthetic writer's characteristics."""
    
    # Identity
    writer_id: str
    base_font: str
    
    # Global transforms
    slant: float = 0.0              # -0.3 to 0.3 (shear factor)
    aspect_ratio: float = 1.0       # 0.8 to 1.2 (width/height)
    baseline_wave_amp: float = 0.0  # 0 to 5 pixels
    baseline_wave_freq: float = 0.1 # cycles per character
    global_scale: float = 1.0       # 0.8 to 1.2
    letter_spacing: float = 1.0     # 0.8 to 1.3 multiplier
    
    # Per-character jitter ranges (sampled each character)
    char_rotation_std: float = 0.0    # degrees, std dev
    char_scale_std: float = 0.0       # multiplier std dev
    char_offset_x_std: float = 0.0    # pixels std dev
    char_offset_y_std: float = 0.0    # pixels std dev
    
    # Per-stroke variation
    stroke_curvature_noise: float = 0.0  # control point displacement
    stroke_start_drift: float = 0.0       # pixels
    stroke_end_drift: float = 0.0         # pixels
    
    # Per-point noise
    point_noise_std: float = 0.0     # gaussian noise on coordinates
    tremor_amplitude: float = 0.0    # high-freq noise amplitude
    tremor_frequency: float = 0.0    # tremor cycles per point
    
    # Speed simulation (affects point density)
    speed_variation: float = 0.0     # 0 = uniform, 1 = high variation
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'WriterStyle':
        return cls(**d)


from svg_parser import Point, Stroke, Glyph


class StyleTransformer:
    """Applies WriterStyle transforms to glyphs."""
    
    def __init__(self, style: WriterStyle, seed: Optional[int] = None):
        self.style = style
        self.rng = np.random.RandomState(seed)
        self.char_index = 0  # for baseline wave
        
    def transform_glyph(self, glyph: Glyph) -> Glyph:
        """Apply all transforms to a glyph."""
        # Work on a copy
        strokes = [Stroke([Point(p.x, p.y, p.t) for p in s.points]) for s in glyph.strokes]
        
        cx, cy = glyph.get_center()
        
        # 1. Apply global transforms
        strokes = self._apply_slant(strokes, cx, cy)
        strokes = self._apply_aspect_ratio(strokes, cx, cy)
        strokes = self._apply_global_scale(strokes, cx, cy)
        
        # 2. Apply per-character jitter
        strokes = self._apply_char_jitter(strokes, cx, cy)
        
        # 3. Apply baseline wave
        strokes = self._apply_baseline_wave(strokes)
        
        # 4. Apply per-stroke variation
        strokes = [self._apply_stroke_variation(s) for s in strokes]
        
        # 5. Apply per-point noise
        strokes = [self._apply_point_noise(s) for s in strokes]
        
        # 6. Apply speed-based resampling
        if self.style.speed_variation > 0:
            strokes = [self._apply_speed_variation(s) for s in strokes]
        
        self.char_index += 1
        
        return Glyph(
            char=glyph.char,
            unicode=glyph.unicode,
            strokes=strokes,
            width=glyph.width * self.style.letter_spacing
        )
    
    def _apply_slant(self, strokes: List[Stroke], cx: float, cy: float) -> List[Stroke]:
        """Apply italic-style shear transform."""
        if self.style.slant == 0:
            return strokes
        
        result = []
        for stroke in strokes:
            new_points = []
            for p in stroke.points:
                # Shear: x' = x + slant * (y - cy)
                new_x = p.x + self.style.slant * (p.y - cy)
                new_points.append(Point(new_x, p.y, p.t))
            result.append(Stroke(new_points))
        return result
    
    def _apply_aspect_ratio(self, strokes: List[Stroke], cx: float, cy: float) -> List[Stroke]:
        """Scale width relative to height."""
        if self.style.aspect_ratio == 1.0:
            return strokes
        
        return [s.scale(self.style.aspect_ratio, 1.0, cx, cy) for s in strokes]
    
    def _apply_global_scale(self, strokes: List[Stroke], cx: float, cy: float) -> List[Stroke]:
        """Apply uniform scaling."""
        if self.style.global_scale == 1.0:
            return strokes
        
        s = self.style.global_scale
        return [stroke.scale(s, s, cx, cy) for stroke in strokes]
    
    def _apply_char_jitter(self, strokes: List[Stroke], cx: float, cy: float) -> List[Stroke]:
        """Apply per-character random variations."""
        # Sample jitter values for this character
        rotation = self.rng.normal(0, self.style.char_rotation_std) if self.style.char_rotation_std > 0 else 0
        scale = 1.0 + self.rng.normal(0, self.style.char_scale_std) if self.style.char_scale_std > 0 else 1.0
        offset_x = self.rng.normal(0, self.style.char_offset_x_std) if self.style.char_offset_x_std > 0 else 0
        offset_y = self.rng.normal(0, self.style.char_offset_y_std) if self.style.char_offset_y_std > 0 else 0
        
        result = strokes
        
        # Apply rotation around center
        if rotation != 0:
            theta = np.radians(rotation)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            new_strokes = []
            for stroke in result:
                new_points = []
                for p in stroke.points:
                    dx, dy = p.x - cx, p.y - cy
                    new_x = cx + dx * cos_t - dy * sin_t
                    new_y = cy + dx * sin_t + dy * cos_t
                    new_points.append(Point(new_x, new_y, p.t))
                new_strokes.append(Stroke(new_points))
            result = new_strokes
        
        # Apply scale
        if scale != 1.0:
            result = [s.scale(scale, scale, cx, cy) for s in result]
        
        # Apply offset
        if offset_x != 0 or offset_y != 0:
            result = [s.translate(offset_x, offset_y) for s in result]
        
        return result
    
    def _apply_baseline_wave(self, strokes: List[Stroke]) -> List[Stroke]:
        """Apply sinusoidal baseline variation."""
        if self.style.baseline_wave_amp == 0:
            return strokes
        
        phase = self.char_index * self.style.baseline_wave_freq * 2 * np.pi
        offset = self.style.baseline_wave_amp * np.sin(phase)
        
        return [s.translate(0, offset) for s in strokes]
    
    def _apply_stroke_variation(self, stroke: Stroke) -> Stroke:
        """Apply per-stroke random variations."""
        if len(stroke.points) < 2:
            return stroke
        
        points = [Point(p.x, p.y, p.t) for p in stroke.points]
        
        # Drift start point
        if self.style.stroke_start_drift > 0:
            dx = self.rng.normal(0, self.style.stroke_start_drift)
            dy = self.rng.normal(0, self.style.stroke_start_drift)
            points[0] = Point(points[0].x + dx, points[0].y + dy, points[0].t)
        
        # Drift end point
        if self.style.stroke_end_drift > 0:
            dx = self.rng.normal(0, self.style.stroke_end_drift)
            dy = self.rng.normal(0, self.style.stroke_end_drift)
            points[-1] = Point(points[-1].x + dx, points[-1].y + dy, points[-1].t)
        
        # Curvature noise on middle points
        if self.style.stroke_curvature_noise > 0 and len(points) > 2:
            for i in range(1, len(points) - 1):
                # Displace perpendicular to stroke direction
                prev_p, curr_p, next_p = points[i-1], points[i], points[i+1]
                dx = next_p.x - prev_p.x
                dy = next_p.y - prev_p.y
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    # Perpendicular direction
                    perp_x, perp_y = -dy/length, dx/length
                    displacement = self.rng.normal(0, self.style.stroke_curvature_noise)
                    points[i] = Point(
                        curr_p.x + perp_x * displacement,
                        curr_p.y + perp_y * displacement,
                        curr_p.t
                    )
        
        return Stroke(points)
    
    def _apply_point_noise(self, stroke: Stroke) -> Stroke:
        """Apply per-point gaussian noise and tremor."""
        if self.style.point_noise_std == 0 and self.style.tremor_amplitude == 0:
            return stroke
        
        new_points = []
        for i, p in enumerate(stroke.points):
            new_x, new_y = p.x, p.y
            
            # Gaussian noise
            if self.style.point_noise_std > 0:
                new_x += self.rng.normal(0, self.style.point_noise_std)
                new_y += self.rng.normal(0, self.style.point_noise_std)
            
            # Tremor (high-frequency sinusoidal noise)
            if self.style.tremor_amplitude > 0:
                phase = i * self.style.tremor_frequency * 2 * np.pi
                # Add tremor perpendicular to approximate stroke direction
                new_x += self.style.tremor_amplitude * np.sin(phase)
                new_y += self.style.tremor_amplitude * np.cos(phase * 1.3)  # slightly different freq
            
            new_points.append(Point(new_x, new_y, p.t))
        
        return Stroke(new_points)
    
    def _apply_speed_variation(self, stroke: Stroke) -> Stroke:
        """Resample stroke to simulate variable writing speed."""
        if len(stroke.points) < 3:
            return stroke
        
        # Convert to array for easier manipulation
        arr = stroke.to_array()
        
        # Compute cumulative arc length
        diffs = np.diff(arr[:, :2], axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cum_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cum_length[-1]
        
        if total_length == 0:
            return stroke
        
        # Generate variable-speed sampling
        # More points on curves (high curvature), fewer on straight segments
        n_points = len(arr)
        
        # Simple speed variation: sinusoidal density variation
        t_uniform = np.linspace(0, 1, n_points * 2)  # oversample
        speed_mod = 1.0 + self.style.speed_variation * np.sin(t_uniform * np.pi * 3)
        speed_mod = speed_mod / speed_mod.mean()  # normalize
        
        # Convert to arc length positions
        cum_speed = np.cumsum(speed_mod)
        cum_speed = cum_speed / cum_speed[-1] * total_length
        
        # Resample to ~original point count
        target_positions = cum_speed[::2][:n_points]
        
        # Interpolate
        new_x = np.interp(target_positions, cum_length, arr[:, 0])
        new_y = np.interp(target_positions, cum_length, arr[:, 1])
        new_t = np.interp(target_positions, cum_length, arr[:, 2])
        
        new_arr = np.column_stack([new_x, new_y, new_t])
        return Stroke.from_array(new_arr)


class WriterGenerator:
    """Generates diverse synthetic writer styles from base fonts."""
    
    # Style parameter ranges for random generation
    PARAM_RANGES = {
        'slant': (-0.25, 0.25),
        'aspect_ratio': (0.85, 1.15),
        'baseline_wave_amp': (0, 3),
        'baseline_wave_freq': (0.05, 0.2),
        'global_scale': (0.85, 1.15),
        'letter_spacing': (0.9, 1.2),
        'char_rotation_std': (0, 3),
        'char_scale_std': (0, 0.05),
        'char_offset_x_std': (0, 2),
        'char_offset_y_std': (0, 2),
        'stroke_curvature_noise': (0, 2),
        'stroke_start_drift': (0, 1.5),
        'stroke_end_drift': (0, 1.5),
        'point_noise_std': (0, 1),
        'tremor_amplitude': (0, 0.5),
        'tremor_frequency': (0.3, 0.8),
        'speed_variation': (0, 0.3),
    }
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def generate_style(self, writer_id: str, base_font: str) -> WriterStyle:
        """Generate a random but coherent writer style."""
        params = {'writer_id': writer_id, 'base_font': base_font}
        
        for param, (low, high) in self.PARAM_RANGES.items():
            # Use beta distribution to favor middle values with occasional extremes
            alpha, beta = 2, 2
            value = low + (high - low) * self.rng.beta(alpha, beta)
            params[param] = value
        
        return WriterStyle(**params)
    
    def generate_style_cluster(
        self, 
        base_style: WriterStyle, 
        n_variations: int,
        variation_scale: float = 0.3
    ) -> List[WriterStyle]:
        """Generate a cluster of similar styles (same person, different days)."""
        styles = [base_style]
        
        for i in range(n_variations - 1):
            params = base_style.to_dict()
            params['writer_id'] = f"{base_style.writer_id}_v{i+1}"
            
            # Add small variations to each parameter
            for param, (low, high) in self.PARAM_RANGES.items():
                if param in params:
                    range_size = high - low
                    noise = self.rng.normal(0, range_size * variation_scale * 0.1)
                    params[param] = np.clip(params[param] + noise, low, high)
            
            styles.append(WriterStyle(**params))
        
        return styles


def generate_training_sample(
    glyph: Glyph,
    style: WriterStyle,
    seed: Optional[int] = None
) -> dict:
    """Generate a single training sample in SDT format."""
    transformer = StyleTransformer(style, seed)
    transformed = transformer.transform_glyph(glyph)
    
    # Convert to SDT format: list of strokes, each stroke is list of [x, y, pen_state]
    strokes_data = []
    for stroke in transformed.strokes:
        stroke_points = []
        for i, p in enumerate(stroke.points):
            # pen_state: 0 = pen down (drawing), 1 = pen up (last point of stroke)
            pen_state = 1 if i == len(stroke.points) - 1 else 0
            stroke_points.append([p.x, p.y, pen_state])
        strokes_data.extend(stroke_points)
    
    return {
        'char': glyph.char,
        'writer_id': style.writer_id,
        'strokes': strokes_data,
        'style_params': style.to_dict()
    }


# Example usage and CLI
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic writers for SDT training')
    parser.add_argument('--fonts_dir', type=str, required=True, help='Directory containing SVG fonts')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_writers', type=int, default=1000, help='Number of synthetic writers')
    parser.add_argument('--samples_per_writer', type=int, default=52, help='Samples per writer (chars)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_writers} synthetic writers...")
    print(f"Fonts dir: {args.fonts_dir}")
    print(f"Output dir: {args.output_dir}")
    
    # This is where you'd integrate with actual font loading
    # For now, just demonstrate the structure
    generator = WriterGenerator(seed=args.seed)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate and save writer styles
    styles = []
    for i in range(args.num_writers):
        style = generator.generate_style(
            writer_id=f"synthetic_{i:05d}",
            base_font=f"font_{i % 100:03d}"  # cycle through ~100 base fonts
        )
        styles.append(style.to_dict())
    
    with open(output_path / 'writer_styles.json', 'w') as f:
        json.dump(styles, f, indent=2)
    
    print(f"Saved {len(styles)} writer styles to {output_path / 'writer_styles.json'}")
