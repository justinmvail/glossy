"""
Handwriting vs Font Classifier / Human-likeness Scorer

Scores fonts based on how natural/human-like they appear.
Uses several perceptual metrics that distinguish real handwriting from fonts.

Key features that make handwriting look human:
1. Stroke width variation (pressure changes)
2. Baseline wobble (not perfectly straight)
3. Letter inconsistency (same letter varies between instances)
4. Slight rotation/slant variation
5. Connection quality (natural pen lifts)
6. Edge roughness (not perfectly smooth curves)
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class HandwritingScore:
    """Detailed breakdown of handwriting quality scores."""
    name: str
    baseline_variation: float  # 0-1, higher = more natural wobble
    stroke_width_variation: float  # 0-1, higher = more pressure variation
    character_consistency: float  # 0-1, LOWER = more natural variation
    edge_roughness: float  # 0-1, higher = more natural irregularity
    slant_variation: float  # 0-1, higher = more natural tilt variation
    overall: float  # Combined score, higher = more human-like

    def to_dict(self):
        return {
            'name': self.name,
            'baseline_variation': self.baseline_variation,
            'stroke_width_variation': self.stroke_width_variation,
            'character_consistency': self.character_consistency,
            'edge_roughness': self.edge_roughness,
            'slant_variation': self.slant_variation,
            'overall': self.overall
        }


class HandwritingScorer:
    """Score fonts on how human-like they appear."""

    def __init__(self, font_size: int = 64):
        self.font_size = font_size
        self.test_chars = 'aehnotrs'  # Common chars to test
        self.test_text = "The quick brown fox"

    def score_font(self, font_path: str) -> Optional[HandwritingScore]:
        """Score a single font file."""
        try:
            font = ImageFont.truetype(font_path, self.font_size)
        except Exception as e:
            print(f"Error loading {font_path}: {e}")
            return None

        name = Path(font_path).stem

        # 1. Baseline variation
        baseline_var = self._measure_baseline_variation(font)

        # 2. Stroke width variation
        stroke_var = self._measure_stroke_variation(font)

        # 3. Character consistency (lower is more human-like)
        char_consistency = self._measure_character_consistency(font)

        # 4. Edge roughness
        edge_rough = self._measure_edge_roughness(font)

        # 5. Slant variation
        slant_var = self._measure_slant_variation(font)

        # Combine scores (weight towards variation metrics)
        # Note: char_consistency is inverted (lower = more human)
        overall = (
            baseline_var * 0.2 +
            stroke_var * 0.25 +
            (1.0 - char_consistency) * 0.2 +  # Invert consistency
            edge_rough * 0.2 +
            slant_var * 0.15
        )

        return HandwritingScore(
            name=name,
            baseline_variation=baseline_var,
            stroke_width_variation=stroke_var,
            character_consistency=char_consistency,
            edge_roughness=edge_rough,
            slant_variation=slant_var,
            overall=overall
        )

    def _render_text(self, font: ImageFont.FreeTypeFont, text: str,
                     width: int = 400, height: int = 100) -> np.ndarray:
        """Render text and return as numpy array."""
        img = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), text, font=font, fill=0)
        return np.array(img)

    def _measure_baseline_variation(self, font) -> float:
        """
        Measure how much the baseline varies.
        Real handwriting has natural wobble; fonts are perfectly straight.
        """
        img = self._render_text(font, self.test_text, 500, 100)

        # Find bottom edge of ink at each x position
        baseline = []
        for x in range(img.shape[1]):
            col = img[:, x]
            ink_pixels = np.where(col < 200)[0]
            if len(ink_pixels) > 0:
                baseline.append(ink_pixels[-1])  # Bottom of ink

        if len(baseline) < 10:
            return 0.0

        baseline = np.array(baseline)

        # Calculate variation (std dev of baseline position)
        # Normalize by font size
        variation = np.std(baseline) / self.font_size

        # Score: 0.05-0.15 std is good handwriting range
        # Too low = too uniform (font-like)
        # Too high = messy
        if variation < 0.02:
            return variation / 0.02 * 0.5  # Low score for too uniform
        elif variation < 0.1:
            return 0.5 + (variation - 0.02) / 0.08 * 0.5  # Good range
        else:
            return max(0.3, 1.0 - (variation - 0.1) * 2)  # Penalize excessive

    def _measure_stroke_variation(self, font) -> float:
        """
        Measure stroke width variation.
        Real handwriting has pressure variation; fonts have uniform strokes.
        """
        img = self._render_text(font, self.test_text, 500, 100)

        # Measure stroke width at multiple points
        widths = []
        for x in range(20, img.shape[1] - 20, 5):
            col = img[:, x]
            ink_pixels = np.where(col < 200)[0]
            if len(ink_pixels) >= 2:
                width = ink_pixels[-1] - ink_pixels[0]
                if width > 0:
                    widths.append(width)

        if len(widths) < 5:
            return 0.0

        widths = np.array(widths)

        # Coefficient of variation (std/mean)
        cv = np.std(widths) / (np.mean(widths) + 1e-6)

        # Score: 0.1-0.3 CV is good handwriting range
        if cv < 0.05:
            return cv / 0.05 * 0.3  # Too uniform
        elif cv < 0.25:
            return 0.3 + (cv - 0.05) / 0.2 * 0.7  # Good range
        else:
            return max(0.4, 1.0 - (cv - 0.25) * 2)  # Penalize excessive

    def _measure_character_consistency(self, font) -> float:
        """
        Measure how consistent same letters are.
        Real handwriting: same letter looks different each time.
        Fonts: same letter is identical.

        Returns 1.0 for perfect consistency (font-like), lower for variation.
        """
        # For fonts, every 'e' is identical, so we can only measure
        # variation between different chars as a proxy

        char_features = []
        for char in self.test_chars:
            img = Image.new('L', (self.font_size * 2, self.font_size * 2), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), char, font=font, fill=0)
            arr = np.array(img)

            # Extract simple features
            ink = arr < 200
            if ink.sum() > 0:
                # Center of mass
                y_coords, x_coords = np.where(ink)
                cx, cy = np.mean(x_coords), np.mean(y_coords)
                # Aspect ratio
                if len(x_coords) > 1 and len(y_coords) > 1:
                    w = x_coords.max() - x_coords.min()
                    h = y_coords.max() - y_coords.min()
                    aspect = w / (h + 1e-6)
                else:
                    aspect = 1.0
                char_features.append([cx, cy, aspect])

        if len(char_features) < 3:
            return 1.0

        # For real variation, we'd need multiple samples of same letter
        # Since fonts give identical output, return high consistency
        # This metric is more useful for comparing rendered handwriting
        return 0.95  # Most fonts are very consistent

    def _measure_edge_roughness(self, font) -> float:
        """
        Measure edge smoothness.
        Real handwriting has slightly rough edges; fonts are smooth.
        """
        img = self._render_text(font, "Hello", 300, 100)

        # Find edges using simple gradient
        dx = np.abs(np.diff(img.astype(float), axis=1))
        dy = np.abs(np.diff(img.astype(float), axis=0))

        # Find edge pixels
        edge_mask = (dx[:-1, :] > 50) | (dy[:, :-1] > 50)

        if edge_mask.sum() < 10:
            return 0.0

        # Measure local variation along edges
        edge_y, edge_x = np.where(edge_mask)

        # Calculate local direction changes (roughness indicator)
        if len(edge_x) < 20:
            return 0.3

        # Simple roughness: how much do neighboring edge pixels vary?
        sorted_idx = np.argsort(edge_x)
        edge_y_sorted = edge_y[sorted_idx]

        # Calculate second derivative (curvature changes)
        if len(edge_y_sorted) > 10:
            diff2 = np.diff(np.diff(edge_y_sorted))
            roughness = np.std(diff2) / self.font_size
        else:
            roughness = 0.1

        # Score: some roughness is good, too much is bad
        if roughness < 0.01:
            return roughness / 0.01 * 0.3  # Too smooth
        elif roughness < 0.1:
            return 0.3 + (roughness - 0.01) / 0.09 * 0.7  # Good range
        else:
            return max(0.4, 1.0 - (roughness - 0.1) * 3)

    def _measure_slant_variation(self, font) -> float:
        """
        Measure variation in letter slant.
        Real handwriting has inconsistent slant; fonts are uniform.
        """
        slants = []

        for char in 'lhkbdf':  # Tall letters good for slant measurement
            img = Image.new('L', (self.font_size * 2, self.font_size * 2), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), char, font=font, fill=0)
            arr = np.array(img)

            ink = arr < 200
            if ink.sum() < 10:
                continue

            # Find the "spine" of the letter
            y_coords, x_coords = np.where(ink)

            # Fit a line to get slant angle
            if len(x_coords) > 10:
                # Simple linear regression
                A = np.vstack([y_coords, np.ones(len(y_coords))]).T
                try:
                    m, c = np.linalg.lstsq(A, x_coords, rcond=None)[0]
                    slant_angle = np.arctan(m) * 180 / np.pi
                    slants.append(slant_angle)
                except:
                    pass

        if len(slants) < 3:
            return 0.3

        # Measure variation in slant
        slant_std = np.std(slants)

        # Score: 1-5 degree variation is good handwriting
        if slant_std < 0.5:
            return slant_std / 0.5 * 0.3  # Too uniform
        elif slant_std < 5:
            return 0.3 + (slant_std - 0.5) / 4.5 * 0.7  # Good range
        else:
            return max(0.4, 1.0 - (slant_std - 5) / 10)


def score_fonts_directory(fonts_dir: str, output_file: str = None) -> List[HandwritingScore]:
    """Score all fonts in a directory."""
    scorer = HandwritingScorer()
    scores = []

    fonts_path = Path(fonts_dir)
    font_files = list(fonts_path.glob('**/*.ttf')) + list(fonts_path.glob('**/*.otf'))

    print(f"Scoring {len(font_files)} fonts...")

    for font_path in font_files:
        score = scorer.score_font(str(font_path))
        if score:
            scores.append(score)
            print(f"  {score.name}: {score.overall:.3f}")

    # Sort by overall score (most human-like first)
    scores.sort(key=lambda x: x.overall, reverse=True)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump([s.to_dict() for s in scores], f, indent=2)
        print(f"\nScores saved to {output_file}")

    return scores


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python handwriting_scorer.py <fonts_dir> [output.json]")
        sys.exit(1)

    fonts_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    scores = score_fonts_directory(fonts_dir, output_file)

    print("\n=== Top 10 Most Human-Like Fonts ===")
    for s in scores[:10]:
        print(f"  {s.name}: {s.overall:.3f}")
        print(f"    baseline={s.baseline_variation:.2f}, stroke={s.stroke_width_variation:.2f}, "
              f"edge={s.edge_roughness:.2f}, slant={s.slant_variation:.2f}")
