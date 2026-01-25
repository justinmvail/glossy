"""
Full Pipeline: Single-Line Fonts → Synthetic Writers → SDT Training Data

This script orchestrates the entire process:
1. Load single-line fonts from SVG files
2. Generate diverse synthetic writer styles
3. Apply transforms to create training samples
4. Export in SDT-compatible format

Output format matches SDT's expected structure:
- character_dict.pkl: {char: index}
- writer_dict.pkl: {writer_id: index}
- train_data.pkl: list of samples with strokes and metadata
"""

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from svg_parser import SVGFontParser, Glyph, Stroke, Point
from generate import WriterStyle, WriterGenerator, StyleTransformer


@dataclass
class SDTSample:
    """A single training sample in SDT format."""
    char: str
    char_idx: int
    writer_id: str
    writer_idx: int
    strokes: np.ndarray  # Shape: (N, 3) where columns are [x, y, pen_state]
    
    def to_dict(self) -> dict:
        return {
            'char': self.char,
            'char_idx': self.char_idx,
            'writer_id': self.writer_id,
            'writer_idx': self.writer_idx,
            'strokes': self.strokes.tolist()
        }


class SDTDataGenerator:
    """Generate SDT-format training data from fonts and synthetic writers."""
    
    # Characters to generate (English alphabet + digits + common punctuation)
    DEFAULT_CHARS = (
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789'
        '.,!?\'"-:;()&'
    )
    
    def __init__(
        self,
        fonts_dir: str,
        output_dir: str,
        num_writers: int = 1000,
        chars: str = None,
        samples_per_char: int = 1,
        seed: int = 42
    ):
        self.fonts_dir = Path(fonts_dir)
        self.output_dir = Path(output_dir)
        self.num_writers = num_writers
        self.chars = chars or self.DEFAULT_CHARS
        self.samples_per_char = samples_per_char
        self.seed = seed
        
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        
        # Initialize
        self.fonts: Dict[str, Dict[str, Glyph]] = {}
        self.char_to_idx: Dict[str, int] = {}
        self.writer_to_idx: Dict[str, int] = {}
        self.styles: List[WriterStyle] = []
        
    def load_fonts(self) -> int:
        """Load all SVG fonts from the fonts directory."""
        parser = SVGFontParser()
        self.fonts = parser.parse_directory(str(self.fonts_dir))
        
        # Filter to fonts that have enough characters
        min_chars = len(self.chars) * 0.5  # At least 50% coverage
        self.fonts = {
            name: glyphs for name, glyphs in self.fonts.items()
            if sum(1 for c in self.chars if c in glyphs) >= min_chars
        }
        
        print(f"Loaded {len(self.fonts)} fonts with sufficient character coverage")
        return len(self.fonts)
    
    def generate_writers(self):
        """Generate synthetic writer styles."""
        generator = WriterGenerator(seed=self.seed)
        font_names = list(self.fonts.keys())
        
        if not font_names:
            raise ValueError("No fonts loaded! Load fonts first.")
        
        self.styles = []
        for i in range(self.num_writers):
            # Assign each writer a base font (cycle through available fonts)
            base_font = font_names[i % len(font_names)]
            
            style = generator.generate_style(
                writer_id=f"writer_{i:05d}",
                base_font=base_font
            )
            self.styles.append(style)
        
        # Build writer index
        self.writer_to_idx = {
            style.writer_id: idx for idx, style in enumerate(self.styles)
        }
        
        print(f"Generated {len(self.styles)} synthetic writer styles")
    
    def build_char_index(self):
        """Build character to index mapping."""
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        print(f"Character vocabulary: {len(self.char_to_idx)} characters")
    
    def generate_samples(self) -> List[SDTSample]:
        """Generate all training samples."""
        samples = []
        
        total = len(self.styles) * len(self.chars) * self.samples_per_char
        print(f"Generating {total} samples...")
        
        for style_idx, style in enumerate(self.styles):
            if style_idx % 100 == 0:
                print(f"  Processing writer {style_idx}/{len(self.styles)}...")
            
            # Get the font for this writer
            font_glyphs = self.fonts.get(style.base_font, {})
            
            for char in self.chars:
                if char not in font_glyphs:
                    continue
                
                glyph = font_glyphs[char]
                
                for sample_idx in range(self.samples_per_char):
                    # Create transformer with unique seed for reproducibility
                    seed = hash((style.writer_id, char, sample_idx)) % (2**31)
                    transformer = StyleTransformer(style, seed=seed)
                    
                    # Transform the glyph
                    transformed = transformer.transform_glyph(glyph)
                    
                    # Normalize to standard size
                    normalized = transformed.normalize(target_height=100.0)
                    
                    # Convert to SDT format
                    strokes_data = self._glyph_to_sdt_strokes(normalized)
                    
                    sample = SDTSample(
                        char=char,
                        char_idx=self.char_to_idx[char],
                        writer_id=style.writer_id,
                        writer_idx=style_idx,
                        strokes=strokes_data
                    )
                    samples.append(sample)
        
        print(f"Generated {len(samples)} total samples")
        return samples
    
    def _glyph_to_sdt_strokes(self, glyph: Glyph) -> np.ndarray:
        """Convert glyph to SDT stroke format."""
        all_points = []
        
        for stroke in glyph.strokes:
            for i, point in enumerate(stroke.points):
                # pen_state: 0 = drawing, 1 = stroke end (pen lifts after this)
                is_last = (i == len(stroke.points) - 1)
                pen_state = 1 if is_last else 0
                all_points.append([point.x, point.y, pen_state])
        
        if not all_points:
            return np.array([[0, 0, 1]])  # Empty glyph fallback
        
        return np.array(all_points, dtype=np.float32)
    
    def save_data(self, samples: List[SDTSample]):
        """Save data in SDT-compatible format."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Character dictionary
        char_dict_path = self.output_dir / 'character_dict.pkl'
        with open(char_dict_path, 'wb') as f:
            pickle.dump(self.char_to_idx, f)
        print(f"Saved character_dict.pkl ({len(self.char_to_idx)} chars)")
        
        # 2. Writer dictionary
        writer_dict_path = self.output_dir / 'writer_dict.pkl'
        with open(writer_dict_path, 'wb') as f:
            pickle.dump(self.writer_to_idx, f)
        print(f"Saved writer_dict.pkl ({len(self.writer_to_idx)} writers)")
        
        # 3. Training data
        train_data = [s.to_dict() for s in samples]
        train_data_path = self.output_dir / 'train_data.pkl'
        with open(train_data_path, 'wb') as f:
            pickle.dump(train_data, f)
        print(f"Saved train_data.pkl ({len(train_data)} samples)")
        
        # 4. Writer styles (for reference/debugging)
        styles_path = self.output_dir / 'writer_styles.json'
        with open(styles_path, 'w') as f:
            json.dump([s.to_dict() for s in self.styles], f, indent=2)
        print(f"Saved writer_styles.json")
        
        # 5. Statistics
        stats = self._compute_stats(samples)
        stats_path = self.output_dir / 'stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats.json")
    
    def _compute_stats(self, samples: List[SDTSample]) -> dict:
        """Compute dataset statistics."""
        chars_per_writer = defaultdict(set)
        samples_per_char = defaultdict(int)
        samples_per_writer = defaultdict(int)
        stroke_lengths = []
        
        for s in samples:
            chars_per_writer[s.writer_id].add(s.char)
            samples_per_char[s.char] += 1
            samples_per_writer[s.writer_id] += 1
            stroke_lengths.append(len(s.strokes))
        
        return {
            'total_samples': len(samples),
            'num_writers': len(self.writer_to_idx),
            'num_characters': len(self.char_to_idx),
            'avg_chars_per_writer': np.mean([len(v) for v in chars_per_writer.values()]),
            'avg_samples_per_char': np.mean(list(samples_per_char.values())),
            'avg_samples_per_writer': np.mean(list(samples_per_writer.values())),
            'avg_stroke_length': np.mean(stroke_lengths),
            'min_stroke_length': min(stroke_lengths),
            'max_stroke_length': max(stroke_lengths),
        }
    
    def run(self):
        """Run the full pipeline."""
        print("=" * 60)
        print("SDT Synthetic Training Data Generator")
        print("=" * 60)
        
        print("\n1. Loading fonts...")
        num_fonts = self.load_fonts()
        if num_fonts == 0:
            print("ERROR: No fonts found!")
            return
        
        print("\n2. Generating synthetic writers...")
        self.generate_writers()
        
        print("\n3. Building character index...")
        self.build_char_index()
        
        print("\n4. Generating training samples...")
        samples = self.generate_samples()
        
        print("\n5. Saving data...")
        self.save_data(samples)
        
        print("\n" + "=" * 60)
        print("DONE!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)


def create_sample_fonts(output_dir: str):
    """Create some sample SVG font files for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Simple block letter 'A'
    svg_a = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <path id="A" d="M 10 90 L 50 10 L 90 90 M 25 60 L 75 60"/>
</svg>'''
    
    # Simple 'B'  
    svg_b = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <path id="B" d="M 20 10 L 20 90 M 20 10 L 60 10 Q 80 10 80 30 Q 80 50 60 50 L 20 50 M 20 50 L 65 50 Q 85 50 85 70 Q 85 90 65 90 L 20 90"/>
</svg>'''

    # Simple 'a' lowercase
    svg_a_lower = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <path id="a" d="M 70 40 Q 70 20 50 20 Q 30 20 30 40 Q 30 60 50 60 Q 70 60 70 40 L 70 60 L 70 90"/>
</svg>'''
    
    (output_path / 'sample_A.svg').write_text(svg_a)
    (output_path / 'sample_B.svg').write_text(svg_b)
    (output_path / 'sample_a.svg').write_text(svg_a_lower)
    
    print(f"Created sample fonts in {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate SDT training data from single-line fonts'
    )
    parser.add_argument(
        '--fonts_dir', type=str, required=True,
        help='Directory containing SVG single-line fonts'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for SDT training data'
    )
    parser.add_argument(
        '--num_writers', type=int, default=1000,
        help='Number of synthetic writers to generate'
    )
    parser.add_argument(
        '--samples_per_char', type=int, default=1,
        help='Number of samples per character per writer'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--create_samples', action='store_true',
        help='Create sample SVG fonts for testing'
    )
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_sample_fonts(args.fonts_dir)
    
    generator = SDTDataGenerator(
        fonts_dir=args.fonts_dir,
        output_dir=args.output_dir,
        num_writers=args.num_writers,
        samples_per_char=args.samples_per_char,
        seed=args.seed
    )
    
    generator.run()
