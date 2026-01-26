"""
InkSight Font Vectorizer

Converts outline fonts to single-line vector strokes using Google's InkSight model.
InkSight is designed for handwriting recognition, which makes it ideal for extracting
natural-looking stroke paths from handwriting fonts.

Pipeline:
1. Render font character/word as image
2. Run InkSight inference to get stroke tokens
3. Post-process: filter artifacts, smooth, connect gaps

Requirements:
    conda activate inksight  # or micromamba
    # TensorFlow 2.15-2.17, tensorflow-text

Usage:
    python inksight_vectorizer.py --font path/to/font.ttf --output output_dir
    python inksight_vectorizer.py --font path/to/font.ttf --word "Hello" --show
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev


@dataclass
class Stroke:
    """A single stroke as a sequence of (x, y) points."""
    points: np.ndarray  # Shape: (N, 2)

    def __len__(self):
        return len(self.points)

    def copy(self):
        return Stroke(self.points.copy())


@dataclass
class InkResult:
    """Result of InkSight vectorization."""
    strokes: List[Stroke]
    image: Image.Image
    word: str
    font_path: str


class InkSightVectorizer:
    """
    Convert fonts to vector strokes using Google's InkSight model.

    InkSight is a Vision-Language model that converts images of handwriting
    to digital ink (vector strokes). We use it to extract stroke paths from
    rendered font images.
    """

    def __init__(self, model_path: str = "/home/server/inksight/model"):
        """
        Initialize the vectorizer.

        Args:
            model_path: Path to InkSight saved_model directory
        """
        self.model_path = model_path
        self.model = None
        self.infer = None

        # Token constants for InkSight
        self.coordinate_length = 224
        self.num_token_per_dim = self.coordinate_length + 1  # 225
        self.start_token = self.num_token_per_dim * 2  # 450

    def load_model(self):
        """Load InkSight TensorFlow model."""
        if self.model is not None:
            return

        import tensorflow as tf
        import tensorflow_text  # Required for model

        print(f"Loading InkSight model from {self.model_path}...")
        self.model = tf.saved_model.load(self.model_path)
        self.infer = self.model.signatures['serving_default']
        print("Model loaded.")

    def render_word(
        self,
        font_path: str,
        word: str,
        size: int = 120,
        canvas_size: int = 512,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        fg_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> Image.Image:
        """
        Render a word using the specified font.

        Args:
            font_path: Path to TTF/OTF font file
            word: Text to render
            size: Font size in pixels
            canvas_size: Output image size (square)
            bg_color: Background color (RGB)
            fg_color: Text color (RGB)

        Returns:
            PIL Image with rendered text
        """
        img = Image.new('RGB', (canvas_size, canvas_size), bg_color)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size)

        # Center the text
        bbox = draw.textbbox((0, 0), word, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (canvas_size - text_width) // 2 - bbox[0]
        y = (canvas_size - text_height) // 2 - bbox[1]

        draw.text((x, y), word, font=font, fill=fg_color)
        return img

    def _detokenize(self, tokens: List[int]) -> List[np.ndarray]:
        """
        Convert InkSight tokens to stroke coordinates.

        Token format:
        - 450 = start of new stroke
        - 0-224 = x coordinate
        - 225-449 = y coordinate (subtract 225)

        Args:
            tokens: List of token integers

        Returns:
            List of numpy arrays, each shape (N, 2) for stroke points
        """
        strokes = []
        current_stroke = []
        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token == self.start_token:
                # Start new stroke
                if current_stroke:
                    strokes.append(np.array(current_stroke, dtype=float))
                current_stroke = []
                i += 1
            elif i + 1 < len(tokens) and tokens[i + 1] != self.start_token:
                # Read x, y pair
                x = tokens[i]
                y = tokens[i + 1] - self.num_token_per_dim

                if 0 <= x < self.coordinate_length and 0 <= y < self.coordinate_length:
                    current_stroke.append([x, y])
                i += 2
            else:
                i += 1

        if current_stroke:
            strokes.append(np.array(current_stroke, dtype=float))

        return strokes

    def infer_strokes(self, image: Image.Image) -> List[np.ndarray]:
        """
        Run InkSight inference on an image.

        Args:
            image: PIL Image (will be resized/processed as needed)

        Returns:
            List of stroke arrays, each shape (N, 2)
        """
        import tensorflow as tf

        self.load_model()

        # Encode image
        image_np = np.array(image)[:, :, :3]
        image_encoded = tf.reshape(
            tf.io.encode_jpeg(image_np, quality=95),
            (1, 1)
        )

        # Run inference with "Recognize and derender" prompt
        input_text = tf.constant(["Recognize and derender."], dtype=tf.string)
        output = self.infer(**{
            'input_text': input_text,
            'image/encoded': image_encoded
        })

        # Parse output tokens
        output_text = output["output_0"].numpy()[0][0].decode()
        tokens = [int(t) for t in re.findall(r"<ink_token_(\d+)>", output_text)]

        # Detokenize to strokes
        strokes = self._detokenize(tokens)

        return strokes

    # ==================== POST-PROCESSING ====================

    @staticmethod
    def smart_filter(strokes: List[np.ndarray], min_points: int = 2) -> List[np.ndarray]:
        """
        Remove artifact strokes (single points at image edges).

        InkSight sometimes produces single-point artifacts at the very edge
        of the 224x224 coordinate space. This removes them while keeping
        legitimate short strokes (like dots on 'i').

        Args:
            strokes: List of stroke arrays
            min_points: Minimum points for automatic inclusion

        Returns:
            Filtered list of strokes
        """
        filtered = []
        for s in strokes:
            if len(s) >= min_points:
                filtered.append(s)
            elif len(s) == 1:
                # Single point - only keep if not at extreme edge
                x, y = s[0]
                if not (x <= 2 or x >= 222 or y <= 2 or y >= 222):
                    filtered.append(s)
        return filtered

    @staticmethod
    def smooth_gaussian(stroke: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        """
        Smooth stroke using Gaussian filter.

        Args:
            stroke: Stroke array shape (N, 2)
            sigma: Gaussian sigma (higher = smoother)

        Returns:
            Smoothed stroke array
        """
        if len(stroke) < 3:
            return stroke
        return np.column_stack([
            gaussian_filter1d(stroke[:, 0], sigma),
            gaussian_filter1d(stroke[:, 1], sigma)
        ])

    @staticmethod
    def smooth_spline(stroke: np.ndarray, smoothing: float = 1.0,
                      num_points: int = None) -> np.ndarray:
        """
        Smooth stroke using B-spline interpolation.

        Args:
            stroke: Stroke array shape (N, 2)
            smoothing: Spline smoothing factor
            num_points: Number of output points (None = 2x input)

        Returns:
            Smoothed stroke array
        """
        if len(stroke) < 4:
            return stroke

        try:
            # Remove duplicate consecutive points
            diffs = np.diff(stroke, axis=0)
            mask = np.concatenate([[True], np.any(diffs != 0, axis=1)])
            pts = stroke[mask]

            if len(pts) < 4:
                return stroke

            # Fit B-spline
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=smoothing, k=min(3, len(pts)-1))

            # Evaluate
            if num_points is None:
                num_points = len(stroke) * 2
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)

            return np.column_stack([x_new, y_new])
        except Exception:
            return stroke

    @staticmethod
    def _ray_segment_intersection(
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        Find intersection of ray with line segment.

        Args:
            ray_origin: Ray starting point
            ray_dir: Ray direction (normalized)
            seg_start: Segment start point
            seg_end: Segment end point

        Returns:
            (t, point) where t is distance along ray, or (None, None)
        """
        v1 = ray_origin - seg_start
        v2 = seg_end - seg_start
        v3 = np.array([-ray_dir[1], ray_dir[0]])  # Perpendicular

        denom = np.dot(v2, v3)
        if abs(denom) < 1e-10:
            return None, None  # Parallel

        t = np.cross(v2, v1) / denom  # Distance along ray
        s = np.dot(v1, v3) / denom    # Position along segment

        if t >= 0 and 0 <= s <= 1:
            intersection = ray_origin + t * ray_dir
            return t, intersection

        return None, None

    def extend_to_connect(
        self,
        strokes: List[np.ndarray],
        max_extension: float = 8.0
    ) -> List[np.ndarray]:
        """
        Extend stroke endpoints to connect nearby strokes.

        For each stroke endpoint, cast a ray along the stroke's trajectory.
        If it intersects another stroke within max_extension pixels,
        extend to that intersection point.

        Args:
            strokes: List of stroke arrays
            max_extension: Maximum extension distance in pixels
                          (8 works well for within-character gaps)

        Returns:
            List of strokes with extensions added
        """
        result = [s.copy() for s in strokes]

        for i, stroke in enumerate(result):
            if len(stroke) < 3:
                continue

            # Check both endpoints
            for is_start in [True, False]:
                if is_start:
                    endpoint = stroke[0].copy()
                    # Direction from interior toward start
                    direction = stroke[0] - stroke[min(3, len(stroke)-1)]
                else:
                    endpoint = stroke[-1].copy()
                    # Direction from interior toward end
                    direction = stroke[-1] - stroke[max(-4, -len(stroke))]

                # Normalize direction
                norm = np.linalg.norm(direction)
                if norm < 0.1:
                    continue
                direction = direction / norm

                # Find nearest intersection with other strokes
                best_t = float('inf')
                best_point = None

                for j, other in enumerate(strokes):  # Use original strokes
                    if i == j or len(other) < 2:
                        continue

                    # Check each segment of other stroke
                    for k in range(len(other) - 1):
                        t, point = self._ray_segment_intersection(
                            endpoint, direction, other[k], other[k+1]
                        )

                        if t is not None and 0 < t < max_extension and t < best_t:
                            best_t = t
                            best_point = point

                # Add extension points if intersection found
                if best_point is not None:
                    n_pts = max(2, int(best_t / 2))
                    ext_pts = np.array([
                        endpoint + direction * t
                        for t in np.linspace(0, best_t, n_pts + 1)[1:]
                    ])

                    if is_start:
                        result[i] = np.vstack([ext_pts[::-1], result[i]])
                    else:
                        result[i] = np.vstack([result[i], ext_pts])

        return result

    def process(
        self,
        font_path: str,
        word: str,
        font_size: int = 120,
        smooth_sigma: float = 1.5,
        max_extension: float = 8.0
    ) -> InkResult:
        """
        Full pipeline: render, infer, post-process.

        Args:
            font_path: Path to font file
            word: Text to vectorize
            font_size: Font rendering size
            smooth_sigma: Gaussian smoothing sigma
            max_extension: Max stroke extension distance

        Returns:
            InkResult with strokes and metadata
        """
        # Render
        image = self.render_word(font_path, word, size=font_size)

        # Infer
        strokes = self.infer_strokes(image)

        # Post-process
        strokes = self.smart_filter(strokes)
        strokes = [self.smooth_gaussian(s, sigma=smooth_sigma) for s in strokes]
        strokes = self.extend_to_connect(strokes, max_extension=max_extension)

        return InkResult(
            strokes=[Stroke(s) for s in strokes],
            image=image,
            word=word,
            font_path=font_path
        )

    def process_charset(
        self,
        font_path: str,
        chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        **kwargs
    ) -> dict:
        """
        Process all characters in a charset.

        Args:
            font_path: Path to font file
            chars: Characters to process
            **kwargs: Passed to process()

        Returns:
            Dict mapping char -> InkResult
        """
        results = {}
        for char in chars:
            try:
                results[char] = self.process(font_path, char, **kwargs)
            except Exception as e:
                print(f"  Failed on '{char}': {e}")
        return results

    # ==================== EXPORT ====================

    @staticmethod
    def to_svg(result: InkResult, stroke_width: float = 2.0) -> str:
        """
        Export strokes to SVG.

        Args:
            result: InkResult from process()
            stroke_width: SVG stroke width

        Returns:
            SVG string
        """
        paths = []
        for stroke in result.strokes:
            if len(stroke.points) < 2:
                continue
            pts = stroke.points
            d = f"M {pts[0, 0]:.1f} {pts[0, 1]:.1f}"
            for p in pts[1:]:
                d += f" L {p[0]:.1f} {p[1]:.1f}"
            paths.append(
                f'  <path d="{d}" fill="none" stroke="black" '
                f'stroke-width="{stroke_width}" stroke-linecap="round" '
                f'stroke-linejoin="round"/>'
            )

        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 224 224">
{chr(10).join(paths)}
</svg>'''

    @staticmethod
    def to_json(result: InkResult) -> dict:
        """
        Export strokes to JSON-serializable dict.

        Args:
            result: InkResult from process()

        Returns:
            Dict with stroke data
        """
        return {
            'word': result.word,
            'font': result.font_path,
            'strokes': [
                stroke.points.tolist()
                for stroke in result.strokes
            ]
        }


def visualize(result: InkResult, output_path: str = None, show: bool = False):
    """
    Visualize InkSight result.

    Args:
        result: InkResult from process()
        output_path: Path to save image (optional)
        show: Whether to display interactively
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    axes[0].imshow(result.image)
    axes[0].set_title(f'Input: "{result.word}"')
    axes[0].axis('off')

    # Strokes
    axes[1].set_xlim(0, 224)
    axes[1].set_ylim(224, 0)  # Flip Y
    axes[1].set_aspect('equal')

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(result.strokes), 1)))
    for i, stroke in enumerate(result.strokes):
        if len(stroke.points) > 1:
            pts = stroke.points
            axes[1].plot(pts[:, 0], pts[:, 1], '-',
                        color=colors[i % 10], linewidth=2.5)
            # Mark endpoints
            axes[1].plot(pts[0, 0], pts[0, 1], 'o',
                        color=colors[i % 10], markersize=4)
            axes[1].plot(pts[-1, 0], pts[-1, 1], 's',
                        color=colors[i % 10], markersize=4)

    axes[1].set_title(f'InkSight: {len(result.strokes)} strokes')
    axes[1].set_facecolor('white')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, facecolor='white', bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Convert fonts to vector strokes using InkSight'
    )
    parser.add_argument('--font', '-f', type=str, required=True,
                        help='Path to TTF/OTF font file')
    parser.add_argument('--word', '-w', type=str, default='Hello',
                        help='Word to vectorize (default: Hello)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for SVG/JSON')
    parser.add_argument('--model', '-m', type=str,
                        default='/home/server/inksight/model',
                        help='Path to InkSight model')
    parser.add_argument('--size', '-s', type=int, default=120,
                        help='Font size (default: 120)')
    parser.add_argument('--smooth', type=float, default=1.5,
                        help='Smoothing sigma (default: 1.5)')
    parser.add_argument('--extend', type=float, default=8.0,
                        help='Max extension distance (default: 8.0)')
    parser.add_argument('--show', action='store_true',
                        help='Show visualization')
    parser.add_argument('--charset', action='store_true',
                        help='Process full charset instead of word')

    args = parser.parse_args()

    vectorizer = InkSightVectorizer(model_path=args.model)

    if args.charset:
        # Process all characters
        print(f"Processing charset from {args.font}...")
        results = vectorizer.process_charset(
            args.font,
            font_size=args.size,
            smooth_sigma=args.smooth,
            max_extension=args.extend
        )
        print(f"Processed {len(results)} characters")

        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save individual SVGs
            for char, result in results.items():
                safe_name = char if char.isalnum() else f"char_{ord(char):04x}"
                svg_path = output_dir / f"{safe_name}.svg"
                svg_path.write_text(vectorizer.to_svg(result))

            # Save combined JSON
            all_data = {char: vectorizer.to_json(r) for char, r in results.items()}
            json_path = output_dir / "strokes.json"
            json_path.write_text(json.dumps(all_data, indent=2))

            print(f"Saved to {output_dir}/")
    else:
        # Process single word
        print(f"Processing '{args.word}' from {args.font}...")
        result = vectorizer.process(
            args.font,
            args.word,
            font_size=args.size,
            smooth_sigma=args.smooth,
            max_extension=args.extend
        )
        print(f"Got {len(result.strokes)} strokes")

        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save SVG
            svg_path = output_dir / f"{args.word}.svg"
            svg_path.write_text(vectorizer.to_svg(result))

            # Save JSON
            json_path = output_dir / f"{args.word}.json"
            json_path.write_text(json.dumps(vectorizer.to_json(result), indent=2))

            # Save visualization
            vis_path = output_dir / f"{args.word}.png"
            visualize(result, output_path=str(vis_path))

            print(f"Saved to {output_dir}/")

        if args.show:
            visualize(result, show=True)


if __name__ == '__main__':
    main()
