"""InkSight Font Vectorizer.

This module converts outline fonts to single-line vector strokes using Google's
InkSight model. InkSight is a Vision-Language model designed for handwriting
recognition, which makes it ideal for extracting natural-looking stroke paths
from handwriting fonts.

The vectorization pipeline:
    1. Render font character/word as a rasterized image
    2. Run InkSight inference to get stroke tokens
    3. Detokenize to extract (x, y) coordinate sequences
    4. Post-process: filter artifacts, smooth curves, connect gaps
    5. Optionally validate with OCR (TrOCR) to ensure readability

InkSight produces stroke orderings that follow natural handwriting patterns,
making the output suitable for applications like pen plotters, laser engravers,
or animation systems that need to draw characters stroke-by-stroke.

Model Information:
    InkSight uses a Vision Transformer encoder with a text decoder that outputs
    special tokens representing ink coordinates. The coordinate space is 224x224
    pixels with tokens encoding x (0-224), y (225-449), and stroke start (450).

Requirements:
    Core dependencies::

        tensorflow>=2.15,<2.18
        tensorflow-text  # Required by InkSight model
        numpy
        Pillow
        scipy

    For OCR validation::

        transformers  # For TrOCR
        torch

    Recommended environment::

        conda activate inksight  # or micromamba

Example:
    Basic vectorization::

        from inksight_vectorizer import InkSightVectorizer, visualize

        vectorizer = InkSightVectorizer(model_path='/path/to/inksight/model')
        result = vectorizer.process('/fonts/MyScript.ttf', 'Hello')

        print(f"Extracted {len(result.strokes)} strokes")
        visualize(result, output_path='output.png')

    With OCR validation::

        from inksight_vectorizer import InkSightVectorizer, OCRValidator

        vectorizer = InkSightVectorizer()
        result = vectorizer.process('/fonts/MyScript.ttf', 'Hello')

        validator = OCRValidator()
        passed, recognized, score = validator.validate(result)
        print(f"OCR: '{recognized}' (similarity: {score:.1%})")

Command-line Usage:
    python inksight_vectorizer.py --font path/to/font.ttf --word "Hello" --show
    python inksight_vectorizer.py --font path/to/font.ttf --charset --output output_dir
    python inksight_vectorizer.py --font path/to/font.ttf --word "Hello" --validate

Attributes:
    Stroke: Dataclass representing a single stroke as (x, y) points.
    InkResult: Dataclass containing vectorization results and metadata.
    InkSightVectorizer: Main class for running InkSight inference.
    OCRValidator: Optional validator using TrOCR for quality checking.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev


@dataclass
class Stroke:
    """A single stroke represented as a sequence of (x, y) points.

    Strokes are the fundamental unit of vector output from InkSight. Each stroke
    represents a continuous pen movement without lifting. The points are ordered
    from start to end of the stroke.

    Attributes:
        points: NumPy array of shape (N, 2) containing [x, y] coordinates.
            Coordinates are in the 224x224 InkSight coordinate space.

    Example:
        >>> stroke = Stroke(np.array([[10, 20], [15, 25], [20, 30]]))
        >>> print(f"Stroke has {len(stroke)} points")
        Stroke has 3 points
        >>> copy = stroke.copy()  # Create independent copy
    """
    points: np.ndarray  # Shape: (N, 2)

    def __len__(self):
        """Return the number of points in the stroke.

        Returns:
            Integer count of coordinate points.
        """
        return len(self.points)

    def copy(self):
        """Create a deep copy of the stroke.

        Returns:
            A new Stroke instance with copied point data.
        """
        return Stroke(self.points.copy())


@dataclass
class InkResult:
    """Complete result of InkSight vectorization.

    Contains all strokes extracted from a font rendering, along with metadata
    about the input. This is the primary output type from InkSightVectorizer.

    Attributes:
        strokes: List of Stroke objects representing the extracted paths.
        image: The original PIL Image that was processed.
        word: The text string that was rendered and vectorized.
        font_path: Path to the font file used for rendering.

    Example:
        >>> result = vectorizer.process('/fonts/Script.ttf', 'Hello')
        >>> print(f"'{result.word}' has {len(result.strokes)} strokes")
        >>> for i, stroke in enumerate(result.strokes):
        ...     print(f"  Stroke {i}: {len(stroke)} points")
    """
    strokes: List[Stroke]
    image: Image.Image
    word: str
    font_path: str


class InkSightVectorizer:
    """Convert fonts to vector strokes using Google's InkSight model.

    InkSight is a Vision-Language model that converts images of handwriting
    to digital ink (vector strokes). This class wraps the model for use with
    font rendering, providing a complete pipeline from font file to strokes.

    The model operates in a 224x224 coordinate space and outputs special tokens
    that encode stroke coordinates. This class handles rendering, inference,
    detokenization, and post-processing.

    Attributes:
        model_path (str): Path to the InkSight saved_model directory.
        model: Loaded TensorFlow SavedModel (None until load_model() called).
        infer: Model inference signature function.
        coordinate_length (int): Size of coordinate space (224).
        num_token_per_dim (int): Number of tokens per dimension (225).
        start_token (int): Token ID indicating stroke start (450).

    Example:
        >>> vectorizer = InkSightVectorizer('/path/to/model')
        >>> vectorizer.load_model()  # Explicit loading (optional)
        >>> result = vectorizer.process('/fonts/Script.ttf', 'Hello')
        >>> svg = vectorizer.to_svg(result)
    """

    def __init__(self, model_path: str = "/home/server/inksight/model"):
        """Initialize the vectorizer with model path.

        The model is not loaded immediately; it will be loaded lazily on first
        inference or can be loaded explicitly with load_model().

        Args:
            model_path: Path to the InkSight TensorFlow saved_model directory.
                This directory should contain 'saved_model.pb' and the
                'variables' subdirectory.
        """
        self.model_path = model_path
        self.model = None
        self.infer = None

        # Token constants for InkSight
        self.coordinate_length = 224
        self.num_token_per_dim = self.coordinate_length + 1  # 225
        self.start_token = self.num_token_per_dim * 2  # 450

    def load_model(self):
        """Load the InkSight TensorFlow model.

        Loads the saved_model from disk and extracts the inference signature.
        This method is idempotent; subsequent calls have no effect.

        The model requires TensorFlow and tensorflow-text to be installed.
        Loading prints progress messages to stdout.

        Raises:
            ImportError: If TensorFlow or tensorflow-text is not installed.
            OSError: If the model files cannot be found at model_path.
        """
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
        """Render a word using the specified font.

        Creates a square image with the text centered. The font size is
        specified directly; use a larger canvas_size for higher resolution
        output.

        Args:
            font_path: Path to the TrueType (.ttf) or OpenType (.otf) font file.
            word: Text string to render. Can be a single character or word.
            size: Font size in pixels. Default 120.
            canvas_size: Output image dimensions (square). Default 512.
            bg_color: Background color as RGB tuple. Default white (255,255,255).
            fg_color: Text color as RGB tuple. Default black (0,0,0).

        Returns:
            PIL Image in RGB mode with the rendered text centered.

        Raises:
            OSError: If the font file cannot be loaded.

        Example:
            >>> image = vectorizer.render_word('/fonts/Arial.ttf', 'Hello', size=100)
            >>> image.save('rendered.png')
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
        """Convert InkSight tokens to stroke coordinates.

        InkSight outputs special tokens that encode coordinates:
        - Token 450: Start of a new stroke
        - Tokens 0-224: X coordinate value
        - Tokens 225-449: Y coordinate value (subtract 225)

        Tokens are consumed in pairs (x, y) between stroke start tokens.

        Args:
            tokens: List of integer token IDs extracted from model output.

        Returns:
            List of numpy arrays, each with shape (N, 2) containing the
            [x, y] coordinates for one stroke.

        Example:
            >>> tokens = [450, 100, 325, 110, 335]  # One stroke, two points
            >>> strokes = vectorizer._detokenize(tokens)
            >>> print(strokes[0])  # [[100, 100], [110, 110]]
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
        """Run InkSight inference on an image.

        Encodes the image, runs the model with the "Recognize and derender"
        prompt, and parses the output tokens into stroke coordinates.

        Args:
            image: PIL Image to process. Should be RGB mode. The image is
                JPEG-encoded before being sent to the model.

        Returns:
            List of stroke arrays, each with shape (N, 2) containing raw
            coordinates in the 224x224 space. These are unprocessed strokes
            that may contain artifacts.

        Note:
            This method calls load_model() if the model is not yet loaded.
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
        """Remove artifact strokes while preserving valid short strokes.

        InkSight sometimes produces single-point artifacts at the edges of the
        224x224 coordinate space (e.g., stray points at [0,0] or [223,223]).
        This filter removes those while keeping legitimate short strokes like
        dots on 'i' or 'j'.

        Args:
            strokes: List of stroke arrays from infer_strokes().
            min_points: Minimum points for automatic inclusion. Strokes with
                at least this many points are always kept. Default 2.

        Returns:
            Filtered list of strokes with edge artifacts removed.

        Example:
            >>> raw_strokes = vectorizer.infer_strokes(image)
            >>> filtered = InkSightVectorizer.smart_filter(raw_strokes)
            >>> print(f"Kept {len(filtered)}/{len(raw_strokes)} strokes")
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
        """Smooth stroke coordinates using Gaussian filter.

        Applies 1D Gaussian smoothing independently to x and y coordinates.
        This reduces noise and jagged edges while preserving overall shape.

        Args:
            stroke: Stroke array of shape (N, 2) with [x, y] coordinates.
            sigma: Standard deviation for Gaussian kernel. Higher values
                produce smoother curves but may lose detail. Default 1.5.

        Returns:
            Smoothed stroke array with same shape as input.
            Returns input unchanged if fewer than 3 points.

        Example:
            >>> smoothed = InkSightVectorizer.smooth_gaussian(stroke, sigma=2.0)
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
        """Smooth stroke using B-spline interpolation.

        Fits a B-spline curve through the stroke points, optionally resampling
        to a different number of points. This produces very smooth curves and
        can increase or decrease point density.

        Args:
            stroke: Stroke array of shape (N, 2) with [x, y] coordinates.
            smoothing: Spline smoothing factor. Higher values allow more
                deviation from original points. Default 1.0.
            num_points: Number of points in output stroke. If None, uses
                2x the input point count for increased smoothness.

        Returns:
            Smoothed stroke array. Returns input unchanged if fewer than
            4 points (minimum for cubic B-spline).

        Example:
            >>> # Double the point density with smoothing
            >>> upsampled = InkSightVectorizer.smooth_spline(stroke, num_points=200)
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
        """Find intersection of a ray with a line segment.

        Uses parametric ray-segment intersection to find where a ray
        starting at ray_origin in direction ray_dir crosses the line
        segment from seg_start to seg_end.

        Args:
            ray_origin: 2D point where the ray starts.
            ray_dir: 2D direction vector for the ray (should be normalized).
            seg_start: 2D point at start of line segment.
            seg_end: 2D point at end of line segment.

        Returns:
            A tuple of (t, point) where:
            - t: Distance along ray to intersection (None if no intersection)
            - point: 2D intersection coordinates (None if no intersection)

            Returns (None, None) if ray is parallel to segment or doesn't
            intersect within the segment bounds.
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
        """Extend stroke endpoints to connect nearby strokes.

        For each stroke endpoint, casts a ray along the stroke's trajectory.
        If the ray intersects another stroke within max_extension pixels,
        adds interpolated points to extend the stroke to that intersection.

        This helps connect strokes that should be joined but have small gaps
        due to model imprecision or font design. Works well for connecting
        within characters (e.g., joining the loop of a 'p').

        Args:
            strokes: List of stroke arrays to potentially extend.
            max_extension: Maximum extension distance in pixels. Strokes
                are only extended if the intersection is within this
                distance. Default 8.0 (suitable for within-character gaps).

        Returns:
            List of strokes with extensions added. Strokes that don't need
            extension are returned unchanged (as copies).

        Example:
            >>> connected = vectorizer.extend_to_connect(strokes, max_extension=10.0)
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
        """Run the full vectorization pipeline.

        Complete workflow from font file to processed strokes:
        1. Render the word as an image
        2. Run InkSight inference
        3. Filter artifact strokes
        4. Apply Gaussian smoothing
        5. Extend endpoints to connect gaps

        Args:
            font_path: Path to the font file (.ttf, .otf).
            word: Text to vectorize. Can be a single character or word.
            font_size: Font size for rendering. Default 120.
            smooth_sigma: Gaussian smoothing sigma. Default 1.5.
            max_extension: Maximum stroke extension distance. Default 8.0.

        Returns:
            InkResult containing the processed strokes and metadata.

        Example:
            >>> result = vectorizer.process('/fonts/Script.ttf', 'Hello')
            >>> print(f"Extracted {len(result.strokes)} strokes")
            >>> svg = InkSightVectorizer.to_svg(result)
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
        """Process all characters in a charset.

        Batch-processes multiple characters from the same font. Failures on
        individual characters are logged but don't stop processing.

        Args:
            font_path: Path to the font file.
            chars: String of characters to process. Default includes
                uppercase, lowercase, and digits.
            **kwargs: Additional arguments passed to process() for each
                character (e.g., font_size, smooth_sigma).

        Returns:
            Dictionary mapping character string to InkResult. Characters
            that failed processing are not included.

        Example:
            >>> results = vectorizer.process_charset('/fonts/Script.ttf', 'ABC')
            >>> for char, result in results.items():
            ...     print(f"'{char}': {len(result.strokes)} strokes")
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
        """Export strokes to SVG format.

        Creates an SVG document with each stroke as a path element. Strokes
        are rendered as black polylines with rounded line caps and joins.

        Args:
            result: InkResult from process() or process_charset().
            stroke_width: Width of strokes in SVG units. Default 2.0.

        Returns:
            Complete SVG document as a string.

        Example:
            >>> svg = InkSightVectorizer.to_svg(result, stroke_width=1.5)
            >>> Path('output.svg').write_text(svg)
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
        """Export strokes to JSON-serializable dictionary.

        Converts the InkResult to a plain dictionary that can be serialized
        with json.dumps(). Stroke coordinates are converted from numpy
        arrays to nested Python lists.

        Args:
            result: InkResult from process() or process_charset().

        Returns:
            Dictionary with keys:
            - 'word': The input word
            - 'font': Path to font file
            - 'strokes': List of strokes, each as [[x, y], ...]

        Example:
            >>> data = InkSightVectorizer.to_json(result)
            >>> json.dumps(data, indent=2)
        """
        return {
            'word': result.word,
            'font': result.font_path,
            'strokes': [
                stroke.points.tolist()
                for stroke in result.strokes
            ]
        }


class OCRValidator:
    """Validate vectorization results using TrOCR handwriting recognition.

    Uses Microsoft's TrOCR model to read rendered strokes and compare against
    the expected text. This provides a quality metric for vectorization
    results: if OCR can read the strokes, they are likely to be human-readable.

    The validator supports two execution modes:
    1. Subprocess mode (default): Runs OCR in a separate process to avoid
       GPU conflicts between TensorFlow (InkSight) and PyTorch (TrOCR).
    2. Direct mode: Runs OCR in the same process (faster but may have issues).

    Subprocess mode maintains a persistent worker process that loads the
    model once and processes multiple images efficiently.

    Attributes:
        model_name (str): HuggingFace model identifier for TrOCR.
        use_subprocess (bool): Whether to use subprocess isolation.
        processor: TrOCR processor (None until loaded, direct mode only).
        model: TrOCR model (None until loaded, direct mode only).
        device: PyTorch device for inference (direct mode only).

    Example:
        >>> validator = OCRValidator()
        >>> result = vectorizer.process('/fonts/Script.ttf', 'Hello')
        >>> passed, text, score = validator.validate(result)
        >>> print(f"OCR read: '{text}' (similarity: {score:.1%})")
    """

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten", use_subprocess: bool = True):
        """Initialize the OCR validator.

        Args:
            model_name: HuggingFace model identifier for TrOCR. Options:
                - "microsoft/trocr-base-handwritten" (default, good balance)
                - "microsoft/trocr-large-handwritten" (more accurate, slower)
                - "microsoft/trocr-small-handwritten" (faster, less accurate)
            use_subprocess: If True (default), runs OCR in a separate Python
                process to avoid TensorFlow/PyTorch GPU conflicts. The worker
                process is started lazily and reused across calls.
        """
        self.model_name = model_name
        self.use_subprocess = use_subprocess
        self.processor = None
        self.model = None
        self.device = None

    def load_model(self):
        """Load TrOCR model and processor.

        In direct mode, loads the model into memory. In subprocess mode,
        this is a no-op since the model is loaded in the worker process.

        Only needs to be called explicitly if you want to preload the model;
        it will be called automatically on first use.

        Raises:
            ImportError: If transformers or torch is not installed.
        """
        if self.use_subprocess:
            return  # Model loaded in subprocess

        if self.model is not None:
            return

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        print(f"Loading TrOCR model: {self.model_name}...")
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self.model = self.model.to('cpu')
        self.model.eval()
        self.device = 'cpu'
        print("TrOCR loaded (CPU).")

    def render_strokes(
        self,
        strokes: List[Stroke],
        size: int = 384,
        stroke_width: int = 3,
        padding: int = 20
    ) -> Image.Image:
        """Render strokes to an image for OCR.

        Creates a centered rendering of the strokes on a white background.
        The strokes are automatically scaled to fit the canvas while
        preserving aspect ratio.

        Args:
            strokes: List of Stroke objects to render.
            size: Output image dimensions (square). Default 384.
            stroke_width: Line thickness in pixels. Default 3.
            padding: Margin around strokes in pixels. Default 20.

        Returns:
            PIL Image in RGB mode with rendered strokes.

        Example:
            >>> img = validator.render_strokes(result.strokes)
            >>> img.save('strokes.png')
        """
        # Get bounding box of all strokes
        all_points = np.vstack([s.points for s in strokes if len(s.points) > 0])
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)

        # Scale to fit in image with padding
        stroke_width_px = max_x - min_x
        stroke_height_px = max_y - min_y

        available = size - 2 * padding
        scale = min(available / max(stroke_width_px, 1), available / max(stroke_height_px, 1))

        # Create image
        img = Image.new('RGB', (size, size), 'white')
        draw = ImageDraw.Draw(img)

        # Center offset
        scaled_w = stroke_width_px * scale
        scaled_h = stroke_height_px * scale
        offset_x = (size - scaled_w) / 2 - min_x * scale
        offset_y = (size - scaled_h) / 2 - min_y * scale

        # Draw strokes
        for stroke in strokes:
            if len(stroke.points) < 2:
                continue
            pts = stroke.points * scale + np.array([offset_x, offset_y])
            pts_list = [(p[0], p[1]) for p in pts]
            draw.line(pts_list, fill='black', width=stroke_width)

        return img

    def recognize(self, image: Image.Image) -> str:
        """Run OCR on an image.

        Dispatches to either subprocess or direct recognition depending
        on the use_subprocess setting.

        Args:
            image: PIL Image to recognize. Should contain rendered text
                or strokes on a white background.

        Returns:
            Recognized text string.

        Raises:
            RuntimeError: If OCR fails (subprocess mode).
        """
        if self.use_subprocess:
            return self._recognize_subprocess(image)
        else:
            return self._recognize_direct(image)

    def _recognize_direct(self, image: Image.Image) -> str:
        """Run OCR directly in this process.

        Args:
            image: PIL Image to recognize.

        Returns:
            Recognized text string.
        """
        import torch
        self.load_model()

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=64)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return text.strip()

    _worker_process = None
    _worker_stdin = None
    _worker_stdout = None

    def _start_worker(self):
        """Start the persistent OCR worker process.

        Launches a Python subprocess that loads the TrOCR model and waits
        for images sent via stdin. The worker stays running until explicitly
        stopped with shutdown_worker().

        The worker protocol:
        - Input: Base64-encoded PNG image, one per line
        - Output: Recognized text, one per line
        - Special: "QUIT" input terminates the worker
        """
        import subprocess

        if OCRValidator._worker_process is not None:
            return

        # Worker code that stays running and processes requests
        worker_code = f'''
import sys
import warnings
warnings.filterwarnings("ignore")
import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load model once
processor = TrOCRProcessor.from_pretrained("{self.model_name}")
model = VisionEncoderDecoderModel.from_pretrained("{self.model_name}")
model.eval()
print("READY", flush=True)

# Process requests
while True:
    try:
        line = sys.stdin.readline().strip()
        if not line or line == "QUIT":
            break

        # Decode image
        img_data = base64.b64decode(line)
        image = Image.open(BytesIO(img_data)).convert("RGB")

        # Run OCR
        pixel_values = processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=64)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(text.strip(), flush=True)
    except Exception as e:
        print(f"ERROR: {{e}}", flush=True)
'''

        print("Starting OCR worker process...")
        OCRValidator._worker_process = subprocess.Popen(
            [sys.executable, '-c', worker_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Wait for READY signal
        ready = OCRValidator._worker_process.stdout.readline().strip()
        if ready != "READY":
            raise RuntimeError(f"Worker failed to start: {ready}")
        print("OCR worker ready.")

    def _recognize_subprocess(self, image: Image.Image) -> str:
        """Run OCR via the persistent worker process.

        Encodes the image as base64 PNG, sends to the worker, and reads
        the recognized text response.

        Args:
            image: PIL Image to recognize.

        Returns:
            Recognized text string.

        Raises:
            RuntimeError: If the worker returns an error.
        """
        import base64
        from io import BytesIO

        self._start_worker()

        # Convert image to base64
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Send to worker
        OCRValidator._worker_process.stdin.write(img_b64 + '\n')
        OCRValidator._worker_process.stdin.flush()

        # Get result
        result = OCRValidator._worker_process.stdout.readline().strip()

        if result.startswith("ERROR:"):
            raise RuntimeError(result)

        return result

    @classmethod
    def shutdown_worker(cls):
        """Shutdown the persistent OCR worker process.

        Sends a quit signal to the worker and waits for it to terminate.
        If the worker doesn't respond within 5 seconds, it is forcefully
        killed.

        This should be called when OCR validation is complete to clean up
        resources. It's safe to call even if no worker is running.

        Example:
            >>> validator = OCRValidator()
            >>> # ... do validation ...
            >>> OCRValidator.shutdown_worker()
        """
        if cls._worker_process is not None:
            try:
                cls._worker_process.stdin.write("QUIT\n")
                cls._worker_process.stdin.flush()
                cls._worker_process.wait(timeout=5)
            except (BrokenPipeError, OSError, TimeoutError):
                cls._worker_process.kill()
            cls._worker_process = None
            print("OCR worker stopped.")

    def validate(
        self,
        result: InkResult,
        threshold: float = 0.8,
        case_sensitive: bool = False
    ) -> Tuple[bool, str, float]:
        """Validate an InkResult by checking if OCR can read it.

        Renders the strokes to an image, runs OCR, and compares the
        recognized text to the expected word using sequence matching.

        Args:
            result: InkResult to validate.
            threshold: Minimum similarity ratio (0-1) to pass. Default 0.8
                means the recognized text must be 80% similar to expected.
            case_sensitive: If False (default), comparison is case-insensitive.

        Returns:
            A tuple of (passed, recognized_text, similarity_score):
            - passed: True if similarity >= threshold
            - recognized_text: What OCR recognized
            - similarity_score: Float 0-1 indicating match quality

        Example:
            >>> passed, text, score = validator.validate(result, threshold=0.7)
            >>> if passed:
            ...     print(f"Valid! OCR read: '{text}'")
            ... else:
            ...     print(f"Failed: expected '{result.word}', got '{text}'")
        """
        from difflib import SequenceMatcher

        # Render strokes to image
        stroke_img = self.render_strokes(result.strokes)

        # Run OCR
        recognized = self.recognize(stroke_img)

        # Compare
        expected = result.word
        if not case_sensitive:
            recognized = recognized.lower()
            expected = expected.lower()

        # Calculate similarity
        similarity = SequenceMatcher(None, expected, recognized).ratio()

        passed = similarity >= threshold

        return passed, recognized, similarity

    def filter_results(
        self,
        results: List[InkResult],
        threshold: float = 0.8,
        verbose: bool = True
    ) -> List[InkResult]:
        """Filter a list of results, keeping only those that pass OCR validation.

        Useful for batch processing where some vectorizations may fail. Only
        results where OCR can recognize the text with sufficient confidence
        are retained.

        Args:
            results: List of InkResult objects to validate.
            threshold: Minimum similarity score to pass. Default 0.8.
            verbose: If True, prints progress for each result.

        Returns:
            Filtered list containing only InkResult objects that passed
            validation.

        Example:
            >>> all_results = [vectorizer.process(font, word) for word in words]
            >>> good_results = validator.filter_results(all_results)
            >>> print(f"Kept {len(good_results)}/{len(all_results)}")
        """
        self.load_model()

        passed = []
        for i, result in enumerate(results):
            is_valid, recognized, score = self.validate(result, threshold)

            if verbose:
                status = "PASS" if is_valid else "FAIL"
                print(f"  [{i+1}/{len(results)}] '{result.word}' -> '{recognized}' "
                      f"({score:.1%}) [{status}]")

            if is_valid:
                passed.append(result)

        if verbose:
            print(f"\nPassed: {len(passed)}/{len(results)} ({len(passed)/len(results):.1%})")

        return passed


def visualize(result: InkResult, output_path: str = None, show: bool = False):
    """Visualize InkSight vectorization result.

    Creates a side-by-side comparison showing the original rendered font
    image and the extracted strokes. Each stroke is drawn in a different
    color with markers at endpoints.

    Args:
        result: InkResult from vectorizer.process().
        output_path: Path to save the visualization image. If None, image
            is not saved (only shown if show=True).
        show: If True, displays the visualization interactively using
            matplotlib. Default False.

    Example:
        >>> result = vectorizer.process('/fonts/Script.ttf', 'Hello')
        >>> visualize(result, output_path='hello_vis.png')
        >>> visualize(result, show=True)  # Interactive display
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
    """Command-line interface for InkSight vectorization.

    Parses command-line arguments and runs vectorization on the specified
    font. Supports single word processing, full charset processing, and
    optional OCR validation.

    Usage:
        python inksight_vectorizer.py --font path/to/font.ttf --word "Hello"
        python inksight_vectorizer.py --font path/to/font.ttf --charset --output output_dir
        python inksight_vectorizer.py --font path/to/font.ttf --word "Test" --validate --show
    """
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
    parser.add_argument('--validate', '-v', action='store_true',
                        help='Validate results with TrOCR (filters bad results)')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='OCR validation threshold (default: 0.8)')

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

        # Validate with OCR if requested
        if args.validate:
            print("\nValidating with TrOCR...")
            validator = OCRValidator()
            valid_results = {}
            for char, result in results.items():
                passed, recognized, score = validator.validate(result, threshold=args.threshold)
                status = "PASS" if passed else "FAIL"
                print(f"  '{char}' -> '{recognized}' ({score:.1%}) [{status}]")
                if passed:
                    valid_results[char] = result
            print(f"\nPassed: {len(valid_results)}/{len(results)} characters")
            results = valid_results

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

        # Validate with OCR if requested
        if args.validate:
            validator = OCRValidator()
            passed, recognized, score = validator.validate(result, threshold=args.threshold)
            status = "PASS" if passed else "FAIL"
            print(f"OCR: '{recognized}' (similarity: {score:.1%}) [{status}]")
            if not passed:
                print("Result failed OCR validation - strokes may not be readable")

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
