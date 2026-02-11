"""OCR Validation for InkSight vectorization results.

This module provides TrOCR-based validation of stroke vectorization output.
It uses Microsoft's TrOCR handwriting recognition model to verify that
vectorized strokes are human-readable.

Extracted from inksight_vectorizer.py to enable:
- Cleaner separation of concerns
- Optional OCR dependency (transformers, torch)
- Easier testing and reuse

Example:
    >>> from ocr_validator import OCRValidator
    >>> from inksight_vectorizer import InkSightVectorizer
    >>>
    >>> vectorizer = InkSightVectorizer()
    >>> result = vectorizer.process('/fonts/Script.ttf', 'Hello')
    >>>
    >>> validator = OCRValidator()
    >>> passed, text, score = validator.validate(result)
    >>> print(f"OCR read: '{text}' (similarity: {score:.1%})")
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from inksight_vectorizer import InkResult, Stroke


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
        strokes: list[Stroke],
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
    ) -> tuple[bool, str, float]:
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
        results: list[InkResult],
        threshold: float = 0.8,
        verbose: bool = True
    ) -> list[InkResult]:
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
