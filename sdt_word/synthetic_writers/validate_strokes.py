"""
Validate Generated Strokes Using OCR

Renders strokes to images and runs them through OCR to verify legibility.
Rejects characters that can't be correctly recognized.

Uses:
- TrOCR (Microsoft) for handwriting recognition
- Or Tesseract as fallback
- Optional: custom confidence thresholds

Requirements:
    pip install transformers torch pillow

Usage:
    from validate_strokes import StrokeValidator
    validator = StrokeValidator()
    is_valid, confidence, recognized = validator.validate_char('A', strokes)
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from PIL import Image, ImageDraw
import json


@dataclass
class ValidationResult:
    """Result of validating a single character."""
    char: str
    recognized: str
    confidence: float
    is_correct: bool
    is_acceptable: bool  # correct OR high-confidence similar
    reason: str = ""


class StrokeRenderer:
    """Render strokes to images for OCR validation."""
    
    def __init__(
        self,
        image_size: int = 128,
        stroke_width: int = 3,
        padding: int = 10,
        bg_color: int = 255,
        stroke_color: int = 0
    ):
        self.image_size = image_size
        self.stroke_width = stroke_width
        self.padding = padding
        self.bg_color = bg_color
        self.stroke_color = stroke_color
    
    def render(self, strokes: List[List[Tuple[float, float]]]) -> Image.Image:
        """Render strokes to PIL Image."""
        if not strokes or not any(strokes):
            # Return blank image
            return Image.new('L', (self.image_size, self.image_size), self.bg_color)
        
        # Flatten all points to find bounds
        all_points = []
        for stroke in strokes:
            all_points.extend(stroke)
        
        if not all_points:
            return Image.new('L', (self.image_size, self.image_size), self.bg_color)
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 and height == 0:
            return Image.new('L', (self.image_size, self.image_size), self.bg_color)
        
        # Calculate scale to fit in image with padding
        available = self.image_size - 2 * self.padding
        scale = available / max(width, height) if max(width, height) > 0 else 1
        
        # Center offset
        scaled_width = width * scale
        scaled_height = height * scale
        offset_x = (self.image_size - scaled_width) / 2 - min_x * scale
        offset_y = (self.image_size - scaled_height) / 2 - min_y * scale
        
        # Create image
        img = Image.new('L', (self.image_size, self.image_size), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw strokes
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            
            # Transform points
            transformed = [
                (p[0] * scale + offset_x, p[1] * scale + offset_y)
                for p in stroke
            ]
            
            # Draw as connected lines
            draw.line(transformed, fill=self.stroke_color, width=self.stroke_width)
        
        return img
    
    def render_word(
        self, 
        word_strokes: List[List[List[Tuple[float, float]]]],
        char_spacing: float = 1.2
    ) -> Image.Image:
        """Render multiple characters as a word."""
        if not word_strokes:
            return Image.new('L', (self.image_size * 4, self.image_size), self.bg_color)
        
        # Render each character
        char_images = []
        for char_strokes in word_strokes:
            img = self.render(char_strokes)
            char_images.append(img)
        
        # Calculate total width
        total_width = sum(img.width for img in char_images)
        spacing_width = int((len(char_images) - 1) * self.image_size * (char_spacing - 1))
        final_width = total_width + spacing_width
        
        # Create combined image
        combined = Image.new('L', (final_width, self.image_size), self.bg_color)
        
        x_offset = 0
        for img in char_images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width + int(self.image_size * (char_spacing - 1))
        
        return combined


class TrOCRValidator:
    """Validate using Microsoft's TrOCR model."""
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self._loaded = False
    
    def _load_model(self):
        """Lazy load the model."""
        if self._loaded:
            return
        
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            
            print(f"Loading TrOCR model: {self.model_name}")
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.model.eval()
            self._loaded = True
            print("TrOCR model loaded successfully")
            
        except ImportError:
            raise ImportError(
                "TrOCR requires: pip install transformers torch\n"
                "Or use TesseractValidator as fallback"
            )
    
    def recognize(self, image: Image.Image) -> Tuple[str, float]:
        """Recognize text in image, return (text, confidence)."""
        self._load_model()
        
        import torch
        
        # Convert to RGB (TrOCR expects RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        
        # Generate with scores
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_length=32
            )
        
        # Decode
        generated_ids = outputs.sequences
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Calculate confidence from scores
        if outputs.scores:
            # Average probability of generated tokens
            probs = []
            for i, score in enumerate(outputs.scores):
                token_id = generated_ids[0, i + 1]  # +1 because first token is BOS
                prob = torch.softmax(score, dim=-1)[0, token_id].item()
                probs.append(prob)
            confidence = np.mean(probs) if probs else 0.0
        else:
            confidence = 0.5  # Default if scores not available
        
        return text.strip(), confidence


class TesseractValidator:
    """Validate using Tesseract OCR (fallback, less accurate for handwriting)."""
    
    def __init__(self, config: str = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"):
        self.config = config
        self._check_tesseract()
    
    def _check_tesseract(self):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
        except Exception:
            raise ImportError(
                "Tesseract not found. Install with:\n"
                "  Ubuntu: sudo apt install tesseract-ocr\n"
                "  Mac: brew install tesseract\n"
                "  Then: pip install pytesseract"
            )
    
    def recognize(self, image: Image.Image) -> Tuple[str, float]:
        """Recognize text in image."""
        import pytesseract
        
        # Get detailed data including confidence
        data = pytesseract.image_to_data(image, config=self.config, output_type=pytesseract.Output.DICT)
        
        # Find the best result
        texts = []
        confidences = []
        
        for i, text in enumerate(data['text']):
            if text.strip():
                texts.append(text.strip())
                conf = data['conf'][i]
                confidences.append(conf / 100.0 if conf > 0 else 0.0)
        
        if texts:
            # Return concatenated text and average confidence
            return ''.join(texts), np.mean(confidences)
        
        return '', 0.0


class StrokeValidator:
    """
    Validate generated strokes by checking OCR legibility.
    
    Usage:
        validator = StrokeValidator()
        
        # Validate single character
        result = validator.validate_char('A', strokes)
        if result.is_acceptable:
            # Use the strokes
            pass
        
        # Validate batch
        results = validator.validate_batch(char_stroke_pairs)
        valid_chars = [r for r in results if r.is_acceptable]
    """
    
    # Characters that commonly get confused
    SIMILAR_CHARS = {
        'O': ['0', 'Q', 'D'],
        '0': ['O', 'Q', 'D'],
        'I': ['1', 'l', '|'],
        '1': ['I', 'l', '|'],
        'l': ['1', 'I', '|'],
        'S': ['5', '$'],
        '5': ['S'],
        'Z': ['2'],
        '2': ['Z'],
        'B': ['8'],
        '8': ['B'],
        'G': ['6'],
        '6': ['G', 'b'],
        'g': ['9', 'q'],
        '9': ['g', 'q'],
        'q': ['9', 'g'],
        'b': ['6', 'd'],
        'd': ['b'],
        'p': ['P'],
        'P': ['p'],
        'n': ['h'],
        'h': ['n'],
        'u': ['v', 'U'],
        'v': ['u', 'V'],
        'm': ['n', 'M'],
        'w': ['W'],
        'c': ['C', '('],
        'C': ['c', '('],
    }
    
    def __init__(
        self,
        ocr_backend: str = 'trocr',  # 'trocr' or 'tesseract'
        confidence_threshold: float = 0.5,
        accept_similar: bool = True,
        image_size: int = 128,
        stroke_width: int = 3
    ):
        self.confidence_threshold = confidence_threshold
        self.accept_similar = accept_similar
        
        # Initialize renderer
        self.renderer = StrokeRenderer(
            image_size=image_size,
            stroke_width=stroke_width
        )
        
        # Initialize OCR backend
        if ocr_backend == 'trocr':
            self.ocr = TrOCRValidator()
        elif ocr_backend == 'tesseract':
            self.ocr = TesseractValidator()
        else:
            raise ValueError(f"Unknown OCR backend: {ocr_backend}")
    
    def validate_char(
        self, 
        expected_char: str, 
        strokes: List[List[Tuple[float, float]]]
    ) -> ValidationResult:
        """
        Validate a single character's strokes.
        
        Args:
            expected_char: The character the strokes should represent
            strokes: List of strokes, each stroke is list of (x, y) points
        
        Returns:
            ValidationResult with recognition details
        """
        # Render to image
        image = self.renderer.render(strokes)
        
        # Run OCR
        recognized, confidence = self.ocr.recognize(image)
        
        # Check if correct
        is_correct = recognized.lower() == expected_char.lower()
        
        # Check if acceptably similar
        is_similar = False
        if not is_correct and self.accept_similar:
            similar = self.SIMILAR_CHARS.get(expected_char, [])
            is_similar = recognized in similar or recognized.lower() in [s.lower() for s in similar]
        
        # Determine acceptability
        is_acceptable = is_correct or (is_similar and confidence >= self.confidence_threshold)
        
        # Build reason
        if is_correct:
            reason = "Correct recognition"
        elif is_similar and is_acceptable:
            reason = f"Similar character accepted ('{recognized}' ≈ '{expected_char}')"
        elif confidence < self.confidence_threshold:
            reason = f"Low confidence ({confidence:.2f} < {self.confidence_threshold})"
        else:
            reason = f"Misrecognized as '{recognized}'"
        
        return ValidationResult(
            char=expected_char,
            recognized=recognized,
            confidence=confidence,
            is_correct=is_correct,
            is_acceptable=is_acceptable,
            reason=reason
        )
    
    def validate_batch(
        self,
        char_stroke_pairs: List[Tuple[str, List[List[Tuple[float, float]]]]]
    ) -> List[ValidationResult]:
        """Validate multiple characters."""
        results = []
        for char, strokes in char_stroke_pairs:
            result = self.validate_char(char, strokes)
            results.append(result)
        return results
    
    def filter_valid(
        self,
        char_stroke_pairs: List[Tuple[str, List[List[Tuple[float, float]]]]],
        verbose: bool = True
    ) -> List[Tuple[str, List[List[Tuple[float, float]]]]]:
        """Filter to only valid character strokes."""
        valid = []
        rejected = []
        
        for char, strokes in char_stroke_pairs:
            result = self.validate_char(char, strokes)
            if result.is_acceptable:
                valid.append((char, strokes))
            else:
                rejected.append((char, result.reason))
        
        if verbose:
            print(f"Validation: {len(valid)}/{len(char_stroke_pairs)} accepted")
            if rejected:
                print(f"Rejected: {rejected[:5]}{'...' if len(rejected) > 5 else ''}")
        
        return valid


class QualityScorer:
    """
    Score stroke quality based on multiple metrics.
    
    Metrics:
    - OCR confidence
    - Stroke smoothness (low jitter)
    - Stroke completeness (no gaps)
    - Aspect ratio reasonableness
    """
    
    def __init__(self, validator: Optional[StrokeValidator] = None):
        self.validator = validator or StrokeValidator()
    
    def score(
        self, 
        char: str, 
        strokes: List[List[Tuple[float, float]]]
    ) -> Dict[str, float]:
        """
        Score stroke quality on multiple dimensions.
        
        Returns dict with scores 0-1 for each metric and overall score.
        """
        scores = {}
        
        # 1. OCR confidence
        result = self.validator.validate_char(char, strokes)
        scores['ocr_confidence'] = result.confidence
        scores['ocr_correct'] = 1.0 if result.is_correct else 0.0
        
        # 2. Smoothness (low variance in point-to-point distances)
        scores['smoothness'] = self._score_smoothness(strokes)
        
        # 3. Completeness (reasonable number of strokes/points)
        scores['completeness'] = self._score_completeness(strokes)
        
        # 4. Aspect ratio
        scores['aspect_ratio'] = self._score_aspect_ratio(strokes)
        
        # Overall weighted score
        weights = {
            'ocr_confidence': 0.4,
            'ocr_correct': 0.3,
            'smoothness': 0.1,
            'completeness': 0.1,
            'aspect_ratio': 0.1
        }
        
        scores['overall'] = sum(scores[k] * weights[k] for k in weights)
        
        return scores
    
    def _score_smoothness(self, strokes: List[List[Tuple[float, float]]]) -> float:
        """Score based on smoothness of strokes."""
        if not strokes:
            return 0.0
        
        variances = []
        for stroke in strokes:
            if len(stroke) < 3:
                continue
            
            # Calculate point-to-point distances
            points = np.array(stroke)
            diffs = np.diff(points, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            
            if len(distances) > 1:
                # Coefficient of variation (lower = smoother)
                cv = np.std(distances) / (np.mean(distances) + 1e-6)
                variances.append(cv)
        
        if not variances:
            return 0.5
        
        avg_cv = np.mean(variances)
        # Convert to 0-1 score (lower CV = higher score)
        return max(0, 1 - avg_cv / 2)
    
    def _score_completeness(self, strokes: List[List[Tuple[float, float]]]) -> float:
        """Score based on having reasonable stroke count and point density."""
        if not strokes:
            return 0.0
        
        num_strokes = len(strokes)
        total_points = sum(len(s) for s in strokes)
        
        # Penalize too few or too many strokes
        stroke_score = 1.0
        if num_strokes < 1:
            stroke_score = 0.0
        elif num_strokes > 10:
            stroke_score = max(0, 1 - (num_strokes - 10) / 20)
        
        # Penalize too few points
        point_score = min(1.0, total_points / 20)
        
        return (stroke_score + point_score) / 2
    
    def _score_aspect_ratio(self, strokes: List[List[Tuple[float, float]]]) -> float:
        """Score based on reasonable aspect ratio."""
        if not strokes:
            return 0.0
        
        all_points = [p for s in strokes for p in s]
        if not all_points:
            return 0.0
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        if height == 0:
            return 0.5
        
        ratio = width / height
        
        # Most characters have aspect ratio between 0.3 and 2.0
        if 0.3 <= ratio <= 2.0:
            return 1.0
        elif 0.1 <= ratio <= 3.0:
            return 0.7
        else:
            return 0.3


# CLI for testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate stroke data')
    parser.add_argument('--input', type=str, required=True, help='JSON file with stroke data')
    parser.add_argument('--backend', type=str, default='tesseract', choices=['trocr', 'tesseract'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Load stroke data
    with open(args.input) as f:
        data = json.load(f)
    
    # Initialize validator
    validator = StrokeValidator(
        ocr_backend=args.backend,
        confidence_threshold=args.threshold
    )
    
    # Validate each character
    print(f"Validating {len(data)} characters...")
    
    results = []
    for char, char_data in data.items():
        strokes = char_data['strokes']
        result = validator.validate_char(char, strokes)
        results.append(result)
        
        status = "✓" if result.is_acceptable else "✗"
        print(f"  {status} '{char}' → '{result.recognized}' ({result.confidence:.2f}) - {result.reason}")
    
    # Summary
    accepted = sum(1 for r in results if r.is_acceptable)
    print(f"\nSummary: {accepted}/{len(results)} characters accepted")
