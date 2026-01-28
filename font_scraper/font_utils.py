"""
Font utilities for the training pipeline.

Includes:
- FontDeduplicator: Find and remove duplicate fonts via perceptual hash
- FontScorer: Score fonts on quality metrics
- CursiveDetector: Detect connected/cursive fonts
- CompletenessChecker: Check font character coverage
- CharacterRenderer: Render individual characters as images
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imagehash
from scipy import ndimage


# ASCII printable characters (95 total)
ASCII_PRINTABLE = (
    ' !"#$%&\'()*+,-./0123456789:;<=>?@'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`'
    'abcdefghijklmnopqrstuvwxyz{|}~'
)


class FontDeduplicator:
    """Find and remove duplicate/similar fonts using perceptual hashing."""

    def __init__(self, threshold: int = 8):
        """
        Args:
            threshold: Max hamming distance to consider fonts as duplicates.
                       Lower = stricter matching. Default 8 works well.
        """
        self.threshold = threshold

    def compute_phash(self, font_path: str, sample_text: str = "The quick brown") -> Optional[str]:
        """Compute perceptual hash for a font."""
        try:
            font = ImageFont.truetype(font_path, 48)
            img = Image.new('L', (256, 64), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), sample_text, font=font, fill=0)
            return str(imagehash.phash(img))
        except Exception:
            return None

    def find_duplicates(self, font_scores: List[Dict]) -> List[List[Dict]]:
        """
        Find duplicate fonts based on perceptual hash.

        Args:
            font_scores: List of dicts with 'path' and 'phash' keys

        Returns:
            List of duplicate groups (each group is a list of font dicts)
        """
        # Group by exact phash first
        hash_groups = defaultdict(list)
        for score in font_scores:
            phash = score.get('phash')
            if phash:
                hash_groups[phash].append(score)

        # Find similar hashes within threshold
        duplicates = []
        processed = set()
        hashes = list(hash_groups.keys())

        for i, h1 in enumerate(hashes):
            if h1 in processed:
                continue

            group = hash_groups[h1].copy()

            for h2 in hashes[i + 1:]:
                if h2 in processed:
                    continue

                try:
                    hash1 = imagehash.hex_to_hash(h1)
                    hash2 = imagehash.hex_to_hash(h2)
                    distance = hash1 - hash2

                    if distance <= self.threshold:
                        group.extend(hash_groups[h2])
                        processed.add(h2)
                except Exception:
                    pass

            if len(group) > 1:
                duplicates.append(group)
            processed.add(h1)

        return duplicates

    def select_best_from_groups(
        self,
        duplicate_groups: List[List[Dict]],
        score_key: str = 'overall'
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        From each duplicate group, select the best font.

        Args:
            duplicate_groups: List of duplicate groups
            score_key: Dict key to use for ranking

        Returns:
            Tuple of (fonts_to_keep, fonts_to_remove)
        """
        keep = []
        remove = []

        for group in duplicate_groups:
            sorted_group = sorted(
                group,
                key=lambda x: x.get(score_key, 0),
                reverse=True
            )
            keep.append(sorted_group[0])
            remove.extend(sorted_group[1:])

        return keep, remove


class FontScorer:
    """Score fonts on multiple quality metrics."""

    def __init__(self, render_size: int = 64):
        self.render_size = render_size

    def score_font(self, font_path: str) -> Dict:
        """
        Score a font on multiple criteria.

        Returns dict with:
            - path, name
            - charset_coverage (0-1)
            - style_score (0-1)
            - phash (for deduplication)
            - overall (weighted average)
        """
        font_path = Path(font_path)
        scores = {
            'path': str(font_path),
            'name': font_path.stem,
            'charset_coverage': 0.0,
            'style_score': 0.0,
            'phash': None,
            'overall': 0.0,
        }

        try:
            font = ImageFont.truetype(str(font_path), self.render_size)
        except Exception as e:
            scores['error'] = str(e)
            return scores

        # Character set coverage
        covered = 0
        for char in ASCII_PRINTABLE:
            try:
                img = Image.new('L', (self.render_size * 2, self.render_size * 2), 255)
                draw = ImageDraw.Draw(img)
                draw.text((5, 5), char, font=font, fill=0)
                if img.getextrema() != (255, 255):
                    covered += 1
            except Exception:
                pass
        scores['charset_coverage'] = covered / len(ASCII_PRINTABLE)

        # Style score (ink coverage heuristic)
        try:
            img = Image.new('L', (self.render_size * 5, self.render_size * 2), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Hello", font=font, fill=0)
            arr = np.array(img)
            ink_ratio = np.sum(arr < 128) / arr.size

            if 0.03 < ink_ratio < 0.25:
                scores['style_score'] = 0.8
            elif 0.01 < ink_ratio < 0.35:
                scores['style_score'] = 0.5
            else:
                scores['style_score'] = 0.2
        except Exception:
            scores['style_score'] = 0.0

        # Perceptual hash
        try:
            img = Image.new('L', (256, 64), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "The quick brown", font=font, fill=0)
            scores['phash'] = str(imagehash.phash(img))
        except Exception:
            pass

        # Overall score
        scores['overall'] = (
            scores['charset_coverage'] * 0.6 +
            scores['style_score'] * 0.4
        )

        return scores


class CursiveDetector:
    """
    Detect connected/cursive fonts using multiple methods:
    1. Stroke connectivity analysis - counts connected components
    2. Contextual glyph comparison - compares isolated vs in-word characters

    Fonts with contextual alternates (different glyphs based on position)
    are problematic for single-character training even if letters don't
    visually connect.
    """

    def __init__(self, connectivity_threshold: float = 0.7, context_threshold: float = 0.15):
        """
        Args:
            connectivity_threshold: Score above this = cursive (connectivity method)
            context_threshold: Difference above this = has contextual alternates
        """
        self.connectivity_threshold = connectivity_threshold
        self.context_threshold = context_threshold

    def check(self, font_path: str, test_word: str = "minimum") -> Tuple[bool, float]:
        """
        Check if a font is cursive by analyzing stroke connectivity.

        Args:
            font_path: Path to font file
            test_word: Word to render (default "minimum" has many vertical strokes)

        Returns:
            Tuple of (is_cursive, connectivity_score)
        """
        try:
            font = ImageFont.truetype(font_path, 64)
        except Exception:
            return False, 0.0

        # Render test word
        try:
            # Get text size
            img = Image.new('L', (500, 100), 255)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), test_word, font=font)
            width = bbox[2] - bbox[0] + 20
            height = bbox[3] - bbox[1] + 20

            # Render at proper size
            img = Image.new('L', (width, height), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), test_word, font=font, fill=0)

            # Binarize
            arr = np.array(img)
            binary = (arr < 128).astype(np.uint8)

            # Count connected components
            labeled, num_components = ndimage.label(binary)

            # Expected components for separated letters
            # "minimum" has 7 letters, but 'i' has 2 parts each (dot + stem)
            # So expected ~9 components for print, ~1-3 for true cursive
            expected_components = len(test_word) + test_word.count('i') + test_word.count('j')

            # Connectivity score:
            # - 1.0 = fully connected (1 component for whole word)
            # - 0.5 = partially connected
            # - 0.0 = all letters separate (components ≈ expected)
            # - Fragmented fonts (components >> expected) also score ~0
            if num_components == 0:
                connectivity_score = 0.0
            elif num_components <= 3:
                # Very few components = truly connected cursive
                connectivity_score = 1.0 - (num_components - 1) / expected_components
            elif num_components < expected_components:
                # Fewer than expected = some letters connecting
                connectivity_score = (expected_components - num_components) / expected_components
            else:
                # At or above expected = print or fragmented
                connectivity_score = 0.0

            is_cursive = connectivity_score >= self.connectivity_threshold

            return is_cursive, connectivity_score

        except Exception:
            return False, 0.0

    def check_contextual(
        self,
        font_path: str,
        test_chars: str = "aeimnou"
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Detect contextual alternates by comparing isolated vs in-word glyphs.

        Cursive fonts often have different glyphs depending on position:
        - Initial (start of word)
        - Medial (middle of word)
        - Final (end of word)
        - Isolated

        Args:
            font_path: Path to font file
            test_chars: Characters to test (lowercase letters common in cursive)

        Returns:
            Tuple of (has_contextual, avg_difference, per_char_differences)
        """
        try:
            font = ImageFont.truetype(font_path, 64)
        except Exception:
            return False, 0.0, {}

        differences = {}

        for char in test_chars:
            try:
                diff = self._compare_glyph_contexts(font, char)
                if diff is not None:
                    differences[char] = diff
            except Exception:
                continue

        if not differences:
            return False, 0.0, {}

        avg_diff = sum(differences.values()) / len(differences)
        has_contextual = avg_diff >= self.context_threshold

        return has_contextual, avg_diff, differences

    def _compare_glyph_contexts(self, font: ImageFont.FreeTypeFont, char: str) -> Optional[float]:
        """
        Compare a character in two different word contexts.

        For print fonts, 'a' in "oao" should look identical to 'a' in "nan".
        For cursive fonts with contextual alternates, they'll differ based on
        surrounding characters.

        Returns normalized difference (0 = identical, 1 = completely different)
        """
        size = 64
        padding = 10

        # Use two different surrounding contexts
        # Context 1: surrounded by round letters
        context1 = f"o{char}o"
        # Context 2: surrounded by letters with vertical strokes
        context2 = f"n{char}n"

        img1 = self._extract_char_from_word(font, context1, 1, size, padding)
        if img1 is None:
            return None

        img2 = self._extract_char_from_word(font, context2, 1, size, padding)
        if img2 is None:
            return None

        # Compare the same character in different contexts
        diff = self._image_difference(img1, img2)

        return diff

    def _render_char_isolated(
        self,
        font: ImageFont.FreeTypeFont,
        char: str,
        size: int,
        padding: int
    ) -> Optional[np.ndarray]:
        """Render a single character in isolation."""
        try:
            # Get bounds
            temp_img = Image.new('L', (size * 2, size * 2), 255)
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), char, font=font)

            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if width <= 0 or height <= 0:
                return None

            # Render centered in a square
            img_size = max(width, height) + padding * 2
            img = Image.new('L', (img_size, img_size), 255)
            draw = ImageDraw.Draw(img)

            x = (img_size - width) // 2 - bbox[0]
            y = (img_size - height) // 2 - bbox[1]
            draw.text((x, y), char, font=font, fill=0)

            # Normalize to standard size
            img = img.resize((size, size), Image.LANCZOS)

            return np.array(img)

        except Exception:
            return None

    def _extract_char_from_word(
        self,
        font: ImageFont.FreeTypeFont,
        word: str,
        char_index: int,
        size: int,
        padding: int
    ) -> Optional[np.ndarray]:
        """Extract a specific character from a rendered word."""
        try:
            # Get full word bounds for rendering
            temp_img = Image.new('L', (500, 200), 255)
            temp_draw = ImageDraw.Draw(temp_img)
            word_bbox = temp_draw.textbbox((0, 0), word, font=font)

            # Use getlength() for accurate advance width positioning
            prefix = word[:char_index]
            target_char = word[char_index]

            # Calculate positions using advance widths
            prefix_width = font.getlength(prefix) if prefix else 0
            char_width = font.getlength(target_char)

            # Render full word
            word_width = int(font.getlength(word)) + padding * 2
            word_height = word_bbox[3] - word_bbox[1] + padding * 2

            img = Image.new('L', (word_width, word_height), 255)
            draw = ImageDraw.Draw(img)
            draw.text((padding - word_bbox[0], padding - word_bbox[1]), word, font=font, fill=0)

            # Calculate character bounds using advance widths
            # Offset by word_bbox[0] to account for left bearing
            char_left = int(prefix_width) + padding - word_bbox[0]
            char_right = int(prefix_width + char_width) + padding - word_bbox[0]

            # Small margin for safety
            margin = 1
            char_left = max(0, char_left - margin)
            char_right = min(word_width, char_right + margin)

            # Extract character region
            char_img = img.crop((char_left, 0, char_right, word_height))

            # Find actual ink bounds
            arr = np.array(char_img)
            ink_rows = np.any(arr < 200, axis=1)
            ink_cols = np.any(arr < 200, axis=0)

            if not np.any(ink_rows) or not np.any(ink_cols):
                return None

            row_min, row_max = np.where(ink_rows)[0][[0, -1]]
            col_min, col_max = np.where(ink_cols)[0][[0, -1]]

            # Crop to ink bounds with padding
            pad = 2
            row_min = max(0, row_min - pad)
            row_max = min(arr.shape[0], row_max + pad + 1)
            col_min = max(0, col_min - pad)
            col_max = min(arr.shape[1], col_max + pad + 1)

            cropped = arr[row_min:row_max, col_min:col_max]

            # Resize to standard size
            cropped_img = Image.fromarray(cropped)
            cropped_img = cropped_img.resize((size, size), Image.LANCZOS)

            return np.array(cropped_img)

        except Exception:
            return None

    def _image_difference(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate normalized difference between two images using perceptual hash.

        Returns value from 0 (identical) to 1 (very different).
        Uses imagehash for robust comparison that handles scale/position differences.
        """
        # Convert to PIL images
        pil1 = Image.fromarray(img1)
        pil2 = Image.fromarray(img2)

        # Compute perceptual hashes
        hash1 = imagehash.phash(pil1, hash_size=16)
        hash2 = imagehash.phash(pil2, hash_size=16)

        # Hamming distance (0 = identical, 256 = completely different for hash_size=16)
        distance = hash1 - hash2

        # Normalize to 0-1 range
        # hash_size=16 means 256 bits, so max distance is 256
        max_distance = 16 * 16
        normalized = distance / max_distance

        return normalized

    def check_all(self, font_path: str) -> Dict:
        """
        Run all cursive detection methods and return combined result.

        Returns dict with:
            - is_cursive: bool (True if ANY method flags it)
            - connectivity_score: float
            - contextual_score: float
            - contextual_details: dict per-character differences
            - method: which method(s) flagged it
        """
        # Run connectivity check
        is_cursive_conn, conn_score = self.check(font_path)

        # Run contextual check
        has_contextual, ctx_score, ctx_details = self.check_contextual(font_path)

        # Combine results
        is_cursive = is_cursive_conn or has_contextual

        methods = []
        if is_cursive_conn:
            methods.append('connectivity')
        if has_contextual:
            methods.append('contextual')

        return {
            'is_cursive': is_cursive,
            'connectivity_score': conn_score,
            'contextual_score': ctx_score,
            'contextual_details': ctx_details,
            'methods': methods
        }


class CompletenessChecker:
    """Check font character coverage."""

    def __init__(self, required_chars: str = ASCII_PRINTABLE):
        self.required_chars = required_chars

    def check(self, font_path: str) -> Tuple[float, List[str]]:
        """
        Check which characters a font can render.

        Args:
            font_path: Path to font file

        Returns:
            Tuple of (completeness_score, missing_chars_list)
        """
        try:
            font = ImageFont.truetype(font_path, 48)
        except Exception:
            return 0.0, list(self.required_chars)

        missing = []
        for char in self.required_chars:
            try:
                img = Image.new('L', (64, 64), 255)
                draw = ImageDraw.Draw(img)
                draw.text((5, 5), char, font=font, fill=0)

                # Check if anything was drawn
                if img.getextrema() == (255, 255):
                    missing.append(char)
            except Exception:
                missing.append(char)

        completeness = 1.0 - (len(missing) / len(self.required_chars))
        return completeness, missing


class CharacterRenderer:
    """Render individual characters as images for InkSight processing."""

    def __init__(self, height: int = 64, padding: int = 4):
        """
        Args:
            height: Target image height (SDT uses 64)
            padding: Padding around character
        """
        self.height = height
        self.padding = padding

    def render(
        self,
        font_path: str,
        char: str,
        output_dir: Optional[str] = None
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Render a single character.

        Args:
            font_path: Path to font file
            char: Character to render
            output_dir: If provided, save image here

        Returns:
            Tuple of (PIL Image, output_path or None)
        """
        try:
            # Load font scaled to target height
            font = ImageFont.truetype(font_path, self.height - self.padding * 2)
        except Exception:
            return None, None

        try:
            # Get character bounds
            temp_img = Image.new('L', (self.height * 2, self.height * 2), 255)
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), char, font=font)

            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]

            if char_width <= 0 or char_height <= 0:
                return None, None

            # Scale to fit height
            scale = (self.height - self.padding * 2) / max(char_height, 1)
            font_size = int((self.height - self.padding * 2) * scale)
            font_size = max(8, min(font_size, self.height * 2))

            font = ImageFont.truetype(font_path, font_size)
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]

            # Create image with proper width
            img_width = char_width + self.padding * 2
            img = Image.new('L', (img_width, self.height), 255)
            draw = ImageDraw.Draw(img)

            # Center character
            x = self.padding - bbox[0]
            y = self.padding - bbox[1]
            draw.text((x, y), char, font=font, fill=0)

            # Save if output_dir provided
            output_path = None
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Safe filename for special characters
                if char.isalnum():
                    filename = f"{char}.png"
                else:
                    filename = f"char_{ord(char):03d}.png"

                output_path = str(output_dir / filename)
                img.save(output_path)

            return img, output_path

        except Exception:
            return None, None

    def render_all(
        self,
        font_path: str,
        output_dir: str,
        chars: str = ASCII_PRINTABLE
    ) -> Dict[str, str]:
        """
        Render all characters from a font.

        Returns:
            Dict mapping char -> output_path (only successful renders)
        """
        results = {}
        for char in chars:
            img, path = self.render(font_path, char, output_dir)
            if path:
                results[char] = path
        return results


def quick_test():
    """Quick test of utilities."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python font_utils.py <font_path>")
        print("\nTests all utilities on the given font.")
        return

    font_path = sys.argv[1]
    print(f"Testing: {font_path}\n")

    # Completeness
    print("=== Completeness Check ===")
    checker = CompletenessChecker()
    score, missing = checker.check(font_path)
    print(f"Score: {score:.1%}")
    if missing:
        print(f"Missing ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # Cursive detection (both methods)
    print("\n=== Cursive Detection ===")
    detector = CursiveDetector()
    result = detector.check_all(font_path)
    print(f"Connectivity score: {result['connectivity_score']:.2f}")
    print(f"Contextual score: {result['contextual_score']:.2f}")
    if result['contextual_details']:
        details = ', '.join(f"{k}:{v:.2f}" for k, v in list(result['contextual_details'].items())[:5])
        print(f"  Per-char: {details}")
    print(f"Is cursive: {result['is_cursive']}", end="")
    if result['methods']:
        print(f" (flagged by: {', '.join(result['methods'])})")
    else:
        print()

    # Scoring
    print("\n=== Font Scoring ===")
    scorer = FontScorer()
    scores = scorer.score_font(font_path)
    print(f"Coverage: {scores['charset_coverage']:.1%}")
    print(f"Style: {scores['style_score']:.1%}")
    print(f"Overall: {scores['overall']:.1%}")
    print(f"PHash: {scores['phash']}")

    # Render sample
    print("\n=== Character Rendering ===")
    renderer = CharacterRenderer()
    img, _ = renderer.render(font_path, 'A')
    if img:
        print(f"Rendered 'A': {img.size[0]}x{img.size[1]} pixels")
    else:
        print("Failed to render 'A'")


if __name__ == '__main__':
    quick_test()
