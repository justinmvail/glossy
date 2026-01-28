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
    """Detect connected/cursive fonts using stroke connectivity analysis."""

    def __init__(self, threshold: float = 0.7):
        """
        Args:
            threshold: Connectivity score above this = cursive.
                       0 = all letters separate, 1 = all letters connected.
        """
        self.threshold = threshold

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

            is_cursive = connectivity_score >= self.threshold

            return is_cursive, connectivity_score

        except Exception:
            return False, 0.0


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

    # Cursive detection
    print("\n=== Cursive Detection ===")
    detector = CursiveDetector()
    is_cursive, connectivity = detector.check(font_path)
    print(f"Connectivity: {connectivity:.2f}")
    print(f"Is cursive: {is_cursive}")

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
