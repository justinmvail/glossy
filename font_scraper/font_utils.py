"""Font utilities for the training pipeline.

This module provides a collection of utilities for font quality assessment,
deduplication, and rendering as part of a font-based training data pipeline.

The utilities handle:
    - Finding and removing duplicate fonts via perceptual hashing
    - Scoring fonts on quality metrics (coverage, style)
    - Detecting connected/cursive fonts unsuitable for character training
    - Checking font character coverage completeness
    - Rendering individual characters as images

Classes:
    FontDeduplicator: Find and remove duplicate fonts via perceptual hash.
    FontScorer: Score fonts on multiple quality metrics.
    CursiveDetector: Detect connected/cursive fonts using multiple methods.
    CompletenessChecker: Check font character coverage.
    CharacterRenderer: Render individual characters as images.

Example:
    Basic font quality assessment::

        from font_utils import FontScorer, CursiveDetector, CompletenessChecker

        font_path = "/path/to/font.ttf"

        # Check completeness
        checker = CompletenessChecker()
        score, missing = checker.check(font_path)
        print(f"Coverage: {score:.1%}, Missing: {len(missing)} chars")

        # Check for cursive
        detector = CursiveDetector()
        result = detector.check_all(font_path)
        if result['is_cursive']:
            print("Font is cursive - not suitable for training")

        # Get overall score
        scorer = FontScorer()
        scores = scorer.score_font(font_path)
        print(f"Overall quality: {scores['overall']:.1%}")
"""

import logging
from collections import defaultdict
from pathlib import Path

import imagehash

logger = logging.getLogger(__name__)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

# ASCII printable characters (95 total)
ASCII_PRINTABLE = (
    ' !"#$%&\'()*+,-./0123456789:;<=>?@'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`'
    'abcdefghijklmnopqrstuvwxyz{|}~'
)
"""str: All 95 printable ASCII characters (space through tilde).

This constant defines the standard character set used for font completeness
checks and coverage scoring. Fonts should ideally render all of these
characters to be considered complete.
"""

# ---------------------------------------------------------------------------
# Perceptual Hash Rendering Constants
# ---------------------------------------------------------------------------

# Font size for perceptual hash rendering
PHASH_FONT_SIZE = 48

# Canvas dimensions for perceptual hash images
PHASH_CANVAS_WIDTH = 256
PHASH_CANVAS_HEIGHT = 64

# Default render size for quality checks (FontScorer, CursiveDetector, etc.)
DEFAULT_RENDER_SIZE = 64


class FontDeduplicator:
    """Find and remove duplicate/similar fonts using perceptual hashing.

    Uses perceptual hashing (pHash) to compare fonts based on their visual
    appearance rather than file contents. This catches fonts that are visually
    identical or very similar even if they have different file names or metadata.

    Attributes:
        threshold: Maximum hamming distance to consider fonts as duplicates.
            Lower values mean stricter matching.

    Example:
        >>> dedup = FontDeduplicator(threshold=8)
        >>> fonts = [{'path': 'font1.ttf', 'phash': dedup.compute_phash('font1.ttf')},
        ...          {'path': 'font2.ttf', 'phash': dedup.compute_phash('font2.ttf')}]
        >>> duplicates = dedup.find_duplicates(fonts)
        >>> keep, remove = dedup.select_best_from_groups(duplicates)
    """

    def __init__(self, threshold: int = 8):
        """Initialize the deduplicator with a similarity threshold.

        Args:
            threshold: Max hamming distance to consider fonts as duplicates.
                Lower values mean stricter matching (fewer false positives).
                Default 8 works well for catching true duplicates while
                allowing stylistic variations.
        """
        self.threshold = threshold

    def compute_phash(self, font_path: str, sample_text: str = "The quick brown") -> str | None:
        """Compute perceptual hash for a font.

        Renders sample text using the font and computes a perceptual hash
        that is robust to minor variations in rendering.

        Args:
            font_path: Path to the font file (TTF, OTF, etc.).
            sample_text: Text to render for hashing. Default uses a
                phrase with varied letter shapes.

        Returns:
            Hex string representation of the perceptual hash, or None
            if the font cannot be loaded or rendered.
        """
        from stroke_rendering import GlyphRenderer

        try:
            renderer = GlyphRenderer(font_path, font_size=PHASH_FONT_SIZE)
            img = renderer.render_for_phash(
                sample_text,
                canvas_width=PHASH_CANVAS_WIDTH,
                canvas_height=PHASH_CANVAS_HEIGHT
            )
            if img is None:
                return None
            return str(imagehash.phash(img))
        except (OSError, ValueError):
            return None

    def find_duplicates(self, font_scores: list[dict]) -> list[list[dict]]:
        """Find duplicate fonts based on perceptual hash.

        Groups fonts by their perceptual hash similarity, finding both
        exact matches and near-duplicates within the threshold.

        Args:
            font_scores: List of dicts with 'path' and 'phash' keys.
                Additional keys (like scores) are preserved.

        Returns:
            List of duplicate groups, where each group is a list of
            font dicts that are duplicates of each other. Only groups
            with 2+ fonts are returned.
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
                except ValueError as e:
                    logger.debug("Hash comparison failed: %s", e)

            if len(group) > 1:
                duplicates.append(group)
            processed.add(h1)

        return duplicates

    def select_best_from_groups(
        self,
        duplicate_groups: list[list[dict]],
        score_key: str = 'overall'
    ) -> tuple[list[dict], list[dict]]:
        """From each duplicate group, select the best font to keep.

        Ranks fonts within each group by a score key and selects the
        highest-scoring font to keep, marking others for removal.

        Args:
            duplicate_groups: List of duplicate groups from find_duplicates().
            score_key: Dict key to use for ranking fonts. Default 'overall'.

        Returns:
            Tuple of (fonts_to_keep, fonts_to_remove):
                - fonts_to_keep: Best font from each duplicate group
                - fonts_to_remove: All other fonts from the groups
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
    """Score fonts on multiple quality metrics.

    Evaluates fonts on character set coverage and visual style characteristics
    to produce an overall quality score for filtering and ranking.

    Attributes:
        render_size: Size in pixels for test renders.

    Example:
        >>> scorer = FontScorer(render_size=64)
        >>> scores = scorer.score_font("/path/to/font.ttf")
        >>> print(f"Coverage: {scores['charset_coverage']:.1%}")
        >>> print(f"Overall: {scores['overall']:.1%}")
    """

    def __init__(self, render_size: int = DEFAULT_RENDER_SIZE):
        """Initialize the scorer.

        Args:
            render_size: Pixel size for test character renders. Larger
                sizes give more accurate style assessment but are slower.
                Default DEFAULT_RENDER_SIZE (64).
        """
        self.render_size = render_size

    def score_font(self, font_path: str) -> dict:
        """Score a font on multiple criteria.

        Evaluates character coverage, visual style, and computes a
        perceptual hash for deduplication.

        Args:
            font_path: Path to the font file.

        Returns:
            Dict containing:
                - path: Original font path
                - name: Font filename stem
                - charset_coverage: Fraction of ASCII chars that render (0-1)
                - style_score: Visual style quality heuristic (0-1)
                - phash: Perceptual hash string for deduplication
                - overall: Weighted average score (0-1)
                - error: Error message if font failed to load (optional)
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
        except OSError as e:
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
            except OSError as e:
                logger.debug("Failed to render char %r: %s", char, e)
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
        except OSError:
            scores['style_score'] = 0.0

        # Perceptual hash
        try:
            img = Image.new('L', (PHASH_CANVAS_WIDTH, PHASH_CANVAS_HEIGHT), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "The quick brown", font=font, fill=0)
            scores['phash'] = str(imagehash.phash(img))
        except (OSError, ValueError) as e:
            logger.debug("Failed to compute phash: %s", e)

        # Overall score
        scores['overall'] = (
            scores['charset_coverage'] * 0.6 +
            scores['style_score'] * 0.4
        )

        return scores


class CursiveDetector:
    """Detect connected/cursive fonts using multiple methods.

    Uses two complementary detection approaches:
        1. Stroke connectivity analysis - counts connected components in
           rendered text to detect physically connected letterforms
        2. Contextual glyph comparison - compares isolated vs in-word
           characters to detect contextual alternates

    Fonts with contextual alternates (different glyphs based on position)
    are problematic for single-character training even if letters don't
    visually connect.

    Attributes:
        connectivity_threshold: Score above this indicates cursive.
        context_threshold: Difference above this indicates contextual alternates.

    Example:
        >>> detector = CursiveDetector()
        >>> result = detector.check_all("/path/to/font.ttf")
        >>> if result['is_cursive']:
        ...     print(f"Cursive detected via: {result['methods']}")
    """

    def __init__(self, connectivity_threshold: float = 0.7, context_threshold: float = 0.10):
        """Initialize the detector with sensitivity thresholds.

        Args:
            connectivity_threshold: Score above this marks font as cursive
                when using the connectivity method. Range 0-1.
            context_threshold: Difference score above this marks font as
                having contextual alternates. Range 0-1.
        """
        self.connectivity_threshold = connectivity_threshold
        self.context_threshold = context_threshold

    def check(self, font_path: str, test_word: str = "minimum") -> tuple[bool, float]:
        """Check if a font is cursive by analyzing stroke connectivity.

        Renders a test word and counts connected components. Cursive fonts
        will have fewer components because letters connect together.

        Args:
            font_path: Path to font file.
            test_word: Word to render. Default "minimum" has many vertical
                strokes that would connect in cursive.

        Returns:
            Tuple of (is_cursive, connectivity_score):
                - is_cursive: True if connectivity score exceeds threshold
                - connectivity_score: 0-1 score (1 = fully connected)
        """
        try:
            font = ImageFont.truetype(font_path, DEFAULT_RENDER_SIZE)
        except OSError:
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

        except OSError:
            return False, 0.0

    def check_contextual(
        self,
        font_path: str,
        test_chars: str = "aeimnou"
    ) -> tuple[bool, float, dict[str, float]]:
        """Detect contextual alternates by comparing isolated vs in-word glyphs.

        Cursive fonts often have different glyphs depending on position
        (initial, medial, final, isolated). This method detects such
        contextual variations by comparing the same character in different
        surrounding contexts.

        Args:
            font_path: Path to font file.
            test_chars: Characters to test. Default uses lowercase letters
                commonly affected by contextual alternates.

        Returns:
            Tuple of (has_contextual, avg_difference, per_char_differences):
                - has_contextual: True if average difference exceeds threshold
                - avg_difference: Mean difference score across test chars
                - per_char_differences: Dict mapping char to its diff score
        """
        try:
            font = ImageFont.truetype(font_path, DEFAULT_RENDER_SIZE)
        except OSError:
            return False, 0.0, {}

        differences = {}

        for char in test_chars:
            try:
                diff = self._compare_glyph_contexts(font, char)
                if diff is not None:
                    differences[char] = diff
            except (OSError, ValueError):
                continue

        if not differences:
            return False, 0.0, {}

        avg_diff = sum(differences.values()) / len(differences)
        has_contextual = avg_diff >= self.context_threshold

        return has_contextual, avg_diff, differences

    def _compare_glyph_contexts(self, font: ImageFont.FreeTypeFont, char: str) -> float | None:
        """Compare a character in two different word contexts.

        For print fonts, 'a' in "oao" should look identical to 'a' in "nan".
        For cursive fonts with contextual alternates, they will differ based
        on the surrounding characters' connection points.

        Args:
            font: Loaded PIL font object.
            char: Single character to compare.

        Returns:
            Normalized difference from 0 (identical) to 1 (completely different),
            or None if comparison failed.
        """
        size = DEFAULT_RENDER_SIZE
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
    ) -> np.ndarray | None:
        """Render a single character in isolation.

        Args:
            font: Loaded PIL font object.
            char: Single character to render.
            size: Target image size (square).
            padding: Padding around the character.

        Returns:
            Numpy array of the rendered grayscale image, or None on failure.
        """
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

        except OSError:
            return None

    def _extract_char_from_word(
        self,
        font: ImageFont.FreeTypeFont,
        word: str,
        char_index: int,
        size: int,
        padding: int
    ) -> np.ndarray | None:
        """Extract a specific character from a rendered word.

        Renders the full word and crops out the target character based
        on advance width calculations.

        Args:
            font: Loaded PIL font object.
            word: Full word to render.
            char_index: Index of the character to extract (0-based).
            size: Target size for the extracted character image.
            padding: Padding for rendering.

        Returns:
            Numpy array of the extracted character image, or None on failure.
        """
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

        except OSError:
            return None

    def _image_difference(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate normalized difference between two images using perceptual hash.

        Uses imagehash for robust comparison that handles minor scale and
        position differences.

        Args:
            img1: First grayscale image as numpy array.
            img2: Second grayscale image as numpy array.

        Returns:
            Normalized difference from 0 (identical) to 1 (very different).
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

    def check_all(self, font_path: str) -> dict:
        """Run all cursive detection methods and return combined result.

        Executes both connectivity and contextual detection, combining
        results for comprehensive cursive detection.

        Args:
            font_path: Path to font file.

        Returns:
            Dict containing:
                - is_cursive: True if ANY method flags the font
                - connectivity_score: Score from connectivity check
                - contextual_score: Average score from contextual check
                - contextual_details: Per-character difference scores
                - methods: List of method names that flagged the font
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
    """Check font character coverage.

    Verifies which characters from a required set can be rendered by a font.
    Useful for filtering fonts that are missing common characters.

    Attributes:
        required_chars: String of characters to check for.

    Example:
        >>> checker = CompletenessChecker()
        >>> score, missing = checker.check("/path/to/font.ttf")
        >>> print(f"Coverage: {score:.1%}")
        >>> if missing:
        ...     print(f"Missing: {missing[:5]}...")
    """

    def __init__(self, required_chars: str = ASCII_PRINTABLE):
        """Initialize with the required character set.

        Args:
            required_chars: String of characters that fonts should support.
                Defaults to all ASCII printable characters.
        """
        self.required_chars = required_chars

    def check(self, font_path: str) -> tuple[float, list[str]]:
        """Check which characters a font can render.

        Attempts to render each required character and checks if any
        pixels were drawn.

        Args:
            font_path: Path to font file.

        Returns:
            Tuple of (completeness_score, missing_chars_list):
                - completeness_score: Fraction of chars that rendered (0-1)
                - missing_chars_list: List of characters that failed to render
        """
        try:
            font = ImageFont.truetype(font_path, PHASH_FONT_SIZE)
        except OSError:
            return 0.0, list(self.required_chars)

        missing = []
        for char in self.required_chars:
            try:
                img = Image.new('L', (DEFAULT_RENDER_SIZE, DEFAULT_RENDER_SIZE), 255)
                draw = ImageDraw.Draw(img)
                draw.text((5, 5), char, font=font, fill=0)

                # Check if anything was drawn
                if img.getextrema() == (255, 255):
                    missing.append(char)
            except OSError:
                missing.append(char)

        completeness = 1.0 - (len(missing) / len(self.required_chars))
        return completeness, missing


class CharacterRenderer:
    """Render individual characters as images for InkSight processing.

    Creates consistent, properly-sized character images from fonts for
    use in training data generation or OCR testing.

    Attributes:
        height: Target image height in pixels.
        padding: Padding around the character.

    Example:
        >>> renderer = CharacterRenderer(height=64, padding=4)
        >>> img, path = renderer.render("/path/to/font.ttf", 'A', '/output')
        >>> print(f"Rendered to {path}")
    """

    def __init__(self, height: int = DEFAULT_RENDER_SIZE, padding: int = 4):
        """Initialize the renderer.

        Args:
            height: Target image height in pixels. SDT uses DEFAULT_RENDER_SIZE (64).
            padding: Padding around the character in pixels.
        """
        self.height = height
        self.padding = padding

    def render(
        self,
        font_path: str,
        char: str,
        output_dir: str | None = None
    ) -> tuple[Image.Image | None, str | None]:
        """Render a single character.

        Creates a grayscale image with the character centered and scaled
        to fit the target height.

        Args:
            font_path: Path to font file.
            char: Single character to render.
            output_dir: If provided, save the image to this directory.

        Returns:
            Tuple of (PIL Image, output_path or None):
                - The rendered PIL Image, or None on failure
                - Path to saved file if output_dir was provided, else None
        """
        try:
            # Load font scaled to target height
            font = ImageFont.truetype(font_path, self.height - self.padding * 2)
        except OSError:
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

        except OSError:
            return None, None

    def render_all(
        self,
        font_path: str,
        output_dir: str,
        chars: str = ASCII_PRINTABLE
    ) -> dict[str, str]:
        """Render all characters from a font.

        Renders each character in the given set and saves to the output
        directory.

        Args:
            font_path: Path to font file.
            output_dir: Directory to save rendered images.
            chars: String of characters to render. Defaults to ASCII printable.

        Returns:
            Dict mapping successfully rendered characters to their output paths.
            Characters that failed to render are not included.
        """
        results = {}
        for char in chars:
            img, path = self.render(font_path, char, output_dir)
            if path:
                results[char] = path
        return results


def quick_test():
    """Quick test of font utilities.

    Runs all utility classes on a font provided as command line argument,
    displaying results for completeness, cursive detection, scoring, and
    rendering.

    Usage:
        python font_utils.py /path/to/font.ttf
    """
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
