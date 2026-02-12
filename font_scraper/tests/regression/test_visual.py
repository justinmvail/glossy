#!/usr/bin/env python3
"""Visual regression tests for stroke rendering.

This module provides visual regression testing capabilities for the font
stroke rendering system. It compares rendered outputs against golden images
to detect unintended visual changes.

Key functionality:
    - Image similarity comparison using structural similarity (SSIM)
    - Golden image management (save, load, update)
    - Rendering consistency verification
    - Character rendering validation against golden baselines

Example:
    Run all visual regression tests::

        $ pytest tests/regression/test_visual.py -v

    Run only slow tests (golden comparison)::

        $ pytest tests/regression/test_visual.py -v -m slow

    Update golden images after intentional changes::

        $ pytest tests/regression/test_visual.py --update-golden
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add parent dir to path for imports
_TEST_DIR = Path(__file__).parent
_PROJECT_ROOT = _TEST_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from stroke_rendering import (
    GlyphRenderer,
    render_char_image,
    render_glyph_mask,
    create_renderer,
    DEFAULT_CANVAS_SIZE,
)

# Directory for golden images
GOLDEN_DIR = _TEST_DIR.parent / "golden"

# Test font - using a font that should be available in the project
TEST_FONT_PATH = _PROJECT_ROOT / "fonts" / "dafont" / "Hello.ttf"

# Alternative test fonts in case primary is not available
FALLBACK_FONTS = [
    _PROJECT_ROOT / "fonts" / "dafont" / "Luna.ttf",
    _PROJECT_ROOT / "fonts" / "dafont" / "daniel.ttf",
]


def get_test_font() -> Path:
    """Get a valid test font path.

    Returns:
        Path to a test font file that exists.

    Raises:
        pytest.skip: If no test fonts are available.
    """
    if TEST_FONT_PATH.exists():
        return TEST_FONT_PATH
    for font in FALLBACK_FONTS:
        if font.exists():
            return font
    pytest.skip("No test fonts available")


# ---------------------------------------------------------------------------
# Image Comparison Utilities
# ---------------------------------------------------------------------------


def images_similar(img1: Image.Image | np.ndarray,
                   img2: Image.Image | np.ndarray,
                   threshold: float = 0.95) -> bool:
    """Compare two images, return True if similarity > threshold.

    Uses a combination of pixel-wise comparison and structural similarity
    to determine if two images are visually equivalent within a tolerance.

    Args:
        img1: First image (PIL Image or numpy array).
        img2: Second image (PIL Image or numpy array).
        threshold: Similarity threshold (0.0-1.0). Default 0.95 means
            95% similarity required to pass.

    Returns:
        True if the images are similar enough (similarity > threshold).

    Example:
        >>> img_a = Image.open("render_v1.png")
        >>> img_b = Image.open("render_v2.png")
        >>> if not images_similar(img_a, img_b):
        ...     print("Visual regression detected!")
    """
    # Convert to numpy arrays if needed
    arr1 = np.array(img1) if isinstance(img1, Image.Image) else img1
    arr2 = np.array(img2) if isinstance(img2, Image.Image) else img2

    # Check shapes match
    if arr1.shape != arr2.shape:
        return False

    # Handle empty images
    if arr1.size == 0:
        return arr2.size == 0

    # Normalize to float [0, 1] for comparison
    arr1_norm = arr1.astype(np.float32) / 255.0
    arr2_norm = arr2.astype(np.float32) / 255.0

    # Compute structural similarity using a simplified SSIM-like metric
    # This considers both pixel values and local structure
    similarity = compute_ssim_like(arr1_norm, arr2_norm)

    return similarity >= threshold


def compute_ssim_like(arr1: np.ndarray, arr2: np.ndarray,
                      window_size: int = 7) -> float:
    """Compute a simplified structural similarity metric.

    This is a simplified implementation inspired by SSIM that captures
    both luminance similarity and structural correlation.

    Args:
        arr1: First normalized image array (values in [0, 1]).
        arr2: Second normalized image array (values in [0, 1]).
        window_size: Local window size for structural comparison.

    Returns:
        Similarity score between 0.0 (completely different) and
        1.0 (identical).
    """
    # Simple MSE-based similarity for baseline
    mse = np.mean((arr1 - arr2) ** 2)
    # Convert MSE to similarity score (exponential decay)
    # MSE of 0 -> similarity 1.0, MSE of 0.1 -> ~0.37
    mse_similarity = np.exp(-10 * mse)

    # Correlation-based structural similarity
    flat1 = arr1.flatten()
    flat2 = arr2.flatten()

    # Handle constant images
    std1 = np.std(flat1)
    std2 = np.std(flat2)

    if std1 < 1e-10 and std2 < 1e-10:
        # Both images are constant (e.g., all white/black)
        corr_similarity = 1.0 if np.allclose(flat1, flat2) else 0.0
    elif std1 < 1e-10 or std2 < 1e-10:
        # One is constant, the other is not - low similarity
        corr_similarity = 0.0
    else:
        # Pearson correlation coefficient
        mean1, mean2 = np.mean(flat1), np.mean(flat2)
        numerator = np.sum((flat1 - mean1) * (flat2 - mean2))
        denominator = np.sqrt(np.sum((flat1 - mean1)**2) * np.sum((flat2 - mean2)**2))
        correlation = numerator / (denominator + 1e-10)
        corr_similarity = (correlation + 1) / 2  # Map [-1, 1] to [0, 1]

    # Combine metrics (weighted average)
    return 0.6 * mse_similarity + 0.4 * corr_similarity


def compute_pixel_similarity(img1: Image.Image | np.ndarray,
                             img2: Image.Image | np.ndarray) -> float:
    """Compute simple pixel-wise similarity between two images.

    Args:
        img1: First image.
        img2: Second image.

    Returns:
        Fraction of pixels that match exactly (0.0-1.0).
    """
    arr1 = np.array(img1) if isinstance(img1, Image.Image) else img1
    arr2 = np.array(img2) if isinstance(img2, Image.Image) else img2

    if arr1.shape != arr2.shape:
        return 0.0

    matching = np.sum(arr1 == arr2)
    total = arr1.size
    return matching / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Golden Image Management
# ---------------------------------------------------------------------------


def get_golden_path(name: str) -> Path:
    """Get the path for a golden image file.

    Args:
        name: Base name for the golden image (without extension).

    Returns:
        Path to the golden image file.
    """
    return GOLDEN_DIR / f"{name}.png"


def save_golden_image(image: Image.Image | np.ndarray, name: str) -> Path:
    """Save an image as a golden reference.

    Args:
        image: Image to save (PIL Image or numpy array).
        name: Base name for the golden file.

    Returns:
        Path to the saved golden image.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    if isinstance(image, np.ndarray):
        # Convert boolean mask to uint8
        if image.dtype == bool:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    path = get_golden_path(name)
    image.save(path, format='PNG')
    return path


def load_golden_image(name: str) -> Image.Image | None:
    """Load a golden reference image.

    Args:
        name: Base name of the golden file.

    Returns:
        PIL Image if the golden exists, None otherwise.
    """
    path = get_golden_path(name)
    if not path.exists():
        return None
    return Image.open(path)


def update_golden_image(image: Image.Image | np.ndarray, name: str,
                        backup: bool = True) -> Path:
    """Update a golden image, optionally keeping a backup.

    Use this when intentionally changing rendering behavior to update
    the golden baseline.

    Args:
        image: New golden image.
        name: Base name for the golden file.
        backup: If True, rename existing golden to .bak before replacing.

    Returns:
        Path to the updated golden image.
    """
    path = get_golden_path(name)

    if backup and path.exists():
        backup_path = path.with_suffix('.png.bak')
        path.rename(backup_path)

    return save_golden_image(image, name)


def golden_exists(name: str) -> bool:
    """Check if a golden image exists.

    Args:
        name: Base name of the golden file.

    Returns:
        True if the golden image exists.
    """
    return get_golden_path(name).exists()


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_font():
    """Fixture providing a valid test font path."""
    return get_test_font()


@pytest.fixture
def renderer(test_font):
    """Fixture providing a GlyphRenderer for the test font."""
    return GlyphRenderer(str(test_font))


@pytest.fixture
def update_golden(request):
    """Fixture to check if goldens should be updated.

    Use with --update-golden pytest flag.
    """
    return request.config.getoption("--update-golden", default=False)


def pytest_addoption(parser):
    """Add custom pytest command line options."""
    try:
        parser.addoption(
            "--update-golden",
            action="store_true",
            default=False,
            help="Update golden images instead of comparing against them"
        )
    except ValueError:
        # Option already added (e.g., by conftest.py)
        pass


# ---------------------------------------------------------------------------
# TestRenderCharMatchesGolden - Golden Image Comparison Tests
# ---------------------------------------------------------------------------


class TestRenderCharMatchesGolden:
    """Test rendered characters match golden reference images.

    These tests compare current rendering output against stored golden
    images to detect visual regressions.
    """

    # Characters to test against golden images
    TEST_CHARS = ['A', 'B', 'M', 'W', 'g', 'y']

    @pytest.mark.slow
    @pytest.mark.parametrize("char", TEST_CHARS)
    def test_render_char_matches_golden(self, renderer, char, update_golden):
        """Render a character and compare to golden image.

        This test renders a character and compares it against a stored
        golden image. If the golden doesn't exist (first run), it creates
        it and skips the comparison.

        Args:
            renderer: GlyphRenderer fixture.
            char: Character to test.
            update_golden: Whether to update goldens instead of comparing.
        """
        golden_name = f"render_char_{char}"

        # Render the character
        img = renderer.render_char(char, canvas_size=DEFAULT_CANVAS_SIZE)
        assert img is not None, f"Failed to render character '{char}'"

        if update_golden:
            # Update mode: save new golden and pass
            save_golden_image(img, golden_name)
            pytest.skip(f"Updated golden image for '{char}'")

        # Load golden
        golden = load_golden_image(golden_name)

        if golden is None:
            # First run: create golden and skip
            save_golden_image(img, golden_name)
            pytest.skip(
                f"Golden image for '{char}' did not exist, created it. "
                "Run test again to verify."
            )

        # Compare
        assert images_similar(img, golden, threshold=0.95), (
            f"Rendered character '{char}' does not match golden image. "
            f"Similarity is below 95% threshold."
        )

    @pytest.mark.slow
    def test_render_mask_matches_golden(self, test_font, update_golden):
        """Test that render_glyph_mask output matches golden."""
        char = 'A'
        golden_name = f"render_mask_{char}"

        # Render the mask
        mask = render_glyph_mask(str(test_font), char, canvas_size=DEFAULT_CANVAS_SIZE)
        assert mask is not None, f"Failed to render mask for '{char}'"

        # Convert mask to image for comparison
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        if update_golden:
            save_golden_image(mask_img, golden_name)
            pytest.skip(f"Updated golden mask for '{char}'")

        golden = load_golden_image(golden_name)

        if golden is None:
            save_golden_image(mask_img, golden_name)
            pytest.skip(f"Golden mask for '{char}' created. Run again to verify.")

        assert images_similar(mask_img, golden, threshold=0.95), (
            f"Rendered mask for '{char}' does not match golden."
        )


# ---------------------------------------------------------------------------
# TestStrokeRenderingConsistency - Determinism Tests
# ---------------------------------------------------------------------------


class TestStrokeRenderingConsistency:
    """Test that stroke rendering is consistent and deterministic.

    These tests verify that the same input always produces the same
    output, which is critical for regression testing reliability.
    """

    def test_same_input_produces_same_output(self, renderer):
        """Verify rendering the same character twice gives identical results.

        This tests rendering determinism - the foundation of reliable
        visual regression testing.
        """
        char = 'A'

        # Render twice
        img1 = renderer.render_char(char, canvas_size=DEFAULT_CANVAS_SIZE)
        img2 = renderer.render_char(char, canvas_size=DEFAULT_CANVAS_SIZE)

        assert img1 is not None, "First render failed"
        assert img2 is not None, "Second render failed"

        # Should be pixel-perfect identical
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        assert np.array_equal(arr1, arr2), (
            "Rendering is not deterministic - same input produced different output"
        )

    def test_same_input_produces_same_mask(self, test_font):
        """Verify mask generation is deterministic."""
        char = 'M'

        mask1 = render_glyph_mask(str(test_font), char, canvas_size=DEFAULT_CANVAS_SIZE)
        mask2 = render_glyph_mask(str(test_font), char, canvas_size=DEFAULT_CANVAS_SIZE)

        assert mask1 is not None, "First mask render failed"
        assert mask2 is not None, "Second mask render failed"

        # Note: masks are cached, so this should definitely be identical
        assert np.array_equal(mask1, mask2), (
            "Mask generation is not deterministic"
        )

    def test_render_glyph_returns_pil_image(self, renderer):
        """Verify render_char returns a PIL Image with correct properties."""
        char = 'B'
        canvas_size = 200

        img = renderer.render_char(char, canvas_size=canvas_size, mode='L')

        assert img is not None, "render_char returned None"
        assert isinstance(img, Image.Image), "render_char should return PIL Image"
        assert img.mode == 'L', f"Expected grayscale mode 'L', got '{img.mode}'"
        assert img.size == (canvas_size, canvas_size), (
            f"Expected size {canvas_size}x{canvas_size}, got {img.size}"
        )

    def test_render_char_rgb_mode(self, renderer):
        """Verify RGB mode rendering works correctly."""
        char = 'C'
        canvas_size = 150

        img = renderer.render_char(char, canvas_size=canvas_size, mode='RGB')

        assert img is not None, "RGB render failed"
        assert img.mode == 'RGB', f"Expected RGB mode, got '{img.mode}'"
        assert img.size == (canvas_size, canvas_size)

    def test_render_char_consistency_across_sizes(self, renderer):
        """Verify character shape is consistent across canvas sizes."""
        char = 'W'

        # Render at different sizes
        img_small = renderer.render_char(char, canvas_size=100)
        img_large = renderer.render_char(char, canvas_size=200)

        assert img_small is not None and img_large is not None

        # Resize small to large and compare structure
        img_small_scaled = img_small.resize((200, 200), Image.Resampling.BILINEAR)

        # Should have similar structure (not pixel-perfect due to scaling)
        similarity = compute_pixel_similarity(img_small_scaled, img_large)

        # Relaxed threshold due to scaling artifacts
        assert similarity > 0.5, (
            f"Character shape differs too much between canvas sizes "
            f"(similarity: {similarity:.2%})"
        )

    def test_mask_binary_values(self, test_font):
        """Verify mask contains only boolean values."""
        mask = render_glyph_mask(str(test_font), 'X', canvas_size=DEFAULT_CANVAS_SIZE)

        assert mask is not None, "Mask render failed"
        assert mask.dtype == bool, f"Expected bool dtype, got {mask.dtype}"

        # Verify it contains both True and False (not empty or fully filled)
        assert mask.any(), "Mask is completely empty"
        assert not mask.all(), "Mask is completely filled"


# ---------------------------------------------------------------------------
# TestRenderingEdgeCases - Edge Case Handling
# ---------------------------------------------------------------------------


class TestRenderingEdgeCases:
    """Test edge cases in rendering."""

    def test_render_space_character(self, renderer):
        """Test rendering a space character (should be mostly white)."""
        img = renderer.render_char(' ', canvas_size=DEFAULT_CANVAS_SIZE)

        if img is None:
            pytest.skip("Font doesn't support rendering space")

        # Space should be mostly white (background)
        arr = np.array(img)
        white_ratio = np.sum(arr == 255) / arr.size

        assert white_ratio > 0.99, (
            f"Space character should be nearly all white, got {white_ratio:.2%}"
        )

    def test_render_various_characters(self, renderer):
        """Test rendering various character types."""
        test_chars = ['A', 'z', '5', '@', '!']

        for char in test_chars:
            img = renderer.render_char(char, canvas_size=DEFAULT_CANVAS_SIZE)
            # Some fonts may not support all characters
            if img is not None:
                assert isinstance(img, Image.Image)
                # Verify there's some content (not all white)
                arr = np.array(img)
                if np.min(arr) < 255:  # Has some non-white pixels
                    assert np.min(arr) < 128, f"Character '{char}' appears too faint"

    def test_create_renderer_returns_none_for_invalid_font(self):
        """Test that create_renderer handles invalid fonts gracefully."""
        renderer = create_renderer("/nonexistent/font.ttf")
        assert renderer is None, "Expected None for invalid font path"

    def test_render_char_image_returns_bytes(self, test_font):
        """Test render_char_image returns PNG bytes."""
        result = render_char_image(str(test_font), 'T', canvas_size=DEFAULT_CANVAS_SIZE)

        assert result is not None, "render_char_image returned None"
        assert isinstance(result, bytes), "Expected bytes output"

        # Verify it's valid PNG (starts with PNG signature)
        png_signature = b'\x89PNG\r\n\x1a\n'
        assert result[:8] == png_signature, "Output is not valid PNG"


# ---------------------------------------------------------------------------
# Test Runner Utilities
# ---------------------------------------------------------------------------


def run_visual_tests():
    """Run visual regression tests and report results.

    Utility function for running tests programmatically.

    Returns:
        int: Exit code (0 for success, non-zero for failures).
    """
    return pytest.main([
        str(_TEST_DIR / "test_visual.py"),
        "-v",
        "--tb=short",
    ])


if __name__ == '__main__':
    # Allow running directly: python test_visual.py
    sys.exit(run_visual_tests())
