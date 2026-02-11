"""Shared configuration for font filter scripts.

This module centralizes common configuration values used by:
    - run_ocr_prefilter.py
    - run_connectivity_filter.py
    - render_all_passing.py

Having these values in one place ensures consistency across scripts
and makes it easy to update settings globally.
"""

# Database path - used by all filter scripts
DB_PATH = 'fonts.db'

# Sample text for connectivity and visual testing
SAMPLE_TEXT = "Hello World!"

# Sample text for OCR validation (includes numbers for broader coverage)
OCR_SAMPLE_TEXT = "Hello World 123"

# Font size for filter rendering (balanced for visibility and speed)
FONT_SIZE = 48
