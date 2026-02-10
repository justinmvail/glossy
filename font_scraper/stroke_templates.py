"""Stroke templates for letter, numpad, and shape-based stroke generation.

This module contains all the template data used by the stroke editor for
generating strokes from font glyphs. Templates are organized into three
complementary systems:

Template Systems:
    LETTER_TEMPLATES:
        Region-based stroke paths using a 3x3 grid notation.
        Regions: TL, TC, TR, ML, MC, MR, BL, BC, BR
        Format: Each stroke is a tuple of regions it passes through.

    NUMPAD_TEMPLATE_VARIANTS:
        Numpad-grid waypoint templates using positions 1-9.
        Supports multiple variants per character for font-specific matching.
        Format: List of waypoint sequences with vertex type annotations.

    NUMPAD_TEMPLATES:
        Legacy single-variant accessor for backward compatibility.
        Returns the first variant for each character.

    NUMPAD_POS:
        Maps numpad positions (1-9) to fractional (x, y) coordinates.

    SHAPE_TEMPLATES:
        Imported from stroke_shape_templates module.
        Parametric shape primitives (lines, arcs, loops).

Coordinate Systems:
    Region Grid (LETTER_TEMPLATES):
        TL  TC  TR     (Top-Left, Top-Center, Top-Right)
        ML  MC  MR     (Middle-Left, Middle-Center, Middle-Right)
        BL  BC  BR     (Bottom-Left, Bottom-Center, Bottom-Right)

    Numpad Grid (NUMPAD_TEMPLATES):
        7   8   9      (positions match phone/keyboard numpad)
        4   5   6
        1   2   3

    Fractional Coordinates (NUMPAD_POS):
        (0.0, 0.0) = top-left corner
        (1.0, 1.0) = bottom-right corner

Waypoint Types (NUMPAD_TEMPLATES):
    int: Termination point - stroke starts/ends at glyph edge near region
    'v(n)': Sharp vertex - abrupt direction change at position n
    'c(n)': Smooth curve vertex - smooth direction change at position n
    'i(n)': Intersection - go straight through for self-crossing strokes
    'down'/'up'/'left'/'right': Direction hint for path tracing
    'straight': Prefer paths with no sharp direction changes

Typical usage example:
    >>> from stroke_templates import LETTER_TEMPLATES, NUMPAD_TEMPLATES
    >>> LETTER_TEMPLATES['A']
    [('TC', 'BL'), ('TC', 'BR'), ('ML', 'MR')]
    >>> NUMPAD_TEMPLATES['A']
    [[1, 'v(8)', 3], [4, 6]]
"""


# -----------------------------------------------------------------------------
# Letter Stroke Templates (Region-based)
# -----------------------------------------------------------------------------

# LETTER_TEMPLATES: Maps characters to region-based stroke definitions.
#
# Each letter maps to a list of strokes. Each stroke is a tuple of regions
# it passes through, specified as (start, end) or (start, via, end).
#
# Region notation uses a 3x3 grid overlay on the glyph bounding box:
#   TL (Top-Left)     TC (Top-Center)     TR (Top-Right)
#   ML (Middle-Left)  MC (Middle-Center)  MR (Middle-Right)
#   BL (Bottom-Left)  BC (Bottom-Center)  BR (Bottom-Right)
#
# Shared endpoint regions between strokes indicate junctions where
# strokes connect. A 'via' region (middle element in 3-tuples)
# distinguishes strokes that share start+end but take different paths.
# For example, B's vertical stem and curved bump both go TL->BL but
# the vertical goes direct while the curve goes via MR.
LETTER_TEMPLATES = {
    # -------------------------------------------------------------------------
    # Uppercase Letters (A-Z)
    # -------------------------------------------------------------------------

    # A: Two diagonal legs from apex + horizontal crossbar
    'A': [('TC', 'BL'), ('TC', 'BR'), ('ML', 'MR')],

    # B: Vertical stem + curved path from top through right back to bottom
    'B': [('TL', 'BL'), ('TL', 'MR', 'BL')],

    # C: Single arc from top-right to bottom-right (opening left)
    'C': [('TR', 'BR')],

    # D: Vertical stem + curved path through right side
    'D': [('TL', 'BL'), ('TL', 'MR', 'BL')],

    # E: Vertical stem + three horizontal bars
    'E': [('TL', 'BL'), ('TL', 'TR'), ('ML', 'MR'), ('BL', 'BR')],

    # F: Vertical stem + two horizontal bars (no bottom bar)
    'F': [('TL', 'BL'), ('TL', 'TR'), ('ML', 'MR')],

    # G: Arc from top-right curving to middle-right + horizontal spur
    'G': [('TR', 'MR'), ('MR', 'MC')],

    # H: Two vertical stems + horizontal crossbar
    'H': [('TL', 'BL'), ('TR', 'BR'), ('ML', 'MR')],

    # I: Single centered vertical stem
    'I': [('TC', 'BC')],

    # J: Vertical stem curving at bottom-center
    'J': [('TR', 'BC')],

    # K: Vertical stem + two diagonals meeting at middle-left
    'K': [('TL', 'BL'), ('TR', 'ML'), ('ML', 'BR')],

    # L: Vertical stem + horizontal base
    'L': [('TL', 'BL'), ('BL', 'BR')],

    # M: Left stem up through center down to right stem
    'M': [('BL', 'TL', 'BC'), ('BC', 'TR', 'BR')],

    # N: Left stem + diagonal + right stem
    'N': [('TL', 'BL'), ('TL', 'BR'), ('TR', 'BR')],

    # O: Closed loop with left and right halves
    'O': [('TC', 'ML', 'BC'), ('TC', 'MR', 'BC')],

    # P: Vertical stem + upper loop returning to middle-left
    'P': [('TL', 'BL'), ('TL', 'MR', 'ML')],

    # Q: Closed loop (like O) + diagonal tail
    'Q': [('TC', 'ML', 'BC'), ('TC', 'MR', 'BC'), ('MC', 'BR')],

    # R: Like P but with diagonal leg from middle-left to bottom-right
    'R': [('TL', 'BL'), ('TL', 'MR', 'ML'), ('ML', 'BR')],

    # S: Single S-curve from top-right to bottom-left
    'S': [('TR', 'BL')],

    # T: Horizontal top bar + vertical stem
    'T': [('TL', 'TR'), ('TC', 'BC')],

    # U: Connected arc from top-left through bottom to top-right
    'U': [('TL', 'BC', 'TR')],

    # V: Two diagonals meeting at bottom-center
    'V': [('TL', 'BC'), ('BC', 'TR')],

    # W: Four connected diagonals forming W shape
    'W': [('TL', 'BL'), ('BL', 'TC'), ('TC', 'BR'), ('BR', 'TR')],

    # X: Two crossing diagonals
    'X': [('TL', 'BR'), ('TR', 'BL')],

    # Y: Two upper diagonals meeting at center + vertical stem
    'Y': [('TL', 'MC'), ('TR', 'MC'), ('MC', 'BC')],

    # Z: Top bar + diagonal + bottom bar
    'Z': [('TL', 'TR'), ('TR', 'BL'), ('BL', 'BR')],

    # -------------------------------------------------------------------------
    # Lowercase Letters (a-z)
    # -------------------------------------------------------------------------

    # a: Right stem + counter-clockwise bowl
    'a': [('TR', 'BR'), ('MR', 'ML', 'BC', 'MR')],

    # b: Left stem + clockwise bowl at bottom
    'b': [('TL', 'BL'), ('BL', 'MR', 'BL')],

    # c: Simple arc opening right
    'c': [('TR', 'BR')],

    # d: Right stem + counter-clockwise bowl
    'd': [('TR', 'BR'), ('BR', 'ML', 'TR')],

    # e: Counter-clockwise loop with horizontal entry
    'e': [('MR', 'ML', 'BC', 'MR')],

    # f: Curved hook at top + vertical stem + crossbar
    'f': [('TR', 'BC'), ('ML', 'MR')],

    # g: Bowl with descending loop
    'g': [('TR', 'MR', 'ML', 'BC', 'TR'), ('TR', 'BR')],

    # h: Left stem + arch to right stem
    'h': [('TL', 'BL'), ('ML', 'MR', 'BR')],

    # i: Simple vertical stem
    'i': [('TC', 'BC')],

    # j: Vertical stem with bottom-left hook
    'j': [('TC', 'BC', 'BL')],

    # k: Left stem + two diagonal arms
    'k': [('TL', 'BL'), ('TR', 'ML'), ('ML', 'BR')],

    # l: Simple vertical stem
    'l': [('TC', 'BC')],

    # m: Left stem + two arches
    'm': [('TL', 'BL'), ('TL', 'MC', 'BC'), ('MC', 'TR', 'BR')],

    # n: Left stem + single arch to right
    'n': [('TL', 'BL'), ('TL', 'TR', 'BR')],

    # o: Closed loop
    'o': [('TC', 'ML', 'BC'), ('TC', 'MR', 'BC')],

    # p: Descending stem + bowl
    'p': [('TL', 'BL'), ('TL', 'MR', 'TL')],

    # q: Bowl + descending stem
    'q': [('TR', 'ML', 'TR'), ('TR', 'BR')],

    # r: Stem + partial arch
    'r': [('TL', 'BL'), ('TL', 'TR')],

    # s: S-curve
    's': [('TR', 'BL')],

    # t: Vertical stem + crossbar
    't': [('TC', 'BC'), ('ML', 'MR')],

    # u: Left descent + arc + right stem
    'u': [('TL', 'BC', 'BR'), ('TR', 'BR')],

    # v: Two diagonals meeting at bottom
    'v': [('TL', 'BC'), ('BC', 'TR')],

    # w: Four diagonals
    'w': [('TL', 'BL'), ('BL', 'MC'), ('MC', 'BR'), ('BR', 'TR')],

    # x: Two crossing diagonals
    'x': [('TL', 'BR'), ('TR', 'BL')],

    # y: Two upper strokes meeting + descender
    'y': [('TL', 'MC'), ('TR', 'MC'), ('MC', 'BL')],

    # z: Zig-zag pattern
    'z': [('TL', 'TR'), ('TR', 'BL'), ('BL', 'BR')],

    # -------------------------------------------------------------------------
    # Digits (0-9)
    # -------------------------------------------------------------------------

    # 0: Closed loop (oval)
    '0': [('TC', 'ML', 'BC'), ('TC', 'MR', 'BC')],

    # 1: Simple vertical stem
    '1': [('TC', 'BC')],

    # 2: Complex path from top through middle to bottom
    '2': [('TL', 'TR', 'MR', 'BL', 'BR')],

    # 3: Two stacked curves opening left
    '3': [('TL', 'MR', 'ML'), ('ML', 'MR', 'BL')],

    # 4: Diagonal/vertical + crossbar + vertical stem
    '4': [('TL', 'ML'), ('ML', 'MR'), ('TR', 'BR')],

    # 5: Top bar + vertical + bottom curve
    '5': [('TR', 'TL', 'ML'), ('ML', 'MR', 'BC', 'BL')],

    # 6: Top curve flowing into bottom loop
    '6': [('TR', 'TL', 'BL', 'BC', 'MR', 'ML')],

    # 7: Top bar + diagonal
    '7': [('TL', 'TR', 'BL')],

    # 8: Two stacked loops
    '8': [('MC', 'TR', 'TL', 'MC'), ('MC', 'BL', 'BR', 'MC')],

    # 9: Top loop + descending curve
    '9': [('MR', 'TL', 'TC', 'MR'), ('MR', 'BL')],
}

# -----------------------------------------------------------------------------
# Numpad-grid Stroke Templates
# -----------------------------------------------------------------------------

# NUMPAD_TEMPLATE_VARIANTS: Maps characters to variant dictionaries.
#
# Each character can have multiple valid stroke patterns. The system will
# try each variant and keep the best-scoring one for the given font.
#
# Numpad position reference:
#     7   8   9     (top row)
#     4   5   6     (middle row)
#     1   2   3     (bottom row)
#
# Waypoint annotation types:
#     int (e.g., 7):     Termination point at glyph edge near region
#     'v(n)' (e.g., 'v(8)'): Sharp vertex - abrupt direction change
#     'c(n)' (e.g., 'c(7)'): Smooth curve - gradual direction change
#     'i(n)' (e.g., 'i(5)'): Intersection - stroke crosses itself
#     'down'/'up'/'left'/'right': Direction hints for path tracing
#     'straight': Prefer paths without sharp direction changes
#
# Format: character -> dict of variant_name -> list of stroke sequences
NUMPAD_TEMPLATE_VARIANTS = {
    # -------------------------------------------------------------------------
    # Uppercase Letters (A-Z)
    # -------------------------------------------------------------------------

    # A: Pointed apex or flat-top variants
    'A': {
        'pointed': [[1, 'v(8)', 3], [4, 6]],
        'flat_top': [[1, 7, 9, 3], [4, 6]],
    },

    # B: Vertical stem + curved bumps
    'B': {
        'default': [[7, 1], [7, 'c(9)', 'v(6)', 'c(3)', 1]],
    },

    # C: Open arc from top-right to bottom-right
    'C': {
        'default': [[9, 'c(7)', 'c(1)', 3]],
    },

    # D: Vertical stem + single curved bump
    'D': {
        'default': [[7, 1], [7, 'c(9)', 'c(3)', 1]],
    },

    # E: Comb shape with three horizontal bars
    'E': {
        'default': [[9, 7, 1, 3], [4, 6]],
    },

    # F: Like E but without bottom bar
    'F': {
        'default': [[9, 7, 1], [4, 6]],
    },

    # G: Arc with horizontal spur
    'G': {
        'default': [[9, 'c(7)', 'c(1)', 3, 6], [6, 5]],
    },

    # H: Two stems with crossbar
    'H': {
        'default': [[7, 1], [9, 3], [4, 6]],
    },

    # I: Simple vertical stem
    'I': {
        'default': [[8, 2]],
    },

    # J: Stem with curved bottom
    'J': {
        'default': [[8, 'c(1)']],
    },

    # K: Stem with diagonal arms meeting at vertex
    'K': {
        'default': [[7, 1], [9, 'v(4)', 3]],
    },

    # L: Stem with horizontal base
    'L': {
        'default': [[7, 1, 3]],
    },

    # M: Connected path with central vertex
    'M': {
        'default': [[1, 7, 'v(2)', 9, 3]],
    },

    # N: Connected path forming N shape
    'N': {
        'default': [[1, 7, 3, 9]],
    },

    # O: Closed loop
    'O': {
        'closed': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8]],
    },

    # P: Stem with upper loop
    'P': {
        'default': [[1, 7, 'c(9)', 'c(6)', 4]],
    },

    # Q: Closed loop with tail
    'Q': {
        'default': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8], [5, 3]],
    },

    # R: Like P with diagonal leg
    'R': {
        'default': [[1, 7, 'c(9)', 'c(6)', 4], [4, 3]],
    },

    # S: S-curve through center
    'S': {
        'default': [[9, 'c(7)', 'c(4)', 'c(6)', 'c(3)', 1]],
    },

    # T: Horizontal top with vertical stem
    'T': {
        'default': [[7, 9], [8, 2]],
    },

    # U: Arc connecting two stems
    'U': {
        'default': [[7, 'c(1)', 'c(3)', 9]],
    },

    # V: Two strokes meeting at vertex
    'V': {
        'default': [[7, 'v(2)', 9]],
    },

    # W: Four connected strokes with vertices
    'W': {
        'default': [[7, 'v(1)', 'v(5)', 'v(3)', 9]],
    },

    # X: Two crossing diagonals
    'X': {
        'default': [[7, 3], [9, 1]],
    },

    # Y: Two upper strokes meeting + descender
    'Y': {
        'default': [[7, 'v(5)', 9], [5, 2]],
    },

    # Z: Zig-zag path
    'Z': {
        'default': [[7, 9, 1, 3]],
    },

    # -------------------------------------------------------------------------
    # Lowercase Letters (a-z)
    # -------------------------------------------------------------------------

    # a: Arc with right stem
    'a': {
        'default': [[9, 'c(7)', 'c(1)', 3], [9, 3]],
    },

    # b: Left stem with bowl
    'b': {
        'default': [[7, 1, 'c(3)', 'c(9)', 4]],
    },

    # c: Open arc
    'c': {
        'default': [[9, 'c(7)', 'c(1)', 3]],
    },

    # d: Arc with right stem (mirror of b)
    'd': {
        'default': [[9, 'c(7)', 'c(1)', 3], [9, 3]],
    },

    # e: Loop with horizontal entry
    'e': {
        'default': [[6, 4, 'c(1)', 'c(3)', 'c(6)']],
    },

    # f: Curved hook with crossbar
    'f': {
        'default': [[9, 'c(8)', 2], [4, 6]],
    },

    # g: Bowl with descending hook
    'g': {
        'default': [[9, 'c(7)', 'c(1)', 'c(3)', 9], [9, 'c(3)']],
    },

    # h: Stem with arch
    'h': {
        'default': [[7, 1], [4, 'c(9)', 3]],
    },

    # i: Simple stem
    'i': {
        'default': [[8, 2]],
    },

    # j: Stem with bottom hook
    'j': {
        'default': [[8, 'c(1)']],
    },

    # k: Stem with diagonal arms
    'k': {
        'default': [[7, 1], [9, 'v(4)', 3]],
    },

    # l: Simple stem
    'l': {
        'default': [[8, 2]],
    },

    # m: Two arches
    'm': {
        'default': [[1, 4, 'c(8)', 5], [5, 'c(9)', 3]],
    },

    # n: Single arch
    'n': {
        'default': [[1, 4, 'c(9)', 3]],
    },

    # o: Closed loop
    'o': {
        'closed': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8]],
    },

    # p: Descending stem with bowl
    'p': {
        'default': [[4, 'c(7)', 'c(9)', 'c(6)', 1]],
    },

    # q: Bowl with descending stem
    'q': {
        'default': [[9, 'c(7)', 'c(1)', 'c(3)', 9], [9, 3]],
    },

    # r: Stem with partial arch
    'r': {
        'default': [[1, 4, 'c(9)']],
    },

    # s: S-curve
    's': {
        'default': [[9, 'c(7)', 'c(4)', 'c(6)', 'c(3)', 1]],
    },

    # t: Stem with crossbar
    't': {
        'default': [[8, 2], [7, 9]],
    },

    # u: Arc with extension
    'u': {
        'default': [[7, 'c(1)', 'c(3)', 9], [9, 3]],
    },

    # v: Two strokes meeting at vertex
    'v': {
        'default': [[7, 'v(2)', 9]],
    },

    # w: Four strokes with vertices
    'w': {
        'default': [[7, 'v(1)', 'v(5)', 'v(3)', 9]],
    },

    # x: Two crossing strokes
    'x': {
        'default': [[7, 3], [9, 1]],
    },

    # y: Two upper strokes meeting + descender with curve
    'y': {
        'default': [[7, 'v(5)'], [9, 'v(5)', 'c(1)']],
    },

    # z: Zig-zag path
    'z': {
        'default': [[7, 9, 1, 3]],
    },

    # -------------------------------------------------------------------------
    # Digits (0-9)
    # -------------------------------------------------------------------------

    # 0: Multiple variants for different digit styles
    '0': {
        'closed': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8]],
        'open': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)']],  # Incomplete loop from top
        'ccw_open': [[7, 'c(4)', 'c(1)', 'c(2)', 'c(3)', 'c(6)', 'c(9)', 'c(8)', 'c(7)', 4]],  # Full CCW loop
    },

    # 1: Multiple variants for serif styles
    '1': {
        'with_serif': [[7, 8, 2]],
        'simple': [[8, 2]],
        'with_base': [[4, 'v(8)', 2], [1, 2, 3]],  # Diagonal start + horizontal base
    },

    # 2: Top curve through diagonal to base
    '2': {
        'default': [[7, 'v(4)', 1, 3]],  # Endpoints: 7 -> vertex at 4 -> 1 or 3
    },

    # 3: Two stacked curves
    '3': {
        'default': [[7, 'c(9)', 'v(4)', 'c(3)', 1]],
    },

    # 4: Open or closed variants
    '4': {
        'open': [[7, 'v(4)', 6], [9, 3]],
        'closed': [[7, 'v(4)', 6, 9], [9, 3]],
    },

    # 5: Top bar + stem + bottom curve
    '5': {
        'default': [[9, 7, 4, 'c(6)', 'c(3)', 1]],
        'two_stroke': [[9, 8, 7], [7, 4, 5, 6, 3, 2, 1]],
    },

    # 6: Top curve into bottom loop
    '6': {
        'default': [[9, 'c(7)', 'c(1)', 'c(3)', 'c(6)', 4]],
        'end_at_1': [[9, 'c(7)', 'c(1)', 'c(3)', 'c(6)', 1]],
    },

    # 7: Top bar + diagonal
    '7': {
        'default': [[7, 9, 1]],
    },

    # 8: Figure-eight with intersection
    '8': {
        'default': [[8, 'c(7)', 'c(4)', 'c(1)', 'c(2)', 'c(3)', 'c(6)', 'c(9)', 8]],
        'from_9': [[9, 8, 7, 4, 'straight', 'i(5)', 'straight', 6, 3, 2, 1, 'i(5)', 9]],
    },

    # 9: Top loop + descending curve
    '9': {
        'default': [[6, 'c(9)', 'c(8)', 'c(7)', 'c(4)', 6], [6, 'c(3)', 1]],
    },
}

# NUMPAD_TEMPLATES: Legacy compatibility accessor.
#
# Returns the first (default) variant for each character. This provides
# backward compatibility with code that expects a single template per
# character rather than the multi-variant system.
NUMPAD_TEMPLATES = {
    char: next(iter(variants.values()))
    for char, variants in NUMPAD_TEMPLATE_VARIANTS.items()
}

# NUMPAD_POS: Maps numpad positions to fractional coordinates.
#
# Each position (1-9) maps to a (col_fraction, row_fraction) tuple
# within the glyph bounding box:
#     - (0.0, 0.0) = top-left corner
#     - (1.0, 1.0) = bottom-right corner
#
# Layout matches standard phone/keyboard numpad:
#     7(0.0,0.0)  8(0.5,0.0)  9(1.0,0.0)
#     4(0.0,0.5)  5(0.5,0.5)  6(1.0,0.5)
#     1(0.0,1.0)  2(0.5,1.0)  3(1.0,1.0)
NUMPAD_POS = {
    7: (0.0, 0.0),  # Top-left
    8: (0.5, 0.0),  # Top-center
    9: (1.0, 0.0),  # Top-right
    4: (0.0, 0.5),  # Middle-left
    5: (0.5, 0.5),  # Middle-center
    6: (1.0, 0.5),  # Middle-right
    1: (0.0, 1.0),  # Bottom-left
    2: (0.5, 1.0),  # Bottom-center
    3: (1.0, 1.0),  # Bottom-right
}
