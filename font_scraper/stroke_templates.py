"""Stroke templates for letter, numpad, and shape-based stroke generation.

This module contains all the template data used by the stroke editor for
generating strokes from font glyphs. Templates are organized by type:

- LETTER_TEMPLATES: Region-based stroke paths (TL, TC, TR, ML, MC, MR, BL, BC, BR)
- NUMPAD_TEMPLATE_VARIANTS: Numpad-grid waypoint templates (1-9 positions)
- NUMPAD_TEMPLATES: Legacy single-variant accessor
- NUMPAD_POS: Numpad region positions as fractions
- SHAPE_TEMPLATES: Parametric shape primitives (imported from stroke_shape_templates)
"""


# --- Letter stroke templates ---
# Each letter maps to a list of strokes. Each stroke is a tuple of regions
# it passes through: (start, end) or (start, via, end).
# Regions use a 3x3 grid: TL TC TR / ML MC MR / BL BC BR
# Shared endpoint regions between strokes indicate junctions.
# A 'via' region distinguishes strokes that share start+end but take
# different paths (e.g. B's vertical vs 3-shape both go TL->BL).
LETTER_TEMPLATES = {
    # --- Uppercase ---
    'A': [('TC', 'BL'), ('TC', 'BR'), ('ML', 'MR')],
    'B': [('TL', 'BL'), ('TL', 'MR', 'BL')],
    'C': [('TR', 'BR')],
    'D': [('TL', 'BL'), ('TL', 'MR', 'BL')],
    'E': [('TL', 'BL'), ('TL', 'TR'), ('ML', 'MR'), ('BL', 'BR')],
    'F': [('TL', 'BL'), ('TL', 'TR'), ('ML', 'MR')],
    'G': [('TR', 'MR'), ('MR', 'MC')],
    'H': [('TL', 'BL'), ('TR', 'BR'), ('ML', 'MR')],
    'I': [('TC', 'BC')],
    'J': [('TR', 'BC')],
    'K': [('TL', 'BL'), ('TR', 'ML'), ('ML', 'BR')],
    'L': [('TL', 'BL'), ('BL', 'BR')],
    'M': [('BL', 'TL', 'BC'), ('BC', 'TR', 'BR')],
    'N': [('TL', 'BL'), ('TL', 'BR'), ('TR', 'BR')],
    'O': [('TC', 'ML', 'BC'), ('TC', 'MR', 'BC')],
    'P': [('TL', 'BL'), ('TL', 'MR', 'ML')],
    'Q': [('TC', 'ML', 'BC'), ('TC', 'MR', 'BC'), ('MC', 'BR')],
    'R': [('TL', 'BL'), ('TL', 'MR', 'ML'), ('ML', 'BR')],
    'S': [('TR', 'BL')],
    'T': [('TL', 'TR'), ('TC', 'BC')],
    'U': [('TL', 'BC', 'TR')],
    'V': [('TL', 'BC'), ('BC', 'TR')],
    'W': [('TL', 'BL'), ('BL', 'TC'), ('TC', 'BR'), ('BR', 'TR')],
    'X': [('TL', 'BR'), ('TR', 'BL')],
    'Y': [('TL', 'MC'), ('TR', 'MC'), ('MC', 'BC')],
    'Z': [('TL', 'TR'), ('TR', 'BL'), ('BL', 'BR')],

    # --- Lowercase ---
    'a': [('TR', 'BR'), ('MR', 'ML', 'BC', 'MR')],
    'b': [('TL', 'BL'), ('BL', 'MR', 'BL')],
    'c': [('TR', 'BR')],
    'd': [('TR', 'BR'), ('BR', 'ML', 'TR')],
    'e': [('MR', 'ML', 'BC', 'MR')],
    'f': [('TR', 'BC'), ('ML', 'MR')],
    'g': [('TR', 'MR', 'ML', 'BC', 'TR'), ('TR', 'BR')],
    'h': [('TL', 'BL'), ('ML', 'MR', 'BR')],
    'i': [('TC', 'BC')],
    'j': [('TC', 'BC', 'BL')],
    'k': [('TL', 'BL'), ('TR', 'ML'), ('ML', 'BR')],
    'l': [('TC', 'BC')],
    'm': [('TL', 'BL'), ('TL', 'MC', 'BC'), ('MC', 'TR', 'BR')],
    'n': [('TL', 'BL'), ('TL', 'TR', 'BR')],
    'o': [('TC', 'ML', 'BC'), ('TC', 'MR', 'BC')],
    'p': [('TL', 'BL'), ('TL', 'MR', 'TL')],
    'q': [('TR', 'ML', 'TR'), ('TR', 'BR')],
    'r': [('TL', 'BL'), ('TL', 'TR')],
    's': [('TR', 'BL')],
    't': [('TC', 'BC'), ('ML', 'MR')],
    'u': [('TL', 'BC', 'BR'), ('TR', 'BR')],
    'v': [('TL', 'BC'), ('BC', 'TR')],
    'w': [('TL', 'BL'), ('BL', 'MC'), ('MC', 'BR'), ('BR', 'TR')],
    'x': [('TL', 'BR'), ('TR', 'BL')],
    'y': [('TL', 'MC'), ('TR', 'MC'), ('MC', 'BL')],
    'z': [('TL', 'TR'), ('TR', 'BL'), ('BL', 'BR')],

    # --- Digits ---
    '0': [('TC', 'ML', 'BC'), ('TC', 'MR', 'BC')],
    '1': [('TC', 'BC')],
    '2': [('TL', 'TR', 'MR', 'BL', 'BR')],
    '3': [('TL', 'MR', 'ML'), ('ML', 'MR', 'BL')],
    '4': [('TL', 'ML'), ('ML', 'MR'), ('TR', 'BR')],
    '5': [('TR', 'TL', 'ML'), ('ML', 'MR', 'BC', 'BL')],
    '6': [('TR', 'TL', 'BL', 'BC', 'MR', 'ML')],
    '7': [('TL', 'TR', 'BL')],
    '8': [('MC', 'TR', 'TL', 'MC'), ('MC', 'BL', 'BR', 'MC')],
    '9': [('MR', 'TL', 'TC', 'MR'), ('MR', 'BL')],
}

# --- Numpad-grid stroke templates ---
# Numpad positions:
#   7  8  9
#   4  5  6
#   1  2  3
# Waypoint types:
#   int        -> termination (stroke starts/ends at glyph edge near region)
#   'v(n)'     -> sharp vertex (abrupt direction change)
#   'c(n)'     -> smooth curve vertex (smooth direction change)
#   'i(n)'     -> intersection (go straight through, for self-crossing strokes)
#   'down'/'up'/'left'/'right' -> direction hint for next segment path tracing
#   'straight' -> prefer paths with no sharp direction changes
# Format: position alone (7) or tuple (deprecated, hints removed)
#
# NUMPAD_TEMPLATE_VARIANTS: Maps character -> dict of variant_name -> template
# Each character can have multiple valid stroke patterns. The system will try
# each variant and keep the best-scoring one.

NUMPAD_TEMPLATE_VARIANTS = {
    # --- Uppercase ---
    'A': {
        'pointed': [[1, 'v(8)', 3], [4, 6]],
        'flat_top': [[1, 7, 9, 3], [4, 6]],
    },
    'B': {
        'default': [[7, 1], [7, 'c(9)', 'v(6)', 'c(3)', 1]],
    },
    'C': {
        'default': [[9, 'c(7)', 'c(1)', 3]],
    },
    'D': {
        'default': [[7, 1], [7, 'c(9)', 'c(3)', 1]],
    },
    'E': {
        'default': [[9, 7, 1, 3], [4, 6]],
    },
    'F': {
        'default': [[9, 7, 1], [4, 6]],
    },
    'G': {
        'default': [[9, 'c(7)', 'c(1)', 3, 6], [6, 5]],
    },
    'H': {
        'default': [[7, 1], [9, 3], [4, 6]],
    },
    'I': {
        'default': [[8, 2]],
    },
    'J': {
        'default': [[8, 'c(1)']],
    },
    'K': {
        'default': [[7, 1], [9, 'v(4)', 3]],
    },
    'L': {
        'default': [[7, 1, 3]],
    },
    'M': {
        'default': [[1, 7, 'v(2)', 9, 3]],
    },
    'N': {
        'default': [[1, 7, 3, 9]],
    },
    'O': {
        'closed': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8]],
    },
    'P': {
        'default': [[1, 7, 'c(9)', 'c(6)', 4]],
    },
    'Q': {
        'default': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8], [5, 3]],
    },
    'R': {
        'default': [[1, 7, 'c(9)', 'c(6)', 4], [4, 3]],
    },
    'S': {
        'default': [[9, 'c(7)', 'c(4)', 'c(6)', 'c(3)', 1]],
    },
    'T': {
        'default': [[7, 9], [8, 2]],
    },
    'U': {
        'default': [[7, 'c(1)', 'c(3)', 9]],
    },
    'V': {
        'default': [[7, 'v(2)', 9]],
    },
    'W': {
        'default': [[7, 'v(1)', 'v(5)', 'v(3)', 9]],
    },
    'X': {
        'default': [[7, 3], [9, 1]],
    },
    'Y': {
        'default': [[7, 'v(5)', 9], [5, 2]],
    },
    'Z': {
        'default': [[7, 9, 1, 3]],
    },

    # --- Lowercase ---
    'a': {
        'default': [[9, 'c(7)', 'c(1)', 3], [9, 3]],
    },
    'b': {
        'default': [[7, 1, 'c(3)', 'c(9)', 4]],
    },
    'c': {
        'default': [[9, 'c(7)', 'c(1)', 3]],
    },
    'd': {
        'default': [[9, 'c(7)', 'c(1)', 3], [9, 3]],
    },
    'e': {
        'default': [[6, 4, 'c(1)', 'c(3)', 'c(6)']],
    },
    'f': {
        'default': [[9, 'c(8)', 2], [4, 6]],
    },
    'g': {
        'default': [[9, 'c(7)', 'c(1)', 'c(3)', 9], [9, 'c(3)']],
    },
    'h': {
        'default': [[7, 1], [4, 'c(9)', 3]],
    },
    'i': {
        'default': [[8, 2]],
    },
    'j': {
        'default': [[8, 'c(1)']],
    },
    'k': {
        'default': [[7, 1], [9, 'v(4)', 3]],
    },
    'l': {
        'default': [[8, 2]],
    },
    'm': {
        'default': [[1, 4, 'c(8)', 5], [5, 'c(9)', 3]],
    },
    'n': {
        'default': [[1, 4, 'c(9)', 3]],
    },
    'o': {
        'closed': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8]],
    },
    'p': {
        'default': [[4, 'c(7)', 'c(9)', 'c(6)', 1]],
    },
    'q': {
        'default': [[9, 'c(7)', 'c(1)', 'c(3)', 9], [9, 3]],
    },
    'r': {
        'default': [[1, 4, 'c(9)']],
    },
    's': {
        'default': [[9, 'c(7)', 'c(4)', 'c(6)', 'c(3)', 1]],
    },
    't': {
        'default': [[8, 2], [7, 9]],
    },
    'u': {
        'default': [[7, 'c(1)', 'c(3)', 9], [9, 3]],
    },
    'v': {
        'default': [[7, 'v(2)', 9]],
    },
    'w': {
        'default': [[7, 'v(1)', 'v(5)', 'v(3)', 9]],
    },
    'x': {
        'default': [[7, 3], [9, 1]],
    },
    'y': {
        'default': [[7, 'v(5)'], [9, 'v(5)', 'c(1)']],
    },
    'z': {
        'default': [[7, 9, 1, 3]],
    },

    # --- Digits ---
    '0': {
        'closed': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8]],
        'open': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)']],  # Incomplete loop from top
        'ccw_open': [[7, 'c(4)', 'c(1)', 'c(2)', 'c(3)', 'c(6)', 'c(9)', 'c(8)', 'c(7)', 4]],  # Full CCW loop: 7-4-1-2-3-6-9-8-7-4
    },
    '1': {
        'with_serif': [[7, 8, 2]],
        'simple': [[8, 2]],
        'with_base': [[4, 'v(8)', 2], [1, 2, 3]],  # Diagonal (4->8->2) + horizontal base
    },
    '2': {
        'default': [[7, 'v(4)', 1, 3]],  # Endpoints: 7 -> junction at 4 -> 1 or 3
    },
    '3': {
        'default': [[7, 'c(9)', 'v(4)', 'c(3)', 1]],
    },
    '4': {
        'open': [[7, 'v(4)', 6], [9, 3]],
        'closed': [[7, 'v(4)', 6, 9], [9, 3]],
    },
    '5': {
        'default': [[9, 7, 4, 'c(6)', 'c(3)', 1]],
        'two_stroke': [[9, 8, 7], [7, 4, 5, 6, 3, 2, 1]],
    },
    '6': {
        'default': [[9, 'c(7)', 'c(1)', 'c(3)', 'c(6)', 4]],
        'end_at_1': [[9, 'c(7)', 'c(1)', 'c(3)', 'c(6)', 1]],
    },
    '7': {
        'default': [[7, 9, 1]],
    },
    '8': {
        'default': [[8, 'c(7)', 'c(4)', 'c(1)', 'c(2)', 'c(3)', 'c(6)', 'c(9)', 8]],
        'from_9': [[9, 8, 7, 4, 'straight', 'i(5)', 'straight', 6, 3, 2, 1, 'i(5)', 9]],
    },
    '9': {
        'default': [[6, 'c(9)', 'c(8)', 'c(7)', 'c(4)', 6], [6, 'c(3)', 1]],
    },
}

# Legacy compatibility: NUMPAD_TEMPLATES returns first variant for each char
NUMPAD_TEMPLATES = {
    char: next(iter(variants.values()))
    for char, variants in NUMPAD_TEMPLATE_VARIANTS.items()
}

# Numpad region positions as (col_fraction, row_fraction) within glyph bbox.
# (0,0) = top-left, (1,1) = bottom-right
NUMPAD_POS = {
    7: (0.0, 0.0),
    8: (0.5, 0.0),
    9: (1.0, 0.0),
    4: (0.0, 0.5),
    5: (0.5, 0.5),
    6: (1.0, 0.5),
    1: (0.0, 1.0),
    2: (0.5, 1.0),
    3: (1.0, 1.0),
}


