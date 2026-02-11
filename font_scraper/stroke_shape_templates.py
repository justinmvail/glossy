"""Shape templates for parametric stroke fitting.

This module contains SHAPE_TEMPLATES, a mapping of characters to lists of
parametric shape primitives that describe how to draw each character. These
templates define the geometric structure of letters, digits, and other
characters using a combination of primitive shapes (lines, arcs, loops).

The templates are used by the stroke fitting system to guide optimization.
Each template specifies:
    - The type and approximate parameters of each shape component
    - Optional parameter bounds for fine-tuning during optimization
    - Optional grouping to join multiple shapes into single strokes

Shape Types and Parameters:
    vline: (x_frac, y_start_frac, y_end_frac)
        Vertical line at fractional x position.

    hline: (y_frac, x_start_frac, x_end_frac)
        Horizontal line at fractional y position.

    diag: (x0_frac, y0_frac, x1_frac, y1_frac)
        Diagonal line between two fractional coordinates.

    arc_right: (cx_frac, cy_frac, rx_frac, ry_frac, ang_start, ang_end)
        Right-opening arc (semicircle or partial ellipse).

    arc_left: (cx_frac, cy_frac, rx_frac, ry_frac, ang_start, ang_end)
        Left-opening arc (mirrored semicircle).

    loop: (cx_frac, cy_frac, rx_frac, ry_frac)
        Full ellipse (closed curve).

    u_arc: (cx_frac, cy_frac, rx_frac, ry_frac)
        U-shaped arc (bottom half of ellipse).

Optional Keys Per Shape:
    group (int): Shapes with the same group value are joined into one stroke.
        Connected at their nearest endpoints.

    bounds (list): Override default parameter bounds for optimization.
        Format is a list of (lo, hi) tuples or None per parameter.
        None entries keep the default from SHAPES[name].get_bounds().

Coordinate System:
    All fractional parameters are in the range [0.0, 1.0] and are relative
    to the glyph bounding box:
        - (0.0, 0.0) = top-left corner
        - (1.0, 1.0) = bottom-right corner
    Arc angles are specified in degrees.

Example:
    The letter 'A' is defined as three shapes:
        - Two diagonal lines from bottom corners meeting at top center
        - A horizontal crossbar in the middle

    >>> SHAPE_TEMPLATES['A']
    [{'shape': 'diag', 'params': (0.0, 1.0, 0.5, 0.0), 'group': 0},
     {'shape': 'diag', 'params': (0.5, 0.0, 1.0, 1.0), 'group': 0},
     {'shape': 'hline', 'params': (0.55, 0.2, 0.8)}]
"""

# SHAPE_TEMPLATES: Character to shape primitives mapping.
#
# Each character maps to a list of shape dictionaries. Each dictionary
# contains:
#   - 'shape': The shape type name (vline, hline, diag, arc_right, etc.)
#   - 'params': Tuple of parameters for the shape function
#   - 'group' (optional): Integer grouping shapes into connected strokes
#   - 'bounds' (optional): List of (min, max) overrides for optimization
#
# The 'group' key assigns shapes to strokes. Shapes with the same group
# are concatenated into a single stroke (joined at their nearest
# endpoints). Shapes without a group key become separate strokes.
SHAPE_TEMPLATES = {
    # -------------------------------------------------------------------------
    # Uppercase Letters (A-Z)
    # -------------------------------------------------------------------------

    # A: Two diagonal legs meeting at apex + horizontal crossbar
    'A': [
        {'shape': 'diag', 'params': (0.0, 1.0, 0.5, 0.0), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 0.0, 1.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (0.55, 0.2, 0.8)},
    ],

    # B: Vertical stem + two stacked right-opening bumps
    'B': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.25, 0.45, 0.24, -90, 90),
         'bounds': [None, (0.10, 0.35), (0.15, 0.65), (0.10, 0.28), (-100, -80), (80, 100)], 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.75, 0.45, 0.24, -90, 90),
         'bounds': [None, (0.65, 0.90), (0.15, 0.65), (0.10, 0.28), (-100, -80), (80, 100)]},
    ],

    # C: Left-opening arc spanning full height
    'C': [
        {'shape': 'arc_left', 'params': (0.85, 0.5, 0.5, 0.5, -90, 90)},
    ],

    # D: Vertical stem + single large right-opening bump
    'D': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.5, 0.5, 0.5, -90, 90), 'group': 0},
    ],

    # E: Vertical stem + three horizontal bars (top, middle, bottom)
    'E': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'hline', 'params': (0.0, 0.15, 1.0)},
        {'shape': 'hline', 'params': (0.5, 0.15, 0.85)},
        {'shape': 'hline', 'params': (1.0, 0.15, 1.0)},
    ],

    # F: Vertical stem + two horizontal bars (top and middle only)
    'F': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'hline', 'params': (0.0, 0.15, 1.0)},
        {'shape': 'hline', 'params': (0.5, 0.15, 0.85)},
    ],

    # G: Left-opening arc + horizontal crossbar entering from right
    'G': [
        {'shape': 'arc_left', 'params': (0.85, 0.5, 0.5, 0.5, -90, 90), 'group': 0},
        {'shape': 'hline', 'params': (0.5, 0.5, 1.0), 'group': 0},
    ],

    # H: Two vertical stems + horizontal crossbar
    'H': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'vline', 'params': (0.85, 0.0, 1.0)},
        {'shape': 'hline', 'params': (0.5, 0.15, 0.85)},
    ],

    # I: Single centered vertical stem
    'I': [
        {'shape': 'vline', 'params': (0.5, 0.0, 1.0)},
    ],

    # J: Vertical stem with U-shaped bottom curve
    'J': [
        {'shape': 'vline', 'params': (0.7, 0.0, 0.7), 'group': 0},
        {'shape': 'u_arc', 'params': (0.5, 0.7, 0.3, 0.3), 'group': 0},
    ],

    # K: Vertical stem + two diagonals meeting at stem
    'K': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.15, 0.5), 'group': 1},
        {'shape': 'diag', 'params': (0.15, 0.5, 1.0, 1.0), 'group': 1},
    ],

    # L: Vertical stem + horizontal base
    'L': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (1.0, 0.15, 1.0), 'group': 0},
    ],

    # M: Two stems + V-shaped center meeting at middle
    'M': [
        {'shape': 'vline', 'params': (0.05, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.05, 0.0, 0.5, 0.5), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 0.5, 0.95, 0.0), 'group': 0},
        {'shape': 'vline', 'params': (0.95, 0.0, 1.0), 'group': 0},
    ],

    # N: Two stems + diagonal connecting top-left to bottom-right
    'N': [
        {'shape': 'vline', 'params': (0.1, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.1, 0.0, 0.9, 1.0), 'group': 0},
        {'shape': 'vline', 'params': (0.9, 0.0, 1.0), 'group': 0},
    ],

    # O: Full ellipse loop
    'O': [
        {'shape': 'loop', 'params': (0.5, 0.5, 0.45, 0.48)},
    ],

    # P: Vertical stem + upper right-opening bump
    'P': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.25, 0.45, 0.25, -90, 90), 'group': 0},
    ],

    # Q: Full ellipse loop + diagonal tail
    'Q': [
        {'shape': 'loop', 'params': (0.5, 0.45, 0.45, 0.45)},
        {'shape': 'diag', 'params': (0.5, 0.7, 0.95, 1.0)},
    ],

    # R: Vertical stem + upper bump + diagonal leg
    'R': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.25, 0.45, 0.25, -90, 90), 'group': 0},
        {'shape': 'diag', 'params': (0.4, 0.5, 1.0, 1.0)},
    ],

    # S: Two connected arcs forming snake-like shape
    'S': [
        {'shape': 'arc_left', 'params': (0.6, 0.28, 0.4, 0.22, -90, 90),
         'bounds': [None, (0.12, 0.32), (0.15, 0.6), (0.05, 0.18), (-100, -80), (80, 100)], 'group': 0},
        {'shape': 'arc_right', 'params': (0.4, 0.72, 0.4, 0.22, -90, 90),
         'bounds': [None, (0.68, 0.88), (0.15, 0.6), (0.05, 0.18), (-100, -80), (80, 100)], 'group': 0},
    ],

    # T: Horizontal top bar + vertical stem
    'T': [
        {'shape': 'hline', 'params': (0.0, 0.0, 1.0)},
        {'shape': 'vline', 'params': (0.5, 0.0, 1.0)},
    ],

    # U: Two vertical stems connected by U-arc at bottom
    'U': [
        {'shape': 'vline', 'params': (0.15, 0.0, 0.65), 'group': 0},
        {'shape': 'u_arc', 'params': (0.5, 0.65, 0.35, 0.35), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.0, 0.65), 'group': 0},
    ],

    # V: Two diagonals meeting at bottom center
    'V': [
        {'shape': 'diag', 'params': (0.0, 0.0, 0.5, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 1.0, 1.0, 0.0), 'group': 0},
    ],

    # W: Four diagonals forming W shape
    'W': [
        {'shape': 'diag', 'params': (0.0, 0.0, 0.25, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.25, 1.0, 0.5, 0.4), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 0.4, 0.75, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.75, 1.0, 1.0, 0.0), 'group': 0},
    ],

    # X: Two crossing diagonals
    'X': [
        {'shape': 'diag', 'params': (0.0, 0.0, 1.0, 1.0)},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.0, 1.0)},
    ],

    # Y: Two upper diagonals meeting at center + vertical stem down
    'Y': [
        {'shape': 'diag', 'params': (0.0, 0.0, 0.5, 0.5), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.5, 0.5), 'group': 0},
        {'shape': 'vline', 'params': (0.5, 0.5, 1.0), 'group': 0},
    ],

    # Z: Three strokes forming zig-zag
    'Z': [
        {'shape': 'hline', 'params': (0.0, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (1.0, 0.0, 1.0), 'group': 0},
    ],

    # -------------------------------------------------------------------------
    # Lowercase Letters (a-z)
    # -------------------------------------------------------------------------

    # a: Left-opening bowl + vertical stem on right
    'a': [
        {'shape': 'arc_left', 'params': (0.7, 0.5, 0.4, 0.45, -90, 90), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.1, 1.0), 'group': 0},
    ],

    # b: Full-height left stem + right-opening bowl
    'b': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.6, 0.4, 0.4, -90, 90), 'group': 0},
    ],

    # c: Left-opening arc (smaller than uppercase C)
    'c': [
        {'shape': 'arc_left', 'params': (0.8, 0.5, 0.45, 0.48, -90, 90)},
    ],

    # d: Left-opening bowl + full-height right stem
    'd': [
        {'shape': 'arc_left', 'params': (0.7, 0.6, 0.4, 0.4, -90, 90), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.0, 1.0), 'group': 0},
    ],

    # e: Horizontal crossbar + left-opening partial arc
    'e': [
        {'shape': 'hline', 'params': (0.45, 0.15, 0.85), 'group': 0},
        {'shape': 'arc_left', 'params': (0.7, 0.55, 0.4, 0.4, 0, 90), 'group': 0},
    ],

    # f: Curved top hook + vertical stem + horizontal crossbar
    'f': [
        {'shape': 'vline', 'params': (0.4, 0.15, 1.0), 'group': 0},
        {'shape': 'arc_left', 'params': (0.7, 0.15, 0.3, 0.15, -90, 0), 'group': 0},
        {'shape': 'hline', 'params': (0.35, 0.15, 0.7)},
    ],

    # g: Left-opening bowl + descending stem + bottom hook
    'g': [
        {'shape': 'arc_left', 'params': (0.7, 0.35, 0.4, 0.35, -90, 90), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.1, 0.85), 'group': 0},
        {'shape': 'u_arc', 'params': (0.5, 0.85, 0.35, 0.15), 'group': 0},
    ],

    # h: Full-height left stem + right arch + right stem
    'h': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.4, 0.35, 0.2, -90, 0), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.4, 1.0), 'group': 0},
    ],

    # i: Simple vertical stem (dot handled separately)
    'i': [
        {'shape': 'vline', 'params': (0.5, 0.25, 1.0)},
    ],

    # j: Vertical stem + bottom hook curve
    'j': [
        {'shape': 'vline', 'params': (0.5, 0.25, 0.8), 'group': 0},
        {'shape': 'u_arc', 'params': (0.3, 0.8, 0.2, 0.2), 'group': 0},
    ],

    # k: Full-height left stem + two diagonal arms
    'k': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'diag', 'params': (0.85, 0.25, 0.15, 0.55), 'group': 1},
        {'shape': 'diag', 'params': (0.15, 0.55, 0.85, 1.0), 'group': 1},
    ],

    # l: Simple full-height vertical stem
    'l': [
        {'shape': 'vline', 'params': (0.5, 0.0, 1.0)},
    ],

    # m: Left stem + two arches + three vertical portions
    'm': [
        {'shape': 'vline', 'params': (0.08, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.08, 0.35, 0.22, 0.18, -90, 0), 'group': 0},
        {'shape': 'vline', 'params': (0.5, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.5, 0.35, 0.22, 0.18, -90, 0), 'group': 0},
        {'shape': 'vline', 'params': (0.92, 0.2, 1.0), 'group': 0},
    ],

    # n: Left stem + arch + right stem
    'n': [
        {'shape': 'vline', 'params': (0.15, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.4, 0.35, 0.22, -90, 0), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.2, 1.0), 'group': 0},
    ],

    # o: Full ellipse loop (x-height)
    'o': [
        {'shape': 'loop', 'params': (0.5, 0.55, 0.4, 0.42)},
    ],

    # p: Descending left stem + right-opening bowl
    'p': [
        {'shape': 'vline', 'params': (0.15, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.35, 0.4, 0.3, -90, 90), 'group': 0},
    ],

    # q: Left-opening bowl + descending right stem
    'q': [
        {'shape': 'arc_left', 'params': (0.7, 0.35, 0.4, 0.3, -90, 90), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.2, 1.0), 'group': 0},
    ],

    # r: Left stem + partial right arch (no right stem)
    'r': [
        {'shape': 'vline', 'params': (0.2, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.2, 0.35, 0.3, 0.18, -90, 0), 'group': 0},
    ],

    # s: Two connected arcs forming snake-like shape (smaller)
    's': [
        {'shape': 'arc_left', 'params': (0.6, 0.32, 0.35, 0.22, -90, 90),
         'bounds': [None, (0.05, 0.45), None, None, None, None], 'group': 0},
        {'shape': 'arc_right', 'params': (0.4, 0.68, 0.35, 0.22, -90, 90),
         'bounds': [None, (0.55, 0.95), None, None, None, None], 'group': 0},
    ],

    # t: Vertical stem + horizontal crossbar
    't': [
        {'shape': 'vline', 'params': (0.4, 0.0, 1.0)},
        {'shape': 'hline', 'params': (0.3, 0.1, 0.75)},
    ],

    # u: Left stem + U-arc + right stem
    'u': [
        {'shape': 'vline', 'params': (0.15, 0.2, 0.65), 'group': 0},
        {'shape': 'u_arc', 'params': (0.5, 0.65, 0.35, 0.35), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.2, 1.0), 'group': 0},
    ],

    # v: Two diagonals meeting at bottom center
    'v': [
        {'shape': 'diag', 'params': (0.0, 0.2, 0.5, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 1.0, 1.0, 0.2), 'group': 0},
    ],

    # w: Four diagonals forming w shape
    'w': [
        {'shape': 'diag', 'params': (0.0, 0.2, 0.25, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.25, 1.0, 0.5, 0.5), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 0.5, 0.75, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.75, 1.0, 1.0, 0.2), 'group': 0},
    ],

    # x: Two crossing diagonals
    'x': [
        {'shape': 'diag', 'params': (0.0, 0.2, 1.0, 1.0)},
        {'shape': 'diag', 'params': (1.0, 0.2, 0.0, 1.0)},
    ],

    # y: Two upper diagonals meeting at center, one continuing to descender
    'y': [
        {'shape': 'diag', 'params': (0.0, 0.2, 0.5, 0.6), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.2, 0.15, 1.0), 'group': 0},
    ],

    # z: Three strokes forming zig-zag (x-height)
    'z': [
        {'shape': 'hline', 'params': (0.2, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.2, 0.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (1.0, 0.0, 1.0), 'group': 0},
    ],

    # -------------------------------------------------------------------------
    # Digits (0-9)
    # -------------------------------------------------------------------------

    # 0: Full ellipse loop
    '0': [
        {'shape': 'loop', 'params': (0.5, 0.5, 0.42, 0.48)},
    ],

    # 1: Simple vertical stem
    '1': [
        {'shape': 'vline', 'params': (0.5, 0.0, 1.0)},
    ],

    # 2: Top arc + diagonal + horizontal base
    '2': [
        {'shape': 'arc_left', 'params': (0.6, 0.25, 0.4, 0.25, -90, 45), 'group': 0},
        {'shape': 'diag', 'params': (0.7, 0.4, 0.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (1.0, 0.0, 1.0), 'group': 0},
    ],

    # 3: Two right-opening bumps stacked vertically
    '3': [
        {'shape': 'arc_right', 'params': (0.35, 0.27, 0.4, 0.27, -90, 90),
         'bounds': [None, (0.05, 0.45), None, None, None, None], 'group': 0},
        {'shape': 'arc_right', 'params': (0.35, 0.73, 0.4, 0.27, -90, 90),
         'bounds': [None, (0.55, 0.95), None, None, None, None], 'group': 0},
    ],

    # 4: Diagonal/vertical + horizontal crossbar + vertical stem
    '4': [
        {'shape': 'diag', 'params': (0.0, 0.0, 0.0, 0.6), 'group': 0},
        {'shape': 'hline', 'params': (0.6, 0.0, 0.85), 'group': 0},
        {'shape': 'vline', 'params': (0.7, 0.0, 1.0)},
    ],

    # 5: Top horizontal + vertical + bottom arc
    '5': [
        {'shape': 'hline', 'params': (0.0, 0.0, 0.9), 'group': 0},
        {'shape': 'vline', 'params': (0.1, 0.0, 0.45), 'group': 0},
        {'shape': 'arc_right', 'params': (0.2, 0.7, 0.45, 0.3, -90, 90), 'group': 0},
    ],

    # 6: Top arc curving into bottom loop
    '6': [
        {'shape': 'arc_left', 'params': (0.75, 0.3, 0.4, 0.35, -90, 60), 'group': 0},
        {'shape': 'loop', 'params': (0.5, 0.65, 0.38, 0.32), 'group': 0},
    ],

    # 7: Horizontal top + diagonal descending
    '7': [
        {'shape': 'hline', 'params': (0.0, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.25, 1.0), 'group': 0},
    ],

    # 8: Two stacked loops
    '8': [
        {'shape': 'loop', 'params': (0.5, 0.27, 0.32, 0.25),
         'bounds': [None, (0.1, 0.45), None, None], 'group': 0},
        {'shape': 'loop', 'params': (0.5, 0.73, 0.38, 0.27),
         'bounds': [None, (0.55, 0.9), None, None], 'group': 0},
    ],

    # 9: Top loop + descending arc
    '9': [
        {'shape': 'loop', 'params': (0.5, 0.35, 0.38, 0.32), 'group': 0},
        {'shape': 'arc_right', 'params': (0.25, 0.7, 0.4, 0.35, -60, 90), 'group': 0},
    ],
}
