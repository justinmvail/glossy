"""Shape templates for parametric stroke fitting.

This module contains SHAPE_TEMPLATES - a mapping of characters to lists of
parametric shape primitives that describe how to draw each character.

Shape types and their parameters:
- vline: (x_frac, y_start_frac, y_end_frac)
- hline: (y_frac, x_start_frac, x_end_frac)
- diag: (x0_frac, y0_frac, x1_frac, y1_frac)
- arc_right: (cx_frac, cy_frac, rx_frac, ry_frac, ang_start, ang_end)
- arc_left: (cx_frac, cy_frac, rx_frac, ry_frac, ang_start, ang_end)
- loop: (cx_frac, cy_frac, rx_frac, ry_frac)
- u_arc: (cx_frac, cy_frac, rx_frac, ry_frac)

Optional keys per shape:
- 'group': int - Shapes with same group are joined into one stroke
- 'bounds': list - Override default parameter bounds for optimization
"""

SHAPE_TEMPLATES = {
    # --- Uppercase ---
    # 'group' key assigns shapes to strokes.  Shapes with the same group
    # are concatenated into a single stroke (joined at their nearest
    # endpoints).  Shapes without a group key become separate strokes.
    'A': [
        {'shape': 'diag', 'params': (0.0, 1.0, 0.5, 0.0), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 0.0, 1.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (0.55, 0.2, 0.8)},
    ],
    'B': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.25, 0.45, 0.24, -90, 90),
         'bounds': [None, (0.10, 0.35), (0.15, 0.65), (0.10, 0.28), (-100, -80), (80, 100)], 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.75, 0.45, 0.24, -90, 90),
         'bounds': [None, (0.65, 0.90), (0.15, 0.65), (0.10, 0.28), (-100, -80), (80, 100)]},
    ],
    'C': [
        {'shape': 'arc_left', 'params': (0.85, 0.5, 0.5, 0.5, -90, 90)},
    ],
    'D': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.5, 0.5, 0.5, -90, 90), 'group': 0},
    ],
    'E': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'hline', 'params': (0.0, 0.15, 1.0)},
        {'shape': 'hline', 'params': (0.5, 0.15, 0.85)},
        {'shape': 'hline', 'params': (1.0, 0.15, 1.0)},
    ],
    'F': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'hline', 'params': (0.0, 0.15, 1.0)},
        {'shape': 'hline', 'params': (0.5, 0.15, 0.85)},
    ],
    'G': [
        {'shape': 'arc_left', 'params': (0.85, 0.5, 0.5, 0.5, -90, 90), 'group': 0},
        {'shape': 'hline', 'params': (0.5, 0.5, 1.0), 'group': 0},
    ],
    'H': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'vline', 'params': (0.85, 0.0, 1.0)},
        {'shape': 'hline', 'params': (0.5, 0.15, 0.85)},
    ],
    'I': [
        {'shape': 'vline', 'params': (0.5, 0.0, 1.0)},
    ],
    'J': [
        {'shape': 'vline', 'params': (0.7, 0.0, 0.7), 'group': 0},
        {'shape': 'u_arc', 'params': (0.5, 0.7, 0.3, 0.3), 'group': 0},
    ],
    'K': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.15, 0.5), 'group': 1},
        {'shape': 'diag', 'params': (0.15, 0.5, 1.0, 1.0), 'group': 1},
    ],
    'L': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (1.0, 0.15, 1.0), 'group': 0},
    ],
    'M': [
        {'shape': 'vline', 'params': (0.05, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.05, 0.0, 0.5, 0.5), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 0.5, 0.95, 0.0), 'group': 0},
        {'shape': 'vline', 'params': (0.95, 0.0, 1.0), 'group': 0},
    ],
    'N': [
        {'shape': 'vline', 'params': (0.1, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.1, 0.0, 0.9, 1.0), 'group': 0},
        {'shape': 'vline', 'params': (0.9, 0.0, 1.0), 'group': 0},
    ],
    'O': [
        {'shape': 'loop', 'params': (0.5, 0.5, 0.45, 0.48)},
    ],
    'P': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.25, 0.45, 0.25, -90, 90), 'group': 0},
    ],
    'Q': [
        {'shape': 'loop', 'params': (0.5, 0.45, 0.45, 0.45)},
        {'shape': 'diag', 'params': (0.5, 0.7, 0.95, 1.0)},
    ],
    'R': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.25, 0.45, 0.25, -90, 90), 'group': 0},
        {'shape': 'diag', 'params': (0.4, 0.5, 1.0, 1.0)},
    ],
    'S': [
        {'shape': 'arc_left', 'params': (0.6, 0.28, 0.4, 0.22, -90, 90),
         'bounds': [None, (0.12, 0.32), (0.15, 0.6), (0.05, 0.18), (-100, -80), (80, 100)], 'group': 0},
        {'shape': 'arc_right', 'params': (0.4, 0.72, 0.4, 0.22, -90, 90),
         'bounds': [None, (0.68, 0.88), (0.15, 0.6), (0.05, 0.18), (-100, -80), (80, 100)], 'group': 0},
    ],
    'T': [
        {'shape': 'hline', 'params': (0.0, 0.0, 1.0)},
        {'shape': 'vline', 'params': (0.5, 0.0, 1.0)},
    ],
    'U': [
        {'shape': 'vline', 'params': (0.15, 0.0, 0.65), 'group': 0},
        {'shape': 'u_arc', 'params': (0.5, 0.65, 0.35, 0.35), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.0, 0.65), 'group': 0},
    ],
    'V': [
        {'shape': 'diag', 'params': (0.0, 0.0, 0.5, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 1.0, 1.0, 0.0), 'group': 0},
    ],
    'W': [
        {'shape': 'diag', 'params': (0.0, 0.0, 0.25, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.25, 1.0, 0.5, 0.4), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 0.4, 0.75, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.75, 1.0, 1.0, 0.0), 'group': 0},
    ],
    'X': [
        {'shape': 'diag', 'params': (0.0, 0.0, 1.0, 1.0)},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.0, 1.0)},
    ],
    'Y': [
        {'shape': 'diag', 'params': (0.0, 0.0, 0.5, 0.5), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.5, 0.5), 'group': 0},
        {'shape': 'vline', 'params': (0.5, 0.5, 1.0), 'group': 0},
    ],
    'Z': [
        {'shape': 'hline', 'params': (0.0, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (1.0, 0.0, 1.0), 'group': 0},
    ],

    # --- Lowercase ---
    'a': [
        {'shape': 'arc_left', 'params': (0.7, 0.5, 0.4, 0.45, -90, 90), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.1, 1.0), 'group': 0},
    ],
    'b': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.6, 0.4, 0.4, -90, 90), 'group': 0},
    ],
    'c': [
        {'shape': 'arc_left', 'params': (0.8, 0.5, 0.45, 0.48, -90, 90)},
    ],
    'd': [
        {'shape': 'arc_left', 'params': (0.7, 0.6, 0.4, 0.4, -90, 90), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.0, 1.0), 'group': 0},
    ],
    'e': [
        {'shape': 'hline', 'params': (0.45, 0.15, 0.85), 'group': 0},
        {'shape': 'arc_left', 'params': (0.7, 0.55, 0.4, 0.4, 0, 90), 'group': 0},
    ],
    'f': [
        {'shape': 'vline', 'params': (0.4, 0.15, 1.0), 'group': 0},
        {'shape': 'arc_left', 'params': (0.7, 0.15, 0.3, 0.15, -90, 0), 'group': 0},
        {'shape': 'hline', 'params': (0.35, 0.15, 0.7)},
    ],
    'g': [
        {'shape': 'arc_left', 'params': (0.7, 0.35, 0.4, 0.35, -90, 90), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.1, 0.85), 'group': 0},
        {'shape': 'u_arc', 'params': (0.5, 0.85, 0.35, 0.15), 'group': 0},
    ],
    'h': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.4, 0.35, 0.2, -90, 0), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.4, 1.0), 'group': 0},
    ],
    'i': [
        {'shape': 'vline', 'params': (0.5, 0.25, 1.0)},
    ],
    'j': [
        {'shape': 'vline', 'params': (0.5, 0.25, 0.8), 'group': 0},
        {'shape': 'u_arc', 'params': (0.3, 0.8, 0.2, 0.2), 'group': 0},
    ],
    'k': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'diag', 'params': (0.85, 0.25, 0.15, 0.55), 'group': 1},
        {'shape': 'diag', 'params': (0.15, 0.55, 0.85, 1.0), 'group': 1},
    ],
    'l': [
        {'shape': 'vline', 'params': (0.5, 0.0, 1.0)},
    ],
    'm': [
        {'shape': 'vline', 'params': (0.08, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.08, 0.35, 0.22, 0.18, -90, 0), 'group': 0},
        {'shape': 'vline', 'params': (0.5, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.5, 0.35, 0.22, 0.18, -90, 0), 'group': 0},
        {'shape': 'vline', 'params': (0.92, 0.2, 1.0), 'group': 0},
    ],
    'n': [
        {'shape': 'vline', 'params': (0.15, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.4, 0.35, 0.22, -90, 0), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.2, 1.0), 'group': 0},
    ],
    'o': [
        {'shape': 'loop', 'params': (0.5, 0.55, 0.4, 0.42)},
    ],
    'p': [
        {'shape': 'vline', 'params': (0.15, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.35, 0.4, 0.3, -90, 90), 'group': 0},
    ],
    'q': [
        {'shape': 'arc_left', 'params': (0.7, 0.35, 0.4, 0.3, -90, 90), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.2, 1.0), 'group': 0},
    ],
    'r': [
        {'shape': 'vline', 'params': (0.2, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.2, 0.35, 0.3, 0.18, -90, 0), 'group': 0},
    ],
    's': [
        {'shape': 'arc_left', 'params': (0.6, 0.32, 0.35, 0.22, -90, 90),
         'bounds': [None, (0.05, 0.45), None, None, None, None], 'group': 0},
        {'shape': 'arc_right', 'params': (0.4, 0.68, 0.35, 0.22, -90, 90),
         'bounds': [None, (0.55, 0.95), None, None, None, None], 'group': 0},
    ],
    't': [
        {'shape': 'vline', 'params': (0.4, 0.0, 1.0)},
        {'shape': 'hline', 'params': (0.3, 0.1, 0.75)},
    ],
    'u': [
        {'shape': 'vline', 'params': (0.15, 0.2, 0.65), 'group': 0},
        {'shape': 'u_arc', 'params': (0.5, 0.65, 0.35, 0.35), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.2, 1.0), 'group': 0},
    ],
    'v': [
        {'shape': 'diag', 'params': (0.0, 0.2, 0.5, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 1.0, 1.0, 0.2), 'group': 0},
    ],
    'w': [
        {'shape': 'diag', 'params': (0.0, 0.2, 0.25, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.25, 1.0, 0.5, 0.5), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 0.5, 0.75, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.75, 1.0, 1.0, 0.2), 'group': 0},
    ],
    'x': [
        {'shape': 'diag', 'params': (0.0, 0.2, 1.0, 1.0)},
        {'shape': 'diag', 'params': (1.0, 0.2, 0.0, 1.0)},
    ],
    'y': [
        {'shape': 'diag', 'params': (0.0, 0.2, 0.5, 0.6), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.2, 0.15, 1.0), 'group': 0},
    ],
    'z': [
        {'shape': 'hline', 'params': (0.2, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.2, 0.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (1.0, 0.0, 1.0), 'group': 0},
    ],

    # --- Digits ---
    '0': [
        {'shape': 'loop', 'params': (0.5, 0.5, 0.42, 0.48)},
    ],
    '1': [
        {'shape': 'vline', 'params': (0.5, 0.0, 1.0)},
    ],
    '2': [
        {'shape': 'arc_left', 'params': (0.6, 0.25, 0.4, 0.25, -90, 45), 'group': 0},
        {'shape': 'diag', 'params': (0.7, 0.4, 0.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (1.0, 0.0, 1.0), 'group': 0},
    ],
    '3': [
        {'shape': 'arc_right', 'params': (0.35, 0.27, 0.4, 0.27, -90, 90),
         'bounds': [None, (0.05, 0.45), None, None, None, None], 'group': 0},
        {'shape': 'arc_right', 'params': (0.35, 0.73, 0.4, 0.27, -90, 90),
         'bounds': [None, (0.55, 0.95), None, None, None, None], 'group': 0},
    ],
    '4': [
        {'shape': 'diag', 'params': (0.0, 0.0, 0.0, 0.6), 'group': 0},
        {'shape': 'hline', 'params': (0.6, 0.0, 0.85), 'group': 0},
        {'shape': 'vline', 'params': (0.7, 0.0, 1.0)},
    ],
    '5': [
        {'shape': 'hline', 'params': (0.0, 0.0, 0.9), 'group': 0},
        {'shape': 'vline', 'params': (0.1, 0.0, 0.45), 'group': 0},
        {'shape': 'arc_right', 'params': (0.2, 0.7, 0.45, 0.3, -90, 90), 'group': 0},
    ],
    '6': [
        {'shape': 'arc_left', 'params': (0.75, 0.3, 0.4, 0.35, -90, 60), 'group': 0},
        {'shape': 'loop', 'params': (0.5, 0.65, 0.38, 0.32), 'group': 0},
    ],
    '7': [
        {'shape': 'hline', 'params': (0.0, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.25, 1.0), 'group': 0},
    ],
    '8': [
        {'shape': 'loop', 'params': (0.5, 0.27, 0.32, 0.25),
         'bounds': [None, (0.1, 0.45), None, None], 'group': 0},
        {'shape': 'loop', 'params': (0.5, 0.73, 0.38, 0.27),
         'bounds': [None, (0.55, 0.9), None, None], 'group': 0},
    ],
    '9': [
        {'shape': 'loop', 'params': (0.5, 0.35, 0.38, 0.32), 'group': 0},
        {'shape': 'arc_right', 'params': (0.25, 0.7, 0.4, 0.35, -60, 90), 'group': 0},
    ],
}
