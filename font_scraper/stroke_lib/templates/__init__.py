"""Stroke templates and matching.

This module provides numpad-based stroke templates for defining character
shapes and a repository system for managing template collections. Templates
use a numeric keypad notation to specify waypoints, making it easy to
define stroke patterns.

The numpad notation uses positions 1-9 corresponding to a 3x3 grid:
    7 8 9  (top row)
    4 5 6  (middle row)
    1 2 3  (bottom row)

The module exports:
    NumpadTemplate: Template class using numpad notation for waypoints.
    NUMPAD_POS: Dictionary mapping region numbers to fractional coordinates.
    TemplateRepository: Collection manager for storing and retrieving templates.

Example usage:
    Creating a simple template::

        from stroke_lib.templates import NumpadTemplate

        # Letter "L" - vertical stroke from top to bottom, then horizontal
        template = NumpadTemplate(
            name='L',
            strokes=[[7, 1, 3]]  # Top-left to bottom-left to bottom-right
        )

        # Get waypoint info
        waypoints = template.get_waypoints(0)
        for wp in waypoints:
            print(f"Region {wp.region}, vertex={wp.is_vertex}")

    Using the repository::

        from stroke_lib.templates import TemplateRepository

        repo = TemplateRepository()
        repo.register('A', template_a)
        repo.register_variant('A', 'serif', template_a_serif)

        # Retrieve templates
        default_a = repo.get('A')
        serif_a = repo.get_variant('A', 'serif')
"""

from .numpad import NUMPAD_POS, NumpadTemplate
from .repository import TemplateRepository

__all__ = ['NumpadTemplate', 'NUMPAD_POS', 'TemplateRepository']
