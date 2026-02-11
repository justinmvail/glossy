"""Template repository for accessing stroke templates.

This module provides the TemplateRepository class for managing collections
of stroke templates. The repository supports default templates for
characters as well as named variants (e.g., serif vs sans-serif versions).

The repository pattern provides:
    - Central storage for all character templates
    - Support for multiple variants per character
    - Factory method for bulk loading from dictionaries
    - Enumeration of available characters

Example usage:
    Basic repository operations::

        from stroke_lib.templates.repository import TemplateRepository
        from stroke_lib.templates.numpad import NumpadTemplate

        repo = TemplateRepository()

        # Register a default template
        template = NumpadTemplate(name='A', strokes=[[7, 'v(8)', 9], [4, 6]])
        repo.register('A', template)

        # Register a variant
        serif_template = NumpadTemplate(name='A_serif', strokes=[[7, 'v(8)', 9], [4, 6], [1, 7], [3, 9]])
        repo.register_variant('A', 'serif', serif_template)

        # Retrieve templates
        default_a = repo.get('A')
        serif_a = repo.get_variant('A', 'serif')

    Bulk loading from dictionaries::

        templates = {
            'A': [[7, 'v(8)', 9], [4, 6]],
            'T': [[7, 9], [8, 2]],
        }
        variants = {
            'A': {'serif': [[7, 'v(8)', 9], [4, 6], [1, 7]]},
        }

        repo = TemplateRepository.from_dict(templates, variants)
"""

from __future__ import annotations

from .numpad import NumpadTemplate


class TemplateRepository:
    """Repository for stroke templates.

    Provides access to predefined templates for characters and allows
    registration of custom templates. Supports both default templates
    and named variants for each character.

    The repository maintains two collections:
        - Default templates: One per character, used when no variant
          is specified
        - Variant templates: Multiple per character, accessed by name

    Attributes:
        _templates: Internal dictionary mapping characters to default
            templates.
        _variants: Internal nested dictionary mapping characters to
            variant name to template.

    Example:
        >>> repo = TemplateRepository()
        >>> repo.register('A', template_a)
        >>> repo.register_variant('A', 'bold', template_a_bold)
        >>> repo.list_characters()
        ['A']
    """

    def __init__(self):
        """Initialize an empty template repository.

        Creates empty dictionaries for default templates and variants.
        """
        self._templates: dict[str, NumpadTemplate] = {}
        self._variants: dict[str, dict[str, NumpadTemplate]] = {}

    def register(self, char: str, template: NumpadTemplate) -> None:
        """Register a template for a character.

        Sets the default template for the specified character. If a
        template was previously registered, it will be replaced.

        Args:
            char: Character to register the template for (e.g., 'A').
            template: NumpadTemplate to use as the default for this
                character.
        """
        self._templates[char] = template

    def register_variant(self, char: str, variant_name: str, template: NumpadTemplate) -> None:
        """Register a variant template for a character.

        Adds a named variant for the specified character. Characters
        can have multiple variants (e.g., 'serif', 'bold', 'italic').

        Args:
            char: Character to register the variant for.
            variant_name: Name of this variant (e.g., 'serif').
            template: NumpadTemplate for this variant.
        """
        if char not in self._variants:
            self._variants[char] = {}
        self._variants[char][variant_name] = template

    def get(self, char: str) -> NumpadTemplate | None:
        """Get the default template for a character.

        Args:
            char: Character to look up.

        Returns:
            Default NumpadTemplate for the character, or None if not
            registered.
        """
        return self._templates.get(char)

    def get_variant(self, char: str, variant_name: str) -> NumpadTemplate | None:
        """Get a specific variant template.

        Args:
            char: Character to look up.
            variant_name: Name of the variant to retrieve.

        Returns:
            NumpadTemplate for the specified variant, or None if not
            registered.
        """
        variants = self._variants.get(char, {})
        return variants.get(variant_name)

    def get_all_variants(self, char: str) -> dict[str, NumpadTemplate]:
        """Get all variants for a character.

        Args:
            char: Character to look up variants for.

        Returns:
            Dictionary mapping variant names to templates. Empty dict
            if no variants are registered.
        """
        return self._variants.get(char, {})

    def has_variants(self, char: str) -> bool:
        """Check if character has multiple variants.

        Args:
            char: Character to check.

        Returns:
            True if the character has more than one registered variant.
        """
        return char in self._variants and len(self._variants[char]) > 1

    def list_characters(self) -> list[str]:
        """List all characters with templates.

        Returns:
            Sorted list of characters that have either a default
            template or variants registered.
        """
        chars = set(self._templates.keys())
        chars.update(self._variants.keys())
        return sorted(chars)

    @classmethod
    def from_dict(
        cls,
        templates: dict[str, list],
        variants: dict[str, dict[str, list]] | None = None
    ) -> TemplateRepository:
        """Create repository from dictionary definitions.

        Factory method to create a fully populated repository from
        dictionary definitions. Useful for loading templates from
        configuration files or static definitions.

        Args:
            templates: Dictionary mapping character -> list of strokes.
                Each stroke list should contain waypoint definitions.
            variants: Optional dictionary mapping character ->
                {variant_name -> strokes}. Allows defining multiple
                variants per character.

        Returns:
            Populated TemplateRepository with all templates registered.

        Example:
            >>> templates = {'A': [[7, 'v(8)', 9]], 'I': [[8, 2]]}
            >>> repo = TemplateRepository.from_dict(templates)
            >>> repo.get('A').name
            'A_default'
        """
        repo = cls()

        for char, strokes in templates.items():
            template = NumpadTemplate(name=f'{char}_default', strokes=strokes)
            repo.register(char, template)

        if variants:
            for char, char_variants in variants.items():
                for var_name, strokes in char_variants.items():
                    template = NumpadTemplate(name=f'{char}_{var_name}', strokes=strokes)
                    repo.register_variant(char, var_name, template)

        return repo
