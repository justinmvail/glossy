"""Template repository for accessing stroke templates."""

from __future__ import annotations
from typing import Dict, List, Optional, Union

from .numpad import NumpadTemplate


class TemplateRepository:
    """Repository for stroke templates.

    Provides access to predefined templates for characters and allows
    registration of custom templates.
    """

    def __init__(self):
        self._templates: Dict[str, NumpadTemplate] = {}
        self._variants: Dict[str, Dict[str, NumpadTemplate]] = {}

    def register(self, char: str, template: NumpadTemplate) -> None:
        """Register a template for a character."""
        self._templates[char] = template

    def register_variant(self, char: str, variant_name: str, template: NumpadTemplate) -> None:
        """Register a variant template for a character."""
        if char not in self._variants:
            self._variants[char] = {}
        self._variants[char][variant_name] = template

    def get(self, char: str) -> Optional[NumpadTemplate]:
        """Get the default template for a character."""
        return self._templates.get(char)

    def get_variant(self, char: str, variant_name: str) -> Optional[NumpadTemplate]:
        """Get a specific variant template."""
        variants = self._variants.get(char, {})
        return variants.get(variant_name)

    def get_all_variants(self, char: str) -> Dict[str, NumpadTemplate]:
        """Get all variants for a character."""
        return self._variants.get(char, {})

    def has_variants(self, char: str) -> bool:
        """Check if character has multiple variants."""
        return char in self._variants and len(self._variants[char]) > 1

    def list_characters(self) -> List[str]:
        """List all characters with templates."""
        chars = set(self._templates.keys())
        chars.update(self._variants.keys())
        return sorted(chars)

    @classmethod
    def from_dict(
        cls,
        templates: Dict[str, List],
        variants: Optional[Dict[str, Dict[str, List]]] = None
    ) -> 'TemplateRepository':
        """Create repository from dictionary definitions.

        Args:
            templates: Dict mapping char -> list of strokes
            variants: Optional dict mapping char -> {variant_name -> strokes}

        Returns:
            Populated TemplateRepository
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
