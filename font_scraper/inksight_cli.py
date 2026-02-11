#!/usr/bin/env python3
"""Command-line interface for InkSight vectorization.

This module provides the CLI for converting fonts to vector strokes using
the InkSight model. Supports single word and full charset processing,
optional OCR validation, and various output formats.

Extracted from inksight_vectorizer.py to enable:
- Cleaner module organization
- Easier CLI customization
- Separation from core vectorization logic

Usage:
    python inksight_cli.py --font path/to/font.ttf --word "Hello" --show
    python inksight_cli.py --font path/to/font.ttf --charset --output output_dir
    python inksight_cli.py --font path/to/font.ttf --word "Hello" --validate

Or run via the main module:
    python -m inksight_cli --font path/to/font.ttf --word "Hello"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Convert fonts to vector strokes using InkSight'
    )
    parser.add_argument('--font', '-f', type=str, required=True,
                        help='Path to TTF/OTF font file')
    parser.add_argument('--word', '-w', type=str, default='Hello',
                        help='Word to vectorize (default: Hello)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for SVG/JSON')
    parser.add_argument('--model', '-m', type=str,
                        default='/home/server/inksight/model',
                        help='Path to InkSight model')
    parser.add_argument('--size', '-s', type=int, default=120,
                        help='Font size (default: 120)')
    parser.add_argument('--smooth', type=float, default=1.5,
                        help='Smoothing sigma (default: 1.5)')
    parser.add_argument('--extend', type=float, default=8.0,
                        help='Max extension distance (default: 8.0)')
    parser.add_argument('--show', action='store_true',
                        help='Show visualization')
    parser.add_argument('--charset', action='store_true',
                        help='Process full charset instead of word')
    parser.add_argument('--validate', '-v', action='store_true',
                        help='Validate results with TrOCR (filters bad results)')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='OCR validation threshold (default: 0.8)')
    return parser


def _process_charset_command(args, vectorizer: 'InkSightVectorizer') -> None:
    """Handle charset processing command.

    Args:
        args: Parsed command-line arguments.
        vectorizer: InkSightVectorizer instance.
    """
    from inksight_vectorizer import visualize
    from ocr_validator import OCRValidator

    print(f"Processing charset from {args.font}...")
    results = vectorizer.process_charset(
        args.font,
        font_size=args.size,
        smooth_sigma=args.smooth,
        max_extension=args.extend
    )
    print(f"Processed {len(results)} characters")

    # Validate with OCR if requested
    if args.validate:
        print("\nValidating with TrOCR...")
        validator = OCRValidator()
        valid_results = {}
        for char, result in results.items():
            passed, recognized, score = validator.validate(result, threshold=args.threshold)
            status = "PASS" if passed else "FAIL"
            print(f"  '{char}' -> '{recognized}' ({score:.1%}) [{status}]")
            if passed:
                valid_results[char] = result
        print(f"\nPassed: {len(valid_results)}/{len(results)} characters")
        results = valid_results

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual SVGs
        for char, result in results.items():
            safe_name = char if char.isalnum() else f"char_{ord(char):04x}"
            svg_path = output_dir / f"{safe_name}.svg"
            svg_path.write_text(vectorizer.to_svg(result))

        # Save combined JSON
        all_data = {char: vectorizer.to_json(r) for char, r in results.items()}
        json_path = output_dir / "strokes.json"
        json_path.write_text(json.dumps(all_data, indent=2))

        print(f"Saved to {output_dir}/")


def _process_word_command(args, vectorizer: 'InkSightVectorizer') -> None:
    """Handle single word processing command.

    Args:
        args: Parsed command-line arguments.
        vectorizer: InkSightVectorizer instance.
    """
    from inksight_vectorizer import visualize
    from ocr_validator import OCRValidator

    print(f"Processing '{args.word}' from {args.font}...")
    result = vectorizer.process(
        args.font,
        args.word,
        font_size=args.size,
        smooth_sigma=args.smooth,
        max_extension=args.extend
    )
    print(f"Got {len(result.strokes)} strokes")

    # Validate with OCR if requested
    if args.validate:
        validator = OCRValidator()
        passed, recognized, score = validator.validate(result, threshold=args.threshold)
        status = "PASS" if passed else "FAIL"
        print(f"OCR: '{recognized}' (similarity: {score:.1%}) [{status}]")
        if not passed:
            print("Result failed OCR validation - strokes may not be readable")

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save SVG
        svg_path = output_dir / f"{args.word}.svg"
        svg_path.write_text(vectorizer.to_svg(result))

        # Save JSON
        json_path = output_dir / f"{args.word}.json"
        json_path.write_text(json.dumps(vectorizer.to_json(result), indent=2))

        # Save visualization
        vis_path = output_dir / f"{args.word}.png"
        visualize(result, output_path=str(vis_path))

        print(f"Saved to {output_dir}/")

    if args.show:
        visualize(result, show=True)


def main() -> None:
    """Command-line interface for InkSight vectorization.

    Parses command-line arguments and runs vectorization on the specified
    font. Supports single word processing, full charset processing, and
    optional OCR validation.

    Usage:
        python inksight_cli.py --font path/to/font.ttf --word "Hello"
        python inksight_cli.py --font path/to/font.ttf --charset --output output_dir
        python inksight_cli.py --font path/to/font.ttf --word "Test" --validate --show
    """
    from inksight_vectorizer import InkSightVectorizer

    parser = _create_argument_parser()
    args = parser.parse_args()

    vectorizer = InkSightVectorizer(model_path=args.model)

    if args.charset:
        _process_charset_command(args, vectorizer)
    else:
        _process_word_command(args, vectorizer)


if __name__ == '__main__':
    main()
