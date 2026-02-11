#!/usr/bin/env python3
"""Run all font scrapers in parallel or sequentially.

This module provides a unified interface for running all font scrapers
(DaFont, FontSpace, and Google Fonts) either in parallel using
ProcessPoolExecutor or sequentially. It aggregates results from all
sources and provides a summary of downloaded fonts.

Each scraper runs as a subprocess, allowing for true parallel execution
and isolation between scrapers. Results are collected and a summary
JSON file is saved to the output directory.

Example:
    Run all scrapers in parallel with default settings::

        $ python scrape_all.py --output ./all_fonts --pages 20

    Run with font limits per source::

        $ python scrape_all.py --output ./all_fonts --pages 50 --max-per-source 200

    Run sequentially (useful for debugging)::

        $ python scrape_all.py --output ./all_fonts --sequential

Output Structure:
    The output directory will contain subdirectories for each source::

        all_fonts/
            dafont/           # Fonts from DaFont.com
            fontspace/        # Fonts from FontSpace.com
            google/           # Fonts from Google Fonts
            summary.json      # Aggregated results summary

Command-line Arguments:
    --output, -o: Base output directory for all scrapers (default: ./all_fonts)
    --pages, -p: Maximum pages to scrape for DaFont and FontSpace (default: 20)
    --max-per-source, -m: Maximum fonts to download per source (default: unlimited)
    --sequential, -s: Run scrapers sequentially instead of in parallel
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Default scraper configuration constants
DAFONT_CATEGORIES = ['601', '603']  # Calligraphy + Handwritten
FONTSPACE_QUERY = 'handwritten'


def run_dafont(output_dir: str, pages: int, max_fonts: int = None) -> dict:
    """Run the DaFont scraper as a subprocess.

    Executes dafont_scraper.py with the specified parameters, downloading
    fonts from the Calligraphy (601) and Handwritten (603) categories.

    Args:
        output_dir: Base output directory. Fonts will be saved to
            a 'dafont' subdirectory within this path.
        pages: Maximum number of pages to scrape per category.
        max_fonts: Maximum number of fonts to download. If None, no limit
            is applied.

    Returns:
        A dictionary containing:
            - source (str): Always "dafont".
            - output (str): stdout from the scraper if successful.
            - error (str): stderr from the scraper if it failed.
    """
    cmd = [
        sys.executable, 'dafont_scraper.py',
        '--output', str(Path(output_dir) / 'dafont'),
        '--pages', str(pages),
        '--categories', *DAFONT_CATEGORIES,
    ]
    if max_fonts:
        cmd.extend(['--max-fonts', str(max_fonts)])

    print(f"[DaFont] Starting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[DaFont] Error: {result.stderr}")
        return {'source': 'dafont', 'error': result.stderr}

    print("[DaFont] Complete")
    return {'source': 'dafont', 'output': result.stdout}


def run_fontspace(output_dir: str, pages: int, max_fonts: int = None) -> dict:
    """Run the FontSpace scraper as a subprocess.

    Executes fontspace_scraper.py with a search query for "handwritten" fonts.

    Args:
        output_dir: Base output directory. Fonts will be saved to
            a 'fontspace' subdirectory within this path.
        pages: Maximum number of search result pages to scrape.
        max_fonts: Maximum number of fonts to download. If None, no limit
            is applied.

    Returns:
        A dictionary containing:
            - source (str): Always "fontspace".
            - output (str): stdout from the scraper if successful.
            - error (str): stderr from the scraper if it failed.
    """
    cmd = [
        sys.executable, 'fontspace_scraper.py',
        '--output', str(Path(output_dir) / 'fontspace'),
        '--query', FONTSPACE_QUERY,
        '--pages', str(pages),
    ]
    if max_fonts:
        cmd.extend(['--max-fonts', str(max_fonts)])

    print(f"[FontSpace] Starting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[FontSpace] Error: {result.stderr}")
        return {'source': 'fontspace', 'error': result.stderr}

    print("[FontSpace] Complete")
    return {'source': 'fontspace', 'output': result.stdout}


def run_google(output_dir: str, max_fonts: int = None) -> dict:
    """Run the Google Fonts scraper as a subprocess.

    Executes google_fonts_scraper.py to download handwriting fonts from
    the curated list of Google Fonts.

    Args:
        output_dir: Base output directory. Fonts will be saved to
            a 'google' subdirectory within this path.
        max_fonts: Maximum number of fonts to download. If None, downloads
            all fonts in the curated handwriting fonts list.

    Returns:
        A dictionary containing:
            - source (str): Always "google".
            - output (str): stdout from the scraper if successful.
            - error (str): stderr from the scraper if it failed.
    """
    cmd = [
        sys.executable, 'google_fonts_scraper.py',
        '--output', str(Path(output_dir) / 'google'),
    ]
    if max_fonts:
        cmd.extend(['--max-fonts', str(max_fonts)])

    print(f"[Google] Starting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[Google] Error: {result.stderr}")
        return {'source': 'google', 'error': result.stderr}

    print("[Google] Complete")
    return {'source': 'google', 'output': result.stdout}


def main():
    """Parse command-line arguments and run all font scrapers.

    This function serves as the entry point when the module is run as a script.
    It configures argument parsing, runs all scrapers either in parallel or
    sequentially based on the --sequential flag, and saves a summary of results.

    The function creates the output directory structure, executes each scraper,
    counts the total downloaded fonts, and saves a summary.json file with
    aggregated statistics.
    """
    parser = argparse.ArgumentParser(description='Run all font scrapers in parallel')
    parser.add_argument('--output', '-o', type=str, default='./all_fonts',
                        help='Base output directory')
    parser.add_argument('--pages', '-p', type=int, default=20,
                        help='Max pages for DaFont/FontSpace')
    parser.add_argument('--max-per-source', '-m', type=int, default=None,
                        help='Max fonts per source')
    parser.add_argument('--sequential', '-s', action='store_true',
                        help='Run sequentially instead of parallel')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Font Scraper - All Sources")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Pages: {args.pages}")
    print(f"Max per source: {args.max_per_source or 'unlimited'}")
    print(f"Mode: {'sequential' if args.sequential else 'parallel'}")
    print("=" * 60)

    start_time = time.time()

    if args.sequential:
        # Run one at a time
        results = []
        results.append(run_dafont(str(output_dir), args.pages, args.max_per_source))
        results.append(run_fontspace(str(output_dir), args.pages, args.max_per_source))
        results.append(run_google(str(output_dir), args.max_per_source))
    else:
        # Run in parallel
        results = []
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(run_dafont, str(output_dir), args.pages, args.max_per_source): 'dafont',
                executor.submit(run_fontspace, str(output_dir), args.pages, args.max_per_source): 'fontspace',
                executor.submit(run_google, str(output_dir), args.max_per_source): 'google',
            }

            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"[{source}] Exception: {e}")
                    results.append({'source': source, 'error': str(e)})

    elapsed = time.time() - start_time

    # Count total fonts
    total_fonts = 0
    for subdir in ['dafont', 'fontspace', 'google']:
        subpath = output_dir / subdir
        if subpath.exists():
            fonts = list(subpath.glob('*.ttf')) + list(subpath.glob('*.otf')) + list(subpath.glob('*.woff2'))
            count = len(fonts)
            total_fonts += count
            print(f"  {subdir}: {count} fonts")

    print("\n" + "=" * 60)
    print("ALL COMPLETE")
    print("=" * 60)
    print(f"Total fonts: {total_fonts}")
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Output: {output_dir}")

    # Save summary
    summary = {
        'total_fonts': total_fonts,
        'elapsed_seconds': elapsed,
        'results': results
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
