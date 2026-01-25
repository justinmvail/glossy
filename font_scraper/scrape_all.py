#!/usr/bin/env python3
"""
Run all font scrapers in parallel.

Usage:
    python scrape_all.py --output ./all_fonts --pages 20
    python scrape_all.py --output ./all_fonts --pages 50 --max-per-source 200
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json


def run_dafont(output_dir: str, pages: int, max_fonts: int = None) -> dict:
    """Run DaFont scraper."""
    cmd = [
        sys.executable, 'dafont_scraper.py',
        '--output', str(Path(output_dir) / 'dafont'),
        '--pages', str(pages),
        '--categories', '601', '603',  # Calligraphy + Handwritten
    ]
    if max_fonts:
        cmd.extend(['--max-fonts', str(max_fonts)])

    print(f"[DaFont] Starting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[DaFont] Error: {result.stderr}")
        return {'source': 'dafont', 'error': result.stderr}

    print(f"[DaFont] Complete")
    return {'source': 'dafont', 'output': result.stdout}


def run_fontspace(output_dir: str, pages: int, max_fonts: int = None) -> dict:
    """Run FontSpace scraper."""
    cmd = [
        sys.executable, 'fontspace_scraper.py',
        '--output', str(Path(output_dir) / 'fontspace'),
        '--query', 'handwritten',
        '--pages', str(pages),
    ]
    if max_fonts:
        cmd.extend(['--max-fonts', str(max_fonts)])

    print(f"[FontSpace] Starting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[FontSpace] Error: {result.stderr}")
        return {'source': 'fontspace', 'error': result.stderr}

    print(f"[FontSpace] Complete")
    return {'source': 'fontspace', 'output': result.stdout}


def run_google(output_dir: str, max_fonts: int = None) -> dict:
    """Run Google Fonts scraper."""
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

    print(f"[Google] Complete")
    return {'source': 'google', 'output': result.stdout}


def main():
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
