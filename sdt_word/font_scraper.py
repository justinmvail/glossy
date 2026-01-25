"""
Font scraper and scorer for building handwriting training dataset.
Downloads fonts from various sources, scores them, and finds duplicates.
"""

import os
import re
import json
import hashlib
import requests
import zipfile
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imagehash
from collections import defaultdict
from tqdm import tqdm
import time


class FontScraper:
    """Download fonts from various sources."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded = []

    def scrape_google_fonts(self, categories=None):
        """
        Download handwriting fonts from Google Fonts.
        Categories: handwriting, display, serif, sans-serif, monospace
        """
        if categories is None:
            categories = ['handwriting']

        api_url = "https://www.googleapis.com/webfonts/v1/webfonts"
        # Note: Requires API key for full access, but we can scrape the CSS

        fonts_downloaded = []

        for category in categories:
            print(f"Fetching Google Fonts category: {category}")

            # Use the developer API (free, no key needed for list)
            try:
                # Get font list from Google Fonts CSS API
                css_url = f"https://fonts.googleapis.com/css2?family="

                # Known handwriting fonts on Google Fonts
                handwriting_fonts = [
                    "Caveat", "Dancing+Script", "Pacifico", "Satisfy", "Great+Vibes",
                    "Kalam", "Indie+Flower", "Shadows+Into+Light", "Amatic+SC",
                    "Permanent+Marker", "Architects+Daughter", "Patrick+Hand",
                    "Handlee", "Gochi+Hand", "Rock+Salt", "Reenie+Beanie",
                    "Just+Another+Hand", "Covered+By+Your+Grace", "Coming+Soon",
                    "Schoolbell", "Short+Stack", "Rancho", "Sue+Ellen+Francisco",
                    "Loved+by+the+King", "La+Belle+Aurore", "Give+You+Glory",
                    "Cedarville+Cursive", "Dawning+of+a+New+Day", "Over+the+Rainbow",
                    "Waiting+for+the+Sunrise", "Zeyada", "Mrs+Saint+Delafield",
                    "Homemade+Apple", "Crafty+Girls", "Annie+Use+Your+Telescope",
                    "The+Girl+Next+Door", "Calligraffitti", "Just+Me+Again+Down+Here",
                    "Swanky+and+Moo+Moo", "Sunshiney", "Walter+Turncoat",
                    "Fontdiner+Swanky", "Kranky", "Cherry+Cream+Soda",
                    "Gloria+Hallelujah", "Nothing+You+Could+Do", "Sedgwick+Ave",
                    "Mali", "Sriracha", "Itim", "Charm", "Charmonman",
                ]

                for font_name in tqdm(handwriting_fonts, desc="Google Fonts"):
                    try:
                        self._download_google_font(font_name)
                        fonts_downloaded.append(font_name.replace("+", " "))
                        time.sleep(0.5)  # Rate limit
                    except Exception as e:
                        print(f"  Failed {font_name}: {e}")

            except Exception as e:
                print(f"Error fetching Google Fonts: {e}")

        return fonts_downloaded

    def _download_google_font(self, font_name):
        """Download a single font from Google Fonts."""
        # Get the CSS which contains font URLs
        css_url = f"https://fonts.googleapis.com/css2?family={font_name}&display=swap"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        resp = requests.get(css_url, headers=headers)
        resp.raise_for_status()

        # Extract TTF/WOFF2 URLs from CSS
        urls = re.findall(r'url\((https://fonts\.gstatic\.com/[^)]+)\)', resp.text)

        if urls:
            # Download the first font file
            font_url = urls[0]
            font_resp = requests.get(font_url)
            font_resp.raise_for_status()

            # Determine extension
            ext = '.woff2' if 'woff2' in font_url else '.ttf'

            # Save font
            font_path = self.output_dir / f"google_{font_name.replace('+', '_')}{ext}"
            font_path.write_bytes(font_resp.content)
            self.downloaded.append(str(font_path))

    def scrape_fontspace(self, category='handwriting', pages=10):
        """
        Scrape fonts from FontSpace.
        Note: Respect robots.txt and rate limits.
        """
        fonts_downloaded = []
        base_url = f"https://www.fontspace.com/category/{category}"

        print(f"Scraping FontSpace category: {category}")

        # This would need proper web scraping with BeautifulSoup
        # For now, return manual list of known good fonts
        print("  FontSpace requires manual download or proper scraping setup")

        return fonts_downloaded

    def scrape_dafont(self, category='handwritten', pages=10):
        """
        Scrape fonts from DaFont.
        Note: Check licenses - many are free for personal use only.
        """
        fonts_downloaded = []
        print(f"Scraping DaFont category: {category}")
        print("  DaFont requires manual download - check licenses!")

        return fonts_downloaded

    def add_local_fonts(self, font_dir):
        """Add fonts from a local directory."""
        font_dir = Path(font_dir)
        extensions = ['.ttf', '.otf', '.woff', '.woff2']

        fonts_added = []
        for ext in extensions:
            for font_path in font_dir.rglob(f'*{ext}'):
                dest = self.output_dir / font_path.name
                if not dest.exists():
                    import shutil
                    shutil.copy(font_path, dest)
                    self.downloaded.append(str(dest))
                    fonts_added.append(font_path.name)

        print(f"Added {len(fonts_added)} local fonts")
        return fonts_added


class FontScorer:
    """Score fonts for suitability as handwriting training data."""

    def __init__(self):
        self.scores = {}

    def score_font(self, font_path, render_size=64):
        """
        Score a font on multiple criteria.
        Returns dict with scores and overall rating.
        """
        font_path = Path(font_path)
        scores = {
            'path': str(font_path),
            'name': font_path.stem,
            'charset_coverage': 0,
            'style_score': 0,
            'legibility_score': 0,
            'uniqueness_score': 0,
            'overall': 0,
        }

        try:
            font = ImageFont.truetype(str(font_path), render_size)
        except Exception as e:
            scores['error'] = str(e)
            return scores

        # 1. Character set coverage
        required_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        covered = 0
        for char in required_chars:
            try:
                img = Image.new('L', (render_size, render_size), 255)
                draw = ImageDraw.Draw(img)
                draw.text((5, 5), char, font=font, fill=0)
                # Check if anything was drawn
                if img.getextrema() != (255, 255):
                    covered += 1
            except:
                pass
        scores['charset_coverage'] = covered / len(required_chars)

        # 2. Style score (is it handwriting-like?)
        # Render sample text and analyze
        try:
            sample_text = "Hello"
            img = Image.new('L', (render_size * 5, render_size * 2), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), sample_text, font=font, fill=0)

            # Handwriting fonts tend to have more variation/curves
            # This is a simple heuristic - could use ML for better scoring
            import numpy as np
            arr = np.array(img)

            # Check for non-uniform stroke widths (handwriting characteristic)
            # More sophisticated analysis would use edge detection
            dark_pixels = np.sum(arr < 128)
            total_pixels = arr.size
            ink_ratio = dark_pixels / total_pixels

            # Handwriting typically has 5-20% ink coverage for sample text
            if 0.03 < ink_ratio < 0.25:
                scores['style_score'] = 0.8
            elif 0.01 < ink_ratio < 0.35:
                scores['style_score'] = 0.5
            else:
                scores['style_score'] = 0.2

        except Exception as e:
            scores['style_score'] = 0

        # 3. Legibility (placeholder - would use TrOCR)
        scores['legibility_score'] = 0.7  # Default, needs TrOCR integration

        # 4. Compute perceptual hash for uniqueness checking
        try:
            img = Image.new('L', (256, 64), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "The quick brown", font=font, fill=0)
            scores['phash'] = str(imagehash.phash(img))
        except:
            scores['phash'] = None

        # Overall score
        scores['overall'] = (
            scores['charset_coverage'] * 0.4 +
            scores['style_score'] * 0.3 +
            scores['legibility_score'] * 0.3
        )

        self.scores[str(font_path)] = scores
        return scores

    def score_directory(self, font_dir):
        """Score all fonts in a directory."""
        font_dir = Path(font_dir)
        extensions = ['.ttf', '.otf', '.woff', '.woff2']

        all_fonts = []
        for ext in extensions:
            all_fonts.extend(font_dir.glob(f'*{ext}'))

        results = []
        for font_path in tqdm(all_fonts, desc="Scoring fonts"):
            score = self.score_font(font_path)
            results.append(score)

        return results


class FontDeduplicator:
    """Find and remove duplicate/similar fonts."""

    def __init__(self, threshold=5):
        """
        threshold: max hamming distance for perceptual hash to consider duplicates
        """
        self.threshold = threshold

    def find_duplicates(self, scores):
        """
        Find duplicate fonts based on perceptual hash.
        Returns list of duplicate groups.
        """
        # Group by similar phash
        hash_groups = defaultdict(list)

        for score in scores:
            if score.get('phash'):
                hash_groups[score['phash']].append(score)

        # Find similar hashes
        duplicates = []
        processed = set()

        hashes = list(hash_groups.keys())
        for i, h1 in enumerate(hashes):
            if h1 in processed:
                continue

            group = hash_groups[h1].copy()

            for j, h2 in enumerate(hashes[i+1:], i+1):
                if h2 in processed:
                    continue

                # Compare hashes
                try:
                    hash1 = imagehash.hex_to_hash(h1)
                    hash2 = imagehash.hex_to_hash(h2)
                    distance = hash1 - hash2

                    if distance <= self.threshold:
                        group.extend(hash_groups[h2])
                        processed.add(h2)
                except:
                    pass

            if len(group) > 1:
                duplicates.append(group)
            processed.add(h1)

        return duplicates

    def select_best_from_groups(self, duplicate_groups):
        """
        From each duplicate group, select the best font.
        Returns list of fonts to keep and list to remove.
        """
        keep = []
        remove = []

        for group in duplicate_groups:
            # Sort by overall score
            sorted_group = sorted(group, key=lambda x: x.get('overall', 0), reverse=True)
            keep.append(sorted_group[0])
            remove.extend(sorted_group[1:])

        return keep, remove


def build_font_catalog(output_dir='/home/server/glossy/sdt_word/fonts'):
    """
    Main function to build a curated font catalog.
    """
    output_dir = Path(output_dir)

    # 1. Scrape fonts
    scraper = FontScraper(output_dir / 'raw')

    print("=== Downloading fonts ===")
    scraper.scrape_google_fonts()

    # Add existing local fonts
    if Path('/home/server/svg-fonts/fonts').exists():
        print("\nAdding existing SVG fonts...")
        # These are already single-line, just catalog them

    # 2. Score fonts
    print("\n=== Scoring fonts ===")
    scorer = FontScorer()
    scores = scorer.score_directory(output_dir / 'raw')

    # 3. Find duplicates
    print("\n=== Finding duplicates ===")
    deduper = FontDeduplicator(threshold=8)
    duplicates = deduper.find_duplicates(scores)
    print(f"Found {len(duplicates)} duplicate groups")

    keep, remove = deduper.select_best_from_groups(duplicates)

    # 4. Filter by quality
    good_fonts = [s for s in scores if s.get('overall', 0) > 0.5 and s not in remove]

    print(f"\n=== Results ===")
    print(f"Total fonts: {len(scores)}")
    print(f"Duplicates removed: {len(remove)}")
    print(f"Good quality fonts: {len(good_fonts)}")

    # 5. Save catalog
    catalog = {
        'fonts': good_fonts,
        'duplicates': [[s['name'] for s in g] for g in duplicates],
        'removed': [s['name'] for s in remove],
    }

    with open(output_dir / 'catalog.json', 'w') as f:
        json.dump(catalog, f, indent=2)

    print(f"\nCatalog saved to {output_dir / 'catalog.json'}")

    return catalog


if __name__ == "__main__":
    catalog = build_font_catalog()

    print("\n=== Top 20 fonts by score ===")
    sorted_fonts = sorted(catalog['fonts'], key=lambda x: x.get('overall', 0), reverse=True)
    for font in sorted_fonts[:20]:
        print(f"  {font['name']}: {font['overall']:.2f}")
