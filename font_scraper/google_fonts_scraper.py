"""
Google Fonts Scraper - Download handwriting fonts from Google Fonts

Uses the CSS API to get font files (no API key needed).

Usage:
    python google_fonts_scraper.py --output ./fonts
    python google_fonts_scraper.py --output ./fonts --category handwriting
"""

import argparse
import re
import time
from pathlib import Path
from typing import List, Set, Dict
from dataclasses import dataclass
import requests
import json


@dataclass
class FontInfo:
    name: str
    family: str
    category: str
    variants: List[str]
    download_urls: Dict[str, str]  # variant -> url

    def to_dict(self):
        return {
            'name': self.name,
            'family': self.family,
            'category': self.category,
            'variants': self.variants,
            'download_urls': self.download_urls
        }


class GoogleFontsScraper:
    """Download handwriting fonts from Google Fonts."""

    # Known handwriting/script fonts on Google Fonts
    # This list can be expanded
    HANDWRITING_FONTS = [
        # Handwriting
        "Caveat", "Dancing Script", "Pacifico", "Satisfy", "Great Vibes",
        "Kalam", "Indie Flower", "Shadows Into Light", "Amatic SC",
        "Permanent Marker", "Architects Daughter", "Patrick Hand",
        "Handlee", "Gochi Hand", "Rock Salt", "Reenie Beanie",
        "Just Another Hand", "Covered By Your Grace", "Coming Soon",
        "Schoolbell", "Short Stack", "Rancho", "Sue Ellen Francisco",
        "Loved by the King", "La Belle Aurore", "Give You Glory",
        "Cedarville Cursive", "Dawning of a New Day", "Over the Rainbow",
        "Waiting for the Sunrise", "Zeyada", "Mrs Saint Delafield",
        "Homemade Apple", "Crafty Girls", "Annie Use Your Telescope",
        "The Girl Next Door", "Calligraffitti", "Just Me Again Down Here",
        "Swanky and Moo Moo", "Sunshiney", "Walter Turncoat",
        "Fontdiner Swanky", "Kranky", "Cherry Cream Soda",
        "Gloria Hallelujah", "Nothing You Could Do", "Sedgwick Ave",
        "Mali", "Sriracha", "Itim", "Charm", "Charmonman",
        # Script/Calligraphy
        "Alex Brush", "Allura", "Bilbo", "Condiment", "Cookie",
        "Courgette", "Damion", "Euphoria Script", "Felipa",
        "Grand Hotel", "Herr Von Muellerhoff", "Italianno",
        "Kaushan Script", "Lavishly Yours", "Leckerli One",
        "Lobster", "Lobster Two", "Lovers Quarrel", "Marck Script",
        "Meddon", "Meie Script", "Merienda", "Miama", "Monsieur La Doulaise",
        "Montez", "Mr Dafoe", "Mr De Haviland", "Ms Madi", "Niconne",
        "Norican", "Oleo Script", "Parisienne", "Petit Formal Script",
        "Pinyon Script", "Princess Sofia", "Qwigley", "Quintessential",
        "Ruge Boogie", "Ruthie", "Sacramento", "Sail", "Seaweed Script",
        "Shadows Into Light Two", "Sofia", "Stalemate", "Tangerine",
        "Yellowtail",
        # More handwriting styles
        "Bad Script", "Bilbo Swash Caps", "Clicker Script", "Dr Sugiyama",
        "Eagle Lake", "Engagement", "Fleur De Leah", "Fondamento",
        "Henny Penny", "Hurricane", "League Script", "Licorice",
        "Liu Jian Mao Cao", "Long Cang", "Ma Shan Zheng", "Mea Culpa",
        "Miss Fajardose", "Molle", "My Soul", "Nanum Brush Script",
        "Nanum Pen Script", "Oooh Baby", "Petemoss", "Playball",
        "Praise", "Pushster", "Redressed", "Road Rage", "Rochester",
        "Rock 3D", "Rouge Script", "Ruge Boogie", "Sassy Frass",
        "Send Flowers", "Snippet", "Square Peg", "Style Script",
        "Syne Tactile", "Tapestry", "Updock", "Vujahday Script",
        "Waterfall", "WindSong", "Whisper", "Zhi Mang Xing",
    ]

    def __init__(self, output_dir: str, rate_limit: float = 0.5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit

        self.session = requests.Session()
        self.session.headers.update({
            # Use a browser user-agent to get TTF instead of WOFF2
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; rv:60.0) Gecko/20100101 Firefox/60.0',
        })

        self.downloaded: Set[str] = set()
        self.failed: List[str] = []

    def get_font_url(self, family: str, variant: str = 'regular') -> str:
        """Get download URL for a font from Google Fonts CSS."""
        # URL-encode the family name
        family_encoded = family.replace(' ', '+')

        # Request CSS with specific variant
        if variant == 'regular':
            css_url = f"https://fonts.googleapis.com/css?family={family_encoded}"
        else:
            css_url = f"https://fonts.googleapis.com/css?family={family_encoded}:{variant}"

        try:
            resp = self.session.get(css_url, timeout=30)
            resp.raise_for_status()

            # Extract font URL from CSS
            # Format: src: url(https://fonts.gstatic.com/...)
            urls = re.findall(r'url\((https://fonts\.gstatic\.com/[^)]+)\)', resp.text)

            if urls:
                return urls[0]

        except Exception as e:
            pass

        return ''

    def download_font(self, family: str) -> bool:
        """Download a font family."""
        if family in self.downloaded:
            return True

        try:
            print(f"  Downloading: {family}")

            url = self.get_font_url(family)
            if not url:
                print(f"    Could not get download URL")
                self.failed.append(family)
                return False

            resp = self.session.get(url, timeout=60)
            resp.raise_for_status()

            # Determine extension from URL
            if 'woff2' in url:
                ext = '.woff2'
            elif 'woff' in url:
                ext = '.woff'
            elif 'ttf' in url:
                ext = '.ttf'
            else:
                ext = '.ttf'  # Default

            # Save font
            safe_name = family.replace(' ', '_')
            out_path = self.output_dir / f"google_{safe_name}{ext}"
            out_path.write_bytes(resp.content)

            print(f"    Saved: {out_path.name}")
            self.downloaded.add(family)
            return True

        except Exception as e:
            print(f"    Error: {e}")
            self.failed.append(family)
            return False

    def scrape_and_download(
        self,
        fonts: List[str] = None,
        max_fonts: int = None
    ) -> dict:
        """Download specified fonts or all known handwriting fonts."""
        print("=" * 60)
        print("Google Fonts Scraper")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")

        if fonts is None:
            fonts = self.HANDWRITING_FONTS.copy()

        print(f"Fonts to download: {len(fonts)}")

        if max_fonts:
            fonts = fonts[:max_fonts]
            print(f"Limiting to {max_fonts} fonts")

        # Download
        print(f"\nDownloading {len(fonts)} fonts...")

        for i, family in enumerate(fonts):
            print(f"[{i+1}/{len(fonts)}]", end='')
            self.download_font(family)
            time.sleep(self.rate_limit)

        # Save metadata
        metadata = {
            'source': 'google_fonts',
            'fonts_requested': len(fonts),
            'fonts_downloaded': len(self.downloaded),
            'fonts_failed': len(self.failed),
            'downloaded': list(self.downloaded),
            'failed': self.failed
        }

        meta_path = self.output_dir / 'google_fonts_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)
        print(f"Fonts downloaded: {len(self.downloaded)}")
        print(f"Failed: {len(self.failed)}")

        return metadata


def main():
    parser = argparse.ArgumentParser(description='Download fonts from Google Fonts')
    parser.add_argument('--output', '-o', type=str, default='./google_fonts',
                        help='Output directory')
    parser.add_argument('--max-fonts', '-m', type=int, default=None,
                        help='Max fonts to download')
    parser.add_argument('--rate-limit', '-r', type=float, default=0.5,
                        help='Seconds between requests')

    args = parser.parse_args()

    scraper = GoogleFontsScraper(args.output, rate_limit=args.rate_limit)
    scraper.scrape_and_download(max_fonts=args.max_fonts)


if __name__ == '__main__':
    main()
