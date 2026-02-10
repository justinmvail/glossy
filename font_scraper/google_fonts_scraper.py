"""Google Fonts Scraper - Download handwriting fonts from Google Fonts.

This module provides a downloader for handwriting and script fonts from
Google Fonts. It uses the CSS API to obtain font file URLs without requiring
an API key, making it simple to use.

The scraper maintains a curated list of known handwriting and script fonts
available on Google Fonts. It downloads font files directly from Google's
static font server (fonts.gstatic.com).

Note:
    The User-Agent header is set to request TTF format when available,
    as older browser signatures receive TTF instead of WOFF2.

Example:
    Basic usage to download all known handwriting fonts::

        $ python google_fonts_scraper.py --output ./fonts

    Download a limited number of fonts::

        $ python google_fonts_scraper.py --output ./fonts --max-fonts 50

    Programmatic usage::

        scraper = GoogleFontsScraper('./output', rate_limit=0.5)
        metadata = scraper.scrape_and_download(max_fonts=100)

        # Or download specific fonts
        scraper.scrape_and_download(fonts=['Caveat', 'Dancing Script', 'Pacifico'])

Attributes:
    HANDWRITING_FONTS (list): Curated list of handwriting and script font
        family names available on Google Fonts.

Command-line Arguments:
    --output, -o: Output directory for downloaded fonts (default: ./google_fonts)
    --max-fonts, -m: Maximum number of fonts to download (default: unlimited)
    --rate-limit, -r: Seconds to wait between requests (default: 0.5)
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
    """Information about a font from Google Fonts.

    Attributes:
        name: The display name of the font (same as family).
        family: The font family name as used in Google Fonts API.
        category: The font category (e.g., 'handwriting', 'script').
        variants: List of available font variants (e.g., ['regular', 'bold']).
        download_urls: Mapping of variant names to their download URLs.
    """
    name: str
    family: str
    category: str
    variants: List[str]
    download_urls: Dict[str, str]  # variant -> url

    def to_dict(self):
        """Convert the FontInfo to a dictionary for JSON serialization.

        Returns:
            dict: A dictionary containing all font information fields.
        """
        return {
            'name': self.name,
            'family': self.family,
            'category': self.category,
            'variants': self.variants,
            'download_urls': self.download_urls
        }


class GoogleFontsScraper:
    """Download handwriting fonts from Google Fonts.

    This class handles downloading fonts from Google Fonts using the CSS API.
    It maintains a curated list of handwriting and script fonts and can
    download them without requiring an API key.

    Attributes:
        HANDWRITING_FONTS (list): Class-level list of known handwriting and
            script font family names on Google Fonts.
        output_dir (Path): Directory where downloaded fonts are saved.
        rate_limit (float): Delay in seconds between HTTP requests.
        session (requests.Session): HTTP session for making requests.
        downloaded (Set[str]): Set of font families successfully downloaded.
        failed (List[str]): List of font families that failed to download.
    """

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
        """Initialize the Google Fonts scraper.

        Args:
            output_dir: Directory path where downloaded fonts will be saved.
                The directory will be created if it does not exist.
            rate_limit: Minimum seconds to wait between HTTP requests.
                Defaults to 0.5 seconds (faster than other scrapers since
                Google's servers handle high traffic well).
        """
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
        """Get the download URL for a font from the Google Fonts CSS API.

        Requests the CSS stylesheet for a font family and extracts the
        direct font file URL from the response.

        Args:
            family: The font family name (e.g., 'Caveat', 'Dancing Script').
            variant: The font variant to download (e.g., 'regular', 'bold',
                '400', '700italic'). Defaults to 'regular'.

        Returns:
            The direct URL to the font file on fonts.gstatic.com, or an
            empty string if the URL could not be obtained.
        """
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
        """Download a font family from Google Fonts.

        Fetches the font URL using the CSS API and downloads the font file
        to the output directory.

        Args:
            family: The font family name to download (e.g., 'Caveat').

        Returns:
            True if the font was successfully downloaded, False otherwise.
        """
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
        """Download specified fonts or all known handwriting fonts.

        This is the main entry point for the scraper. It downloads fonts
        from the provided list or defaults to the curated HANDWRITING_FONTS
        list.

        Args:
            fonts: List of font family names to download. If None, uses
                the built-in HANDWRITING_FONTS list containing curated
                handwriting and script fonts.
            max_fonts: Maximum number of fonts to download. If None,
                downloads all fonts in the list.

        Returns:
            A dictionary containing download statistics and metadata:
                - source (str): Always "google_fonts".
                - fonts_requested (int): Number of fonts in the download list.
                - fonts_downloaded (int): Number of fonts successfully downloaded.
                - fonts_failed (int): Number of fonts that failed to download.
                - downloaded (list): List of successfully downloaded font names.
                - failed (list): List of font names that failed to download.
        """
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
    """Parse command-line arguments and run the Google Fonts scraper.

    This function serves as the entry point when the module is run as a script.
    It configures argument parsing and initiates the download process.
    """
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
