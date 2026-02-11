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

    Using the common FontSource interface::

        from font_source import ScraperConfig
        scraper = GoogleFontsScraper('./output')
        config = ScraperConfig(max_fonts=50)
        metadata = scraper.scrape_and_download(config)

Attributes:
    HANDWRITING_FONTS (list): Curated list of handwriting and script font
        family names available on Google Fonts.

Command-line Arguments:
    --output, -o: Output directory for downloaded fonts (default: ./google_fonts)
    --max-fonts, -m: Maximum number of fonts to download (default: unlimited)
    --rate-limit, -r: Seconds to wait between requests (default: 0.5)
"""

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from font_source import FontMetadata, FontSource, ScraperConfig, create_scrape_metadata

logger = logging.getLogger(__name__)


class GoogleFontsScraper(FontSource):
    """Download handwriting fonts from Google Fonts.

    This class handles downloading fonts from Google Fonts using the CSS API.
    It maintains a curated list of handwriting and script fonts and can
    download them without requiring an API key.

    Inherits from FontSource to provide a consistent interface with other
    font scrapers (DaFont, FontSpace).

    Attributes:
        SOURCE_NAME (str): Identifier for this source ('google').
        HANDWRITING_FONTS (list): Class-level list of known handwriting and
            script font family names on Google Fonts.
        output_dir (Path): Directory where downloaded fonts are saved.
        rate_limit (float): Delay in seconds between HTTP requests.
        session (requests.Session): HTTP session for making requests.
        fonts_found (list): List of discovered FontMetadata objects.
        downloaded (Set[str]): Set of font families successfully downloaded.
        failed (List[str]): List of font families that failed to download.
    """

    SOURCE_NAME = "google"

    # HTTP timeout constants (seconds)
    PAGE_TIMEOUT = 30
    DOWNLOAD_TIMEOUT = 60

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
        super().__init__(output_dir, rate_limit)

        # Override User-Agent to get TTF instead of WOFF2
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; rv:60.0) Gecko/20100101 Firefox/60.0',
        })

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
            resp = self.get_with_retry(css_url, timeout=self.PAGE_TIMEOUT)
            resp.raise_for_status()

            # Extract font URL from CSS
            # Format: src: url(https://fonts.gstatic.com/...)
            urls = re.findall(r'url\((https://fonts\.gstatic\.com/[^)]+)\)', resp.text)

            if urls:
                return urls[0]

        except Exception as e:
            logger.debug("Failed to get font URL for %s: %s", family, e)

        return ''

    def scrape_fonts(self, config: ScraperConfig) -> list[FontMetadata]:
        """Discover fonts from Google Fonts.

        For Google Fonts, this creates FontMetadata objects from either
        the provided fonts list or the curated HANDWRITING_FONTS list.

        Args:
            config: ScraperConfig with options. Uses config.fonts if provided,
                otherwise defaults to HANDWRITING_FONTS.

        Returns:
            List of FontMetadata objects for discovered fonts.
        """
        # Use provided fonts list or default to curated list
        font_names = config.fonts if config.fonts else self.HANDWRITING_FONTS

        fonts = []
        for family in font_names:
            fonts.append(FontMetadata(
                name=family,
                family=family,
                source=self.SOURCE_NAME,
                category='handwriting',
                url=f"https://fonts.google.com/specimen/{family.replace(' ', '+')}",
            ))
        return fonts

    def download_font(self, font: FontMetadata | str) -> bool:
        """Download a font family from Google Fonts.

        Fetches the font URL using the CSS API and downloads the font file
        to the output directory. Tracks failures in self.failed.

        Args:
            font: FontMetadata object or font family name string.

        Returns:
            True if the font was successfully downloaded, False otherwise.
        """
        # Support both FontMetadata and string (backward compatibility)
        if isinstance(font, FontMetadata):
            family = font.family or font.name
        else:
            family = font

        if family in self.downloaded:
            return True

        url = self.get_font_url(family)
        if not url:
            logger.warning("Could not get download URL for %s", family)
            self.failed.append(family)
            return False

        # Use base class download method
        content = self.download_font_file(url, family, self.DOWNLOAD_TIMEOUT)
        if content is None:
            self.failed.append(family)
            return False

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
        safe_name = self.safe_filename(family.replace(' ', '_'))
        out_path = self.output_dir / f"google_{safe_name}{ext}"
        out_path.write_bytes(content)

        logger.debug("Saved: %s", out_path.name)
        self.downloaded.add(family)
        return True

    def scrape_and_download(
        self,
        config: ScraperConfig | None = None,
        fonts: list[str] = None,
        max_fonts: int = None
    ) -> dict:
        """Download specified fonts or all known handwriting fonts.

        This is the main entry point for the scraper. It downloads fonts
        from the provided list or defaults to the curated HANDWRITING_FONTS
        list.

        Supports both the new ScraperConfig interface and the legacy
        keyword arguments for backward compatibility.

        Args:
            config: ScraperConfig with options. If provided, fonts and
                max_fonts arguments are ignored.
            fonts: List of font family names to download. If None, uses
                the built-in HANDWRITING_FONTS list containing curated
                handwriting and script fonts.
            max_fonts: Maximum number of fonts to download. If None,
                downloads all fonts in the list.

        Returns:
            A dictionary containing download statistics and metadata:
                - source (str): Always "google" or "google_fonts".
                - fonts_requested/fonts_found (int): Number of fonts in the download list.
                - fonts_downloaded/downloaded (int): Number of fonts successfully downloaded.
                - fonts_failed/failed (int): Number of fonts that failed to download.
                - downloaded (list): List of successfully downloaded font names.
                - failed (list): List of font names that failed to download.
        """
        # Support both ScraperConfig and legacy arguments
        if config is not None:
            # Use the base class implementation
            return super().scrape_and_download(config)

        # Legacy interface
        logger.info("=" * 60)
        logger.info("Google Fonts Scraper")
        logger.info("=" * 60)
        logger.info("Output directory: %s", self.output_dir)

        if fonts is None:
            fonts = self.HANDWRITING_FONTS.copy()

        logger.info("Fonts to download: %d", len(fonts))

        if max_fonts:
            fonts = fonts[:max_fonts]
            logger.info("Limiting to %d fonts", max_fonts)

        # Download
        logger.info("Downloading %d fonts...", len(fonts))

        for i, family in enumerate(fonts, 1):
            self.log_progress(i, len(fonts), family)
            # download_font now handles adding to downloaded/failed internally
            self.download_font(family)
            time.sleep(self.rate_limit)

        # Build standardized metadata
        metadata = create_scrape_metadata(
            source=self.SOURCE_NAME,
            fonts_found=len(fonts),
            fonts_downloaded=len(self.downloaded),
            fonts_failed=len(self.failed),
            output_dir=str(self.output_dir),
        )

        meta_path = self.output_dir / 'google_fonts_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 60)
        logger.info("COMPLETE")
        logger.info("=" * 60)
        logger.info("Fonts downloaded: %d", len(self.downloaded))
        logger.info("Failed: %d", len(self.failed))

        return metadata


def main() -> None:
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
