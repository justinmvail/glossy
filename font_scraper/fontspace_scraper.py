"""
FontSpace Scraper - Download handwriting fonts from FontSpace.com

Usage:
    python fontspace_scraper.py --output ./fonts --pages 10
"""

import argparse
import os
import re
import time
import zipfile
import io
from pathlib import Path
from typing import List, Set
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlencode
import json


@dataclass
class FontInfo:
    name: str
    url: str
    download_url: str
    downloads: int = 0

    def to_dict(self):
        return {
            'name': self.name,
            'url': self.url,
            'download_url': self.download_url,
            'downloads': self.downloads
        }


class FontSpaceScraper:
    """Scrape handwriting fonts from FontSpace.com."""

    BASE_URL = "https://www.fontspace.com"

    def __init__(self, output_dir: str, rate_limit: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        })

        self.fonts_found: List[FontInfo] = []
        self.downloaded: Set[str] = set()
        self.failed: List[str] = []

    def scrape_search(self, query: str = "handwritten", max_pages: int = 10) -> List[FontInfo]:
        """Scrape fonts from search results."""
        print(f"\nSearching FontSpace for: {query}")

        fonts = []
        page = 1

        while page <= max_pages:
            url = f"{self.BASE_URL}/search?q={query}&p={page}"
            print(f"  Page {page}: {url}")

            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()

                page_fonts = self._parse_search_results(resp.text)

                if not page_fonts:
                    print(f"  No fonts found on page {page}, stopping.")
                    break

                fonts.extend(page_fonts)
                print(f"  Found {len(page_fonts)} fonts (total: {len(fonts)})")

                page += 1
                time.sleep(self.rate_limit)

            except requests.RequestException as e:
                print(f"  Error fetching page {page}: {e}")
                break

        return fonts

    def scrape_category(self, category: str = "handwriting", max_pages: int = 10) -> List[FontInfo]:
        """Scrape fonts from a category."""
        print(f"\nScraping FontSpace category: {category}")

        fonts = []
        page = 1

        while page <= max_pages:
            url = f"{self.BASE_URL}/category/{category}?p={page}"
            print(f"  Page {page}: {url}")

            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()

                page_fonts = self._parse_search_results(resp.text)

                if not page_fonts:
                    print(f"  No fonts found on page {page}, stopping.")
                    break

                fonts.extend(page_fonts)
                print(f"  Found {len(page_fonts)} fonts (total: {len(fonts)})")

                page += 1
                time.sleep(self.rate_limit)

            except requests.RequestException as e:
                print(f"  Error fetching page {page}: {e}")
                break

        return fonts

    def _parse_search_results(self, html: str) -> List[FontInfo]:
        """Parse font list from search/category page."""
        soup = BeautifulSoup(html, 'html.parser')
        fonts = []

        # FontSpace uses font cards with links
        # Look for font links - they typically have /font/ in the URL
        for link in soup.find_all('a', href=re.compile(r'/font/[^/]+$')):
            try:
                font_name = link.get_text(strip=True)
                font_url = link.get('href', '')

                if not font_name or len(font_name) < 2:
                    continue

                # Skip navigation/UI links
                if font_name.lower() in ['download', 'preview', 'share', 'embed']:
                    continue

                if font_url and not font_url.startswith('http'):
                    font_url = urljoin(self.BASE_URL, font_url)

                # Build download URL from font page URL
                # FontSpace download: /font/fontname -> click download button
                # We'll get the actual download URL when we visit the font page
                font_slug = font_url.split('/font/')[-1] if '/font/' in font_url else ''

                fonts.append(FontInfo(
                    name=font_name,
                    url=font_url,
                    download_url='',  # Will be fetched per-font
                    downloads=0
                ))

            except Exception:
                continue

        # Deduplicate by URL
        seen = set()
        unique_fonts = []
        for f in fonts:
            if f.url not in seen:
                seen.add(f.url)
                unique_fonts.append(f)

        return unique_fonts

    def _get_download_url(self, font_url: str) -> str:
        """Fetch the actual download URL from a font's page."""
        try:
            resp = self.session.get(font_url, timeout=30)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Look for download button/link
            dl_link = soup.find('a', href=re.compile(r'/download/'))
            if dl_link:
                url = dl_link.get('href', '')
                if url and not url.startswith('http'):
                    url = urljoin(self.BASE_URL, url)
                return url

            # Alternative: look for direct ZIP link
            zip_link = soup.find('a', href=re.compile(r'\.zip'))
            if zip_link:
                url = zip_link.get('href', '')
                if url and not url.startswith('http'):
                    url = urljoin(self.BASE_URL, url)
                return url

        except Exception as e:
            print(f"    Error getting download URL: {e}")

        return ''

    def download_font(self, font: FontInfo) -> bool:
        """Download a font and extract TTF/OTF files."""
        if font.name in self.downloaded:
            return True

        try:
            print(f"  Downloading: {font.name}")

            # Get download URL if we don't have it
            if not font.download_url:
                font.download_url = self._get_download_url(font.url)
                time.sleep(0.5)

            if not font.download_url:
                print(f"    Could not find download URL")
                self.failed.append(font.name)
                return False

            resp = self.session.get(font.download_url, timeout=60)
            resp.raise_for_status()

            # Check if it's a ZIP file
            if resp.content[:4] == b'PK\x03\x04':
                extracted = self._extract_zip(resp.content)
                if extracted > 0:
                    print(f"    Extracted {extracted} font file(s)")
                    self.downloaded.add(font.name)
                    return True
                else:
                    print(f"    No TTF/OTF files in ZIP")
                    return False
            else:
                # Might be a direct font file
                content_type = resp.headers.get('content-type', '')
                if 'font' in content_type or font.download_url.endswith(('.ttf', '.otf')):
                    ext = '.ttf' if '.ttf' in font.download_url else '.otf'
                    safe_name = self._safe_filename(font.name) + ext
                    out_path = self.output_dir / safe_name
                    out_path.write_bytes(resp.content)
                    print(f"    Saved {safe_name}")
                    self.downloaded.add(font.name)
                    return True

            print(f"    Unknown file format")
            return False

        except Exception as e:
            print(f"    Error: {e}")
            self.failed.append(font.name)
            return False

    def _extract_zip(self, content: bytes) -> int:
        """Extract font files from ZIP content."""
        extracted = 0
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for name in zf.namelist():
                    lower_name = name.lower()
                    if lower_name.endswith(('.ttf', '.otf')):
                        safe_name = self._safe_filename(os.path.basename(name))
                        out_path = self.output_dir / safe_name
                        if not out_path.exists():
                            with zf.open(name) as src, open(out_path, 'wb') as dst:
                                dst.write(src.read())
                            extracted += 1
        except zipfile.BadZipFile:
            pass
        return extracted

    def _safe_filename(self, name: str) -> str:
        """Create a safe filename."""
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe = safe.strip('. ')
        return safe or 'unnamed_font'

    def scrape_and_download(
        self,
        query: str = "handwritten",
        max_pages: int = 10,
        max_fonts: int = None,
        use_category: bool = False
    ) -> dict:
        """Scrape and download fonts."""
        print("=" * 60)
        print("FontSpace Scraper")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Query/Category: {query}")
        print(f"Max pages: {max_pages}")

        # Scrape
        if use_category:
            fonts = self.scrape_category(query, max_pages)
        else:
            fonts = self.scrape_search(query, max_pages)

        self.fonts_found = fonts

        print(f"\nTotal fonts found: {len(fonts)}")

        # Limit if specified
        if max_fonts:
            fonts = fonts[:max_fonts]
            print(f"Limiting to {max_fonts} fonts")

        # Download
        print(f"\nDownloading {len(fonts)} fonts...")

        for i, font in enumerate(fonts):
            print(f"[{i+1}/{len(fonts)}]", end='')
            self.download_font(font)
            time.sleep(self.rate_limit)

        # Save metadata
        metadata = {
            'source': 'fontspace',
            'query': query,
            'fonts_found': len(self.fonts_found),
            'fonts_downloaded': len(self.downloaded),
            'fonts_failed': len(self.failed),
            'font_list': [f.to_dict() for f in fonts]
        }

        meta_path = self.output_dir / 'fontspace_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)
        print(f"Fonts found: {len(self.fonts_found)}")
        print(f"Fonts downloaded: {len(self.downloaded)}")
        print(f"Failed: {len(self.failed)}")

        return metadata


def main():
    parser = argparse.ArgumentParser(description='Scrape fonts from FontSpace')
    parser.add_argument('--output', '-o', type=str, default='./fontspace_fonts',
                        help='Output directory')
    parser.add_argument('--query', '-q', type=str, default='handwritten',
                        help='Search query')
    parser.add_argument('--pages', '-p', type=int, default=10,
                        help='Max pages to scrape')
    parser.add_argument('--max-fonts', '-m', type=int, default=None,
                        help='Max fonts to download')
    parser.add_argument('--category', '-c', action='store_true',
                        help='Use category instead of search')
    parser.add_argument('--rate-limit', '-r', type=float, default=1.0,
                        help='Seconds between requests')

    args = parser.parse_args()

    scraper = FontSpaceScraper(args.output, rate_limit=args.rate_limit)
    scraper.scrape_and_download(
        query=args.query,
        max_pages=args.pages,
        max_fonts=args.max_fonts,
        use_category=args.category
    )


if __name__ == '__main__':
    main()
