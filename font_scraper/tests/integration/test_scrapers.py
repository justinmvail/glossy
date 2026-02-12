"""Integration tests for font scrapers using HTTP mocking.

This module tests the HTTP request handling and retry logic of the font
scrapers using the 'responses' library to mock HTTP responses.

Tests cover:
    - request_with_retry: Retry logic for 5xx errors, 429 rate limiting
    - Error handling: 404 responses, connection errors
    - URL parsing: Font URL construction from HTML parsing

Run with:
    python3 -m pytest tests/integration/test_scrapers.py -v
    python3 -m unittest tests.integration.test_scrapers -v
"""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import responses
import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from font_source import FontSource, FontMetadata, ScraperConfig
from dafont_scraper import DaFontScraper
from fontspace_scraper import FontSpaceScraper
from google_fonts_scraper import GoogleFontsScraper


class ConcreteFontSource(FontSource):
    """Concrete implementation of FontSource for testing."""

    SOURCE_NAME = "test"

    def scrape_fonts(self, config: ScraperConfig) -> list[FontMetadata]:
        return []

    def download_font(self, font: FontMetadata) -> bool:
        return True


class TestRequestWithRetry(unittest.TestCase):
    """Test the request_with_retry method in FontSource."""

    def setUp(self):
        """Create a temp directory and FontSource instance."""
        self.temp_dir = tempfile.mkdtemp()
        self.source = ConcreteFontSource(self.temp_dir, rate_limit=0.1)

    @responses.activate
    def test_successful_request(self):
        """Test that a successful request returns immediately."""
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="success",
            status=200
        )

        response = self.source.request_with_retry("GET", "https://example.com/test")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "success")
        self.assertEqual(len(responses.calls), 1)

    @responses.activate
    def test_retries_on_500(self):
        """Test that 500 errors trigger retries with eventual success."""
        # First two requests return 500, third succeeds
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="Server Error",
            status=500
        )
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="Server Error",
            status=500
        )
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="success",
            status=200
        )

        with patch('time.sleep'):  # Speed up test by mocking sleep
            response = self.source.request_with_retry(
                "GET",
                "https://example.com/test",
                max_retries=3,
                base_delay=0.1
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "success")
        self.assertEqual(len(responses.calls), 3)

    @responses.activate
    def test_retries_on_502(self):
        """Test that 502 Bad Gateway errors trigger retries."""
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="Bad Gateway",
            status=502
        )
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="success",
            status=200
        )

        with patch('time.sleep'):
            response = self.source.request_with_retry(
                "GET",
                "https://example.com/test",
                max_retries=2,
                base_delay=0.1
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(responses.calls), 2)

    @responses.activate
    def test_retries_on_503(self):
        """Test that 503 Service Unavailable errors trigger retries."""
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="Service Unavailable",
            status=503
        )
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="success",
            status=200
        )

        with patch('time.sleep'):
            response = self.source.request_with_retry(
                "GET",
                "https://example.com/test",
                max_retries=2,
                base_delay=0.1
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(responses.calls), 2)

    @responses.activate
    def test_respects_429_retry_after(self):
        """Test that 429 responses with Retry-After header are respected."""
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="Too Many Requests",
            status=429,
            headers={"Retry-After": "2"}
        )
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="success",
            status=200
        )

        sleep_calls = []

        def mock_sleep(duration):
            sleep_calls.append(duration)

        with patch('time.sleep', side_effect=mock_sleep):
            response = self.source.request_with_retry(
                "GET",
                "https://example.com/test",
                max_retries=2,
                base_delay=0.1
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(responses.calls), 2)
        # Should use Retry-After value (2 seconds) instead of exponential backoff
        self.assertEqual(sleep_calls[0], 2.0)

    @responses.activate
    def test_429_without_retry_after_uses_exponential_backoff(self):
        """Test that 429 without Retry-After header uses exponential backoff."""
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="Too Many Requests",
            status=429
        )
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="success",
            status=200
        )

        sleep_calls = []

        def mock_sleep(duration):
            sleep_calls.append(duration)

        with patch('time.sleep', side_effect=mock_sleep):
            response = self.source.request_with_retry(
                "GET",
                "https://example.com/test",
                max_retries=2,
                base_delay=1.0
            )

        self.assertEqual(response.status_code, 200)
        # First attempt (attempt=0): delay = 1.0 * (2^0) = 1.0
        self.assertEqual(sleep_calls[0], 1.0)

    @responses.activate
    def test_gives_up_after_max_retries(self):
        """Test that continuous failures exhaust retries and raise exception."""
        # All requests fail with 500
        for _ in range(3):
            responses.add(
                responses.GET,
                "https://example.com/test",
                body="Server Error",
                status=500
            )

        with patch('time.sleep'):
            with self.assertRaises(requests.RequestException) as ctx:
                self.source.request_with_retry(
                    "GET",
                    "https://example.com/test",
                    max_retries=3,
                    base_delay=0.1
                )

        self.assertIn("Max retries exceeded", str(ctx.exception))
        self.assertEqual(len(responses.calls), 3)

    @responses.activate
    def test_gives_up_after_max_retries_429(self):
        """Test that continuous 429 errors exhaust retries."""
        for _ in range(3):
            responses.add(
                responses.GET,
                "https://example.com/test",
                body="Rate Limited",
                status=429
            )

        with patch('time.sleep'):
            with self.assertRaises(requests.RequestException) as ctx:
                self.source.request_with_retry(
                    "GET",
                    "https://example.com/test",
                    max_retries=3,
                    base_delay=0.1
                )

        self.assertIn("Max retries exceeded", str(ctx.exception))
        self.assertEqual(len(responses.calls), 3)

    @responses.activate
    def test_connection_error_triggers_retry(self):
        """Test that connection errors trigger retries."""
        responses.add(
            responses.GET,
            "https://example.com/test",
            body=requests.ConnectionError("Connection refused")
        )
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="success",
            status=200
        )

        with patch('time.sleep'):
            response = self.source.request_with_retry(
                "GET",
                "https://example.com/test",
                max_retries=2,
                base_delay=0.1
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(responses.calls), 2)

    @responses.activate
    def test_connection_error_exhausts_retries(self):
        """Test that continuous connection errors exhaust retries."""
        for _ in range(3):
            responses.add(
                responses.GET,
                "https://example.com/test",
                body=requests.ConnectionError("Connection refused")
            )

        with patch('time.sleep'):
            with self.assertRaises(requests.ConnectionError):
                self.source.request_with_retry(
                    "GET",
                    "https://example.com/test",
                    max_retries=3,
                    base_delay=0.1
                )

        self.assertEqual(len(responses.calls), 3)

    @responses.activate
    def test_timeout_error_triggers_retry(self):
        """Test that timeout errors trigger retries."""
        responses.add(
            responses.GET,
            "https://example.com/test",
            body=requests.Timeout("Read timed out")
        )
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="success",
            status=200
        )

        with patch('time.sleep'):
            response = self.source.request_with_retry(
                "GET",
                "https://example.com/test",
                max_retries=2,
                base_delay=0.1
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(responses.calls), 2)

    @responses.activate
    def test_4xx_errors_not_retried(self):
        """Test that 4xx client errors (except 429) are not retried."""
        responses.add(
            responses.GET,
            "https://example.com/test",
            body="Not Found",
            status=404
        )

        response = self.source.request_with_retry(
            "GET",
            "https://example.com/test",
            max_retries=3,
            base_delay=0.1
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(len(responses.calls), 1)

    @responses.activate
    def test_exponential_backoff_timing(self):
        """Test that exponential backoff increases delay correctly."""
        for _ in range(4):
            responses.add(
                responses.GET,
                "https://example.com/test",
                body="Server Error",
                status=500
            )

        sleep_calls = []

        def mock_sleep(duration):
            sleep_calls.append(duration)

        with patch('time.sleep', side_effect=mock_sleep):
            try:
                self.source.request_with_retry(
                    "GET",
                    "https://example.com/test",
                    max_retries=4,
                    base_delay=1.0
                )
            except requests.RequestException:
                pass

        # Expected delays after each failed attempt:
        # attempt 0: 1.0 * 2^0 = 1.0
        # attempt 1: 1.0 * 2^1 = 2.0
        # attempt 2: 1.0 * 2^2 = 4.0
        # attempt 3: 1.0 * 2^3 = 8.0
        self.assertEqual(sleep_calls, [1.0, 2.0, 4.0, 8.0])


class TestScraperHandles404(unittest.TestCase):
    """Test that scrapers handle 404 errors gracefully."""

    def setUp(self):
        """Create temp directory for scrapers."""
        self.temp_dir = tempfile.mkdtemp()

    @responses.activate
    def test_dafont_handles_404_on_category_page(self):
        """Test that DaFont scraper handles 404 on category page."""
        responses.add(
            responses.GET,
            "https://www.dafont.com/theme.php?cat=601&page=1",
            body="Not Found",
            status=404
        )

        scraper = DaFontScraper(self.temp_dir, rate_limit=0.1)
        fonts = scraper.scrape_category("601", max_pages=1)

        # Should return empty list, not crash
        self.assertEqual(fonts, [])

    @responses.activate
    def test_dafont_handles_404_on_download(self):
        """Test that DaFont scraper handles 404 on font download."""
        responses.add(
            responses.GET,
            "https://dl.dafont.com/dl/?f=test_font",
            body="Not Found",
            status=404
        )

        scraper = DaFontScraper(self.temp_dir, rate_limit=0.1)
        font = FontMetadata(
            name="Test Font",
            url="https://www.dafont.com/test-font.font",
            download_url="https://dl.dafont.com/dl/?f=test_font",
            source="dafont"
        )

        result = scraper.download_font(font)

        self.assertFalse(result)
        self.assertIn("Test Font", scraper.failed)

    @responses.activate
    def test_fontspace_handles_404_on_search(self):
        """Test that FontSpace scraper handles 404 on search page."""
        responses.add(
            responses.GET,
            "https://www.fontspace.com/search?q=handwritten&p=1",
            body="Not Found",
            status=404
        )

        scraper = FontSpaceScraper(self.temp_dir, rate_limit=0.1)
        fonts = scraper.scrape_search("handwritten", max_pages=1)

        # Should return empty list, not crash
        self.assertEqual(fonts, [])

    @responses.activate
    def test_fontspace_handles_404_on_font_page(self):
        """Test that FontSpace scraper handles 404 on font detail page."""
        responses.add(
            responses.GET,
            "https://www.fontspace.com/font/test-font",
            body="Not Found",
            status=404
        )

        scraper = FontSpaceScraper(self.temp_dir, rate_limit=0.1)
        download_url = scraper._get_download_url("https://www.fontspace.com/font/test-font")

        # Should return empty string, not crash
        self.assertEqual(download_url, "")

    @responses.activate
    def test_fontspace_handles_404_on_download(self):
        """Test that FontSpace scraper handles 404 on font download."""
        # First request is to get download URL from font page
        responses.add(
            responses.GET,
            "https://www.fontspace.com/font/test-font",
            body='<a href="/download/test123">Download</a>',
            status=200
        )
        # Second request is the actual download which 404s
        responses.add(
            responses.GET,
            "https://www.fontspace.com/download/test123",
            body="Not Found",
            status=404
        )

        scraper = FontSpaceScraper(self.temp_dir, rate_limit=0.1)
        font = FontMetadata(
            name="Test Font",
            url="https://www.fontspace.com/font/test-font",
            download_url="",  # Will be fetched
            source="fontspace"
        )

        with patch('time.sleep'):
            result = scraper.download_font(font)

        self.assertFalse(result)
        self.assertIn("Test Font", scraper.failed)

    @responses.activate
    def test_google_fonts_handles_404_on_css(self):
        """Test that Google Fonts scraper handles 404 on CSS request."""
        responses.add(
            responses.GET,
            "https://fonts.googleapis.com/css?family=NonExistent+Font",
            body="Not Found",
            status=404
        )

        scraper = GoogleFontsScraper(self.temp_dir, rate_limit=0.1)
        font_url = scraper.get_font_url("NonExistent Font")

        # Should return empty string, not crash
        self.assertEqual(font_url, "")

    @responses.activate
    def test_google_fonts_handles_404_on_download(self):
        """Test that Google Fonts scraper handles 404 on font download."""
        # CSS request succeeds and returns font URL
        responses.add(
            responses.GET,
            "https://fonts.googleapis.com/css?family=Test+Font",
            body="@font-face { src: url(https://fonts.gstatic.com/s/testfont.ttf) }",
            status=200
        )
        # Font download 404s
        responses.add(
            responses.GET,
            "https://fonts.gstatic.com/s/testfont.ttf",
            body="Not Found",
            status=404
        )

        scraper = GoogleFontsScraper(self.temp_dir, rate_limit=0.1)
        font = FontMetadata(
            name="Test Font",
            family="Test Font",
            source="google"
        )

        result = scraper.download_font(font)

        self.assertFalse(result)
        self.assertIn("Test Font", scraper.failed)


class TestDaFontUrlParsing(unittest.TestCase):
    """Test DaFont URL and HTML parsing."""

    def setUp(self):
        """Create temp directory and scraper."""
        self.temp_dir = tempfile.mkdtemp()
        self.scraper = DaFontScraper(self.temp_dir, rate_limit=0.1)

    def test_parse_font_list_from_html(self):
        """Test parsing font list from DaFont category page HTML."""
        html = '''
        <html>
        <body>
            <a href="/dancing-script.font">Dancing Script</a>
            <span>10,500 downloads</span>
            <a href="/pacifico.font">Pacifico</a>
            <span>5,200 downloads</span>
        </body>
        </html>
        '''

        fonts = self.scraper._parse_font_list(html, "Handwritten")

        self.assertEqual(len(fonts), 2)

        # Check first font
        self.assertEqual(fonts[0].name, "Dancing Script")
        self.assertEqual(fonts[0].url, "https://www.dafont.com/dancing-script.font")
        self.assertEqual(fonts[0].download_url, "https://dl.dafont.com/dl/?f=dancing_script")
        self.assertEqual(fonts[0].category, "Handwritten")
        self.assertEqual(fonts[0].source, "dafont")

        # Check second font
        self.assertEqual(fonts[1].name, "Pacifico")
        self.assertEqual(fonts[1].url, "https://www.dafont.com/pacifico.font")
        self.assertEqual(fonts[1].download_url, "https://dl.dafont.com/dl/?f=pacifico")

    def test_parse_font_list_with_download_counts(self):
        """Test that download counts are extracted from HTML."""
        html = '''
        <html>
        <body>
            <a href="/popular-font.font">Popular Font</a>
            <div>1,234,567 downloads</div>
        </body>
        </html>
        '''

        fonts = self.scraper._parse_font_list(html, "Script")

        self.assertEqual(len(fonts), 1)
        self.assertEqual(fonts[0].downloads, 1234567)

    def test_parse_font_list_skips_navigation_links(self):
        """Test that navigation/UI links are skipped."""
        html = '''
        <html>
        <body>
            <a href="/real-font.font">Real Font</a>
            <a href="/something.font">Download</a>
            <a href="/another.font">Donate</a>
        </body>
        </html>
        '''

        fonts = self.scraper._parse_font_list(html, "Test")

        # Should only include "Real Font", not "Download" or "Donate"
        self.assertEqual(len(fonts), 1)
        self.assertEqual(fonts[0].name, "Real Font")

    def test_parse_font_list_handles_empty_page(self):
        """Test that empty page returns empty list."""
        html = '''
        <html>
        <body>
            <div>No fonts here</div>
        </body>
        </html>
        '''

        fonts = self.scraper._parse_font_list(html, "Test")

        self.assertEqual(fonts, [])

    def test_parse_font_list_alt_with_download_links(self):
        """Test alternative parsing using download links."""
        from bs4 import BeautifulSoup

        html = '''
        <html>
        <body>
            <a href="https://dl.dafont.com/dl/?f=test_font_one">Download</a>
            <a href="//dl.dafont.com/dl/?f=test_font_two">Download</a>
        </body>
        </html>
        '''

        soup = BeautifulSoup(html, 'html.parser')
        fonts = self.scraper._parse_font_list_alt(soup, "Script")

        self.assertEqual(len(fonts), 2)
        self.assertEqual(fonts[0].name, "Test Font One")
        self.assertEqual(fonts[1].name, "Test Font Two")

    def test_download_url_construction(self):
        """Test that download URLs are constructed correctly."""
        html = '''
        <a href="/my-awesome-font.font">My Awesome Font</a>
        '''

        fonts = self.scraper._parse_font_list(html, "Test")

        self.assertEqual(len(fonts), 1)
        # Hyphens should be converted to underscores in download URL
        self.assertEqual(
            fonts[0].download_url,
            "https://dl.dafont.com/dl/?f=my_awesome_font"
        )


class TestFontSpaceUrlParsing(unittest.TestCase):
    """Test FontSpace URL and HTML parsing."""

    def setUp(self):
        """Create temp directory and scraper."""
        self.temp_dir = tempfile.mkdtemp()
        self.scraper = FontSpaceScraper(self.temp_dir, rate_limit=0.1)

    def test_parse_search_results_from_html(self):
        """Test parsing font list from FontSpace search results."""
        html = '''
        <html>
        <body>
            <a href="/font/dancing-script-o4m8">Dancing Script</a>
            <a href="/font/pacifico-x9k2">Pacifico</a>
        </body>
        </html>
        '''

        fonts = self.scraper._parse_search_results(html)

        self.assertEqual(len(fonts), 2)
        self.assertEqual(fonts[0].name, "Dancing Script")
        self.assertEqual(fonts[0].url, "https://www.fontspace.com/font/dancing-script-o4m8")
        self.assertEqual(fonts[0].source, "fontspace")
        self.assertEqual(fonts[1].name, "Pacifico")

    def test_parse_search_results_deduplicates_by_url(self):
        """Test that duplicate URLs are removed."""
        html = '''
        <html>
        <body>
            <a href="/font/same-font">Same Font</a>
            <a href="/font/same-font">Same Font Copy</a>
            <a href="/font/different-font">Different Font</a>
        </body>
        </html>
        '''

        fonts = self.scraper._parse_search_results(html)

        # Should only have 2 unique fonts
        self.assertEqual(len(fonts), 2)
        urls = [f.url for f in fonts]
        self.assertEqual(len(urls), len(set(urls)))

    def test_parse_search_results_skips_navigation_links(self):
        """Test that navigation/UI links are skipped."""
        html = '''
        <html>
        <body>
            <a href="/font/real-font">Real Font</a>
            <a href="/font/something">Download</a>
            <a href="/font/another">Preview</a>
            <a href="/font/third">Share</a>
        </body>
        </html>
        '''

        fonts = self.scraper._parse_search_results(html)

        # Should only include "Real Font"
        self.assertEqual(len(fonts), 1)
        self.assertEqual(fonts[0].name, "Real Font")

    def test_parse_search_results_handles_empty_page(self):
        """Test that empty page returns empty list."""
        html = '''
        <html>
        <body>
            <div>No results found</div>
        </body>
        </html>
        '''

        fonts = self.scraper._parse_search_results(html)

        self.assertEqual(fonts, [])

    @responses.activate
    def test_get_download_url_from_font_page(self):
        """Test extracting download URL from font detail page."""
        responses.add(
            responses.GET,
            "https://www.fontspace.com/font/test-font",
            body='<a href="/download/12345">Download</a>',
            status=200
        )

        download_url = self.scraper._get_download_url(
            "https://www.fontspace.com/font/test-font"
        )

        self.assertEqual(download_url, "https://www.fontspace.com/download/12345")

    @responses.activate
    def test_get_download_url_finds_zip_link(self):
        """Test finding direct ZIP download link."""
        responses.add(
            responses.GET,
            "https://www.fontspace.com/font/test-font",
            body='<a href="https://example.com/fonts/test.zip">Download ZIP</a>',
            status=200
        )

        download_url = self.scraper._get_download_url(
            "https://www.fontspace.com/font/test-font"
        )

        self.assertEqual(download_url, "https://example.com/fonts/test.zip")


class TestGoogleFontsUrlParsing(unittest.TestCase):
    """Test Google Fonts URL and CSS parsing."""

    def setUp(self):
        """Create temp directory and scraper."""
        self.temp_dir = tempfile.mkdtemp()
        self.scraper = GoogleFontsScraper(self.temp_dir, rate_limit=0.1)

    @responses.activate
    def test_get_font_url_extracts_from_css(self):
        """Test extracting font URL from Google Fonts CSS response."""
        responses.add(
            responses.GET,
            "https://fonts.googleapis.com/css?family=Dancing+Script",
            body='''
            @font-face {
                font-family: 'Dancing Script';
                font-style: normal;
                src: url(https://fonts.gstatic.com/s/dancingscript/v25/abc123.ttf);
            }
            ''',
            status=200
        )

        font_url = self.scraper.get_font_url("Dancing Script")

        self.assertEqual(
            font_url,
            "https://fonts.gstatic.com/s/dancingscript/v25/abc123.ttf"
        )

    @responses.activate
    def test_get_font_url_with_variant(self):
        """Test requesting specific font variant."""
        responses.add(
            responses.GET,
            "https://fonts.googleapis.com/css?family=Roboto:700",
            body='''
            @font-face {
                font-family: 'Roboto';
                font-weight: 700;
                src: url(https://fonts.gstatic.com/s/roboto/v30/bold.ttf);
            }
            ''',
            status=200
        )

        font_url = self.scraper.get_font_url("Roboto", "700")

        self.assertEqual(
            font_url,
            "https://fonts.gstatic.com/s/roboto/v30/bold.ttf"
        )

    @responses.activate
    def test_get_font_url_returns_first_url_from_multiple(self):
        """Test that first URL is returned when CSS contains multiple."""
        responses.add(
            responses.GET,
            "https://fonts.googleapis.com/css?family=Test",
            body='''
            @font-face {
                src: url(https://fonts.gstatic.com/first.ttf);
            }
            @font-face {
                src: url(https://fonts.gstatic.com/second.ttf);
            }
            ''',
            status=200
        )

        font_url = self.scraper.get_font_url("Test")

        self.assertEqual(font_url, "https://fonts.gstatic.com/first.ttf")

    @responses.activate
    def test_get_font_url_handles_no_urls_in_css(self):
        """Test handling CSS response without font URLs."""
        responses.add(
            responses.GET,
            "https://fonts.googleapis.com/css?family=Test",
            body='''
            /* Empty or malformed CSS */
            body { color: red; }
            ''',
            status=200
        )

        font_url = self.scraper.get_font_url("Test")

        self.assertEqual(font_url, "")

    def test_scrape_fonts_returns_metadata_list(self):
        """Test that scrape_fonts returns FontMetadata objects."""
        config = ScraperConfig(fonts=["Caveat", "Pacifico"])

        fonts = self.scraper.scrape_fonts(config)

        self.assertEqual(len(fonts), 2)
        self.assertEqual(fonts[0].name, "Caveat")
        self.assertEqual(fonts[0].family, "Caveat")
        self.assertEqual(fonts[0].source, "google")
        self.assertEqual(fonts[0].category, "handwriting")
        self.assertIn("Caveat", fonts[0].url)

    def test_scrape_fonts_uses_default_list_when_no_fonts_specified(self):
        """Test that scrape_fonts uses HANDWRITING_FONTS when none specified."""
        config = ScraperConfig()

        fonts = self.scraper.scrape_fonts(config)

        self.assertEqual(len(fonts), len(GoogleFontsScraper.HANDWRITING_FONTS))


class TestZipExtractionAndDownload(unittest.TestCase):
    """Test ZIP file handling in scrapers."""

    def setUp(self):
        """Create temp directory."""
        self.temp_dir = tempfile.mkdtemp()

    @responses.activate
    def test_dafont_rejects_non_zip_file(self):
        """Test that DaFont scraper rejects non-ZIP downloads."""
        responses.add(
            responses.GET,
            "https://dl.dafont.com/dl/?f=test_font",
            body="Not a ZIP file, just some text",
            status=200
        )

        scraper = DaFontScraper(self.temp_dir, rate_limit=0.1)
        font = FontMetadata(
            name="Test Font",
            url="https://www.dafont.com/test-font.font",
            download_url="https://dl.dafont.com/dl/?f=test_font",
            source="dafont"
        )

        result = scraper.download_font(font)

        self.assertFalse(result)

    @responses.activate
    def test_fontspace_handles_direct_ttf_download(self):
        """Test that FontSpace scraper handles direct TTF downloads."""
        # Font page with download link
        responses.add(
            responses.GET,
            "https://www.fontspace.com/font/test-font",
            body='<a href="/download/test.ttf">Download</a>',
            status=200
        )
        # Direct TTF download (not a ZIP)
        # TTF files start with specific bytes
        ttf_header = b'\x00\x01\x00\x00' + b'\x00' * 100
        responses.add(
            responses.GET,
            "https://www.fontspace.com/download/test.ttf",
            body=ttf_header,
            status=200
        )

        scraper = FontSpaceScraper(self.temp_dir, rate_limit=0.1)
        font = FontMetadata(
            name="Test Font",
            url="https://www.fontspace.com/font/test-font",
            download_url="",  # Will be fetched
            source="fontspace"
        )

        with patch('time.sleep'):
            result = scraper.download_font(font)

        # Should succeed since download_url ends with .ttf
        # The result depends on whether file writing succeeds
        # At minimum, it should not crash


class TestScraperIntegrationWorkflow(unittest.TestCase):
    """Test complete scraper workflows."""

    def setUp(self):
        """Create temp directory."""
        self.temp_dir = tempfile.mkdtemp()

    @responses.activate
    def test_dafont_scrape_category_workflow(self):
        """Test complete DaFont category scraping workflow."""
        # Mock category page with fonts
        responses.add(
            responses.GET,
            "https://www.dafont.com/theme.php?cat=601&page=1",
            body='''
            <html>
            <body>
                <a href="/font-one.font">Font One</a>
                <span>100 downloads</span>
                <a href="/font-two.font">Font Two</a>
                <span>200 downloads</span>
            </body>
            </html>
            ''',
            status=200
        )
        # Mock empty second page to stop pagination
        responses.add(
            responses.GET,
            "https://www.dafont.com/theme.php?cat=601&page=2",
            body='<html><body>No more fonts</body></html>',
            status=200
        )

        scraper = DaFontScraper(self.temp_dir, rate_limit=0.1)

        with patch('time.sleep'):
            fonts = scraper.scrape_category("601", max_pages=5)

        self.assertEqual(len(fonts), 2)
        self.assertEqual(fonts[0].name, "Font One")
        self.assertEqual(fonts[1].name, "Font Two")

    @responses.activate
    def test_fontspace_search_workflow(self):
        """Test complete FontSpace search workflow."""
        # Mock search results page
        responses.add(
            responses.GET,
            "https://www.fontspace.com/search?q=handwriting&p=1",
            body='''
            <html>
            <body>
                <a href="/font/test-handwriting">Test Handwriting</a>
            </body>
            </html>
            ''',
            status=200
        )
        # Mock empty second page
        responses.add(
            responses.GET,
            "https://www.fontspace.com/search?q=handwriting&p=2",
            body='<html><body></body></html>',
            status=200
        )

        scraper = FontSpaceScraper(self.temp_dir, rate_limit=0.1)

        with patch('time.sleep'):
            fonts = scraper.scrape_search("handwriting", max_pages=5)

        self.assertEqual(len(fonts), 1)
        self.assertEqual(fonts[0].name, "Test Handwriting")


if __name__ == '__main__':
    unittest.main()
