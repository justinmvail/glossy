# Font Scrapers

Download handwriting fonts from multiple sources for training data.

## Sources

| Source | Script | Fonts Available |
|--------|--------|-----------------|
| DaFont | `dafont_scraper.py` | ~2000+ handwriting |
| FontSpace | `fontspace_scraper.py` | ~1000+ handwriting |
| Google Fonts | `google_fonts_scraper.py` | ~120 handwriting |

## Quick Start

```bash
# Run all scrapers in parallel
python scrape_all.py --output ./all_fonts --pages 20

# Or run individually
python dafont_scraper.py --output ./dafont --pages 10
python fontspace_scraper.py --output ./fontspace --pages 10
python google_fonts_scraper.py --output ./google
```

## Options

### scrape_all.py
```
--output, -o     Base output directory (default: ./all_fonts)
--pages, -p      Max pages for DaFont/FontSpace (default: 20)
--max-per-source Max fonts per source (default: unlimited)
--sequential     Run one at a time instead of parallel
```

### dafont_scraper.py
```
--output, -o     Output directory
--categories     Category IDs: 601=Calligraphy, 603=Handwritten
--pages, -p      Pages per category (25 fonts/page)
--max-fonts, -m  Limit total fonts
--rate-limit     Seconds between requests (default: 1.0)
```

### fontspace_scraper.py
```
--output, -o     Output directory
--query, -q      Search query (default: handwritten)
--pages, -p      Pages to scrape
--category, -c   Use category instead of search
--max-fonts, -m  Limit total fonts
--rate-limit     Seconds between requests (default: 1.0)
```

### google_fonts_scraper.py
```
--output, -o     Output directory
--max-fonts, -m  Limit fonts (default: all ~120)
--rate-limit     Seconds between requests (default: 0.5)
```

## Output

Each scraper creates:
- Font files (TTF, OTF, WOFF2)
- `*_metadata.json` with font info

## License Notes

- **Google Fonts**: All open source (OFL/Apache)
- **DaFont**: Mixed - check each font's license
- **FontSpace**: Mixed - many free for personal use

For commercial use, filter to fonts with appropriate licenses.

## Docker-based InkSight Vectorization

For GPU-accelerated font vectorization using InkSight, use the Docker containers. This isolates TensorFlow (InkSight) and PyTorch (TrOCR) to avoid dependency conflicts.

### Setup

```bash
# Build both containers (requires NVIDIA Docker runtime)
./docker/build.sh

# Or build individually
docker build -t inksight:latest docker/inksight
docker build -t trocr:latest docker/trocr
```

### Usage

```python
from docker.inksight_docker import InkSightDocker

# Initialize (checks Docker availability)
inksight = InkSightDocker()

# Process single word
result = inksight.process('fonts/dafont/MyFont.ttf', 'hello')
print(f"Strokes: {result['num_strokes']}, Points: {result['total_points']}")

# Save visualization
result = inksight.process('fonts/dafont/MyFont.ttf', 'hello',
                          output_image='output.png')

# Batch process (more efficient for multiple words)
results = inksight.batch_process('fonts/dafont/MyFont.ttf',
                                  ['hello', 'world', 'test'])
```

### Requirements

- Docker with NVIDIA runtime (`nvidia-docker2`)
- GPU with CUDA support
- InkSight model at `/home/server/inksight/model` (or specify custom path)

### Testing

```bash
# Quick test
./docker/run_test.sh

# Or manually
python docker/test_inksight.py
```

## Training Pipeline

For the full font-to-stroke training pipeline (scraping, filtering, vectorization, validation), see **[PIPELINE.md](PIPELINE.md)**.

### Utilities

```python
from font_utils import (
    FontDeduplicator,  # Find duplicate fonts via perceptual hash
    FontScorer,        # Score fonts on quality metrics
    CursiveDetector,   # Detect connected/cursive fonts
    CompletenessChecker,  # Check character coverage
    CharacterRenderer,    # Render individual characters
)

# Quick test on a font
python font_utils.py /path/to/font.ttf
```

### Database

```python
from db_schema import init_db, FontDB

# Initialize
init_db('fonts.db')

# Use helper class
with FontDB('fonts.db') as db:
    font_id = db.add_font('MyFont', '/path/to/font.ttf', source='dafont')
    db.update_checks(font_id, completeness_score=0.95, is_cursive=False)
```
