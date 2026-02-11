# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/claude-code) when working with this codebase.

## Project Overview

Font scraper and stroke editor for generating training data for SDT (Stroke Diffusion Transformer) character models. The system:
1. Scrapes handwriting fonts from DaFont, FontSpace, and Google Fonts
2. Filters fonts (completeness, duplicates, cursive detection, OCR validation)
3. Extracts stroke paths from font glyphs via skeleton analysis
4. Provides a web-based editor for reviewing and correcting strokes
5. Stores results in SQLite for training pipeline consumption

## Quick Start

```bash
# Initialize database (creates fonts.db with all tables)
python3 setup_database.py

# Run the stroke editor web UI
python3 stroke_editor.py
# Open http://localhost:5000

# Run tests
python3 -m unittest test_flask_routes -v
python3 test_minimal_strokes.py
```

## Architecture

### Core Modules

| Module | Purpose |
|--------|---------|
| `stroke_editor.py` | Flask app entry point (77 lines) |
| `stroke_flask.py` | Flask config, DB helpers, route utilities |
| `stroke_routes_core.py` | Main API routes (render, save, process) |
| `stroke_routes_batch.py` | Batch processing routes |
| `stroke_routes_stream.py` | SSE streaming routes for optimization |
| `stroke_core.py` | Core stroke API: `min_strokes()`, `auto_fit()`, `skel_strokes()` |
| `stroke_pipeline.py` | MinimalStrokePipeline orchestration |
| `stroke_pipeline_stream.py` | Streaming version with SSE events |

### Stroke Processing

| Module | Purpose |
|--------|---------|
| `stroke_rendering.py` | Glyph mask rendering with caching |
| `stroke_skeleton.py` | Skeletonization wrapper |
| `stroke_merge.py` | Stroke merging and stub absorption |
| `stroke_scoring.py` | Coverage and topology scoring |
| `stroke_affine.py` | Affine transformation optimization |
| `stroke_shapes.py` | Parametric shape fitting |
| `stroke_contour.py` | Contour extraction utilities |

### Supporting Code

| Module | Purpose |
|--------|---------|
| `stroke_lib/` | Reusable stroke library (analysis, templates, optimization) |
| `font_utils.py` | Font utilities (dedup, cursive detection, rendering) |
| `template_morph.py` | Character-specific vertex detection |
| `db_schema.py` | Database schema and migrations |

### Scrapers

| Module | Purpose |
|--------|---------|
| `dafont_scraper.py` | DaFont scraper |
| `fontspace_scraper.py` | FontSpace scraper |
| `google_fonts_scraper.py` | Google Fonts scraper |
| `scrape_all.py` | Run all scrapers in parallel |

## Database

SQLite database at `fonts.db` with tables:
- `fonts` - Font metadata (name, source, file_path)
- `characters` - Per-character stroke data (strokes_raw, strokes_processed, markers)
- `font_checks` - Quality metrics (completeness, cursive, duplicates)
- `font_removals` - Rejected fonts with reasons
- `removal_reasons` - Lookup table for rejection codes
- `test_runs` - Batch test results
- `ocr_runs` - OCR validation history

Use `setup_database.py` to initialize. Use `get_db_context()` from stroke_flask.py for connections.

## Key API Endpoints

```
GET  /                          Font list page
GET  /chars/<font_id>           Character grid page
GET  /api/char/<id>?c=X         Get character data (strokes, markers)
POST /api/save/<id>?c=X         Save character strokes
GET  /api/render/<id>?c=X       Render glyph as PNG
GET  /api/preview/<id>?c=X      Preview strokes overlaid on glyph
POST /api/process/<id>?c=X      Process strokes (skeleton → optimize)
GET  /api/optimize-stream/<id>  SSE stream for real-time optimization
GET  /api/check-connected/<id>  Check font quality metrics
```

## Coding Conventions

- **Style**: snake_case, Google-style docstrings, type hints on public functions
- **Linting**: ruff configured in pyproject.toml (E, F, I, B, C4, UP, SIM)
- **Logging**: Use module-level `logger = logging.getLogger(__name__)`
- **DB access**: Use repository pattern (FontRepository, TestRunRepository)
- **Routes**: Extract common patterns to helpers in stroke_flask.py

## Testing

```bash
# Flask route integration tests (24 tests)
python3 -m unittest test_flask_routes -v

# Stroke quality tests (26 chars across fonts)
python3 test_minimal_strokes.py

# Compare against baseline
python3 test_minimal_strokes.py --compare test_baseline.json
```

## Common Tasks

### Add a new route
1. Add route function to appropriate `stroke_routes_*.py`
2. Use helpers: `get_font_or_error()`, `get_font_and_mask()`, `send_pil_image_as_png()`
3. Add test case to `test_flask_routes.py`

### Modify stroke processing
1. Core logic in `stroke_core.py` or `stroke_pipeline.py`
2. Run `python3 test_minimal_strokes.py` to verify quality metrics

### Update database schema
1. Add migration to `db_schema.py`
2. Update `setup_database.py` SCHEMA constant
3. Run `python3 setup_database.py --force` to recreate (dev only)

## File Locations

- Templates: `templates/*.html` (font_list, char_grid, editor)
- Static: `static/` (CSS, JS)
- Fonts: `fonts/` subdirectories by source
- Database: `fonts.db` (SQLite)
