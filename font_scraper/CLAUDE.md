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
| `stroke_merge.py` | Stroke merging via MergePipeline (Strategy pattern) |
| `stroke_scoring.py` | Coverage scoring via CompositeScorer (Composite pattern) |
| `stroke_affine.py` | Optimization via OptimizationStrategy (Strategy pattern) |
| `stroke_shapes.py` | Shape classes via SHAPES registry (Polymorphism) |
| `stroke_contour.py` | Contour extraction utilities |

### Supporting Code

| Module | Purpose |
|--------|---------|
| `stroke_lib/` | Reusable stroke library (analysis, templates, optimization) |
| `font_utils.py` | Font utilities (dedup, cursive detection, rendering) |
| `template_morph.py` | Character vertex detection via VERTEX_FINDERS registry |
| `db_schema.py` | Database schema and migrations |

## Design Patterns

The codebase uses several design patterns to maximize extensibility:

### Shape Classes (Polymorphism) - `stroke_shapes.py`

```python
from stroke_shapes import SHAPES, Shape

# Use the registry
shape = SHAPES['arc_right']
points = shape.generate(params, bbox)
bounds = shape.get_bounds()

# Add new shape: subclass Shape, register in SHAPES
class MyShape(Shape):
    def generate(self, params, bbox, offset=(0,0), n_pts=60): ...
    def get_bounds(self): return [(0, 1), (0, 1)]
    @property
    def param_count(self): return 2
```

### Optimization Strategy - `stroke_affine.py`

```python
from stroke_affine import (
    OptimizationStrategy, NelderMeadStrategy,
    DifferentialEvolutionStrategy, ChainedStrategy,
    OptimizationConfig, create_default_affine_strategy
)

# Use default chained strategy
strategy = create_default_affine_strategy()
best_params, best_score = strategy.optimize(objective_fn, initial, bounds)

# Or customize
config = OptimizationConfig(max_iterations=200, tolerance=0.0001)
strategy = ChainedStrategy([
    NelderMeadStrategy(config),
    DifferentialEvolutionStrategy(config)
])
```

### Composite Scoring - `stroke_scoring.py`

```python
from stroke_scoring import (
    CompositeScorer, ScoringPenalty,
    SnapPenalty, EdgePenalty, OverlapPenalty
)

# Default scorer
scorer = CompositeScorer()
score = scorer.score(stroke_arrays, context)

# Custom penalties
scorer = CompositeScorer(penalties=[
    SnapPenalty(weight=0.8),
    EdgePenalty(weight=0.2),
    OverlapPenalty(weight=0.5),
])
```

### Vertex Finder Registry - `template_morph.py`

```python
from template_morph import VERTEX_FINDERS, VertexFinder, find_vertices

# Use the registry (automatic via find_vertices)
vertices = find_vertices('A', font_mask, bbox)

# Add new character: subclass VertexFinder, register
class VertexFinderX(VertexFinder):
    def find(self, font_mask, bbox, outline_xy): ...

VERTEX_FINDERS['X'] = VertexFinderX()
```

### Merge Strategy Pipeline - `stroke_merge.py`

```python
from stroke_merge import (
    MergePipeline, MergeStrategy, MergeContext,
    DirectionMergeStrategy, TJunctionMergeStrategy,
    StubAbsorptionStrategy, OrphanRemovalStrategy
)

# Use default pipeline
pipeline = MergePipeline.create_default()
strokes = pipeline.run(strokes, junction_clusters, assigned)

# Or use preset configurations
pipeline = MergePipeline.create_aggressive()  # More merging
pipeline = MergePipeline.create_conservative()  # Less merging

# Or customize
pipeline = MergePipeline([
    DirectionMergeStrategy(max_angle=np.pi/6),
    TJunctionMergeStrategy(),
    StubAbsorptionStrategy(conv_threshold=15),
])
```

### Pipeline Factory - `stroke_pipeline.py`

```python
from stroke_pipeline import PipelineFactory, PipelineConfig

# Named configurations
pipeline = PipelineFactory.create_default(font_path, char)
pipeline = PipelineFactory.create_fast(font_path, char)      # 128px, speed
pipeline = PipelineFactory.create_high_quality(font_path, char)  # 448px, quality

# Custom configuration
config = PipelineConfig(canvas_size=300, resample_points=40)
pipeline = PipelineFactory.create_with_config(font_path, char, config)
```

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
2. Use design patterns for extensibility (see Design Patterns section):
   - New shape type: subclass `Shape` in `stroke_shapes.py`
   - New scoring penalty: subclass `ScoringPenalty` in `stroke_scoring.py`
   - New merge strategy: subclass `MergeStrategy` in `stroke_merge.py`
   - New optimization algorithm: subclass `OptimizationStrategy` in `stroke_affine.py`
3. Run `python3 test_minimal_strokes.py` to verify quality metrics

### Update database schema
1. Add migration to `db_schema.py`
2. Update `setup_database.py` SCHEMA constant
3. Run `python3 setup_database.py --force` to recreate (dev only)

## File Locations

- Templates: `templates/*.html` (font_list, char_grid, editor)
- Static: `static/` (CSS, JS)
- Fonts: `fonts/` subdirectories by source
- Database: `fonts.db` (SQLite)

## Complete Codebase Audit

When asked to perform a "complete audit", follow this systematic checklist:

### 1. Code Quality

| Check | Command/Method | Fix |
|-------|----------------|-----|
| Dead code | `Grep` for unused functions, check imports | Remove unused code |
| Code duplication | Look for repeated patterns across files | Extract to shared helpers |
| Long functions | Functions > 50 lines | Extract helpers, reduce nesting |
| Deep nesting | 4+ levels of indentation | Early returns, extract functions |
| Magic numbers | Hardcoded literals | Define as named constants |
| Missing type hints | Public functions without annotations | Add type hints |
| Inconsistent naming | Mixed conventions | Standardize to snake_case |

### 2. Architecture

| Check | Command/Method | Fix |
|-------|----------------|-----|
| Circular imports | Import errors, `Grep` for cross-imports | Lazy imports, restructure |
| God modules | Files > 500 lines with mixed concerns | Split into focused modules |
| Tight coupling | Excessive cross-module imports | Introduce interfaces/abstractions |
| Missing abstractions | Direct DB/API calls scattered | Repository pattern, service layer |
| Inconsistent patterns | Mixed approaches to same problem | Standardize on one pattern |

### 3. Design Patterns

| Check | Command/Method | Fix |
|-------|----------------|-----|
| Dict dispatch | `THING_FNS = {'name': fn}` patterns | Polymorphism (base class + registry) |
| if/elif chains | Character/type-specific branches | Strategy pattern or registry |
| Hardcoded algorithms | Can't swap implementations | Strategy pattern |
| Scattered config | Constants across multiple files | Config dataclasses, factories |
| Procedural pipelines | Sequential function calls | Pipeline/chain pattern |

### 4. Performance

| Check | Command/Method | Fix |
|-------|----------------|-----|
| O(n²) algorithms | Nested loops over same data | Build indexes, use sets/dicts |
| Missing caching | Repeated expensive operations | LRU cache, memoization |
| N+1 queries | Loop with DB query inside | Batch queries, joins |
| Blocking I/O | Synchronous file/network ops | Async or background tasks |
| Memory leaks | Unbounded caches, reference cycles | Size limits, weak refs |

### 5. Reliability

| Check | Command/Method | Fix |
|-------|----------------|-----|
| Bare except | `except:` without exception type | Specific exception types |
| Silent failures | Errors swallowed without logging | Add logging or re-raise |
| Missing validation | User input used directly | Validate at boundaries |
| Resource leaks | Files/connections not closed | Context managers (`with`) |
| Race conditions | Shared mutable state | Locks or immutable patterns |

### 6. Security

| Check | Command/Method | Fix |
|-------|----------------|-----|
| SQL injection | String formatting in queries | Parameterized queries (`?`) |
| Command injection | `shell=True`, string commands | List args, no shell |
| Path traversal | User input in file paths | Validate, use safe joins |
| Hardcoded secrets | Passwords/keys in code | Environment variables |
| Error leaks | `str(e)` in HTTP responses | Generic error messages |

### 7. Testing

| Check | Command/Method | Fix |
|-------|----------------|-----|
| Missing tests | Critical paths untested | Add unit/integration tests |
| Flaky tests | Non-deterministic failures | Fix race conditions, mock time |
| Slow tests | Test suite > 30s | Parallelize, mock I/O |
| No integration tests | Only unit tests | Add end-to-end tests |
| Untestable code | Tightly coupled, global state | Dependency injection |

### 8. Documentation

| Check | Command/Method | Fix |
|-------|----------------|-----|
| Missing docstrings | Public functions undocumented | Add Google-style docstrings |
| Outdated docs | Docs don't match code | Update or remove |
| No README | Missing setup instructions | Add README.md |
| Tribal knowledge | Undocumented decisions | Add comments or ADRs |

### 9. Dependencies

| Check | Command/Method | Fix |
|-------|----------------|-----|
| Outdated packages | `pip list --outdated` | Update with testing |
| Unused deps | Imports not used | Remove from requirements |
| Unpinned versions | No version constraints | Add `>=` constraints |
| Security vulns | `pip-audit` or manual check | Update affected packages |

### 10. Developer Experience

| Check | Command/Method | Fix |
|-------|----------------|-----|
| Complex setup | Many manual steps to run | Add setup scripts |
| No linting | Inconsistent style | Add ruff/pyproject.toml |
| Missing dev tools | Hard to debug | Add logging, visualization |

### Audit Workflow

1. **Scan first**: Use Grep/Glob to identify issues before fixing
2. **Prioritize**: Security > Reliability > Performance > Code Quality
3. **Test after each fix**: Run `test_flask_routes` and `test_minimal_strokes.py`
4. **Commit incrementally**: One commit per category or major fix
5. **Update docs**: Keep CLAUDE.md current with patterns used

### Verification Commands

```bash
# Syntax check all Python files
python3 -m py_compile *.py

# Run all tests
python3 -m unittest test_flask_routes -v
python3 test_minimal_strokes.py

# Check for common issues
grep -r "except:" *.py           # Bare excepts
grep -r "shell=True" *.py        # Command injection risk
grep -r "% s" *.py               # SQL injection risk (string formatting)
```
