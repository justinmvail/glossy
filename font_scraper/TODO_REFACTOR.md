# TODO Refactor

Codebase audit findings and refactoring opportunities.

Generated: 2026-02-11

---

## Ordered Refactoring List

1. [x] `run_prefilters.py` - `run_pipeline()` 273 lines → split into stage handlers
2. [x] `run_ocr_prefilter.py` - `run_prefilter()` 197 lines → extract validation logic
3. [x] `run_ocr_prefilter.py` - `run_batch_ocr()` 140 lines → extract batch helpers
4. [x] `stroke_pipeline_stream.py` - `stream_minimal_strokes()` 131 lines → extract phases
5. [x] `inksight_vectorizer.py` - `main()` 125 lines → extract CLI subcommands
6. [x] `stroke_routes_stream.py` - `optimize_stream_generator()` 117 lines → extract phases
7. [ ] `stroke_routes_stream.py` - `api_minimal_strokes_stream()` 102 lines → extract helpers
8. [ ] `setup_database.py:256` - SQL injection risk → validate table names
9. [ ] `stroke_flask` -> `stroke_rendering` circular import → review/fix
10. [ ] Add docstrings to 53 functions (Shape subclasses, strategies)
11. [ ] Add return type hints to 37 functions
12. [ ] Extract magic numbers to constants
13. [ ] Add logging to `stroke_merge.py`, `stroke_scoring.py`
14. [ ] Review `pass` in exception handlers
15. [ ] Split `inksight_vectorizer.py` into 3 modules
16. [ ] Add unit tests for design pattern classes

---

## Details

### 1. Long Functions (76 functions over 60 lines)

Functions that should be broken down into smaller pieces:

| File | Function | Lines | Action |
|------|----------|-------|--------|
| `run_prefilters.py:333` | `run_pipeline()` | 84 | ✓ DONE - split into 4 helpers |
| `run_ocr_prefilter.py:513` | `run_prefilter()` | 90 | ✓ DONE - split into 4 helpers |
| `run_ocr_prefilter.py:309` | `run_batch_ocr()` | 57 | ✓ DONE - extracted script + 2 helpers |
| `stroke_pipeline_stream.py:516` | `stream_minimal_strokes()` | 107 | ✓ DONE - extracted 2 phase helpers |
| `inksight_vectorizer.py:1349` | `main()` | 24 | ✓ DONE - extracted 3 command handlers |
| `stroke_routes_stream.py:476` | `optimize_stream_generator()` | 90 | ✓ DONE - extracted 3 helpers |
| `stroke_routes_stream.py:616` | `api_minimal_strokes_stream()` | 102 | Extract SSE helpers |

**Approach:** Extract helper functions, use strategy pattern for phases.

### 2. SQL Injection Risk

| File | Line | Issue |
|------|------|-------|
| `setup_database.py:256` | `cursor.execute(f"SELECT COUNT(*) FROM {table_name}")` | f-string in SQL |

**Fix:** Validate `table_name` against allowlist of known tables before executing.

### 3. Potential Circular Imports

| Modules | Status |
|---------|--------|
| `stroke_core` -> `stroke_pipeline` | Handled (lazy import) |
| `stroke_flask` -> `stroke_rendering` | **Needs review** |

**Action:** Verify `stroke_flask` -> `stroke_rendering` doesn't cause issues at import time.

---

## Priority: MEDIUM

### 4. Missing Docstrings (53 functions)

Most are Shape subclass methods that inherit behavior from base class:

| File | Functions Missing Docstrings |
|------|------------------------------|
| `stroke_shapes.py` | `generate()`, `get_bounds()`, `param_count()` on all 7 Shape subclasses (21 functions) |
| `stroke_merge.py` | Strategy class methods |
| `template_morph.py` | VertexFinder subclass methods |

**Action:** Add brief docstrings or document "See base class" pattern.

### 5. Type Hints Coverage

- **With return types:** 129 functions (77%)
- **Without return types:** 37 functions (23%)

Files needing type hints:
- `dafont_scraper.py`
- `fontspace_scraper.py`
- `google_fonts_scraper.py`
- `run_*.py` scripts
- `test_*.py` files

**Action:** Add return type hints to remaining public functions.

### 6. Deep Nesting (465 instances of 4+ indentation levels)

Worst offenders:

| File | Occurrences |
|------|-------------|
| `stroke_lib/analysis/skeleton.py` | 76 |
| `stroke_merge.py` | 52 |
| `stroke_pipeline.py` | 46 |
| `inksight_vectorizer.py` | 30 |

**Action:** Use early returns, extract helper functions, consider state machines.

### 7. Large Files (potential god modules)

| File | Lines | Concern |
|------|-------|---------|
| `stroke_pipeline.py` | 1364 | Multiple responsibilities |
| `inksight_vectorizer.py` | 1361 | CLI + vectorizer + OCR validator |
| `stroke_merge.py` | 1221 | Many merge strategies (OK - cohesive) |
| `stroke_routes_core.py` | 1115 | Many routes (OK - Flask pattern) |
| `stroke_routes_batch.py` | 1018 | Many routes (OK - Flask pattern) |

**Action:** Consider splitting `inksight_vectorizer.py` into separate modules:
- `inksight_vectorizer.py` - core vectorization
- `ocr_validator.py` - OCR validation
- `inksight_cli.py` - CLI interface

### 8. Magic Numbers

Found hardcoded values that could be constants:

| File | Value | Context |
|------|-------|---------|
| `run_ocr_prefilter.py:66` | `font_size = 48` | Rendering config |
| `run_ocr_prefilter.py:67` | `padding = 20` | Rendering config |
| `stroke_skeleton.py:602` | `max_steps = 500` | Path tracing limit |
| `stroke_scoring.py:716` | `radius = 6.0` | Stroke half-width |
| `stroke_rendering.py:253` | `norm_size = 64` | Normalization size |
| `stroke_rendering.py:325` | `canvas = 400` | Canvas size |
| `template_morph.py:864-867` | `cols=6, rows=5, cell=300` | Grid layout |

**Action:** Extract to named constants at module level or config dataclass.

---

## Priority: LOW

### 9. Logging Coverage

Only 20 logging statements across 6 files:

| File | Statements |
|------|------------|
| `stroke_lib/api/services.py` | 8 |
| `stroke_core.py` | 5 |
| `stroke_routes_stream.py` | 4 |
| Others | 3 |

**Modules needing logging:**
- `stroke_merge.py` - merge operations
- `stroke_scoring.py` - scoring calculations
- `stroke_pipeline.py` - pipeline stages
- Scrapers - download progress

### 10. Empty Pass Statements

18 `pass` statements found. Review for incomplete implementations:

| File | Line | Context |
|------|------|---------|
| `stroke_shapes.py:78,87,93` | Abstract method stubs (OK) |
| `stroke_affine.py:153` | Abstract method stub (OK) |
| `stroke_scoring.py:164` | Abstract method stub (OK) |
| `stroke_merge.py:118` | Abstract method stub (OK) |
| `template_morph.py:165` | Abstract method stub (OK) |
| `font_utils.py:166,278,305` | Exception handlers - **review** |
| `stroke_flask.py:230` | Exception handler - **review** |
| `google_fonts_scraper.py:206` | Exception handler - **review** |
| `fontspace_scraper.py:402` | Exception handler - **review** |

**Action:** Review exception handlers with `pass` - should they log?

### 11. Test Coverage Gaps

Current test files:
- `test_flask_routes.py` - 24 integration tests
- `test_minimal_strokes.py` - 26 character quality tests
- `test_ocr_single.py` - OCR testing
- `test_inksight_timing.py` - Performance testing

**Missing test coverage:**
- Unit tests for design pattern classes (Shape, Strategy, Penalty)
- Unit tests for `stroke_merge.py` merge strategies
- Unit tests for `stroke_scoring.py` scoring penalties
- Integration tests for scrapers

---

## Completed (from previous audit)

- [x] Dead code removal (414 lines)
- [x] Route helper extraction (3 helpers)
- [x] Design patterns implementation (6 patterns)
- [x] Backwards-compat wrapper removal (276 lines)
- [x] Bare `except:` clauses fixed
- [x] `shell=True` removed
- [x] SQL parameterization (mostly complete)
- [x] Type hints on route functions
- [x] Ruff linting configured
- [x] Requirements.txt with version constraints

---

## Refactoring Checklist

When addressing items above:

1. [ ] Create branch for refactoring
2. [ ] Run tests before changes: `python3 -m unittest test_flask_routes && python3 test_minimal_strokes.py`
3. [ ] Make incremental changes
4. [ ] Run tests after each change
5. [ ] Commit with descriptive messages
6. [ ] Update CLAUDE.md if patterns change

---

## Quick Wins (< 30 min each)

1. Add docstrings to Shape subclass methods
2. Extract magic numbers to constants in `stroke_rendering.py`
3. Validate table names in `setup_database.py`
4. Add logging to `stroke_merge.py`
5. Review `pass` in exception handlers
