# Testing Strategy for font_scraper

## Current State
- **test_flask_routes.py**: 24 integration tests (API endpoints)
- **test_minimal_strokes.py**: 26 quality regression tests (stroke extraction)

## Recommended Test Layers

```
┌─────────────────────────────────────────────────────────────┐
│  E2E Tests (few)        - Full workflow: font → strokes    │
├─────────────────────────────────────────────────────────────┤
│  Integration Tests      - API routes, DB operations        │
├─────────────────────────────────────────────────────────────┤
│  Unit Tests (many)      - Algorithms, utilities, scoring   │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Unit Tests - HIGH VALUE

### stroke_scoring.py - Critical for quality
```python
# Test each penalty class independently
class TestSnapPenalty:
    def test_no_penalty_when_inside_mask(self)
    def test_full_penalty_when_outside_mask(self)
    def test_partial_penalty_at_boundary(self)

class TestCompositeScorer:
    def test_combines_penalties_with_weights(self)
    def test_coverage_calculation(self)
```

### stroke_shapes.py - Each shape generates correct geometry
```python
class TestVLineShape:
    def test_generates_vertical_points(self)
    def test_respects_bbox(self)
    def test_param_bounds_valid(self)

# Parametrized test for all shapes
@pytest.mark.parametrize("shape_name", SHAPES.keys())
def test_shape_generates_valid_points(shape_name):
    shape = SHAPES[shape_name]
    points = shape.generate(params, bbox)
    assert len(points) > 0
    assert all points within bbox
```

### stroke_lib/utils/geometry.py - Pure functions, easy to test
```python
def test_point_distance()
def test_point_distance_squared()
def test_pick_straightest_neighbor()
def test_infer_direction_from_regions()
```

### stroke_merge_strategies.py - Critical merge logic
```python
class TestDirectionMerge:
    def test_merges_collinear_strokes(self)
    def test_rejects_perpendicular_strokes(self)
    def test_respects_max_angle(self)

class TestTJunctionMerge:
    def test_detects_t_junction(self)
    def test_merges_stem_to_crossbar(self)
```

### template_morph.py - Vertex finders
```python
@pytest.mark.parametrize("char", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
def test_vertex_finder_returns_valid_vertices(char):
    vertices = find_vertices(char, mock_mask, bbox)
    assert all vertices within bbox
    assert required vertices present for char
```

---

## 2. Integration Tests - MEDIUM VALUE

### Database operations (stroke_flask.py)
```python
class TestFontRepository:
    def test_get_font_by_id_returns_font(self)
    def test_get_font_by_id_returns_none_for_missing(self)
    def test_reject_fonts_batch_rejects_multiple(self)
    def test_save_character_strokes(self)
```

### API routes - Expand existing test_flask_routes.py
```python
# Add error case coverage
def test_render_char_invalid_char_returns_400(self)
def test_save_char_invalid_json_returns_400(self)
def test_process_empty_strokes_returns_400(self)

# Add SSE streaming tests
def test_optimize_stream_yields_events(self)
def test_minimal_strokes_stream_completes(self)
```

### Scraper retry logic (font_source.py)
```python
class TestRequestWithRetry:
    @responses.activate
    def test_retries_on_500(self)

    @responses.activate
    def test_respects_429_retry_after(self)

    @responses.activate
    def test_gives_up_after_max_retries(self)
```

---

## 3. Regression Tests - HIGH VALUE

### Stroke quality - Already have test_minimal_strokes.py
```python
# Expand to multiple fonts
@pytest.mark.parametrize("font_path", get_test_fonts())
def test_stroke_quality_above_threshold(font_path):
    results = run_minimal_strokes(font_path)
    assert results['combined_score'] >= 0.70
    assert results['pass_rate'] >= 0.90
```

### Visual regression - Golden image comparison
```python
def test_render_char_matches_golden(self):
    img = render_char(font, 'A')
    assert images_similar(img, 'golden/A.png', threshold=0.95)
```

---

## 4. What NOT to Test

| Skip Testing | Reason |
|--------------|--------|
| `stroke_dataclasses.py` | Simple data containers |
| `filter_config.py` | Just constants |
| `stroke_editor.py` | Thin Flask app wrapper |
| `db_schema.py` | Schema definitions |
| Trivial getters/setters | No logic |
| `__init__.py` re-exports | Just imports |
| CLI argument parsing | Argparse does this |

---

## 5. Test Infrastructure

### Fixtures (conftest.py)
```python
@pytest.fixture
def test_font_path():
    return "fonts/test/TestFont.ttf"

@pytest.fixture
def mock_glyph_mask():
    # 64x64 binary mask with letter shape
    return np.array(...)

@pytest.fixture
def sample_strokes():
    return [np.array([[0,0], [10,10], [20,20]])]

@pytest.fixture
def db_session():
    # In-memory SQLite for tests
    conn = sqlite3.connect(':memory:')
    setup_schema(conn)
    yield conn
    conn.close()
```

### Mocking external services
```python
# Use responses library for HTTP mocking
@responses.activate
def test_scraper_handles_404():
    responses.add(responses.GET, "https://dafont.com/...", status=404)
    result = scraper.scrape_font(url)
    assert result is None
```

---

## 6. Coverage Targets

| Module | Target | Rationale |
|--------|--------|-----------|
| stroke_scoring.py | 90% | Critical for quality |
| stroke_shapes.py | 95% | Simple, easy to test |
| stroke_merge*.py | 80% | Complex algorithms |
| stroke_pipeline.py | 70% | Hard to unit test, covered by integration |
| stroke_routes_*.py | 60% | Covered by API tests |
| *_scraper.py | 50% | External dependencies |
| stroke_rendering.py | 40% | Visual output, hard to verify |

**Overall target: 70% line coverage**

---

## 7. Implementation Order

1. **Week 1**: Unit tests for `stroke_scoring.py`, `stroke_shapes.py`, `geometry.py`
2. **Week 2**: Unit tests for `stroke_merge*.py`, `template_morph.py`
3. **Week 3**: Expand integration tests, add DB tests
4. **Week 4**: Scraper tests with mocking, visual regression setup

---

## 8. Test Commands

```bash
# Run all tests with coverage
pytest --cov=. --cov-report=html

# Run fast unit tests only
pytest tests/unit -x

# Run slow integration tests
pytest tests/integration --slow

# Run specific module tests
pytest tests/unit/test_scoring.py -v

# Check coverage for specific file
pytest --cov=stroke_scoring --cov-fail-under=90
```

---

## 9. File Structure

```
font_scraper/
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── unit/
│   │   ├── test_scoring.py
│   │   ├── test_shapes.py
│   │   ├── test_geometry.py
│   │   ├── test_merge.py
│   │   └── test_vertex_finders.py
│   ├── integration/
│   │   ├── test_flask_routes.py  # (move existing)
│   │   ├── test_database.py
│   │   └── test_scrapers.py
│   ├── regression/
│   │   ├── test_minimal_strokes.py  # (move existing)
│   │   └── test_visual.py
│   └── golden/                   # Golden images for visual tests
│       ├── A.png
│       └── ...
```
