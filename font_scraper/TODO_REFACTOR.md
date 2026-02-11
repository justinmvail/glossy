# TODO Refactor

Complete codebase audit findings and refactoring opportunities.

Generated: 2026-02-11

---

## Ordered Refactoring List

### CRITICAL - Security & Reliability
1. [x] Add input validation bounds for text parameter in stroke_routes_core.py:661
2. [x] Add input validation bounds for limit parameter in stroke_routes_batch.py:199
3. [x] Fix database connection handling in run_ocr_prefilter.py (use context managers)
4. [x] Add thread-safe initialization to stroke_routes_batch.py _diffvg (use Lock)
5. [x] Fix race condition in ocr_validator.py _worker_process singleton

### CRITICAL - Performance O(n²) Algorithms
6. [x] stroke_merge.py:985-992 - absorb_convergence_stubs uses O(n²) nested loops
7. [x] stroke_merge.py:1059-1075 - absorb_junction_stubs has triple nested loops
8. [x] stroke_core.py:330-339 - _merge_to_expected_count brute force pair search
9. [x] stroke_merge.py:452-490 - endpoint_cluster() called 100+ times without caching

### CRITICAL - God Modules (>1000 lines, mixed concerns)
10. [x] Split stroke_pipeline.py (1364 lines) into pipeline + variant evaluator + path tracing
11. [x] Split stroke_merge.py (1234 lines) into strategies + utilities + legacy
12. [x] Split stroke_routes_core.py (1115 lines) into routes + handlers + services
13. [x] Split stroke_routes_batch.py (1018 lines) into routes + handlers + services

### HIGH - Architecture Issues
14. [x] stroke_core.py has 10 internal dependencies - create StrokeProcessor service
15. [x] Direct DB calls scattered - create CharacterRepository
16. [x] Rendering logic scattered - consolidate into GlyphRenderer abstraction
17. [x] Inconsistent scraper interfaces - create common FontSource base class
18. [x] Multiple template systems - unify NUMPAD_TEMPLATES and template_morph.py

### HIGH - Code Quality
19. [ ] stroke_merge.py - Extract magic thresholds (18, 20, 15, 25) to constants
20. [ ] font_utils.py - Extract PHASH_FONT_SIZE, PHASH_IMG_WIDTH constants
21. [ ] stroke_utils.py - Extract SNAP_DEEP_INSIDE_MIN_DEPTH, SNAP_MARGIN constants
22. [ ] stroke_rendering.py - Extract SMALLCAPS_MIN_SIZE = 5 constant
23. [ ] Replace broad `except Exception:` in font_utils.py (8 occurrences)

### HIGH - Testing Gaps
24. [ ] Add unit tests for stroke_pipeline.py (1364 LOC, 0 tests)
25. [ ] Add unit tests for stroke_merge.py merge strategies
26. [ ] Add unit tests for stroke_scoring.py penalties
27. [ ] Add integration tests for scrapers (dafont, fontspace, google)
28. [ ] Add isolated database tests for FontRepository, TestRunRepository

### MEDIUM - Performance Caching
29. [ ] stroke_utils.py:458-469 - find_skeleton_waypoints: use KD-tree instead of min()
30. [ ] stroke_rendering.py:64 - Increase LRU cache from 32 to 256
31. [ ] stroke_merge.py - Cache endpoint_cluster results in absorb functions

### MEDIUM - Code Duplication
32. [ ] Extract bbox_width_height() utility (appears 5+ times)
33. [ ] Extract point_distance_squared() utility (appears 6+ times)
34. [ ] Extract render_text_on_image() utility (appears 4+ times)
35. [ ] Extract binarize_image() utility (appears 6+ times)

### MEDIUM - Deep Nesting (4+ levels)
36. [ ] stroke_merge.py:551-610 - _find_best_merge_pair has 6 nesting levels
37. [ ] stroke_merge.py:958-1013 - absorb_convergence_stubs has 7 nesting levels
38. [ ] stroke_merge.py:1038-1087 - absorb_junction_stubs similar pattern

### MEDIUM - Long Functions (>50 lines)
39. [ ] stroke_merge.py - absorb_convergence_stubs (86 lines) - extract helpers
40. [ ] stroke_merge.py - absorb_junction_stubs (72 lines) - extract helpers
41. [ ] stroke_rendering.py - check_case_mismatch (84 lines) - extract helpers

### MEDIUM - Type Hints
42. [ ] stroke_utils.py - Add precise tuple types to geometry functions
43. [ ] stroke_utils.py:334 - build_guide_path needs precise list/tuple types

### MEDIUM - Documentation Gaps
44. [ ] Add module docstrings to stroke_editor.py, stroke_templates.py
45. [ ] Add module docstrings to stroke_shape_templates.py
46. [ ] Document stroke_lib/optimization/*.py (minimal docstrings)
47. [ ] Document stroke_lib/analysis/segments.py

### LOW - Memory/Data Structures
48. [ ] stroke_merge.py:452-490 - Use dict for point_to_cluster instead of list[set]
49. [ ] stroke_merge.py:633-630 - Use deque instead of list for O(1) pop
50. [ ] stroke_merge.py:1155-1160 - Avoid repeated list allocation in merge

### LOW - Code Style
51. [ ] Standardize logger naming (_logger vs logger)
52. [ ] Standardize boolean prefixes (is_ prefix for predicates)
53. [ ] Replace print() with logger in dafont_scraper.py

### LOW - Deprecated Code
54. [ ] stroke_scoring.py - Remove deprecated _compute_*_penalty functions

---

## Details

### 1-5. Security & Reliability Issues

| File | Line | Issue | Fix |
|------|------|-------|-----|
| stroke_routes_core.py | 661 | Text parameter no length limit | Add `if len(txt) > 500: return error` |
| stroke_routes_batch.py | 199 | Limit parameter unbounded | Add `if limit < 1 or limit > 100: limit = 10` |
| run_ocr_prefilter.py | 205, 533 | Manual `conn.close()` not in context manager | Use `with sqlite3.connect() as conn:` |
| stroke_routes_batch.py | 64-82 | Race condition on `_diffvg` initialization | Use `threading.Lock` |
| ocr_validator.py | 234-236 | TOCTOU race on `_worker_process` | Use lock for singleton check |

### 6-9. Performance O(n²) Issues

**stroke_merge.py:985-992 - absorb_convergence_stubs**
```python
# Current: O(n²) - nested loop checking all strokes
for sj in range(len(strokes)):  # For each stroke
    if endpoint_cluster(strokes[sj], False, assigned) == cluster_id:
        others_at_cluster += 1
```
**Fix:** Build cluster index once with `_build_cluster_index()`, get O(1) lookups.

**stroke_core.py:330-339 - _merge_to_expected_count**
```python
# Current: O(k * n²) - checks all pairs of strokes
for i in range(len(strokes)):
    for j in range(i + 1, len(strokes)):
        # Check 4 endpoint combinations
```
**Fix:** Use KD-tree for nearest endpoint pairs in O(n log n).

### 10-13. God Module Splits

**stroke_pipeline.py (1364 lines) → 3 modules:**
- `stroke_pipeline.py` (400 lines) - MinimalStrokePipeline, PipelineConfig, PipelineFactory
- `stroke_pipeline_core.py` (300 lines) - Path tracing, waypoint resolution
- `stroke_variant_evaluator.py` (250 lines) - Variant evaluation, scoring

**stroke_merge.py (1234 lines) → 4 modules:**
- `stroke_merge.py` (300 lines) - MergePipeline, MergeContext, MergeStrategy base
- `stroke_merge_strategies.py` (400 lines) - DirectionMerge, TJunction, StubAbsorption
- `stroke_merge_utils.py` (300 lines) - Geometry helpers, distance calculations
- `stroke_merge_legacy.py` (200 lines) - Legacy function API (deprecated)

**stroke_routes_core.py (1115 lines) → 3 modules:**
- `stroke_routes_core.py` (400 lines) - @app.route decorators only
- `stroke_services_core.py` (300 lines) - CharacterService, RenderingService
- `stroke_handlers_core.py` (350 lines) - Business logic handlers

### 14-18. Architecture Refactoring

**Create StrokeProcessor service (item 14):**
```python
@dataclass
class StrokeProcessorDependencies:
    render_glyph_mask: Callable
    find_skeleton_segments: Callable
    merge_pipeline: MergePipeline
    scorer: CompositeScorer

class StrokeProcessor:
    def __init__(self, deps: StrokeProcessorDependencies):
        self.deps = deps
```

**Unify scraper interfaces (item 17):**
```python
class FontSource(ABC):
    @abstractmethod
    def scrape_fonts(self, config: ScraperConfig) -> List[FontMetadata]

@dataclass
class ScraperConfig:
    max_fonts: int = None
    max_pages: int = None
    rate_limit: float = 1.0
```

### 24-28. Testing Gaps

| Component | LOC | Tests | Priority |
|-----------|-----|-------|----------|
| stroke_pipeline.py | 1364 | 0 | CRITICAL |
| stroke_merge.py | 1234 | 0 | CRITICAL |
| stroke_scoring.py | 734 | 0 | CRITICAL |
| dafont_scraper.py | 512 | 0 | HIGH |
| fontspace_scraper.py | 540 | 0 | HIGH |
| FontRepository | 150 | 0 | HIGH |

### 32-35. Code Duplication

**bbox_width_height() - appears 5+ times:**
```python
# Current (repeated in font_utils.py, stroke_utils.py, stroke_rendering.py)
w = bbox[2] - bbox[0]
h = bbox[3] - bbox[1]

# Extract to stroke_utils.py:
def bbox_width_height(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return bbox[2] - bbox[0], bbox[3] - bbox[1]
```

**point_distance_squared() - appears 6+ times:**
```python
# Current (repeated in stroke_merge.py, stroke_utils.py)
d = (end_i[0] - end_j[0])**2 + (end_i[1] - end_j[1])**2

# Extract to stroke_utils.py:
def point_distance_squared(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
```

---

## Summary

| Category | Count | Severity |
|----------|-------|----------|
| Security & Reliability | 5 | CRITICAL |
| Performance O(n²) | 4 | CRITICAL |
| God Modules | 4 | CRITICAL |
| Architecture | 5 | HIGH |
| Code Quality | 5 | HIGH |
| Testing Gaps | 5 | HIGH |
| Performance Caching | 3 | MEDIUM |
| Code Duplication | 4 | MEDIUM |
| Deep Nesting | 3 | MEDIUM |
| Long Functions | 3 | MEDIUM |
| Type Hints | 2 | MEDIUM |
| Documentation | 4 | MEDIUM |
| Memory/Data Structures | 3 | LOW |
| Code Style | 3 | LOW |
| Deprecated Code | 1 | LOW |
| **TOTAL** | **54** | |

---

## Verification Commands

```bash
# Run all tests after changes
python3 -m unittest test_flask_routes test_design_patterns -v
python3 test_minimal_strokes.py

# Check for common issues
grep -r "except:" *.py --include="*.py" | grep -v "except Exception"
grep -rn "shell=True" *.py
python3 -m py_compile *.py
```

---

## Completed

- [x] Previous audit items 1-16 (see git history)
