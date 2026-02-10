# Font Scraper Codebase Refactoring TODO

This document tracks code quality issues identified through a comprehensive review of the glossy/font_scraper codebase. Issues are prioritized and will be checked off as they are fixed.

**Last Updated**: 2026-02-10
**Review Scope**: All stroke_*.py files, stroke_lib package, Flask routes

---

## Summary Statistics

| Category | High | Medium | Low | Total |
|----------|------|--------|-----|-------|
| Long Functions (>50 lines) | 15 | 12 | - | 27 |
| Code Duplication | 8 | 6 | - | 14 |
| Magic Numbers/Strings | 3 | 25 | - | 28 |
| Missing Error Handling | 8 | 10 | - | 18 |
| Missing Type Hints | 5 | 15 | - | 20 |
| Tight Coupling | 6 | 8 | - | 14 |
| Poor Naming | - | 12 | 5 | 17 |
| Testability Issues | 8 | 6 | - | 14 |
| Dead Code | - | 4 | 3 | 7 |
| Inconsistent Patterns | - | 8 | - | 8 |

---

## HIGH PRIORITY

### 1. Long Functions (>80 lines)

- [ ] **stroke_pipeline_stream.py:312** - `stream_minimal_strokes()` - 185 lines
  - Extract phase generators for each processing stage
  - Consider state machine pattern

- [ ] **stroke_routes_stream.py:307** - `optimize_stream_generator()` - 136 lines
  - Already has helpers but main function still too long
  - Extract initialization and finalization phases

- [ ] **stroke_pipeline_stream.py:201** - `_stream_skeleton_evaluation()` - 109 lines
  - Split penalty calculation and comparison logic

- [ ] **stroke_skeleton.py:168** - `find_skeleton_segments()` - 106 lines
  - Extract segment tracing and classification

- [ ] **stroke_scoring.py:46** - `score_all_strokes()` - 107 lines
  - Extract penalty calculations into separate functions

- [ ] **stroke_lib/analysis/segments.py:105** - `find_best_vertical_chain()` - 105 lines
  - Extract graph building, BFS, and scoring phases

- [ ] **stroke_utils.py:322** - `build_guide_path()` - 100 lines
  - Extract nested functions to module level
  - Split waypoint processing from path building

- [ ] **stroke_merge.py:164** - `run_merge_pass()` - 96 lines
  - Extract merge candidate selection logic

- [ ] **stroke_pipeline.py:834** - `_trace_resolved_waypoints()` - 95 lines
  - Split arrival branch management and apex extension

- [ ] **stroke_merge.py:262** - `merge_t_junctions()` - 94 lines
  - Extract junction detection and merge execution

- [ ] **stroke_skeleton.py:306** - `trace_skeleton_path()` - 93 lines
  - Extract neighbor scoring logic

- [ ] **stroke_dataclasses.py:191** - `parse_stroke_template()` - 91 lines
  - Extract regex parsing and waypoint construction

### 2. Critical Code Duplication

- [ ] **stroke_rendering.py:64-78 & 123-138** - Font scaling logic duplicated
  - Create `_scale_font_to_fit(pil_font, char, canvas_size)` helper

- [ ] **stroke_rendering.py:83-87, 143-147, 210-217** - Image centering logic 3x
  - Create `_compute_centered_position(bbox, canvas_size)` helper

- [ ] **stroke_routes_core.py:460-522 & 526-582** - Quality check logic duplicated
  - Extract `_check_font_quality(pil_font)` shared function

- [ ] **stroke_pipeline.py:789-792 & 816-819** - `extract_region()` defined twice
  - Move to module level or class method

- [ ] **stroke_merge.py:469-742** - Four stub absorption functions with similar patterns
  - Create generic `_absorb_stubs_generic()` with strategy pattern

- [ ] **stroke_scoring.py:134 & 228** - `FREE_OVERLAP = 0.25` defined twice
  - Move to module-level constant

### 3. Missing Error Handling

- [ ] **stroke_flask.py:269-290** - `get_font()` doesn't use context manager
  - Connection not closed on fetchone() failure
  - Refactor to use `get_db_context()`

- [ ] **stroke_pipeline.py:382-392** - `resolve_waypoint()` no region validation
  - Add check that `wp.region` is 1-9

- [ ] **stroke_affine.py:320-404** - `optimize_affine()` no exception handling
  - Add try-except for invalid fonts, missing glyphs

- [ ] **stroke_routes_stream.py:217-223, 247-254** - Generic `except Exception`
  - Replace with specific exceptions, add logging

- [ ] **stroke_lib/api/services.py:324-427** - Bare `except Exception` (3 instances)
  - Distinguish between "not found" and "error"
  - Add logging for debugging

- [ ] **stroke_routes_batch.py:706-710** - `ALTER TABLE` silently catches error
  - Validate column was actually added

### 4. Testability Issues

- [ ] **stroke_pipeline.py:115-180** - Constructor requires 11 callback parameters
  - Create factory method with sensible defaults
  - Use dependency injection container or config object

- [ ] **stroke_lib/api/services.py:306-427** - `FontService` hardcodes sqlite3.connect()
  - Accept connection/factory as parameter
  - Enable mock database for testing

- [ ] **stroke_routes_core.py:93-133** - Routes directly query database
  - Extract data access layer for testability

- [ ] **stroke_scoring.py:46-152** - `score_all_strokes()` takes 10 parameters
  - Bundle into `ScoringContext` dataclass

- [ ] **stroke_lib/analysis/skeleton.py:552-605** - Complex nested functions
  - Promote to class methods for direct testing

### 5. Tight Coupling

- [ ] **stroke_lib/utils/rendering.py:49** - Imports from external stroke_rendering
  - Clarify dependency or copy implementation

- [ ] **stroke_pipeline.py:859-928** - `global_traced` set mutated as side effect
  - Return new set instead of mutating parameter

- [ ] **stroke_routes_batch.py:55-59** - Global `_diffvg` variable
  - Use lazy initialization pattern or dependency injection

---

## MEDIUM PRIORITY

### 6. Magic Numbers and Strings

- [ ] **stroke_core.py:86, 90, 129** - `merge_distance=12` undefined constant
  - Define `SKELETON_MERGE_DISTANCE = 12` at module level

- [ ] **stroke_pipeline.py:232** - `60 <= abs(s['angle']) <= 120` magic angle range
  - Define `VERTICAL_ANGLE_MIN = 60`, `VERTICAL_ANGLE_MAX = 120`

- [ ] **stroke_pipeline.py:701** - `waist_tolerance = h * 0.15` magic multiplier
  - Define `WAIST_TOLERANCE_RATIO = 0.15`

- [ ] **stroke_pipeline.py:747** - `waist_margin = h * 0.05` magic multiplier
  - Define `WAIST_MARGIN_RATIO = 0.05`

- [ ] **stroke_pipeline_stream.py:60** - `[:500]` hardcoded skeleton limit
  - Define `MAX_SKELETON_MARKERS = 500`

- [ ] **stroke_routes_core.py:519-520** - Quality thresholds (10, 15, 0.225, 2)
  - Define named constants with documentation

- [ ] **stroke_routes_stream.py:245** - Affine bounds `[(-15, 15), ...]`
  - Define `AFFINE_TRANSLATION_BOUNDS = (-15, 15)` etc.

- [ ] **stroke_routes_stream.py:219** - Optimizer params `maxfev=400, xatol=0.2`
  - Define `NM_MAX_EVALUATIONS = 400` etc.

- [ ] **stroke_rendering.py:68, 128** - `0.9` canvas fill threshold
  - Define `CANVAS_FILL_THRESHOLD = 0.9`

- [ ] **stroke_rendering.py:150, 224** - `128` binarization threshold
  - Define `BINARIZATION_THRESHOLD = 128`

- [ ] **stroke_shapes.py:485-495** - Magic values in adaptive_radius
  - Document why `1.5`, `6.0`, `95th percentile`

- [ ] **stroke_skeleton.py:369-384** - Neighbor scoring weights
  - Define `DIRECTION_BIAS_WEIGHT = 10`, `AVOID_PENALTY = 1000`

- [ ] **stroke_lib/analysis/skeleton.py:221** - `near_vertex_dist = 5`
  - Make configurable parameter

- [ ] **stroke_lib/analysis/skeleton.py:715** - `stub_threshold = 20`
  - Make configurable parameter

### 7. Missing Type Hints

- [ ] **stroke_core.py** - Functions missing type hints:
  - `_skel(mask)` - line 63
  - `skel_markers(mask)` - line 107
  - `skel_strokes(mask, min_len, min_stroke_len)` - line 133
  - `_merge_to_expected_count(strokes, char)` - line 219
  - `min_strokes(fp, c, cs, tpl, ret_var)` - line 278
  - `auto_fit(fp, c, cs, ret_mark)` - line 345

- [ ] **stroke_pipeline.py:115** - Constructor params need type hints
  - Add types for all 13 parameters

- [ ] **stroke_pipeline_stream.py:114-116** - `_stream_variant_strokes()` params
  - Add proper type hints

- [ ] **stroke_utils.py:158-186** - `parse_waypoint()` missing return type
  - Add `-> tuple[int, str]`

- [ ] **stroke_rendering.py:241, 280, 320** - `pil_font` parameter untyped
  - Add `PIL.ImageFont.FreeTypeFont` type

- [ ] **stroke_dataclasses.py:129, 134** - `info: dict`, `skel_tree: object`
  - Use `dict[str, Any]` and proper cKDTree type

### 8. Poor Naming

- [ ] **stroke_pipeline.py:179** - `trace()` → `_trace_segment_from_endpoints()`
- [ ] **stroke_pipeline.py:338** - `dist()` → `_euclidean_distance()`
- [ ] **stroke_pipeline.py:565** - `angle_matches()` → `_angle_in_range()`
- [ ] **stroke_pipeline.py:632** - `region_to_rc()` → `_region_to_row_col()`
- [ ] **stroke_core.py:63** - `_skel()` → `_analyze_skeleton_legacy()`
- [ ] **stroke_core.py:176** - `eps`, `jps`, `jcs` → full names
- [ ] **stroke_routes_core.py:130** - `q` → `query`
- [ ] **stroke_routes_stream.py:127-140** - `cloud_tree`, `snap_yi` → clearer names
- [ ] **stroke_merge.py:219** - `_cid` → `cluster_id`
- [ ] **stroke_lib/analysis/skeleton.py:552** - `trace` → `_trace_path_from_point`
- [ ] **stroke_lib/analysis/skeleton.py:643** - `seg_dir` → `_compute_segment_direction`

### 9. Inconsistent Patterns

- [ ] **stroke_flask.py:148-191 vs 117-145** - `get_db_context()` vs `get_db()`
  - Standardize on context manager everywhere

- [ ] **stroke_pipeline.py vs stroke_pipeline_stream.py** - Analysis initialization
  - Pipeline uses lazy `@property`, stream computes eagerly
  - Document or standardize

- [ ] **stroke_routes_batch.py:701-710** - Uses `get_db()` with manual cleanup
  - Convert to `get_db_context()`

- [ ] **stroke_lib/analysis/skeleton.py** - Returns `List` for endpoints
  - **stroke_lib/domain/skeleton.py** - Uses `Set` for same data
  - Standardize data structures

- [ ] **stroke_routes_stream.py:342 vs 349** - Inconsistent path resolution
  - `min_strokes()` with raw path vs `render_glyph_mask()` with resolved path

### 10. Algorithm Inefficiencies

- [ ] **stroke_skeleton.py:112-165** - `_merge_junction_clusters()` is O(n²)
  - Use spatial index (KD-tree) for distance queries

- [ ] **stroke_merge.py:235-242** - Endpoint cluster lookups repeated 4x per pair
  - Cache cluster assignments

- [ ] **stroke_merge.py:205-259** - While-loop restarts scan after each merge
  - Use queue-based approach for O(n²) instead of O(n³)

- [ ] **stroke_skeleton.py:276-303** - `snap_to_skeleton()` brute-force O(n)
  - Use KD-tree (already available in codebase)

---

## LOW PRIORITY

### 11. Dead Code

- [ ] **stroke_core.py:331** - `adjust_stroke_paths_fn=lambda st, c, m: st` no-op
  - Remove or implement actual functionality

- [ ] **stroke_utils.py:494-507** - Backwards compatibility aliases
  - Remove if not used elsewhere

- [ ] **stroke_lib/analysis/skeleton.py:45** - Unused `cKDTree` import
  - Remove

- [ ] **All stroke_*.py files** - Underscore aliases at EOF
  - Remove `_analyze_skeleton = analyze_skeleton` pattern

### 12. Documentation Gaps

- [ ] **stroke_pipeline.py:630-649** - Direction inference logic not explained
  - Document why diagonal moves return None

- [ ] **stroke_core.py:257-274** - Greedy merge algorithm not documented
  - Explain endpoint combination strategy

- [ ] **stroke_shapes.py:485-495** - `adaptive_radius` magic values
  - Document why 1.5, 6.0, 95th percentile chosen

- [ ] **stroke_lib module docstring** - Missing architecture overview
  - Document relationship with main codebase

### 13. Configuration Issues

- [ ] **stroke_editor.py:71** - Hardcoded `debug=True`, host, port
  - Use environment variables

- [ ] **stroke_affine.py:216-231** - Hardcoded optimizer hyperparameters
  - Make function parameters with defaults

- [ ] **stroke_routes_batch.py:301** - Different STROKE_COLORS than stroke_flask.py
  - Centralize color definitions

---

## Testing Strategy (To Add After Refactoring)

### Unit Tests Needed

- [ ] `stroke_scoring.py` - Test penalty calculations individually
- [ ] `stroke_merge.py` - Test each merge strategy in isolation
- [ ] `stroke_skeleton.py` - Test path tracing with known skeletons
- [ ] `stroke_pipeline.py` - Test waypoint resolution
- [ ] `stroke_lib/analysis/` - Test skeleton analysis components

### Integration Tests Needed

- [ ] End-to-end stroke generation for sample fonts
- [ ] Flask route tests with mock database
- [ ] Streaming endpoint tests

### Test Infrastructure

- [ ] Create fixtures for common test data (masks, skeletons)
- [ ] Mock factory for database connections
- [ ] Test fonts with known expected outputs

---

## Progress Tracking

### Completed
- [x] Initial code review (2026-02-10)
- [x] Created this tracking document

### In Progress
- [ ] None currently

### Blocked
- [ ] None currently

---

## Notes

- Prioritize HIGH items that block testability
- Magic numbers can be fixed incrementally
- Long functions should be split before adding tests
- Consider creating a `constants.py` module for all magic numbers
