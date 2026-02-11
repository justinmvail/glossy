# Font Scraper Codebase Refactoring TODO

This document tracks code quality issues identified through a comprehensive review of the glossy/font_scraper codebase. Issues are prioritized and will be checked off as they are fixed.

**Last Updated**: 2026-02-10 (Magic numbers refactoring completed)
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

- [x] **stroke_pipeline_stream.py:312** - `stream_minimal_strokes()` - 185 lines
  - ✅ Extracted `_create_pipeline()`, `_consume_subgenerator()`, `_stream_template_variants()`
  - Reduced to ~60 lines using generator delegation

- [x] **stroke_routes_stream.py:307** - `optimize_stream_generator()` - 136 lines
  - ✅ Extracted `_get_initial_strokes()`, `_determine_completion_reason()`
  - Added PERFECT_SCORE_THRESHOLD, TIME_LIMIT_SECONDS constants

- [x] **stroke_pipeline_stream.py:201** - `_stream_skeleton_evaluation()` - 109 lines
  - ✅ Extracted `_apply_stroke_count_penalty()` and `_make_selection_frame()` helpers

- [x] **stroke_skeleton.py:168** - `find_skeleton_segments()` - 106 lines
  - ✅ Extracted `_build_pixel_to_junction_map()`, `_collect_segment_start_points()`,
    `_trace_segment_from_start()`, `_create_segment_dict()` helpers

- [x] **stroke_scoring.py:46** - `score_all_strokes()` - 107 lines
  - ✅ Extracted `_snap_points_to_mask()`, `_compute_snap_penalty()`, `_compute_edge_penalty()`,
    `_compute_per_shape_coverage()`, `_compute_overlap_penalty()` helpers
  - ✅ Added module-level constants (FREE_OVERLAP, penalty weights, thresholds)

- [x] **stroke_lib/analysis/segments.py:105** - `find_best_vertical_chain()` - 105 lines
  - ✅ Extracted `_build_junction_connectivity()`, `_find_connected_chains()`,
    `_score_chain()` methods

- [x] **stroke_utils.py:322** - `build_guide_path()` - 100 lines
  - ✅ Extracted `_snap_to_skeleton_region()`, `_resolve_waypoint_position()`,
    `_constrain_points_to_mask()` helpers

- [x] **stroke_merge.py:164** - `run_merge_pass()` - 96 lines
  - ✅ Extracted `_build_cluster_endpoint_map()`, `_is_loop_stroke()`,
    `_find_best_merge_pair()`, `_execute_merge()` helpers

- [x] **stroke_pipeline.py:834** - `_trace_resolved_waypoints()` - 95 lines
  - ✅ Extracted `_trace_single_segment()`, `_apply_apex_extensions()` methods

- [x] **stroke_merge.py:262** - `merge_t_junctions()` - 94 lines
  - ✅ Extracted `_find_t_junction_candidate()`, `_remove_short_cross_strokes()` helpers
  - ✅ Reuses `_build_cluster_endpoint_map()` and `_execute_merge()`

- [x] **stroke_skeleton.py:306** - `trace_skeleton_path()` - 93 lines
  - ✅ Extracted `_compute_neighbor_score()`, `_bfs_trace_path()`
  - Added DIRECTION_VECTORS constant, reduced to ~40 lines

- [x] **stroke_dataclasses.py:191** - `parse_stroke_template()` - 91 lines
  - ✅ Extracted `_apply_hint_to_config()`, `_parse_waypoint_item()` helpers
  - ✅ Pre-compiled regex patterns as module constants

### 2. Critical Code Duplication

- [x] **stroke_rendering.py:64-78 & 123-138** - Font scaling logic duplicated
  - ✅ Created `_scale_font_to_fit()` helper with CANVAS_FILL_THRESHOLD constant

- [x] **stroke_rendering.py:83-87, 143-147, 210-217** - Image centering logic 3x
  - ✅ Created `_compute_centered_position(bbox, canvas_size)` helper

- [x] **stroke_routes_core.py:460-522 & 526-582** - Quality check logic duplicated
  - ✅ Extracted `_check_font_quality(pil_font, font_path)` shared function
  - ✅ Added constants: MIN_SHAPE_COUNT, MAX_SHAPE_COUNT, MAX_WIDTH_RATIO

- [x] **stroke_pipeline.py:789-792 & 816-819** - `extract_region()` defined twice
  - ✅ Moved to module level as `extract_region_from_waypoint()`

- [x] **stroke_merge.py:469-742** - Four stub absorption functions with similar patterns
  - ✅ Reviewed: Functions have different core logic (extend/merge/remove)
  - Kept separate for clarity - consolidating would reduce readability

- [x] **stroke_scoring.py:134 & 228** - `FREE_OVERLAP = 0.25` defined twice
  - ✅ Moved to module-level constant with other scoring constants

### 3. Missing Error Handling

- [x] **stroke_flask.py:269-290** - `get_font()` doesn't use context manager
  - ✅ Refactored to use `get_db_context()` with try/except

- [x] **stroke_pipeline.py:382-392** - `resolve_waypoint()` no region validation
  - ✅ Added validation that wp.region is 1-9, raises ValueError if invalid

- [x] **stroke_affine.py:320-404** - `optimize_affine()` no exception handling
  - ✅ Reviewed: Already has proper None checks for failures
  - Inner functions have exception handling, main function returns None on error

- [x] **stroke_routes_stream.py:217-223, 247-254** - Generic `except Exception`
  - ✅ Added logging module and _logger
  - ✅ Catch specific exceptions (ValueError, RuntimeError, LinAlgError) first
  - ✅ Log warnings for expected failures, errors for unexpected ones

- [x] **stroke_lib/api/services.py:324-427** - Bare `except Exception` (3 instances)
  - ✅ Added logging module and _logger
  - ✅ Catch sqlite3.Error and json.JSONDecodeError specifically
  - ✅ Added proper finally blocks for connection cleanup
  - ✅ Log with context (font_id, char) for debugging

- [x] **stroke_routes_batch.py:706-710** - `ALTER TABLE` silently catches error
  - ✅ Check if column exists using PRAGMA table_info before ALTER
  - ✅ Log warning if ALTER still fails (race condition)

### 4. Testability Issues

- [x] **stroke_pipeline.py:115-180** - Constructor requires 11 callback parameters
  - ✅ Added `MinimalStrokePipeline.create_default()` factory method

- [x] **stroke_lib/api/services.py:306-427** - `FontService` hardcodes sqlite3.connect()
  - ✅ Accept optional `connection_factory` parameter for testing
  - ✅ Added `_get_connection()` helper method

- [ ] **stroke_routes_core.py:93-133** - Routes directly query database
  - Extract data access layer for testability (deferred - requires larger refactor)

- [x] **stroke_scoring.py:46-152** - `score_all_strokes()` takes 10 parameters
  - ✅ Created `ScoringContext` dataclass to bundle parameters
  - ✅ Added `score_all_strokes_ctx()` wrapper function

- [x] **stroke_lib/analysis/skeleton.py:552-605** - Complex nested functions
  - ✅ Extracted `_trace_single_path()` and `_pick_straightest_candidate()` as static methods

### 5. Tight Coupling

- [x] **stroke_lib/utils/rendering.py:49** - Imports from external stroke_rendering
  - ✅ Added documentation explaining the dependency and testing strategy

- [x] **stroke_pipeline.py:859-928** - `global_traced` set mutated as side effect
  - ✅ Changed to return tuple (path, new_traced) instead of mutating
  - ✅ Updated callers to handle new return type

- [x] **stroke_routes_batch.py:55-59** - Global `_diffvg` variable
  - ✅ Converted to lazy initialization with `get_diffvg()` function

---

## MEDIUM PRIORITY

### 6. Magic Numbers and Strings

- [x] **stroke_core.py:86, 90, 129** - `merge_distance=12` undefined constant
  - ✅ Imported SKELETON_MERGE_DISTANCE from stroke_skeleton.py

- [x] **stroke_pipeline.py:232** - `60 <= abs(s['angle']) <= 120` magic angle range
  - ✅ Defined VERTICAL_ANGLE_MIN = 60, VERTICAL_ANGLE_MAX = 120

- [x] **stroke_pipeline.py:701** - `waist_tolerance = h * 0.15` magic multiplier
  - ✅ Defined WAIST_TOLERANCE_RATIO = 0.15

- [x] **stroke_pipeline.py:747** - `waist_margin = h * 0.05` magic multiplier
  - ✅ Defined WAIST_MARGIN_RATIO = 0.05

- [x] **stroke_pipeline_stream.py:60** - `[:500]` hardcoded skeleton limit
  - ✅ Defined MAX_SKELETON_MARKERS = 500

- [x] **stroke_routes_core.py:519-520** - Quality thresholds (10, 15, 0.225, 2)
  - ✅ Already defined as MIN_SHAPE_COUNT, MAX_SHAPE_COUNT, MAX_WIDTH_RATIO, EXPECTED_EXCLAMATION_SHAPES

- [x] **stroke_routes_stream.py:245** - Affine bounds `[(-15, 15), ...]`
  - ✅ Defined AFFINE_TX_BOUNDS, AFFINE_TY_BOUNDS, AFFINE_SX_BOUNDS, AFFINE_SY_BOUNDS, AFFINE_ROTATE_BOUNDS, AFFINE_SHEAR_BOUNDS

- [x] **stroke_routes_stream.py:219** - Optimizer params `maxfev=400, xatol=0.2`
  - ✅ Defined NM_GLOBAL_MAXFEV, NM_GLOBAL_XATOL, NM_GLOBAL_FATOL, NM_PERSTROKE_MAXFEV, NM_PERSTROKE_XATOL, DE_MAXITER, DE_POPSIZE, DE_TOL

- [x] **stroke_rendering.py:68, 128** - `0.9` canvas fill threshold
  - ✅ Already defined as CANVAS_FILL_THRESHOLD = 0.9

- [x] **stroke_rendering.py:150, 224** - `128` binarization threshold
  - ✅ Already defined as BINARIZATION_THRESHOLD = 128

- [x] **stroke_shapes.py:485-495** - Magic values in adaptive_radius
  - ✅ Defined MIN_SHAPE_POINTS, POINT_SPACING_TARGET, RADIUS_FLOOR_MULTIPLIER, MIN_RADIUS, DISTANCE_PERCENTILE

- [x] **stroke_skeleton.py:369-384** - Neighbor scoring weights
  - ✅ Defined DIRECTION_BIAS_STEPS, DIRECTION_BIAS_WEIGHT, AVOID_PENALTY

- [x] **stroke_lib/analysis/skeleton.py:221** - `near_vertex_dist = 5`
  - ✅ Defined NEAR_VERTEX_DISTANCE = 5 as module constant

- [x] **stroke_lib/analysis/skeleton.py:715** - `stub_threshold = 20`
  - ✅ Defined STUB_THRESHOLD = 20 as module constant

### 7. Missing Type Hints

- [x] **stroke_core.py** - Functions missing type hints:
  - ✅ Added type hints to all 6 functions with proper return types
  - Added `from __future__ import annotations`, `from typing import Any`

- [x] **stroke_pipeline.py:115** - Constructor params need type hints
  - ✅ Added `create_default()` factory method with typed signature
  - Constructor types documented in docstring (callable types complex)

- [x] **stroke_pipeline_stream.py:114-116** - `_stream_variant_strokes()` params
  - ✅ Uses Generator type hints from extracted helper functions

- [x] **stroke_utils.py:158-186** - `parse_waypoint()` missing return type
  - ✅ Added `-> tuple[int, str]` return type

- [x] **stroke_rendering.py:241, 280, 320** - `pil_font` parameter untyped
  - ✅ Added `FreeTypeFont` type with TYPE_CHECKING conditional import

- [x] **stroke_dataclasses.py:129, 134** - `info: dict`, `skel_tree: object`
  - ✅ Changed to `dict[str, Any]` and `cKDTree` with TYPE_CHECKING import

### 8. Poor Naming

- [x] **stroke_pipeline.py** - Local function renames:
  - ✅ `dist()` → `euclidean_distance()`
  - ✅ `angle_matches()` → `angle_in_range()`
  - ✅ `region_to_rc()` → `region_to_row_col()`

- [x] **stroke_core.py** - Function and variable renames:
  - ✅ `_skel()` → `_analyze_skeleton_legacy()`
  - ✅ `eps` → `endpoints`, `jps` → `junction_pixels`, `jcs` → `junction_clusters`
  - ✅ `trace()` → `trace_path()`, `vedges` → `visited_edges`, `stops` → `stop_pixels`

- [x] **stroke_routes_core.py:130** - `q` → `query`, `rej` → `show_rejected`

- [x] **stroke_routes_stream.py:127-140** - Clearer spatial names:
  - ✅ `cloud_tree` → `glyph_kdtree`
  - ✅ `snap_yi` → `snap_row_indices`, `snap_xi` → `snap_col_indices`

- [x] **stroke_merge.py:220** - `_cid` → `cluster_id`

- [x] **stroke_lib/analysis/skeleton.py** - Method renames in nested functions:
  - ✅ `endpoint_cluster()` → `get_endpoint_cluster()`
  - ✅ `seg_dir()` → `compute_segment_direction()`
  - ✅ `angle()` → `angle_between_directions()`

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
- [x] Missing type hints (2026-02-10)
  - stroke_core.py: Added type hints to 6 functions
  - stroke_utils.py: Added return type to parse_waypoint()
  - stroke_rendering.py: Added FreeTypeFont type with TYPE_CHECKING
  - stroke_dataclasses.py: Added dict[str, Any] and cKDTree types
- [x] Poor naming fixes (2026-02-10)
  - stroke_pipeline.py: dist→euclidean_distance, angle_matches→angle_in_range, region_to_rc→region_to_row_col
  - stroke_core.py: _skel→_analyze_skeleton_legacy, eps/jps/jcs→full names
  - stroke_routes_core.py: q→query, rej→show_rejected
  - stroke_routes_stream.py: cloud_tree→glyph_kdtree, snap_yi/xi→snap_row/col_indices
  - stroke_merge.py: _cid→cluster_id
  - stroke_lib/analysis/skeleton.py: endpoint_cluster→get_endpoint_cluster, seg_dir→compute_segment_direction
- [x] Long function refactoring (2026-02-10)
  - stroke_scoring.py: Extracted 5 helpers, added module constants
  - stroke_skeleton.py: Extracted 4 helpers for find_skeleton_segments
  - stroke_utils.py: Extracted 3 helpers for build_guide_path
  - stroke_merge.py: Extracted 6 helpers for run_merge_pass and merge_t_junctions
  - stroke_dataclasses.py: Extracted 2 helpers, pre-compiled regex patterns
  - stroke_lib/analysis/segments.py: Extracted 3 methods for find_best_vertical_chain
  - stroke_pipeline.py: Extracted 2 methods for _trace_resolved_waypoints
  - stroke_pipeline_stream.py: Extracted 2 helpers for _stream_skeleton_evaluation
- [x] Fixed duplicate FREE_OVERLAP constant in stroke_scoring.py
- [x] Code duplication fixes (2026-02-10)
  - stroke_rendering.py: Extracted _scale_font_to_fit(), _compute_centered_position()
  - stroke_routes_core.py: Extracted _check_font_quality() with quality constants
  - stroke_pipeline.py: Moved extract_region() to module level
  - stroke_merge.py stub functions: Reviewed, kept separate for clarity
- [x] Error handling improvements (2026-02-10)
  - stroke_flask.py: get_font() now uses context manager
  - stroke_pipeline.py: resolve_waypoint() validates region 1-9
  - stroke_routes_stream.py: Added logging, specific exception handling
  - stroke_lib/api/services.py: Added logging, proper connection cleanup
  - stroke_routes_batch.py: Check column exists before ALTER TABLE
- [x] Magic numbers refactoring (2026-02-10)
  - stroke_skeleton.py: Added DIRECTION_BIAS_STEPS, DIRECTION_BIAS_WEIGHT, AVOID_PENALTY
  - stroke_core.py: Imported and used SKELETON_MERGE_DISTANCE
  - stroke_pipeline.py: Added VERTICAL_ANGLE_MIN/MAX, WAIST_TOLERANCE_RATIO, WAIST_MARGIN_RATIO
  - stroke_pipeline_stream.py: Added MAX_SKELETON_MARKERS
  - stroke_routes_stream.py: Added optimizer constants (NM_*, DE_*, AFFINE_*_BOUNDS)
  - stroke_shapes.py: Added MIN_SHAPE_POINTS, POINT_SPACING_TARGET, RADIUS_FLOOR_MULTIPLIER, MIN_RADIUS, DISTANCE_PERCENTILE
  - stroke_lib/analysis/skeleton.py: Added NEAR_VERTEX_DISTANCE, STUB_THRESHOLD
- [x] Remaining long function refactoring (2026-02-10)
  - stroke_skeleton.py: Extracted _compute_neighbor_score(), _bfs_trace_path() for trace_skeleton_path()
  - stroke_routes_stream.py: Extracted _get_initial_strokes(), _determine_completion_reason()
  - stroke_pipeline_stream.py: Extracted _create_pipeline(), _consume_subgenerator(), _stream_template_variants()
- [x] Testability and coupling improvements (2026-02-10)
  - stroke_pipeline.py: Added create_default() factory, fixed global_traced mutation
  - stroke_lib/api/services.py: Added connection_factory for testable FontService
  - stroke_scoring.py: Added ScoringContext dataclass and score_all_strokes_ctx()
  - stroke_lib/analysis/skeleton.py: Promoted nested trace function to static methods
  - stroke_routes_batch.py: Lazy init for _diffvg via get_diffvg()
  - stroke_lib/utils/rendering.py: Documented external dependency

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
