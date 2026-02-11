# TODO Refactor

Complete codebase audit findings and refactoring opportunities.

Generated: 2026-02-11 (Audit #2)
**COMPLETED: 2026-02-11** - All 106 items addressed

---

## Ordered Refactoring List

### CRITICAL - Broken/Undefined Code
1. [x] stroke_routes_core.py:141 - EXPECTED_EXCLAMATION_SHAPES undefined (import missing from stroke_services_core) - FIXED
2. [x] stroke_services_core.py:121,155 - Calls `font_repository.get_by_id()` but method is `get_font_by_id()` - FIXED

### CRITICAL - Placeholder/Stub Code
3. [x] stroke_lib/optimization/strategies.py:239 - `_compute_coverage()` returns hardcoded 0.5 - FIXED
4. [x] stroke_lib/optimization/strategies.py:261 - `_params_to_strokes()` returns empty list - FIXED
5. [x] stroke_lib/optimization/strategies.py:358 - `_optimize_single_shape()` returns (None, 0.0) - FIXED
6. [x] stroke_lib/optimization/strategies.py:466,488,508 - More placeholder implementations - FIXED

### CRITICAL - Security Issues
7. [x] stroke_routes_core.py:323-324 - JSON parsing without schema validation - NOT AN ISSUE (parses from database, not user input)
8. [x] stroke_routes_batch.py:1001 - Backend exceptions propagated to client (info disclosure) - FIXED (now logs + generic message)
9. [x] stroke_routes_core.py:1051-1055 - TOCTOU race condition in font rejection - DOCUMENTED as benign race (comment added)

### HIGH - Silent Exception Swallowing
10. [x] stroke_routes_core.py:567-568 - `except Exception:` with no logging - FIXED
11. [x] stroke_routes_core.py:690-691 - `except Exception:` with no logging - FIXED
12. [x] stroke_routes_batch.py:367-370 - `except Exception: pass` - NOT FOUND (may have been fixed previously)
13. [x] stroke_routes_stream.py:369,409,674,773 - Silent `except Exception:` (4 locations) - FIXED
14. [x] google_fonts_scraper.py:186,266 - Broad `except Exception as e:` catches - FIXED (print->logging)
15. [x] run_ocr_prefilter.py:71,246 - Silent/broad exception catches - FIXED (added logging, line 246 already records error)

### HIGH - Code Duplication (Routes)
16. [x] Font lookup pattern: 5 different ways to get font across route files - FIXED: Added get_font_or_error() helper
17. [x] Character param validation duplicated 5+ times with different response types - FIXED: Added get_char_param_or_error() helper
18. [x] SSE event formatting: 3 separate implementations in stroke_routes_stream.py - FIXED: Added format_sse_event() helper
19. [x] Error response creation: 5+ patterns (JSON, plain text, SSE) - FIXED: Added error_response() helper with format param

### HIGH - Code Duplication (Core)
20. [x] Waypoint resolution: stroke_pipeline.py and stroke_pipeline_stream.py duplicate logic - FIXED: Extracted infer_direction_from_regions() to geometry.py
21. [x] Path tracing: stroke_lib/analysis/skeleton.py:540-594 duplicates segments.py:311-389 - FIXED: Extracted pick_straightest_neighbor() to geometry.py
22. [x] Distance calculations: Angle/direction computed inline in 5+ places vs using utilities - FIXED: Added point_distance() to geometry.py
23. [x] Snap point calculation: Called redundantly in score_all_strokes and score_raw_strokes - FIXED: Extracted _score_strokes_internal() shared helper

### HIGH - Code Duplication (Scrapers)
24. [x] download_font() logic nearly identical across dafont, fontspace, google scrapers - FIXED: Added download_font_file() to FontSource base
25. [x] Pagination loop structure duplicated across all 3 scrapers - FIXED: Added paginate() generator to FontSource base
26. [x] ZIP extraction and font file handling duplicated 3x - FIXED: Added extract_font_from_zip() to FontSource base

### HIGH - Complex Functions (>70 lines)
27. [x] stroke_pipeline.py:681-761 - `_resolve_terminal()` 80 lines, 5+ nesting levels - FIXED: Split into 4 helper methods
28. [x] stroke_pipeline.py:763-846 - `_find_junction_for_direction()` 83 lines - FIXED: Split into 4 helper methods
29. [x] stroke_pipeline_stream.py:516-623 - `stream_minimal_strokes()` 107 lines - OK: 70 lines are docstring, code is 37 lines
30. [x] stroke_core.py:435-514 - `_find_closest_endpoint_pair()` 79 lines - FIXED: Split into 3 helper functions
31. [x] stroke_affine.py:562-646 - `optimize_affine()` 84 lines - FIXED: Split into 2 helper functions
32. [x] font_utils.py:257-337 - `score_font()` 81 lines, too many responsibilities - FIXED: Split into 3 helper methods

### HIGH - Missing Error Handling
33. [x] stroke_core.py:462 - Index access `strokes[i][-1]` without length check - FIXED: Added bounds check
34. [x] stroke_pipeline.py:498-500 - `bbox[0]` access without verifying bbox is valid - FIXED: Added bbox validation
35. [x] stroke_scoring.py:407 - Array indexing without bounds verification - FIXED: Added bounds check
36. [x] stroke_utils.py:272 - Dynamic import without try/except - FIXED: Added try/except with fallback
37. [x] font_utils.py:680-683 - Hash comparison may fail silently - FIXED: Added logging on failure

### HIGH - Inconsistent Logging
38. [x] fontspace_scraper.py - Uses print() instead of logger (30+ occurrences) - FIXED
39. [x] google_fonts_scraper.py - Mixes print() and logger.debug() - FIXED (all print->logger)
40. [x] run_ocr_prefilter.py - Added logging module, print() used for CLI progress output (acceptable)
41. [x] run_connectivity_filter.py - Added logging module, print() used for CLI progress output (acceptable)
42. [x] font_utils.py - Uses print() in utility functions - NOT AN ISSUE (docstrings + CLI test output)

### HIGH - Missing Retry Logic (Scrapers)
43. [x] All scrapers: No retry on transient failures (RequestException breaks loop) - FIXED: Added get_with_retry() to FontSource base class
44. [x] All scrapers: No 429 rate limit detection or backoff - FIXED: Added 429 handling with Retry-After header support
45. [x] All scrapers: No circuit breaker pattern for consistently failing hosts - FIXED: Implemented in font_source.py request_with_retry()

### MEDIUM - Magic Numbers
46. [x] stroke_pipeline.py:544 - Angle thresholds 75, 105 degrees not constants - FIXED: Added TRULY_VERTICAL_ANGLE_MIN/MAX
47. [x] stroke_pipeline.py:796-799 - Direction angle ranges 45, 135, -135, -45 hardcoded - FIXED: Added DIRECTION_ANGLE_* constants
48. [x] stroke_pipeline.py:810,827,837 - Distance thresholds 10, 3, 5 not constants - FIXED: Added DISTANCE_THRESHOLD_* constants
49. [x] stroke_merge_utils.py:279,288 - Tail point limit 8 appears twice, no constant - FIXED: Added TAIL_POINT_LIMIT constant
50. [x] stroke_utils.py:527 - k_candidates=50 magic number in KD-tree query - FIXED: Added KD_TREE_K_CANDIDATES constant
51. [x] stroke_utils.py:183 - `best_depth * 0.5` magic multiplier - FIXED: Added DEPTH_MULTIPLIER constant
52. [x] stroke_rendering.py:638,831,883 - Hardcoded 128 threshold (should use BINARIZATION_THRESHOLD) - FIXED: Now uses BINARIZATION_THRESHOLD
53. [x] font_utils.py:313-318 - Magic ink ratio thresholds 0.03, 0.25, 0.01, 0.35 - FIXED: Added INK_RATIO_* constants
54. [x] font_utils.py:332-335 - Magic weights 0.6 and 0.4 undocumented - FIXED: Added COVERAGE_WEIGHT, STYLE_WEIGHT constants

### MEDIUM - Response Format Inconsistencies
55. [x] Success responses: Mix of `{"ok": true}`, `{"strokes": [...]}`, plain data - FIXED: Added success_response(), data_response() helpers
56. [x] Error responses: Mix of `{"error": "..."}` JSON and plain text strings - FIXED: Standardized via error_response() helper
57. [x] Image endpoints: 3 different patterns for returning PNG - FIXED: Standardized via send_pil_image_as_png() helper

### MEDIUM - Inconsistent Scraper Interfaces
58. [x] Metadata dict keys differ: fonts_found vs fonts_requested, categories included/missing - FIXED: Added create_scrape_metadata() helper
59. [x] Progress output format differs across all 3 scrapers - FIXED: Added log_progress() method to FontSource
60. [x] Error tracking: Some track in self.failed, some don't - FIXED: All scrapers now consistently track in self.failed

### MEDIUM - Performance Issues
61. [x] stroke_core.py:460-468 - O(n^2) brute force for n<=6 strokes (use KD-tree always) - FIXED: Always use KD-tree
62. [x] stroke_pipeline.py:546-566 - Chain building has no early termination - FIXED: Added EARLY_EXIT_THRESHOLD
63. [x] stroke_merge_strategies.py:207-249 - T-junction search has no early exit - ALREADY IMPLEMENTED: Returns on first valid match
64. [x] stroke_utils.py:442-443 - Dual distance transform computation (could cache) - FIXED: Added compute_distance_cache() helper
65. [x] font_utils.py:691 - O(n^2) character comparison (render all at once instead) - FIXED: Inlined and optimized
66. [x] stroke_rendering.py - Repeated font loading bypasses _cached_font() - FIXED: Now uses _cached_font() consistently

### MEDIUM - Missing Caching
67. [x] stroke_pipeline.py:resolve_waypoint() - cKDTree.query() results not cached - FIXED: Added _kdtree_cache
68. [x] stroke_pipeline.py:find_best_vertical_segment() - Rebuilds junction map each call - FIXED: Added _junction_segment_map cache
69. [x] stroke_merge.py:MergePipeline.run() - Builds cluster maps 4+ times per iteration - FIXED: Added cache fields to MergeContext

### MEDIUM - Scattered Configuration
70. [x] NelderMead params: Defined in both stroke_affine.py and stroke_routes_stream.py - FIXED: Consolidated in stroke_affine.py, imported elsewhere
71. [x] DE params: Defined in both stroke_affine.py and stroke_routes_stream.py - FIXED: Consolidated in stroke_affine.py, imported elsewhere
72. [x] Per-script constants: run_ocr_prefilter.py, run_connectivity_filter.py, render_all_passing.py all define DB_PATH, SAMPLE_TEXT, FONT_SIZE independently - FIXED: Created filter_config.py

### MEDIUM - Missing Type Hints (Public Functions)
73. [x] stroke_core.py:165 - `min_strokes()` missing return type - FIXED
74. [x] stroke_core.py:183 - `auto_fit()` missing return type - FIXED
75. [x] stroke_pipeline.py:568 - `euclidean_distance()` missing return type - FIXED: Added -> float
76. [x] stroke_pipeline.py:802 - `angle_in_range()` missing return type - FIXED: Added -> bool
77. [x] template_morph.py:641 - `interpolate_stroke()` uses bare `list` not `list[tuple]` - FIXED (list[tuple] -> np.ndarray)

### MEDIUM - Inconsistent APIs
78. [x] Tuple vs List returns: stroke_utils.py returns tuples, stroke_contour.py returns lists - FIXED: Added type hints, documented conventions
79. [x] Error signaling: Some return None, some raise, some return empty results - FIXED: Documented conventions in module docstrings
80. [x] Waypoint formats: stroke_utils.py and stroke_dataclasses.py have separate parsers - FIXED: Consolidated to use stroke_dataclasses._parse_waypoint_item()

### MEDIUM - N+1 Database Queries
81. [x] stroke_routes_core.py:610-625 - Loop with individual font_repository.reject_font() calls - FIXED: Added reject_fonts_batch() method

### MEDIUM - Tight Coupling
82. [x] stroke_core.py manually constructs MinimalStrokePipeline with 11 callbacks - FIXED: Now uses PipelineFactory.create_with_config()
83. [x] stroke_pipeline_stream.py duplicates pipeline construction - FIXED: Now uses PipelineFactory.create_with_config()
84. [x] stroke_pipeline_stream.py imports private `_analyze_skeleton_legacy()` from stroke_core - FIXED: Renamed to public analyze_skeleton_legacy()
85. [x] stroke_scoring.py has both legacy and composite scorer paths in same function - FIXED: Both now use CompositeScorer consistently

### LOW - Missing Module Docstrings
86. [x] run_ocr_prefilter.py - Missing module docstring - ALREADY HAS DOCSTRING
87. [x] run_connectivity_filter.py - Missing module docstring - ALREADY HAS DOCSTRING
88. [x] run_prefilters.py - Missing module docstring - ALREADY HAS DOCSTRING
89. [x] scrape_all.py - Missing module docstring - ALREADY HAS DOCSTRING
90. [x] test_minimal_strokes.py - Missing module docstring - ALREADY HAS DOCSTRING
91. [x] test_flask_routes.py - Missing module docstring - ALREADY HAS DOCSTRING
92. [x] visualize_all_chars.py - Missing module docstring - ALREADY HAS DOCSTRING
93. [x] visualize_strokes.py - Missing module docstring - ALREADY HAS DOCSTRING
94. [x] inksight_cli.py - Missing module docstring - ALREADY HAS DOCSTRING
95. [x] docker/*.py files - Missing module docstrings - ALREADY HAVE DOCSTRINGS

### LOW - Unused/Dead Code
96. [x] stroke_utils.py:29 - `gaussian_filter1d` imported but never used (noqa present) - ALREADY REMOVED: Import no longer exists
97. [x] stroke_utils.py:304-305 - point_in_region comment but function not used - NOT DEAD: Re-exported from geometry.py, used by multiple files
98. [x] font_utils.py:295-303 - Loop generates discarded intermediate images - ALREADY OPTIMIZED: Reuses single image with draw.rectangle()

### LOW - Module-Level Aliases
99. [x] stroke_routes_core.py:112 - `_font = get_font` alias (still in use, would require changes to 5 call sites) - FIXED: Removed alias, updated call sites
100. [x] stroke_routes_batch.py:82-85 - Multiple aliases to imported functions - REMOVED (3 unused aliases)
101. [x] stroke_routes_stream.py:95 - `_font = get_font` alias - REMOVED (unused)

### LOW - stroke_lib Package Issues
102. [x] stroke_lib/optimization/__init__.py - Missing `create_default_optimizer` in __all__ - FIXED: Added to __all__
103. [x] stroke_lib/optimization/strategies.py - Late imports inside methods (time, scipy) - FIXED: Moved imports to module top

### LOW - Hard-Coded Scraper Values
104. [x] HTTP timeouts hard-coded (30s pages, 60s downloads) - should be configurable - FIXED: Added PAGE_TIMEOUT, DOWNLOAD_TIMEOUT constants to all scrapers
105. [x] Regex patterns hard-coded in scraper methods - should be class constants - FIXED: Compiled patterns as class attributes
106. [x] scrape_all.py:74,111 - Hard-coded categories and queries - FIXED: Added DAFONT_CATEGORIES, FONTSPACE_QUERIES constants

---

## Previously Completed (Audit #1)

All 54 items from previous audit completed:
- [x] 1-5: Security & Reliability (input validation, context managers, thread safety)
- [x] 6-9: Performance O(n^2) algorithms (cluster indexing, KD-tree)
- [x] 10-13: God module splits (pipeline, merge, routes)
- [x] 14-18: Architecture (StrokeProcessor, repositories, FontSource base)
- [x] 19-23: Code quality (constants extraction, specific exceptions)
- [x] 24-28: Testing gaps (skipped per user request)
- [x] 29-31: Performance caching (KD-tree, LRU cache, endpoint cache)
- [x] 32-35: Code duplication utilities extracted
- [x] 36-41: Deep nesting and long functions refactored
- [x] 42-47: Type hints and documentation added
- [x] 48-54: Memory, code style, logger standardization

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Broken/Undefined Code | 2 | COMPLETE |
| Placeholder/Stub Code | 4 | COMPLETE |
| Security Issues | 3 | COMPLETE |
| Silent Exception Swallowing | 6 | COMPLETE |
| Code Duplication (Routes) | 4 | COMPLETE |
| Code Duplication (Core) | 4 | COMPLETE |
| Code Duplication (Scrapers) | 3 | COMPLETE |
| Complex Functions | 6 | COMPLETE |
| Missing Error Handling | 5 | COMPLETE |
| Inconsistent Logging | 5 | COMPLETE |
| Missing Retry Logic | 3 | COMPLETE |
| Magic Numbers | 9 | COMPLETE |
| Response Inconsistencies | 3 | COMPLETE |
| Scraper Interfaces | 3 | COMPLETE |
| Performance Issues | 6 | COMPLETE |
| Missing Caching | 3 | COMPLETE |
| Scattered Configuration | 3 | COMPLETE |
| Missing Type Hints | 5 | COMPLETE |
| Inconsistent APIs | 3 | COMPLETE |
| N+1 Queries | 1 | COMPLETE |
| Tight Coupling | 4 | COMPLETE |
| Missing Docstrings | 10 | COMPLETE |
| Unused/Dead Code | 3 | COMPLETE |
| Module Aliases | 3 | COMPLETE |
| stroke_lib Issues | 2 | COMPLETE |
| Hard-Coded Scrapers | 3 | COMPLETE |
| **TOTAL** | **106** | **ALL COMPLETE** |

---

## Verification Commands

```bash
# Run all tests after changes
python3 -m unittest test_flask_routes -v
python3 test_minimal_strokes.py

# Syntax check all Python files
python3 -m py_compile *.py

# Check for remaining issues
grep -rn "except Exception:" *.py
grep -rn "except:" *.py | grep -v "except Exception"
grep -rn "print(" *.py | grep -v "# noqa" | grep -v test_ | grep -v "def print"
```
