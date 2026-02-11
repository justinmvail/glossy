# TODO Refactor

Complete codebase audit findings and refactoring opportunities.

Generated: 2026-02-11 (Audit #2)

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
9. [ ] stroke_routes_core.py:1051-1055 - TOCTOU race condition in font rejection - LOW PRIORITY (benign race)

### HIGH - Silent Exception Swallowing
10. [x] stroke_routes_core.py:567-568 - `except Exception:` with no logging - FIXED
11. [x] stroke_routes_core.py:690-691 - `except Exception:` with no logging - FIXED
12. [x] stroke_routes_batch.py:367-370 - `except Exception: pass` - NOT FOUND (may have been fixed previously)
13. [x] stroke_routes_stream.py:369,409,674,773 - Silent `except Exception:` (4 locations) - FIXED
14. [x] google_fonts_scraper.py:186,266 - Broad `except Exception as e:` catches - FIXED (print->logging)
15. [x] run_ocr_prefilter.py:71,246 - Silent/broad exception catches - FIXED (added logging, line 246 already records error)

### HIGH - Code Duplication (Routes)
16. [ ] Font lookup pattern: 5 different ways to get font across route files
17. [ ] Character param validation duplicated 5+ times with different response types
18. [ ] SSE event formatting: 3 separate implementations in stroke_routes_stream.py
19. [ ] Error response creation: 5+ patterns (JSON, plain text, SSE)

### HIGH - Code Duplication (Core)
20. [ ] Waypoint resolution: stroke_pipeline.py and stroke_pipeline_stream.py duplicate logic
21. [ ] Path tracing: stroke_lib/analysis/skeleton.py:540-594 duplicates segments.py:311-389
22. [ ] Distance calculations: Angle/direction computed inline in 5+ places vs using utilities
23. [ ] Snap point calculation: Called redundantly in score_all_strokes and score_raw_strokes

### HIGH - Code Duplication (Scrapers)
24. [ ] download_font() logic nearly identical across dafont, fontspace, google scrapers
25. [ ] Pagination loop structure duplicated across all 3 scrapers
26. [ ] ZIP extraction and font file handling duplicated 3x

### HIGH - Complex Functions (>70 lines)
27. [ ] stroke_pipeline.py:681-761 - `_resolve_terminal()` 80 lines, 5+ nesting levels
28. [ ] stroke_pipeline.py:763-846 - `_find_junction_for_direction()` 83 lines
29. [ ] stroke_pipeline_stream.py:516-623 - `stream_minimal_strokes()` 107 lines
30. [ ] stroke_core.py:435-514 - `_find_closest_endpoint_pair()` 79 lines
31. [ ] stroke_affine.py:562-646 - `optimize_affine()` 84 lines
32. [ ] font_utils.py:257-337 - `score_font()` 81 lines, too many responsibilities

### HIGH - Missing Error Handling
33. [ ] stroke_core.py:462 - Index access `strokes[i][-1]` without length check
34. [ ] stroke_pipeline.py:498-500 - `bbox[0]` access without verifying bbox is valid
35. [ ] stroke_scoring.py:407 - Array indexing without bounds verification
36. [ ] stroke_utils.py:272 - Dynamic import without try/except
37. [ ] font_utils.py:680-683 - Hash comparison may fail silently

### HIGH - Inconsistent Logging
38. [x] fontspace_scraper.py - Uses print() instead of logger (30+ occurrences) - FIXED
39. [x] google_fonts_scraper.py - Mixes print() and logger.debug() - FIXED (all print->logger)
40. [x] run_ocr_prefilter.py - Added logging module, print() used for CLI progress output (acceptable)
41. [x] run_connectivity_filter.py - Added logging module, print() used for CLI progress output (acceptable)
42. [x] font_utils.py - Uses print() in utility functions - NOT AN ISSUE (docstrings + CLI test output)

### HIGH - Missing Retry Logic (Scrapers)
43. [ ] All scrapers: No retry on transient failures (RequestException breaks loop)
44. [ ] All scrapers: No 429 rate limit detection or backoff
45. [ ] All scrapers: No circuit breaker pattern for consistently failing hosts

### MEDIUM - Magic Numbers
46. [ ] stroke_pipeline.py:544 - Angle thresholds 75, 105 degrees not constants
47. [ ] stroke_pipeline.py:796-799 - Direction angle ranges 45, 135, -135, -45 hardcoded
48. [ ] stroke_pipeline.py:810,827,837 - Distance thresholds 10, 3, 5 not constants
49. [ ] stroke_merge_utils.py:279,288 - Tail point limit 8 appears twice, no constant
50. [ ] stroke_utils.py:527 - k_candidates=50 magic number in KD-tree query
51. [ ] stroke_utils.py:183 - `best_depth * 0.5` magic multiplier
52. [ ] stroke_rendering.py:638,831,883 - Hardcoded 128 threshold (should use BINARIZATION_THRESHOLD)
53. [ ] font_utils.py:313-318 - Magic ink ratio thresholds 0.03, 0.25, 0.01, 0.35
54. [ ] font_utils.py:332-335 - Magic weights 0.6 and 0.4 undocumented

### MEDIUM - Response Format Inconsistencies
55. [ ] Success responses: Mix of `{"ok": true}`, `{"strokes": [...]}`, plain data
56. [ ] Error responses: Mix of `{"error": "..."}` JSON and plain text strings
57. [ ] Image endpoints: 3 different patterns for returning PNG

### MEDIUM - Inconsistent Scraper Interfaces
58. [ ] Metadata dict keys differ: fonts_found vs fonts_requested, categories included/missing
59. [ ] Progress output format differs across all 3 scrapers
60. [ ] Error tracking: Some track in self.failed, some don't

### MEDIUM - Performance Issues
61. [ ] stroke_core.py:460-468 - O(n²) brute force for n≤6 strokes (use KD-tree always)
62. [ ] stroke_pipeline.py:546-566 - Chain building has no early termination
63. [ ] stroke_merge_strategies.py:207-249 - T-junction search has no early exit
64. [ ] stroke_utils.py:442-443 - Dual distance transform computation (could cache)
65. [ ] font_utils.py:691 - O(n²) character comparison (render all at once instead)
66. [ ] stroke_rendering.py - Repeated font loading bypasses _cached_font()

### MEDIUM - Missing Caching
67. [ ] stroke_pipeline.py:resolve_waypoint() - cKDTree.query() results not cached
68. [ ] stroke_pipeline.py:find_best_vertical_segment() - Rebuilds junction map each call
69. [ ] stroke_merge.py:MergePipeline.run() - Builds cluster maps 4+ times per iteration

### MEDIUM - Scattered Configuration
70. [ ] NelderMead params: Defined in both stroke_affine.py and stroke_routes_stream.py
71. [ ] DE params: Defined in both stroke_affine.py and stroke_routes_stream.py
72. [ ] Per-script constants: run_ocr_prefilter.py, run_connectivity_filter.py, render_all_passing.py all define DB_PATH, SAMPLE_TEXT, FONT_SIZE independently

### MEDIUM - Missing Type Hints (Public Functions)
73. [ ] stroke_core.py:165 - `min_strokes()` missing return type
74. [ ] stroke_core.py:183 - `auto_fit()` missing return type
75. [ ] stroke_pipeline.py:568 - `euclidean_distance()` missing return type
76. [ ] stroke_pipeline.py:802 - `angle_in_range()` missing return type
77. [ ] template_morph.py:641 - `interpolate_stroke()` uses bare `list` not `list[tuple]`

### MEDIUM - Inconsistent APIs
78. [ ] Tuple vs List returns: stroke_utils.py returns tuples, stroke_contour.py returns lists
79. [ ] Error signaling: Some return None, some raise, some return empty results
80. [ ] Waypoint formats: stroke_utils.py and stroke_dataclasses.py have separate parsers

### MEDIUM - N+1 Database Queries
81. [ ] stroke_routes_batch.py:610-625 - Loop with individual font_repository.reject_font() calls

### MEDIUM - Tight Coupling
82. [ ] stroke_core.py manually constructs MinimalStrokePipeline with 11 callbacks
83. [ ] stroke_pipeline_stream.py duplicates pipeline construction
84. [ ] stroke_pipeline_stream.py imports private `_analyze_skeleton_legacy()` from stroke_core
85. [ ] stroke_scoring.py has both legacy and composite scorer paths in same function

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
96. [ ] stroke_utils.py:29 - `gaussian_filter1d` imported but never used (noqa present)
97. [ ] stroke_utils.py:304-305 - point_in_region comment but function not used
98. [ ] font_utils.py:295-303 - Loop generates discarded intermediate images

### LOW - Module-Level Aliases
99. [ ] stroke_routes_core.py:112 - `_font = get_font` alias (still in use, would require changes to 5 call sites)
100. [x] stroke_routes_batch.py:82-85 - Multiple aliases to imported functions - REMOVED (3 unused aliases)
101. [x] stroke_routes_stream.py:95 - `_font = get_font` alias - REMOVED (unused)

### LOW - stroke_lib Package Issues
102. [ ] stroke_lib/optimization/__init__.py - Missing `create_default_optimizer` in __all__
103. [ ] stroke_lib/optimization/strategies.py - Late imports inside methods (time, scipy)

### LOW - Hard-Coded Scraper Values
104. [ ] HTTP timeouts hard-coded (30s pages, 60s downloads) - should be configurable
105. [ ] Regex patterns hard-coded in scraper methods - should be class constants
106. [ ] scrape_all.py:74,111 - Hard-coded categories and queries

---

## Previously Completed (Audit #1)

All 54 items from previous audit completed:
- [x] 1-5: Security & Reliability (input validation, context managers, thread safety)
- [x] 6-9: Performance O(n²) algorithms (cluster indexing, KD-tree)
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

| Category | Count | Severity |
|----------|-------|----------|
| Broken/Undefined Code | 2 | CRITICAL |
| Placeholder/Stub Code | 4 | CRITICAL |
| Security Issues | 3 | CRITICAL |
| Silent Exception Swallowing | 6 | HIGH |
| Code Duplication (Routes) | 4 | HIGH |
| Code Duplication (Core) | 4 | HIGH |
| Code Duplication (Scrapers) | 3 | HIGH |
| Complex Functions | 6 | HIGH |
| Missing Error Handling | 5 | HIGH |
| Inconsistent Logging | 5 | HIGH |
| Missing Retry Logic | 3 | HIGH |
| Magic Numbers | 9 | MEDIUM |
| Response Inconsistencies | 3 | MEDIUM |
| Scraper Interfaces | 3 | MEDIUM |
| Performance Issues | 6 | MEDIUM |
| Missing Caching | 3 | MEDIUM |
| Scattered Configuration | 3 | MEDIUM |
| Missing Type Hints | 5 | MEDIUM |
| Inconsistent APIs | 3 | MEDIUM |
| N+1 Queries | 1 | MEDIUM |
| Tight Coupling | 4 | MEDIUM |
| Missing Docstrings | 10 | LOW |
| Unused/Dead Code | 3 | LOW |
| Module Aliases | 3 | LOW |
| stroke_lib Issues | 2 | LOW |
| Hard-Coded Scrapers | 3 | LOW |
| **TOTAL** | **106** | |

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

---

## Priority Order

### Phase 1: Critical (Fix immediately)
- Items 1-9: Broken code, stubs, security issues

### Phase 2: High (Fix soon)
- Items 10-45: Exception handling, duplication, complex functions, logging

### Phase 3: Medium (Refactor)
- Items 46-85: Magic numbers, consistency, performance, caching

### Phase 4: Low (Cleanup)
- Items 86-106: Docstrings, dead code, minor issues
