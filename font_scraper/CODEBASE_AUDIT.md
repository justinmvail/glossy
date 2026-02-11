# Codebase Audit Checklist

Systematic review of font_scraper codebase for quality improvements.

## 1. Code Quality

- [x] Dead code - unused functions/imports/variables
  - **Status:** 58 potentially unused functions found
  - Notable: `analyze_skeleton`, `batch_process`, `contour_to_strokes`, etc.
  - Many are utility functions - need manual review to confirm unused
- [x] Code duplication - repeated logic across files
  - **Status:** 1358 duplicate 5-line blocks detected
  - Hotspots: stroke_routes_*.py (route boilerplate), font_utils.py
  - Recommendation: Extract common route patterns to decorators/helpers
- [x] Long functions - functions over 50-100 lines
  - **Status:** 20 functions over 50 lines found
  - Worst: run_prefilters.py:run_pipeline() 275 lines
  - Others: template_morph.py:find_vertices() 225 lines
  - Note: stroke_editor.py already refactored per earlier plan
- [x] Deep nesting - high cyclomatic complexity
  - **Status:** 10 functions with 4+ nesting levels
  - Worst: template_morph.py:find_vertices() depth 10
  - Others: run_ocr_prefilter.py:run_batch_ocr() depth 9
- [x] Magic numbers/strings - hardcoded literals
  - **Status:** Common values found (3, 60, 5, 90, 400, 500, 80)
  - Many are already constants in stroke_affine.py
  - HTTP codes (400, 404, 500) are acceptable
- [x] Inconsistent naming - naming convention violations
  - **Status:** PASS - consistent snake_case throughout
- [x] Missing type hints - unannotated functions
  - **Status:** 89 public functions without type hints
  - Hotspots: template_morph.py, font_utils.py, stroke_utils.py

## 2. Architecture

- [x] Circular dependencies - import cycles
  - **Status:** FIXED - stroke_core ↔ stroke_pipeline cycle broken
  - Fix: Moved MinimalStrokePipeline import to lazy (inside function)
- [x] God classes/modules - files with too many responsibilities
  - **Status:** Noted - inksight_vectorizer.py (1364 lines), stroke_pipeline.py (1198 lines)
  - These are complex but cohesive - no immediate action needed
- [x] Tight coupling - excessive cross-module dependencies
  - **Status:** Acceptable - stroke_flask imported by 9 files (it's the Flask app core)
- [x] Missing abstraction layers - direct DB/API calls scattered
  - **Status:** FIXED - Added TestRunRepository for test_runs table
  - Migrated 6 direct DB calls in stroke_routes_batch.py to use repository
- [x] Inconsistent patterns - mixed approaches to same problem
  - **Status:** PASS - consistent repository pattern now used for DB access

## 3. Performance

- [ ] O(n²) algorithms - inefficient loops
- [ ] Repeated expensive operations - missing caching
- [ ] N+1 queries - database query patterns
- [ ] Blocking I/O - synchronous bottlenecks
- [ ] Memory leaks - reference cycles, unclosed resources
- [ ] Unindexed queries - slow database operations

## 4. Reliability

- [ ] Missing error handling - bare try/except, unhandled exceptions
- [ ] Silent failures - errors swallowed without logging
- [ ] Race conditions - concurrent access issues
- [ ] Resource leaks - unclosed files/connections
- [ ] Hardcoded timeouts - non-configurable wait times

## 5. Security

- [x] SQL injection - string formatting in queries
  - **Status:** PASS - All queries use parameterized `?` placeholders
- [x] Hardcoded secrets - passwords, keys, tokens in code
  - **Status:** PASS - No secrets found (only InkSight model tokens)
- [x] Missing input validation - unvalidated user input
  - **Status:** FIXED - Changed `int(request.args.get())` to `request.args.get(type=int)` with bounds checking
  - Files: stroke_routes_core.py, stroke_routes_batch.py
- [x] Insecure dependencies - known vulnerabilities
  - **Status:** Manual audit - cryptography 41.0.7 has CVE-2023-50782 (needs 42.0.0+)
  - Note: System Python locked, upgrade requires --break-system-packages or venv
- [x] Overly permissive CORS/permissions
  - **Status:** PASS - No CORS configured (local tool)
- [x] Command injection - subprocess with shell=True
  - **Status:** PASS - All subprocess calls use list args, no shell=True
- [x] Error message leaks - exposing internal exceptions
  - **Status:** FIXED - Replaced `str(e)` with generic messages in HTTP responses
  - Files: stroke_routes_core.py, stroke_routes_stream.py
- [x] Path traversal - user input in file paths
  - **Status:** PASS - File paths come from DB, not user input
- [x] Deserialization - pickle/yaml.load
  - **Status:** PASS - No unsafe deserialization found

## 6. Testing

- [ ] Low coverage - untested critical paths
- [ ] Flaky tests - non-deterministic failures
- [ ] Slow tests - tests taking too long
- [ ] Missing integration tests - no end-to-end coverage
- [ ] Untestable code - tightly coupled, hard to mock

## 7. Configuration & Operations

- [ ] Hardcoded config - IPs, ports, paths in code
- [ ] Missing logging - insufficient observability
- [ ] No health checks - missing service health endpoints
- [ ] Manual deployments - no CI/CD
- [ ] No monitoring - missing metrics/alerting

## 8. Documentation

- [ ] Missing README - inadequate setup/usage docs
- [ ] Outdated docs - documentation doesn't match code
- [ ] No API docs - missing docstrings
- [ ] Missing architecture diagrams - complex flows undocumented
- [ ] Tribal knowledge - undocumented decisions

## 9. Dependencies

- [ ] Outdated packages - old versions with fixes available
- [ ] Unused dependencies - packages not actually used
- [ ] Unpinned versions - missing version constraints
- [ ] Conflicting versions - dependency conflicts
- [ ] Abandoned packages - unmaintained dependencies

## 10. Developer Experience

- [ ] Slow builds - long setup/build times
- [ ] Complex setup - difficult onboarding
- [ ] No linting/formatting - inconsistent style
- [ ] Inconsistent style - mixed conventions
- [ ] Missing dev tools - no debugging/profiling helpers

---

## Progress Log

### Session 1: Initial Setup
- Created audit checklist

### Session 1: Security Audit
- Scanned for SQL injection: PASS (all parameterized queries)
- Scanned for hardcoded secrets: PASS (none found)
- Scanned for command injection: PASS (no shell=True)
- Scanned for path traversal: PASS (DB-sourced paths)
- Scanned for deserialization: PASS (no pickle/yaml.load)
- Fixed input validation in 3 locations
- Fixed error message leaks in 4 locations

### Session 1: Code Quality Audit
- Long functions: 20 over 50 lines (worst: 275 lines)
- Deep nesting: 10 functions with depth 4+ (worst: depth 10)
- Code duplication: 1358 duplicate blocks (route boilerplate)
- Dead code: 58 potentially unused functions
- Type hints: 89 public functions missing annotations
- Naming: PASS (consistent snake_case)

### Session 1: Code Quality Fixes
- Extracted `_find_proximity_merge_target()` from `absorb_proximity_stubs()` - reduced nesting depth 6→4
- Extracted `_process_qcurve()` from `extract_contours()` - reduced nesting depth 8→4
- Extracted `_find_waist_height()` from `find_vertices()` - reduced duplication for B/P/R chars
- Removed dead code: `catmull_rom_point()`, `catmull_rom_segment()` from stroke_utils.py

### Session 1: Architecture Fixes
- Fixed circular dependency: stroke_core ↔ stroke_pipeline (lazy import)
- Added TestRunRepository class to stroke_flask.py
- Migrated stroke_routes_batch.py to use TestRunRepository (6 DB calls)

