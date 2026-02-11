# Codebase Audit Checklist

Systematic review of font_scraper codebase for quality improvements.

## 1. Code Quality

- [ ] Dead code - unused functions/imports/variables
- [ ] Code duplication - repeated logic across files
- [ ] Long functions - functions over 50-100 lines
- [ ] Deep nesting - high cyclomatic complexity
- [ ] Magic numbers/strings - hardcoded literals
- [ ] Inconsistent naming - naming convention violations
- [ ] Missing type hints - unannotated functions

## 2. Architecture

- [ ] Circular dependencies - import cycles
- [ ] God classes/modules - files with too many responsibilities
- [ ] Tight coupling - excessive cross-module dependencies
- [ ] Missing abstraction layers - direct DB/API calls scattered
- [ ] Inconsistent patterns - mixed approaches to same problem

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

- [ ] SQL injection - string formatting in queries
- [ ] Hardcoded secrets - passwords, keys, tokens in code
- [ ] Missing input validation - unvalidated user input
- [ ] Insecure dependencies - known vulnerabilities
- [ ] Overly permissive CORS/permissions

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

