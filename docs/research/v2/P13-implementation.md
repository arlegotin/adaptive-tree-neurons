# P13 Simple References Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. Root explicitly prohibits additional agents; root owns the independent final review.

**Goal:** Exact29 F-only closed-form references, tiny parity and strict portable deployment.

**Architecture:** One new P13 module consumes existing FitRequest, plain schema/frame helpers and pinned NumPy/Pandas. Preprocessing and saved arrays remain distinct from legacy estimators.

**Tech Stack:** Existing core27 Python3.12.13, NumPy2.5.4, Pandas3.0.6; no installation.

**Spec:** [approved contract](P13-simple-references.md), commit2900a65 plus root's explicit feature-exclusion clarification.

## Constraints and review focus

Own new P13 module/tests/docs only. Preserve sum-loss alpha and all29 configurations. No arbitrary name blacklist. No G access or Torch/NODE imports. Raw-data/HPO/full-workload admission/hard resource and recovery harness remain pending. Scoped tests intentionally exclude scientific fits in the full repository suite.

1. Typed category collisions/unseen values preserve explicit unknown geometry.
2. Numeric imputation/knots use F only, with categories linear in spline models.
3. Rank-deficient and constant fits preserve declared solve semantics.
4. Snapshot schema/geometry/dtype/source/runtime changes reject before restore.
5. Fit failure preserves fitted state, original exception and paid timing.

## Tasks

- [x] T1: Write failing support/geometry/role/numerical tests in tests/test_p13_simple.py; verify missing API RED; implement only src/treeneuron/research/p13_simple.py; run scoped GREEN.
- [x] T2: Write failing portable-state/lazy-import/error tests; verify RED; implement strict serialization/identity/receipts; run scoped GREEN.
- [x] T3: Add all29 unchanged-reference parity and independent numeric/category solves; verify RED when new behavior absent; run combined P13+legacy simple/transform tests in one-thread processes; retain logs. Hold source for root's independent final review, then one RED/GREEN correction pass if requested.

## Progress and rulings

- Setup: native managed worktree0dbd, branch research-v2-p13-simple from4cc937d; approved contract2900a65.
- Ruling: root clarification supersedes generic protected-name rejection. Explicit semantic schemas own exclusions; no inferred feature dropping.
- Ruling: only scoped deterministic engineering and unchanged-reference tests run; bare full pytest could start forbidden scientific fits. No installs or extra agents.

- TDD evidence: missing API first produced 30 failures; the first implementation passed 30 tests. Extended tiny fixtures and unchanged references passed 87 tests. Three self-review regressions (category-squared prediction allocation, impossible constant-column knots, malformed F identity container) and two installed-runtime binding regressions were each observed RED before correction.
- Held-source verification: P13 + unchanged simple baseline + unchanged transform tests passed 92 tests in 14.77 seconds. Exact command, raw logs, content commitments and scope are retained in [evidence receipt](P13-evidence/receipt.json).
- Independent review: source held for the one root-approved read-only review by diagnostic_review. Findings and disposition will be recorded here; no second review is planned.
- Remaining prerequisites: prepared transfer data and semantic lineage, complete finite transformed workloads, external hard process/resource limits, interruption/retry harness, search/cost attribution, and Gate D remain unqualified. The adapter performs no G selection and these engineering fixtures make no practical efficacy claim.
