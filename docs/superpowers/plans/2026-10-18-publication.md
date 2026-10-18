# Public Research Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline preparation and an independent final review.

**Goal:** Publish an honest, usable research repository with clear value, results, licensing and citation.

**Architecture:** Documentation and release metadata surround unchanged scientific source and evidence. Local history inspection and clean-clone checks gate the first remote push and public visibility.

**Tech Stack:** Existing Python3.12/uv/PyTorch lock, GitHub CLI, CFF, Markdown; checksum-verified Gitleaks8.30.1.

**Spec:** ../specs/2026-10-18-publication-design.md

## Global Constraints

- Stay on current main; public push/visibility are authorized by the owner.
- Preserve source/evidence/historical hashes and separate16-problem and8-problem studies.
- AGPL-3.0-only software; CC-BY-4.0 original prose/results; retain third-party exceptions.
- Do not invent DOI, superior accuracy, hard-execution savings or confirmation.
- No ignored archives/environments/credentials uploaded; no destructive history rewrite.

## Review Focus

- Fresh checkout: commands must work without ignored experiment artifacts.
- Full result authentication: hashes and aggregates alone are insufficient.
- Comparison labels: fresh ordinary and warm widening controls remain distinct.
- Licensing: packaged vendor code retains its original notices and terms.
- Publication boundary: scan complete reachable history, then verify actual public HEAD.

## Task 1: Prepare public documentation and metadata

- [ ] Read recent owner research READMEs/citations/licensing and exact requested writing skill.
- [ ] Rewrite README around mechanism, reusable parts, scoped results, runnable checks and citation.
- [ ] Add LICENSE, licenses/CC-BY-4.0.txt, LICENSING.md, NOTICE, THIRD_PARTY_NOTICES.md, CITATION.cff; update package metadata without changing dependency resolution.
- [ ] Add REPRODUCE.md, docs/PUBLICATION.md and small contribution guidance. Verify README inline example, both reference scripts, CFF structure and local links; retain publication evidence under provenance/publication/.
- [ ] Commit the concrete public-facing preparation.

## Task 2: Check an artifact-free clone and review publication

- [ ] Create a fresh local clone at the preparation commit, reuse locked installed dependencies with project imports from that clone, and run documented checks plus one full software-suite attempt.
- [ ] Diagnose actual failures; preserve commands and outputs. Any required source correction gets scoped tests and review before publishing; scientific results are not rerun/rebound.
- [ ] Obtain independent review of claims, README usability, licenses/citation, release scope and changed files; resolve important findings.
- [ ] Scan all reachable Git history and final publication files with redacted Gitleaks; retain actual exit/count/tool version and inspect any findings. Verify ignored research archive exclusion and small synthetic-fixture scope.
- [ ] Commit verified release provenance and final documentation.

## Task 3: Publish and verify

- [ ] Push prepared main to arlegotin/adaptive-tree-neurons while still private; set description/topics.
- [ ] Change visibility to public as explicitly requested.
- [ ] Verify authenticated settings, anonymous README/repository access, license/citation recognition, remote HEAD equality and clean local branch. Report public URL and any remaining reproducibility limits.
