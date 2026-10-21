# Project sunset — 21 October 2026

The owner ended the project and requested removal of its local experiment
artifacts, auxiliary worktrees and project-specific caches. The public repository
remains the evidence record. The main source checkout is retained as a small
local copy of that public material.

Cleanup removed all 17 auxiliary worktrees, 213 inventoried project-specific
temporary entries, and ignored research archives, checkpoints, environments,
download caches and runtime caches in the main checkout. Shared caches and other
projects were not targeted. Git initially protected nested test repositories;
a follow-up was restricted to the two remaining ignored artifact directories.
The final check found no ignored cleanup entries and only the main worktree.

All 2,580 previously tracked files were checked byte for byte after cleanup and
before the sunset documentation updates. Tracked source, compact results,
selected numerical fixtures and audit evidence were preserved, including the
small deliberately tracked software records under `artifacts/`. Scientific
code, tests, configurations, locks and historical reports were not rewritten.
The [machine-readable receipt](cleanup.json) records the checks and scope.

Historical receipts document audits performed while their inputs were available.
Their old local paths are now provenance locators, not retained raw archives.
Public hashes and compact tables cannot reconstruct the deleted original bytes.
Software checks remain possible after a fresh installation; full raw-result
authentication is limited by the missing assets.

The negative selector result and the unfinished practical benchmark keep their
original scope. Sunset adds no experiment, protected evaluation or stronger
scientific claim. No research execution is scheduled.
