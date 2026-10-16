Missing/empty trained Adam history accepted as lazy state — ADDRESSED.

Finding evidence

- `src/treeneuron/train.py:320` now rejects absent/empty records at positive selected updates and incomplete nonempty records. Selective gate/contrast and widened added blocks require explicit records even at update zero. Ordinary/static original parameters retain legitimate update-zero lazy state. The guard does not equate individual Adam steps with the global update or manufacture replacement history.
- `src/treeneuron/train.py:349` binds serialized parameter names to unique integer identifiers before checking records, rejecting unmatched saved entries. The guard runs before optimizer reconstruction/loading at line 556, in live packing at line 374, and before continuation at line 707.
- `src/treeneuron/growth/insert.py:61` applies the same guard to registered live optimizer parameters, covering insertion/widening preflight without incorrectly requiring the not-yet-registered new group during migration.
- `src/treeneuron/checkpoint.py:112` and `:116` validate current and embedded-best histories before publication. Lines 165 and 168 validate both histories before any optimizer loading during public recovery.
- `tests/test_adam_history.py:61`, `:81`, and `:94` cover deleted/empty records across ordinary, centered/direct-leaf static, old/new widening, and old/new selective growth, plus load/save/preflight rejection and unchanged RNG/pointers. Lines 111, 130, and 143 cover valid lazy originals, mandatory explicit added step-zero records, exact valid replay, and distinct inherited/fresh ages. Line 156 checks embedded-best corruption with recomputed checksums to reach semantic validation.

New breakage in the fix diff

- None identified. The checkpoint-only fixture amendment initializes explicit zero records for its manually added growth parameters and preserves the existing reconstruction assertions. Replacement-Parameter widening semantics remain consistent with root's clarified contract; this fix does not change that representation.

Out-of-scope observations

- None.

Checks

- Reviewed the supplied fix diff once in bounded chunks: d8c5591 → 75e7742, five changed files. No source crawl, git commands, additional agents, suite reruns, or probe reruns.
- Read retained RED summary: 42 failed, 7 passed in 13.97s. Read retained focused GREEN: 175 passed in 51.94s. Read retained final full-suite log: 569 passed, 48 skipped in 150.55s; no warnings shown.
- Read the new CLI capability artifact: implemented checks passed, 13 B1–B5 recipes, B0 mean/ridge verified, spline pending EvidenceTask2, B6/B7 pending, duplication deferred. Source identity is d4e6102013d8e3eb5d2618590d9479bae3783e11d89629ea27cfe62b17066601.
- Read the preservation log reporting 219 original environment artifacts preserved and the unchanged CPU producer digest applicable. No new native/performance evidence or source-hash waiver is claimed. Artifact-index hashes were not independently recomputed in this scoped review.

Fix round verdict: All findings addressed, no new Critical/Important breakage. Spec compliance and code quality approved for this scoped fix.
