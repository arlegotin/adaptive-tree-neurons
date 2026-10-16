### Spec Compliance

- ✅ Spec compliant for the scoped fix round, baseline `b877fc7` to head `272e4ea`.
- **ADDRESSED — observed-coordinate additivity:** `src/treeneuron/data/controls.py:194` applies sine and quadratic functions before summing weighted observed coordinates, eliminating mixed feature interactions. The dense two-coordinate regression fixture checks the behavior that the original one-axis fixture missed (`tests/test_synthetic.py:292`).
- **ADDRESSED — ordered schema bound into identities:** `src/treeneuron/data/types.py:130` binds ordered names/schema hash into the assignment identity; `:155` binds them into the dataset identity and expected development manifest. Rename/reorder and stale supplied-manifest cases are covered (`tests/test_access.py:66`).
- **ADDRESSED — serialized control metadata and teacher thresholds:** `src/treeneuron/data/controls.py:108` validates exact coefficient keys/values and `:112` validates equation/distribution/normalization consistency; `src/treeneuron/data/synthetic.py:127` validates thresholds against recorded prevalences. Load-path regression tests cover both prior failures (`tests/test_synthetic.py:312`, `:320`).
- **ADDRESSED — controller's paired-axis rotation ruling:** `src/treeneuron/data/synthetic.py:236` draws the axis-aligned base before rotating coefficients; `:176` and `:286` apply the same stored transform to reference and observation coordinates. The paired teacher regression checks vectors, gates, reference coordinates, nuisance coordinates, normalized signals and noisy labels (`tests/test_synthetic.py:329`). Controls use the matching input transform (`src/treeneuron/data/controls.py:209`); additive+rotated is explicitly rejected (`:238`). Changed rotated semantics have a distinct teacher version, and rotation presence is validated (`src/treeneuron/data/synthetic.py:137`).

### Strengths

- ✅ The fix tests behavior rather than only equation strings: dense mixed differences, schema identity changes, paired transformed populations and unchanged primary byte identities (`tests/test_synthetic.py:292`, `:329`, `:375`; `tests/test_access.py:66`).

### Issues

- **NEW breakage:** None found in the scoped fix diff.
- **Cross-task obligations remain:** The runner/training handoff must apply F-only preprocessing and copy to the required Torch representation; future raw transfer types must support missing predictors before imputation. This fix intentionally does not implement those later-task responsibilities (`task-5-report.md:135`, `progress.md:108`).

### Assessment

**Task quality:** Approved.

**Reasoning:** All three prior findings and the controller's rotation ruling are addressed without changing the primary isotropic population semantics. The paired coefficient/input transform is mathematically consistent and preserves the axis teacher's signal and noise to floating-point tolerance.

**Verification:** Reviewed the fix diff once in three chunks and the appended report/recorded rulings. No tests were rerun: the report records 73 focused and 194 full-suite tests passing without warnings (`task-5-report.md:119`, `:120`), and the scoped diff raised no additional unanswered correctness risk requiring a targeted execution.
