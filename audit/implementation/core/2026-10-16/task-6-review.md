### Spec Compliance

- ✅ Spec compliant: all seven requested implementation/test files are present; the two additional package initializers are minimal scaffolding. `FitRequest` and `Predictor` match design §11 exactly (`src/treeneuron/baselines/api.py:9`).
- ✅ F-only population preprocessing, median/all-missing handling, constant flags, fixed indicator policy, reserved unknown categories and original-unit target inversion satisfy design §7 and the clarified task scope (`src/treeneuron/data/transforms.py:85`, `src/treeneuron/data/transforms.py:114`). Strict reconstruction validates schema, numeric state, category tokens and reserved metadata names (`src/treeneuron/data/transforms.py:146`).
- ✅ Metrics implement design §9's raw and F-variance-normalized risk, explicit zero-variance reason, separate noise-free risk, group macro/worst-group and tail errors, with shape/nonfinite rejection (`src/treeneuron/evaluation/metrics.py:15`).
- ✅ Mean/ridge use F only, return finite original-unit vectors, persist preprocessing/estimator state and disclose capabilities/unavailable counters (`src/treeneuron/baselines/simple.py:38`, `src/treeneuron/baselines/simple.py:67`, `src/treeneuron/baselines/simple.py:88`, `src/treeneuron/baselines/simple.py:110`). Ridge leaves the intercept unpenalized and supports alpha zero through least squares (`src/treeneuron/baselines/simple.py:115`).
- ✅ Transfer loaders/source allowlists, prespecified regime selection, HPO, inference, protected evaluation and complete execution resource measurement remain downstream responsibilities; their absence is not a Task 6 defect. The WP02 recommendation is explicitly limited to this task's synthetic/harness scope (`task-6-report.md`, “WP02 decision and remaining boundaries”).

### Strengths

- Schema/order checks and copied JSON state make preprocessing reproducible; tests cover complete synthetic width, missingness, unknown categories, roundtrip isolation and project metadata rejection (`tests/test_transforms.py:22`, `tests/test_transforms.py:31`, `tests/test_transforms.py:43`, `tests/test_transforms.py:124`).
- Baseline tests use forbidden G inputs and compare predictions against an independently assembled original-target augmented design, checking alpha 0, 0.7 and 100 with an unpenalized intercept (`tests/test_simple_baselines.py:13`, `tests/test_simple_baselines.py:34`).
- Metric tests distinguish row-weighted and group-macro risk, reject mixed group labels and exercise finite aggregation of very large squared losses (`tests/test_metrics.py:23`, `tests/test_metrics.py:61`, `tests/test_metrics.py:68`).

### Issues

- Critical: none.
- Important: none.
- Minor: none.

### Assessment

**Task quality:** Approved.

**Reasoning:** The implementation is focused, separates preprocessing/evaluation/adapters cleanly, and implements the task's declared edge cases without crossing information boundaries. Recorded evidence reports 65 focused and 259 full-suite passes without warnings; no tests were rerun because code review raised no specific unresolved correctness risk requiring another execution.

**Review checks:** Read the supplied diff and binding design §7, §9 metrics and §11, task brief/context/global constraints, and implementer report. No broader codebase inspection, protected-data access, commits, branch operations or subagents.
