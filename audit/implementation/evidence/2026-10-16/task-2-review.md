### Spec Compliance

- ✅ Spec compliant for Evidence Task 2's explicitly phased B0–B6 scope. B6 independently reconstructs the complete selected parent for each candidate, preserves common eligibility and per-ID contrasts, ranks finite G results with deterministic ties, and returns the fixed effective count even when improvements are negative (`src/treeneuron/baselines/greedy.py:34`, `:48`, `:77`, `:136`). Fresh joint continuation is integrated through B6 in `src/treeneuron/baselines/causal.py:277` and exercised in `tests/test_greedy.py:35`.
- ✅ The requested files and corresponding behavior are present: new greedy/registry modules, new greedy/adapter tests, additive spline in the simple module, and causal/practical CLI dispatch. Greedy provenance is explicitly distinct from B5 (`src/treeneuron/baselines/causal.py:180`, `:292`; `tests/test_greedy.py:235`, `:243`). No unrequested practical adapter or general search framework was added.
- ✅ B7 remains pending and `track_ready=false`; the causal exit status certifies the declared implemented scope, not complete causal readiness (`src/treeneuron/baselines/causal_checks.py:34`, `:129`; `src/treeneuron/cli.py:45`; `DECISIONS.md:162`). Required practical methods remain pending with nonzero CLI status, and optional methods have reasoned exclusions (`src/treeneuron/baselines/registry.py:53`, `:70`; `tests/test_adapters.py:76`, `:85`). This matches the Task 3 closure rule.
- ⚠️ Cannot independently verify every indexed bulk checkpoint, the historical preservation hashes, or the reported shell exit statuses from the diff alone. The package includes the 329-entry index, retained outputs, and curation audit (`environment/evidence2-artifact-hashes.json:1`; `environment/evidence2-curation-audit.txt:1`). This review did not repeat that complete hash audit or execute the CLI. These limits do not contradict the inspected implementation and retained results.

### Strengths

- Complete-parent reconstruction is inside the candidate loop; inherited parameter values, Adam moments/steps, sampler permutation/cursor, scheduler, identical training row streams, and fresh joint restart receive real training checks (`src/treeneuron/baselines/greedy.py:136`; `tests/test_greedy.py:35`). Process RNG is restored even after publication failure (`src/treeneuron/baselines/greedy.py:218`; `tests/test_greedy.py:210`).
- Eligibility uses the shared allocator, original rows are retained for contrast lookup, incomplete candidate sets reject, and shortened probes cannot silently enter the ranking (`src/treeneuron/baselines/greedy.py:34`, `:40`, `:48`, `:150`; `tests/test_greedy.py:85`, `:95`, `:185`, `:194`).
- Registration precedes reconstruction/insertion/training. Coordinator records retain known provenance, attempted failures, losers, keep, unattempted candidates, and enclosing attempt costs without reopening terminal children. Secondary publication errors preserve the primary exception and independent cost publication is still attempted (`src/treeneuron/baselines/greedy.py:115`, `:129`, `:155`, `:160`, `:201`; `tests/test_greedy.py:106`, `:144`, `:210`).
- Spline fitting uses F-derived numerical transforms and per-feature knots, an additive basis, and an unpenalized intercept. Prediction overrides the mean estimator path, and strict nonpickled JSON reload checks basis/schema/finite values/width (`src/treeneuron/baselines/simple.py:144`, `:156`, `:180`, `:193`). Behavioral tests cover additivity, original-unit predictions, F quantiles, missingness, categorical rejection, malformed snapshots, and exact reload (`tests/test_adapters.py:9`, `:33`, `:47`, `:92`).
- Retained final evidence is clean: 189 focused passes with one existing skip and 657 full-suite passes with 48 existing skips; neither final output contains warnings (`environment/evidence2-final-focused.txt:4`; `environment/evidence2-final-pytest.txt:11`). Earlier missing-API, provenance, publication, policy, and test-setup failures are retained rather than misrepresented as final success (`environment/evidence2-red.txt:1`; `environment/evidence2-provenance-red2.txt:1`; `environment/evidence2-publication-red.txt:1`; `environment/evidence2-policy-red.txt:1`).

### Issues

#### Critical (Must Fix)

- None found in the reviewed task scope.

#### Important (Should Fix)

- None found in the reviewed task scope.

#### Minor (Nice to Have)

- **M1 — The standalone probe coordinator remains running after completion.** `src/treeneuron/baselines/greedy.py:111` advances the supplied coordinator to `running`; final publication at `:201` writes the completed/failed probe report and event but does not terminalize that coordinator. The concrete owner is `src/treeneuron/baselines/causal_checks.py:98`: it creates a dedicated `probe_paths`, calls the probe function at `:99`, reads the report at `:100`, and never calls `finish_run` for those paths. The retained coordinator manifest confirms `identity.status="running"` and an unavailable finish time despite the completed probe report (`environment/evidence2-causal-runs/eb76c6b1a2e9460fb44df25eda7612e0/manifest.json:1`). This can confuse later manifest-only inventory/audit consumers; the completed `greedy.json` and child outcomes remain accurate, so it does not invalidate selection or measured costs. Make lifecycle ownership explicit: the dedicated CLI caller should finish its coordinator after final observations, including failure handling; future search may intentionally keep a caller-owned enclosing coordinator open until its broader work finishes. Do not unconditionally close an enclosing search run inside `run_greedy_probes` without deciding that ownership contract.

### Focused Checks and Review Bounds

- Reviewed package `review-b29d360..28bcb5c.diff` for base `b29d360` through head `28bcb5c`, in bounded sections; large JSON/evidence sections were parsed and summarized, with repeated stdout/report payloads treated as duplicate evidence. Recovered tool-truncated portions rather than regenerating tests.
- Named external-contract check — independent restoration and continuation policy: inspected `src/treeneuron/train.py:556` and `:702`, plus the canonical-parent/context helpers at `src/treeneuron/baselines/causal.py:42` and `:127` that are outside the supplied changed hunks. Reconstruction copies optimizer tensor stores, sampler state, RNGs and stage metadata; continuation restores the saved RNG and preserves paired policy. No additional execution was necessary.
- Named external-contract check — common eligibility: inspected `src/treeneuron/growth/allocate.py:25`; the shared finite-positive score/contrast predicate matches the greedy helper's use.
- Named external-contract check — spline column layout, finite output, and inherited save/resource behavior: inspected `src/treeneuron/data/transforms.py:115`, `:141`, `:149`, and the inherited methods in `src/treeneuron/baselines/simple.py:55` and `:87` beyond the spline addition's supplied context. Numerical feature columns precede missing indicators, matching the new basis; inverse transformation validates finite original-unit output, and inherited save dynamically dispatches the spline state.
- Named external-contract check — standalone coordinator lifecycle: inspected `src/treeneuron/artifacts.py:177`, `:194` and the one retained coordinator manifest cited in M1. A completed event does not set terminal manifest status; `finish_run` owns that transition.
- No suite, CLI, training, or benchmark rerun; no source, test, artifact, index, HEAD, or branch mutation. Only this requested review report was written.

### Assessment

**Task quality:** Approved, with one nonblocking lifecycle follow-up (M1).

**Reasoning:** The implementation meets the restricted B6 and additive spline contracts with meaningful state, failure, provenance, and numerical tests. Evidence and capabilities preserve the B7/practical gates and avoid scientific overclaims; the remaining caller-owned coordinator lifecycle mismatch should be retained for search/audit integration.
