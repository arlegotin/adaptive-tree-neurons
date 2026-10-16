# Whole-core review — Tasks 1–12 / WP00–WP04

Reviewed base `4cacd69` through head `f0eeca8938ae89279bd8eee44b236d575cc1a822`, against the design specification, core plan, `DECISIONS.md`, and `global-constraints.md`. Current package source digest independently observed: `6223991050f82a7e40c35e20112c893ea852988a3abcebc1e4c8ce60077d1d5d`. The supplied review diff is 1,358,491 bytes. Review used bounded passes over the production modules and relevant numerical, recovery, leakage, adapter and integration tests, and structured inspection of the large evidence records. Earlier reviews informed the deferred-item audit; they did not replace code inspection.

This is a review of the completed bounded CPU experiment. The later evidence plan is gated future work. No branch, index, HEAD, source, test, checkpoint or existing artifact was changed. No helper was dispatched. This report is the only file written by the reviewer. No full suite, mathematical verification command, backend calibration or comparative study was rerun.

## Strengths

- The scientific comparison is implemented coherently across modules. `growth/score.py:199` differentiates summed row losses and accumulates signed CPU64 moments with the correct F normalization; `growth/allocate.py:40` establishes a common eligible population and shares per-ID contrasts; `experiments/paired.py:330` independently reconstructs each parent fork and verifies old values, Adam histories, streams and predictions before continuation. The score is treated as a local training heuristic, not a generalization theorem.
- State recovery is substantially stronger than weight-only checkpointing. `train.py:772` selects strict earliest G-best complete snapshots, including update zero; `checkpoint.py:105` validates current and embedded best state before durable pointer publication; `growth/insert.py:190` preserves inherited state and initializes added Adam ages separately. Blockwise widening preserves those age semantics without pretending that an enlarged packed Adam tensor has two scalar clocks (`baselines/causal.py:55`).
- Data and evaluation boundaries are concrete: independent teacher/reference/observation domains, immutable F/G/V data identities, F-only transforms, original-unit predictions, G-only training selection, descriptive V results, and no protected T interface in the runner. Required static, ordinary and placement controls are present. Fresh ordinary/static controls receive two stages and the full declared opportunity (`experiments/paired.py:361`).
- The reported conclusion matches the retained evidence. Both historical studies have 17 completed methods; the primary top/random V MSE values are approximately 0.147531/0.143840 at equal 1,697 parameters. The separate 2D study is correctly described as a different mechanism diagnostic. Historical execution remains pinned to `46d537c` / source `ec52c0…`; the newer small study is explicitly engineering verification. Costs distinguish shared scoring, enclosing study work, unmeasured standalone random work and overlapping memory counters. No superiority, completed confirmation or practical competitiveness claim is inferred.

## Issues

### Critical — none

No Critical defect was identified. The findings below do not establish that the completed default-config historical studies have incorrect numerical results.

### Important — 3

**I1 — Reject scientific growth settings that the paired runner does not execute.**

- References: `src/treeneuron/config.py:59`, `src/treeneuron/config.py:62`, `src/treeneuron/experiments/paired.py:194`, `src/treeneuron/experiments/paired.py:286`, `src/treeneuron/experiments/paired.py:296`, `src/treeneuron/growth/score.py:228`.
- Trigger: a caller changes a valid YAML field such as `growth.policy: keep`, `growth.direction: gaussian`, or `growth.projection_rms_floor: 1.0` and invokes `run`.
- Behavior: all three settings pass resolution and runner preflight and are persisted in the resolved scientific configuration. The runner nevertheless constructs its fixed Adam/eigenvector proposed arm and fixed comparison inventory, and the scorer/allocator use a literal `1e-12` floor. There is no execution read of these config fields. Thus an accepted, hashed configuration can describe a scientifically different request from the computation performed. The fixed inventory is appropriate for this core; accepting incompatible selectors silently is the defect.
- Verification: a focused no-I/O check loaded the actual starter config, changed each field separately, and replaced only `paired.create_run` with a sentinel. All three requests reached run creation. Source inspection establishes the subsequent hard-coded behavior. No experiment was started.
- Minimal fix: reject unsupported nondefaults at the paired execution boundary before creating artifacts, and make the fixed inventory explicit in its resolved execution semantics. For the fixed formula, constrain the RMS floor to its supported value unless a separately declared variant is implemented consistently in scoring and Gaussian allocation. Add entrypoint rejection tests. The existing default-config studies do not require numerical reruns.

**I2 — Preserve ineligible diagnostic rows through finite score-table serialization.**

- References: `src/treeneuron/growth/score.py:211`, `src/treeneuron/growth/score.py:216`, `src/treeneuron/experiments/paired.py:290`, `src/treeneuron/experiments/paired.py:294`.
- Trigger: an individual unit has nonfinite moment statistics or a caught eigendecomposition failure, while the model/loss and other units remain valid. This is an explicitly handled scorer outcome, including the existing test `tests/test_score.py:182`.
- Behavior: the scorer marks the unit ineligible and deliberately retains NaN diagnostic eigenvalues. The paired runner converts every tensor directly to lists and sends the record to finite-only `canonical_json`. Serialization raises before allocation, aborting the whole teacher and preventing the promised diagnostic table from being retained. The valid common eligible count and zero-growth retention paths therefore do not compose with this supported scorer outcome.
- Verification: a tiny in-memory CPU64 score check injected one `torch.linalg.LinAlgError` at the existing eigensolver seam. The resulting statuses were `eigendecomposition_failed, eligible, eligible, eligible`; allocation correctly returned effective count 2. Applying the runner's exact record conversion then failed with `Expected finite JSON value, got float`. No files or study were created.
- Minimal fix: give the exported score table a finite, explicit representation for unavailable diagnostic scalars, preserving per-unit status/reasons and the original table identity. Keep valid rows usable and let allocation proceed from the original table. Validate serialized bytes before opening the output file. Cover mixed valid/ineligible and all-ineligible exports plus summary reconstruction. Do not change the scoring formula or reinterpret nonfinite model/loss runs as valid.

**I3 — Establish known child provenance before training can finalize a failed attempt.**

- References: `src/treeneuron/experiments/paired.py:263`, `src/treeneuron/experiments/paired.py:366`, `src/treeneuron/experiments/paired.py:370`, `src/treeneuron/baselines/causal.py:217`, `src/treeneuron/train.py:736`, `src/treeneuron/artifacts.py:230`.
- Trigger: parent training or a warm/fresh child fit raises, for example an orderly `KeyboardInterrupt`, nonfinite training loss, or a backend/setup error after the child run was created.
- Behavior: `fit_parent`/`continue_fit` record the failure and terminalize the child before returning. The runner supplies coordinator/parent/score links, teacher normalization, V role/counts and other already known study observations only after fit succeeds. For adapter fits it also assigns `open_child` and appends the child's stage record only after `adapter.fit` returns. The terminal failure consequently retains generic placeholders, can still label V as hyperparameter selection, and has no explicit method-stage/coordinator linkage in the study record. The observation API correctly refuses a later terminal update, so the runner's exception handler cannot repair that ordering. The files remain retained, but the failed attempt does not meet the promised attribution/provenance contract.
- Evidence: this follows directly from the fit → `_record_failure` → `finish_run` call path. Existing Task12 tests cover a factory failure before a child exists and a prediction failure after fit succeeds (`tests/test_paired.py:128`, `tests/test_paired.py:190`); neither covers a training failure after child creation. The original I2 fix is present for successful stages and final-prediction failure, but does not close this branch.
- Minimal fix: expose/register each new child with the coordinator immediately after run creation and publish the known study context before training may write terminal status. Record its run/path in the method attempt record at that point. State-dependent values can remain accurately unavailable until observed; do not fabricate them. Ensure growth/adapter metadata used for immutable hashes is established consistently. Add focused parent-interruption and child-training-failure tests that inspect terminal manifests and study linkage. Preserve all historical artifacts and their source identities.

### Minor — 3

**M1 — Bind the audited paired stream to the verified event stream.**

- References: `src/treeneuron/experiments/paired.py:84`, `src/treeneuron/analysis/exploratory.py:35`.
- Status: original Task12 M1 remains open. `audit_pairing` compares duplicated `executed_row_ids`, while the summarizer separately checks events against `stages[*].row_hashes`. Neither requires those representations/counts to agree. Setting a duplicate stream to empty makes the minimum common prefix zero and can still return `passed=True`.
- Consequence: semantic corruption of the duplicated study metadata can bypass the pairing assertion even while the event audit succeeds. This is not evidence that the recorded completed streams differ.
- Fix: derive the paired stream from verified final-stage events, or enforce exact equality among duplicate stream, stage hashes and actual update count before comparing legitimate early-stopped prefixes. Cover empty, truncated and changed duplicate streams without disallowing genuinely different G stopping lengths.

**M2 — Restore process settings even when final publication fails.**

- Reference: `src/treeneuron/experiments/paired.py:412` and `src/treeneuron/experiments/paired.py:436`.
- Status: original Task12 M2 remains open. Observation updates, hashing, `study.json` publication and terminal publication all occur before restoration within the same `finally` block.
- Trigger/consequence: an I/O or serialization exception there skips restoration of `BACKEND_REPORT_PATH`, process RNGs and Torch threads, contaminating a subsequent in-process caller; it can also mask the original experiment exception.
- Fix: place restoration in an unconditional inner `finally`; preserve the original failure and attach secondary publication errors where appropriate. Test a failure during final observation/publication, not only a factory failure inside the main body.

**M3 — Complete the stated gate/leaf diagnostic fields.**

- References: `src/treeneuron/experiments/paired.py:136`, `src/treeneuron/experiments/paired.py:149`, design specification `:298`, core plan `:293`.
- Behavior: selected-state diagnostics retain gate mean/min/max/saturated fraction and, only for centered coordinates, leaf-separation RMS. They omit the specified gate variance/quantiles, `E[g(1-g)]`, input norms and synthetic regime occupancy. Direct-leaf controls omit leaf disagreement even though both leaf modules are available. These omissions also appear in the retained primary and small-integration records.
- Consequence: the basic pipeline and reported negative placement result remain interpretable, but the promised mechanistic diagnostic record is incomplete and treats equivalent coordinate families differently.
- Fix: add a declared finite diagnostic schema covering the readily available F-derived fields and direct-leaf disagreement; record explicit inapplicability for populations/models without gates. Keep oracle regime observations diagnostic-only. This request does not require the future B6 ranking correlation study, new predictive inventories or retrospective rewriting of historical records.

## Deferred-item and prior-fix triage

| Recorded item | Current disposition |
|---|---|
| Task1 failed-probe stdout | Closed: `doctor.py:40` preserves redacted stdout as a failed-probe reason when stderr is absent; regression coverage exists. |
| Task2 derived-count overflow | Closed: `config.py:228` catches float conversion/product overflow and rejects nonfinite derived counts; quarter splits use integer arithmetic. |
| Task2 known manifest observations | Successful/final-prediction paths fixed by Task12; training-failure branch remains I3 above. |
| Task3 status-only/partial reference success | Closed: `verification.py:55` requires real schemas, numerical results, environments and contracts. |
| Task5 serialized metadata consistency | Closed for the recorded findings: control equation/coefficient/distribution/normalization checks are at `controls.py:108`; teacher threshold/prevalence consistency at `synthetic.py:127`; load-path regressions at `test_synthetic.py:312` and `:320`. |
| Task5/6 F-only conversion and raw missing-data types | F-only conversion is integrated in training/scoring. Synthetic types are explicitly complete/finite; raw transfer types remain a separately documented future task. |
| Task7 standalone random cost | Explicitly unmeasured, with honest work counts; no current standalone-speed claim. Future practical costing must implement/measure that path. |
| Task8 capped isolated timing precision | Retained and disclosed; complete-path blocks met their rule. Later pipeline timing/calibration remains a gate, not a completed claim. |
| Task9 enclosing cost | Resolved by Task12's enclosing method/study timers; imports/final index publication and nested validation are explicitly distinguished. |
| Task10 producer/table and widening dependencies | Actual score/allocation integration and Task11 mapped widening are present; no open issue from the earlier dependency-only cautions. |
| Task11 complete Adam-history review defect | Closed: current/embedded publication and reconstruction require complete trained/added records; original update-zero lazy state remains valid. |
| Task11 secondary duplication, spline/B6/B7, practical/raw data, protected architecture | Explicitly deferred/pending capabilities; not falsely represented as executed. |
| Task12 I1 interruption bookkeeping | Closed for its identified window: safety is set only after update and due-G bookkeeping return (`train.py:840`–`:847`); boundary regressions cover it. |
| Task12 I2 metadata correction | Successful-stage ordering and the 46-manifest historical sidecar are present. I3 is the uncovered training-failure path, not a request to rewrite that sidecar or old numerical evidence. |
| Task12 M1/M2 | Still open, retained as M1/M2 above. |

## Chronological ruling review

All ten chronological rulings in `progress.md` are public in D003–D009. None requires reversal on the inspected evidence.

1. **Observed-coordinate additivity (D003):** mathematically appropriate for the intended additive negative control; the implemented equation avoids introducing projection interactions.
2. **Paired axis-to-rotation control (D003):** coefficient, reference and observation transformations are consistent, preserving paired signal/noise while changing alignment.
3. **Reserved metadata rejection (D003):** a narrow explicit guard, correctly distinguished from future source-specific predictor allowlists.
4. **Checkpoint retention (D004):** durable last/best targets and failed/unpublished attempts are preserved; only journaled superseded payloads are pruned. It does not remove losing outcomes.
5. **Small CPU diagnostics (D005):** actual workloads and uncalibrated performance scope are recorded; they do not claim the primary throughput measurement.
6. **Separate added widening blocks (D007):** correctly gives new tensors fresh Adam age while preserving inherited ages.
7. **Pre-freeze implementation order (D007):** resolves a real plan defect: adding execution/analysis/audit code after the real source freeze would invalidate the intended source guards. The revised order preserves protected execution gates.
8. **Widening pointer-identity clarification (D007):** independent validated replacement Parameters preserve the scientific history requirement; literal pointer identity was unnecessary for this control.
9. **Checkpoint cadence amendment (D008):** motivated by retained engineering cost evidence before allocation results; the defined cadence keeps G/stage/growth boundaries and complete event histories. The I1 fix closes its identified unsafe bookkeeping window.
10. **Confirmation cohort floor (D009):** resolves a real prospective plan defect: `K_calc=1` cannot estimate teacher uncertainty. Retaining `K_calc`, flooring at the completed accepted pilot count, rejecting degenerate superiority inference and retaining resource/estimation branches are explicit pre-freeze safeguards. This is not a claim that eight teachers guarantee coverage.

## Verification and evidence limits

- Read actual final logs: `environment/task12-fix1-pytest.txt` reports **596 passed / 48 skipped in 167.62s**; `environment/task12-fix1-focused.txt` reports **115 passed in 38.93s**. The skipped MPS cases are not counted as native training evidence.
- Parsed `environment/task12-fix1-causal.json`: implemented checks passed, 13 causal recipes, with spline/B6/B7/duplication status explicitly bounded. Read the current-source small-study record: 17 completed methods, 22 manifests in its audit evidence, six paired continuation updates, no T.
- Parsed both original completed studies and curated evidence: each has 17 methods and eight warm forks; capacities, two-stage allowances, source identities, top/random risks and enclosing times agree with the public account. Inspected the metadata-only correction structure and historical source separation. The controller's exhaustive 486-new/1,026-old artifact hash audit was supplied evidence; this review did not redundantly hash the full bulk inventory or deserialize old checkpoints under new source.
- Two new focused executions were solely in-memory checks for I1 and I2, both under `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -`. I1 stopped at a mocked run-creation boundary; I2 exercised a tiny real scorer plus exporter conversion with one injected eigensolver failure. I3 and M1–M3 are established by inspected control flow/records; no new failing study was needed.

### Exact focused reproductions

I1 used this no-I/O entrypoint check; the observed output was `accepted: policy keep`, `accepted: direction gaussian`, and `accepted: projection_rms_floor 1.0`:

```python
from pathlib import Path
from unittest.mock import patch
from treeneuron.config import load_experiment, resolve_experiment
from treeneuron.experiments import paired

class ReachedRunCreation(Exception):
    pass

for field, value in [('policy', 'keep'), ('direction', 'gaussian'),
                     ('projection_rms_floor', 1.0)]:
    raw = load_experiment(Path('configs/explore.yaml')).to_dict()
    raw['growth'][field] = value
    with patch.object(paired, 'create_run', side_effect=ReachedRunCreation):
        try:
            paired.run_paired(resolve_experiment(raw))
        except ReachedRunCreation:
            print('accepted:', field, value)
```

I2 used the following scorer/export boundary. It yielded one failed diagnostic row, three eligible rows, effective count 2, then the finite-JSON exception:

```python
from dataclasses import fields
from unittest.mock import patch
import torch
from treeneuron.train import SelectiveRegressor
from treeneuron.growth.score import score_units
from treeneuron.growth.allocate import allocate
from treeneuron.config import canonical_json

torch.manual_seed(61)
model = SelectiveRegressor(2, 4).double().eval()
x = torch.randn(13, 2, dtype=torch.float64)
y = torch.randn(13, dtype=torch.float64)
eigh, calls = torch.linalg.eigh, [0]
def one_failed_unit(matrix):
    calls[0] += 1
    if calls[0] == 1:
        raise torch.linalg.LinAlgError('Injected isolated unit failure')
    return eigh(matrix)
with patch.object(torch.linalg, 'eigh', side_effect=one_failed_unit):
    table = score_units(model, x, y, epsilon=.05, tau=1., adam_epsilon=1e-8,
                        batch_size=7, unit_block_size=2)
print(table.status, allocate(table, 2, policy='adam', seed=3).effective_count)
record = {f.name: (getattr(table, f.name).tolist()
          if isinstance(getattr(table, f.name), torch.Tensor)
          else getattr(table, f.name)) for f in fields(table)}
canonical_json(record)
```

The review invocation wrapped I2 in `torch.random.fork_rng()` and caught the final `ValueError` to print its message; these do not change the reproduced boundary. For I3's regression, an implementer can reuse the existing small paired fixture and interrupt the first main-training Adam step with two parameter groups (the grown top child); the causal receipt probe itself constructs one group. The expected invariant is that the newly interrupted child is already explicitly linked to the top method/coordinator/parent and has the known descriptive V role, counts and normalization. This proposed file-producing test was not run during this read-only review.

## Recommendations

Use one scoped fix wave for the six findings, with focused config/export/failure-provenance/audit/restoration/diagnostic tests. Changes should preserve the existing default numerical recipe and historical execution identities. A small current-source integration can verify the new record contracts; the findings do not justify rerunning the historical full inventories. Run the relevant required checks after source changes and retain their actual outputs.

## Declined to judge

- **Protected T generation/access enforcement, real freeze, confirmatory inference and cohort completion:** unimplemented future EvidencePlan gates; the core explicitly refuses protected phase and makes no completed-confirmation claim.
- **B0 spline, B6/NODE, practical vendor adapters, grouped/temporal raw transfer pipelines and H3 competitiveness:** explicitly staged future work, with current capabilities and claims appropriately restricted.
- **Secondary duplication/symmetry control:** expressly deferred in the Task11 plan/capability matrix; warm zero-outgoing widening is the implemented B2 control.
- **Native MPS complete training/growth recovery:** the selected causal path is CPU, and the public evidence limits native MPS claims to prior kernel-level checks. Full MPS recovery is not represented as completed.
- **Standalone sampled-ID random timing, cold-process latency and calibrated practical HPO budgets:** explicitly unmeasured/future measurements; current recorded shared/enclosing times were judged within their stated boundaries.
- **Population superiority, plateau sufficiency of the 2,000-update limits, and efficacy of the unexecuted 4,000-update amendments:** no such result is claimed; these require the later calibrated independent design. Current negative descriptive G/V evidence was assessed.
- **Full score–probe improvement correlations and future clipping/group-out/extension analyses:** later B6/pilot or separately named variants; this does not excuse the current diagnostic-field omissions in M3.
- **Broad novelty, legal freedom to operate and future asset/license eligibility:** no such claim or acquisition is part of this core; provenance and the R12 provisional hold are explicit. No new external-source/legal assessment was performed.

## Assessment

**Ready for the next evidence stage: With fixes.**

**Counts: 0 Critical, 3 Important, 3 Minor.** The default completed experiment is scientifically restrained and its main algebra, information boundaries, paired state handling and reported negative result are well supported. Close the accepted-config, ineligible-score export and failed-child provenance gaps before building the next evidence machinery on these interfaces; the three minor items can be resolved in the same bounded wave.
