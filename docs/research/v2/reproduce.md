# Research v2 reproduction and continuation

Start with the spec, plan and execution ledger linked from this directory. The
ledger is authoritative for which packages and scientific gates have completed.
Historical reproduction is still documented in the root REPRODUCE.md.

Current scientific state: M1 and M2 are complete; M2 passed the opportunity screen.
The later sixteen-teacher M3 joint-selector study closed Gate B negative.
The four-teacher B0 native-backbone screen also closed negative, admitting no
residual follow-up. The full M0 reference cohort, native-suite qualification,
production practical search, real calibration and protected confirmation remain
unfinished. Each completed study retains its own immutable scientific source;
current implementation HEAD cannot silently replace it. Start with the
[plain status](status.md), [M3 report](../../../reports/research-v2-M3.md) and
[B0 report](../../../reports/research-v2-B0.md).

Use a fresh Python3.12.13 environment:

```sh
UV_PROJECT_ENVIRONMENT=.venv-research-v2 UV_CACHE_DIR=$PWD/.uv-cache \
  uv sync --frozen --python .venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=0
ATN_PY="$PWD/.venv-research-v2/bin/python"
```

The explicit .venv interpreter selects the original qualified local Python. On a
clean machine choose an actual3.12 interpreter and create a new runtime identity;
never assume this path exists. A fresh setuptools editable build may leave source
egg-info beside installed dist-info; inventory and retain generated metadata before
resolving duplicate discovery. Do not reuse another project's VIRTUAL_ENV.

Existing verification commands:

```sh
"$ATN_PY" tests/reference_math_checks.py
"$ATN_PY" tests/reference_score_checks.py
"$ATN_PY" -c 'import torch,pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main(["-q","tests/test_research_runner.py","tests/test_state_v2.py"]))'
```

The v2 CLI can expand strict finite M0/M1 plans without loading observations:

```sh
"$ATN_PY" -m treeneuron.research.cli plan \
  --config configs/research/v2/M0-reference.json \
  --output artifacts/research-v2/M0-plan.json
"$ATN_PY" -m treeneuron.research.cli plan \
  --config configs/research/v2/M1-controls.json \
  --output artifacts/research-v2/M1-plan.json
```

Each output must be fresh. Resolved plans include source/config hashes and exact
seed/role identities. M0 has64 final fits and520 nominal probe trials; M1 has160
positive and96 negative final fits. Positive coordinates share four parent/data
instances, not16 independent teachers. The pinned M1 cohort has completed; a new
execution of these commands has a new source identity and must not replace it.
Native execution requires a fresh backend receipt matching the current
whole-package source, and measured resource admission before a research matrix.
The tiny native fixtures are software tests with deliberately reduced budgets.

Checkpoint training metadata is version2. It declares a birth update for every
named parameter and the unfrozen global-minus-birth Adam age. Both model and moment
tensors use the declared shapes/float32 dtype (grown IDs int64), with nonnegative
second moments. Deserialization cannot infer missing births. Locally assembled
new blocks register only at complete age0/zero moments. Freezing/pruning require
the later explicit applied-update extension; they are not implemented by this
metadata version. Historical schema1 checkpoints require their original source;
there is no silent migration or source-hash rewrite.

The WP03 M2 dispatcher completed its source-bound scientific run under measured
admission. The native-anchor engineering `qualify` command and its strict
qualification-plan dispatch now pass fixture checks and an independently audited
real P13 primary-split command check; see the
[command contract](native-qualification-runner.md). They grant no Gate D or
production-quality eligibility. Freeze/confirm/general-report entry points still
require their remaining production implementations and prerequisites. Separate native adapter,
provisional method/seed and pure analysis APIs are documented in their component
contracts; they do not make those CLI stages complete. The WP07 prepare-data slice
below is implemented independently of baseline qualification. Native trainer
checkpoints resume exactly. The outer legacy reference
graph uses one declared deterministic same-seed infrastructure rerun under
`--resume`, retaining both transaction directories and all events. OS leases
prevent concurrent execution. Recovery cost bounds and unrecorded process-loss
time explicitly block formal efficiency claims; they are never reported as exact
standalone savings. Source/lock/runtime/backend receipt identity cannot change on
resume. M2 has a source-bound measured forecast and rolling live admission;
the full practical program still lacks measured native workload forecasts.
Read the live ledger before executing production matrices. Synthetic protected T
has not been created. Transfer T roles are sealed by the preparation below;
their labels have not been accessed for fitting, selection or predictive evaluation.

One predeclared first M0 pair can be measured independently for resource
qualification after creating the source/runtime-matching CPU receipt:

```sh
"$ATN_PY" -m treeneuron.evaluation.search_worker \
  --backend-receipt environment/research-v2-WP02-cpu.json
"$ATN_PY" docs/research/v2/scripts/measure_reference.py \
  --config configs/research/v2/M0-reference.json \
  --output artifacts/research-v2/WP02/M0-native-qualification
```

The output must be fresh. This executes8 native final roles plus the parent and
all singleton/keep probes for teacher0/repetition0, retains observations and
checkpoints, and emits hash-bound resource measurements. It does not complete
M0, provide a variance estimate, or constitute resource admission by itself.
Its measured observations and seeds remain development evidence permanently.

The planted-control resource qualification uses the full M1 plan's first declared
instance/repeat at gamma2/sigma.1, with five finite fit roles. It does not complete
the full control matrix or decide Gate A. Create a fresh source-bound CPU receipt
and fresh output; do not rerun an existing qualification destination:

```sh
"$ATN_PY" -m treeneuron.evaluation.search_worker \
  --backend-receipt environment/research-v2-control-qualified-cpu.json
"$ATN_PY" docs/research/v2/scripts/measure_positive_control.py \
  --config configs/research/v2/M1-native-qualification.json \
  --output artifacts/research-v2/WP02/M1-native-qualification
```

The qualifier stores actual F512/G256/V256/E16384 arrays, raw teacher coefficients,
all native role states and coordinator predictions/metrics. Its true-contrast
oracle is information-advantaged and diagnostic-only. Every outcome becomes
development information; one instance/repeat is not a population result.

Transfer acquisition retains official ZIPs, license pages, failed request
receipts, exact raw hashes and a complete file inventory. The actual acquisition
manifest/summary is bound by `WP07-acquisition-receipt.json`; raw bytes are local
artifacts. To acquire anew in a fresh default directory, copy the original
receipt-producing scripts, then execute them:

```sh
mkdir -p artifacts/research-v2/WP07/acquisition
cp docs/research/v2/scripts/acquire_uci/download.py artifacts/research-v2/WP07/acquisition/
cp docs/research/v2/scripts/acquire_uci/verify_extract.py artifacts/research-v2/WP07/acquisition/
"$ATN_PY" artifacts/research-v2/WP07/acquisition/download.py --network-context authorized-uci-acquisition
"$ATN_PY" artifacts/research-v2/WP07/acquisition/verify_extract.py
```

Never overwrite a previous acquisition manifest or raw file. A new acquisition
has new receipt timestamps/digest: bind that new manifest in a new copy of
P0-data.json while retaining the six fixed raw SHA256 pins. The checked-in P0
config names this execution's manifest and is not a floating download locator.

Install optional spreadsheet readers in a separate overlay; this leaves core
training dependencies unchanged. Official wheel hashes are in the data lock:

```sh
uv pip install --python "$ATN_PY" --require-hashes --only-binary :all: \
  --target environment/research-v2-data-site \
  -r locks/research-v2/data-requirements.txt
PYTHONPATH="$PWD/src:$PWD/environment/research-v2-data-site" \
  "$ATN_PY" -m treeneuron.research.cli prepare-data \
  --config configs/research/v2/P0-data.json --output data/research-v2
```

Preparation selects CCPP Sheet1, validates all six raw schemas/row/group counts,
stores the concrete five splits and embargo omissions, and separates V/T labels
from fitting-worker views. All output paths must be fresh; partial failure
artifacts and preparation-status.json remain. It does not fit preprocessing or
models, qualify native adapters, authorize secondary development, evaluate T, or
freeze a scientific procedure. The public raw targets are parsed for partition
serialization; sealing is an audited role boundary, not cryptographic nonaccess
to the source tables. Data encoders fit only when an adapter receives F.

Data dependencies were verified against the official
[openpyxl](https://pypi.org/project/openpyxl/3.1.5/),
[xlrd](https://pypi.org/project/xlrd/2.0.2/) and
[et-xmlfile](https://pypi.org/project/et-xmlfile/2.0.0/) release records.

### Full M1 measured admission and completed source identity

The forecast recipe now exists at scripts/admit_control_matrix.py. It consumes a
resolved M1 config and completed positive/negative qualification directories,
verifies their source/runtime/training/profile commitments, and writes a fresh
plan, forecast and proposed admission. Publication is separate. See
WP02-M1-source-604cffb-receipt.json for its original admission and exact paths.
Its scientific checkout remains detached604cffb even as implementation advances.
Never replace its config/source/lock/receipt on resume; use the retained exact
command with --resume only after confirming its previous process has stopped.
The full cohort completed and its report rebuilt byte-for-byte, as recorded
below. M2 dispatch now runs separately under e1b1018. No practical benchmark or
protected confirmation result is implied by either development study.

Ordinary-neural adapter engineering can be checked with
`tests/test_ordinary_neural_contract.py`, `tests/test_ordinary_neural_training.py`
and the explicitly tiny `tests/test_ordinary_neural_native.py` fixtures. The
builder receipt and integration receipt document exact tested source identities,
pinned upstream source/license assets and F-only target-scaling adaptation.
Engineering fixtures are quality-ineligible and do not qualify a full anchor.

After the full M1 worker has stopped, the retained prediction audit can run under
a current qualified core without loading old-source model states:

```python
from pathlib import Path
from treeneuron.research.control_audit import audit_control_study
result = audit_control_study(
    Path('artifacts/research-v2/M1-controls-604cffb'),
    expected_plan_hash='cfe48a305c3c721eae7afe058ffe9af8f27c00dd7321be63359a3999abd6859d',
)
```

An active study lease rejects this call. Save its result only to a fresh artifact
and bind the invoking auditor source/module/runtime/command in a separate receipt.
This recomputes original-unit metrics from retained arrays and row identities;
it does not reexecute models, validate the checkpoint metadata/tensor chain or
regenerate teacher/data bytes. It retains missing statuses and blocks complete
sensitivity when any of the required four-instance/R2 observations is absent.
The full cohort is complete. The audit remains read-only and still rejects an
active worker lease if replayed against any running source-bound study.

## Completed M1 development study

All256 fits of the pinned604cffb cohort completed. The complete raw arithmetic,
no-op checks, inventory and report are recorded in
[WP02-M1-full-receipt.json](WP02-M1-full-receipt.json). With the original local
raw archive available, rebuild to fresh destinations:

```sh
PYTHONPATH=src "$ATN_PY" docs/research/v2/scripts/audit_full_controls.py \
  --study artifacts/research-v2/M1-controls-604cffb \
  --expected-plan-hash cfe48a305c3c721eae7afe058ffe9af8f27c00dd7321be63359a3999abd6859d \
  --output artifacts/research-v2/WP02/M1-audit-fresh
PYTHONPATH=src "$ATN_PY" docs/research/v2/scripts/record_m1_execution.py \
  --study artifacts/research-v2/M1-controls-604cffb \
  --expected-plan-hash cfe48a305c3c721eae7afe058ffe9af8f27c00dd7321be63359a3999abd6859d \
  --output artifacts/research-v2/WP02/M1-execution-fresh
PYTHONPATH=src "$ATN_PY" docs/research/v2/scripts/report_m1_controls.py \
  --audit-dir artifacts/research-v2/WP02/M1-audit-fresh \
  --execution-dir artifacts/research-v2/WP02/M1-execution-fresh \
  --output reports/research-v2-M1-rebuilt.md
```

The retained report rebuilt byte-for-byte. The report helper deliberately binds
the original producer plan; a new source/runtime run is a new identity and is
not silently substituted. The raw archive is local, not publicly uploaded.
Its6353-file inventory supplies relative paths and exact hashes; existing producer
records also contain original absolute locators. Relocation requires an explicit
locator mapping before using these audit helpers, rather than editing raw bytes.
No protected access, model/teacher replay or checkpoint tensor-chain authentication
is implied by the arithmetic report.

## Active M2 and later implementation work

The active scientific checkout is `/private/tmp/atn-research-v2-M2-run` at
e1b1018, package SHA256
`e322da64b518f29a14bde3385bfcf6d683993b717ed4c1fc13a7a5440a1f9a47`.
Its complete native qualification, measured admission and immutable queue precede
execution. [M2 execution](M2-execution.md) records the source/admission contract;
WP03-M2-measured-admission.json retains the forecast. The original local launcher
is `artifacts/research-v2/WP03/M2-e1b1018-launch/launch.py`. It records argv,
environment, process clocks, interruptions and terminal status. Resume only that
unchanged queue after its worker stops, with the same source/config/receipts.

Current implementation HEAD includes later worker/search, provisional protocol
and analysis components. It must not load old source-bound training state as a
current run or reuse an old backend receipt. New native execution requires fresh
qualification; old records retain their original identities. Full M2 collection
and independent prediction auditing run only after the scientific worker is
inactive. The joint selector still requires the resulting Gate A decision.

[Provisional method locks](provisional-method-lock.md),
[fresh seed commitments](confirmation-seed-commitment.md) and
[independent prediction arithmetic](prediction-arithmetic.md) are fixture-tested
components. They grant no calibration/protected access, do not freeze a study,
and do not complete the final evaluator, artifact audit or reporter.

For additional tests, choose a fresh explicit `pytest --basetemp` inside the
retained artifact namespace and reject an existing destination before launching.
This avoids pytest cleanup of earlier evidence and a second physical copy of
large temporary archives. Preserve original failures and same-source recovery
records; do not weaken native memory/backend guards under host contention.

## Completed B0 native-backbone development screen

The [B0 report](../../../reports/research-v2-B0.md) and
[completion receipt](WP07-B0-completion-receipt.json) describe four fresh teachers,
two paired repeats and nine fixed native recipes. All 72 fits completed. The
screen closed negative (zero of four teacher means met the 5% benefit rule,
versus three required). No extra teachers, configurations or residual follow-up
are admitted. The independent arithmetic is development evidence, not independent
training or protected confirmation.

The compact analysis, exact executed operator, read-only auditor, renderer,
resolved queue plan and full source reconciliation are published under
`audit/research-v2/B0/reproduction/`, with hashes in its inventory. This command
regenerates the report and all compact tables using only Python's standard
library, without loading labels or requiring retained model assets. Outputs
must be fresh:

```sh
"$ATN_PY" -P -B audit/research-v2/B0/reproduction/report.py \
  --analysis audit/research-v2/B0/reproduction/analysis.json \
  --expected-sha256 cd401dc768ca6aa496f6e7c611b5b22af8fbde34968a473d60838524c8cd48c0 \
  --output /private/tmp/atn-B0-report-replay.md \
  --tables /private/tmp/atn-B0-table-replay
```

The report must match `reports/research-v2-B0.md` byte for byte, and all four table
files must match `audit/research-v2/B0/`. This verifies compact arithmetic/report
reproduction, not raw-file authentication. The full raw audit requires the local
immutable queue at `artifacts/research-v2/WP07/B0-native-primary-backbones-80afc0d/`
and the held producer commit `80afc0dd9966e74e8e6d251bc8a976335bcd72d5` with
its qualified environment. Its original operator depends on retained historical
seed/exclusion/software assets; the published copy is the executed source, not
a promise that a new checkout can recreate those assets from filenames. No
silent path, source or model migration is supported. Use `-P` to prevent a local
`operator.py` from shadowing Python's standard library. The failed report-only
invocation and corrected safe-path invocation are retained in the raw study.

## Completed M3 development study

See [the M3 report](../../../reports/research-v2-M3.md) and
[completion receipt](WP04-M3-completion-receipt.json). Scientific source53bd029
(e78e027f…) and auditor654a2cc(4dbf2141…) are held separately; their
original immutable raw queues are external study outputs, not included in a clone.
The pinned auditor command remains in the recorded invocation.
The complete sixteen-teacher/R2 study has320final models and640V/Eprediction
vectors, all reconstructed arithmetically. Gate B closes negative; no correction
or confirmation was run. The compact tracked evidence is under
`audit/research-v2/M3/`, including all teacher/procedure risks, all temporary
fractions, descriptive uncertainty, selected stages/sets and the branch decision.
`docs/research/v2/scripts/review_m3_development.py` exports the review from the
exact queue/audit commitments in its recorded invocation. It rehashes consumed
job/selection metadata and summarizes saved local diagnostics; it does not
retrain a model or replace the independent prediction audit. The new summary
checks and retained-prediction tests passed24cases in17.61seconds.
Both descriptive t and whole-teacher bootstrap intervals remain development
quantities after the one predeclared extension, not confirmation guarantees.

## Native-anchor validation survey

[The fixed decision](primary-anchor-development-survey.md) was committed before
reading V labels; the externally committed120-cell plan admitted115existing
completed native predictions. No new fit was run. The local operational planning,
scoring and independent original-unit fsum scripts/logs are bound in
[the receipt](WP07-primary-anchor-development-survey-receipt.json).
The planning attempt that confused an inventory count with its list stopped
before label access; its source/error log remain retained. The corrected plan
verifies the externally committed inventory and audit file hashes. Original
native-generation sources remain explicit rather than migrating old models to
current code. Primary-only V scoring uses the already-tested coordinator API;
independent fsum reconstruction agrees on all805metrics for115predictions.
The [complete table](../../../reports/research-v2-primary-anchors.md) and
`audit/research-v2/primary-anchor-development/` retain every planned anchor/status
and source/prediction commitment. This grants no engineering-to-production
promotion, final Gate D, tuned-family result or protected claim. T remains sealed.

Candidate inspection fixtures: `tests/test_native_candidate_inspection.py`,
`tests/test_native_generation_chain.py`, `tests/test_prediction_replay_v2.py`
and `tests/test_validation_scoring_v2.py` exercise the composed primary inspection
with real owned native/replay subprocesses. Numerical threads must be1before
imports; native `/bin/ps` access is needed for the existing RSS supervisor.
The71-case affected suite passed65.91seconds; initial failures and source/log
hashes are retained inWP08-native-candidate-inspection-receipt.json. Synthetic
sealed fixtures are not official acquisition or predictive evidence.

The new through-M3 exclusion files are listed by
`protocol/research-v2/exclusions/current-known-exclusions.json`. They preserve the
original575-file scan's registry as an unchanged historical input and add17actual
M3/qualification metadata identities. The collection script/input commitments,
sorted-registry loader check and dual identity arithmetic are retained in
WP09-M3-teacher-exclusion-extension-receipt.json. Raw observation arrays were not
opened for this update. Completeness/confirmation authority remain false.

Search-journal fixtures: run `tests/test_native_search_audit.py`,
`tests/test_search_controller_v2.py` and `tests/test_search_v2.py` with the
qualified interpreter and pinned search overlay. The67-case affected result
is recorded inWP08-native-search-journal-audit-receipt.json. This exact declared
journal replay does not grant native-quality eligibility or T access; compose
the separate qualification, generation, validation and protocol evidence first.

The inspected-native callback fixtures are in`tests/test_native_trial_bridge.py`.
Together with candidate inspection, native supervision, search-journal,
controller and semantic-support checks,104affected cases passed43.89seconds.
Use native owned-process inspection permissions and the qualified interpreter;
the actual tiny fits grant no native workload or acquisition qualification.
P06 official scheduling is not implemented by this bridge; it must retain its
separate pinned sampler. See native-inspected-trial-bridge.md.

The follow-up P06 sampler connection is now implemented. Run
`tests/test_native_schedule_worker.py` and the pinned `tests/test_realmlp_search.py`
official-parity checks with the native-process permissions and existing private
overlay. The97-case affected result also covers trial/journal/controller/support
checks. Its actual full native P06 model uses synthetic sealed fixtures only.
See native-official-schedule-worker.md and its source/log receipt; production
qualification, primary selection and final protocol are still required.

The fixed software-verification prerequisite is exercised by
`tests/test_native_implementation_verification.py`;58affected checks including
owned-process and worker guards passed28.30seconds. The real simple-profile
execution supplies software evidence only. See native-qualification-contract.md
for the still-required real native/support registry and final-source executions;
this helper grants no predictive eligibility or protected release.

Immutable read-only staging and its native integration are covered by
`tests/test_immutable_input_staging.py`, native candidate/trial, generation and
checkpoint-supervision tests. The44-case affected suite passed59.93seconds.
Training recovery uses separate copies; same-device inference/input sharing is
explicitly recorded. See immutable-input-staging.md for storage/cost scope.

Joined anchor execution checks are in`tests/test_native_anchor_execution.py`.
The32-case affected suite with software-verification, native inspection and
generation checks passed75.49seconds. Its actual software/native P13 fixture
does not authenticate official data. See native-anchor-execution.md; the complete
raw-data/support/workload registry and production admission are still required.

Prepared-source/raw/sealed-byte integrity uses
`tests/test_prepared_data_lineage.py` plus transfer/access/workload regressions;
52affected cases passed2.44seconds. Protected archives are hashed opaquely with
no NumPy/pandas decoding or metrics. This is distinct from protected evaluation
and does not rerun historical preparation. See prepared-data-lineage.md.

The actual retained-suite integrity replay is in
`artifacts/research-v2/WP07/prepared-lineage-1efc095/`: operator, invocation,
audit and result hashes are recorded inWP07-retained-data-lineage-audit-receipt.json.
It verifies the exact original suite/acquisition hashes and retained preparation
commit without numerical decoding. It does not grant the remaining native/
workload/protocol authority.

The first newly completed NODE V tranche is retained at
`artifacts/research-v2/WP07/primary-node-anchor-development-96707ea/`. Its fixed
five-cell plan/decision preceded V, and its scorer, separate raw arithmetic,
strict metadata review and report generator are retained. Rebuild the report
from saved scores with`report-generation.py`; it opens no labels. The original
survey remains unchanged. See WP07-NODE-Airfoil-completion-and-V-receipt.json
and reports/research-v2-native-anchors-follow-up.md. Do not duplicate the active
Concrete fit: serial session13703 owns the remaining native queue.
