# Selectively growing tree-neurons

A research proof of concept where selected neurons grow two specialists and a switch that blends them for each input. The question is whether a cheap score can choose where to grow better than random placement, using the same added capacity.

The genuine exploratory pilot has retained **15 independent problems, trained twice**, with no model-pair failures. Guided growth averages normalized E MSE **0.0593955**, random placement **0.0590092**, and the upper ordinary parameter bracket **0.0586740**. These descriptive results establish no guided-placement advantage. The original run timed out after a user-requested suspension; the [prospective continuation](docs/research/2026-10-17-genuine-pilot-interruption.md) will complete only the final predeclared problem while preserving the original failed record. There is no independent confirmatory result.

An earlier single-teacher exploratory comparison gave a negative result for the score: on one 16-dimensional synthetic teacher, top-scored growth had validation MSE **0.14753**, versus **0.14384** for random placement at the same 1,697 parameters. Both improved on continuing the 1,153-parameter parent (**0.16066**). Ordinary warm widening to 1,693/1,711 parameters gave **0.15644/0.15881**. This keeps conditional growth interesting on this problem, but supplies no evidence that the score chooses better locations.

Two further [common-recipe calibration problems](docs/research/2026-10-17-common-recipe-exploratory.md) give mixed outcomes: selected growth has a tiny lead over random on one, while both match the parent on the other and an ordinary wider network does better. These remain exploratory, with one repetition per problem. [Finite development controls](docs/research/2026-10-17-finite-development-controls.md) also give mixed results: guided growth wins on one sparse-data draw, while random placement wins on several other conditions. These calibration cases are separate from the genuine pilot, whose primary recipe stays unchanged.

These are descriptive G/V results from one teacher and optimizer seed, with provisional update limits. Top was still improving at its limit. The separate noiseless 2-D control learns the intended opposite slopes; keep, top and random all give validation MSE about **0.001338**, with no useful extra-growth gain. It is a different diagnostic population, not another primary replicate. The [curated evidence](docs/research/2026-10-16-exploratory-paired-evidence.json) records all controls, costs, hashes and limitations; [DECISIONS.md](DECISIONS.md#d008--16-october-2026-exploratory-checkpoint-cost-and-first-paired-study) records the bounded conclusion and next gates.

The [plain-language interpretation](docs/research/2026-10-17-exploratory-interpretation.md) explains the earlier results. Protocol protection, the independent-cohort runner and audit/report code passed implementation review and scoped fixes. The reviewed storage correction reproduces all 4,000 updates exactly while reducing checkpoint metadata from 20.6 MB to 2.6 MB. The complete two-repetition engineering workload took about eight minutes, and [strict scientific readiness](docs/research/2026-10-17-scientific-pilot-readiness.md) preceded the genuine pilot. Completing that pilot record, a supported numeric design, independent confirmation and actual scientific reproduction remain distinct gates.

The repository implements the compact selective affine layer, CPU64 moment score, deterministic CPU training, complete checkpoint recovery, state-preserving growth, F-only preprocessing and serial paired controls. Algebra, native CPU/MPS kernel checks and the CPU causal path have been tested. The soft affine-stump model belongs to an established gated family; the research target is the allocation procedure and its measured limits.

Start with the [specification](docs/superpowers/specs/2026-10-16-selective-tree-neurons-design.md), then follow the plans in gate order:

1. [Core plan: WP00–WP04](docs/superpowers/plans/2026-10-16-selective-tree-neurons-core.md) — environment verification, algebra, compact layer, score, data/metrics, paired training, optimizer recovery and the first complete exploratory experiment.
2. [Evidence plan: remaining WP02/WP03 scope and WP05–WP09](docs/superpowers/plans/2026-10-16-selective-tree-neurons-evidence.md) — faithful baselines, grouped/temporal data, pilot-derived design, protocol freeze, independent confirmation, justified transfer and reproduction.

The plans contain file/API boundaries, tests, verification commands and acceptance gates. `doctor` and `verify-math` are available. Core environment, reference/layer/score algebra and native CPU/MPS operation gates pass. Parent/continuation training and complete CPU checkpoint recovery are implemented and reviewed. One-event growth preserves old optimizer/schedule/RNG state on CPU. Causal ordinary/widening and static-gated controls are implemented and reviewed; the complete serial exploratory study runner and artifact-only report command are available.

Install the locked core locally with `uv` (Python first tries 3.12):

```sh
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
uv sync --frozen --python 3.12
.venv/bin/python -m pytest -q
env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m treeneuron.cli doctor --output environment/current-report.json
.venv/bin/python -m treeneuron.cli verify-math --backend-report environment/current-report.json --output environment/reference-math.json
```

The core lock resolves Torch 2.14.1, NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, PyYAML 6.0.3 and development pytest 9.1.1. It is scoped to Python 3.12. Bulk runs and acquired data/model assets stay outside Git; selected verification logs, provenance and hash manifests are versioned. Doctor records denied probes as null with reasons, strips serial/UUID fields and returns nonzero for indispensable package import failures. Sandboxed observations can differ from native process observations; Doctor runs actual isolated kernel probes; a saved report is revalidated against execution source, lock and runtime. Sandboxed MPS unavailability remains separate from the executed native MPS evidence.

For the measured resident-data d16/h64/F2306 calibration, CPU was faster: about 36.1 ms versus 66.9 ms for MPS across base epoch, CPU64 scoring and grown epoch. The initial recipe uses one CPU thread and physical/logical batch 128. This is a bounded kernel calibration. The complete primary exploratory study subsequently took 611.7 seconds on CPU, including data generation, setup, training, shared scoring and artifact work; calibrated search and practical comparisons remain pending. Two isolated stage timing caps are retained in the [gate evidence](docs/research/2026-10-16-core-gate-evidence.json); all four complete-path blocks met their stability rule.

[Source provenance](provenance/sources.json) records exact installed wheel terms and license hashes. The [focused prior-art revisit](docs/research/2026-10-17-prior-art-revisit.md) narrows the claim: informed-versus-random neuron allocation already has precedent, and R12's full text is accessible. The exact combined rule was not located in the three inspected sources; priority remains unproved.

[Evidence and machine record](docs/research/2026-10-16-evidence-and-machine-record.md) distinguishes the supplied Linux checks/reference ledger from local discovery. The observed target is an M3 Max with 36 GiB memory, macOS 26.7.1 and about 21 GiB free disk; default native Python 3.14.6 has no Torch; this project uses a separate managed Python 3.12.13. Recheck mutable resource observations before execution.

[DECISIONS.md](DECISIONS.md) records project state and the next step. The [planning review](docs/superpowers/reviews/2026-10-16-selective-tree-neurons-review.md) records the fresh builder check and plan self-review. Work remains on the current branch, `main`.

Generate a current CPU-only receipt before running causal checks. The historical Task11/Task12 receipts retain their original source identities and do not authorize the changed backend:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m treeneuron.evaluation.search_worker --backend-receipt environment/current-cpu.json
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m treeneuron.cli verify-baselines --track causal --backend-report environment/current-cpu.json --output environment/causal-capabilities.json
```

That local receipt is an ignored runtime artifact. Use a new output filename if it already exists. The [finite CPU envelope](docs/research/2026-10-17-h1-cpu-envelope.md) documents the additional measured receipt needed for high-density or larger-batch H1 execution. The [original control evidence](docs/research/2026-10-16-causal-control-evidence.json), [costed probe and spline checks](docs/research/2026-10-16-evidence2-capabilities.json) and [NODE checks](docs/research/2026-10-16-evidence3-node.json) retain their source-specific correctness results. NODE's full-size three-update check ends before official checkpoint selection and is ineligible as a quality result. Long NODE instrumentation and warning handling remain conditional work under [Ruling14](audit/implementation/evidence/2026-10-17/ruling-14-conditional-node-instrumentation.md).

The first paired study is explicitly exploratory. Its builder-chosen starting
allowances are 2,000 parent and 2,000 continuation updates, evaluated on G every
25 updates with patience 20. Fresh static/wider controls receive both stages.
All methods use the same logical batch 128 and durable snapshot interval 25;
every G evaluation, stage boundary and safe complete-update interruption is also
snapshotted. Every update event is retained. Legacy callers default to interval 1.
V metrics are descriptive. There is
no protected T access or confirmatory claim.

The executed starter configs and raw receipt bytes remain historical inputs.
For a new exploratory study, create a versioned config with the fresh receipt
path and a new artifact namespace, then use `python -m treeneuron.cli run
--config PATH`. Preserve the original config rather than changing its recorded
source or receipt identity. `report --mode exploratory --study-dir PATH
--output PATH` rebuilds its development metrics from saved artifacts.

`execution.backend_report` and `execution.artifacts_root` are explicit config
fields. The referenced local receipt is ignored evidence, not a shipped hidden
input. On a fresh checkout, run `doctor --output environment/current-report.json`
and put that path in the config; the selected device/batches must match the
config, and strict source/lock/runtime guards still apply. Historical
`environment/report.json` does not match the current producer digest. The existing
CPU-only receipt remains applicable while its bound package, producer, lock and
runtime are unchanged; it does not establish current native MPS recovery or comparisons.

Each new invocation preserves an independent UUID directory, including failures.
`study.json` binds config/source/checkout, receipt and hardware reports, every
fork's parent, score table, float64 contrasts and actual float32 insertion, raw
row streams, G/V predictions, selected states and enclosing costs. The reporter
checks hashes and rebuilds metrics from these artifacts. Shared scoring is timed
once. A standalone sampled random scorer is explicitly unmeasured; required work
counts are not timings. Validation is nested inside training, and process peak
RSS is not incremental per-method memory. Bulk datasets, predictions and
checkpoints stay outside Git; curated scalar evidence and hashes belong in
`docs/research/`.

The executed starter YAMLs remain at 2,000 + 2,000 updates. Because improving
curves reached those limits, `configs/explore-next-allowance.yaml` and
`configs/explore-positive-control-next-allowance.yaml` separately declare a
symmetric 4,000 + 4,000 next-run allowance. They are **unexecuted and calibration
pending**. They change no batch, scoring, checkpoint-selection or patience rule.
Definitive H1 calibration follows the required spline/B6/B7 and search/timing
gates, plus all pre-freeze H1 implementation. Grouped/temporal raw data and
practical adapters gate the separate transfer study. These historical starter configs do not describe the current 8,000-update
common-recipe pilot linked above. The [confirmation-cohort amendment](protocol/amendments/2026-10-16-minimum-confirmation-cohort.md)
prevents a one-teacher design from producing a falsely precise population claim;
its study-specific floor still requires finite-sample uncertainty checks.

The [Task 12 review fixes](docs/research/2026-10-16-exploratory-fix1-evidence.json)
close an interruption-bookkeeping gap and populate coordinator/child manifests
before terminal publication. A small complete CPU study checks that behavior.
The original predictive studies remain pinned to their original source; the
[historical manifest correction sidecar](docs/research/2026-10-16-exploratory-manifest-corrections-fix1.json)
supplies hash-linked metadata overlays without changing their results or files.

The core has now passed its whole-implementation review and scoped fix review.
The [reviewed handoff](docs/research/2026-10-16-reviewed-core-handoff.md) records
629 passing tests, 48 sandbox MPS skips, the final engineering checks and all
review decisions. The exhaustive probe control now exists and charges every attempted neuron.
Search and timing machinery has also passed its [implementation review and fix
review](audit/implementation/evidence/2026-10-16/task-7-fix1-review.md): model
timings share balanced blocks, precision uses complete blocks, and resumed
searches retain cumulative costs. The [bounded fix evidence](docs/research/2026-10-16-evidence7-fix1.json)
remains below its timing-precision requirement. Pilot-design, protected evaluation
and audit code are now implemented and reviewed. The final H1 review and its
scoped fixes close the ledger handoff, CPU density/batch coverage and advisory
cost issues. The full-suite stage retained 958 passes, 2 failures and 9 setup
errors; the final affected checks passed 190 tests after those paths were fixed.
Native checks cover 40 numerical cells, 162 timing cells, 24 scoring comparisons
and 8 high-density parent reloads. The [final review evidence](audit/implementation/evidence/2026-10-17/h1-final-index.json)
preserves each source identity. These are implementation checks. Independent
confirmation and scientific audit remain pending; the early predictive results
above are unchanged. Resident timing stays within the explicit smaller
[Ruling15 scope](audit/implementation/evidence/2026-10-17/ruling-15-resident-timing-scope.md).

A later [narrow storage correction](audit/implementation/evidence/2026-10-17/h1-history-storage-index.json)
preserves full raw sample logs while replacing redundant checkpoint row lists
with versioned counts and ordered hashes. The actual 4,000-update replay matches
all selected model/optimizer/sampler/RNG state, predictions and ordered update
records exactly. Its fitting boundary fell from 47.6 to 14.1 seconds in one run.
This is an engineering observation. The current exploratory pilot record is
linked above; protected independent confirmation remains pending.

### Auditing and portable saved evidence (pre-freeze implementation)

Task 13 supplies audit/report tooling. The genuine pilot has retained the
partial record above; a supported numeric design, full independent cohort and
actual scientific reproduction remain pending. `reports/final.md`
is a blocked pre-study status, not a completed research result. After a reviewed
freeze and complete execution, use:

```sh
.venv/bin/python -m treeneuron.cli audit --protocol protocol/frozen.yaml --study-dir protocol/confirmation/PROTOCOL_HASH --output audit/study.json
.venv/bin/python -m treeneuron.cli report --mode audited --protocol protocol/frozen.yaml --study-dir protocol/confirmation/PROTOCOL_HASH --output reports/final.md
```

The report validates the frozen cohort, every attempt and selected graph (including
B6 losers), checkpoint topology/optimizer/resume state, original protected access,
raw predictions, F variance, original-unit metrics and declared statistical policy.
Incomplete or corrupt evidence cannot produce a success table. Whole teachers,
including unfavorable and zero-growth cases, remain the independent units.
Unknown execution costs remain null with a reason; nested child clocks are shown
separately from the enclosing preparation/execution cost.

Restore allowed artifacts at the same repository-relative locations in a clean
checkout of the pinned execution source. Retain Git history, locks, accepted review
artifacts, numerical receipts, prepared F/G/V files, frozen protocol/history,
confirmation runs and access ledger. The original absolute root remains in the
frozen identity; locator translation resolves the restored bytes and records the
current checkout separately. It does not rewrite config strings or hashes. Run
from a locked environment installed for that checkout, with the declared CPU/BLAS
thread settings and fallback disabled. Redundant editable-install `src/*.egg-info`
is an untracked executable-input failure: preserve generated metadata outside
`src` as documented by the Task 9 install-metadata relocation record; never relax
the source guard. Ordinary fit/predict/resume and protected access remain strict.

Before any amendment or execution-source change, preserve a prospective archive:

```sh
.venv/bin/python -m treeneuron.cli seal-archive --protocol protocol/frozen.yaml --study-dir protocol/confirmation/PROTOCOL_HASH --output audit/archives/STUDY
.venv/bin/python -m treeneuron.cli audit --archive audit/archives/STUDY --output audit/reanalysis.json
.venv/bin/python -m treeneuron.cli report --mode archived --archive audit/archives/STUDY --output reports/reanalysis.md
```

The v1 archive copies a hashed inventory and the original access-ledger prefix.
Later ordinary evidence reads do not change those copied bytes. Archived analysis
uses only originally authorized sealed rows, preserves original source/model
identities, records its current clean analysis source and labels claims
exploratory. It supplies no `FrozenProtocol`, new T access, refit, prediction,
label generation or old-source training resume. Missing a seal before amendment
means archive eligibility is unavailable; restoring source does not undo an
amendment. This prospective-only boundary is [Ruling 12](audit/implementation/evidence/2026-10-17/ruling-12-archive-sealing.md).

R12's full text was retrieved in WP00; the former access obstacle is historical.
The bounded mechanism comparison remains provisional. Actual final claims still
require the closest-prior-art revisit, locked clean reproduction and post-cohort
scientific audit. H1 component evidence does not establish H3 practical advantage;
unavailable strongest models remain explicit, Energy stays separate as simulation,
and Track C references stay outside local measured tables.

Audit-start markers and immutable audit-work receipts retain interrupted or completed audit work. Nested primary-analysis receipts are linked and excluded from the outer total. Archive copying/reanalysis and Markdown rendering have separate copy-work, archive-work and report-work receipts; these variable measurements are outside deterministic report content.

The [implementation review and fixes](audit/implementation/evidence/2026-10-17/task-13-index.json)
are accepted. Failed attempts retain the partial artifacts they actually wrote;
an unknown hard-stop history stays explicitly incomplete. Terminal confirmation
records unsealed archive eligibility until a verified seal exists. Repository-local
seals create a portable binding; external bundles remain available by explicit
saved-evidence path, as recorded in [Ruling 13](audit/implementation/evidence/2026-10-17/ruling-13-live-archive-bindings.md).
The final implementation review and measured scientific calibration are next.

Pass `--archive audit/archives/STUDY` to `amend` to bind the verified prospective
archive digest to the amendment. Without it the amendment explicitly records
`unavailable_no_archive_supplied`. A mismatched or damaged supplied archive
rejects before writing the amendment; original protocol bytes remain unchanged.
