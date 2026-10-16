# Selectively growing tree-neurons

A research proof of concept for testing whether a training-derived score can choose which learned neurons should acquire conditional affine specialists better than random placement at the same capacity budget.

The first complete exploratory comparison gives a negative result for the score: on one 16-dimensional synthetic teacher, top-scored growth had validation MSE **0.14753**, versus **0.14384** for random placement at the same 1,697 parameters. Both improved on continuing the 1,153-parameter parent (**0.16066**). Ordinary warm widening to 1,693/1,711 parameters gave **0.15644/0.15881**. This keeps conditional growth interesting on this problem, but supplies no evidence that the score chooses better locations.

These are descriptive G/V results from one teacher and optimizer seed, with provisional update limits. Top was still improving at its limit. The separate noiseless 2-D control learns the intended opposite slopes; keep, top and random all give validation MSE about **0.001338**, with no useful extra-growth gain. It is a different diagnostic population, not another primary replicate. The [curated evidence](docs/research/2026-10-16-exploratory-paired-evidence.json) records all controls, costs, hashes and limitations; [DECISIONS.md](DECISIONS.md#d008--16-october-2026-exploratory-checkpoint-cost-and-first-paired-study) records the bounded conclusion and next gates.

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

The core lock resolves Torch 2.14.1, NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, PyYAML 6.0.3 and development pytest 9.1.1. It is scoped to Python 3.12. Runtime reports and acquired source artifacts stay outside Git. Doctor records denied probes as null with reasons, strips serial/UUID fields and returns nonzero for indispensable package import failures. Sandboxed observations can differ from native process observations; Doctor runs actual isolated kernel probes; a saved report is revalidated against execution source, lock and runtime. Sandboxed MPS unavailability remains separate from the executed native MPS evidence.

For the measured resident-data d16/h64/F2306 calibration, CPU was faster: about 36.1 ms versus 66.9 ms for MPS across base epoch, CPU64 scoring and grown epoch. The initial recipe uses one CPU thread and physical/logical batch 128. This is a bounded kernel calibration. The complete primary exploratory study subsequently took 611.7 seconds on CPU, including data generation, setup, training, shared scoring and artifact work; calibrated search and practical comparisons remain pending. Two isolated stage timing caps are retained in the [gate evidence](docs/research/2026-10-16-core-gate-evidence.json); all four complete-path blocks met their stability rule.

[Source provenance](provenance/sources.json) records exact installed wheel terms and license hashes. [Prior-art status](protocol/prior_art_delta.md) records successful official R12 PDF retrieval and the remaining provisional overlap decision.

[Evidence and machine record](docs/research/2026-10-16-evidence-and-machine-record.md) distinguishes the supplied Linux checks/reference ledger from local discovery. The observed target is an M3 Max with 36 GiB memory, macOS 26.7.1 and about 21 GiB free disk; default native Python 3.14.6 has no Torch; this project uses a separate managed Python 3.12.13. Recheck mutable resource observations before execution.

[DECISIONS.md](DECISIONS.md) records project state and the next step. The [planning review](docs/superpowers/reviews/2026-10-16-selective-tree-neurons-review.md) records the fresh builder check and plan self-review. Work remains on the current branch, `main`.

The current causal checks use an explicit CPU-only receipt after a shared-source documentation update:

```sh
.venv/bin/python -m treeneuron.cli verify-baselines --track causal --backend-report environment/task11-parent-backend.json --output environment/causal-capabilities.json
```

That local receipt is an ignored runtime artifact. On a fresh checkout, generate current doctor evidence first and pass its output path. The [control evidence](docs/research/2026-10-16-causal-control-evidence.json) distinguishes executed diagnostic recipes from pending spline/B6/B7 controls. The [evidence plan](docs/superpowers/plans/2026-10-16-selective-tree-neurons-evidence.md#execution-order-before-the-source-freeze) now requires all confirmation/analysis/audit code to be committed before the definitive pilot and real source freeze.

The first paired study is explicitly exploratory. Its builder-chosen starting
allowances are 2,000 parent and 2,000 continuation updates, evaluated on G every
25 updates with patience 20. Fresh static/wider controls receive both stages.
All methods use the same logical batch 128 and durable snapshot interval 25;
every G evaluation, stage boundary and safe complete-update interruption is also
snapshotted. Every update event is retained. Legacy callers default to interval 1.
V metrics are descriptive. There is
no protected T access or confirmatory claim.

Run the CPU-only study with the explicitly selected, source-validated receipt.
The older Task11 training receipt lacks the math aggregate's schema header;
create a derived envelope (the original receipt and all probes stay unchanged):

```bash
.venv/bin/python - <<'PYCODE'
import json, hashlib
from pathlib import Path
source = Path('environment/task11-parent-backend.json')
r = json.loads(source.read_text())
assert r['schema_version'] == 1 and 'schema_version' not in r['backend_probes']
r['backend_probes']['schema_version'] = 1
r['derived_from'] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'change': 'Add required aggregate schema_version=1; all probe/decision/comparison content unchanged'}
Path('environment/task12-cpu-math-receipt.json').write_text(json.dumps(r, indent=2, sort_keys=True) + '\n')
PYCODE
```

```bash
.venv/bin/python -m treeneuron.cli verify-math --backend-report environment/task12-cpu-math-receipt.json --output environment/task12-math-final.json
.venv/bin/python -m treeneuron.cli verify-baselines --track causal --backend-report environment/task11-parent-backend.json --artifacts-dir environment/task12-causal-runs --output environment/task12-causal.json
.venv/bin/python -m treeneuron.cli run --config configs/explore.yaml
.venv/bin/python -m treeneuron.cli run --config configs/explore-positive-control.yaml
.venv/bin/python -m treeneuron.cli report --mode exploratory --study-dir artifacts/exploratory/STUDY_UUID --output environment/exploratory-summary.json
```

`execution.backend_report` and `execution.artifacts_root` are explicit config
fields. The referenced local receipt is ignored evidence, not a shipped hidden
input. On a fresh checkout, run `doctor --output environment/current-report.json`
and put that path in the config; the selected device/batches must match the
config, and strict source/lock/runtime guards still apply. Historical
`environment/report.json` does not match the current producer digest. The existing
CPU-only receipt remains applicable while those producer files and the runtime
are unchanged; it does not establish current native MPS recovery or comparisons.

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
practical adapters gate the separate transfer study. No repeated inventory run
or variance pilot is claimed here. The [confirmation-cohort amendment](protocol/amendments/2026-10-16-minimum-confirmation-cohort.md)
prevents a one-teacher design from producing a falsely precise population claim;
its study-specific floor still requires finite-sample uncertainty checks.

The [Task 12 review fixes](docs/research/2026-10-16-exploratory-fix1-evidence.json)
close an interruption-bookkeeping gap and populate coordinator/child manifests
before terminal publication. A small complete CPU study checks that behavior.
The original predictive studies remain pinned to their original source; the
[historical manifest correction sidecar](docs/research/2026-10-16-exploratory-manifest-corrections-fix1.json)
supplies hash-linked metadata overlays without changing their results or files.
