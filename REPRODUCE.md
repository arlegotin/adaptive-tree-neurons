# Reproduction

There are three distinct tasks: check the implementation, authenticate the saved
research, or run a new experiment. A successful software check does not establish
the allocation hypothesis.

## Install the measured core

The measured platform was a native M3 Max with 36 GiB RAM, macOS 26.7.1,
Python 3.12.13 and PyTorch 2.14.1. CPU float32 training and CPU float64 scoring
were used for the reported experiments. MPS kernel checks exist separately;
no predictive MPS advantage is asserted. Other platforms need their own checks.

```sh
git clone https://github.com/arlegotin/adaptive-tree-neurons.git
cd adaptive-tree-neurons
uv sync --frozen --python 3.12
uv run --frozen python -m treeneuron.cli --help
```

`uv.lock` retains the exact resolved dependencies. Optional NODE and search
environments have separate records under `locks/`; the core setup does not
install them. Full baseline experiments may require those separate environments.

## Check the software without the research archive

The README's layer example uses only PyTorch and the local package. Run its code
with `uv run --frozen python` or activate `.venv` before using plain `python`.
The layer returns preactivations; apply ReLU in the surrounding network.

```sh
uv run --frozen python tests/reference_math_checks.py
uv run --frozen python tests/reference_score_checks.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
uv run --frozen python -m pytest -q \
  tests/test_algebra.py tests/test_growth.py tests/test_score.py tests/test_probe_diagnostic.py
```

These authored checks need no benchmark download, model weights, paid API or
ignored experiment artifact. They exercise affine-tree/gated equivalence,
insertion, score derivatives and the diagnostic's bookkeeping. Checkpoint,
training and optimizer tests additionally build temporary synthetic cases and
fresh backend receipts.

The complete suite can be attempted with the same thread settings and
`uv run --frozen python -m pytest -q`. It includes native MPS and optional
environment paths that can skip when unavailable. Historical whole-suite output
was not all green: it recorded 958 passes, 2 failures and 9 setup errors, followed
by 190 passing affected checks after scoped fixes. The latest longer-probe
implementation passed its 55 affected tests. Those are source-specific records,
not a claim that every later checkout's whole suite passes. Publication checks
and their actual outputs are recorded in [publication scope](docs/PUBLICATION.md).

## Inspect and authenticate recorded results

| Study | Summary | Machine-readable analysis |
| --- | --- | --- |
| Earlier 16-problem/R2 pilot | [Report](reports/final.md) | [Research account](docs/research/2026-10-17-complete-exploratory-pilot.md) |
| Fresh 8-problem/R2 long-search diagnostic | [Findings](docs/research/2026-10-18-long-probe-diagnostic.md), [all candidates](reports/long-probe-diagnostic.md) | [Independent arithmetic](audit/diagnostics/2026-10-18-long-probes/reproduction/results/independent.json), [reviewed analysis](audit/diagnostics/2026-10-18-long-probes/reproduction/results/analysis.json) |

The public analyses expose method risks, paired effects and aggregation. They
allow inspection of the reported comparisons. Complete authentication requires
the external original observations, per-row predictions, event logs, checkpoint
payloads and source-bound backend receipts identified by the inventories.
Hashes identify those assets; they cannot reconstruct absent bytes.

For the latest diagnostic, use the retained archive and its recorded execution
source to run:

```sh
python -m treeneuron.pilot.probe_diagnostic analyze --output PATH_TO_RESTORED_COHORT
python -m treeneuron.pilot.probe_diagnostic report \
  --output PATH_TO_RESTORED_COHORT --destination NEW_REPORT.md
```

The reporter requires a fresh output filename and verifies source, dependency,
runtime, receipt and artifact identities. Restore original locators where
required; never edit a historical manifest to make changed bytes appear valid.
The diagnostic's scientific execution commit was `5e46ce1`, with implementation
at `b2a9937`. Its [replay record](audit/diagnostics/2026-10-18-long-probes/reproduction/coordination/task-3-report.md)
documents exact comparisons and the permitted normalization of run locators.
The unchanged generated report's SHA-256 is
`9baa18c96a55a0023adf469984b8f5bebf948d1985a73675fbde115dc5e5335b`.

Historical scripts and inventories retain their original machine paths. Some
refer to completed `.superpowers` coordination workspaces. Their durable copies
are identified by `archive_path` in the publication audit indexes. A path string
is provenance, not an instruction to recreate another user's home directory.

## Run a new experiment

Create a separate output namespace and new seeds. Generate an actual receipt
for the intended backend and batch support; the historical receipts do not
authorize a changed environment or package source.

```sh
mkdir -p environment
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
uv run --frozen python -m treeneuron.evaluation.search_worker \
  --backend-receipt environment/new-cpu-receipt.json
```

Copy an exploratory config, bind its explicit `execution.backend_report` to the
new receipt, and choose a new `execution.artifacts_root`. Review its data,
training and resource requirements before running `python -m treeneuron.cli run
--config NEW_CONFIG`. Existing configs are historical or proposed recipes, not
portable approvals for an automatic multi-hour cohort.

The [research design](docs/superpowers/specs/2026-10-16-selective-tree-neurons-design.md)
and [decision record](DECISIONS.md) define F/G/V/T information roles and the
unmet confirmation gates. The latest diagnostic used separate E observations
and never accessed protected T. A new exploratory run must not be called a
confirmation of a method adjusted after seeing the old evaluation results.
