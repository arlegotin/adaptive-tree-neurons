# Selectively growing tree-neurons

A research proof of concept for testing whether a training-derived score can choose which learned neurons should acquire conditional affine specialists better than random placement at the same capacity budget.

The repository now has a native Python 3.12 core environment and an environment doctor. The remaining implementation follows the specification and gated plans. No predictive benchmark or training result is claimed. The soft affine-stump model belongs to an established gated family; the research target is the allocation procedure and its measured limits, including a negative result.

Start with the [specification](docs/superpowers/specs/2026-10-16-selective-tree-neurons-design.md), then follow the plans in gate order:

1. [Core plan: WP00–WP04](docs/superpowers/plans/2026-10-16-selective-tree-neurons-core.md) — environment verification, algebra, compact layer, score, data/metrics, paired training, optimizer recovery and the first complete exploratory experiment.
2. [Evidence plan: remaining WP02/WP03 scope and WP05–WP09](docs/superpowers/plans/2026-10-16-selective-tree-neurons-evidence.md) — faithful baselines, grouped/temporal data, pilot-derived design, protocol freeze, independent confirmation, justified transfer and reproduction.

The plans contain file/API boundaries, tests, verification commands and acceptance gates. Core Task 1 provides the `doctor` command; later study commands remain unimplemented.

Install the locked core locally with `uv` (Python first tries 3.12):

```sh
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
uv sync --frozen --python 3.12
.venv/bin/python -m pytest -q
.venv/bin/python -m treeneuron.cli doctor --output environment/report.json
```

The core lock resolves Torch 2.14.1, NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, PyYAML 6.0.3 and development pytest 9.1.1. It is scoped to Python 3.12. Runtime reports and acquired source artifacts stay outside Git. Doctor records denied probes as null with reasons, strips serial/UUID fields and returns nonzero for indispensable package import failures. Sandboxed observations can differ from native process observations; MPS availability does not pass the kernel/parity gate in Task 8.

[Source provenance](provenance/sources.json) records exact installed wheel terms and license hashes. [Prior-art status](protocol/prior_art_delta.md) records successful official R12 PDF retrieval and the remaining provisional overlap decision.

[Evidence and machine record](docs/research/2026-10-16-evidence-and-machine-record.md) distinguishes the supplied Linux checks/reference ledger from local discovery. The observed target is an M3 Max with 36 GiB memory, macOS 26.7.1 and about 21 GiB free disk; default native Python 3.14.6 has no Torch; this project uses a separate managed Python 3.12.13. Recheck mutable resource observations before execution.

[DECISIONS.md](DECISIONS.md) records project state and the next step. The [planning review](docs/superpowers/reviews/2026-10-16-selective-tree-neurons-review.md) records the fresh builder check and plan self-review. Work remains on the current branch, `main`.
