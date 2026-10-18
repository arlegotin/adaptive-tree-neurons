# Research v2 reproduction and continuation

Start with the spec, plan and execution ledger linked from this directory. The
ledger is authoritative for which packages and scientific gates have completed.
Historical reproduction is still documented in the root REPRODUCE.md.

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
instances, not16 independent teachers. No production M0/M1 completion is currently
claimed. Native execution requires a fresh backend receipt matching the current
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

CLI commands for WP03 onward currently fail with an owning-package prerequisite
message. Native trainer checkpoints resume exactly. The outer legacy reference
graph uses one declared deterministic same-seed infrastructure rerun under
`--resume`, retaining both transaction directories and all events. OS leases
prevent concurrent execution. Recovery cost bounds and unrecorded process-loss
time explicitly block formal efficiency claims; they are never reported as exact
standalone savings. Source/lock/runtime/backend receipt identity cannot change on
resume. Measured full resource forecasts remain pending.
Read the live ledger before executing production matrices. Protected T has not
been created or accessed by the continuation.

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
