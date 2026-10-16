# Core Task 3 report — 2026-10-16

Implemented on existing `main` from `baf30ed`; no branch/worktree switch, subagents, training, predictive experiment, source acquisition, package layer/score implementation, or MPS execution. Read Task 3 brief, global constraints, preserved A/B recipes, spec §5, TDD/writing-good-tests and verification-before-completion skills. Per dispatch instruction, no plans, progress ledger or DECISIONS.md edits; this report records the local evidence for the supervising agent's gate decision.

## Implementation

- `tests/reference_math_checks.py`: standalone Appendix A, CPU float64, one thread, seed 1602026, N=37/d=7/h=5, exact preserved draw order and original eigensolver sign. Shared-tensor forward/gradient identity, scalar-head collapse, exactly centered insertion, zero contrast gradient, analytic gate gradient, descending tiny gate-only step, soft-hard identity and ReLU discrepancy bound.
- `tests/reference_score_checks.py`: standalone Appendix B, fresh CPU float64 process, one thread, seed 1602027, N=41/d=5/h=4. Literal dense normalized reference score, sum-of-row-loss derivative, symmetric moments, largest-coordinate-positive eigenvector sign, RMS contrast scaling, added-parameter normalization, positive rescaling/permutation invariance, analytic gradient, and one fixed-parent/fixed-contrast AdamW gate step (LR=1e-6/epsilon=1e-8/decay=0).
- Standalone success prints finite JSON containing environment, shapes, seed, constants, errors, norms and losses; B also emits contrasts/gradients/scores. Explicit AssertionError checks survive Python optimization and propagate nonzero exit; failures do not emit success JSON.
- `src/treeneuron/verification.py` invokes each checkout reference through `sys.executable` in a fresh process, records script SHA-256/returncode/report/reason, and preserves caller RNG/default dtype/thread state. Nonzero, malformed/nonfinite JSON, missing files, timeouts and missing passed status cannot become success.
- CLI `verify-math [--output PATH]` always prints aggregate JSON and optionally saves the same report. Exit 0 means current reference CPU checks passed. `scope=reference_cpu_only`, `complete=false`; package_layer_math (Task 4), package_score_math (Task 7), cpu_backend and mps_backend (Task 8) explicitly remain `not_implemented`. The existing doctor command is preserved and covered by the full suite.

## TDD / execution evidence

1. Added subprocess/boundary tests first. `.venv/bin/python -m pytest tests/test_reference_checks.py -q`: RED, **7 failed, 4 errors in 1.87s**, specifically absent reference scripts, missing verification module and unregistered verify-math command. Fixture errors wrap real missing-script exit codes, not syntax errors in the tests.
2. Implemented the scripts/helper/CLI. Same command: GREEN, **11 passed in 11.75s**. Tests check environment/thresholds/historical values by tolerance, exact centered zeros, real optimizer-step assertion propagation, caller-state isolation, optional output, and CLI failure for real failing/malformed/failed-status child scripts.
3. Ran both standalone scripts independently: each exited 0 with JSON; saved independent runs as `environment/reference-a.json` and `environment/reference-b.json`. Ran `.venv/bin/python -m treeneuron.cli verify-math --output environment/reference-math.json`: exit 0, complete=false, reference_cpu_only. These ignored JSON artifacts remain local.
4. Initial full suite: `.venv/bin/python -m pytest -q`, **73 passed in 23.07s**. Strengthened numerical assertions for deterministic contrast sign and the hand-derived historical score (8.941316578e-7 / [1e-6 × 12]). A deliberate temporary removal of score denominator factor 2 produced RED: **1 failed, 1 passed, 9 deselected in 2.65s**, observed 0.14902194296710355 versus required approximately 0.0745109714833. Restored the exact original production file after this mutation check.
5. Final full suite after all changes: `.venv/bin/python -m pytest -q`, **73 passed in 14.16s**, no warnings/failures/skips. `git diff --check` passed. Only task-owned files staged.

## Actual local environment and numerical results

Native arm64, macOS-26.7.1-arm64-arm-64bit; Python **3.12.13**, PyTorch **2.14.1**, CPU **torch.float64**, **1 Torch thread** for each check. These are new local observations, distinct from historical Linux x86-64 / Python 3.13.5 / Torch 2.10.0+cpu results retained unchanged in the evidence record.

| Appendix A quantity | Local result |
|---|---:|
| Forward max abs | 8.881784197001252e-16 |
| Shared-coordinate gradient max abs | 1.3322676295501878e-15 |
| Scalar-head collapse max abs | 1.7763568394002505e-15 |
| Growth preservation max abs | 0.0 |
| Analytic gate gradient max abs | 2.7755575615628914e-17 |
| Initial contrast-gradient norm | 0.0 |
| Initial gate-gradient norm | 0.4451935794578945 |
| Tiny gate step size | 6.919488255511828e-5 |
| Loss before / after | 18.735481890132764 / 18.735468175897527 |
| Soft-hard gap max abs | 8.881784197001252e-16 |
| ReLU bound max violation | 6.661338147750939e-16 |

| Appendix B quantity | Local result |
|---|---:|
| Rescaled gate gradient max abs | 1.6653345369377348e-16 |
| Rescaled score max abs | 1.734723475976807e-17 |
| Permuted score max abs | 2.0816681711721685e-17 |
| Analytic gate gradient max abs | 5.551115123125783e-17 |
| Selected unit | 0 |
| Selected gradient / contrast norm | 0.4498939250427178 / 0.12533186496102378 |
| Loss before / after | 6.973686776382635 / 6.973685882251051 |
| Predicted / observed Adam loss decrease | 8.941316578026212e-7 / 8.941315847721398e-7 |
| Relative prediction error | 8.167754798008742e-8 |

All required A/B error/invariance checks are below 1e-12; centered zeros are exact; both gate-only steps descend; B relative error is below 1e-4. Small error differences from historical Linux evidence fit the declared tolerances. This establishes local reference algebra only, not package parity, generalization, allocation quality, training-kernel correctness, MPS availability/performance or MPS validation.

## Evidence SHA-256

| File | SHA-256 |
|---|---|
| environment/reference-a.json | `2bfbdbc3fb4e4ae70e24c2320917d080857707a5aadc111d1ab74c0a395e561b` |
| environment/reference-b.json | `20f06dc44a4626dcd230a9bd844838c19e165914bf34105d940d600306997faa` |
| environment/reference-math.json | `0a7a01d36448e064179b410429069d459e47e2fd127365b11c2788bec66d6099` |
| tests/reference_math_checks.py | `e14dc49d6654223033a511ad1605df0c2b7ff5f70715b3f2d9be31f8e772fb22` |
| tests/reference_score_checks.py | `f75f7308970aeace6ab614ffc8bf0736db4e698a7bc784b9f5557701fb3810e2` |
| uv.lock | `13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5` |
| docs/research/2026-10-16-evidence-and-machine-record.md | `2bf263b911b0591ba8cfd61e65d3bd8a2180a7d7b566a143adc83dcb1e26ac79` |

No blocker observed. Later Tasks 4/7/8 must replace their pending CLI gate statuses with actual independent checks; reference success must never be mistaken for those gates. References are checkout test scripts, so an installation lacking the source tests reports failure rather than inventing a pass. Next command for reproducibility: `.venv/bin/python -m treeneuron.cli verify-math --output environment/reference-math.json`.

## Commit

`a75197a` — `test: reproduce reference algebra and normalized score checks` (five task-owned tracked files, 417 insertions). Initial sandbox staging/commit failed because `.git/index.lock` was denied; authorized escalated retry succeeded. This ignored task report is intentionally local, as are the environment JSON artifacts. No unrelated tracked changes were included.
