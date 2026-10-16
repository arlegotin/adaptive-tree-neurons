# Decision record

## D000 — 16 October 2026: specification and gated implementation handoff

**Inputs:** User-supplied research proposal, cut off 16 October 2026; empty repository on current branch `main`; local read-only hardware/tool discovery. No existing project code or repository guidance was present.

**Evidence:** [Machine/source record](docs/research/2026-10-16-evidence-and-machine-record.md); [builder check and plan review](docs/superpowers/reviews/2026-10-16-selective-tree-neurons-review.md). Actual memory is 38,654,705,664 bytes (36 GiB), logical CPUs 14, OS macOS 26.7.1, native arm64, free disk approximately 21 GiB. Default Python is 3.14.6 without Torch. The supplied numerical checks remain historical Linux evidence; they were not executed locally during planning.

**Decision:** Adopt [the specification](docs/superpowers/specs/2026-10-16-selective-tree-neurons-design.md) and split execution into [a core plan](docs/superpowers/plans/2026-10-16-selective-tree-neurons-core.md) and [a gated evidence plan](docs/superpowers/plans/2026-10-16-selective-tree-neurons-evidence.md). Preserve H1 as the sole initial confirmatory superiority hypothesis. A valid negative or inconclusive outcome satisfies the research purpose when its declared study and audit are complete. Use the current branch; no worktree or branch switch.

**Handoff choices:** The specification explicitly identifies author-chosen gate-type probabilities, positive-control distribution/noise, smooth gate orientation, Energy geometry grouping, date boundary rounding, all-missing fill, growth input conventions and checkpoint/tie rules. Frozen-artifact/documentation commits may differ from the pinned source commit only when execution/scientific input digests match. These are visible draft implementation decisions, not experimental findings or claims of user preference.

**State:** Planning documents written; no WP is complete. No dependencies installed, datasets/weights acquired, predictive models trained, confirmatory teachers/tests generated, or protocol frozen. No scientific result is claimed. The fresh builder review resolved one missing control definition. Plan self-review checks and their outcomes are recorded separately.

**Unresolved execution gates:** Resolve native compatible Python/PyTorch and core lock; test CPU/MPS operations and reference algebra; measure current RAM pressure/storage and resource forecasts; verify consequential current sources/licenses/assets, including R12’s unresolved exact overlap. Runtime budgets, numerical tolerances and confirmatory counts must come from pilot measurements, never guessed to complete the document.

**Next step:** Execute Core Task 1 when implementation begins. Its first real discovery command is `uname -m`; subsequent commands/acceptance are in that task. `python -m treeneuron.cli doctor --output environment/report.json` is a proposed interface that Task 1 must implement, not an existing command. Start no comparative claim before the prerequisites, mathematics and harness gates pass.

Future completed work packages append inputs/hashes, evidence, decision, blockers and next command here. Do not relabel this planning entry as WP00 completion.

## D001 — 16 October 2026: native core prerequisites established

**Inputs/evidence:** Core Task 1 on current `main`; pre-acquisition forecast reserved 5 GiB installation/cache headroom against ~21 GiB free. `uv` 0.11.29 installed managed native CPython 3.12.13 inside `.uv-python`, with `.uv-cache` and `.venv` inside the checkout. No system Python changes. Actual directory sizes: interpreter 70 MiB, cache 763 MiB, environment 754 MiB (filesystem sharing means these are not necessarily additive physical usage); free disk still ~21 GiB. Initial RAM reservation 9 GiB. Current swap used ~6206.62 MiB; measure pressure/swap deltas before work.

**Decision:** Resolve stable packages from the registry under Python 3.12 rather than guessing a Torch minimum or following the stale version selector. Lock SHA-256 `13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5`: Torch 2.14.1, NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, PyYAML 6.0.3, development pytest 9.1.1. These are the spec core packages, with resolver-selected transitive dependencies and setuptools build backend. Native compiled wheel tags are cp312 arm64; pytest is pure Python. License expressions/metadata, bundled notices and their hashes are recorded in `provenance/sources.json`; separate data/weight terms are not inferred.

**Verification:** Expected TDD RED: four tests failed for missing `treeneuron` APIs/CLI. Doctor file tests subsequently passed (4/4); required full suite verification is recorded in the task report. The implemented CLI produces finite JSON, retains denied counters/reasons/return codes, redacts serial/UUIDs, records installed package imports, lock/source digests and fallback settings, and exits nonzero for failed indispensable CPU imports. Runtime reports are ignored: `environment/report.json` (sandboxed), `environment/report-native.json` (permitted native observation). Both observe arm64/Python 3.12.13; CPU imports succeed. Sandboxed sysctl reads fail and MPS availability is false; native observations give 36 GiB memory/14 CPUs and Torch MPS built/available true. Native fallback was explicitly `0`; recommended working set 30,150,672,384 bytes, allocated 0 bytes, driver 475,136 bytes. These counters overlap unified RAM and are not summed. No scientific MPS kernels or parity tests ran; WP00 remains open until Task 8.

**Source status:** Official R12 PDF retrieval succeeded; SHA-256 `f1628581fc83f79a4c32d4013299e56f2cf3693eb50821cb355b8c2c5926cd65`. OpenReview still challenges access. Scoped text/algorithm inspection is recorded in `protocol/prior_art_delta.md`; exact overlap and broad novelty stay provisional. No weights/data, vendor code copying, agreement acceptance, or predictive experiment occurred.

**Next step:** Core Task 2 (strict experiment configuration/run ledger), then numerical gates in plan order. No confirmatory or practical-comparison claim is unlocked.
