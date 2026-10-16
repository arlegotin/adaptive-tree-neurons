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
