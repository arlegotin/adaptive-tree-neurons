# Selectively growing tree-neurons

A research proof of concept for testing whether a training-derived score can choose which learned neurons should acquire conditional affine specialists better than random placement at the same capacity budget.

The current repository contains the specification and implementation plans. No project package, dependency environment, predictive benchmark or M3 experiment has been implemented. The soft affine-stump model belongs to an established gated family; the research target is the allocation procedure and its measured limits, including a negative result.

Start with the [specification](docs/superpowers/specs/2026-10-16-selective-tree-neurons-design.md), then follow the plans in gate order:

1. [Core plan: WP00–WP04](docs/superpowers/plans/2026-10-16-selective-tree-neurons-core.md) — environment verification, algebra, compact layer, score, data/metrics, paired training, optimizer recovery and the first complete exploratory experiment.
2. [Evidence plan: remaining WP02/WP03 scope and WP05–WP09](docs/superpowers/plans/2026-10-16-selective-tree-neurons-evidence.md) — faithful baselines, grouped/temporal data, pilot-derived design, protocol freeze, independent confirmation, justified transfer and reproduction.

The plans contain exact file/API boundaries, test assertions, verification commands, acceptance gates and proposed commit boundaries. Commands using `python -m treeneuron.cli` are future interfaces to implement; they are not available yet. Begin implementation at Core Task 1 rather than running a later study command.

[Evidence and machine record](docs/research/2026-10-16-evidence-and-machine-record.md) distinguishes the supplied Linux checks/reference ledger from local discovery. The observed target is an M3 Max with 36 GiB memory, macOS 26.7.1 and about 21 GiB free disk; default native Python 3.14.6 has no Torch. Recheck mutable resource observations before execution.

[DECISIONS.md](DECISIONS.md) records project state and the next step. The [planning review](docs/superpowers/reviews/2026-10-16-selective-tree-neurons-review.md) records the fresh builder check and plan self-review. Work remains on the current branch, `main`.
