# Contributing

Useful contributions include a reproducible bug report, a cheaper allocation
policy with matched controls, a faithful baseline adapter, or a clearly scoped
reproduction on another platform. Open an issue before proposing a new study.

Use Python 3.12 and the locked setup in [REPRODUCE.md](REPRODUCE.md). Run the
focused tests for the affected behavior; report backend, dtype and exact command.
Include a regression check for a numerical or state-recovery fix.

Keep historical protocols, reports, source bindings and failed-run records
unchanged. New experiments need separate output roots, explicit seeds and
prospective comparison rules. A new implementation must not inherit the old
experiment's scientific result merely because its interface is compatible.

Do not commit credentials, environments, downloaded weights, external datasets
or full training archives. Retain upstream notices for any copied source. Original
software contributions use AGPL-3.0-only and original prose uses CC BY 4.0 as
specified in [LICENSING.md](LICENSING.md); separately licensed material must be
identified explicitly.
