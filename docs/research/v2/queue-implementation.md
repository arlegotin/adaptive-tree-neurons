# Finite prerequisite queue implementation contract

This is an internal execution primitive for the already specified M2 DAG, not a
new experiment or an operational public M2 command. The initial M2 graph compiles
and individual model/data/evaluation/local-diagnostic handlers work on fixtures;
production dispatch, conditional admission, metric/gate analysis and measured resource admission
remain separate work. No fitting begins merely because this queue is implemented.

The queue freezes identity, ordered job IDs/kinds/payloads/dependencies and a
canonical digest before work. A stable topological order resolves dependencies;
missing IDs, duplicates and cycles fail. Every job has one exclusive attempt
directory and committed output inventory. Terminal results bind all file bytes,
and dependency results are authenticated before reuse. Source/runtime checks
remain a required caller-supplied check before creation and each job. No model
checkpoint selection can rewind this outer history.

The inventory is exact: new files/directories, changed bytes and symbolic links
invalidate reuse. Failed/unsafe worker outputs remain retained but unused; their
failure result is published separately in a fresh coordinator directory, so one
bad output does not prevent unrelated work from reaching a terminal status.

Ordinary unavailable prerequisites skip their dependent nominal jobs visibly.
Failed prerequisites produce explicit missing dependents; unrelated jobs continue.
An explicit frozen `all_terminal` dependency policy is used only for aggregators
that must see every nominal candidate status. In particular, the small-pair
oracle receives ineligible pairs rather than skipping the whole oracle because
dead unit7 appears in seven nominal pairs. An eligible pair failure makes that
oracle missing; it cannot select an apparently best pair from an incomplete search.
No missing prediction receives an invented numerical loss. Numerical or dependency
failures do not change seeds or retry. Resource exhaustion/interruption permits
one same-payload, same-seed retry in a fresh directory, retaining both attempts
and costs. A second infrastructure failure closes that logical job as missing.
Recovering an orphaned running event requires the exclusive OS lease, retains a
wall-time upper bound including unknown downtime, and never calls that bound an
efficiency measurement. Worker checkpoint resume may be added only through a
validated handler; the generic fallback is the authorized single rerun.
Closing a second lost worker uses a new coordinator publication directory and
can itself be retried without starting a third scientific worker.

The coordinator owns publication, terminal status and its original failure. A
failed result write must not replace the worker exception. Consumed attempt time
never decreases. Job timers are disjoint work intervals inside the containing
queue timer; standalone procedure costs later traverse prerequisite ancestry
once per procedure. Caches cannot erase prerequisite costs. The queue itself
makes no standalone cost claim. Startup/import time remains the CLI's enclosing
process measurement responsibility.

Tests use harmless temporary-file handlers to exercise actual ordering,
unavailability, tampering, numerical failures, interruption/retry and paid history.
They are not predictive results or native-workload qualification.

`compile_initial_graph` authenticates the complete resolved M2 plan and expands
2,472 jobs:8 data,48 parent,32 score,8 horizon-binding,1,168 fit,8 head,
1,184 evaluator,8 local-diagnostic and8 pair-oracle jobs. This represents exactly
the initial1,176 nominal procedures plus8 G-selected subset-oracle evaluations.
The compiled identity explicitly says execution is unimplemented. Conditional
fallback/extension admission, production dispatch, resource qualification and Gate A
remain pending; no scientific M2 fits have run.
