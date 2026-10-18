# Practical program expansion

The immutable metadata planner expands all14mandatory families over six datasets,
five outer splits and two search repetitions:840jobs, partitioned into168primary
and672secondary jobs. Bike uses its last future block as primary; grouped datasets
use fold0. All primary jobs precede secondary jobs. Each job binds the supplied
sealed manifest/split, support, source, dependency-lock and runtime digests.

The same HPO randomization label is deliberately paired across families. Sixty
logical dataset/fold/repetition labels have independently allocated32-bit seeds,
with deterministic collision accounting. Search repetitions remain separate jobs
and cannot become candidates for choosing a favorable seed. No label arrays are
opened by the planner.

Each job retains the32attempt/120minute ceiling and descriptive10/30minute
prefixes. P13 has29unique configurations. Consequently26,880is the conservative
32attempt ceiling and26,700is the finite unique-configuration ceiling; the
corresponding primary counts are5,376and5,340. The1,680/336worker-hour ceilings
are arithmetic upper limits, not measured runtime forecasts.

This is metadata planning, not execution admission. Manifest hashes are checked
but raw files and native support semantics are not qualified here. Secondary jobs
explicitly require a frozen global procedure and all168primary selected-model
hashes. The planner grants no label access, and the existing TransferStore still
rejects secondary development access. Protocol-enforced release, native supports,
full anchor qualification, measured practical resource admission, sequential TPE
and interruption/cost accounting remain pending.

Seventeen missing-interface RED failures became GREEN. The final affected
workload/metric/design/transfer suite passed67tests in2.07s. Tests cover actual
job counts, caps, disjoint tranches, Bike primary assignment, pairing, malformed
pins, changed/duplicate/missing split metadata and immutable returned values.
These are engineering fixtures, with no transfer model fitting or protected
outcome access. scripts/check_practical_plan.py can expand actual prepared
metadata using an explicitly unqualified support placeholder; that output cannot
authorize a production search.
