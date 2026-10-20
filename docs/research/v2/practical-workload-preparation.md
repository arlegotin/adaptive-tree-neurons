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

Explicit native-quality mode separately fixes one model seed per
dataset/fold/repetition, shared across all14families and every trial. Its canonical
labels end in `'fit'`, replacing `'HPO'`, under root2026101802. Store `fit_seed64`,
the worker's signed-range31-bit `fit_seed`, `fit_seed_bits=31`, and
`fit_seed31_collision_counter`. Native-quality plans declare
`native_fit_seed_policy='paired-signed31-v1'`. Allocate
model seeds after all60HPO library values, excluding those and prior model
labels, using the separate `'library31'` derivation namespace. Existing HPO and
fit-root64 values remain unchanged. This prospective compatibility repair changes
planned initialization numbers before any native-quality production search;
[the amendment](2026-10-20-model-seed-range-amendment.md) retains its reason and
verification. The primary inventory checks
planned model seeds against each search and retained trial, and rejects even
self-rehashed job seed changes. Distinct architectures consume randomness
differently; shared labels do not promise identical initial tensors.

Explicit `native_quality=True` retains32attempts and allows each native fit to
finish without a wall cutoff. Its10/30/120minute prefixes include only models
completed by each deadline. P13 has29unique configurations. Consequently26,880is the conservative
32attempt ceiling and26,700is the finite unique-configuration ceiling; the
corresponding primary counts are5,376and5,340. Native-quality mode has no worker
hour ceiling; actual measured forecasts are required. Default legacy mode still
serializes its original120minute limit and1,680/336hour ceilings for historical
compatibility. Those limits do not govern the prospective results-first view.

This is metadata planning, not execution admission. Manifest hashes are checked
but raw files and native support semantics are not qualified here. Secondary jobs
explicitly require a frozen global procedure and all168primary selected-model
hashes. The planner grants no label access, and the existing TransferStore still
rejects secondary development access. Sequential TPE, paid queue/retry accounting,
native budget translation and inventory checks have component implementations.
Production integration, full native support/workload qualification, resource
forecasts and protocol-enforced release remain pending.

Seventeen missing-interface RED failures became GREEN. The final affected
workload/metric/design/transfer suite passed67tests in2.07s. Tests cover actual
job counts, caps, disjoint tranches, Bike primary assignment, pairing, malformed
pins, changed/duplicate/missing split metadata and immutable returned values.
These are engineering fixtures, with no transfer model fitting or protected
outcome access. scripts/check_practical_plan.py can expand actual prepared
metadata using an explicitly unqualified support placeholder; that output cannot
authorize a production search.

The native-quality fit-seed extension observed three RED failures, then78affected
checks passed in26.41seconds. This includes actual file-backed168-primary
search fixtures, rejection of consistent unplanned search/trial seeds, and a
hand-derived canonical hash check. These are metadata/search fixtures, without
native training or protected access. See
[the seed-binding receipt](WP08-practical-fit-seed-receipt.json).
