# Method-specific deployment measurements

This Task 10 slice supplies a fresh-process, feature-only deployment worker.
It loads one externally committed closed model snapshot through its existing
adapter and measures CPU prediction including the adapter's F-fitted native
preprocessing. It never fits, reads labels, chooses a configuration or authorizes
a cost claim. Its first actual checks use tiny simple-reference and grown-model
snapshots; every registered family still needs full deployment-workload
qualification under the eventual frozen policy.

The strict request binds scientific package bytes, family, feature schema, model
metadata digest and a feature-only query NPZ. It declares batch sizes 1 and 1024,
warmup calls and sample count. Query X is finite float32 with at least 1024 rows;
row IDs are unique. Unexpected fields, target arrays, source/model drift and
schema/family mismatches fail before method import. The query prefix used for
each batch is fixed and shared by compared requests; no timing-based selection
of inputs occurs. Training precision and native prediction precision remain the
adapter's recorded policy; float32 input does not make a float64 ridge model a
float32 model.

The worker's module imports standard-library code only. Separate measured phases
cover common numerical/frame imports, feature decoding/frame construction,
method-specific import, model load, first prediction, declared warmups and
resident predictions. Native preprocessing is inside every timed prediction.
Output validation/equality checks stay outside those prediction timers. Every
sample must reproduce its batch's first predictions exactly. All durations and
prediction commitments are retained; no fastest-sample selection is performed.

An outer owned-process supervisor must retain the complete fresh-process wall
time, deadline/RSS treatment, process reap and startup environment. The worker's
inner clocks cannot supply interpreter startup or prove strict peak RAM. Thirty
paired randomized blocks, frozen cache/power/thread policy, native qualification,
and an authenticated quality bound remain required before the cost analyzer can
support a deployment claim. Resident batch-1 prediction is the primary deployment
workload; batch 1024 is secondary. Cold model loading is reported separately.
The worker does not install a global OpenMP workaround or import another model
family to initialize a runtime.

Use the already qualified fresh-worker launcher environment: remove an inherited
`KMP_DUPLICATE_LIB_OK` override in the child environment before starting it and
record that removal, as the existing trial bridge does. The worker and native
adapters retain their guards and do not remove/set it after startup. A pinned
dependency's documented environment default remains recorded by its native
adapter. Parent-side fixture fitting can otherwise contaminate a later child's
startup environment; that is a launch-treatment error, not a model-load repair.

Sixteen missing-component RED cases preceded the implementation. The initial
simple-reference/analysis run passed 34 checks. Adding an actual grown-model
fresh-process fixture exposed an inherited parent environment override; that
failure and its receipt are retained. Applying the existing qualified launcher
policy to the test child's environment, without weakening the native guard,
passed 35 affected checks in 12.23 s. P13 loaded no tracked heavy model namespace;
P00 loaded Torch only. Both batch sizes reproduced independently saved fixture
predictions exactly and retained all three resident samples plus one declared
warmup. These are contended engineering observations, not comparative timing
evidence or a claim about real-data quality. See [the receipt](WP10-deployment-worker-receipt.json).

## Paired block coordinator contract

The implemented coordinator is `research/deployment_blocks_v2.py`. It expands
exactly 30 paired blocks into 60 sequential owned-process jobs. The order uses
the existing cost analyzer's seed namespace and RNG law. Both requests bind
identical query bytes, feature schema, batch sizes, warmups and sample count;
their model/family can differ. No seed or fastest block is selected. Complete
request/source/interpreter bindings and per-child deadline/RSS settings enter the
queue identity. Child-only environment treatment matches the qualified trial
bridge, including its recorded removal of inherited OpenMP suppression.

Each worker runs under the existing owned-process supervisor. Live disk and
available-memory proxies can pause the unchanged queue before a child starts.
Every attempt remains paid and terminal statuses remain visible. No automatic
measurement rerun is admitted; interruption/recovery events remain explicit.
The collector authenticates the inactive queue and every stored process/worker
record. One missing member makes the complete paired comparison unavailable;
there is no complete-case filtering or invented duration. Each process block
uses the rounded arithmetic mean of all its declared resident samples as its
batch duration. Thirty process blocks, not the individual resident calls, are
the paired uncertainty units. Cold process, load and resident clocks remain
separate. Pauses and contended host conditions must be disclosed.

This engineering coordinator produces inputs for the existing paired timing
arithmetic. It grants no protected access, hardware/precision/workload admission,
dedicated timing qualification, quality-bound authentication or efficiency claim.
Those require the final frozen measurement policy and complete audit.

Nine missing-component RED cases preceded its implementation. The first affected
run passed 44 checks, including 60 actual fresh P13 worker processes. Two later
failure fixtures exposed an incorrect success status after a nonzero child exit
and an orphan `running` status reported as absent. Both failed before correction.
The final affected run passed 46 checks in 54.31 s, again including the complete
60-process fixture and a deliberately paused partial queue. Every final fixture
child was reaped; missing members prevent paired summaries. The independent
inactive collection retained 31.072220707 s of paid queue jobs containing
27.315774372 s of supervisor-enclosing process clocks. These contended prototype
measurements compare identical mean-model fixtures and establish no speed gain.
Initial and final cohorts are retained separately and are never pooled for
inference. See [the coordinator receipt](WP10-deployment-blocks-receipt.json).
