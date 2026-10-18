# P13 independent review disposition

18 October 2026. Reviewer: `/root/diagnostic_review`. Read-only held source:
`d7db80ab58a0c30baa3a3ddd6dc9ed24d72f5593`, based on `4cc937d` with approved
contract `2900a65`. The reviewer read the new P13 module/tests/docs and dependent
contracts; verified the two owned-code hashes, six raw-log hashes and two unchanged
lock hashes. The reviewer ran no fits, import probes, tests, edits or extra agents.

## Material finding and one correction pass

P2: source/runtime identity was captured only after preprocessing, basis construction
and solve (`p13_simple.py` held line269). An edit during computation could label
already-executed old code with new bytes. No implemented immutable-worker harness
replaces this adapter check.

Two deterministic regressions stub a source or runtime identity change during a
mocked solve. Both first failed because fit completed instead of rejecting drift;
the original RED log is retained. The correction captures source/runtime before
computation, compares both before success publication, and retains those initial
identities in completed state. The failed-refit fixtures require unchanged prior
state and predictions plus a failed paid-time receipt. Both targeted regressions
then passed. Root explicitly approved this correction; no second review is used.

## Optional deferred hardening

- F identity digests currently require string length64 but do not additionally
  validate hexadecimal characters. Snapshot restoration still requires the external
  trusted raw manifest hash and all saved-array/file commitments. Add canonical
  digest-shape validation with future schema hardening.
- Geometry receipts report encoded/basis widths and parameter counts, but do not
  measure allocation bytes. Byte allocation and actual process peaks belong to the
  pending finite-workload/resource harness. The adapter's widths and timers do not
  qualify a full workload or establish standalone efficiency.

## Declared scope held outside this review

Prepared transfer data and semantic target/proxy lineage remain coordinator work;
the adapter preserves explicitly declared feature names. Complete transformed
workloads, hard process/resource limits, interruption/retry harness, search/cost
attribution and Gate D are separate prerequisites. No prepared data, HPO, full
practical comparison, scientific efficacy or transfer-quality claim follows from
these tiny engineering fixtures. The review found no other material role,
serialization, numerical geometry or declared29-config semantic issue.
