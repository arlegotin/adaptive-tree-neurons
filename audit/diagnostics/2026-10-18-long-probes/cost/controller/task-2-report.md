# Task 2 Phase A report

Status: STOPPED at outer receipt-wrapper failure. The actual narrow CPU128 receipt passed and was validated. The engineering cost comparison has NOT started. No cohort run, automatic retry, source/config/test/lock mutation, dependency install, broad test, MPS probe or additional efficacy unit occurred.

## Execution and failure

The existing `.venv` launched one fresh producer with all five BLAS variables1 and MPS fallback0 present before imports. The producer set Torch CPU1/inter-op1, called exactly `backend.probe_backend('cpu', seed=1602030, batch_sizes=(128,))`, required `passed`, called `validate_backend_report`, and wrote the prescribed aggregate:

```python
{'backend_probes': {'probes': {'cpu': raw}, 'comparison_blocks': [],
                    'decision': backend.select_backend({'cpu': raw})}}
```

The public probe itself launched its isolated fresh interpreter. The producer's stdout confirms `passed` and validation. Raw and aggregate bytes are retained separately. The wrapper invoked `/usr/bin/time -l` around that producer to capture native process counters. After the producer completed successfully, that utility printed:

```text
        2.62 real         2.16 user         0.43 sys
time: sysctl kern.clockrate: Operation not permitted
```

The utility returned1. This is an instrumentation failure after successful numerical receipt production. The harness stopped immediately; no cost invocation or cost output directory was created. Zero retries were performed. The receipt outer subprocess duration was **2.630338166 seconds**; this is receipt overhead and is not a comparison-cost estimate. Native time counters beyond the printed real/user/sys values were unavailable through that utility in this sandbox.

`sysctl vm.swapusage` also returned1 with an explicit permission-denied observation. This did not conceal native pressure evidence: `memory_pressure -Q` and `vm_stat` succeeded. The final observation showed64% coarse availability,4,786,629 cumulative swapouts, AC power and39,638,814,720 free disk bytes. Pre/post observations, including failed counters, are retained in the outer measurement. These are resource observations, not a cohort resource decision.

## Provenance

- HEAD: `839590310619a1836c47cc606a16f055abb18cab`; accepted source implementation `b2a9937`.
- Source SHA256: `089b03d13e1950b8e0c1ff54a147a121cf78db66acaaa2dcedd9ebf29afc3543`.
- Backend implementation SHA256: `6f702a34dda4b260fbef75d800ef183fcfeb1f716f266b8895bef50d07189401`.
- Lock SHA256: `13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5`.
- Runtime SHA256: `af9618d1f31b7f55bd414c489f0cf65b3f96547df5b9c57128207a50b7871b68`.
- Raw receipt SHA256: `62c1e83808d41ef7f2056a7d112518f02a89e0da0d765fc26b35964439b2fbc5`.
- Aggregate receipt SHA256: `e506511d50107138c0e94b1341687c5d9ff97e2fecac5b500a0d1946d1a93f43`.
- Runtime: Python3.12.13, Torch2.14.1, macOS26.7.1 arm64.

The preflight records SHA256 for every tracked regular file. The final stop record confirms all tracked bytes unchanged; `git status --short` was empty after execution. Process-scoped caffeinate covered this execution and was terminated in `finally`. No elapsed-time kill was applied by the outer harness; the backend producer's existing bounded240-second timeout was unchanged.

## Retained artifacts and review boundary

All execution evidence is under `environment/long-probes-20261018/`:

- `receipt-producer.py`, `phase-a.py`: exact producer and serial receipt/cost harness.
- `preflight.json`: initial AC/native resources, environment, source HEAD and full tracked-file hashes.
- `receipt-invocation.json`: exact process argv, working directory and environment.
- `cpu128-raw.json`, `cpu128-receipt.json`, `receipt-provenance.json`: passed actual receipt, exact aggregate and identity record.
- `receipt-stdout.txt`, `receipt-stderr.txt`: unmodified producer/time stdout and stderr.
- `receipt-outer-measurement.json`: launch-to-exit clock, returncode and native pre/post observations.
- `phase-a-stop.json`: explicit failed wrapper status, unchanged tracked bytes, zero retries and no cohort.
- `phase-a-inventory.json`: SHA256 and byte inventory of all new regular artifacts and this report, excluding its own self-referential hash. It explicitly lists its own filename.

No cost measurement, complete nine-role member, short/long search, candidate stop counts, cost peak RSS or post-publication cost retained-byte total exists. There is therefore no valid16-member forecast or cost resource gate yet. No E accuracy was reported or used. The intended later cost boundary remains the complete outer measure command through all publication and its final cost-directory bytes, distinct from the implementation's pre-final-seal enclosing clock and byte snapshot.

Parent action required: inspect this instrumentation failure and authorize the next bounded execution. A safe follow-up can reuse this validated receipt and launch the single unstarted cost comparison with a standard-library outer timer plus retained `resource.getrusage`/existing native resource observations, avoiding the denied `/usr/bin/time -l` counter query. This report does not authorize or perform that follow-up.
