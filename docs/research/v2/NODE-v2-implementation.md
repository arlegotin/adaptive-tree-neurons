# NODE-v2 Implementation Plan

> **For agentic workers:** Execute inline with superpowers:executing-plans. Root explicitly prohibits extra agents; root owns the one final independent review.

**Goal:** Implement approved P12 NODE-v2 with strict portable endpoint state and bounded native engineering parity.

**Architecture:** One new lazy module consumes FitRequest and unchanged pinned low-level NODE/QHAdam kernels; one owned test helper invokes archived official kernels/source bodies independently. Versioned P12 lock binds existing archives/runtime files without editing them.

**Tech Stack:** Existing pinned Python3.12.13/native arm64 CPU, Torch2.14.1/NumPy2.5.4/SciPy1.18.1/sklearn1.9.1/Pandas3.0.6. No installs.

**Spec:** [approved contract2f5a93f](NODE-v2-contract.md), Task7 and root's approval clarifying that every forbidden facade must raise if reached.

## Global constraints and rulings

- Only new research/node_v2.py, P12-owned tests/docs and locks/research-v2/node-v2. Legacy/vendor/core/shared registries and active scientific source unchanged.
- Native/search remain2048/512/1024 trees with frozen100-report/5000-patience policy. Actual runs only engineering16/32trees,<=128updates, all threads1 and source held unchanged while running.
- All-column F-only quantiles, float64 F target statistics/unit constant scale, nativefloat32 labels and float64 inverse are approved explicit adaptations.
- Ruling: root's scoped fixture/no-scientific-fit instruction supersedes whole bare pytest; affected P12/P13/transform/portable contract tests only. Full2048 anchor, prepared data/HPO/GateD/resource/recovery harness remain pending.
- Ruling: use this durable owned ledger instead of modifying a shared execution ledger or creating extra agents. Root already approved inline implementation; no additional plan approval is needed.

## Review focus

1. Missing/typed category collisions and unknown columns must not use G vocab/statistics.
2. First/late G report, rolling reduction order and strict patience must never turn a raw capped endpoint into an eligible native checkpoint.
3. Native failures and partial optimizer entry retain exact seed/config/rows and original error even when receipt publication fails.
4. Structured rehashed snapshots must reject mismatched model/optimizer/quantile/selection geometry and dtype rather than silently cast or initialize.
5. Source/runtime drift and load construction must preserve previous fitted state and caller RNG.

## Tasks

- [x] T1: Write config/lazy-import/F-geometry/role/target tests; observe missing API RED; implement strict config, encoder/quantile/target and pinned identity lock. Run focused GREEN.
- [x] T2: Write model/initialization/full-row/report/rolling/patience/error tests and independent archived native reference. Observe RED; implement bounded native fit with trace and caller restoration. Run tiny sequential native GREEN including actual100 report.
- [x] T3: Write portable endpoint/external hash/structured corruption/dtype/source/runtime/RNG tests; observe RED; implement JSON+NPZ state with exact model/optimizer/window/trace validation. Run affected suite and hold source for one independent review.
- [x] T4: Record final review; material findings get one RED→GREEN correction pass, minors ledgered. Final affected verification, source/lock/evidence hash inventory, commit; preserve isolated branch for root.

Each task cycle retains raw logs and run directories under /private/tmp/atn-node-v2-evidence-20261018 with unique paths; no hidden retries. Package files are edited only between completed native runs.

## Progress

- Root approved complete contract2f5a93f and all four named adaptations before implementation. Source-stage14 archived records/6 runtime pairs byte verified; P13 branch preserved.

- T1: first missing-API run5RED; config/typed F encoding/all-column native quantiles/target boundaries/lazy source helper implemented. Focused5GREEN0.98s; expected native short-F quantile warnings retained. Hand-checked typed token test fixture corrected before implementation (str:1 row is a known category). Root-approved32bit native NumPy seed accepted without modulo; unsupported seeds fail explicitly.

- T2: missing predictor/training API6RED. Native fit implemented without kernel changes;11GREEN35.55s including actual100-update report, source-defined noisy Dataset preprocessing, direct DenseBlock/entmax/QHAdam training, real source-defined checkpoint averaging I/O, full-F/incomplete batches, partial numerical failure traces and RNG/old-model preservation. Native TorchScript/add_ deprecations, short-F initialization/quantile warnings and upstream escape warning retained. No facade callable was invoked. Reference source-body coverage is Dataset/fetch definitions/registry (only custom-data Dataset.__init__ runs), Trainer.average_checkpoints, unchanged imported arch/nn/ODST/utils and official-wheel QHAdam. No whole notebook or trainer logger invocation claimed.

- T3 serialization:20missing-save/load RED, then34GREEN82.36s. Self-review added retained prediction/transform/role binding, explicit indeterminate partial-optimizer labeling and actual loaded-vendor origin authentication (4RED plus1originRED). Fresh-process reconstruction exposed map-order-dependent binding; an unchanged-array positive control exposed scalar-array shape promotion (1RED). Corrected canonical map packing and actual scalar geometry;7targetedGREEN26.59s. Mutation tests now require their specific semantic failure domain, with a valid unchanged reseal as positive control.
- Additional bounded coverage: mixed numeric/missing/category unknown with constant F target matches independently hand-encoded official native path, exact snapshot roundtrip; raw endpoint before any G report roundtrips without a fake selected checkpoint. Failed partial updates have exact positive per-phase nanosecond accounting and an explicit possible-partial flag (phase-cost RED then3boundaryGREEN29.87s).
- Combined source before final CPU-only seed correction passed136tests149.02s. Reading pinned torch/random.py showed manual_seed also calls non-CPU backend seed APIs. A forbidden-backend test reproduced1RED. Use the CPU default generator directly; this preserves native CPU random draws and avoids off-device caller state changes. Final production code passed137tests152.42s; a distinct-callable mock correction then passed the isolated CPU guard5.65s. The final test tree passed137tests/44warnings154.76s. The intermediate aliased-callable mock failed in TorchDynamo introspection; its exact log remains separate from the original off-device production RED. No hidden retry or failed artifact was removed.
- Ruling: static construction guards F/G<=20000, encoded width<=4096 and estimated trainable parameters<=10M are unmeasured input guards before model construction, not finite-workload admission or hard RSS guarantees. They do not reduce training rows. Actual full transformed workloads and process budgets remain separate GateD prerequisites; cost if too narrow is an explicit rejected workload rather than a weakened fit.
- Ruling: caller RNG means this CPU adapter must not seed other device generators. CPU default_generator.manual_seed has the same native CPU seed effect; direct official reference remains CPU numerical-path parity. Cost if wrong is a direct parity failure, checked by the combined suite.

- T3 final hold: combined137GREEN154.76s on final package/test tree. Source identity and all owned source/test/lock hashes are committed in NODE-v2-evidence/receipt.json; every raw log is copied byte-exact and all raw artifacts/symlink aliases are inventoried without following aliases. T4 awaits root-owned single read-only review.

- T4 complete: root-owned single static review found two P2s (caller-controlled construction defaults; missing equality of same-step optimizer copies). Four focused regressions first failed20.90s, then passed16.13s after one combined correction. Final affected141GREEN/44warnings175.15s. No second review; exact Python MT index type/Torch RNG structural validation is deferred, with prediction-only restoration unchanged. Final source/raw commitments are separate from the preserved pre-review receipts in NODE-v2-evidence/final-receipt.json.
