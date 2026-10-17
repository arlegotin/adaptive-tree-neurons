# Checkpoint row-history storage correction

Status: DONE for the narrow implementation/evidence task. Root's fresh task
specification/quality review and acceptance bindings remain pending. No genuine
pilot, freeze, protected T, scientific claim, branch/worktree switch, helper,
push, tuning change, or source-identity waiver occurred.

## Source and evidence identity

- Baseline: `680429908735a8077fbfab972468b9d420b7a9f1`.
- Scoped source/tests/representation-doc commit on current main:
  `388c82c59e3dccb6425c55a4a3832dadf80e55fb`.
- Final package SHA256:
  `d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c`.
- Strict execution SHA256:
  `07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c`.
- Strict snapshot:
  `environment/h1-v1/history-storage-fix/strict-snapshot.json`, also copied into
  the audit directory below. Producer and final comparator checked the same
  committed snapshot, including tests, without executable/scientific drift.
- Compact raw evidence index:
  `audit/implementation/evidence/2026-10-17/h1-history-storage-fix/index.json`,
  SHA256 `fc36d2db999babe1563b0ee84197e3006fd4c2207f2436a747568e085a37f166`.
  It indexes 113 files / 31,131,569 bytes, excluding its own bytes and Python
  caches, including all new raw fits, receipts, comparisons, wrappers and failure
  logs. Raw numerical artifacts remain in their original environment/artifact
  directories; small measurements, logs and the strict snapshot are copied into
  this source-specific audit directory.

Before editing, `reference-validation.json` verified the supplied provenance
SHA256 `1ef69743ca5fdbd11a3d4e65799dc3b026f428d7f360854bd93ac9e113e26c95`,
all seven indexed reference files and byte counts, and the original config,
checkpoint pointer, raw events, measurement and backend receipt hashes. Root's
independent 289-file phase-A inventory was retained; no historical file was
overwritten. The old checkpoint was never loaded under changed source. The
old reference's plain tensor/metadata dictionaries were read for comparison.

## Root cause and implementation

The measured defect was full ordered `row_ids` being appended to both
`state.history` and per-stage `events.jsonl`. Every checkpoint serialized that
history and every causal/greedy fork copied it. Replay uses order, cursor,
sampler generator and RNG state; actual pairing and audit consumers read raw
stage events. No historical ID list is needed by training mathematics.

The only production edit is at `train._run::record` (ten added lines, one removed
line). Newly recorded update history removes `row_ids` and adds:

```json
{"row_ids_commitment":{"schema_version":1,"encoding":"ordered_int64_array_hash","count":128,"sha256":"…"}}
```

The hash is the existing `array_hash(np.asarray(ids, dtype=np.int64))`, including
dtype, shape and ordered contiguous bytes. `rows`, loss, learning rate, both
update counters and every other event field remain present. All events remain
in order. The original event object is appended to the raw log without any
change, retaining its full ordered IDs. No new framework, dependency, log
compression, conversion of legacy opaque histories, or cross-source permission
was introduced. The outer checkpoint format and numerical producers/lock are
unchanged. Representation documentation is committed at
`docs/research/2026-10-17-checkpoint-row-history.md`.

## RED/GREEN and affected-path verification

All pytest jobs used unique `/tmp/h1-history-storage-*` basetemps. Following
the first retained setup failure, fresh interpreters explicitly set Torch CPU
threads=1 and inter-op threads=1 before pytest; all BLAS thread environment
variables were 1 and fallback was 0. Jobs were serialized.

Three new regressions exercise real training, full raw events, independent
dtype/shape/ordered-byte SHA construction, exact count/other fields, persisted
best/last histories, exact optimizer/model/sampler/RNG round trips and resumed
continuations, growth prefix preservation, and batch4 versus batch128 history
size on F257. All three failed against the original recording boundary in
`red-2.log` for the intended missing compact representation.

The affected run covered `test_history_storage`, `test_train`,
`test_checkpoint`, `test_checkpoint_cadence`, `test_optimizer_growth`,
`test_greedy`, `test_causal_baselines`, `test_paired`,
`test_paired_diagnostics`, and `test_pilot_calibrate`: **225 passed, nine failed**.
The nine failures were old assertions that directly read history `row_ids`:

- `test_resume_through_growth_matches_uninterrupted[before,after,during]`.
- `test_actual_shapes_interrupted_recovery[lower,upper,centered,direct_leaf,widen94,widen95]`.

Those assertions now compare compact commitments and additionally require full
ordered raw row-stream equality across interrupted attempt plus retry against
the uninterrupted stream. The fresh final run of these nine migrated cases and
the three new regressions passed **12/12 in 12.93 seconds**. Thus all 234 affected
cases were exercised successfully across the passing initial cases and focused
rerun; there was no single clean 234-case rerun. Unchanged passing cases were
not repeated. Paired exploration and pilot graph checks exercise the existing
raw-event consumers, including artifact reconstruction and corruption rejection.

The initial affected run emitted the existing `torch.jit.script` FutureWarning
in `test_node_gate_failure_preserves_completed_b0_b6_checks`; no warning was
suppressed. No bare full suite, standalone full negative-toy replay, unchanged
eight-batch fitting sweep or scientific pilot was launched. `git diff --check`
passed before the scoped commit. Self-review confirmed the change has no
additional consumer, replay or numerical dependency: mutation of raw logs,
omission of history events, use of an unordered hash, or incorrect counts would
fail the new exact assertions and actual comparison.

## Renewed final-package numerical receipt

The existing CPU128 kernel and finite-envelope producers were each run once
after source bytes and the committed execution snapshot were stable. This
renewal is necessary because downstream envelope validation binds the whole
package source hash. The producer itself was unchanged. Total producer wrapper
time was 19.017008959 seconds, using a native fresh CPU1/inter-op1/BLAS1/fallback0
process with original read-only resource checks.

`environment/h1-v1/history-storage-fix/backend.json` SHA256:
`4cfb4983fbe8d152decf7e621dc79b2e77b7f4e3825cb379534ce6a4e05e9e42`.

The separate kernel SHA256 is
`dccf78d3771d04ce39105d8f64cef929552175f11b7f9f68cdcb21a5e33735a1`;
envelope SHA256 is
`9c1e08bac7f7fefb571e29383cb0b734310ae5d4e3acc7c810b5892ff6966d98`.
Scope is the existing kernel batch128 and full finite envelope: 40 arithmetic
operation cells, 162 timing cells, 24 scoring parity cells, the existing three
densities and eight admitted logical batches plus the diagnostic saturation
request. This certifies its existing finite operation envelope; it is not a
fresh study calibration or complete fitting graph. The old receipt remains
unchanged. Producer outputs and wrapper hashes are indexed.

## Actual same-seed replay and exact comparison

The fresh fit used the original batch128/LR.005 recipe, teacher261017100,
data261017101, optimizer261017102 and all other original seeds/configuration.
Only execution artifact/report paths changed. Original canonical config SHA256
is `6003fa2d88b1a62d3aa688579b71e9d9eeaf0f8e18ef00b0091b74ddfa878631`;
new config SHA256 is
`83d286aa2fd6b58fbd808ad8729a4af6c350b7254cb63f3e9b88fcb9f1b05a45`.
Original config bytes, old checkpoint identity and root reference remain retained.

The matching fresh fit is:
`artifacts/h1-v1/history-storage-fix/reference-replay-4/runs/430743ae8de04a1da241816ef7cab5ec`.
It executed 4000 updates / 485,540 examples and selected update3675, exactly as
the old reference. The successful final comparison is
`environment/h1-v1/history-storage-fix/replay-4-completed-2/replay-measurement.json`
with `status=passed`, `failures=[]`, preserved prior postprocessing failure, and
the originating fit measurement path. This final comparison reuses that
completed new-source fit; it performs no further optimizer update.

Exact checks include:

- All 20 F/G/V/E arrays, teacher/split metadata and independent exploratory
  E4096(seed261017990), with original typed hashes and exact array bytes. V
  arrays are diagnostic reference material only; no V selection/evaluation or T
  access was added to training.
- All 21 selected snapshot tensors, including model parameters, every Adam
  moment/age, order, sampler generator and Python/NumPy/Torch CPU RNG, plus the
  complete serialized scalar/container state. Dtypes, shapes, devices and
  tensor bytes agree exactly. MPS RNG state/reason also agree exactly in the
  final matching sandbox runtime: actual MPS unavailable and stored state None.
- Every metadata field except exactly five explicitly recorded differences:
  execution artifact path, execution backend receipt path, config identity,
  package code identity and backend receipt content hash. Optimizer groups,
  scheduler position/horizon/LR, stage selection/counters, preprocessing,
  variance, topology, growth state, data identities, runtime and lock were
  compared without an exclusion.
- F/G/E prediction dtype/bytes/hashes exactly match the original. Typed hashes:
  F `14c3ee2546e07a5fa51911200fef04c30af883e41117e054fbfc9fd4afff2af3`,
  G `8dec179c4f7a2c110e0793124c8cfd2324945dcf7d03a1ce4c0ed054dafc71b9`,
  E `b5d00a05aa6b5421d2195a076a3989614f820af6790c1f3910876d266b47542f`.
- All 4324 raw events compare exactly except the 161 checkpoint digest fields
  required to change with source/config/representation. All 4000 update rows,
  ordered-int64 byte and typed commitments, counts, losses and LRs compare
  exactly. Raw update-stream SHA256 (canonical JSON mapping containing the
  update list) is identical old/new:
  `3bc8a02b02bf4e79839d125215467a3717809bc39172e0c0186fc7fc5bc7848b`.
- New compact history is checked event by event against the complete raw stage
  log. New-source best-checkpoint reload validates current source/receipt and
  preserves every selected tensor, entire compact history and next ordered IDs.

Source identity, paths, representation, checkpoint digests and clocks are
explicit differences. No numerical state is silently excluded.

## Measured storage, time and peak

| Boundary | Old measured/rehashed source | Matching corrected fit |
|---|---:|---:|
| Two actual retained checkpoint metadata files, bytes | 20,613,624 | 2,567,456 |
| Same parent run-directory boundary, bytes | 31,118,960 | 13,072,965 |
| Fresh fit + terminal manifest enclosing seconds | 47.640123083 | 14.143509333 |
| Observed peak bytes | 998,473,728 (old full wrapper) | 413,581,312 (through fit) |

Metadata declined 87.54%; the complete run directory declined 57.99%. The old
phase-A table's 31,787,418 bytes uses the larger candidate-directory boundary,
including teacher/splits/measurement; it is not substituted for the rehashed
same-run boundary above. New metadata files are 1,279,794 and 1,287,662 bytes.
Raw IDs still occupy the unchanged stage trace. Fit clocks exclude imports,
reference verification and subsequent numerical comparison, using the old
wrapper's same fitting boundary. These are actual single-run observations, not
a repeated speed confidence interval. The old wrapper peak includes stage
record construction; the corrected fit peak is recorded immediately after the
fit. Corrected full fit-process peak through comparison/postprocessing was
502,841,344 bytes; the later separate retained-state comparator peaked at
538,935,296 bytes. These memory boundaries are explicitly different.

Actual canonical event-history prefixes (mapping wrapper adds the same fixed
overhead to old and new) demonstrate removal of the past-batch-row factor:

| Updates | Actual examples | Old history bytes | Compact history bytes |
|---:|---:|---:|---:|
|25|3,074|65,510|8,336|
|100|12,170|258,047|31,838|
|1000|121,448|2,573,213|316,158|
|2000|242,770|5,146,061|634,456|
|4000|485,540|10,294,956|1,271,740|

Each update stores a fixed-size commitment and small count/counter fields;
duplication no longer grows with all past minibatch IDs. This fixes the measured
redundancy. A complete resource teacher and actual study readiness still require
root's subsequent calibration; no universal local storage feasibility follows.

## Every retained failure and recovery

`verification-runs.json` records exact logs and scopes. No failed run was
overwritten or relabelled successful.

1. `red-1.log`: three setup failures (live inter-op default versus CPU receipt;
   F257 storage fixture initially retained configured F13). Corrected setup,
   then obtained intended RED.
2. `red-2.log`: three intended missing-representation failures before source edit.
3. `green-1.log`: two passed, one overly strong raw JSON ratio assertion failed
   (6590 versus1192 bytes, below10x). Fixed event overhead and epoch tails justify
   the conservative4x raw-size check; compact sizes remained within100 bytes.
4. `targeted-1.log`: nine legacy history-row assertions described above; 225
   other cases passed. Final migrated/new cases passed12/12.
5. Replay1 stopped prefit: Python tuples versus parsed JSON lists. Canonical
   metadata was identical; no optimizer update ran. Wrapper/log/config/measurement
   preserved with `failed-1` names.
6. Replay2 stopped prefit: exclusive writer refused the existing config path.
   No optimizer update or measurement ran. Wrapper/log retained as `failed-2`.
7. Native replay3 completed4000 updates, then full-state comparison failed on
   reference MPS state None versus native MPS tensor. Independent retained-state
   comparison finds exactly two differences: MPS state type and availability
   reason; all21 common tensors and scheduler match. Native fit12.877012083s,
   peak through fit415,301,632 bytes, run13,073,595 bytes, metadata2,567,456 bytes.
   This native fit remains a failed full-state comparison, separately scoped in
   `native-state-observation.json`. No source/receipt rule was weakened.
8. Sandbox replay4 completed4000 updates and all data/state/prediction/raw-event/
   commitment checks, then summary formatting passed a list to mapping-only
   `canonical_json`. Original wrapper/log/measurement remain retained. No new
   fit was needed to finish the comparison.
9. First retained-state completion again reached a mapping-only writer with the
   summary list payload. Its wrapper/log/output directory remain retained.
   Final completion wraps summary measurements in a mapping and passes every
   comparison, storage summary and same-source reload check.

Initial sandbox Git index-write denial was resolved with authorized scoped
escalation. A read-only `ps` request was denied and no process action occurred.
Some read-only path searches referred to absent files. These caused no source
or historical artifact mutation. There was no automatic approval rejection.

The single source commit contains only `train.py`, the new history tests, the
two required recovery-test migrations and the representation document. Root's
README/DECISIONS/calibration archive changes and Ruling16 remain unstaged.
The source-specific audit evidence is left unstaged for root's review/archival
acceptance. No additional numerical work remains active.
