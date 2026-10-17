# Finite H1 CPU operation envelope

The H1 extension admits F577, F2306 and F9224 with d16, parent h64,
requested growth m16, and the derived ordinary/widened widths through95.
Logical optimizer batches are128,256,512,1024,2048,4096,8192,16384.
Physical equals logical; each update ends at the epoch boundary. A request
above F consumes F actual rows. Prediction chunks are also bounded by F.
The existing CPU diagnostic bounds and MPS/physical-divisor calibration
remain separate. This extension adds no accumulation, model depth, optimizer,
dimension, or MPS operations.

The finite extension covers effective selective growth0..16, ordinary
widths64..95, widening from64 through95, and centered static count0..16 at
width64. Other extended shapes reject. Full first-layer H1 recipes retain
one common batch across forks. The representative envelope measures parent64,
grown16, ordinary95, widened64→95, and static16 operations; live causal
controls also receive their own actual-topology arithmetic/replay checks.

## Producer and consumers

`treeneuron.cpu_envelope.probe_cpu_envelope()` launches a fresh CPU1,
inter-op1, BLAS1, fallback0 interpreter. It returns actual measurements and
numerical checks, or retained partial failure evidence. It does not choose
the study batch. Combine it with a fresh legacy CPU kernel report:

```python
import json
from pathlib import Path
from treeneuron.backend import probe_backend, select_backend
from treeneuron.cpu_envelope import probe_cpu_envelope

probe = probe_backend('cpu', seed=1602030, batch_sizes=(128,))
envelope = probe_cpu_envelope()
assert probe['status'] == envelope['status'] == 'passed'
probes = {'cpu': probe}
report = {'backend_probes': {'probes': probes, 'comparison_blocks': [],
                           'decision': select_backend(probes)},
          'cpu_envelope': envelope}
Path('environment/NEW-RECEIPT.json').write_text(json.dumps(report, sort_keys=True))
```

Run fitting and reload in a fresh CPU1/inter-op1/BLAS1/fallback0 process.
The resolved `execution.backend_report` must identify this receipt; existing
API launchers that supply `train.BACKEND_REPORT_PATH` must supply the same
file. Training validates both kernel evidence and the new envelope.
Checkpoint metadata retains `backend.cpu_envelope={path,sha256}`. Reload
checks that exact file, current whole-package source, core lock, runtime,
actual topology, and current thread/environment settings. Existing relocation
contexts translate paths without changing hashes. Old receipts/checkpoints
retain their original bytes and identities; no source mismatch is waived.

The envelope schema is version1 with check
`h1_cpu_logical_batch_density_envelope`. It contains source/lock/runtime
SHA256 values, explicit environment, logical batches, densities, diagnostic
batch32768, fixed tolerance, exact representative topologies,40 operation
cells,162 timing cells,24 score-parity cells, and resource observations.
Each operation retains requested and actual batch rows, CPU64 raw/conditioned
reference checks, and causal schema2 evidence. Timing cells retain requested
logical/physical batch, actual maximum/tail rows, update count, one discarded
warmup and five measurements. Training epochs include model reconstruction
and optimizer creation; scoring includes CPU64 model copy. These boundaries
exclude checkpoint IO, G selection, and the full paired graph.

Requested16384 and32768 both saturate at actualF9224;32768 is diagnostic-only.
Timing variance and ambient interference remain explicit. A finite saturated
operation measurement is not a universal throughput optimum or a completed
common-recipe calibration. The existing primary128 recipe remains valid.

## ReLU reference semantics, version2

A retained widened95 fixture has one preactivation at row1417/unit66:
CPU64−1.0459109800464716e−8 versus CPU32+3.725290298461914e−9.
Its predictions agree within the fixed forward tolerance, but the ReLU
branches differ and unconditional gradients fail. The original failed
measurements and exact tensors are retained; neither is relabelled as a pass.

For an affine preactivation with d products and bias, let k=2d+2,
S=sum(abs(x_i*w_i))+abs(b), u32=2^-24, u64=2^-53 and
`gamma(k,u)=k*u/(1-k*u)`. The certificate radius is
`(gamma(k,u32)+gamma(k,u64))*S + k*tiny32`, centered on an independent
CPU64 evaluation from the exact float32 operands. The operation count covers
unfused multiplication/reduction/bias; fused evaluation has no larger bound.
The second gamma term covers center evaluation; the tiny term conservatively
covers underflow. Both computed preactivations must lie in that interval,
and the interval must contain zero before differing masks are admitted.

The independent reference preserves its forward ReLU values and computes
its VJP on the actual CPU32 branch. This is reference-only conditioning: the
training model, score, optimizer, inputs, seed and dtype are unchanged.
Forward, loss and branch-conditioned gradients retain their previous
numerical tolerances. Own-gradient first-AdamW formula checks and exact
optimizer/checkpoint replay remain mandatory. Raw unconditional gradient
errors, ambiguity counts, indices, preactivations, centers and radii remain
in each reference check. A false mask outside its interval, an incorrect
backward, or an incorrect AdamW update still rejects.

This affine certificate applies to ordinary rows, widened rows, and ordinary
rows within mixed models. Gated/contrast rows do not receive an affine-only
certificate: a mask disagreement there fails closed because no composed
interval has been established. No global warning suppression or vendor math
change accompanies the extension.

Causal receipts use schema2 and add `actual_batch_size` and `reference_check`.
The check labels `reference_semantics=relu-affine-branch-conditioned-v2` and
retains `raw_gradient_max_normalized_error`,
`conditioned_gradient_max_normalized_error`, `forward_max_normalized_error`,
`loss_max_normalized_error`, and the branch certificate. A raw derivative
failure is permitted only with an explicit certified ambiguity; it is never
reported as unconditional derivative parity.
