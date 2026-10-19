"""Full2048/depth6 actual interruption/recovery proof; never a quality result.

Run under a pinned source with one numerical thread and an outer RSS supervisor.
The fresh output directory is argv1. Deliberately retain every attempted fit.
"""
from pathlib import Path
import copy
import json
import os
import sys
import time

started=time.perf_counter_ns()
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron.baselines.api import FitRequest
from treeneuron.research import node_v2 as node

source=node.source_identity()
node._write(root/'started.json',dict(argv=sys.argv,cwd=os.getcwd(),source=source,
    operator_sha256=node._sha(__file__),utc_ns=time.time_ns(),quality_eligible=False,T_accessed=False))
rng=np.random.RandomState(2026101907)
F=rng.normal(size=(1025,16)).astype(np.float64);G=rng.normal(size=(65,16)).astype(np.float64)
fy=1.+.2*F[:,0]+.1*F[:,1];gy=1.+.2*G[:,0]+.1*G[:,1]
np.savez(root/'inputs.npz',F=F,G=G,F_y=fy,G_y=gy)
cfg=dict(schema='research/v2-node-config',family='P12',mode='native_anchor',execution='bounded_memory_v1',
    feature_schema=dict(columns=[f'x{i}' for i in range(16)],categorical=[]),parameters={})
def request(name):return FitRequest(F,fy,G,gy,17,cfg,{'max_updates':105},root/name)
print('reference:105 native full-shape updates',flush=True)
reference=node.NodeV2Predictor().fit(request('reference-fit'))
reference.save(root/'reference-snapshot')
original=node._optimizer_step;calls=0
def interrupt(optimizer):
    global calls
    calls+=1
    if calls==103:raise KeyboardInterrupt('Declared recovery proof before update103')
    return original(optimizer)
print('interruption: checkpoint100, then interrupt before103',flush=True)
node._optimizer_step=interrupt
try:
    try:node.NodeV2Predictor().fit(request('interrupted-fit'),training_checkpoints=root/'checkpoints')
    except KeyboardInterrupt:pass
    else:raise AssertionError('Controlled interruption did not occur')
finally:node._optimizer_step=original
pointer=node._read(root/'checkpoints/latest.json');assert pointer['step']==100
print('resume: raw checkpoint100 through105',flush=True)
recovered=node.NodeV2Predictor().fit(request('resumed-fit'),training_checkpoints=root/'checkpoints',resume=True)

def state_digest(state):
    value=copy.deepcopy(state)
    for row in value['trace']:row.pop('elapsed_ns')
    arrays={};packed=node._pack(value,arrays)
    return node._digest(dict(payload=packed,arrays={k:node._array_identity(v) for k,v in arrays.items()}))
assert state_digest(reference._state)==state_digest(recovered._state)
np.testing.assert_array_equal(reference.predict(G),recovered.predict(G))
recovered.save(root/'resumed-snapshot')
manifest_sha=node._sha(root/'resumed-snapshot/manifest.json')
restored=node.NodeV2Predictor.load(root/'resumed-snapshot',expected_manifest_sha256=manifest_sha)
np.testing.assert_array_equal(restored.predict(G),recovered.predict(G))
assert node.source_identity()==source
failure=node._read(root/'interrupted-fit/fit-receipt.json')
assert failure['executed_updates']==102
result=dict(schema='research/v2-node-full-shape-recovery-proof',status='passed',source=source,
    operator_sha256=node._sha(__file__),inputs_sha256=node._sha(root/'inputs.npz'),
    F_rows=1025,G_rows=65,width=16,trees=2048,depth=6,effective_batch=512,initialization_rows=1000,
    reference_updates=105,interrupted_applied_updates=102,resume_step=100,resumed_final_updates=105,
    state_digest_without_elapsed_times=state_digest(recovered._state),exact_training_state=True,
    exact_predictions=True,exact_closed_reload=True,final_snapshot_sha256=manifest_sha,
    current_payload_count=len(list((root/'checkpoints').glob('checkpoint-*/arrays.npz'))),
    total_process_body_ns=time.perf_counter_ns()-started,quality_eligible=False,Gate_D=False,T_accessed=False,
    paid_attempts=3,training_work_not_duplicated_in_cost_sums=False,
    limitation='Capped105-update numerical/recovery workload, not native-patience completion or predictive comparison')
node._write(root/'result.json',result)
print(json.dumps({k:result[k] for k in ('status','exact_training_state','exact_predictions','resume_step')}),flush=True)
