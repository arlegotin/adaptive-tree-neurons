"""Real h64/d16/F2306 exact selector resource/insertion proof, not a teacher study."""
import copy
from dataclasses import fields
from pathlib import Path
import sys
import time

import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.artifacts import create_run
from treeneuron.backend import probe_backend,select_backend
from treeneuron.config import canonical_hash,resolve_experiment
from treeneuron.data.types import DevelopmentData,ObservationSet
from treeneuron.research.diagnostics import _parent,score_parent
from treeneuron.research.joint_allocation import build_gate_blocks,joint_ridge_select,selection_to_allocation
from treeneuron.research.joint_selector import JointSelectionConfig
from treeneuron.research.local_updates import diagnose_local_updates
from treeneuron.research.roles import fitting_view
from treeneuron.research.ledger import write_new_json
from treeneuron.growth.insert import insert_growth

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False);started=time.perf_counter_ns()
source=train.source_hash()
write_new_json(root/'started.json',dict(argv=sys.argv,cwd=str(Path.cwd()),source_sha256=source,
    scope='Numerical workload fixture; no primary synthetic teacher, V labels, E or protected T',quality_eligible=False))
probe=probe_backend('cpu',seed=1602030,batch_sizes=(128,));assert probe['status']=='passed'
report={'backend_probes':dict(probes={'cpu':probe},comparison_blocks=[],decision=select_backend({'cpu':probe}))}
write_new_json(root/'backend.json',report);train.BACKEND_REPORT_PATH=root/'backend.json'
config=resolve_experiment(dict(study_id='joint-workload-engineering',model=dict(d=16,h=64),
    data=dict(n_fit=2306,n_growth=577,n_validation=577),growth=dict(m_requested=16),
    training=dict(batch_size=128,learning_rate=.005,weight_decay=.001,base_updates=30,continuation_updates=30,
        validation_interval=25,patience=20,scheduler='constant'),backend=dict(cpu_threads=1)))
rng=np.random.default_rng(2026101908)
def rows(n,start):
    X=rng.normal(size=(n,16));y=X[:,0]+.2*X[:,1]+.1*np.sin(X[:,2])
    ids=np.arange(start,start+n,dtype=np.int64)
    return ObservationSet(X,y,ids,ids)
data=DevelopmentData(rows(2306,0),rows(577,10000),rows(577,20000),'engineering-only-joint-workload')
with (root/'inputs.npz').open('xb') as stream:
    np.savez(stream,**{role+'_'+key:getattr(getattr(data,role),key) for role in ('F','G') for key in ('X','y','row_id','group_id')})
view=fitting_view(data);paths=create_run(root/'parent',config.to_dict())
parent=train.fit_parent(view,config,paths);assert parent.update>0
_,before=_parent(parent);scored=score_parent(parent,view.F)
np.savez(root/'score-table.npz',**{field.name:getattr(scored.table,field.name).numpy() for field in fields(scored.table)
    if isinstance(getattr(scored.table,field.name),torch.Tensor)})
write_new_json(root/'score-identity.json',dict(table_sha256=scored.table_sha256,parent_sha256=scored.parent_sha256,
    status=list(scored.table.status),epsilon=scored.table.epsilon,tau=scored.table.tau,adam_epsilon=scored.table.adam_epsilon,
    formula_version=scored.table.formula_version,elapsed_ns=scored.elapsed_ns))
print('full parent-conditioned selector',flush=True)
selection=joint_ridge_select(parent,view.F,build_gate_blocks(parent,view.F,scored),JointSelectionConfig(),scored=scored)
write_new_json(root/'selection.json',selection.to_dict())
print('full selected-set direct solve and temporary nonlinear updates',flush=True)
diagnostic=diagnose_local_updates(parent,view.F,view.G,scored,selection)
write_new_json(root/'local-diagnostic.json',diagnostic)
assert _parent(parent)[1]==before
allocation=selection_to_allocation(parent,view.F,scored,selection,seed=29)
old={name:p for name,p in parent.model.named_parameters()}
moments={name:copy.deepcopy(parent.optimizer.state[p]) for name,p in old.items()}
prediction=parent.predict(data.G.X)
insert_growth(parent,allocation)
np.testing.assert_array_equal(prediction,parent.predict(data.G.X))
def equal(a,b):
    if isinstance(a,torch.Tensor):assert a.dtype==b.dtype and torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b
for name,p in old.items():
    assert dict(parent.model.named_parameters())[name] is p
    equal(parent.optimizer.state[p],moments[name])
for name,p in parent.model.named_parameters():
    if name not in old:assert parent.optimizer.state[p]['step']==0
from treeneuron.checkpoint import save_checkpoint,load_checkpoint
save_checkpoint(parent,root/'grown.json')
restored=load_checkpoint(root/'grown.json',expected_hashes=parent.identity_hashes,device='cpu')
np.testing.assert_array_equal(prediction,restored.predict(data.G.X))
assert train.source_hash()==source
result=dict(schema='research/v2-joint-native-workload',status='passed',source_sha256=source,
    F_rows=2306,G_rows=577,d=16,h=64,requested_count=16,effective_count=allocation.effective_count,
    selected_parent_update=parent.update,parent_parameters=sum(p.numel() for p in old.values()),
    grown_parameters=sum(p.numel() for p in parent.model.parameters()),selection_sha256=selection.sha256,
    full_direct_reference_error=diagnostic['linear_solve']['matched_Schur_objective_error'],
    exact_function_preservation=True,old_moments_unchanged=True,newborn_age_zero=True,closed_checkpoint_reload_exact=True,
    process_body_ns=time.perf_counter_ns()-started,quality_eligible=False,new_primary_teachers=0,T_accessed=False,
    interpretation='Linear-plus-sine numerical fixture only; no predictive benefit, Gate B or M3 study outcome')
write_new_json(root/'result.json',result);print(result,flush=True)
