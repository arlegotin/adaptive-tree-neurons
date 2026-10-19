"""Read-only small-control export under the exact frozen M2 producer source.

New selector analysis uses separate F-only geometry and already-public V/E tables.
This does not load historical state under a changed package identity or refit.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.config import canonical_hash
from treeneuron.growth.allocate import allocate
from treeneuron.research.ledger import AttemptLedger,write_new_json
from treeneuron.research.opportunity_data import load_fitting_data
from treeneuron.research.opportunity_inputs import load_scores
from treeneuron.research.parent_artifacts import load_parent_input
from treeneuron.research.workqueue import _read

collection_path,queue_path,output=map(Path,sys.argv[1:]);started=time.perf_counter_ns()
assert train.source_hash()=='e322da64b518f29a14bde3385bfcf6d683993b717ed4c1fc13a7a5440a1f9a47'
assert hashlib.sha256(collection_path.read_bytes()).hexdigest()=='52bda3705bd2f78cde2de3d89b6c263f7a5cb07872cf3b6889d372a33c922a1c'
collection=json.loads(collection_path.read_text());plan=json.loads((queue_path/'plan.json').read_text())
assert plan['plan_hash']=='d0537c5c7caee7c6e4c6d09e3e2f68b2c819f8e38853ab1cad67850718d6381a'
assert plan['plan_hash']==canonical_hash({k:v for k,v in plan.items() if k!='plan_hash'})
lease=(queue_path/'worker-lease.json').open('r')
fcntl.flock(lease.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
ledger=AttemptLedger.resume(queue_path,plan)
assert ledger.events[-1]['event_hash']==collection['ledger_event_hash']
committed={r['job_id']:r for r in collection['result_commitments']}
jobs={j['job_id']:j for j in plan['attempts']};used={}
def read(name):
 if name not in used:
  record=committed[name];used[name]=_read(record['result'],queue_path,jobs[name])
 return used[name]['value']
output.mkdir(parents=True,exist_ok=False)
write_new_json(output/'request.json',dict(schema='research/v2-small-joint-extraction',source_sha256=train.source_hash(),queue_plan_hash=plan['plan_hash'],collection_sha256=hashlib.sha256(collection_path.read_bytes()).hexdigest(),ledger_event_hash=collection['ledger_event_hash'],quality_claim=False,new_fits=0,T_accessed=False))
max_autograd_error=0.;cases=[]
for teacher in range(4):
 data=read(f'data:positive_small/teacher-{teacher}')
 view=load_fitting_data(data['artifact_path'])
 for repeat in range(2):
  parent_key=f'positive_small/teacher-{teacher}/repeat-{repeat}/ungrown'
  native_parent=read('parent:'+parent_key);score=read('score:'+parent_key+'/late_G_best')
  parent=load_parent_input(native_parent['artifact_path'],'late_G_best',view)
  scored=load_scores(score['artifact_path'],parent)
  model=copy.deepcopy(parent.model).double().eval()
  X=torch.tensor(parent.preprocessing.transform_X(view.F.X),dtype=torch.float64)
  y=parent.preprocessing.transform_y(view.F.y);aug=torch.cat([X,torch.ones((len(X),1),dtype=torch.float64)],dim=1)
  parameters=dict(model.named_parameters());names=list(parameters)
  jac=torch.autograd.functional.jacobian(lambda *values:torch.func.functional_call(model,dict(zip(names,values)),(X,)),tuple(parameters.values()),vectorize=True)
  J=np.column_stack([a.detach().numpy().reshape(len(X),-1) for a in jac])
  with torch.no_grad():
   beta=model.head.weight[0];mask=model.layer.base(X)>0;prediction=model(X).numpy()
   blocks={unit:((beta[unit]*mask[:,unit]*(aug@scored.table.contrasts[unit]))[:,None]*aug/(2*model.layer.tau)).numpy() for unit in range(8)}
  # Independently differentiate each actual newborn gate on a separate model.
  for unit in range(8):
   temporary=copy.deepcopy(model);temporary.layer.grow(torch.tensor([unit]),scored.table.contrasts[[unit]])
   gate=torch.cat([temporary.layer.gate.weight[0],temporary.layer.gate.bias])
   def forward(g):
    state=dict(temporary.named_parameters());state['layer.gate.weight']=g[:-1][None,:];state['layer.gate.bias']=g[-1:]
    return torch.func.functional_call(temporary,state,(X,))
   observed=torch.autograd.functional.jacobian(forward,gate,vectorize=True).detach().numpy()
   error=float(np.max(np.abs(observed-blocks[unit])));max_autograd_error=max(max_autograd_error,error)
   np.testing.assert_allclose(observed,blocks[unit],rtol=1e-10,atol=1e-12)
  name=f'teacher-{teacher}-repeat-{repeat}';directory=output/name;directory.mkdir()
  eligible=allocate(scored.table,8,policy='adam',seed=0).ids.tolist()
  arrays=dict(parent_J=J,residual=np.asarray(y-prediction,dtype=np.float64),**{'gate_'+str(k):v for k,v in blocks.items()})
  with (directory/'F-geometry.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
  cases.append(dict(case_id=name,teacher_index=teacher,repetition=repeat,eligible_ids=eligible,parent_sha256=scored.parent_sha256,table_sha256=scored.table_sha256,geometry_sha256=hashlib.sha256((directory/'F-geometry.npz').read_bytes()).hexdigest(),current_top=allocate(scored.table,2,policy='adam',seed=0).ids.tolist(),matched_random=allocate(scored.table,2,policy='random',seed=parent.config.seeds.allocation).ids.tolist(),allocation_seed=parent.config.seeds.allocation,native_training=parent.config.to_dict()['training']))
  outcomes=[]
  for job in plan['attempts']:
   p=job['payload']
   if job['kind']!='fit' or p.get('matrix')!='M2d' or (p['teacher_index'],p['repetition'])!=(teacher,repeat):continue
   fitted=read(job['job_id']);row=dict(unit_ids=p['unit_ids'],status=fitted['status'],procedure_job_id=job['job_id'],standalone_cost=collection['standalone_costs'][job['job_id']])
   if fitted['status']=='completed_eligible':
    evaluated=read('evaluate:'+p['attempt_id']);assert evaluated['status']=='completed_eligible'
    row.update(selected_g_mse=fitted['report']['selected_g_mse'],metrics=evaluated['metrics']['selected'])
   outcomes.append(row)
  assert len(outcomes)==37
  write_new_json(directory/'outcomes.json',dict(source='Previously audited M2d finite fit table; no new fit or seed',records=outcomes))
write_new_json(output/'export.json',dict(schema='research/v2-small-joint-inputs',status='completed_eligible',source_sha256=train.source_hash(),cases=cases,max_gate_autograd_error=max_autograd_error,used_commitments=[committed[name] for name in sorted(used)],elapsed_ns=time.perf_counter_ns()-started,new_fits=0,T_accessed=False,scope='Four existing planted bases/R2; legacy M2 native stopping/WD0; not the prospective competent M3 population/treatment'))
print(json.dumps(dict(status='passed',cases=len(cases),new_fits=0,max_gate_autograd_error=max_autograd_error)),flush=True)
