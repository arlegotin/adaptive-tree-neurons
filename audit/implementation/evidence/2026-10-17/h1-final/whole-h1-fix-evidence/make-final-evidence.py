"""Fresh bounded engineering receipts/parents. No scientific teachers or E/T."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron.backend import probe_backend,select_backend,implementation_hash,lock_hash,runtime_hash,_resources,_guard
from treeneuron.cpu_envelope import probe_cpu_envelope,LOGICAL_BATCHES
from treeneuron.train import source_hash
from treeneuron import train
from treeneuron.config import resolve_experiment
from treeneuron.data.types import ObservationSet,DevelopmentData
from treeneuron.artifacts import create_run,finish_run
from treeneuron.checkpoint import load_checkpoint
root=Path('.superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/whole-h1-fix-evidence/native-final')
root.mkdir(exist_ok=False)
def write(name,value):
 with (root/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
source=source_hash();before=_resources(torch);_guard(before)
write('launch.json',dict(package_sha256=source,kernel_producer_sha256=implementation_hash(),lock_sha256=lock_hash(),runtime_sha256=runtime_hash(),resources=before,scope='fresh bounded CPU engineering receipts and one-update parents, no scientific pilot/cohort/E/T'))
probe=probe_backend('cpu',seed=1602030,batch_sizes=(128,))
write('kernel.json',probe)
if probe['status']!='passed':raise RuntimeError(probe.get('reason'))
envelope=probe_cpu_envelope();write('envelope.json',envelope)
if envelope['status']!='passed':raise RuntimeError(envelope.get('reason'))
probes={'cpu':probe}
report=dict(backend_probes=dict(probes=probes,comparison_blocks=[],decision=select_backend(probes)),cpu_envelope=envelope)
write('backend.json',report)
receipt=(root/'backend.json').resolve();train.BACKEND_REPORT_PATH=receipt
rng=np.random.default_rng(401)
def rows(n,start):
 x=rng.normal(size=(n,16));ids=np.arange(start,start+n,dtype=np.int64)
 return ObservationSet(x,x[:,0]-x[:,1],ids,ids)
data=DevelopmentData(rows(9224,0),rows(2306,10000),rows(577,20000),'bounded-engineering-fixture')
parents=[]
for batch in LOGICAL_BATCHES:
 _guard(_resources(torch));tick=time.perf_counter_ns()
 config=resolve_experiment(dict(study_id='h1-final-fix-native-engineering',model=dict(d=16,h=64),growth=dict(m_requested=16),
  data=dict(n_fit=9224,n_growth=2306,n_validation=577),training=dict(batch_size=batch,base_updates=1,continuation_updates=1,validation_interval=1,patience=2),
  backend=dict(cpu_threads=1,interop_threads=1),execution=dict(backend_report=str(receipt))))
 paths=create_run(root/f'batch-{batch}',config.to_dict());state=train.fit_parent(data,config,paths)
 finish_run(paths,{'selected_g_mse':state.stage['best_mse']},'completed')
 pointer=paths.directory/'checkpoints/best.json'
 loaded=load_checkpoint(pointer,expected_hashes=state.identity_hashes,device='cpu')
 np.testing.assert_array_equal(state.predict(data.G.X),loaded.predict(data.G.X))
 parents.append(dict(logical_batch=batch,actual_max_rows=min(batch,9224),run=str(paths.directory),checkpoint=str(pointer),
  checkpoint_sha256=hashlib.sha256(pointer.read_bytes()).hexdigest(),identities=state.identity_hashes,
  requested_updates=1,selected_update=state.update,elapsed_ns=time.perf_counter_ns()-tick,resources=_resources(torch)))
 if source_hash()!=source:raise RuntimeError('Source changed during native evidence')
write('parents.json',dict(status='passed',source_sha256=source,parents=parents,resources_after=_resources(torch)))
print(json.dumps(dict(status='passed',source_sha256=source,receipt=str(receipt),parents=len(parents))))
