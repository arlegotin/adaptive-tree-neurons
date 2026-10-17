"""Prechange plain-state numerical reference; never a cross-source resume grant."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.config import resolve_experiment,canonical_hash
from treeneuron.checkpoint import load_checkpoint
from treeneuron.data.synthetic import make_from_config,sample_teacher
from treeneuron.data.types import array_hash
from treeneuron.experiments.paired import file_hash,write_json
from treeneuron.protocol import execution_snapshot

base=Path('environment/h1-v1/calibration');config_path=base/'parent-b128-lr0p005-v2.json'
candidate=Path('artifacts/h1-v1/parent-safety/parent-b128-lr0p005-v2')
measured=json.loads((candidate/'measurement.json').read_text())
config=resolve_experiment(json.loads(config_path.read_text()))
assert train.source_hash()=='5c66c55693825e7dec51f0e59c74bd995da98b54c36f34acc445039d9ef85495'
source=execution_snapshot(Path.cwd(),'680429908735a8077fbfab972468b9d420b7a9f1')
assert len(list(Path('artifacts/h1-v1/parent-safety').glob('*/measurement.json')))==6
out=base/'old-source-reference';out.mkdir(exist_ok=False)
teacher,data=make_from_config(config)
parent=Path(measured['parent_path']);checkpoint=parent/'checkpoints/best.json'
state=load_checkpoint(checkpoint,expected_hashes=measured['identity_hashes'],device='cpu')
snapshot=train.pack_state(state)
torch.save(snapshot,out/'selected-snapshot.pt')
write_json(out/'selected-metadata.json',snapshot['metadata'])
write_json(out/'teacher.json',teacher.to_dict());write_json(out/'splits.json',data.manifest.to_dict())
E_seed=261017990;E=sample_teacher(teacher,4096,E_seed,config.data.noise_sigma)
assert not set(E.row_id)&set(data.manifest.row_id)
arrays={}
for name,rows in [('F',data.F),('G',data.G),('V',data.V),('E',E)]:
    for field in ('X','y','row_id','group_id','noise_free_target'):
        if getattr(rows,field) is not None:arrays[f'{name}_{field}']=getattr(rows,field)
np.savez_compressed(out/'observations.npz',**arrays)
predictions={name:state.predict(rows.X) for name,rows in [('F',data.F),('G',data.G),('E',E)]}
np.savez_compressed(out/'predictions.npz',**predictions)
events=[json.loads(line) for line in (parent/'events.jsonl').read_text().splitlines()]
updates=[]
for event in events:
    if event['type']!='update':continue
    rows=np.asarray(event['row_ids'],dtype=np.int64)
    updates.append(dict(update=event['update'],stage_update=event['stage_update'],rows=event['rows'],
        array_hash=array_hash(rows),ordered_int64_bytes_sha256=hashlib.sha256(rows.tobytes(order='C')).hexdigest(),
        loss_standardized_mse=event['loss_standardized_mse'],learning_rate=event['learning_rate']))
write_json(out/'ordered-update-commitments.json',dict(count=len(updates),examples=sum(u['rows'] for u in updates),
    dtype='int64 native little-endian on recorded arm64 runtime',hash_scope='array_hash includes dtype/shape; ordered_int64_bytes_sha256 hashes only ordered native int64 bytes',updates=updates))
provenance=dict(status='completed_prechange_diagnostic_reference',cross_source_resume_authorized=False,
    utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256=train.source_hash(),
    execution_sha256=source['execution_sha256'],pinned_commit=source['pinned_commit'],
    config_path=str(config_path),config_file_sha256=file_hash(config_path),config_sha256=canonical_hash(config.to_dict()),
    checkpoint_path=str(checkpoint),checkpoint_pointer_sha256=file_hash(checkpoint),
    checkpoint_metadata_sha256=json.loads(checkpoint.read_text())['sha256'],
    raw_events_path=str(parent/'events.jsonl'),raw_events_sha256=file_hash(parent/'events.jsonl'),
    measurement_path=str(candidate/'measurement.json'),measurement_sha256=file_hash(candidate/'measurement.json'),
    backend_receipt_path=config.execution.backend_report,backend_receipt_sha256=file_hash(Path(config.execution.backend_report)),
    wrapper_sha256=file_hash(Path(__file__)),runtime=dict(cpu=torch.get_num_threads(),interop=torch.get_num_interop_threads()),
    identity_hashes=measured['identity_hashes'],teacher_id=teacher.teacher_id,split_sha256=data.manifest.assignment_hash,
    selected_update=state.update,executed_updates=len(updates),actual_examples=sum(u['rows'] for u in updates),
    selected_epoch=state.epoch,selected_cursor=state.cursor,selected_schedule_position=state.scheduler.position,
    fit_target_variance=state.fit_target_variance,preprocessing=state.preprocessing.to_state(),
    E=dict(seed=E_seed,rows=4096,role='independent exploratory numeric prediction reference only; never pilot/confirmation',disjoint_from_FGV=True),
    arrays={name:dict(dtype=str(a.dtype),shape=list(a.shape),sha256=array_hash(a)) for name,a in arrays.items()},
    predictions={name:dict(dtype=str(a.dtype),shape=list(a.shape),sha256=array_hash(a)) for name,a in predictions.items()},
    files=[dict(path=str(p),sha256=file_hash(p),bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file()])
write_json(out/'provenance.json',provenance)
print(json.dumps(dict(path=str(out),provenance_sha256=file_hash(out/'provenance.json'),selected=state.update,
    executed=len(updates),examples=provenance['actual_examples'],source=train.source_hash())),flush=True)
