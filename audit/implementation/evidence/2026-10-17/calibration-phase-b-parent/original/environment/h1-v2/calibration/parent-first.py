"""Single authorized Phase-B parent, no automatic forks or next coordinate."""
import os,sys,json,time,resource,subprocess
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from threadpoolctl import threadpool_info
from treeneuron import train
from treeneuron.config import resolve_experiment,canonical_hash
from treeneuron.data.synthetic import make_from_config
from treeneuron.artifacts import create_run,finish_run
from treeneuron.experiments.paired import _stage_record,write_json,file_hash
from treeneuron.protocol import execution_snapshot

config_path=Path(sys.argv[1]);config=resolve_experiment(json.loads(config_path.read_text()))
expected=dict(source='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c',
    execution='07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c',
    receipt='4cfb4983fbe8d152decf7e621dc79b2e77b7f4e3825cb379534ce6a4e05e9e42',
    commit='067f3913912b35ca5a4703f09b493659c007b5ea')
train.BACKEND_REPORT_PATH=Path(config.execution.backend_report).resolve()
assert train.source_hash()==expected['source']
assert file_hash(train.BACKEND_REPORT_PATH)==expected['receipt']
snapshot=execution_snapshot(Path.cwd(),expected['commit'])
assert snapshot['execution_sha256']==expected['execution'] and len(snapshot['files'])==315
root=Path(config.execution.artifacts_root)/config.study_id;root.mkdir(exist_ok=False,parents=True)
record=dict(status='running',scope='one fresh parent only; no allocation/forks/genuinepilot/T',
    config=config.to_dict(),config_sha256=canonical_hash(config.to_dict()),
    config_file_sha256=file_hash(config_path),wrapper_sha256=file_hash(Path(__file__)),
    source_sha256=train.source_hash(),execution_sha256=snapshot['execution_sha256'],
    pinned_commit=expected['commit'],checkout_commit=snapshot['checkout_commit'],
    receipt_sha256=file_hash(train.BACKEND_REPORT_PATH),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    runtime=dict(cpu=torch.get_num_threads(),interop=torch.get_num_interop_threads(),threadpools=threadpool_info(),
        env={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK')}),failures=[])
started=time.perf_counter_ns()
try:
    tick=time.perf_counter_ns();teacher,data=make_from_config(config)
    write_json(root/'teacher.json',teacher.to_dict());write_json(root/'splits.json',data.manifest.to_dict())
    np.savez_compressed(root/'observations.npz',**{f'{s}_{k}':getattr(getattr(data,s),k) for s in 'FGV' for k in ('X','y','row_id','group_id')})
    record.update(setup_ns=time.perf_counter_ns()-tick,teacher_id=teacher.teacher_id,split_hash=data.manifest.assignment_hash)
    assert teacher.teacher_id=='2bfc126302b1f4913acfce64b20f97362128b1d3926bc519229133ea3d9cc09f'
    assert data.manifest.assignment_hash=='3026548e1f90cbf15a1ddaa9eb95e2509bd78832818e01284733603886fd019f'
    paths=create_run(root/'runs',config.to_dict());tick=time.perf_counter_ns()
    state=train.fit_parent(data,config,paths);finish_run(paths,{},'completed')
    fit_ns=time.perf_counter_ns()-tick;stage=_stage_record(state,paths,root)
    bad=0
    for evaluation in reversed(stage['evaluations']):
        if evaluation['improved']:break
        bad+=1
    record.update(parent_enclosing_ns=fit_ns,stage=stage,identity_hashes=state.identity_hashes,
        parent_path=str(paths.directory),normalized_G=state.stage['best_mse']/state.fit_target_variance,
        actual_updates=stage['executed_updates'],actual_examples=sum(e['rows'] for e in state.history if e['type']=='update'),
        selected_epoch=state.epoch,selected_cursor=state.cursor,final_consecutive_nonimprovements=bad)
    record['status']='completed'
except BaseException as error:
    record['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
    record['failures'].append(f'{type(error).__name__}: {error}');raise
finally:
    record.update(enclosing_ns=time.perf_counter_ns()-started,peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        final_pre_receipt_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
    if 'parent_path' in record:
        check=Path(record['parent_path'])/'checkpoints'
        record['checkpoint_metadata_files']=[dict(path=str(p),bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(check.glob('checkpoint-*.meta.json'))]
        record['checkpoint_metadata_bytes']=sum(x['bytes'] for x in record['checkpoint_metadata_files'])
        record['events_bytes']=(Path(record['parent_path'])/'events.jsonl').stat().st_size
    write_json(root/'measurement.json',record)
print(json.dumps({k:record[k] for k in ('status','parent_enclosing_ns','actual_updates','actual_examples','normalized_G','peak_memory_bytes','events_bytes','checkpoint_metadata_bytes')}),flush=True)
