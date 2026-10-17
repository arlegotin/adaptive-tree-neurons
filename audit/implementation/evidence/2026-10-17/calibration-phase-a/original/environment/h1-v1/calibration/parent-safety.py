"""Bounded root-authorized existing-policy parent-only comparison."""
import os,sys,json,time,resource,hashlib,subprocess
from pathlib import Path
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.config import resolve_experiment,canonical_hash
from treeneuron.data.synthetic import make_from_config
from treeneuron.artifacts import create_run,finish_run
from treeneuron.experiments.paired import _stage_record,write_json,file_hash
from treeneuron.protocol import execution_snapshot

config_path=Path(sys.argv[1]);config=resolve_experiment(json.loads(config_path.read_text()))
train.BACKEND_REPORT_PATH=Path(config.execution.backend_report).resolve()
root=Path(config.execution.artifacts_root)/config.study_id;root.mkdir(exist_ok=False,parents=True)
record=dict(status='running',scope='fresh parent-only development candidate; no allocation/forks/pilot/T',
    config=config.to_dict(),config_sha256=canonical_hash(config.to_dict()),
    config_file_sha256=file_hash(config_path),wrapper_sha256=file_hash(Path(__file__)),
    source_sha256=train.source_hash(),execution_sha256=execution_snapshot(Path.cwd(),
    '680429908735a8077fbfab972468b9d420b7a9f1')['execution_sha256'],
    receipt_sha256=file_hash(train.BACKEND_REPORT_PATH),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    runtime=dict(cpu=torch.get_num_threads(),interop=torch.get_num_interop_threads(),
        env={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK')}),failures=[])
started=time.perf_counter_ns()
try:
    tick=time.perf_counter_ns();teacher,data=make_from_config(config)
    write_json(root/'teacher.json',teacher.to_dict());write_json(root/'splits.json',data.manifest.to_dict())
    record.update(setup_ns=time.perf_counter_ns()-tick,teacher_id=teacher.teacher_id,
        split_hash=data.manifest.assignment_hash)
    paths=create_run(root/'runs',config.to_dict());tick=time.perf_counter_ns()
    state=train.fit_parent(data,config,paths);finish_run(paths,{},'completed')
    record.update(parent_enclosing_ns=time.perf_counter_ns()-tick,stage=_stage_record(state,paths,root),
        identity_hashes=state.identity_hashes,parent_path=str(paths.directory),normalized_G=state.stage['best_mse']/state.fit_target_variance,
        actual_updates=state.stage['updates_completed'],actual_examples=sum(e['rows'] for e in state.history if e['type']=='update'))
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
        record['two_metadata_probe_floor_bytes']=sum(x['bytes'] for x in record['checkpoint_metadata_files'])
    write_json(root/'measurement.json',record)
print(json.dumps({k:record[k] for k in ('status','parent_enclosing_ns','actual_updates','actual_examples','normalized_G','peak_memory_bytes','final_pre_receipt_bytes','two_metadata_probe_floor_bytes')}),flush=True)
