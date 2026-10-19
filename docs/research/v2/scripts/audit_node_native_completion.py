"""Audit an inactive uncapped NODE anchor, without fitting or query labels."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--study',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();started=time.perf_counter_ns()
source=args.source.resolve();root=args.study.resolve()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE'):
    os.environ[key]='1'
os.environ.pop('KMP_DUPLICATE_LIB_OK',None)
assert sys.dont_write_bytecode
sys.path.insert(0,str(source/'src'))
from treeneuron.research.method_worker import scientific_source_sha256,_initialize_native_threads
from treeneuron.research.trial_bridge_v2 import EngineeringTrialBridge
from treeneuron.research.ledger import write_new_json
from native_node_trace_reference import check_native_trace

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

@contextmanager
def inactive_study(path):
    lease=path/'worker-lease.json'
    if lease.is_symlink():raise ValueError('Ordinary producer lease required')
    with lease.open('r') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as error:raise PermissionError('Cannot audit an active native producer') from error
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

with inactive_study(root):
    identity=json.loads((root/'study.json').read_text())
    assert scientific_source_sha256()==identity['source_sha256']
    assert identity['budget']=={'native_stopping':True} and identity['wall_deadline_ns'] is None
    assert identity['fit_seed']==17 and identity['V_labels_read'] is identity['T_accessed'] is False
    attempt=root/'attempt-000';terminal=json.loads((attempt/'terminal.json').read_text())
    process_path=attempt/'process/process.json';process=json.loads(process_path.read_text())
    assert terminal['status']=='completed_ineligible' and sha(process_path)==terminal['process_receipt_sha256']
    assert process['execution_status']=='exited' and process['returncode']==0 and process['child_reaped'] is True
    try:os.killpg(process['child_pid'],0)
    except ProcessLookupError:pass
    else:raise AssertionError('Owned worker process group remains active')
    for ref in process['files']:
        p=attempt/'process'/ref['path']
        assert not p.is_symlink() and sha(p)==ref['sha256'] and p.stat().st_size==ref['bytes']
    samples=[json.loads(line) for line in (attempt/'process/resource-samples.jsonl').read_text().splitlines()]
    peak=max(s['root_rss_bytes'] for s in samples if s.get('root_rss_bytes') is not None)
    assert peak==process['observed_peak_root_rss_bytes'] and len(samples)==process['rss_sample_count']
    assert peak<=identity['sampled_root_rss_limit_bytes']
    request_path=root/'input/request.json';request=json.loads(request_path.read_text())
    assert request['scientific_source_sha256']==identity['source_sha256']
    assert request['budget']=={'native_stopping':True} and request['config']==identity['config']
    assert request['fit_seed']==17 and request['observation_identity']['split_sha256']==identity['split_sha256']
    worker=EngineeringTrialBridge._worker_terminal(attempt/'worker',request,sha(request_path))
    assert worker['snapshot_reload_exact'] is True and worker['torch_threads']==dict(intra_op=1,inter_op=1)
    assert worker['loaded_heavy_modules']==['torch','treeneuron.baselines._node_vendor']
    _initialize_native_threads('P12')
    import numpy as np
    import pandas as pd
    from treeneuron.research.node_v2 import NodeV2Predictor
    arrays={}
    for role,ref in request['files'].items():
        p=request_path.parent/ref['path'];assert not p.is_symlink() and sha(p)==ref['sha256']
        with np.load(p,allow_pickle=False) as archive:arrays[role]={key:archive[key].copy() for key in archive.files}
    assert set(arrays)=={'F','G','V_features'} and 'y' not in arrays['V_features']
    fit=json.loads((attempt/'worker/fit/fit-receipt.json').read_text())
    model=NodeV2Predictor.load(attempt/'worker/snapshot',expected_manifest_sha256=worker['snapshot_manifest_sha256'])
    state=model._state;summary=model.resource_summary()
    assert state['max_updates'] is None and summary['stop_reason']=='G_patience'
    assert summary['config']==identity['config'] and state['trace']==fit['trace'] and state['summary']==fit['summary']
    steps=summary['executed_updates'];n=len(arrays['F']['y'])
    assert n==state['n_fit'] and len(arrays['G']['y'])==state['n_select']
    assert steps==len(state['trace'])==fit['executed_updates']
    reports=state['reports']
    trace_reference=check_native_trace(trace=state['trace'],reports=reports,n_fit=n,
        batch_size=512,report_frequency=100,patience=5000,latest_checkpoints=5,
        selected_step=summary['selected_step'],executed_updates=steps)
    best_step=trace_reference['selected_step']
    q=arrays['V_features']
    with np.load(attempt/'worker/predictions.npz',allow_pickle=False) as archive:
        assert set(archive.files)=={'prediction','row_id','group_id'}|({'timestamps'} if 'timestamps' in q else set())
        assert np.array_equal(archive['row_id'],q['row_id']) and np.array_equal(archive['group_id'],q['group_id'])
        if 'timestamps' in q:np.testing.assert_array_equal(archive['timestamps'],q['timestamps'])
        repeated=model.predict(pd.DataFrame(q['X'],columns=identity['config']['feature_schema']['columns']))
        np.testing.assert_array_equal(repeated,archive['prediction'])
    assert scientific_source_sha256()==identity['source_sha256']
    result=dict(schema='research/v2-NODE-uncapped-completion-audit',status='native_anchor_completed_and_replayed',
        source_sha256=identity['source_sha256'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        operator_sha256=sha(__file__),trace_reference_sha256=sha(Path(__file__).with_name('native_node_trace_reference.py')),
        trace_reference=trace_reference,dataset=request['observation_identity']['dataset'],
        study_sha256=sha(root/'study.json'),request_sha256=sha(request_path),
        process_receipt_sha256=sha(process_path),worker_terminal_sha256=sha(attempt/'worker/terminal.json'),
        selected_snapshot_manifest_sha256=worker['snapshot_manifest_sha256'],prediction_sha256=sha(attempt/'worker/predictions.npz'),
        executed_updates=steps,selected_step=best_step,native_G_evaluations=len(reports),
        training_rows=n,query_rows=len(q['row_id']),encoded_width=summary['encoded_width'],
        trainable_parameters=sum(p.numel() for p in model._model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model._model.parameters())+sum(b.numel() for b in model._model.buffers()),
        native_stopping_independently_checked=True,exact_selected_prediction_replay=True,
        observed_peak_root_rss_bytes=peak,process_elapsed_ns=process['elapsed_ns'],containing_elapsed_ns=terminal['elapsed_ns'],
        timers_nested=True,wall_deadline_ns=None,scientific_quality_comparison=False,
        full_six_dataset_P12_qualification=False,V_labels_read=False,T_accessed=False,Gate_D=False,
        audit_elapsed_ns=time.perf_counter_ns()-started)
    write_new_json(args.output,result)
    print(json.dumps(result,sort_keys=True))
