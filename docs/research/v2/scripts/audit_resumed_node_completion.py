"""Audit an inactive NODE anchor, retaining an explicit ENOSPC predecessor.

Derivative of the original c33 single-attempt operator. No old producer
terminal is fabricated, and an unavailable prior cost tail remains unavailable.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
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
parser.add_argument('--attempt-index',type=int,choices=(0,1),required=True)
for name in ('interruption-receipt','checkpoint-validation-receipt','resume-invocation'):
    parser.add_argument('--'+name,type=Path)
    parser.add_argument('--expected-'+name+'-sha256')
parser.add_argument('--evidence-root',type=Path)
parser.add_argument('--predecessor-only',action='store_true',help='Check only immutable closed predecessor evidence; no endpoint/model/query access')
args=parser.parse_args();started=time.perf_counter_ns()
source=args.source.resolve();root=args.study.resolve()

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

external={}
for name in ('interruption_receipt','checkpoint_validation_receipt','resume_invocation'):
    path=getattr(args,name);digest=getattr(args,'expected_'+name+'_sha256')
    if args.attempt_index==0:
        if path is not None or digest is not None:raise ValueError('Baseline audit cannot claim a predecessor')
    else:
        if (path is None or digest is None or path.is_symlink() or not path.is_file() or sha(path)!=digest):
            raise ValueError('Externally hashed interruption/checkpoint/resume evidence required before native imports')
        external[name]=dict(path=str(path.resolve()),sha256=digest,value=json.loads(path.read_text()))
if args.attempt_index==1:
    if args.evidence_root is None:raise ValueError('Explicit original evidence root required')
    evidence=args.evidence_root.resolve()
    if evidence.is_symlink() or not root.is_relative_to(evidence):raise ValueError('Original study must remain inside its evidence root')
elif args.evidence_root is not None:raise ValueError('No baseline predecessor evidence root')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE'):
    os.environ[key]='1'
os.environ.pop('KMP_DUPLICATE_LIB_OK',None)
assert sys.dont_write_bytecode
sys.path.insert(0,str(source/'src'))
from treeneuron.research.method_worker import scientific_source_sha256,_initialize_native_threads
from treeneuron.research.trial_bridge_v2 import EngineeringTrialBridge
from treeneuron.research.ledger import write_new_json
trace_spec=importlib.util.spec_from_file_location('atn_native_node_trace_reference',Path(__file__).with_name('native_node_trace_reference.py'))
trace_module=importlib.util.module_from_spec(trace_spec)
trace_spec.loader.exec_module(trace_module)
check_native_trace=trace_module.check_native_trace

@contextmanager
def inactive_study(path):
    lease=path/'worker-lease.json'
    if lease.is_symlink():raise ValueError('Ordinary producer lease required')
    with lease.open('r') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as error:raise PermissionError('Cannot audit an active native producer') from error
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

def check_predecessor(identity):
    prior_observations=None
    if args.attempt_index==1:
        interruption=external['interruption_receipt']['value'];validated=external['checkpoint_validation_receipt']['value']
        resume=external['resume_invocation']['value'];old=root/'attempt-000'
        assert interruption['schema']=='research/v2-native-queue-storage-interruption'
        assert interruption['dataset']=='parkinsons' and interruption['status']=='paused_resource_exhausted_incomplete_terminal_records'
        assert interruption['held_source_sha256']==identity['source_sha256'] and interruption['checkpoint_step']==4000
        assert interruption['T_accessed'] is False and interruption['source_or_seed_changed'] is False
        assert not (old/'terminal.json').exists() and not (old/'process/process.json').exists()
        assert validated['schema']=='research/v2-held-node-storage-checkpoint-read' and validated['checkpoint_step']==4000
        assert validated['source_sha256']==identity['source_sha256']
        assert validated['original_complete_checkpoint_loader_passed'] is validated['checkpoint_inventory_unchanged'] is True
        assert validated['parameter_updates_executed']==validated['model_predictions_executed']==0 and validated['T_accessed'] is False
        assert resume['schema']=='research/v2-explicit-native-storage-resume'
        assert resume['source_sha256']==identity['source_sha256'] and resume['unchanged']['fit_seed']==17
        assert resume['argv'][-1]=='--resume' and resume['argv'][resume['argv'].index('--output')+1]==str(root)
        assert resume['argv'][resume['argv'].index('--expected-source-sha256')+1]==identity['source_sha256']
        assert resume['failed_attempt_retained'] is resume['previous_terminal_tail_unavailable'] is True
        assert resume['previous_missing_receipts_not_backfilled'] is True and resume['T_accessed'] is False
        assert sha(resume['launcher']['path'])==resume['launcher']['sha256']==identity['launcher_sha256']
        starting=Path(external['resume_invocation']['path']).parent/'starting-checkpoint-pointer.json'
        assert sha(starting)==resume['starting_checkpoint']['sha256']
        pointer=json.loads(starting.read_text());assert pointer['step']==4000
        saved_manifest=root/'checkpoints'/pointer['manifest']['path']
        assert sha(saved_manifest)==pointer['manifest']['sha256'] and saved_manifest.stat().st_size==pointer['manifest']['bytes']
        old_process_files={'started.json','spawned.json','stderr.log','resource-samples.jsonl'}
        refs={Path(row['path']).name:row for row in interruption['files'] if Path(row['path']).parent==Path(str(old/'process').removeprefix(str(evidence)+'/'))}
        assert set(refs)==old_process_files
        for name,ref in refs.items():
            path=evidence/ref['path'];assert path==old/'process'/name and path.stat().st_size==ref['bytes'] and sha(path)==ref['sha256']
        original_start=json.loads((old/'process/started.json').read_text())
        assert original_start['identity']['source_sha256']==identity['source_sha256']
        assert original_start['identity']['request_sha256']==sha(root/'input/request.json')
        old_samples=[json.loads(line) for line in (old/'process/resource-samples.jsonl').read_text().splitlines()]
        assert old_samples and all(type(r['elapsed_ns']) is int for r in old_samples)
        prior_observations=dict(attempt_index=0,status='resource_exhausted_missing_producer_terminal',
            last_resource_sample_elapsed_ns=max(r['elapsed_ns'] for r in old_samples),
            observed_peak_root_rss_bytes=max(r['root_rss_bytes'] for r in old_samples),
            exact_terminal_cost_tail_available=False,original_terminal_backfilled=False,
            historical_checkpoint_payload_retention='Original rolling-two-payload policy may trim old arrays; immutable manifests/history and the separately validated starting-state observation remain retained')
    return prior_observations

if args.predecessor_only:
    if args.attempt_index!=1:raise ValueError('Only explicit resumed predecessor metadata can be checked here')
    identity=json.loads((root/'study.json').read_text())
    assert scientific_source_sha256()==identity['source_sha256']
    prior=check_predecessor(identity)
    write_new_json(args.output,dict(schema='research/v2-NODE-closed-predecessor-check',status='checked_closed_predecessor_only',source_sha256=identity['source_sha256'],operator_sha256=sha(__file__),prior=prior,model_or_prediction_read=False,native_workload_qualified=False,quality_eligible=False,T_accessed=False))
    raise SystemExit(0)

with inactive_study(root):
    identity=json.loads((root/'study.json').read_text())
    assert scientific_source_sha256()==identity['source_sha256']
    assert identity['budget']=={'native_stopping':True} and identity['wall_deadline_ns'] is None
    assert identity['fit_seed']==17 and identity['V_labels_read'] is identity['T_accessed'] is False
    prior_observations=check_predecessor(identity)
    assert {p.name for p in root.glob('attempt-*')}=={f'attempt-{i:03}' for i in range(args.attempt_index+1)}
    attempt=root/f'attempt-{args.attempt_index:03}';terminal=json.loads((attempt/'terminal.json').read_text())
    process_path=attempt/'process/process.json';process=json.loads(process_path.read_text())
    assert terminal['status']=='completed_ineligible' and sha(process_path)==terminal['process_receipt_sha256']
    assert process['execution_status']=='exited' and process['returncode']==0 and process['child_reaped'] is True
    # The inactive producer lease and closed/reaped process receipt are checked.
    # Never probe an old PID/process group that could have been reused.
    assert (process['argv'][-1]=='--resume' if args.attempt_index else '--resume' not in process['argv'])
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
        audit_elapsed_ns=time.perf_counter_ns()-started,attempt_index=args.attempt_index,
        prior_interruption_observations=prior_observations,
        prior_external_references={k:dict(path=v['path'],sha256=v['sha256']) for k,v in external.items()},
        all_attempt_terminal_receipts_available=args.attempt_index==0,
        whole_training_procedure_cost_authenticated=False,
        containing_clock_scope='Successful named invocation only; predecessor available RSS-time lower bound and unavailable terminal tail are separate',
        model_training_independently_reexecuted=False,old_pid_probed=False)
    for item in external.values():assert sha(item['path'])==item['sha256']
    write_new_json(args.output,result)
    print(json.dumps(result,sort_keys=True))
