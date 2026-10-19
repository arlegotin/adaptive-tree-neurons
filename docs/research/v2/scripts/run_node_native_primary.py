"""One pinned primary-split/seed17 native fit, without a wall cap, with exact recovery.

Run from the held source checkout. No HPO, V labels, protected T or quality claim.
Each invocation retains a new owned-process attempt and all preceding costs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from treeneuron.research import method_worker
from treeneuron.research.ledger import write_new_json
from treeneuron.research.opportunity_execution import live_resources
from treeneuron.research.process_supervision_v2 import supervise_process
from treeneuron.research.transfer import TransferStore
from treeneuron.research.workqueue import StudyLease

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',choices=('airfoil','concrete','ccpp','parkinsons','bike'),required=True)
parser.add_argument('--prepared',type=Path,required=True)
parser.add_argument('--expected-manifest-sha256',required=True)
parser.add_argument('--expected-source-sha256',required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--resume',action='store_true')
args=parser.parse_args();started=time.perf_counter_ns()
project=Path(method_worker.__file__).resolve().parents[3]
assert Path.cwd().resolve()==project
assert method_worker.scientific_source_sha256()==args.expected_source_sha256
assert sha(args.prepared/'manifest.json')==args.expected_manifest_sha256
root=args.output.resolve()
if not args.resume:root.mkdir(parents=True,exist_ok=False)
elif not root.is_dir() or root.is_symlink():raise ValueError('Existing recovery study required')

with StudyLease(root):
    store=TransferStore(args.prepared)
    assert store.manifest['dataset']==args.dataset
    fold=4 if args.dataset=='bike' else 0
    view=store.development(fold)
    config=dict(schema='research/v2-node-config',family='P12',mode='native_anchor',execution='bounded_memory_v1',
        feature_schema=dict(columns=store.manifest['feature_names'],categorical=store.manifest['categorical_features']),parameters={})
    identity=dict(schema='research/v2-NODE-native-recovery-study',source_sha256=args.expected_source_sha256,
        prepared_manifest_sha256=args.expected_manifest_sha256,split_sha256=view.split_hash,
        launcher_sha256=sha(__file__),interpreter=str(Path(sys.executable)),interpreter_sha256=sha(sys.executable),
        config=config,budget={'native_stopping':True},fit_seed=17,wall_deadline_ns=None,
        sampled_root_rss_limit_bytes=4*1024**3,memory_admission_proxy_bytes=11*1024**3//2,
        disk_admission_bytes=1536*1024**2,keep_committed_payloads=2,
        V_labels_read=False,T_accessed=False,quality_eligible=False,Gate_D=False)
    if args.resume:
        assert json.loads((root/'study.json').read_text())==identity
        request_path=root/'input/request.json'
        if (root/'checkpoints').exists() and not (root/'checkpoints/latest.json').is_file():
            raise ValueError('Interrupted before first checkpoint; a new explicit same-seed study must retain this failed attempt')
        if any(json.loads(p.read_text()).get('status')=='completed_ineligible' for p in root.glob('attempt-*/terminal.json')):
            raise ValueError('Native study already completed; no repeat fit admitted')
    else:
        write_new_json(root/'study.json',identity)
        inputs=root/'input';inputs.mkdir();files={}
        for role in ('F','G','V_features'):
            relative=f'outer-{fold}/'+('V-features.npz' if role=='V_features' else role+'.npz')
            original=store.root/relative;expected=store.manifest['files'][relative]['sha256']
            assert not original.is_symlink() and sha(original)==expected
            target=inputs/(role+'.npz')
            try:os.link(original,target)
            except OSError as error:
                import errno
                if error.errno!=errno.EXDEV:raise
                shutil.copyfile(original,target)
            assert sha(target)==expected;files[role]=dict(path=target.name,sha256=expected)
        request=dict(schema='research/v2-method-worker-request',purpose='engineering_qualification',
            job_id=f'native-recovery/P12/{args.dataset}/primary',family_id='P12',fit_seed=17,
            scientific_source_sha256=args.expected_source_sha256,config=config,budget={'native_stopping':True},
            observation_identity=dict(dataset=args.dataset,split_kind='temporal' if args.dataset=='bike' else 'grouped',split_sha256=view.split_hash),files=files)
        request_path=inputs/'request.json';write_new_json(request_path,request)
    # The established worker validator authenticates every archive before a launch.
    request_digest=sha(request_path)
    method_worker._validate_request(request_path,request_digest,recoverable_training=True)
    attempts=sorted(root.glob('attempt-*'));index=len(attempts)
    attempt=root/f'attempt-{index:03d}';attempt.mkdir()
    live=live_resources(root)
    admission=dict(live,admitted=live['available_memory_bytes'] is not None
        and live['available_memory_bytes']>=identity['memory_admission_proxy_bytes']
        and live['free_disk_bytes']>=identity['disk_admission_bytes'])
    write_new_json(attempt/'admission.json',admission)
    write_new_json(attempt/'invocation.json',dict(argv=sys.argv,cwd=str(project),started_utc_ns=time.time_ns(),
        source_sha256=args.expected_source_sha256,study_sha256=sha(root/'study.json'),request_sha256=request_digest,
        resume=args.resume,all_previous_attempts_retained=[p.name for p in attempts],
        cost_scope='This invocation plus separately retained prior attempts/proofs; never erase the earlier capped7200s run'))
    if not admission['admitted']:
        write_new_json(attempt/'terminal.json',dict(status='paused_before_launch',admission=admission,
            elapsed_ns=time.perf_counter_ns()-started,T_accessed=False,quality_eligible=False))
        print('Paused for RAM/disk admission; no model changed.',flush=True);raise SystemExit(3)
    argv=[sys.executable,'-B','-m','treeneuron.research.method_worker','--request',str(request_path),
        '--expected-request-sha256',request_digest,'--output',str(attempt/'worker'),
        '--training-checkpoints',str(root/'checkpoints')]
    if args.resume and (root/'checkpoints/latest.json').is_file():argv.append('--resume')
    environment={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}
    environment.update(PYTHONPATH=str(project/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0',KMP_DUPLICATE_LIB_OK=None)
    result=supervise_process(argv=argv,cwd=project,env_overrides=environment,output=attempt/'process',
        timeout_ns=None,rss_limit_bytes=identity['sampled_root_rss_limit_bytes'],poll_interval_ns=1000000000,
        identity=dict(study_sha256=sha(root/'study.json'),source_sha256=args.expected_source_sha256,
            request_sha256=request_digest,recovery_attempt=index))
    success=result['execution_status']=='exited' and result['returncode']==0
    pointer=json.loads((root/'checkpoints/latest.json').read_text()) if (root/'checkpoints/latest.json').is_file() else None
    terminal=dict(status='completed_ineligible' if success else result['execution_status'],
        process_receipt_sha256=sha(attempt/'process/process.json'),resume_checkpoint_at_end=pointer,
        elapsed_ns=time.perf_counter_ns()-started,process_ns=result['elapsed_ns'],timers_nested=True,
        source_sha256=args.expected_source_sha256,V_labels_read=False,T_accessed=False,quality_eligible=False,Gate_D=False)
    write_new_json(attempt/'terminal.json',terminal)
    print(json.dumps(terminal,sort_keys=True),flush=True)
    raise SystemExit(0 if success else 4)
