"""Managed full-size numerical recovery proof, separate from native quality."""
import copy
import base64
import csv
import hashlib
import os
from pathlib import Path
import time

import numpy as np

from treeneuron.config import canonical_hash
from .ledger import write_new_json
from .method_worker import _ROOT,_THREADS,_sha,_digest,scientific_source_sha256
from .native_node_workload_v2 import _Array,_metadata,_stream_arrays
from .native_workload_v2 import _bound_reader,_bounded_tensor_archive,_process_observation
from .process_supervision_v2 import supervise_process
from .workqueue import _inventory


_OPERATOR='docs/research/v2/scripts/check_node_training_recovery.py'


def _identity():
    from .node_v2 import source_identity
    return dict(source_sha256=scientific_source_sha256(),native_source=source_identity(),operator_sha256=_sha(_ROOT/_OPERATOR),
        core_manifest_sha256=_sha(_ROOT/'locks/research-v2/core.json'))


def qualify_fullsize_node_recovery(*,output,interpreter,rss_limit_bytes=4*1024**3):
    root=Path(output);interpreter=Path(interpreter)
    if (root.is_symlink() or root.exists() or not interpreter.is_absolute() or not interpreter.is_file()
            or type(rss_limit_bytes) is not int or rss_limit_bytes<=0
            or any(root.resolve().is_relative_to((_ROOT/name).resolve()) for name in ('src','tests','locks','configs','protocol'))):
        raise ValueError('Fresh full-size recovery output, qualified interpreter and RSS guard required')
    root=root.resolve();root.mkdir(parents=True);identity=_identity();write_new_json(root/'request.json',identity)
    request_sha=_sha(root/'request.json');started=time.perf_counter_ns()
    argv=[str(interpreter),'-P','-B',str(_ROOT/_OPERATOR),str(root/'workload')]
    env={k:'1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE')}
    env.update(PYTHONPATH=str(_ROOT/'src'),KMP_DUPLICATE_LIB_OK=None,PYTORCH_ENABLE_MPS_FALLBACK='0',
        MPLCONFIGDIR=str(root/'cache/mpl'),XDG_CACHE_HOME=str(root/'cache/xdg'))
    process=supervise_process(argv=argv,cwd=_ROOT,env_overrides=env,output=root/'process',timeout_ns=None,
        rss_limit_bytes=rss_limit_bytes,identity=dict(purpose='fullsize_NODE_recovery',request_sha256=request_sha))
    if identity!=_identity():raise ValueError('Recovery source/operator changed during execution')
    result_file=root/'workload/result.json'
    completed=process['execution_status']=='exited' and process['returncode']==0 and process['child_reaped'] is True and result_file.is_file()
    result=dict(schema='research/v2-fullsize-recovery-execution',status='completed_recovery_qualification' if completed else 'recovery_qualification_failed',
        request_sha256=request_sha,process_sha256=_sha(root/'process/process.json'),
        workload_result_sha256=_sha(result_file) if result_file.is_file() else None,elapsed_before_publication_ns=time.perf_counter_ns()-started,
        native_workload_qualified=False,quality_eligible=False,protected_access_authorized=False)
    write_new_json(root/'result.json',result)
    return dict(result,reference=dict(root=str(root),inventory_sha256=canonical_hash(_inventory(root))))


def _state_digest(state,declared):
    """Independent portable encoding of recorded state, excluding only clocks."""
    from .node_v2 import _digest as digest
    value=copy.deepcopy(state);arrays={}
    for row in value['trace']:
        if type(row.get('elapsed_ns')) is not int or row['elapsed_ns']<=0:raise ValueError('Invalid recovery trace clock')
        row.pop('elapsed_ns')
    def pack(v):
        if isinstance(v,_Array):
            key=f'a{len(arrays)}';arrays[key]=declared[v.name];return dict(array=key,tensor=v.tensor)
        if type(v) is dict:return {'mapping':[[pack(k),pack(v[k])] for k in sorted(v,key=lambda k:(type(k).__name__,k))]}
        if type(v) in (list,tuple):return {'tuple' if type(v) is tuple else 'list':[pack(x) for x in v]}
        return v
    payload=pack(value);return digest(dict(payload=payload,arrays=arrays))


def _geometry(state,declared,resolved):
    from .node_v2 import _canonical
    prefix='0.0.';shapes={prefix+'response':[2048,3,64],prefix+'feature_selection_logits':[16,2048,6],
        prefix+'feature_thresholds':[2048,6],prefix+'log_temperatures':[2048,6],
        prefix+'bin_codes_1hot':[6,64,2],prefix+'_is_initialized_tensor':[]}
    codes=((np.arange(64)[None,:]//(2**np.arange(6))[:,None])%2).astype(np.float32)
    code_sha=hashlib.sha256(np.stack([codes,1-codes],axis=-1).tobytes()).hexdigest()
    def check_model(model):
        if type(model) is not dict or set(model)!=set(shapes):raise ValueError('Recovery model geometry differs')
        for name,shape in shapes.items():
            ref=model[name];dtype='uint8' if name.endswith('_is_initialized_tensor') else 'float32'
            if (not isinstance(ref,_Array) or not ref.tensor or declared[ref.name]['dtype']!=dtype
                    or _canonical(declared[ref.name]['shape'])!=_canonical(shape)
                    or name.endswith('_is_initialized_tensor') and declared[ref.name]['sha256']!=hashlib.sha256(b'\1').hexdigest()
                    or name.endswith('bin_codes_1hot') and declared[ref.name]['sha256']!=code_sha):
                raise ValueError('Recovery typed native tensor geometry/initialization differs')
    for name in ('initial_model','raw_model','selected_model'):check_model(state[name])
    for name in ('rolling_window','selected_window'):
        for row in state[name]:check_model(row['model'])
    parameter_shapes={i:shapes[prefix+name] for i,name in enumerate(('response','feature_selection_logits','feature_thresholds','log_temperatures'),1)}
    nonnegative=set()
    for name,age in (('raw_optimizer',105),('selected_optimizer',100)):
        optimizer=state[name];expected=dict(resolved['optimizer'],params=list(range(6)))
        if (set(optimizer)!={'state','param_groups'} or _canonical(optimizer['param_groups'])!=_canonical([expected])
                or set(optimizer['state'])!=set(parameter_shapes)):raise ValueError('Recovery optimizer geometry/groups differ')
        weights=[0.,0.]
        for _ in range(age):weights=[1+beta*w for beta,w in zip(resolved['optimizer']['betas'],weights)]
        for i,slot in optimizer['state'].items():
            if (type(i) is not int or set(slot)!={'beta1_weight','beta2_weight','exp_avg','exp_avg_sq'}
                    or any(type(slot[k]) is not float or slot[k]!=w for k,w in zip(('beta1_weight','beta2_weight'),weights))):
                raise ValueError('Recovery optimizer applied ages differ')
            for key in ('exp_avg','exp_avg_sq'):
                ref=slot[key]
                if (not isinstance(ref,_Array) or not ref.tensor or declared[ref.name]['dtype']!='float32'
                        or _canonical(declared[ref.name]['shape'])!=_canonical(parameter_shapes[i])):
                    raise ValueError('Recovery moment tensor geometry differs')
                if key=='exp_avg_sq':nonnegative.add(ref.name)
    return nonnegative


def _runtime(runtime,argv,identity,read):
    core=read_core=_read_core()
    expected={r['name'].lower().replace('_','-'):r for r in core['distributions']}
    if (runtime.get('executable')!=argv[0] or argv[0]!=core['executable']
            or runtime.get('executable_sha256')!=_sha(Path(argv[0]))
            or runtime.get('python')!=core['python'] or runtime.get('platform')!=core['platform']
            or runtime.get('core_sha256')!=identity['core_manifest_sha256']
            or runtime.get('versions')!={name:row['version'] for name,row in expected.items()}):
        raise ValueError('Recovery interpreter/runtime core identity differs')
    for row in runtime['files']:
        dist=expected.get(row.get('distribution'));path=Path(row['path'])
        if dist is None or set(row)!={'distribution','path','sha256'}:raise ValueError('Recovery runtime import schema differs')
        base=Path(dist['path']).parent;record=Path(dist['path'])/'RECORD'
        if not path.is_relative_to(base) or path.is_symlink() or _sha(record)!=dist['record_sha256'] or _sha(path)!=row['sha256']:
            raise ValueError('Recovery runtime import bytes differ')
        relative=path.relative_to(base).as_posix();entry=next((r for r in csv.reader(record.read_text().splitlines()) if r[0]==relative),None)
        if not entry or entry[1]!='sha256='+base64.urlsafe_b64encode(bytes.fromhex(row['sha256'])).decode().rstrip('='):
            raise ValueError('Recovery runtime import RECORD differs')


def _read_core():
    from .deployment_worker_v2 import _json
    return _json((_ROOT/'locks/research-v2/core.json').read_bytes())


def _snapshot(root,read,name,identity):
    from .node_v2 import NodeV2Config
    manifest=read('workload/'+name+'/manifest.json')
    config=dict(schema='research/v2-node-config',family='P12',mode='native_anchor',execution='bounded_memory_v1',
        feature_schema=dict(columns=[f'x{i}' for i in range(16)],categorical=[]),parameters={})
    if (manifest.get('schema')!='research/v2-node-snapshot' or manifest.get('config')!=config
            or manifest.get('resolved')!=NodeV2Config.parse(config).resolved()
            or manifest.get('identity',{}).get('source')!=identity['native_source']):
        raise ValueError('Full-size recovery snapshot/source/configuration differs')
    path=root/'workload'/name/'arrays.npz';ref=manifest['array_file']
    if ref.get('path')!='arrays.npz' or path.is_symlink():raise ValueError('Recovery snapshot array reference differs')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=ref['sha256']:raise ValueError('Recovery snapshot array bytes differ')
    seen=set();state=_metadata(manifest['payload'],manifest['arrays'],seen)
    if seen!=set(manifest['arrays']):raise ValueError('Unreferenced recovery state array')
    nonnegative=_geometry(state,manifest['arrays'],manifest['resolved'])
    _stream_arrays(path,manifest['arrays'],nonnegative)
    if (type(state['n_fit']) is not int or state['n_fit']!=1025 or type(state['n_select']) is not int or state['n_select']!=65
            or type(state['max_updates']) is not int or state['max_updates']!=105
            or type(state['seed']) is not int or state['seed']!=17
            or state['summary']['executed_updates']!=105 or state['summary']['stop_reason']!='max_updates'
            or len(state['trace'])!=105 or any(r['applied'] is not True for r in state['trace'])):
        raise ValueError('Recovery snapshot engineering horizon/rows differ')
    return dict(digest=_state_digest(state,manifest['arrays']),identity=manifest['identity'],config=config)


def load_fullsize_node_recovery(reference):
    reference=copy.deepcopy(reference)
    if (type(reference) is not dict or set(reference)!={'root','inventory_sha256'}
            or not _digest(reference['inventory_sha256'])):raise ValueError('Exact recovery inventory commitment required')
    root=Path(reference['root'])
    if not root.is_absolute() or root.is_symlink() or root.resolve()!=root:raise ValueError('Ordinary recovery proof root required')
    inventory=_inventory(root)
    if canonical_hash(inventory)!=reference['inventory_sha256']:raise ValueError('Recovery inventory bytes differ')
    read,refs=_bound_reader(root,inventory);request=read('request.json');identity=_identity()
    if request!=identity:raise ValueError('Recovery proof source/operator identity differs')
    outer=read('result.json');process=read('process/process.json');result=read('workload/result.json');started=read('workload/started.json')
    if (outer.get('schema')!='research/v2-fullsize-recovery-execution' or outer.get('status')!='completed_recovery_qualification'
            or outer.get('request_sha256')!=refs['request.json'] or outer.get('process_sha256')!=refs['process/process.json']
            or outer.get('workload_result_sha256')!=refs['workload/result.json']
            or any(outer.get(k) is not False for k in ('native_workload_qualified','quality_eligible','protected_access_authorized'))
            or process.get('execution_status')!='exited' or process.get('returncode')!=0 or process.get('child_reaped') is not True
            or process.get('timeout_ns') is not None or process.get('cwd')!=str(_ROOT)
            or process.get('identity')!=dict(purpose='fullsize_NODE_recovery',request_sha256=refs['request.json'])):
        raise ValueError('Managed recovery process/result/authority differs')
    argv=process['argv'];env=process['environment_overrides']
    if (argv[1:]!=['-P','-B',str(_ROOT/_OPERATOR),str(root/'workload')]
            or started.get('argv')!=argv[3:] or started.get('cwd')!=str(_ROOT)
            or started.get('operator_sha256')!=identity['operator_sha256'] or started.get('source')!=identity['native_source']
            or any(env.get(k)!='1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE'))
            or env.get('PYTHONPATH')!=str(_ROOT/'src') or env.get('KMP_DUPLICATE_LIB_OK') is not None
            or env.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0'
            or env.get('MPLCONFIGDIR')!=str(root/'cache/mpl') or env.get('XDG_CACHE_HOME')!=str(root/'cache/xdg')):
        raise ValueError('Recovery managed invocation/source differs')
    expected=dict(schema='research/v2-node-full-shape-recovery-proof',status='passed',source=identity['native_source'],
        operator_sha256=identity['operator_sha256'],F_rows=1025,G_rows=65,width=16,trees=2048,depth=6,effective_batch=512,initialization_rows=1000,
        reference_updates=105,interrupted_applied_updates=102,resume_step=100,resumed_final_updates=105,
        exact_training_state=True,exact_predictions=True,exact_closed_reload=True,current_payload_count=2,
        quality_eligible=False,Gate_D=False,T_accessed=False,paid_attempts=3,training_work_not_duplicated_in_cost_sums=False,
        limitation='Capped105-update numerical/recovery workload, not native-patience completion or predictive comparison')
    if any(type(result.get(k)) is not type(v) or result.get(k)!=v for k,v in expected.items()):
        raise ValueError('Recovery numerical capability/configuration/counters differ')
    if (result['inputs_sha256']!=refs['workload/inputs.npz']
            or result['final_snapshot_sha256']!=refs['workload/resumed-snapshot/manifest.json']):
        raise ValueError('Recovery input/endpoint commitments differ')
    # These are only the predeclared engineering F/G arrays, never V/T.
    expected_payload={name:dict(dtype='float64',shape=shape) for name,shape in
        (('F',[1025,16]),('G',[65,16]),('F_y',[1025]),('G_y',[65]))}
    inputs=root/'workload/inputs.npz';_bounded_tensor_archive(inputs,expected_payload,4*1024**2)
    rng=np.random.RandomState(2026101907);F=rng.normal(size=(1025,16));G=rng.normal(size=(65,16))
    expected_inputs=dict(F=F,G=G,F_y=1.+.2*F[:,0]+.1*F[:,1],G_y=1.+.2*G[:,0]+.1*G[:,1])
    with np.load(inputs,allow_pickle=False) as data:
        if any(not np.array_equal(data[k],v) for k,v in expected_inputs.items()):raise ValueError('Recovery engineering input constructor differs')
    snapshots=[_snapshot(root,read,name,identity) for name in ('reference-snapshot','resumed-snapshot')]
    if (snapshots[0]['digest']!=snapshots[1]['digest'] or snapshots[0]['digest']!=result['state_digest_without_elapsed_times']
            or snapshots[0]['identity']!=snapshots[1]['identity']):raise ValueError('Recovery state digest or runtime differs')
    _runtime(snapshots[0]['identity']['runtime'],argv,identity,read)
    receipts={name:read('workload/'+name+'-fit/fit-receipt.json') for name in ('reference','interrupted','resumed')}
    cfg=snapshots[0]['config']
    for name,row in receipts.items():
        if (row.get('config')!=cfg or type(row.get('seed')) is not int or row['seed']!=17 or row.get('budget')!={'max_updates':105}
                or type(row.get('executed_updates')) is not int or row['executed_updates']!=(102 if name=='interrupted' else 105)
                or row.get('status')!=('failed' if name=='interrupted' else 'completed_ineligible')):
            raise ValueError('Recovery paid attempt configuration/counters differ')
    failed=receipts['interrupted'];resumed=receipts['resumed']
    if (failed.get('error_type')!='KeyboardInterrupt' or len(failed['trace'])!=103
            or sum(r['applied'] is True for r in failed['trace'])!=102 or failed['trace'][-1]['applied'] is not False
            or type(resumed.get('resume_step')) is not int or resumed['resume_step']!=100):
        raise ValueError('Recovery interrupted/restored applied updates differ')
    observation=_process_observation(root,read,refs)
    if _identity()!=identity or _inventory(root)!=inventory:raise ValueError('Recovery evidence changed during inspection')
    return dict(schema='research/v2-authenticated-fullsize-node-recovery',reference=reference,source_sha256=identity['source_sha256'],
        recovery_qualified=True,state_bytes_independently_compared=True,trees=2048,depth=6,effective_batch=512,
        reference_updates=105,interrupted_applied_updates=102,resume_step=100,resumed_new_updates=5,total_applied_updates=212,
        paid_attempts=3,additional_unapplied_gradient_attempts=1,process_observation=observation,
        actual_updates_independently_reexecuted_by_reader=False,new_model_prediction_by_reader=False,
        native_workload_qualified=False,quality_eligible=False,protected_access_authorized=False,
        scope='Managed full-size capped numerical recovery with independent streamed saved-state comparison, including raw/selected parameters/moments/RNG/sampler/windows/rows except elapsed clocks; not native stopping or predictive quality. Closed prediction equality is trusted pinned operator execution.')
