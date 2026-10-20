"""Read NODE's recorded endpoint without constructing a model or loading all weights.

The managed optimizer and its recovery execution remain trusted. Numeric array
commitments are streamed; only small transform/sampler arrays are materialized.
"""
from dataclasses import dataclass
import copy
import hashlib
import math
import zipfile

import numpy as np

from .native_workload_v2 import _bounded_tensor_archive,_finite_number


@dataclass(frozen=True)
class _Array:
    name:str
    tensor:bool


def _metadata(value,declared,seen):
    if type(value) is not dict:
        if value is None or type(value) in (str,int,bool) or type(value) is float and math.isfinite(value):return value
        raise ValueError('Invalid NODE portable metadata scalar')
    fields=set(value)
    if fields=={'array','tensor'}:
        key=value['array']
        if type(key) is not str or key not in declared or key in seen or type(value['tensor']) is not bool:
            raise ValueError('Invalid NODE portable array reference')
        seen.add(key);return _Array(key,value['tensor'])
    if fields=={'mapping'}:
        result={}
        for pair in value['mapping']:
            if type(pair) is not list or len(pair)!=2:raise ValueError('Invalid NODE portable mapping')
            key=_metadata(pair[0],declared,seen)
            if type(key) not in (str,int) or key in result:raise ValueError('Invalid NODE portable mapping key')
            result[key]=_metadata(pair[1],declared,seen)
        return result
    if fields in ({'list'},{'tuple'}):
        tag=next(iter(fields));result=[_metadata(x,declared,seen) for x in value[tag]]
        return tuple(result) if tag=='tuple' else result
    raise ValueError('Unknown NODE portable metadata node')


def _stream_arrays(path,declared,nonnegative):
    _bounded_tensor_archive(path,declared,1024**3,allowed_dtypes=('float32','float64','int64','uint32','uint8','bool'))
    with zipfile.ZipFile(path) as archive:
        for name,expected in declared.items():
            with archive.open(name+'.npy') as stream:
                version=np.lib.format.read_magic(stream)
                header=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                shape,fortran,dtype=header(stream)
                if fortran:raise ValueError('NODE portable array must retain its native C layout')
                remaining=math.prod(shape)*dtype.itemsize;digest=hashlib.sha256()
                while remaining:
                    raw=stream.read(min(1024**2,remaining));remaining-=len(raw);digest.update(raw)
                    if not raw or len(raw)%dtype.itemsize:raise ValueError('Truncated NODE numeric payload')
                    value=np.frombuffer(raw,dtype=dtype)
                    if not np.isfinite(value).all() or name in nonnegative and np.any(value<0):
                        raise ValueError('Nonfinite NODE tensor or negative optimizer second moment')
                if digest.hexdigest()!=expected['sha256']:raise ValueError('NODE numeric array commitment differs')


def _prediction_binding(state,config,declared):
    from .node_v2 import _digest
    arrays={}
    def pack(value):
        if isinstance(value,_Array):
            key=f'a{len(arrays)}';arrays[key]=declared[value.name]
            return dict(array=key,tensor=value.tensor)
        if type(value) is dict:return {'mapping':[[pack(k),pack(value[k])] for k in sorted(value,key=lambda k:(type(k).__name__,k))]}
        if type(value) in (list,tuple):return {'tuple' if type(value) is tuple else 'list':[pack(v) for v in value]}
        return value
    value=dict(config=config.to_state(),transform=state['transform'],F_identity=state['F_identity'],G_identity=state['G_identity'],
        seed=state['seed'],initialization=state['initialization'],selected_step=state['summary']['selected_step'],
        prediction_model=state['selected_model'] if state['selected_model'] is not None else state['raw_model'])
    payload=pack(value);return _digest(dict(payload=payload,arrays=arrays))


def _trace(state,resolved,n,gradient_names):
    rows=state['trace'];reports=state['reports'];cursor=0;epoch=0;seen=set();ordered=[];visited=set()
    if type(rows) is not list or not rows:raise ValueError('Missing NODE update trace')
    for step,row in enumerate(rows,1):
        if cursor==n:cursor=0;epoch+=1;seen=set();ordered=[]
        batch=row.get('rows');norms=row.get('gradient_norms')
        if (set(row)!={'step','epoch','rows','loss','gradient_norms','optimizer_entered','applied','elapsed_ns'}
                or type(row['step']) is not int or row['step']!=step or type(row['epoch']) is not int or row['epoch']!=epoch
                or type(batch) is not list or len(batch)!=min(resolved['batch_size'],n-cursor)
                or any(type(i) is not int or not 0<=i<n for i in batch) or len(set(batch))!=len(batch) or seen.intersection(batch)
                or not _finite_number(row['loss']) or row['optimizer_entered'] is not True or row['applied'] is not True
                or type(row['elapsed_ns']) is not int or row['elapsed_ns']<=0
                or type(norms) is not dict or set(norms)!=gradient_names or not all(_finite_number(v) for v in norms.values())):
            raise ValueError('NODE paid row/update/gradient trace differs')
        cursor+=len(batch);seen.update(batch);ordered.extend(batch);visited.update(batch)
    frequency=resolved['report_frequency'];expected_steps=list(range(frequency,len(rows)+1,frequency))
    if type(reports) is not list or len(reports)!=len(expected_steps):raise ValueError('NODE native G report grid differs')
    by_step={}
    for step,row in zip(expected_steps,reports,strict=True):
        window=list(range(max(frequency,step-4*frequency),step+1,frequency))
        if (set(row)!={'step','g_mse','snapshot_steps'} or type(row['step']) is not int or row['step']!=step
                or not _finite_number(row['g_mse']) or type(row['snapshot_steps']) is not list
                or any(type(s) is not int for s in row['snapshot_steps']) or row['snapshot_steps']!=window):
            raise ValueError('NODE native G/window record differs')
        by_step[step]=row
    winner=None;best=math.inf;best_step=0;stop=None
    for step in range(1,len(rows)+1):
        report=by_step.get(step)
        if report and report['g_mse']<best:best=report['g_mse'];best_step=step;winner=report
        if step>best_step+resolved['patience']:stop=step;break
    expected=dict(executed_updates=len(rows),selected_step=best_step if winner else None,stop_reason='G_patience',
        checkpoint_available=winner is not None,selection_status='G_selected' if winner else 'raw_unselected_at_cap',
        quality_eligible=False,native_full_anchor_qualified=False)
    from .node_v2 import _canonical
    if stop!=len(rows) or _canonical(state['summary'])!=_canonical(expected) or winner is None:
        raise ValueError('NODE native patience or selected checkpoint differs')
    for name,steps in (('rolling_window',reports[-1]['snapshot_steps']),('selected_window',winner['snapshot_steps'])):
        window=state[name]
        if (type(window) is not list or len(window)!=len(steps)
                or any(type(row.get('step')) is not int or row['step']!=step or set(row)!={'step','model'} for row,step in zip(window,steps,strict=True))):
            raise ValueError('NODE retained selected/current window differs')
    return dict(cursor=cursor,epoch=epoch,seen=seen,ordered=ordered,visited=visited)


def inspect_node_records(root,read,request,arrays):
    import pandas as pd
    from . import node_v2 as native
    config=native.NodeV2Config.parse(request['config']);resolved=config.resolved();manifest=read('worker/snapshot/manifest.json')
    if (manifest.get('schema')!='research/v2-node-snapshot' or config.state['mode'] not in ('native_anchor','search')
            or native._canonical(manifest.get('config'))!=native._canonical(config.to_state())
            or native._canonical(manifest.get('resolved'))!=native._canonical(resolved)
            or manifest.get('identity',{}).get('source')!=native.source_identity()):
        raise ValueError('NODE native snapshot/config/source differs')
    declared=manifest['arrays'];seen=set();state=_metadata(manifest['payload'],declared,seen)
    if seen!=set(declared):raise ValueError('Unreferenced NODE numeric array')
    fields={'initial_model','selected_model','selected_optimizer','raw_model','raw_optimizer','rolling_window','selected_window',
        'reports','trace','sampler','rng','summary','transform','seed','max_updates','n_fit','n_select','F_identity','G_identity','initialization','prediction_binding'}
    if type(state) is not dict or set(state)!=fields:raise ValueError('NODE native endpoint fields differ')
    receipt=read('worker/fit/fit-receipt.json');resources=read('worker/terminal.json')['adapter_resources']
    if (receipt.get('status')!='completed_ineligible' or receipt.get('budget')!={'native_stopping':True}
            or receipt.get('identity')!=manifest['identity'] or receipt.get('config')!=config.to_state()
            or receipt.get('resolved')!=resolved or type(receipt.get('seed')) is not int or receipt['seed']!=request['fit_seed']
            or receipt.get('phase')!='complete' or type(receipt.get('executed_updates')) is not int
            or receipt['executed_updates']!=len(state['trace'])
            or native._canonical(receipt.get('summary'))!=native._canonical(state['summary'])
            or native._canonical(receipt.get('trace'))!=native._canonical(state['trace'])
            or any(native._canonical(resources.get(k))!=native._canonical(v) for k,v in state['summary'].items())
            or resources.get('config')!=config.to_state() or resources.get('fitted') is not True or resources.get('full_workload_qualified') is not False
            or state.get('max_updates') is not None or type(state.get('seed')) is not int or state['seed']!=request['fit_seed']):
        raise ValueError('NODE native receipt/selected state differs')
    path=root/'worker/snapshot/arrays.npz';ref=manifest['array_file']
    if (ref.get('path')!='arrays.npz' or path.is_symlink() or path.stat().st_size>1024**3):raise ValueError('NODE array allocation differs')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=ref['sha256']:raise ValueError('NODE array bytes differ')
    # Only transform/sampler arrays are loaded. All model/optimizer arrays are
    # streamed and checked below without retaining their simultaneous copies.
    def small(value):
        if not isinstance(value,_Array) or math.prod(declared[value.name]['shape'])*np.dtype(declared[value.name]['dtype']).itemsize>40*1024**2:
            raise ValueError('NODE small metadata array allocation differs')
        with np.load(path,allow_pickle=False) as saved:return saved[value.name].copy()
    _bounded_tensor_archive(path,declared,1024**3,allowed_dtypes=('float32','float64','int64','uint32','uint8','bool'))
    transform_state={k:small(v) if isinstance(v,_Array) else copy.deepcopy(v) for k,v in state['transform'].items()}
    transform=native.NodeTransform.from_state(transform_state)
    schema=config.state['feature_schema'];frames={k:pd.DataFrame(arrays[k]['X'],columns=schema['columns']) for k in ('F','G')}
    nF=len(frames['F']);nG=len(frames['G']);yF=native._target(arrays['F']['y'])
    if (type(state.get('n_fit')) is not int or type(state.get('n_select')) is not int or state['n_fit']!=nF or state['n_select']!=nG
            or transform.state['n_fit']!=nF or transform.state['seed']!=request['fit_seed'] or transform.state['schema']!=schema
            or native._canonical(transform.state['preprocessing'])!=native._canonical(native._preprocess(frames['F'],yF,schema))):
        raise ValueError('NODE complete F/G/preprocessing context differs')
    features={k:transform.transform(v) for k,v in frames.items()}
    for role in ('F','G'):
        expected=dict(features=native._array_identity(features[role]),target=native._array_identity(native._target(arrays[role]['y'])))
        if native._canonical(state[role+'_identity'])!=native._canonical(expected):raise ValueError('NODE full transformed inputs differ')
    count=min(1000,nF);initialization=dict(requested_rows=1000,actual_rows=count,ordered_indices=list(range(count)),prefix=native._array_identity(features['F'][:count]))
    if native._canonical(state['initialization'])!=native._canonical(initialization):raise ValueError('NODE full native initialization prefix differs')
    width=features['F'].shape[1];trees=resolved['trees'];depth=resolved['depth'];prefix='0.0.'
    shapes={prefix+'response':[trees,3,2**depth],prefix+'feature_selection_logits':[width,trees,depth],
        prefix+'feature_thresholds':[trees,depth],prefix+'log_temperatures':[trees,depth],
        prefix+'bin_codes_1hot':[depth,2**depth,2],prefix+'_is_initialized_tensor':[]}
    gradient_names=set(shapes)-{prefix+'bin_codes_1hot',prefix+'_is_initialized_tensor'}
    codes=((np.arange(2**depth)[None,:]//(2**np.arange(depth))[:,None])%2).astype(np.float32)
    code_digest=hashlib.sha256(np.stack([codes,1-codes],axis=-1).tobytes()).hexdigest()
    def model(value):
        if type(value) is not dict or set(value)!=set(shapes):raise ValueError('NODE model geometry differs')
        for name,shape in shapes.items():
            ref=value[name];dtype='uint8' if name.endswith('_is_initialized_tensor') else 'float32'
            if (not isinstance(ref,_Array) or not ref.tensor or declared[ref.name]['shape']!=shape or declared[ref.name]['dtype']!=dtype
                    or name.endswith('_is_initialized_tensor') and declared[ref.name]['sha256']!=hashlib.sha256(b'\1').hexdigest()
                    or name.endswith('bin_codes_1hot') and declared[ref.name]['sha256']!=code_digest):
                raise ValueError('NODE model tensor geometry or native initialization differs')
    progress=_trace(state,resolved,nF,gradient_names)
    for name in ('initial_model','raw_model','selected_model'):model(state[name])
    for name in ('rolling_window','selected_window'):
        for row in state[name]:model(row['model'])
    if type(state['sampler']) is not dict or set(state['sampler'])!={'permutation','cursor','epoch'}:
        raise ValueError('NODE sampler schema differs')
    permutation=small(state['sampler']['permutation'])
    if (permutation.dtype!=np.dtype('int64') or permutation.shape!=(nF,) or not np.array_equal(np.sort(permutation),np.arange(nF))
            or type(state['sampler']['cursor']) is not int or state['sampler']['cursor']!=progress['cursor']
            or type(state['sampler']['epoch']) is not int or state['sampler']['epoch']!=progress['epoch']
            or permutation[:progress['cursor']].tolist()!=progress['ordered']):raise ValueError('NODE sampler endpoint differs from paid rows')
    nonnegative=set();parameter_shapes={i:shapes[prefix+name] for i,name in enumerate(('response','feature_selection_logits','feature_thresholds','log_temperatures'),1)}
    for name,age in (('raw_optimizer',len(state['trace'])),('selected_optimizer',state['summary']['selected_step'])):
        optimizer=state[name];expected_group=dict(resolved['optimizer'],params=list(range(6)))
        if (set(optimizer)!={'state','param_groups'} or native._canonical(optimizer['param_groups'])!=native._canonical([expected_group])
                or set(optimizer['state'])!=set(parameter_shapes)):raise ValueError('NODE QHAdam native groups/slots differ')
        weights=[0.,0.]
        for _ in range(age):weights=[1+beta*w for beta,w in zip(resolved['optimizer']['betas'],weights)]
        for i,slot in optimizer['state'].items():
            if (type(i) is not int or set(slot)!={'beta1_weight','beta2_weight','exp_avg','exp_avg_sq'}
                    or any(type(slot[k]) is not float or slot[k]!=w for k,w in zip(('beta1_weight','beta2_weight'),weights))):
                raise ValueError('NODE QHAdam applied-update age differs')
            for key in ('exp_avg','exp_avg_sq'):
                ref=slot[key]
                if (not isinstance(ref,_Array) or not ref.tensor or declared[ref.name]['dtype']!='float32'
                        or declared[ref.name]['shape']!=parameter_shapes[i]):raise ValueError('NODE QHAdam moment geometry differs')
                if key=='exp_avg_sq':nonnegative.add(ref.name)
    _stream_arrays(path,declared,nonnegative)
    if state['prediction_binding']!=_prediction_binding(state,config,declared):raise ValueError('NODE prediction-state/transform binding differs')
    return dict(procedure='native_NODE_qhadam',F_context_rows=nF,G_context_rows=nG,encoded_width=width,
        executed_updates=len(state['trace']),selected_step=state['summary']['selected_step'],native_stop=state['summary']['stop_reason'],
        visited_F_rows=len(progress['visited']),finite_recorded_gradients_checked=True,finite_tensor_payload_checked=True,
        optimizer_ages_and_second_moments_checked=True,native_batch_size=resolved['batch_size'],trees=trees,depth=depth,
        rolling_checkpoints=resolved['latest_checkpoints'],microbatch_rows=resolved.get('execution',{}).get('microbatch_rows'),
        actual_optimization_independently_reexecuted=False,averaged_weight_transition_independently_reexecuted=False,
        RNG_transition_independently_reexecuted=False,feature_transform_fitting_independently_reexecuted=False,
        persistent_training_resume='Separate source-bound native checkpoint/recovery qualification required',
        scope='Recorded full-native F/G/initialization, paid rows/gradients, strict G reports and retained windows, sampler endpoint, QHAdam ages and streamed finite tensors; actual optimization, averaging and RNG recovery execution remain trusted')
