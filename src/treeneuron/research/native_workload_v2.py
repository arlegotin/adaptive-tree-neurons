"""Recorded native procedures and resource observations, without promotion.

The managed generation remains trusted execution evidence. This reader does not
refit, predict, independently reproduce optimization or grant quality authority.
Family-specific observation checks are admitted explicitly as implemented.
"""
import copy
import hashlib
import io
import math
from pathlib import Path
import zipfile

import numpy as np

from treeneuron.config import canonical_hash
from .deployment_worker_v2 import _json
from .native_generation_v2 import audit_native_generation
from .workqueue import _inventory


def _bound_reader(root,inventory):
    refs={row['path']:row['sha256'] for row in inventory['files']}
    def read(relative):
        path=root/relative
        if relative not in refs or path.is_symlink():raise ValueError('Missing committed workload record')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=refs[relative]:raise ValueError('Workload record bytes changed')
        return _json(raw)
    return read,refs


def _process_observation(root,read,refs):
    process=read('process/process.json');start=read('process/started.json');spawn=read('process/spawned.json')
    keys=('argv','cwd','environment_overrides','identity','started_utc_ns','timeout_ns',
        'rss_limit_bytes','poll_interval_ns','terminate_grace_ns','log_limit_bytes')
    if (start.get('schema')!='research/v2-process-start' or any(start.get(k)!=process.get(k) for k in keys)
            or spawn.get('schema')!='research/v2-owned-process' or spawn.get('new_session') is not True
            or type(process.get('child_pid')) is not int or process['child_pid']<=0
            or spawn.get('child_pid')!=process['child_pid'] or spawn.get('child_pgid')!=process['child_pid']
            or type(spawn.get('observed_utc_ns')) is not int or spawn['observed_utc_ns']<start['started_utc_ns']
            or type(process.get('elapsed_ns')) is not int or process['elapsed_ns']<=0
            or type(process.get('rss_limit_bytes')) is not int or process['rss_limit_bytes']<=0):
        raise ValueError('Owned workload process start/closure differs')
    name='process/resource-samples.jsonl';digest=hashlib.sha256();count=0;peak=0;last=0
    if name not in refs or (root/name).is_symlink():raise ValueError('Missing committed RSS samples')
    with (root/name).open('rb') as stream:
        for raw in stream:
            digest.update(raw);row=_json(raw)
            if (not raw.endswith(b'\n') or type(row) is not dict or set(row)!={'elapsed_ns','root_rss_bytes'}
                    or type(row['elapsed_ns']) is not int or not last<row['elapsed_ns']<=process['elapsed_ns']
                    or type(row['root_rss_bytes']) is not int or row['root_rss_bytes']<0):
                raise ValueError('Invalid recorded RSS sample')
            last=row['elapsed_ns'];peak=max(peak,row['root_rss_bytes']);count+=1
    if (digest.hexdigest()!=refs[name] or count==0 or peak<=0
            or process.get('rss_sample_count')!=count or process.get('observed_peak_root_rss_bytes')!=peak
            or peak>process['rss_limit_bytes']):
        raise ValueError('Recorded RSS peak/count/guard differs from raw observations')
    return dict(observed_peak_root_rss_bytes=peak,sample_count=count,
        rss_limit_bytes=process['rss_limit_bytes'],process_elapsed_ns=process['elapsed_ns'],
        scope='Root worker samples only; no aggregate-descendant or strict peak guarantee')


def _inspect_p13(root,read,request,arrays):
    # Replay F-derived arithmetic, never a fit/least-squares solve or inference.
    import pandas as pd
    from .p13_simple import _array_identity,_basis,_config,_features,_preprocessing,_source,_validate_preprocessing
    state=read('worker/snapshot/manifest.json');config=_config(request['config'])
    fields={'schema','config','preprocessing','basis','encoded_width','basis_width','seed',
        'F_identity','source_identity','runtime_identity','arrays','array_file'}
    if (type(state) is not dict or set(state)!=fields or state.get('schema')!='research/v2-p13-snapshot' or state.get('config')!=config
            or state.get('seed')!=request['fit_seed'] or state.get('source_identity')!=_source()):
        raise ValueError('Closed P13 training identity differs')
    F=arrays['F'];y=F['y'].astype(np.float64);frame=pd.DataFrame(F['X'],columns=config['feature_schema']['columns'])
    preprocessing=state['preprocessing'];_validate_preprocessing(preprocessing,config['feature_schema'])
    if preprocessing!=_preprocessing(frame,y,config['feature_schema']):
        raise ValueError('P13 preprocessing is not derived from complete F')
    features,numeric=_features(frame,config['feature_schema'],preprocessing)
    if canonical_hash(state.get('F_identity'))!=canonical_hash(dict(features=_array_identity(features),target=_array_identity(y))):
        raise ValueError('P13 recorded full F context differs')
    basis=state['basis'];method=config['estimator'];knots=[]
    if method=='spline':
        for values in features[:,:numeric].T:
            k=np.unique(np.quantile(values,np.linspace(0,1,config['parameters']['n_knots']+2)[1:-1],method='linear'))
            knots.append(k[(k>values.min())&(k<values.max())].tolist())
    if canonical_hash(basis)!=canonical_hash(dict(kind=method,knots=knots,objective='sum_squared_error_plus_alpha_l2_unpenalized_intercept')):
        raise ValueError('P13 basis is not the declared complete-F design')
    design=_basis(features,numeric,method,knots)
    if (any(type(state.get(k)) is not int for k in ('encoded_width','basis_width'))
            or state['encoded_width']!=features.shape[1] or state['basis_width']!=design.shape[1]):
        raise ValueError('P13 recorded design geometry differs')
    reference=state['array_file'];path=root/'worker/snapshot/arrays.npz'
    if (type(reference) is not dict or set(reference)!={'path','sha256'} or reference['path']!='arrays.npz'
            or path.is_symlink() or not path.is_file() or path.stat().st_size>8*1024**2):
        raise ValueError('Invalid P13 array payload reference')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('P13 array payload bytes differ')
    names={'coefficients','knots','knot_offsets'}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members=archive.infolist()
            if (len(members)!=3 or {m.filename for m in members}!={k+'.npy' for k in names}
                    or sum(m.file_size for m in members)>8*1024**2):raise ValueError('P13 array payload allocation differs')
        with np.load(io.BytesIO(raw),allow_pickle=False) as saved:
            payload={k:saved[k].copy() for k in saved.files}
    except (OSError,KeyError,TypeError,ValueError,zipfile.BadZipFile) as error:
        raise ValueError('Invalid P13 array payload') from error
    if (set(payload)!=names or canonical_hash(state['arrays'])!=canonical_hash({k:_array_identity(v) for k,v in payload.items()})
            or any(v.ndim!=1 or v.dtype!=(np.dtype('int64') if k=='knot_offsets' else np.dtype('float64'))
                or not np.isfinite(v).all() for k,v in payload.items())
            or payload['coefficients'].shape!=(design.shape[1],)
            or not np.array_equal(payload['knots'],np.asarray([v for group in knots for v in group],dtype=np.float64))
            or not np.array_equal(payload['knot_offsets'],np.cumsum([0]+[len(group) for group in knots],dtype=np.int64))
            or method=='mean' and payload['coefficients'][0]!=0.):
        raise ValueError('P13 array/coefficient geometry, dtype or finiteness differs')
    worker=read('worker/terminal.json');resources=worker['adapter_resources']
    if (resources.get('fitted') is not True or resources.get('full_workload_qualified') is not False
            or any(type(resources.get(k)) is not int for k in ('encoded_width','basis_width','total_parameters'))
            or resources.get('encoded_width')!=features.shape[1] or resources.get('basis_width')!=design.shape[1]
            or resources.get('total_parameters')!=design.shape[1]):
        raise ValueError('P13 worker design record differs')
    return dict(procedure='closed_form',estimator=method,F_context_rows=len(y),
        F_identity=copy.deepcopy(state['F_identity']),encoded_width=features.shape[1],basis_width=design.shape[1],
        finite_coefficient_payload_checked=True,iterative_gradients='not_applicable',actual_optimization_independently_reexecuted=False,
        scope='Recorded complete-F design and target/preprocessing arithmetic; managed closed fit/reload is trusted')


def _gated_shapes(family,recipe,summary):
    d=summary['encoded_width'];h=recipe['width'];m=summary['effective_count'];shapes={}
    def linear(name,outputs,inputs,bias=True):
        shapes[name+'.weight']=[outputs,inputs]
        if bias:shapes[name+'.bias']=[outputs]
    if family=='P05':
        if m<h:linear('ordinary',h-m,d)
        if m:
            linear('gate',m,d)
            for name in (('center','contrast') if recipe['variant'].startswith('centered') else ('left','right')):
                linear(name,m,d)
        linear('head',1,h);buffers={}
    else:
        prefix='incumbent.' if family=='P02' else ''
        linear(prefix+'layer.base',h,d);linear(prefix+'head',1,h)
        ids=[] if family=='P02' else summary['selected_ids']
        buffers={prefix+'layer.grown_ids':np.asarray(ids,dtype=np.int64)}
        if family=='P02':
            extra=summary['ordinary_width']-h
            if extra:linear('added_affine',extra,d);linear('added_head',1,extra,bias=False)
        elif m:
            linear('layer.gate',m,d);linear('layer.contrast',m,d)
    return shapes,buffers


def _gated_payload(root,state,recipe,summary):
    from .p13_simple import _array_identity,_hash
    shapes,buffers=_gated_shapes(recipe['family'],recipe,summary)
    expected={k:dict(dtype='float32',shape=v) for k,v in shapes.items()}
    expected.update({k:dict(dtype='int64',shape=list(v.shape)) for k,v in buffers.items()})
    if canonical_hash(state.get('payload'))!=canonical_hash(expected):raise ValueError('Gated tensor geometry differs')
    path=root/'worker/snapshot/weights.npz'
    if path.is_symlink() or path.stat().st_size>8*1024**2:raise ValueError('Gated tensor payload allocation differs')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=state['weights_sha256']:raise ValueError('Gated tensor payload bytes differ')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members=archive.infolist()
        if (len(members)!=len(expected) or {m.filename for m in members}!={k+'.npy' for k in expected}
                or sum(m.file_size for m in members)>8*1024**2):raise ValueError('Gated tensor payload allocation differs')
    with np.load(io.BytesIO(raw),allow_pickle=False) as saved:values={k:saved[k].copy() for k in saved.files}
    if (set(values)!=set(expected) or any(v.dtype!=np.dtype(expected[k]['dtype'])
            or list(v.shape)!=expected[k]['shape'] or not np.isfinite(v).all() for k,v in values.items())
            or any(not np.array_equal(values[k],v) for k,v in buffers.items())
            or {r['name'] for r in summary['parameter_ages']}!=set(shapes)
            or sum(values[k].size for k in shapes)!=summary['total_parameters']
            or _hash({k:_array_identity(v) for k,v in values.items()})!=summary['selected_native_state_sha256']):
        raise ValueError('Gated tensor values, parameter ages or selected identity differ')


def _finite_number(value):return type(value) in (int,float) and math.isfinite(value) and value>=0


def _gated_stages(root,refs,recipe,summary):
    name='worker/fit/events.jsonl';expected=refs.get(name)
    if not expected or summary['event_stream_sha256']!=expected:raise ValueError('Gated update stream commitment differs')
    stage_index=0;result=[];digest=hashlib.sha256();nF=summary['fit_rows'];batch=recipe['batch_size']
    per_epoch=math.ceil(nF/batch)
    def fresh():
        return dict(updates=0,last_evaluation=None,best=None,best_offset=None,bad=0,visited=set(),epoch=None,epoch_rows=set(),first_slot=None)
    progress=fresh()
    with (root/name).open('rb') as stream:
        for raw in stream:
            digest.update(raw);event=_json(raw)
            if (not raw.endswith(b'\n') or type(event) is not dict or stage_index>=2
                    or event.get('stage')!=('base','continuation')[stage_index]):
                raise ValueError('Gated stage ordering or record differs')
            limit=(recipe['base_updates'],recipe['continuation_updates'])[stage_index]
            absolute=0 if stage_index==0 else summary['stages'][0]['selected_absolute_update']
            offset=progress['updates'];kind=event.get('type')
            if kind=='evaluation':
                if (set(event)!={'type','stage','update','stage_update','G_mse','improved'}
                        or type(event['stage_update']) is not int or event['stage_update']!=offset
                        or type(event['update']) is not int or event['update']!=absolute+offset or progress['last_evaluation']==offset
                        or offset and offset%recipe['validation_interval'] and offset!=limit
                        or not _finite_number(event['G_mse'])):raise ValueError('Gated G evaluation grid differs')
                improved=progress['best'] is None or event['G_mse']<progress['best']
                if event['improved'] is not improved:raise ValueError('Gated strict-best decision differs')
                if improved:progress.update(best=event['G_mse'],best_offset=offset,bad=0)
                else:progress['bad']+=1
                progress['last_evaluation']=offset
            elif kind=='update':
                if (set(event)!={'type','stage','update','stage_update','row_indices','F_online_mse','learning_rates','gradient_norms'}
                        or progress['last_evaluation'] is None or offset>=limit or progress['bad']>=recipe['patience']
                        or offset%recipe['validation_interval']==0 and progress['last_evaluation']!=offset
                        or type(event['stage_update']) is not int or event['stage_update']!=offset+1
                        or type(event['update']) is not int or event['update']!=absolute+offset+1
                        or not _finite_number(event['F_online_mse'])):raise ValueError('Gated applied update/grid differs')
                update=event['update'];slot=(update-1)%per_epoch;epoch=(update-1)//per_epoch;rows=event['row_indices']
                if (type(rows) is not list or len(rows)!=min(batch,nF-slot*batch)
                        or any(type(r) is not int or not 0<=r<nF for r in rows) or len(set(rows))!=len(rows)):
                    raise ValueError('Gated sampler batch differs')
                if progress['epoch']!=epoch:progress.update(epoch=epoch,epoch_rows=set(),first_slot=slot)
                if progress['epoch_rows'].intersection(rows):raise ValueError('Repeated gated row within sampler epoch')
                progress['epoch_rows'].update(rows);progress['visited'].update(rows)
                if slot==per_epoch-1 and progress['first_slot']==0 and len(progress['epoch_rows'])!=nF:
                    raise ValueError('Incomplete gated F epoch')
                extra=(recipe['family'] in ('P00','P01') and summary['effective_count']>0
                    or recipe['family']=='P02' and summary['ordinary_width']>recipe['width'])
                groups=2 if stage_index==1 and extra else 1
                norms=event['gradient_norms'];rates=event['learning_rates']
                factor=1. if recipe['scheduler']=='constant' else (1+math.cos(math.pi*(update-1)/recipe['schedule_horizon']))/2
                if (type(norms) is not list or len(norms)!=groups or not all(_finite_number(v) for v in norms)
                        or type(rates) is not list or len(rates)!=groups or not all(_finite_number(v) for v in rates)
                        or rates!=[recipe['lr']*factor]*groups):raise ValueError('Gated recorded gradients or scheduled learning rates differ')
                progress['updates']+=1
            elif kind=='selection':
                selected=progress['best_offset'];stop='patience' if progress['bad']>=recipe['patience'] else 'max_updates'
                if (set(event)!={'type','stage','selected_absolute_update','selected_stage_update','executed_stage_updates','stop'}
                        or any(type(event[k]) is not int for k in ('selected_absolute_update','selected_stage_update','executed_stage_updates'))
                        or offset<=0 or progress['last_evaluation']!=offset or selected is None
                        or stop=='max_updates' and offset!=limit
                        or event['selected_absolute_update']!=absolute+selected or event['selected_stage_update']!=selected
                        or event['executed_stage_updates']!=offset or event['stop']!=stop):
                    raise ValueError('Gated selected/executed stage endpoint differs')
                endpoint=dict(selected_absolute_update=absolute+selected,selected_stage_update=selected,
                    executed_stage_updates=offset,executed_absolute_update=absolute+offset,stop=stop,G_mse=progress['best'])
                if canonical_hash(endpoint)!=canonical_hash(summary['stages'][stage_index]):raise ValueError('Gated stage summary differs from update stream')
                result.append(dict(stage=event['stage'],executed_updates=offset,selected_updates=selected,
                    visited_F_rows=len(progress['visited']),complete_recorded_F_coverage=len(progress['visited'])==nF,stop=stop))
                stage_index+=1;progress=fresh()
            else:raise ValueError('Unknown gated training event')
    if digest.hexdigest()!=expected or stage_index!=2:raise ValueError('Gated training stream is incomplete or changed')
    return result


def _inspect_gated(root,read,refs,request,arrays):
    import pandas as pd
    from .gated_config import GatedConfig
    from .gated_native import _matrix,_validate_summary
    from .p13_simple import _source,_array_identity,_preprocessing
    family=request['family_id'];config=GatedConfig.parse(family,request['config']);recipe=config.resolved()
    state=read('worker/snapshot/manifest.json');worker=read('worker/terminal.json');summary=state['resources']
    if (state.get('schema')!='research/v2-gated-prediction-snapshot' or state.get('family')!=family
            or canonical_hash(state.get('config'))!=canonical_hash(config.to_state()) or state.get('seed')!=request['fit_seed']
            or state.get('source')!=_source() or summary.get('source')!=state['source']
            or canonical_hash(worker['adapter_resources'])!=canonical_hash(summary)
            or summary.get('status')!='completed_ineligible' or summary.get('time_limit_exceeded') is not False
            or summary.get('seed')!=request['fit_seed'] or summary.get('dependencies')!=state['dependencies']):
        raise ValueError('Gated native snapshot/worker identity differs')
    _validate_summary(config,summary)
    if config.to_state()['mode'] not in ('native_anchor','search'):raise ValueError('Shortened gated fixture is not a native workload')
    frames={k:pd.DataFrame(v['X'],columns=config.feature_schema['columns']) for k,v in arrays.items() if k in ('F','G')}
    yF=arrays['F']['y'].astype(np.float64);yG=arrays['G']['y'].astype(np.float64)
    preprocessing=_preprocessing(frames['F'],yF,config.feature_schema)
    if (canonical_hash(preprocessing)!=canonical_hash(state['preprocessing'])
            or canonical_hash(preprocessing)!=canonical_hash(summary['preprocessing'])
            or summary['fit_rows']!=len(yF) or summary['select_rows']!=len(yG)):
        raise ValueError('Gated complete F/G/preprocessing context differs')
    F64,F=_matrix(frames['F'],config.feature_schema,preprocessing);_,G=_matrix(frames['G'],config.feature_schema,preprocessing)
    inputs=read('worker/fit/input-manifest.json')
    expected=dict(schema='research/v2-gated-fit-input',source=state['source'],dependencies=state['dependencies'],
        config=config.to_state(),resolved_parameters=recipe,seed=request['fit_seed'],preprocessing=preprocessing,
        F_features=_array_identity(F64),G_features=_array_identity(G),F_target=_array_identity(yF),G_target=_array_identity(yG),accessed_roles=['F','G'])
    if canonical_hash(inputs)!=canonical_hash(expected):raise ValueError('Gated recorded transformed inputs differ')
    _gated_payload(root,state,recipe,summary);stages=_gated_stages(root,refs,recipe,summary)
    return dict(procedure='two_stage_adamw',F_context_rows=len(yF),G_context_rows=len(yG),
        encoded_width=F.shape[1],total_parameters=summary['total_parameters'],effective_count=summary['effective_count'],
        stages=stages,finite_recorded_gradients_checked=True,finite_tensor_payload_checked=True,
        actual_optimization_independently_reexecuted=False,persistent_training_resume=False,
        scope='Recorded native stages, full input/visited rows, gradients, schedule, structural tensors and strict G selection; managed optimization remains trusted')


def _epoch_orders(root,refs,summary):
    """Stream one epoch at a time, even for a long native trajectory."""
    name='worker/fit/F-row-orders.npz';path=root/name
    if path.is_symlink() or refs.get(name)!=summary['row_orders_sha256']:
        raise ValueError('Native epoch order commitment differs')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    if digest.hexdigest()!=refs[name]:raise ValueError('Native epoch order bytes differ')
    n=summary['fit_rows'];epochs=summary['executed_epochs'];expected=np.arange(n,dtype=np.int64)
    with zipfile.ZipFile(path) as archive:
        members=archive.infolist()
        if len(members)!=1 or members[0].filename!='row_indices.npy' or members[0].file_size!=epochs*n*8+128:
            raise ValueError('Native epoch order archive allocation differs')
        with archive.open(members[0]) as stream:
            version=np.lib.format.read_magic(stream)
            if version!=(1,0):raise ValueError('Unsupported native epoch order array format')
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
            if shape!=(epochs,n) or fortran or dtype!=np.dtype('int64'):
                raise ValueError('Native epoch order geometry/dtype differs')
            for row in summary['history']:
                raw=stream.read(n*8)
                if (len(raw)!=n*8 or hashlib.sha256(raw).hexdigest()!=row['row_order_sha256']
                        or not np.array_equal(np.sort(np.frombuffer(raw,dtype=np.int64)),expected)):
                    raise ValueError('Native epoch order is not a complete F permutation')
            if stream.read(1):raise ValueError('Unexpected native epoch order tail')


def _epoch_payload(root,state,summary,family):
    """Check finite retained tensors without constructing or initializing a model."""
    path=root/'worker/snapshot/weights.npz';limit=128*1024**2
    payload=state['payload']
    if (type(payload) is not dict or not payload or len(payload)>2048 or path.is_symlink()
            or path.stat().st_size>limit):raise ValueError('Native epoch tensor allocation differs')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=state['weights_sha256']:raise ValueError('Native epoch tensor bytes differ')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members=archive.infolist()
        if (len(members)!=len(payload) or {m.filename for m in members}!={k+'.npy' for k in payload}
                or sum(m.file_size for m in members)>limit):raise ValueError('Native epoch tensor allocation differs')
    identity={}
    with np.load(io.BytesIO(raw),allow_pickle=False) as saved:
        for name in saved.files:
            value=saved[name]
            if (value.dtype not in (np.dtype('float32'),np.dtype('int64'),np.dtype('bool'))
                    or not np.isfinite(value).all()
                    or canonical_hash(payload[name])!=canonical_hash(dict(dtype=str(value.dtype),shape=list(value.shape)))):
                raise ValueError('Native epoch tensor dtype, shape or finiteness differs')
            if value.dtype==np.dtype('int64'):
                if name=='category_offsets' and family!='P07':
                    offsets=np.cumsum([0]+summary['cardinalities'][:-1],dtype=np.int64)
                    if not np.array_equal(value,offsets):raise ValueError('Native category offsets differ')
                elif family=='P04' and name.endswith('.num_batches_tracked'):
                    selected=summary['selected_epoch']*math.ceil(summary['fit_rows']/summary['resolved_parameters']['batch_size'])
                    if value.shape!=() or value.item()!=selected:raise ValueError('Native selected BatchNorm age differs')
                else:raise ValueError('Unexpected native epoch integer tensor')
            if value.dtype==np.dtype('bool') and not (family=='P07' and state['config']['variant']=='piecewise'
                    and name.rsplit('.',1)[-1] in ('mask','single_bin_mask')):
                raise ValueError('Unexpected native epoch boolean tensor')
            if name.endswith('running_var') and np.any(value<0):raise ValueError('Negative native running variance')
            identity[name]=dict(shape=list(value.shape),dtype='torch.'+str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if canonical_hash(identity)!=summary['selected_native_state_sha256']:
        raise ValueError('Native epoch selected tensor identity differs')


def _inspect_epochs(root,read,refs,request,arrays):
    import pandas as pd
    family=request['family_id'];tabm=family=='P07'
    if tabm:
        from . import tabm_native as native
        config=native.TabMConfig.parse(request['config']);transform_type=native.TabMTransform;target_type=native.TabMTarget
    else:
        from . import ordinary_neural as native
        config=native.OrdinaryNeuralConfig.parse(family,request['config']);transform_type=native.NeuralTransform;target_type=native.TargetTransform
    state=read('worker/snapshot/manifest.json');worker=read('worker/terminal.json');summary=state['resources']
    prefix='tabm' if tabm else 'ordinary-neural'
    if (state.get('schema')!='research/v2-'+prefix+'-snapshot' or not tabm and state.get('family')!=family
            or canonical_hash(state.get('config'))!=canonical_hash(config.to_state())
            or type(state.get('seed')) is not int or state['seed']!=request['fit_seed']
            or state.get('source')!=native._source() or canonical_hash(worker['adapter_resources'])!=canonical_hash(summary)
            or canonical_hash(read('worker/fit/attempt.json'))!=canonical_hash(summary)
            or config.mode not in ('native_anchor','search') or summary.get('time_limit_exceeded') is not False
            or summary.get('fitted') is not True or summary.get('seed')!=request['fit_seed']
            or tabm and canonical_hash(summary.get('dependencies'))!=canonical_hash(state['dependencies'])
            or summary.get('budget')!={} or summary.get('accessed_roles')!=['F','G']):
        raise ValueError('Native epoch snapshot/worker/input identity differs')
    counters=('fit_rows','select_rows','executed_epochs','selected_epoch','applied_updates','total_parameters','trainable_parameters')
    if (any(type(summary.get(k)) is not int for k in counters)
            or not 1<=summary['total_parameters']<=10_000_000
            or not 0<summary['trainable_parameters']<=summary['total_parameters']):
        raise ValueError('Native epoch counters differ')
    native._validate_summary(config,summary)
    transform=transform_type.from_state(state['feature_transform']);target=target_type.from_state(state['target_transform'])
    frames={k:pd.DataFrame(arrays[k]['X'],columns=config.feature_schema['columns']) for k in ('F','G')}
    yF=arrays['F']['y'].astype(np.float64);yG=arrays['G']['y'].astype(np.float64)
    if (summary['fit_rows']!=len(yF) or summary['select_rows']!=len(yG) or transform.fit_rows!=len(yF)
            or canonical_hash(transform.schema)!=canonical_hash(config.feature_schema)
            or canonical_hash(dict(values=transform.cardinalities))!=canonical_hash(dict(values=summary['cardinalities']))
            or canonical_hash(target.to_state())!=canonical_hash(target_type.fit(yF).to_state())
            or summary['feature_transform_sha256']!=native._canonical(state['feature_transform'])
            or summary['target_transform_sha256']!=native._canonical(state['target_transform'])):
        raise ValueError('Native epoch full F/G transform or target context differs')
    F=transform.transform(frames['F']);G=transform.transform(frames['G'])
    expected=dict(schema='research/v2-'+prefix+'-attempt',source=state['source'],dependencies=state['dependencies'],
        seed=request['fit_seed'],config=config.to_state(),resolved_parameters=config.resolved(),
        feature_transform=transform.to_state(),target_transform=target.to_state(),budget={},
        F_target_sha256=hashlib.sha256(yF.tobytes()).hexdigest(),G_target_sha256=hashlib.sha256(yG.tobytes()).hexdigest(),
        fit_rows=len(yF),select_rows=len(yG),accessed_roles=['F','G'])
    if tabm:
        # This native implementation commits raw targets after its float32
        # conversion, before standardization. Ordinary neural uses float64.
        expected['F_target_sha256']=hashlib.sha256(yF.astype(np.float32).tobytes()).hexdigest()
        expected['G_target_sha256']=hashlib.sha256(yG.astype(np.float32).tobytes()).hexdigest()
        native._validated_bins(config,F[0].shape[1],state['bins']);expected['bins']=state['bins']
        if summary.get('n_numeric')!=F[0].shape[1] or summary['bins_sha256']!=native._canonical(state['bins']):
            raise ValueError('Native TabM numerical geometry/bins differ')
        for role,value in (('F',F),('G',G)):
            expected[role+'_numeric_sha256']=hashlib.sha256(value[0].tobytes()).hexdigest()
            expected[role+'_categorical_sha256']=hashlib.sha256(value[1].tobytes()).hexdigest()
    else:
        expected.update(F_features_sha256=hashlib.sha256(F.tobytes()).hexdigest(),G_features_sha256=hashlib.sha256(G.tobytes()).hexdigest())
    if canonical_hash(read('worker/fit/input-manifest.json'))!=canonical_hash(expected):
        raise ValueError('Native epoch transformed input commitments differ')
    origin=0 if tabm else 1;updates=math.ceil(len(yF)/config.resolved()['batch_size'])
    for index,row in enumerate(summary['history']):
        if (type(row['epoch']) is not int or row['epoch']!=index+origin
                or type(row['applied_updates']) is not int or row['applied_updates']!=updates*(index+1)
                or not _finite_number(row['gradient_norm_min']) or not _finite_number(row['gradient_norm_max'])
                or row['gradient_norm_max']<row['gradient_norm_min']):
            raise ValueError('Native epoch gradient or applied-update record differs')
    _epoch_orders(root,refs,summary);_epoch_payload(root,state,summary,family)
    return dict(procedure='native_epoch_adamw',F_context_rows=len(yF),G_context_rows=len(yG),
        executed_epochs=summary['executed_epochs'],selected_epoch=summary['selected_epoch'],native_epoch_index_origin=origin,
        native_stop=summary['native_stop'],full_F_coverage_each_epoch_verified=True,
        finite_recorded_gradients_checked=True,finite_tensor_payload_checked=True,
        training_objective='mean_member_mse' if tabm else 'mse',
        selected_F_mean=summary.get('selected_F_mean'),actual_optimization_independently_reexecuted=False,
        feature_transform_fitting_independently_reexecuted=False,persistent_training_resume=False,
        scope='Recorded native epochs, complete row permutations, transformed inputs, gradients, tensor payload and native G selection; managed optimization and fitted quantiles remain trusted')


def inspect_native_workload(*,request_path,expected_request_sha256,native_reference,prepared_root,
        expected_manifest_sha256,outer_fold,expected_fit_seed,expected_configuration):
    args=copy.deepcopy(dict(request_path=request_path,expected_request_sha256=expected_request_sha256,
        native_reference=native_reference,prepared_root=prepared_root,expected_manifest_sha256=expected_manifest_sha256,
        outer_fold=outer_fold,expected_fit_seed=expected_fit_seed,expected_configuration=expected_configuration))
    config=args['expected_configuration'];family=config.get('family') if type(config) is dict else None
    if family not in ('P00','P01','P02','P03','P04','P05','P07','P13'):raise ValueError('Recorded native workload observation for this family is not implemented')
    generation=audit_native_generation(**args);root=Path(args['native_reference']['root'])
    inventory=_inventory(root)
    if canonical_hash(inventory)!=args['native_reference']['inventory_sha256']:
        raise ValueError('Native workload inventory changed')
    read,refs=_bound_reader(root,inventory);process=_process_observation(root,read,refs)
    from .method_worker import _validate_request
    request,arrays=_validate_request(Path(args['request_path']),args['expected_request_sha256'])
    if family=='P13':training=_inspect_p13(root,read,request,arrays)
    elif family in ('P03','P04','P07'):training=_inspect_epochs(root,read,refs,request,arrays)
    else:training=_inspect_gated(root,read,refs,request,arrays)
    if audit_native_generation(**args)!=generation or _inventory(root)!=inventory:
        raise ValueError('Native workload inputs or evidence changed during inspection')
    return dict(schema='research/v2-recorded-native-workload-observation',status='checked_recorded_native_workload',
        family_id=family,source_sha256=generation['source_sha256'],configuration_sha256=generation['configuration_sha256'],
        generation=generation,input_rows={k:len(v['X']) for k,v in arrays.items()},training=training,
        process_observation_recomputed=True,process_observation=process,
        native_workload_qualified=False,quality_eligible=False,T_accessed=False,
        scope='Recorded native procedure, primary role/context and raw resource observations; no complete qualification registry, new fit/inference, V labels or protected authority')
