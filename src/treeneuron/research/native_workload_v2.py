"""Recorded native procedures and resource observations, without promotion.

The managed generation remains trusted execution evidence. This reader does not
refit, predict, independently reproduce optimization or grant quality authority.
Family-specific observation checks are admitted explicitly as implemented.
"""
import copy
import hashlib
import io
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


def inspect_native_workload(*,request_path,expected_request_sha256,native_reference,prepared_root,
        expected_manifest_sha256,outer_fold,expected_fit_seed,expected_configuration):
    args=copy.deepcopy(dict(request_path=request_path,expected_request_sha256=expected_request_sha256,
        native_reference=native_reference,prepared_root=prepared_root,expected_manifest_sha256=expected_manifest_sha256,
        outer_fold=outer_fold,expected_fit_seed=expected_fit_seed,expected_configuration=expected_configuration))
    config=args['expected_configuration'];family=config.get('family') if type(config) is dict else None
    if family!='P13':raise ValueError('Recorded native workload observation for this family is not implemented')
    generation=audit_native_generation(**args);root=Path(args['native_reference']['root'])
    inventory=_inventory(root)
    if canonical_hash(inventory)!=args['native_reference']['inventory_sha256']:
        raise ValueError('Native workload inventory changed')
    read,refs=_bound_reader(root,inventory);process=_process_observation(root,read,refs)
    from .method_worker import _validate_request
    request,arrays=_validate_request(Path(args['request_path']),args['expected_request_sha256'])
    training=_inspect_p13(root,read,request,arrays)
    if audit_native_generation(**args)!=generation or _inventory(root)!=inventory:
        raise ValueError('Native workload inputs or evidence changed during inspection')
    return dict(schema='research/v2-recorded-native-workload-observation',status='checked_recorded_native_workload',
        family_id=family,source_sha256=generation['source_sha256'],configuration_sha256=generation['configuration_sha256'],
        generation=generation,input_rows={k:len(v['X']) for k,v in arrays.items()},training=training,
        process_observation_recomputed=True,process_observation=process,
        native_workload_qualified=False,quality_eligible=False,T_accessed=False,
        scope='Recorded native procedure, primary role/context and raw resource observations; no complete qualification registry, new fit/inference, V labels or protected authority')
