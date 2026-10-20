"""Saved-state feature-query primitives; no public protected data reader.

The public preflight requires the genuine final protocol and owner journal.
Its output is metadata, not permission to bypass a later owned coordinator.
Private forward fixtures authenticate their saved bytes and numerical output
only, never training generation or protected-data access.
"""
import copy
import errno
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

from .deployment_worker_v2 import _digest,_json,_source,_THREADS,_ROOT
from .protected_journal_v2 import _path,_reference,_canonical,_hash,_write_new

_FALSE=dict(native_qualification_authenticated=False,protected_access_authorized=False,T_accessed=False)
_QUERY_FIELDS={'query_id','content_identity','job_id','fit_seed','dataset','outer_fold','family_id','split_hash',
    'manifest_hash','query_role','feature_reference','model_binding'}


def _snapshot_inventory(root):
    """Streaming byte-equivalent of the established queue inventory format."""
    root=_path(root)
    if not root.is_dir():raise ValueError('Closed native snapshot directory required')
    files=[];directories=[]
    for path in sorted(root.rglob('*')):
        _path(path)
        if path.is_dir():directories.append(path.relative_to(root).as_posix());continue
        if not path.is_file():raise ValueError('Snapshot contains a nonregular artifact')
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        files.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=digest))
    return dict(files=files,directories=directories)


def _feature_schema(schema):
    if (type(schema) is not dict or set(schema)!={'columns','categorical'} or type(schema['columns']) is not list
            or not 1<=len(schema['columns'])<=128 or type(schema['categorical']) is not list
            or any(type(s) is not str or not s for s in schema['columns']+schema['categorical'])
            or len(set(schema['columns']))!=len(schema['columns']) or len(set(schema['categorical']))!=len(schema['categorical'])
            or not set(schema['categorical'])<=set(schema['columns'])):raise ValueError('Exact semantic native feature schema required')


def _metadata_request(request):
    fields={'schema','source_sha256','protocol_sha256','prediction_manifest_sha256','query','feature_schema','split_kind'}
    if (type(request) is not dict or set(request)!=fields or request['schema']!='research/v2-saved-protected-feature-query'
            or request['source_sha256']!=_source() or not _digest(request['protocol_sha256'])
            or not _digest(request['prediction_manifest_sha256']) or request['split_kind'] not in ('grouped','temporal')):
        raise ValueError('Exact current-source saved-state feature request required')
    _feature_schema(request['feature_schema']);q=request['query']
    if (type(q) is not dict or set(q)!=_QUERY_FIELDS or not _digest(q['query_id']) or type(q['content_identity']) is not dict
            or q['query_role']!='T_features' or type(q['family_id']) is not str or q['family_id'] not in tuple(f'P{i:02}' for i in range(14))
            or type(q['fit_seed']) is not int or not 0<=q['fit_seed']<2**63 or type(q['outer_fold']) is not int or not 0<=q['outer_fold']<5
            or any(type(q[k]) is not str or not q[k] for k in ('job_id','dataset'))
            or any(not _digest(q[k]) for k in ('split_hash','manifest_hash'))):raise ValueError('Frozen explicit T-feature query identity required')
    model=q['model_binding'];fields={'attempt_index','selected_metadata','snapshot_root','snapshot_inventory_sha256','generation_sha256'}
    if (type(model) is not dict or set(model)!=fields or type(model['attempt_index']) is not int or not 0<=model['attempt_index']<32
            or any(not _digest(model[k]) for k in ('snapshot_inventory_sha256','generation_sha256'))):raise ValueError('Complete chosen native snapshot binding required')
    root=_path(model['snapshot_root']);meta=model['selected_metadata'];feature=q['feature_reference']
    for ref in (meta,feature):
        if type(ref) is not dict or set(ref)!={'path','sha256'} or not _digest(ref['sha256']):raise ValueError('Exact feature/model references required')
        _path(ref['path'])
    filename='state.json' if q['family_id'] in ('P08','P09','P10','P11') else 'manifest.json'
    if Path(meta['path'])!=root/filename or Path(feature['path']).name!='T-features.npz' or Path(feature['path']).parent.name!=f"outer-{q['outer_fold']}":
        raise ValueError('Query must retain the original explicit T-feature and selected-model roles')
    identity={k:v for k,v in q.items() if k not in ('query_id','content_identity','model_binding')}
    identity['model_content']=dict(attempt_index=model['attempt_index'],generation_sha256=model['generation_sha256'],
        snapshot_inventory_sha256=model['snapshot_inventory_sha256'],selected_metadata_sha256=meta['sha256'],selected_metadata_name=filename)
    if identity!=q['content_identity'] or _hash(identity)!=q['query_id']:raise ValueError('Frozen query content/ID changed')
    return q,model


def _query_inputs(request):
    """Validate bytes and archive fields before importing any numerical method."""
    q,model=_metadata_request(request);_,raw=_reference(model['selected_metadata']);meta=_json(raw)
    payload=meta.get('payload') if q['family_id'] in ('P08','P09','P10','P11') and type(meta) is dict else meta
    if (type(payload) is not dict or type(payload.get('config')) is not dict
            or payload['config'].get('family')!=q['family_id'] or payload['config'].get('feature_schema')!=request['feature_schema']):
        raise ValueError('Saved native recipe family/schema differs from the frozen feature query')
    root=_path(model['snapshot_root']);inventory=_snapshot_inventory(root)
    if _hash(inventory)!=model['snapshot_inventory_sha256']:raise ValueError('Complete saved native snapshot bytes changed')
    path,features=_reference(q['feature_reference']);fields=['X','row_id','group_id']+(['timestamps'] if request['split_kind']=='temporal' else [])
    try:
        with zipfile.ZipFile(io.BytesIO(features)) as archive:
            members=archive.infolist()
            if (len(members)!=len(fields) or {m.filename for m in members}!={k+'.npy' for k in fields}
                    or sum(m.file_size for m in members)>64*1024**2):raise ValueError('Bounded feature-only T query required; labels/extra fields forbidden')
    except zipfile.BadZipFile as error:raise ValueError('Invalid feature-only query archive') from error
    import numpy as np
    with np.load(io.BytesIO(features),allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in fields}
    X=arrays['X'];rows=arrays['row_id'];groups=arrays['group_id'];n=len(X)
    if (X.ndim!=2 or X.dtype!=np.float64 or not 1<=n<=20000 or X.shape[1]!=len(request['feature_schema']['columns']) or not np.isfinite(X).all()
            or rows.dtype!=np.int64 or rows.shape!=(n,) or len(np.unique(rows))!=n or groups.dtype!=np.int64 or groups.shape!=(n,)):
        raise ValueError('Finite native float64 features and unique aligned int64 row/group identities required')
    if request['split_kind']=='temporal':
        times=arrays['timestamps']
        if (times.shape!=(n,) or times.dtype.kind!='M' or np.datetime_data(times.dtype)[0] not in ('h','m','s','ms','us','ns')
                or np.isnat(times).any() or np.any(times[1:]<=times[:-1])):raise ValueError('Actual increasing hourly/subhour query timestamps required')
    return q,model,root,inventory,arrays


def _output_location(output,request):
    root=_path(output);q=request.get('query',{});model=q.get('model_binding',{});feature=q.get('feature_reference',{})
    protected=[Path(sys.prefix).resolve(),Path(sys.executable).resolve().parent]
    protected.extend((_ROOT/n).resolve() for n in ('src','tests','configs','locks','environment','.git','.agents','.codex','.aws'))
    if 'snapshot_root' in model:protected.append(_path(model['snapshot_root']))
    if 'path' in feature:protected.append(_path(feature['path']).parent)
    if root.exists() or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError('Fresh query output must preserve native model/features/source/runtime')
    return root


def _saved_query(request,*,output):
    """Private inference-only composition, for later owned-worker integration."""
    started=time.perf_counter_ns();request=copy.deepcopy(request);root=_output_location(output,request);root.mkdir(parents=True,exist_ok=False)
    record=dict(schema='research/v2-saved-feature-query-terminal',status='input_or_dependency_failure',request_sha256=_hash(request),
        adapter_import_started=False,finite_aligned_prediction_artifact=False,feature_query_read=False,labels_read=False,**_FALSE,
        scope='Saved-byte/finite-output fixture primitive only; no native training, protocol/access or whole-cost authority')
    try:
        q,model,snapshot,inventory,arrays=_query_inputs(request)
        if not sys.dont_write_bytecode or any(os.environ.get(k)!='1' for k in _THREADS):
            raise ValueError('Saved query requires disabled bytecode and declared single-thread environment')
        record.update(query_id=q['query_id'],family_id=q['family_id'],logical_fit_seed=q['fit_seed'],source_sha256=request['source_sha256'],
            protocol_sha256=request['protocol_sha256'],prediction_manifest_sha256=request['prediction_manifest_sha256'],feature_query_read=True,
            query_rows=len(arrays['X']),feature_reference=copy.deepcopy(q['feature_reference']),model_binding=copy.deepcopy(model))
        import numpy as np
        import pandas as pd
        record['adapter_import_started']=True
        from .method_worker import _ADAPTERS,_initialize_native_threads,loaded_method_modules
        record['torch_threads']=_initialize_native_threads(q['family_id'])
        module,name,_=_ADAPTERS[q['family_id']];cls=getattr(importlib.import_module('treeneuron.research.'+module),name)
        argument='expected_state_sha256' if q['family_id'] in ('P08','P09','P10','P11') else 'expected_manifest_sha256'
        # The loader restores its own saved native seed/config. Logical query seed is not an initialization seed.
        model_instance=cls.load(snapshot,**{argument:model['selected_metadata']['sha256']})
        prediction=np.asarray(model_instance.predict(pd.DataFrame(arrays['X'],columns=request['feature_schema']['columns'])))
        if prediction.dtype not in (np.dtype('float32'),np.dtype('float64')) or prediction.shape!=(len(arrays['X']),) or not np.isfinite(prediction).all():
            raise FloatingPointError('Saved native query produced invalid original-unit predictions')
        _reference(q['feature_reference']);_reference(model['selected_metadata'])
        if _source()!=request['source_sha256'] or _snapshot_inventory(snapshot)!=inventory:raise ValueError('Source/model/features changed during saved-state prediction')
        path=root/'prediction.npz';pending=root/'.prediction.npz.pending'
        with pending.open('xb') as stream:
            np.savez(stream,prediction=prediction,**{k:v for k,v in arrays.items() if k!='X'})
            stream.flush();os.fsync(stream.fileno())
        os.link(pending,path);pending.unlink()
        record.update(status='completed_ineligible',finite_aligned_prediction_artifact=True,prediction_dtype=str(prediction.dtype),
            prediction_reference=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),loaded_heavy_modules=loaded_method_modules())
        return record
    except BaseException as error:
        exhausted=isinstance(error,MemoryError) or isinstance(error,OSError) and error.errno in (errno.ENOMEM,errno.ENOSPC)
        record.update(status='resource_exhausted' if exhausted else 'numerical_failure' if isinstance(error,FloatingPointError)
            else 'interrupted_recoverable' if isinstance(error,KeyboardInterrupt) else 'input_or_dependency_failure',
            error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
        raise
    finally:
        record.update(worker_body_ns=time.perf_counter_ns()-started,
            clock_scope='Private core entry through prediction publication; own terminal publication/return and containing process excluded')
        _write_new(root/'terminal.json',record)


def _compile_query_request(payload,state,*,query_id):
    """Untrusted metadata compiler; no model/feature/label bytes opened here."""
    if (state['phase']!='owners_ready' or state['active'] is not None or query_id in state['committed_queries']
            or state['last_failures'].get(query_id) in ('numerical_failure','irrecoverable_missing')
            or state['last_failures'].get(query_id)=='input_or_dependency_failure' and state['reruns'].get(query_id,0)):
        raise ValueError('Query must be the same available frozen query at the unambiguous prediction stage')
    queries=[q for q in payload['prediction_manifest']['queries'] if q['query_id']==query_id]
    if len(queries)!=1:raise ValueError('Exactly one frozen query required')
    query=queries[0];data=payload['data'][query['dataset']];fold=data['folds'][query['outer_fold']]
    if (fold['outer_fold']!=query['outer_fold'] or fold['split_hash']!=query['split_hash']
            or data['manifest_hash']!=query['manifest_hash'] or fold['roles']['T_features']['reference']!=query['feature_reference']):
        raise ValueError('Query cannot change its frozen data feature role')
    return dict(schema='research/v2-saved-protected-feature-query',source_sha256=payload['source_sha256'],
        protocol_sha256=state['binding']['protocol_sha256'],prediction_manifest_sha256=payload['prediction_manifest']['manifest_hash'],
        query=copy.deepcopy(query),feature_schema=dict(columns=data['feature_names'],categorical=data['categorical_features']),
        split_kind='temporal' if query['dataset']=='bike' else 'grouped')


def prepare_protected_query(reference,*,directory,query_id,episode_root,models_reference,global_barrier_reference,
        qualification_reference,preT_costs_reference,readiness_reference):
    """Genuine protocol/owner gates first; return read-only draft metadata only."""
    from .practical_T_protocol_v2 import load_practical_T_protocol
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,
        qualification_reference=qualification_reference,preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol=load_practical_T_protocol(reference,**kwargs)
    from .protected_journal_v2 import inspect_protected_study
    state=inspect_protected_study(reference,directory=directory,**kwargs)
    request=_compile_query_request(protocol.payload,state,query_id=query_id)
    root=_path(episode_root);expected=_path(directory)/'queries'/query_id/f"episode-{state['attempts'].get(query_id,0):04}"
    if root!=expected or root.exists():raise ValueError('Fresh exact frozen query/attempt output required')
    raw=(_canonical(request)+'\n').encode()
    return dict(schema='research/v2-protected-query-preflight',request=request,request_bytes=raw,request_sha256=hashlib.sha256(raw).hexdigest(),
        episode_root=str(root),journal_sha256=state['journal_sha256'],genesis_sha256=state['binding']['genesis_sha256'],
        execution_implemented=False,**_FALSE,
        scope='Read-only draft only. Actual coordinator must reserve the query and bind owned nonce/request/journal before any worker feature access.')
