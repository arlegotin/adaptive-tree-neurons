"""Fresh label-free replay of closed native models and committed predictions.

Exact replay authenticates these artifacts, not their training/protocol history.
The containing supervisor owns process wall time, RSS, deadlines and reaping.
"""
import argparse
import errno
import hashlib
import importlib
import io
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

from .deployment_worker_v2 import _bytes,_json,_path,_source,_THREADS


def _inventory(root):
    if root.is_symlink() or not root.is_dir():raise ValueError('Ordinary closed snapshot directory required')
    files=[]
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('Snapshot symlinks are forbidden')
        if path.is_dir():continue
        if not path.is_file():raise ValueError('Nonregular snapshot artifact')
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        files.append(dict(path=path.relative_to(root).as_posix(),sha256=digest,bytes=path.stat().st_size))
    return files


def _archive(raw,fields):
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members=archive.infolist()
            if (len(members)!=len(fields) or {m.filename for m in members}!={k+'.npy' for k in fields}
                    or sum(m.file_size for m in members)>64*1024**2):
                raise ValueError('Exact bounded feature-only query/prediction archive required')
    except zipfile.BadZipFile as error:raise ValueError('Invalid closed replay archive') from error


def _request(path,expected):
    if path.is_symlink():raise ValueError('Replay request cannot be a symlink')
    value=_json(_bytes(path,expected))
    keys={'schema','purpose','family_id','scientific_source_sha256','feature_schema','snapshot','query','prediction'}
    if (type(value) is not dict or set(value)!=keys or value['schema']!='research/v2-prediction-replay-request'
            or value['purpose']!='engineering_qualification' or type(value['family_id']) is not str
            or value['family_id'] not in tuple(f'P{i:02}' for i in range(14))
            or value['scientific_source_sha256']!=_source()):
        raise ValueError('Strict current-source engineering replay request required')
    schema=value['feature_schema']
    if (type(schema) is not dict or set(schema)!={'columns','categorical'}
            or type(schema['columns']) is not list or not 1<=len(schema['columns'])<=128
            or type(schema['categorical']) is not list
            or any(type(s) is not str or not s for s in schema['columns']+schema['categorical'])
            or len(set(schema['columns']))!=len(schema['columns'])
            or len(set(schema['categorical']))!=len(schema['categorical'])
            or not set(schema['categorical'])<=set(schema['columns'])):
        raise ValueError('Explicit finite semantic replay schema required')
    for key in ('snapshot','query','prediction'):
        fields={'path','metadata_sha256' if key=='snapshot' else 'sha256'}
        if type(value[key]) is not dict or set(value[key])!=fields:raise ValueError('Exact closed replay references required')
    snapshot=_path(path.parent,value['snapshot']['path'])
    metadata=snapshot/('state.json' if value['family_id'] in ('P08','P09','P10','P11') else 'manifest.json')
    if metadata.is_symlink():raise ValueError('Snapshot metadata cannot be a symlink')
    saved=_json(_bytes(metadata,value['snapshot']['metadata_sha256']))
    payload=saved.get('payload') if value['family_id'] in ('P08','P09','P10','P11') and type(saved) is dict else saved
    if (type(payload) is not dict or type(payload.get('config')) is not dict
            or payload['config'].get('family')!=value['family_id']
            or payload['config'].get('feature_schema')!=schema):
        raise ValueError('Replay model family/schema differs from its native metadata')
    query=_path(path.parent,value['query']['path']);prediction=_path(path.parent,value['prediction']['path'])
    query_raw=_bytes(query,value['query']['sha256']);prediction_raw=_bytes(prediction,value['prediction']['sha256'])
    try:
        with zipfile.ZipFile(io.BytesIO(query_raw)) as archive:temporal='timestamps.npy' in archive.namelist()
    except zipfile.BadZipFile as error:raise ValueError('Invalid replay query archive') from error
    identities=['row_id','group_id']+(['timestamps'] if temporal else [])
    _archive(query_raw,['X',*identities]);_archive(prediction_raw,['prediction',*identities])
    if not sys.dont_write_bytecode or any(os.environ.get(k)!='1' for k in _THREADS):
        raise ValueError('Fresh replay requires disabled bytecode and declared numerical threads')
    return value,snapshot,metadata,query,prediction,query_raw,prediction_raw,identities


def run_prediction_replay(request_path,*,expected_request_sha256,output):
    started=time.perf_counter_ns();path=Path(request_path);root=Path(output)
    if root.exists() or root.is_symlink():raise ValueError('Fresh replay output required; no overwrite')
    if root.resolve().is_relative_to(path.parent.resolve()):raise ValueError('Replay output cannot modify its input directory')
    root.mkdir(parents=True)
    record=dict(schema='research/v2-prediction-replay-terminal',status='input_or_dependency_failure',
        request_sha256=expected_request_sha256,adapter_import_started=False,prediction_reload_exact=False,
        labels_read=False,quality_eligible=False,model_generation_authenticated=False,
        protected_access_authorized=False,T_accessed=False,started_utc_ns=time.time_ns(),
        whole_process_wall_authenticated=False,strict_peak_memory_limit_qualified=False)
    try:
        value,snapshot,metadata,query,prediction,query_raw,prediction_raw,identities=_request(path,expected_request_sha256)
        import numpy as np
        import pandas as pd
        with np.load(io.BytesIO(query_raw),allow_pickle=False) as z:features={k:z[k].copy() for k in ['X',*identities]}
        with np.load(io.BytesIO(prediction_raw),allow_pickle=False) as z:expected={k:z[k].copy() for k in ['prediction',*identities]}
        X=features['X'];rows=features['row_id'];groups=features['group_id'];values=expected['prediction']
        if (X.ndim!=2 or X.dtype!=np.float64 or not 1<=len(X)<=20000
                or X.shape[1]!=len(value['feature_schema']['columns']) or not np.isfinite(X).all()
                or rows.dtype!=np.int64 or rows.shape!=(len(X),) or len(np.unique(rows))!=len(rows)
                or groups.dtype!=np.int64 or groups.shape!=rows.shape
                or values.dtype not in (np.dtype('float32'),np.dtype('float64'))
                or values.shape!=rows.shape or not np.isfinite(values).all()):
            raise ValueError('Finite native float64 query and aligned unique int64 IDs/predictions required')
        if 'timestamps' in identities:
            times=features['timestamps']
            if (times.shape!=rows.shape or times.dtype.kind!='M'
                    or np.datetime_data(times.dtype)[0] not in ('h','m','s','ms','us','ns')
                    or np.isnat(times).any() or np.any(times[1:]<=times[:-1])):
                raise ValueError('Increasing actual replay timestamps required')
        if any(features[k].dtype!=expected[k].dtype or not np.array_equal(features[k],expected[k]) for k in identities):
            raise ValueError('Saved predictions do not match query row/group/time identities')
        files=_inventory(snapshot)
        record.update(family_id=value['family_id'],scientific_source_sha256=value['scientific_source_sha256'],
            query_rows=len(X),query_dtype=str(X.dtype),prediction_dtype=str(values.dtype),
            query_sha256=value['query']['sha256'],prediction_sha256=value['prediction']['sha256'],
            snapshot_metadata_sha256=value['snapshot']['metadata_sha256'],snapshot_files=files)
        record['adapter_import_started']=True
        from .method_worker import _ADAPTERS,_initialize_native_threads,loaded_method_modules
        record['torch_threads']=_initialize_native_threads(value['family_id'])
        module,name,_=_ADAPTERS[value['family_id']]
        cls=getattr(importlib.import_module('treeneuron.research.'+module),name)
        argument='expected_state_sha256' if value['family_id'] in ('P08','P09','P10','P11') else 'expected_manifest_sha256'
        model=cls.load(snapshot,**{argument:value['snapshot']['metadata_sha256']})
        actual=np.asarray(model.predict(pd.DataFrame(X,columns=value['feature_schema']['columns'])))
        if actual.dtype!=values.dtype or actual.shape!=values.shape or not np.array_equal(actual,values):
            raise FloatingPointError('Closed native model does not reproduce saved prediction dtype/values exactly')
        _bytes(path,expected_request_sha256);_bytes(metadata,value['snapshot']['metadata_sha256'])
        _bytes(query,value['query']['sha256']);_bytes(prediction,value['prediction']['sha256'])
        if _source()!=value['scientific_source_sha256'] or _inventory(snapshot)!=files:
            raise ValueError('Replay source or complete snapshot changed during prediction')
        record.update(status='completed_ineligible',prediction_reload_exact=True,loaded_heavy_modules=loaded_method_modules(),
            scope='Closed artifact prediction replay only; training generation, qualification and protected protocol are not authenticated')
        return record
    except BaseException as error:
        exhausted=isinstance(error,MemoryError) or isinstance(error,OSError) and error.errno in (errno.ENOMEM,errno.ENOSPC)
        record.update(status='resource_exhausted' if exhausted else 'numerical_failure' if isinstance(error,FloatingPointError)
            else 'interrupted_recoverable' if isinstance(error,KeyboardInterrupt) else 'input_or_dependency_failure',
            error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
        raise
    finally:
        record['worker_body_ns']=time.perf_counter_ns()-started
        from .method_worker import _write
        _write(root/'terminal.json',record)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--expected-request-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:run_prediction_replay(args.request,expected_request_sha256=args.expected_request_sha256,output=args.output)
    except Exception as error:
        print(f'prediction-replay/v2: {type(error).__name__}: {error}',file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
