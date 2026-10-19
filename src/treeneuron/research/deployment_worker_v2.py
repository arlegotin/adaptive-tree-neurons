"""Fresh CPU deployment measurements; standard-library imports until validation.

This engineering worker grants no quality, protected-access or timing claim.
The outer supervisor owns complete interpreter/process wall time and RSS limits.
"""
import argparse
import errno
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import zipfile


_ROOT=Path(__file__).resolve().parents[3]
_THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def _source():
    digest=hashlib.sha256()
    for path in sorted((_ROOT/'src/treeneuron').rglob('*.py')):
        digest.update(path.relative_to(_ROOT/'src/treeneuron').as_posix().encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def _digest(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _json(content):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate deployment field')
            result[key]=value
        return result
    return json.loads(content,object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite deployment metadata')))


def _path(root,relative):
    if type(relative) is not str or not relative:raise ValueError('Relative deployment path required')
    parts=Path(relative)
    if parts.is_absolute() or '..' in parts.parts:raise ValueError('Deployment path escapes request root')
    path=root
    for part in parts.parts:
        path=path/part
        if path.is_symlink():raise ValueError('Deployment paths cannot use symlinks')
    return path


def _bytes(path,digest):
    if not _digest(digest) or not path.is_file():raise ValueError('Missing external deployment file commitment')
    content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('Deployment input/model commitment differs')
    return content


def _request(path,expected):
    if path.is_symlink():raise ValueError('Deployment request cannot be a symlink')
    content=_bytes(path,expected);value=_json(content)
    keys={'schema','purpose','family_id','scientific_source_sha256','feature_schema','snapshot','query',
        'batch_sizes','warmup_calls','samples_per_batch'}
    if (type(value) is not dict or set(value)!=keys or value['schema']!='research/v2-deployment-worker-request'
            or value['purpose']!='engineering_qualification' or value['family_id'] not in tuple(f'P{i:02}' for i in range(14))
            or value['scientific_source_sha256']!=_source() or type(value['batch_sizes']) is not list
            or value['batch_sizes']!=[1,1024] or any(type(n) is not int for n in value['batch_sizes'])
            or type(value['warmup_calls']) is not int or not 0<=value['warmup_calls']<=16
            or type(value['samples_per_batch']) is not int or not 1<=value['samples_per_batch']<=128):
        raise ValueError('Strict source-bound engineering deployment request required')
    schema=value['feature_schema']
    if (type(schema) is not dict or set(schema)!={'columns','categorical'}
            or type(schema['columns']) is not list or not 1<=len(schema['columns'])<=128
            or type(schema['categorical']) is not list
            or any(type(s) is not str or not s for s in schema['columns']+schema['categorical'])
            or len(set(schema['columns']))!=len(schema['columns'])
            or len(set(schema['categorical']))!=len(schema['categorical'])
            or not set(schema['categorical'])<=set(schema['columns'])):
        raise ValueError('Explicit native deployment feature schema required')
    snapshot=value['snapshot'];query=value['query']
    if (type(snapshot) is not dict or set(snapshot)!={'path','metadata_sha256'}
            or type(query) is not dict or set(query)!={'path','sha256'}):raise ValueError('Closed snapshot/query references required')
    directory=_path(path.parent,snapshot['path']);tree=value['family_id'] in ('P08','P09','P10','P11')
    metadata=directory/('state.json' if tree else 'manifest.json')
    if metadata.is_symlink():raise ValueError('Snapshot metadata cannot be a symlink')
    saved=_json(_bytes(metadata,snapshot['metadata_sha256']));payload=saved.get('payload') if tree else saved
    if type(payload) is not dict or type(payload.get('config')) is not dict:
        raise ValueError('Missing native model recipe')
    config=payload['config']
    if config.get('family')!=value['family_id'] or config.get('feature_schema')!=schema:
        raise ValueError('Deployment family/schema differs from the retained model')
    query_path=_path(path.parent,query['path']);raw=_bytes(query_path,query['sha256'])
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members=archive.infolist()
            if (len(members)!=2 or {m.filename for m in members}!={'X.npy','row_id.npy'}
                    or sum(m.file_size for m in members)>64*1024**2):
                raise ValueError('Deployment query must contain bounded features/IDs only')
    except zipfile.BadZipFile as error:raise ValueError('Invalid deployment query archive') from error
    if not sys.dont_write_bytecode or any(os.environ.get(k)!='1' for k in _THREADS):
        raise ValueError('Fresh deployment requires disabled bytecode and one numerical thread')
    return value,directory,metadata,query_path,raw


def run_deployment(request_path,*,expected_request_sha256,output):
    """Measure one closed CPU model, retaining all declared prediction samples."""
    started=time.perf_counter_ns();path=Path(request_path);root=Path(output)
    if root.exists() or root.is_symlink():raise ValueError('Fresh deployment output required')
    root.mkdir(parents=True)
    record=dict(schema='research/v2-deployment-worker-terminal',status='input_or_dependency_failure',
        adapter_import_started=False,cost_claim_authorized=False,protected_access_authorized=False,T_accessed=False,
        whole_fresh_process_wall_authenticated=False,strict_peak_memory_limit_qualified=False,
        request_sha256=expected_request_sha256,started_utc_ns=time.time_ns(),batches=[],
        numerical_threads={k:os.environ.get(k) for k in _THREADS},cpu=platform.processor(),machine=platform.machine(),
        startup_environment={k:os.environ.get(k) for k in ('KMP_DUPLICATE_LIB_OK','PYTHONDONTWRITEBYTECODE')},
        primary_workload='resident_batch_1_CPU_prediction_including_native_preprocessing',
        clock_scope='Function entry through measurement/input validation; outer supervisor owns complete fresh process and final publication',
        prediction_includes_native_preprocessing=True)
    try:
        value,directory,metadata,query_path,raw=_request(path,expected_request_sha256)
        begin=time.perf_counter_ns()
        import numpy as np
        import pandas as pd
        record['common_import_ns']=time.perf_counter_ns()-begin
        begin=time.perf_counter_ns()
        with np.load(io.BytesIO(raw),allow_pickle=False) as saved:X=saved['X'].copy();rows=saved['row_id'].copy()
        if (X.ndim!=2 or X.dtype!=np.float32 or not 1024<=len(X)<=20000
                or X.shape[1]!=len(value['feature_schema']['columns']) or not np.isfinite(X).all()
                or rows.dtype!=np.int64 or rows.shape!=(len(X),) or len(np.unique(rows))!=len(rows)):
            raise ValueError('Finite float32 features and unique int64 row IDs in the measured envelope required')
        frame=pd.DataFrame(X,columns=value['feature_schema']['columns'])
        record['feature_decode_and_frame_ns']=time.perf_counter_ns()-begin
        record.update(family_id=value['family_id'],scientific_source_sha256=value['scientific_source_sha256'],
            query_rows=len(X),input_dtype=str(X.dtype),query_sha256=value['query']['sha256'],snapshot_sha256=value['snapshot']['metadata_sha256'])
        begin=time.perf_counter_ns();record['adapter_import_started']=True
        from .method_worker import _ADAPTERS,loaded_method_modules,_initialize_native_threads
        record['torch_threads']=_initialize_native_threads(value['family_id'])
        module,name,_=_ADAPTERS[value['family_id']]
        cls=getattr(importlib.import_module('treeneuron.research.'+module),name)
        record['adapter_import_ns']=time.perf_counter_ns()-begin
        argument='expected_state_sha256' if value['family_id'] in ('P08','P09','P10','P11') else 'expected_manifest_sha256'
        begin=time.perf_counter_ns();model=cls.load(directory,**{argument:value['snapshot']['metadata_sha256']})
        record['model_load_ns']=time.perf_counter_ns()-begin
        for size in value['batch_sizes']:
            query=frame.iloc[:size]
            def predict():
                begin=time.perf_counter_ns();result=model.predict(query,batch_size=size);elapsed=time.perf_counter_ns()-begin
                result=np.asarray(result)
                if result.shape!=(size,) or result.dtype.kind not in 'fiu' or not np.isfinite(result).all():
                    raise FloatingPointError('Invalid original-unit deployment prediction')
                return elapsed,result
            first,golden=predict();golden=golden.copy()
            def sample():
                elapsed,result=predict()
                if result.dtype!=golden.dtype or not np.array_equal(result,golden):
                    raise FloatingPointError('Resident deployment prediction changed for identical rows')
                return elapsed
            warmup=[sample() for _ in range(value['warmup_calls'])]
            resident=[sample() for _ in range(value['samples_per_batch'])]
            record['batches'].append(dict(batch_size=size,first_prediction_ns=first,warmup_prediction_ns=warmup,
                resident_prediction_ns=resident,prediction_dtype=str(golden.dtype),predictions_equal_exact=True,
                prediction_values_sha256=hashlib.sha256(golden.tobytes()).hexdigest(),
                row_ids_sha256=hashlib.sha256(rows[:size].tobytes()).hexdigest()))
        _bytes(path,expected_request_sha256);_bytes(metadata,value['snapshot']['metadata_sha256']);_bytes(query_path,value['query']['sha256'])
        if _source()!=value['scientific_source_sha256']:raise ValueError('Deployment scientific source changed')
        record.update(status='completed_ineligible',loaded_heavy_modules=loaded_method_modules(),
            native_family_deployment_qualified=False,paired_30_block_protocol_authenticated=False)
        return record
    except BaseException as error:
        exhausted=isinstance(error,MemoryError) or isinstance(error,OSError) and error.errno in (errno.ENOMEM,errno.ENOSPC)
        record.update(status='resource_exhausted' if exhausted else 'numerical_failure' if isinstance(error,FloatingPointError)
            else 'interrupted_recoverable' if isinstance(error,KeyboardInterrupt) else 'input_or_dependency_failure',
            error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        record['inner_elapsed_ns']=time.perf_counter_ns()-started
        record['process_lifetime_peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
        with (root/'terminal.json').open('x') as stream:
            json.dump(record,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--expected-request-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:run_deployment(args.request,expected_request_sha256=args.expected_request_sha256,output=args.output)
    except Exception as error:
        print(f'deployment/v2: {type(error).__name__}: {error}',file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
