"""Bounded feature-only native inference; no fitting or protected authority."""
import argparse
import errno
import hashlib
import io
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import zipfile

from .deployment_worker_v2 import _ROOT,_THREADS,_source,_digest,_json,_path,_bytes
from .ledger import write_new_json
from .prediction_replay_v2 import _inventory


def _predict_chunks(predict,frame,*,chunk_rows):
    import numpy as np
    if type(chunk_rows) is not int or not 1<=chunk_rows<=1024 or not len(frame):
        raise ValueError('Nonempty query and bounded positive inference chunk required')
    chunks=[];dtype=None
    for offset in range(0,len(frame),chunk_rows):
        part=frame.iloc[offset:offset+chunk_rows] if hasattr(frame,'iloc') else frame[offset:offset+chunk_rows]
        values=np.asarray(predict(part))
        if (values.dtype not in (np.dtype('float32'),np.dtype('float64')) or values.shape!=(len(part),)
                or not np.isfinite(values).all() or dtype is not None and values.dtype!=dtype):
            raise FloatingPointError('Invalid native prediction dtype, shape, finiteness or chunk consistency')
        dtype=values.dtype;chunks.append(values.copy())
    return np.concatenate(chunks),len(chunks)


def _request(path,expected):
    if path.is_symlink():raise ValueError('Query request cannot be a symlink')
    value=_json(_bytes(path,expected))
    fields={'schema','purpose','family_id','scientific_source_sha256','feature_schema','snapshot','query','chunk_rows'}
    if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-native-query-request'
            or value['purpose']!='engineering_qualification' or value['family_id'] not in tuple(f'P{i:02}' for i in range(14))
            or value['scientific_source_sha256']!=_source() or type(value['chunk_rows']) is not int or value['chunk_rows']!=1024):
        raise ValueError('Strict current-source feature-only native query request required')
    schema=value['feature_schema']
    if (type(schema) is not dict or set(schema)!={'columns','categorical'} or type(schema['columns']) is not list
            or not 1<=len(schema['columns'])<=128 or any(type(s) is not str or not s for s in schema['columns'])
            or len(set(schema['columns']))!=len(schema['columns']) or type(schema['categorical']) is not list
            or any(type(s) is not str for s in schema['categorical']) or len(set(schema['categorical']))!=len(schema['categorical'])
            or not set(schema['categorical'])<=set(schema['columns'])):raise ValueError('Exact finite native feature schema required')
    for key,fields in (('snapshot',{'path','metadata_sha256'}),('query',{'path','sha256'})):
        if type(value[key]) is not dict or set(value[key])!=fields:raise ValueError('Exact source-bound model/query references required')
    snapshot=_path(path.parent,value['snapshot']['path'])
    metadata=snapshot/('state.json' if value['family_id'] in ('P08','P09','P10','P11') else 'manifest.json')
    saved=_json(_bytes(metadata,value['snapshot']['metadata_sha256']))
    payload=saved.get('payload') if value['family_id'] in ('P08','P09','P10','P11') and type(saved) is dict else saved
    if (type(payload) is not dict or type(payload.get('config')) is not dict
            or payload['config'].get('family')!=value['family_id'] or payload['config'].get('feature_schema')!=schema):
        raise ValueError('Closed native model family/schema differs from query request')
    query=_path(path.parent,value['query']['path']);raw=_bytes(query,value['query']['sha256'])
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members=archive.infolist();names={m.filename for m in members}
            identities=['row_id','group_id']+(['timestamps'] if 'timestamps.npy' in names else [])
            if (len(members)!=1+len(identities) or names!={k+'.npy' for k in ['X',*identities]}
                    or sum(m.file_size for m in members)>512*1024**2):
                raise ValueError('Query archive must contain bounded features/identifiers only; no labels')
    except zipfile.BadZipFile as error:raise ValueError('Invalid feature-only native query archive') from error
    import numpy as np
    with np.load(io.BytesIO(raw),allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in ['X',*identities]}
    X=arrays['X'];rows=arrays['row_id'];groups=arrays['group_id']
    if (X.ndim!=2 or X.dtype!=np.float64 or not 1<=len(X)<=262144 or X.shape[1]!=len(schema['columns'])
            or not np.isfinite(X).all() or rows.dtype!=np.int64 or rows.shape!=(len(X),)
            or len(np.unique(rows))!=len(rows) or groups.dtype!=np.int64 or groups.shape!=rows.shape):
        raise ValueError('Finite float64 features and matching unique int64 row/group identities required')
    if 'timestamps' in arrays:
        dates=arrays['timestamps']
        if (dates.shape!=rows.shape or dates.dtype.kind!='M' or np.datetime_data(dates.dtype)[0] not in ('h','m','s','ms','us','ns')
                or np.isnat(dates).any() or np.any(dates[1:]<=dates[:-1])):
            raise ValueError('Actual strictly increasing native query timestamps required')
    return value,snapshot,metadata,query,arrays,identities


def run_query_worker(request_path,*,expected_request_sha256,output):
    started=time.perf_counter_ns();path=Path(request_path);output=Path(output)
    if output.exists() or output.is_symlink():raise ValueError('Fresh native query output required')
    output.mkdir(parents=True,exist_ok=False)
    record=dict(schema='research/v2-native-query-terminal',status='input_or_dependency_failure',
        request_sha256=expected_request_sha256,adapter_import_started=False,fitting_performed=False,
        labels_read=False,quality_eligible=False,protected_access_authorized=False,
        prediction_rows=0,chunks_completed=0)
    try:
        value,snapshot,metadata,query,arrays,identities=_request(path,expected_request_sha256)
        if not sys.dont_write_bytecode or any(os.environ.get(k)!='1' for k in _THREADS):
            raise ValueError('Fresh native prediction requires disabled bytecode and one numerical thread')
        snapshot_files=_inventory(snapshot)
        import numpy as np
        import pandas as pd
        from .method_worker import _factory,_initialize_native_threads,loaded_method_modules
        record.update(family_id=value['family_id'],scientific_source_sha256=value['scientific_source_sha256'],
            query_sha256=value['query']['sha256'],snapshot_metadata_sha256=value['snapshot']['metadata_sha256'],snapshot_files=snapshot_files)
        record['adapter_import_started']=True
        record['torch_threads']=_initialize_native_threads(value['family_id'])
        cls=type(_factory(value['family_id']))
        argument='expected_state_sha256' if value['family_id'] in ('P08','P09','P10','P11') else 'expected_manifest_sha256'
        model=cls.load(snapshot,**{argument:value['snapshot']['metadata_sha256']})
        frame=pd.DataFrame(arrays['X'],columns=value['feature_schema']['columns'])
        prediction,count=_predict_chunks(model.predict,frame,chunk_rows=value['chunk_rows'])
        _bytes(path,expected_request_sha256);_bytes(query,value['query']['sha256'])
        _bytes(metadata,value['snapshot']['metadata_sha256'])
        if _source()!=value['scientific_source_sha256'] or _inventory(snapshot)!=snapshot_files:
            raise ValueError('Native source or complete snapshot changed during inference')
        with (output/'predictions.npz').open('xb') as stream:
            np.savez(stream,prediction=prediction,**{key:arrays[key] for key in identities})
            stream.flush();os.fsync(stream.fileno())
        record.update(status='completed_ineligible',prediction_rows=len(prediction),chunks_completed=count,
            prediction_dtype=str(prediction.dtype),loaded_heavy_modules=loaded_method_modules(),
            prediction_sha256=hashlib.sha256((output/'predictions.npz').read_bytes()).hexdigest(),
            scope='Complete bounded original-unit closed-model inference only; no labels, fitting, quality qualification or protected authorization')
        return record
    except BaseException as error:
        record.update(status='resource_exhausted' if isinstance(error,MemoryError) or isinstance(error,OSError) and error.errno in (errno.ENOMEM,errno.ENOSPC)
            else 'interrupted_recoverable' if isinstance(error,(KeyboardInterrupt,SystemExit))
            else 'numerical_failure' if isinstance(error,FloatingPointError) else 'input_or_dependency_failure',
            error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
        raise
    finally:
        record['worker_body_ns']=time.perf_counter_ns()-started
        write_new_json(output/'terminal.json',record)


def predict_native_query(request_path,*,expected_request_sha256,output,interpreter,
        extra_pythonpath=(),rss_limit_bytes=4*1024**3):
    from .process_supervision_v2 import supervise_process
    started=time.perf_counter_ns();path=Path(request_path).resolve();output=Path(output);interpreter=Path(interpreter)
    _bytes(path,expected_request_sha256);source=_source()
    if (not interpreter.is_absolute() or not interpreter.is_file() or type(rss_limit_bytes) is not int or rss_limit_bytes<=0
            or output.exists() or output.is_symlink() or output.resolve().is_relative_to(path.parent)
            or any(output.resolve().is_relative_to(_ROOT/p) for p in ('src','tests','locks','configs','protocol'))):
        raise ValueError('Fresh external output, qualified interpreter and owned RSS guard required')
    overlays=[]
    for entry in extra_pythonpath:
        directory=Path(entry).resolve()
        if not directory.is_dir() or not directory.is_relative_to((_ROOT/'environment').resolve()):
            raise ValueError('Native query overlays must use the pinned project environment')
        overlays.append(str(directory))
    output.mkdir(parents=True,exist_ok=False);output=output.resolve()
    env={key:'1' for key in (*_THREADS,'PYTHONDONTWRITEBYTECODE')}
    env.update(PYTHONPATH=os.pathsep.join([str(_ROOT/'src'),*overlays]),KMP_DUPLICATE_LIB_OK=None,
        PYTORCH_ENABLE_MPS_FALLBACK='0',MPLCONFIGDIR=str(output/'cache/mpl'),XDG_CACHE_HOME=str(output/'cache/xdg'))
    process=supervise_process(argv=[str(interpreter),'-B','-m','treeneuron.research.native_query_v2',
        '--request',str(path),'--expected-request-sha256',expected_request_sha256,'--output',str(output/'worker')],
        cwd=_ROOT,env_overrides=env,output=output/'process',timeout_ns=None,rss_limit_bytes=rss_limit_bytes,
        identity=dict(purpose='feature_only_native_query',source_sha256=source,request_sha256=expected_request_sha256))
    status=process['execution_status'];worker=None
    if status=='exited':
        if process['returncode'] in (-signal.SIGINT,-signal.SIGTERM,-signal.SIGKILL):status='interrupted_recoverable'
        else:
            worker=_json((output/'worker/terminal.json').read_bytes());status=worker['status']
            if (worker.get('schema')!='research/v2-native-query-terminal' or worker.get('request_sha256')!=expected_request_sha256
                    or any(worker.get(k) is not False for k in ('fitting_performed','labels_read','quality_eligible','protected_access_authorized'))
                    or (process['returncode']==0)!=(status=='completed_ineligible')):
                raise ValueError('Native query managed process and closed worker disagree')
            if status=='completed_ineligible':
                prediction=output/'worker/predictions.npz'
                if hashlib.sha256(prediction.read_bytes()).hexdigest()!=worker['prediction_sha256']:
                    raise ValueError('Published query prediction bytes changed')
                value=_json(_bytes(path,expected_request_sha256))
                snapshot=_path(path.parent,value['snapshot']['path'])
                metadata=snapshot/('state.json' if value['family_id'] in ('P08','P09','P10','P11') else 'manifest.json')
                query=_path(path.parent,value['query']['path'])
                _bytes(query,value['query']['sha256']);_bytes(metadata,value['snapshot']['metadata_sha256'])
                if (worker.get('scientific_source_sha256')!=source or worker.get('family_id')!=value['family_id']
                        or worker.get('query_sha256')!=value['query']['sha256']
                        or worker.get('snapshot_metadata_sha256')!=value['snapshot']['metadata_sha256']
                        or _inventory(snapshot)!=worker.get('snapshot_files')):
                    raise ValueError('Native prediction source or complete closed model changed after worker exit')
    _bytes(path,expected_request_sha256)
    if _source()!=source:raise ValueError('Scientific source changed during native query inference')
    result=dict(schema='research/v2-native-query-result',status=status,request_sha256=expected_request_sha256,
        scientific_source_sha256=source,child_reaped=process['child_reaped'],child_started=process['child_started'],
        process_receipt_sha256=hashlib.sha256((output/'process/process.json').read_bytes()).hexdigest(),
        prediction_rows=worker['prediction_rows'] if worker else 0,chunks_completed=worker['chunks_completed'] if worker else 0,
        labels_read=False,quality_eligible=False,protected_access_authorized=False,
        elapsed_ns=time.perf_counter_ns()-started,
        clock_scope='Helper entry through result preparation; process time nested; caller pays final publication/return')
    write_new_json(output/'result.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--expected-request-sha256',required=True);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    try:run_query_worker(args.request,expected_request_sha256=args.expected_request_sha256,output=args.output)
    except BaseException as error:
        print(f'native-query: {type(error).__name__}: {error}',file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
