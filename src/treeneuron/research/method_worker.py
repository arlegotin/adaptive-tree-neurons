"""Fresh engineering-only method worker with F/G and feature-only query inputs.

This entry point cannot grant quality eligibility or protected access. Its
supervisor must measure process startup and enforce the outer deadline/RSS limit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

import numpy as np


_ROOT = Path(__file__).resolve().parents[3]
_THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
_ADAPTERS = {
    'P00': ('gated_native','GatedPredictor',True),
    'P01': ('gated_native','GatedPredictor',True),
    'P02': ('gated_native','GatedPredictor',True),
    'P05': ('gated_native','GatedPredictor',True),
    'P03': ('ordinary_neural','OrdinaryNeuralPredictor',True),
    'P04': ('ordinary_neural','OrdinaryNeuralPredictor',True),
    'P06': ('realmlp_native','RealMLPPredictor',False),
    'P07': ('tabm_native','TabMPredictor',False),
    'P08': ('native_trees','NativeTreePredictor',True),
    'P09': ('native_trees','NativeTreePredictor',True),
    'P10': ('native_trees','NativeTreePredictor',True),
    'P11': ('native_trees','NativeTreePredictor',True),
    'P12': ('node_v2','NodeV2Predictor',False),
    'P13': ('p13_simple','P13Predictor',False),
}


def scientific_source_sha256():
    """Same byte algorithm as train.source_hash, without importing Torch."""
    root = _ROOT/'src/treeneuron'
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def loaded_method_modules():
    """Presence of tracked namespaces, not their measured import cost."""
    names=('torch','treeneuron.baselines._node_vendor','pytabkit','tabm',
           'rtdl_num_embeddings','catboost','xgboost','lightgbm')
    return [name for name in names if any(k==name or k.startswith(name+'.') for k in sys.modules)]


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _digest(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _read_json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate worker request field')
            result[key]=value
        return result
    return json.loads(path.read_text(),object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite worker request')))


def _write(path,value):
    text=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    with path.open('x') as stream:
        stream.write(text);stream.flush();os.fsync(stream.fileno())


def _identifiers(values,n):
    if values.ndim!=1 or len(values)!=n or values.dtype.kind not in 'iuU':
        raise ValueError('Worker row/group identifiers must be integer or Unicode vectors')
    result=values.tolist()
    if any(type(v) is str and not v for v in result):raise ValueError('Empty worker row/group identifier')
    return set((type(v).__name__,v) for v in result)


def _load_inputs(path,value):
    columns=value['config'].get('feature_schema',{}).get('columns')
    categories=value['config'].get('feature_schema',{}).get('categorical')
    if (type(columns) is not list or not 1<=len(columns)<=128
            or any(type(s) is not str or not s for s in columns) or len(set(columns))!=len(columns)
            or type(categories) is not list or any(type(s) is not str for s in categories)
            or len(set(categories))!=len(categories) or not set(categories)<=set(columns)):
        raise ValueError('Worker requires an explicit finite semantic feature schema')
    temporal=value['observation_identity']['split_kind']=='temporal'
    result={};row_sets=[];group_sets=[]
    for role in ('F','G','V_features'):
        ref=value['files'][role]
        if type(ref) is not dict or set(ref)!={'path','sha256'} or type(ref['path']) is not str or not _digest(ref['sha256']):
            raise ValueError('Invalid worker input commitment')
        relative=Path(ref['path']);file=path.parent/relative
        if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                or file.is_symlink() or not file.resolve().is_relative_to(path.parent.resolve())
                or not file.is_file() or _sha(file)!=ref['sha256']):
            raise ValueError('Worker input changed or escaped its committed input directory')
        expected={'X','row_id','group_id'}|({'y'} if role!='V_features' else set())|({'timestamps'} if temporal else set())
        with zipfile.ZipFile(file) as zipped:
            members=zipped.infolist()
            if (len(members)!=len(expected) or {m.filename for m in members}!={k+'.npy' for k in expected}
                    or sum(m.file_size for m in members)>64*1024**2):
                raise ValueError('Worker role archive has unexpected/forbidden fields or allocation')
        with np.load(file,allow_pickle=False) as saved:
            if len(saved.files)!=len(expected) or set(saved.files)!=expected:
                raise ValueError('Worker query must contain features only; no forbidden labels')
            arrays={k:saved[k].copy() for k in saved.files}
        x=arrays['X'];n=len(x)
        if (x.ndim!=2 or x.shape[1]!=len(columns) or not (2 if role=='F' else 1)<=n<=20000
                or x.dtype.kind not in 'iuf' or not np.isfinite(x).all()):
            raise ValueError('Worker feature shape/dtype/finiteness is outside its finite engineering envelope')
        row_ids=_identifiers(arrays['row_id'],n);group_ids=_identifiers(arrays['group_id'],n)
        if len(row_ids)!=n:raise ValueError('Worker observation row IDs are duplicated')
        if any(row_ids&prior for prior in row_sets):raise ValueError('Worker roles overlap observation rows')
        if any(group_ids&prior for prior in group_sets):raise ValueError('Worker roles overlap split groups/days')
        row_sets.append(row_ids);group_sets.append(group_ids)
        if role!='V_features':
            y=arrays['y']
            if y.shape!=(n,) or y.dtype.kind not in 'iuf' or not np.isfinite(y).all():
                raise ValueError('Worker F/G targets must be finite real vectors')
        if temporal:
            dates=arrays['timestamps']
            if (dates.shape!=(n,) or dates.dtype.kind!='M' or np.isnat(dates).any()
                    or np.any(dates[1:]<=dates[:-1])):
                raise ValueError('Worker temporal rows must have actual increasing timestamps')
            if result:
                previous=result['F' if role=='G' else 'G']['timestamps'][-1]
                if dates[0]-previous<np.timedelta64(24,'h'):
                    raise ValueError('Worker temporal roles violate chronology/24-clock-hour embargo')
        result[role]=arrays
    return result


def _validate_request(path,expected):
    if (path.is_symlink() or not path.is_file() or not _digest(expected) or _sha(path)!=expected):
        raise ValueError('Worker request differs from external trusted digest')
    value=_read_json(path)
    fields={'schema','purpose','job_id','family_id','fit_seed','scientific_source_sha256',
            'config','budget','observation_identity','files'}
    if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-method-worker-request'
            or value['purpose']!='engineering_qualification' or type(value['job_id']) is not str
            or not value['job_id'] or len(value['job_id'])>256 or value['family_id'] not in _ADAPTERS
            or type(value['fit_seed']) is not int or not 0<=value['fit_seed']<2**32
            or not _digest(value['scientific_source_sha256'])
            or value['scientific_source_sha256']!=scientific_source_sha256()
            or type(value['config']) is not dict or value['config'].get('family')!=value['family_id']
            or type(value['files']) is not dict or set(value['files'])!={'F','G','V_features'}):
        raise ValueError('Invalid/unsupported engineering worker request or source identity')
    budget=value['budget'];identity=value['observation_identity']
    if (type(budget) is not dict or set(budget)!={'max_seconds'}
            or type(budget['max_seconds']) not in (int,float)
            or not math.isfinite(budget['max_seconds']) or not 0<budget['max_seconds']<=7200
            or type(identity) is not dict or set(identity)!={'dataset','split_kind','split_sha256'}
            or type(identity['dataset']) is not str or not identity['dataset']
            or identity['split_kind'] not in ('grouped','temporal') or not _digest(identity['split_sha256'])):
        raise ValueError('Worker budget or declared split identity is invalid')
    return value,_load_inputs(path,value)


def _factory(family):
    module,name,with_family=_ADAPTERS[family]
    cls=getattr(importlib.import_module('treeneuron.research.'+module),name)
    return cls(family) if with_family else cls()


def _initialize_native_threads(family):
    """Own the fresh neural process's Torch policy; keep other methods lazy.

    OMP/BLAS environment variables do not set Torch's inter-op pool. Native
    adapters retain their caller-policy guards instead of changing caller
    globals; the containing fresh worker sets this treatment before fitting.
    """
    if family not in ('P00','P01','P02','P03','P04','P05','P06','P07','P12'):
        return None
    torch=importlib.import_module('torch')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    threads=dict(intra_op=torch.get_num_threads(),inter_op=torch.get_num_interop_threads())
    if threads!=dict(intra_op=1,inter_op=1):raise ValueError('Fresh worker Torch thread initialization failed')
    return threads


def run_engineering_trial(request_path,*,expected_request_sha256,output):
    """Retain a fresh engineering attempt; no retry or eligibility promotion."""
    started=time.perf_counter_ns();root=Path(output);request_path=Path(request_path)
    if root.exists() or root.is_symlink():raise ValueError('Fresh worker output required; no overwrite/retry')
    root.mkdir(parents=True)
    record=dict(schema='research/v2-method-worker-terminal',status='input_or_dependency_failure',
        purpose='engineering_qualification',request_path=str(request_path.resolve()),
        expected_request_sha256=expected_request_sha256,adapter_import_started=False,
        checkpoint_eligible=False,quality_eligible=False,protected_access_authorized=False,
        T_accessed=False,started_utc_ns=time.time_ns(),files=[],
        startup_environment={k:os.environ.get(k) for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE','KMP_DUPLICATE_LIB_OK')})
    model=None
    try:
        value,arrays=_validate_request(request_path,expected_request_sha256)
        if not sys.dont_write_bytecode or any(os.environ.get(k)!='1' for k in _THREADS):
            raise ValueError('Fresh worker requires disabled bytecode and declared single numerical threads')
        record.update(job_id=value['job_id'],family_id=value['family_id'],fit_seed=value['fit_seed'],
            scientific_source_sha256=value['scientific_source_sha256'],
            observation_identity=value['observation_identity'],accessed_roles=['F_labels','G_labels','V_features'])
        # Inputs and source have passed before any adapter import or constructor.
        record['adapter_import_started']=True
        record['torch_threads']=_initialize_native_threads(value['family_id'])
        model=_factory(value['family_id'])
        from treeneuron.baselines.api import FitRequest
        import pandas as pd
        schema=value['config']['feature_schema'];columns=schema['columns']
        frames={role:pd.DataFrame(a['X'],columns=columns) for role,a in arrays.items()}
        model.fit(FitRequest(frames['F'],arrays['F']['y'],frames['G'],arrays['G']['y'],
            value['fit_seed'],value['config'],value['budget'],root/'fit'))
        query=arrays['V_features'];prediction=np.asarray(model.predict(frames['V_features']))
        if (prediction.shape!=(len(query['X']),) or prediction.dtype.kind not in 'iuf'
                or not np.isfinite(prediction).all()):
            raise FloatingPointError('Worker produced invalid original-unit query predictions')
        model.save(root/'snapshot')
        # Native tree adapters publish state.json and require its external hash.
        # Keep their existing closed format rather than fabricating a manifest.
        snapshot_name,hash_argument=(('state.json','expected_state_sha256')
            if value['family_id'] in ('P08','P09','P10','P11') else
            ('manifest.json','expected_manifest_sha256'))
        snapshot_digest=_sha(root/'snapshot'/snapshot_name)
        restored=type(model).load(root/'snapshot',**{hash_argument:snapshot_digest})
        if not np.array_equal(prediction,np.asarray(restored.predict(frames['V_features']))):
            raise ValueError('Worker closed snapshot reload changed its query predictions')
        with (root/'predictions.npz').open('xb') as stream:
            np.savez(stream,prediction=prediction,row_id=query['row_id'],group_id=query['group_id'],
                **({'timestamps':query['timestamps']} if 'timestamps' in query else {}))
            stream.flush();os.fsync(stream.fileno())
        if scientific_source_sha256()!=value['scientific_source_sha256']:
            raise ValueError('Worker source changed during execution')
        if _sha(request_path)!=expected_request_sha256:
            raise ValueError('Worker request changed during execution')
        for ref in value['files'].values():
            file=request_path.parent/ref['path']
            if (file.is_symlink() or not file.resolve().is_relative_to(request_path.parent.resolve())
                    or not file.is_file() or _sha(file)!=ref['sha256']):
                raise ValueError('Worker committed input changed during execution')
        record.update(status='completed_ineligible',
            snapshot_commitment=dict(path='snapshot/'+snapshot_name,sha256=snapshot_digest),
            snapshot_reload_exact=True,adapter_resources=model.resource_summary(),
            loaded_heavy_modules=loaded_method_modules(),
            hard_limits_enforced='Requires containing supervisor; this worker records its body only',
            prediction_rows=len(prediction),eligible_scope='Engineering result only; native workload/GateD qualification pending')
        record['snapshot_state_sha256' if snapshot_name=='state.json' else 'snapshot_manifest_sha256']=snapshot_digest
        if time.perf_counter_ns()-started>value['budget']['max_seconds']*1e9:
            record.update(status='timeout',reason='Worker body exceeded requested budget; outputs retained ineligible')
    except BaseException as error:
        status='interrupted_recoverable' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'timeout' if isinstance(error,TimeoutError) else (
            'resource_exhausted' if isinstance(error,MemoryError) or isinstance(error,OSError) and error.errno in (12,28) else
            'numerical_failure' if isinstance(error,ArithmeticError) else 'input_or_dependency_failure')
        record.update(status=status,error_type=type(error).__module__+'.'+type(error).__qualname__,
            reason=str(error),traceback=traceback.format_exc())
        if model is not None:record['adapter_resources']=model.resource_summary()
        raise
    finally:
        record['worker_body_ns']=time.perf_counter_ns()-started
        record['timer_scope']='After module imports through retained artifact hashing; process startup/final publication require outer clock'
        for p in sorted(root.rglob('*')):
            if p.is_symlink():raise ValueError('Worker artifact symlink is outside its closed publication contract')
            if p.is_file():record['files'].append(dict(path=p.relative_to(root).as_posix(),sha256=_sha(p),bytes=p.stat().st_size))
        record['worker_body_ns']=time.perf_counter_ns()-started
        _write(root/'terminal.json',record)
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--expected-request-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        result=run_engineering_trial(args.request,expected_request_sha256=args.expected_request_sha256,output=args.output)
    except Exception as error:
        print(f'method-worker: {type(error).__name__}: {error}',file=sys.stderr);return 2
    print(json.dumps(dict(status=result['status'],quality_eligible=False),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
