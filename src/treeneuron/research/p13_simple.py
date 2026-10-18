"""Versioned P13 closed-form references; F-only geometry and lazy deployment.

Tiny fixtures do not qualify transformed workloads, process limits or recovery.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from treeneuron.baselines.api import FitRequest
from treeneuron.research.native_trees import _schema, _frame, _category

_ROOT=Path(__file__).resolve().parents[3]
ALPHAS=(1e-6,1e-4,1e-2,1.,1e2,1e4,1e6)
KNOTS=(4,8,16)


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _hash(value):return hashlib.sha256(_json(value).encode()).hexdigest()
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate snapshot field')
            result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))


def _write(path,value):
    with Path(path).open('x') as stream:
        stream.write(_json(value)+'\n');stream.flush()
        import os
        os.fsync(stream.fileno())


def _config(value):
    if (type(value) is not dict or set(value)!={'schema','family','estimator','feature_schema','parameters'}
            or value['schema']!='research/v2-p13-config' or value['family']!='P13'
            or value['estimator'] not in ('mean','ridge','spline')):
        raise ValueError('Invalid P13 configuration')
    schema=_schema(value['feature_schema']);params=value['parameters'];method=value['estimator']
    fields=set() if method=='mean' else {'alpha'}|({'n_knots'} if method=='spline' else set())
    if type(params) is not dict or set(params)!=fields:raise ValueError('Invalid P13 parameters')
    if method!='mean' and (type(params['alpha']) not in (int,float) or params['alpha'] not in ALPHAS):
        raise ValueError('Alpha is outside the fixed P13 support')
    if method=='spline' and (type(params['n_knots']) is not int or params['n_knots'] not in KNOTS):
        raise ValueError('Knot count is outside the fixed P13 support')
    return dict(schema=value['schema'],family='P13',estimator=method,feature_schema=schema,
        parameters=copy.deepcopy(params))


def p13_configurations(feature_schema):
    schema=_schema(feature_schema)
    def row(method,parameters):
        return dict(schema='research/v2-p13-config',family='P13',estimator=method,
            feature_schema=copy.deepcopy(schema),parameters=parameters)
    return [row('mean',{})]+[row('ridge',dict(alpha=a)) for a in ALPHAS]+[
        row('spline',dict(alpha=a,n_knots=k)) for k in KNOTS for a in ALPHAS]


def _source():
    files={str(p.relative_to(_ROOT)):_sha(p) for p in sorted((_ROOT/'src/treeneuron').rglob('*.py'))}
    for name in ('uv.lock','locks/research-v2/core.json'):files[name]=_sha(_ROOT/name)
    return dict(sha256=_hash(files),files=files)


def _runtime():
    core=_read(_ROOT/'locks/research-v2/core.json')
    pinned={r['name']:r for r in core['distributions']}
    packages={}
    for name in ('numpy','pandas'):
        distribution=importlib.metadata.distribution(name)
        if distribution.version!=pinned[name]['version']:raise ValueError('Numerical package version differs from core pin')
        records=[item for item in distribution.files if str(item).endswith('.dist-info/RECORD')]
        if len(records)!=1 or _sha(distribution.locate_file(records[0]))!=pinned[name]['record_sha256']:
            raise ValueError('Numerical package RECORD differs from core pin')
        files={}
        for item in distribution.files:
            relative=str(item)
            if relative==name+'/__init__.py' or relative.endswith(('.py','.so','.dylib')):
                path=Path(distribution.locate_file(item));digest=_sha(path)
                import base64
                encoded=base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')
                if item.hash is None or item.hash.mode!='sha256' or item.hash.value!=encoded:
                    raise ValueError('Installed numerical source/library differs from RECORD')
                files[relative]=digest
        module=importlib.import_module(name)
        if Path(module.__file__).resolve()!=Path(distribution.locate_file(name+'/__init__.py')).resolve():
            raise ValueError('Numerical module origin differs from installed distribution')
        packages[name]=dict(version=distribution.version,files=files)
    return dict(python=platform.python_version(),platform=sys.platform,machine=platform.machine(),
        packages=packages,dtype='float64')


def _missing(value):
    return value is None or value is pd.NA or isinstance(value,(float,np.floating)) and np.isnan(value)


def _numeric(values):
    result=[]
    for value in values:
        if _missing(value):result.append(np.nan);continue
        if isinstance(value,np.generic):value=value.item()
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('Numeric features require finite real values or missing values')
        result.append(value)
    return np.asarray(result,dtype=np.float64)


def _token(value):return None if _missing(value) else _category(value)


def _finite(value):return type(value) in (int,float) and math.isfinite(value)


def _preprocessing(X,y,schema):
    frame=_frame(X,schema)
    if not len(frame) or len(frame)!=len(y):raise ValueError('F feature/target rows differ or are empty')
    columns=[]
    for name in schema['columns']:
        if name in schema['categorical']:
            tokens=[_token(x) for x in frame[name]];known=sorted(set(t for t in tokens if t is not None))
            columns.append(dict(name=name,kind='categorical',categories=known,unknown_index=len(known)))
        else:
            values=_numeric(frame[name]);missing=np.isnan(values)
            fill=0. if missing.all() else float(np.median(values[~missing]))
            values=np.where(missing,fill,values);mean=float(values.mean());scale=float(values.std(ddof=0))
            columns.append(dict(name=name,kind='numeric',fill=fill,mean=mean,scale=scale or 1.,
                indicator=bool(missing.any()),constant=scale==0.,all_missing=bool(missing.all())))
    scale=float(y.std(ddof=0))
    state=dict(columns=columns,target_mean=float(y.mean()),target_scale=scale or 1.,target_constant=scale==0.)
    _validate_preprocessing(state,schema)
    return state


def _valid_token(token):
    if type(token) is not str or ':' not in token:return False
    kind,value=token.split(':',1)
    if kind=='str':return True
    if kind=='bool':return value in ('True','False')
    try:
        if kind=='int':return str(int(value))==value
        if kind=='float':return math.isfinite(float.fromhex(value)) and float.fromhex(value).hex()==value
    except (ValueError,OverflowError):pass
    return False


def _validate_preprocessing(state,schema):
    if (type(state) is not dict or set(state)!={'columns','target_mean','target_scale','target_constant'}
            or not _finite(state['target_mean']) or not _finite(state['target_scale']) or state['target_scale']<=0
            or type(state['target_constant']) is not bool
            or state['target_constant'] and state['target_scale']!=1.
            or type(state['columns']) is not list or len(state['columns'])!=len(schema['columns'])):
        raise ValueError('Invalid P13 preprocessing')
    for name,col in zip(schema['columns'],state['columns']):
        categorical=name in schema['categorical']
        fields={'name','kind','categories','unknown_index'} if categorical else {
            'name','kind','fill','mean','scale','indicator','constant','all_missing'}
        if type(col) is not dict or set(col)!=fields or col['name']!=name or col['kind']!=('categorical' if categorical else 'numeric'):
            raise ValueError('Preprocessing column schema differs')
        if categorical:
            cats=col['categories']
            if (type(cats) is not list or any(not _valid_token(c) for c in cats) or cats!=sorted(set(cats))
                    or type(col['unknown_index']) is not int or col['unknown_index']!=len(cats)):
                raise ValueError('Invalid typed F category vocabulary')
        elif (any(not _finite(col[k]) for k in ('fill','mean','scale')) or col['scale']<=0
                or any(type(col[k]) is not bool for k in ('indicator','constant','all_missing'))
                or col['constant'] and col['scale']!=1.
                or col['all_missing'] and not (col['indicator'] and col['constant'] and col['fill']==col['mean']==0.)):
            raise ValueError('Invalid F numeric statistics')


def _features(X,schema,state):
    frame=_frame(X,schema);numeric=[];indicators=[];categorical=[]
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        for col in state['columns']:
            if col['kind']=='numeric':
                values=_numeric(frame[col['name']]);missing=np.isnan(values)
                numeric.append((np.where(missing,col['fill'],values)-col['mean'])/col['scale'])
                if col['indicator']:indicators.append(missing.astype(np.float64))
            else:
                codes={t:i for i,t in enumerate(col['categories'])}
                indices=np.asarray([codes.get(_token(t),col['unknown_index']) for t in frame[col['name']]],dtype=np.int64)
                one_hot=np.zeros((len(frame),len(codes)+1),dtype=np.float64)
                one_hot[np.arange(len(frame)),indices]=1.
                categorical.extend(one_hot.T)
        columns=numeric+indicators+categorical
        values=np.column_stack(columns) if columns else np.empty((len(frame),0),dtype=np.float64)
    if not np.isfinite(values).all():raise FloatingPointError('Nonfinite encoded features')
    return values,len(numeric)


def _basis(features,count,kind,knots):
    if kind=='mean':return np.ones((len(features),1),dtype=np.float64)
    pieces=[np.ones((len(features),1),dtype=np.float64)]
    with np.errstate(over='raise',invalid='raise'):
        if kind=='spline':
            for j,k in enumerate(knots):
                z=features[:,j:j+1];pieces.extend((z,z*z,z*z*z,np.maximum(z-np.asarray(k),0.)**3))
            pieces.append(features[:,count:])
        else:pieces.append(features)
        result=np.concatenate(pieces,axis=1)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite spline basis')
    return result


def _array_identity(value):
    a=np.ascontiguousarray(value)
    return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())


class P13Predictor:
    def __init__(self):self._state=None;self._coefficients=None

    def _fitted(self):
        if self._state is None:raise RuntimeError('Predictor is not fitted')
        return self._state,self._coefficients

    def fit(self,request):
        if not isinstance(request,FitRequest):raise ValueError('P13 requires FitRequest')
        root=Path(request.artifact_dir);root.mkdir(parents=True,exist_ok=False)
        started=time.perf_counter_ns()
        try:
            cfg=_config(request.config)
            if type(request.seed) is not int or not 0<=request.seed<2**63:raise ValueError('Invalid P13 seed')
            raw=np.asarray(request.y_fit)
            if raw.ndim!=1 or raw.dtype.kind not in 'iuf' or not len(raw):raise ValueError('F target must be a real vector')
            y=raw.astype(np.float64)
            if not np.isfinite(y).all():raise ValueError('F target must be finite')
            state=_preprocessing(request.X_fit,y,cfg['feature_schema'])
            features,count=_features(request.X_fit,cfg['feature_schema'],state)
            method=cfg['estimator'];knots=[]
            if method=='spline':
                for values in features[:,:count].T:
                    k=np.unique(np.quantile(values,np.linspace(0,1,cfg['parameters']['n_knots']+2)[1:-1],method='linear'))
                    knots.append(k[(k>values.min())&(k<values.max())].tolist())
            design=_basis(features,count,method,knots);basis_width=design.shape[1]
            target=(y-state['target_mean'])/state['target_scale']
            if not np.isfinite(target).all():raise FloatingPointError('Nonfinite standardized F target')
            if method=='mean':coefficients=np.zeros(1,dtype=np.float64);rank=1
            else:
                width=basis_width-1
                augmented=np.vstack((design,np.column_stack((np.zeros(width),np.sqrt(cfg['parameters']['alpha'])*np.eye(width)))))
                coefficients,_,rank,_=np.linalg.lstsq(augmented,np.r_[target,np.zeros(width)],rcond=None)
            if not np.isfinite(coefficients).all():raise FloatingPointError('Nonfinite fit coefficients')
            candidate=dict(schema='research/v2-p13-snapshot',config=cfg,preprocessing=state,
                basis=dict(kind=method,knots=knots,objective='sum_squared_error_plus_alpha_l2_unpenalized_intercept'),
                encoded_width=features.shape[1],basis_width=basis_width,seed=request.seed,
                F_identity=dict(features=_array_identity(features),target=_array_identity(y)),
                source_identity=_source(),runtime_identity=_runtime())
            receipt=dict(status='completed',elapsed_ns=time.perf_counter_ns()-started,seed=request.seed,
                config=cfg,encoded_width=features.shape[1],basis_width=basis_width,rank=int(rank),
                selection_labels_used=False,scope='closed-form adapter; whole-worker cost and workload qualification external')
            _write(root/'fit-receipt.json',receipt)
        except BaseException as error:
            receipt=dict(status='failed',elapsed_ns=time.perf_counter_ns()-started,error_type=type(error).__name__,
                reason=str(error),selection_labels_used=False)
            try:_write(root/'fit-receipt.json',receipt)
            except BaseException as publication:error.add_note(f'P13 failure receipt publication also failed: {publication}')
            raise
        self._state,self._coefficients=candidate,coefficients.copy()
        return self

    def predict(self,X,*,batch_size=None):
        state,coefficients=self._fitted()
        if batch_size is not None and (type(batch_size) is not int or batch_size<=0):raise ValueError('Invalid prediction batch size')
        features,count=_features(X,state['config']['feature_schema'],state['preprocessing'])
        design=_basis(features,count,state['config']['estimator'],state['basis']['knots'])
        with np.errstate(over='raise',invalid='raise'):
            result=(design@coefficients)*state['preprocessing']['target_scale']+state['preprocessing']['target_mean']
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite original-unit predictions')
        return result

    def save(self,directory):
        state,coefficients=self._fitted();root=Path(directory);root.mkdir(parents=True,exist_ok=False)
        if state['source_identity']!=_source() or state['runtime_identity']!=_runtime():raise ValueError('Fitted source/runtime changed')
        arrays={'coefficients':coefficients}
        flat=np.asarray([x for k in state['basis']['knots'] for x in k],dtype=np.float64)
        offsets=np.cumsum([0]+[len(k) for k in state['basis']['knots']],dtype=np.int64)
        arrays.update(knots=flat,knot_offsets=offsets)
        np.savez(root/'arrays.npz',**arrays)
        manifest=copy.deepcopy(state);manifest.update(arrays={k:_array_identity(v) for k,v in arrays.items()},
            array_file=dict(path='arrays.npz',sha256=_sha(root/'arrays.npz')))
        _write(root/'manifest.json',manifest)

    @classmethod
    def load(cls,directory,*,expected_manifest_sha256):
        root=Path(directory);manifest_path=root/'manifest.json'
        if (root.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file()
                or type(expected_manifest_sha256) is not str or _sha(manifest_path)!=expected_manifest_sha256):
            raise ValueError('External P13 snapshot commitment differs')
        state=_read(manifest_path)
        fields={'schema','config','preprocessing','basis','encoded_width','basis_width','seed','F_identity','source_identity','runtime_identity','arrays','array_file'}
        if type(state) is not dict or set(state)!=fields or state['schema']!='research/v2-p13-snapshot':raise ValueError('Invalid snapshot schema')
        cfg=_config(state['config']);_validate_preprocessing(state['preprocessing'],cfg['feature_schema'])
        if state['source_identity']!=_source() or state['runtime_identity']!=_runtime():raise ValueError('Snapshot source/runtime identity differs')
        if type(state['seed']) is not int or not 0<=state['seed']<2**63:raise ValueError('Invalid saved seed')
        for key in ('encoded_width','basis_width'):
            if type(state[key]) is not int or state[key]<0:raise ValueError('Invalid design width')
        basis=state['basis'];method=cfg['estimator']
        if (type(basis) is not dict or set(basis)!={'kind','knots','objective'} or basis['kind']!=method
                or basis['objective']!='sum_squared_error_plus_alpha_l2_unpenalized_intercept'
                or type(basis['knots']) is not list):raise ValueError('Invalid basis declaration')
        numeric=[c for c in state['preprocessing']['columns'] if c['kind']=='numeric'];count=len(numeric)
        encoded=count+sum(c['indicator'] for c in numeric)+sum(len(c['categories'])+1 for c in state['preprocessing']['columns'] if c['kind']=='categorical')
        if (len(basis['knots'])!=(count if method=='spline' else 0)
                or any(type(k) is not list or len(k)>cfg['parameters']['n_knots'] or any(not _finite(v) for v in k)
                    or any(a>=b for a,b in zip(k,k[1:])) for k in basis['knots'])
                or method=='spline' and any(c['constant'] and k for c,k in zip(numeric,basis['knots']))):raise ValueError('Invalid saved knot geometry')
        width=1 if method=='mean' else 1+encoded if method=='ridge' else 1+sum(3+len(k) for k in basis['knots'])+encoded-count
        if state['encoded_width']!=encoded or state['basis_width']!=width:raise ValueError('Saved design geometry differs')
        if type(state['F_identity']) is not dict:raise ValueError('Invalid F identity container')
        for label,identity in state['F_identity'].items():
            if (label not in ('features','target') or type(identity) is not dict or set(identity)!={'dtype','shape','sha256'}
                    or identity['dtype']!='float64' or type(identity['shape']) is not list
                    or not identity['shape'] or any(type(n) is not int or n<=0 for n in identity['shape'])
                    or type(identity['sha256']) is not str or len(identity['sha256'])!=64):raise ValueError('Invalid F identity')
        if (set(state['F_identity'])!={'features','target'}
                or state['F_identity']['features']['shape']!=[state['F_identity']['target']['shape'][0],encoded]
                or len(state['F_identity']['target']['shape'])!=1):raise ValueError('F identity geometry differs')
        reference=state['array_file'];path=root/'arrays.npz'
        if (type(reference) is not dict or set(reference)!={'path','sha256'} or reference['path']!='arrays.npz'
                or path.is_symlink() or not path.is_file() or _sha(path)!=reference['sha256']
                or set(p.name for p in root.iterdir())!={'manifest.json','arrays.npz'}):raise ValueError('Invalid snapshot file inventory')
        with np.load(path,allow_pickle=False) as archive:
            if set(archive.files)!={'coefficients','knots','knot_offsets'} or len(archive.files)!=3:raise ValueError('Invalid array inventory')
            arrays={name:archive[name].copy() for name in archive.files}
        if set(state['arrays'])!=set(arrays):raise ValueError('Invalid declared array inventory')
        for name,value in arrays.items():
            dtype=np.dtype('int64') if name=='knot_offsets' else np.dtype('float64')
            if value.dtype!=dtype or value.ndim!=1 or not np.isfinite(value).all() or _array_identity(value)!=state['arrays'][name]:
                raise ValueError('Array dtype, finiteness or commitment differs')
        flat=np.asarray([v for k in basis['knots'] for v in k],dtype=np.float64)
        offsets=np.cumsum([0]+[len(k) for k in basis['knots']],dtype=np.int64)
        if (arrays['coefficients'].shape!=(width,) or not np.array_equal(arrays['knots'],flat)
                or not np.array_equal(arrays['knot_offsets'],offsets)
                or method=='mean' and arrays['coefficients'][0]!=0.):raise ValueError('Saved array geometry differs')
        model=cls();model._coefficients=arrays['coefficients'];model._state={k:v for k,v in state.items() if k not in ('arrays','array_file')}
        return model

    def resource_summary(self):
        state=self._state
        return dict(fitted=state is not None,full_workload_qualified=False,
            encoded_width=None if state is None else state['encoded_width'],
            basis_width=None if state is None else state['basis_width'],
            total_parameters=None if self._coefficients is None else len(self._coefficients),
            capabilities=dict(selection_labels_used=False,output_units='original_target',dtype='float64',
                categories='F-only typed one-hot with explicit unknown column',internal_validation=False,
                backend='numpy_cpu',fallback=False,batch_composition_dependent=False),
            resource_scope='adapter geometry only; finite workload, process limits, recovery and standalone timing require harness')
