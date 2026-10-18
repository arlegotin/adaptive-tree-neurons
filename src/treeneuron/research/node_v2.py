"""P12 NODE-v2: explicit F-only adaptations and portable native endpoint state.

Kernels/notices are the unchanged pinned NODE/QHAdam vendor sources. Engineering
fixtures do not qualify full workloads, hard process limits, recovery or Gate D.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd

from treeneuron.baselines.api import FitRequest
from treeneuron.research.native_trees import _schema, _frame, _category
from treeneuron.research.p13_simple import _read, _write, _numeric, _missing, _valid_token, _array_identity

_ROOT=Path(__file__).resolve().parents[3]
_THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def _digest(value):return hashlib.sha256(_canonical(value).encode()).hexdigest()
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _finite(value):return type(value) in (int,float) and math.isfinite(value)
def _seed(value):
    if type(value) is not int or not 0<=value<2**32:raise ValueError('NODE requires an explicit uint32 NumPy seed')
    return value


def source_identity():
    lockpath=_ROOT/'locks/research-v2/node-v2/source-lock.json';lock=_read(lockpath)
    for name,digest in lock['files'].items():
        if _sha(_ROOT/name)!=digest:raise ValueError('Pinned NODE source/lock bytes changed')
    files={str(p.relative_to(_ROOT)):_sha(p) for p in sorted((_ROOT/'src/treeneuron').rglob('*.py'))}
    files.update(lock['files']);files[str(lockpath.relative_to(_ROOT))]=_sha(lockpath)
    return dict(package_sha256=_digest(files),files=files)


@dataclass(frozen=True)
class NodeV2Config:
    state:dict

    @classmethod
    def parse(cls,value):
        if (type(value) is not dict or set(value)!={'schema','family','mode','feature_schema','parameters'}
                or value['schema']!='research/v2-node-config' or value['family']!='P12'
                or value['mode'] not in ('native_anchor','search','engineering_fixture')):
            raise ValueError('Invalid versioned NODE configuration')
        schema=_schema(value['feature_schema']);p=value['parameters'];mode=value['mode']
        if type(p) is not dict:raise ValueError('NODE parameters must be a mapping')
        support={} if mode=='native_anchor' else dict(trees=(512,1024,2048),depth=(4,6),lr_multiplier=(.5,1,2))
        if mode=='engineering_fixture':support=dict(trees=(16,32),depth=(2,4),batch_size=(8,16,32),report_frequency=(1,2,100),patience=(1,2,5000))
        if not set(p)<=set(support) or mode=='search' and set(p)!=set(support):raise ValueError('Unknown setting or implicit NODE recipe override')
        for key,item in p.items():
            if type(item) not in ((int,float) if key=='lr_multiplier' else (int,)) or item not in support[key]:
                raise ValueError('NODE setting outside frozen support')
        return cls(dict(schema=value['schema'],family='P12',mode=mode,feature_schema=schema,parameters=copy.deepcopy(p)))

    def to_state(self):return copy.deepcopy(self.state)

    def resolved(self):
        mode=self.state['mode'];p=self.state['parameters']
        result=dict(trees=16 if mode=='engineering_fixture' else 2048,depth=2 if mode=='engineering_fixture' else 6,
            layers=1,tree_dim=3,batch_size=8 if mode=='engineering_fixture' else 512,
            report_frequency=2 if mode=='engineering_fixture' else 100,patience=5000,latest_checkpoints=5,
            initialization_rows=1000,validation_chunk=1024,choice='entmax15',bin='entmoid15',
            head='channel0_mean_over_trees',dtype='float32',device='cpu')
        result.update({k:v for k,v in p.items() if k!='lr_multiplier'})
        result['optimizer']=dict(lr=.001*p.get('lr_multiplier',1),betas=[.95,.998],nus=[.7,1.],eps=1e-8,weight_decay=0.,decouple_weight_decay=False)
        return result


def _quantile_params(seed):
    return dict(n_quantiles=1000,output_distribution='normal',ignore_implicit_zeros=False,subsample=10000,random_state=seed,copy=True)


def _preprocess(X,y,schema):
    frame=_frame(X,schema)
    if len(frame)<2 or y.shape!=(len(frame),):raise ValueError('NODE F requires at least two matched rows')
    columns=[]
    for name in schema['columns']:
        if name in schema['categorical']:
            tokens=[None if _missing(v) else _category(v) for v in frame[name]]
            categories=sorted(set(v for v in tokens if v is not None))
            columns.append(dict(name=name,kind='categorical',categories=categories,unknown_index=len(categories)))
        else:
            raw=_numeric(frame[name]);missing=np.isnan(raw)
            fill=0. if missing.all() else float(np.median(raw[~missing]))
            columns.append(dict(name=name,kind='numeric',fill=fill,indicator=bool(missing.any()),all_missing=bool(missing.all())))
    mean=float(y.mean());constant=bool(np.all(y==y[0]));scale=1. if constant else float(y.std(ddof=0))
    if not _finite(mean) or not _finite(scale) or scale<=0:raise ValueError('Nonfinite or underflowed F target statistics')
    return dict(columns=columns,target_mean=mean,target_scale=scale,target_constant=constant)


def _validate_preprocessing(value,schema):
    if (type(value) is not dict or set(value)!={'columns','target_mean','target_scale','target_constant'}
            or not _finite(value['target_mean']) or not _finite(value['target_scale']) or value['target_scale']<=0
            or type(value['target_constant']) is not bool or value['target_constant'] and value['target_scale']!=1.
            or type(value['columns']) is not list or len(value['columns'])!=len(schema['columns'])):
        raise ValueError('Invalid NODE F preprocessing state')
    width=0
    for name,column in zip(schema['columns'],value['columns']):
        category=name in schema['categorical']
        fields={'name','kind','categories','unknown_index'} if category else {'name','kind','fill','indicator','all_missing'}
        if type(column) is not dict or set(column)!=fields or column['name']!=name or column['kind']!=('categorical' if category else 'numeric'):
            raise ValueError('NODE preprocessing column schema differs')
        if category:
            categories=column['categories']
            if (type(categories) is not list or any(not _valid_token(v) for v in categories)
                    or categories!=sorted(set(categories)) or type(column['unknown_index']) is not int
                    or column['unknown_index']!=len(categories)):raise ValueError('Invalid typed NODE vocabulary')
            width+=len(categories)+1
        else:
            if (not _finite(column['fill']) or type(column['indicator']) is not bool or type(column['all_missing']) is not bool
                    or column['all_missing'] and not (column['indicator'] and column['fill']==0.)):
                raise ValueError('Invalid NODE numeric imputation')
            width+=1+column['indicator']
    return width


def _encode_features(X,schema,state):
    frame=_frame(X,schema);numeric=[];indicators=[];categories=[]
    for column in state['columns']:
        values=frame[column['name']]
        if column['kind']=='numeric':
            raw=_numeric(values);missing=np.isnan(raw)
            numeric.append(np.where(missing,column['fill'],raw))
            if column['indicator']:indicators.append(missing.astype(np.float32))
        else:
            known={v:i for i,v in enumerate(column['categories'])};block=np.zeros((len(frame),len(known)+1),dtype=np.float32)
            codes=[known.get(None if _missing(v) else _category(v),column['unknown_index']) for v in values]
            block[np.arange(len(frame)),codes]=1.;categories.extend(block.T)
    with np.errstate(over='raise',invalid='raise'):
        encoded=np.column_stack(numeric+indicators+categories).astype(np.float32)
    if not np.isfinite(encoded).all():raise ValueError('Nonfinite NODE encoded features')
    return encoded


class NodeTransform:
    def __init__(self,state,quantile):self.state=state;self.quantile=quantile

    @classmethod
    def fit(cls,X,y,schema,seed):
        _seed(seed);schema=_schema(schema);target=_target(y)
        preprocessing=_preprocess(X,target,schema);encoded=_encode_features(X,schema,preprocessing)
        np.random.seed(seed);noisy=encoded.copy()
        with np.errstate(over='raise',invalid='raise'):
            noisy+=(.001/np.maximum(np.std(noisy,axis=0,keepdims=True),.001))*np.random.randn(*noisy.shape)
        qt=importlib.import_module('sklearn.preprocessing').QuantileTransformer(**_quantile_params(seed)).fit(noisy)
        state=dict(schema=schema,preprocessing=preprocessing,seed=seed,n_fit=len(encoded),width=encoded.shape[1],
            params=_quantile_params(seed),quantiles=qt.quantiles_.copy(),references=qt.references_.copy())
        return cls.from_state(state)

    @classmethod
    def from_state(cls,state):
        fields={'schema','preprocessing','seed','n_fit','width','params','quantiles','references'}
        if type(state) is not dict or set(state)!=fields:raise ValueError('Invalid NODE transform schema')
        schema=_schema(state['schema']);_seed(state['seed']);width=_validate_preprocessing(state['preprocessing'],schema)
        if (type(state['n_fit']) is not int or state['n_fit']<2 or type(state['width']) is not int or state['width']!=width
                or state['params']!=_quantile_params(state['seed'])):raise ValueError('NODE transform geometry/recipe differs')
        q,r=state['quantiles'],state['references'];n=min(1000,state['n_fit'])
        if (not isinstance(q,np.ndarray) or not isinstance(r,np.ndarray) or q.dtype!=np.dtype('float64')
                or r.dtype!=np.dtype('float64') or q.shape!=(n,width) or r.shape!=(n,)
                or not np.isfinite(q).all() or not np.isfinite(r).all() or np.any(np.diff(q,axis=0)<0)
                or not np.array_equal(r,np.linspace(0.,1.,n))):raise ValueError('Invalid NODE quantile arrays/dtypes')
        qt=importlib.import_module('sklearn.preprocessing').QuantileTransformer(**state['params'])
        qt.quantiles_=q.copy();qt.references_=r.copy();qt.n_quantiles_=n;qt.n_features_in_=width
        return cls(copy.deepcopy(state),qt)

    def transform(self,X):
        encoded=_encode_features(X,self.state['schema'],self.state['preprocessing'])
        if not len(encoded):return encoded
        value=self.quantile.transform(encoded)
        if value.dtype!=np.float32 or not np.isfinite(value).all():raise FloatingPointError('Invalid native NODE transformed features')
        return value

    def target(self,y):
        value=_target(y);p=self.state['preprocessing']
        with np.errstate(over='raise',invalid='raise'):
            result=((value-p['target_mean'])/p['target_scale']).astype(np.float32)
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite NODE normalized targets')
        return result

    def inverse(self,prediction):
        p=self.state['preprocessing']
        with np.errstate(over='raise',invalid='raise'):
            result=prediction.astype(np.float64)*p['target_scale']+p['target_mean']
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite NODE original-unit prediction')
        return result

    def to_state(self):return copy.deepcopy(self.state)


def _target(value):
    raw=np.asarray(value)
    if raw.ndim!=1 or raw.dtype.kind not in 'iuf' or not len(raw):raise ValueError('NODE target must be a finite real vector')
    with np.errstate(over='raise',invalid='raise'):raw=raw.astype(np.float64)
    if not np.isfinite(raw).all():raise ValueError('NODE target must be finite')
    return raw
