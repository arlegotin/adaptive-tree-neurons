"""Pinned shallow NODE regression, CPU float32, F transforms / G selection.

Numerical kernels are the archived MIT vendor sources. The notebook's optimizer
was unpinned; QHOptim 1.1.0 is an explicit modern dependency resolution, not a
claim to reproduce the historical YEAR run. Bounded fits report truncation.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import time
from collections import deque
from typing import Any, Mapping
import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from treeneuron.baselines.api import FitRequest
from treeneuron.baselines._node_vendor.arch import DenseBlock
from treeneuron.baselines._node_vendor.nn_utils import entmax15, entmoid15, Lambda
from treeneuron.baselines._node_vendor.qhadam import QHAdam

SOURCE_COMMIT='3bae6a8a63f0205683270b6d566d9cfa659403e4'
QHOPTIM_COMMIT='0e0dc6053c4f7f95bc26cfeb336c6471fea4c5ea'
DEFAULTS=dict(recipe='year_node_shallow',trees=2048,layers=1,depth=6,tree_dim=3,
              initialization_rows=1000,batch_size=512,validation_chunk=1024,
              report_frequency=100,patience=5000,latest_checkpoints=5,device='cpu')
OPTIMIZER=dict(lr=.001,betas=(.95,.998),nus=(.7,1.),eps=1e-8,
               weight_decay=0.,decouple_weight_decay=False)


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def node_identity():
    from importlib.metadata import version
    from treeneuron.train import source_hash,lock_hash
    lock_path=Path(__file__).resolve().parents[3]/'locks/node/environment.json'
    locked=json.loads(lock_path.read_text())
    packages={k:version(k) for k in ('torch','numpy','scipy','scikit-learn')}
    if (any(locked['packages'][k]!=v for k,v in packages.items())
            or locked['python']!=platform.python_version() or locked['architecture']!=platform.machine()
            or locked['core_lock_sha256']!=lock_hash()):
        raise RuntimeError('Pinned NODE native environment mismatch')
    return dict(package_source=source_hash(),core_lock=lock_hash(),node_commit=SOURCE_COMMIT,
                qhoptim_commit=QHOPTIM_COMMIT,node_lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                python=platform.python_version(),machine=platform.machine(),packages=packages)


def _matrix(x):
    a=np.asarray(x)
    if a.ndim!=2 or a.dtype.kind not in 'fiu' or not a.shape[1] or not np.isfinite(a).all():
        raise ValueError('NODE requires finite numerical matrix; categorical/missing path is unsupported')
    return np.array(a,dtype=np.float32,copy=True)


def _state(model):
    return {k:v.detach().clone() for k,v in model.state_dict().items()}


def average_states(states):
    """Newest parameter snapshots; integer initialized flags retain their type."""
    result={}
    for key,value in states[-1].items():
        if value.is_floating_point():result[key]=sum(s[key] for s in reversed(states))/len(states)
        else:
            if not all(torch.equal(s[key],value) for s in states):raise ValueError('Initialization state differs')
            result[key]=value.clone()
    return result


class NodePredictor:
    def __init__(self, config: Mapping[str,Any]):
        unknown=set(config)-set(DEFAULTS)
        if unknown:raise ValueError(f'Unknown NODE configuration keys: {sorted(unknown)}')
        self.config={**DEFAULTS,**dict(config)}
        if self.config['recipe'] not in ('year_node_shallow','engineering_fixture'):
            raise ValueError('Unknown named NODE recipe')
        if self.config['device']!='cpu':raise ValueError('Only explicitly verified CPU path is available')
        if self.config['recipe']=='year_node_shallow' and self.config!=DEFAULTS:
            raise ValueError('The named shallow anchor cannot silently change defaults')
        for key,value in self.config.items():
            if key not in ('recipe','device') and (type(value) is not int or value<=0):
                raise ValueError(f'{key} must be a positive integer')
        # Shallow fixture varies scale/cadence only, not the experimental functions/head.
        if self.config['layers']!=1 or self.config['tree_dim']!=3 or self.config['latest_checkpoints']!=5:
            raise ValueError('Only shallow three-channel five-snapshot recipe is implemented')
        self.model=None;self.fitted=False;self.costs={}

    def build_model(self, width):
        return torch.nn.Sequential(DenseBlock(width,self.config['trees'],num_layers=1,
            depth=self.config['depth'],tree_dim=3,flatten_output=False,
            choice_function=entmax15,bin_function=entmoid15),
            Lambda(lambda x:x[...,0].mean(dim=-1))).cpu().float()

    def build_optimizer(self, parameters):
        return QHAdam(parameters,**OPTIMIZER)

    def fit(self, request: FitRequest):
        limit=request.budget.get('max_updates')
        if type(limit) is not int or limit<=0:
            raise ValueError('Explicit positive max_updates resource budget is required')
        x=_matrix(request.X_fit);g=_matrix(request.X_select)
        y=np.asarray(request.y_fit);gy=np.asarray(request.y_select)
        if not len(x) or not len(g) or x.shape[1]!=g.shape[1] or y.shape!=(len(x),) or gy.shape!=(len(g),):
            raise ValueError('NODE F/G rows, target shape or feature width invalid')
        if not np.isfinite(y).all() or not np.isfinite(gy).all():raise ValueError('Nonfinite targets')
        self.fitted=False
        self.seed=request.seed;self.identity=node_identity()
        self.n_fit=len(x);self.n_select=len(g);self.width=x.shape[1]
        self.y_mean=float(y.mean());self.y_std=float(y.std())
        if self.y_std<=0:raise ValueError('NODE anchor requires nonconstant F targets')
        before=(np.random.get_state(),torch.get_rng_state(),random.getstate())
        started=time.perf_counter()
        try:
            np.random.seed(request.seed);torch.manual_seed(request.seed);random.seed(request.seed)
            self.quantile=QuantileTransformer(n_quantiles=1000,output_distribution='normal',
                ignore_implicit_zeros=False,subsample=10000,random_state=request.seed,copy=True)
            noisy=x.copy();noise_std=1e-3/np.maximum(np.std(noisy,axis=0,keepdims=True),1e-3)
            noisy+=noise_std*np.random.randn(*noisy.shape)
            self.quantile.fit(noisy)
            tx=torch.from_numpy(self.quantile.transform(x).astype('float32'))
            tg=torch.from_numpy(self.quantile.transform(g).astype('float32'))
            ty=torch.from_numpy(((y-self.y_mean)/self.y_std).astype('float32'))
            self.costs['preprocessing_seconds']=time.perf_counter()-started
            self.model=self.build_model(self.width)
            self.initialization_rows=min(len(x),self.config['initialization_rows'])
            self.initialization_prefix_sha256=hashlib.sha256(x[:self.initialization_rows].tobytes()).hexdigest()
            t=time.perf_counter()
            with torch.no_grad():self.model(tx[:self.initialization_rows])
            self.costs['initialization_seconds']=time.perf_counter()-t
            self.initial_state=_state(self.model)
            self.optimizer=self.build_optimizer(self.model.parameters())
            self.training_row_indices=[];self.report_history=[];self.selected_step=None
            snapshots=deque(maxlen=5);snapshot_steps=deque(maxlen=5)
            self.best_state=None;self.best_optimizer_state=None;best_mse=float('inf');best_step=0;step=0
            indices=np.arange(len(x));self.stop_reason='budget';self.costs['step_seconds']=[]
            fit_start=time.perf_counter()
            while step<limit:
                np.random.shuffle(indices)
                for start in range(0,len(x),self.config['batch_size']):
                    batch=indices[start:start+self.config['batch_size']]
                    t=time.perf_counter();self.model.train();self.optimizer.zero_grad()
                    loss=(self.model(tx[batch])-ty[batch]).square().mean()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite NODE training loss')
                    loss.backward();self.optimizer.step();step+=1
                    self.costs['step_seconds'].append(time.perf_counter()-t)
                    self.training_row_indices.extend(int(i) for i in batch)
                    if step%self.config['report_frequency']==0:
                        raw=_state(self.model);snapshots.append(raw);snapshot_steps.append(step)
                        optimizer_before=copy.deepcopy(self.optimizer.state_dict())
                        averaged=average_states(list(snapshots));self.model.load_state_dict(averaged)
                        pred=self._normalized_prediction(tg,self.config['validation_chunk'])
                        mse=float(np.mean((pred-(gy-self.y_mean)/self.y_std)**2))
                        if mse<best_mse:
                            best_mse=mse;best_step=step;self.best_state=_state(self.model);self.selected_step=step
                            self.best_optimizer_state=copy.deepcopy(self.optimizer.state_dict())
                        self.model.load_state_dict(raw)
                        raw_error=max(float((self.model.state_dict()[k]-v).abs().max()) for k,v in raw.items())
                        optimizer_error=max((float((self.optimizer.state_dict()['state'][k][name]-v).abs().max())
                            for k,s in optimizer_before['state'].items() for name,v in s.items() if isinstance(v,torch.Tensor)),default=0.)
                        self.report_history.append(dict(step=step,g_mse=mse,snapshot_steps=list(snapshot_steps),
                            raw_restore_max_abs=raw_error,optimizer_restore_max_abs=optimizer_error))
                    # Notebook stop test runs every update, not only on reports.
                    if step>best_step+self.config['patience']:
                        self.stop_reason='G_patience';break
                    if step>=limit:break
                if self.stop_reason=='G_patience':break
            self.costs['training_seconds']=time.perf_counter()-fit_start
            self.executed_updates=step;self.raw_state=_state(self.model)
            self.raw_optimizer_state=copy.deepcopy(self.optimizer.state_dict())
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)
                self.optimizer.load_state_dict(self.best_optimizer_state)
            self.model.eval();self.fitted=True
            self.costs['fit_seconds']=time.perf_counter()-started
            return self
        finally:
            np.random.set_state(before[0]);torch.set_rng_state(before[1]);random.setstate(before[2])

    def _normalized_prediction(self, x, batch_size):
        self.model.eval()
        with torch.no_grad():return np.concatenate([self.model(b).cpu().numpy() for b in x.split(batch_size)])

    def predict(self, X, *, batch_size=None):
        if not self.fitted:raise RuntimeError('NODE must be fitted before prediction')
        chunk=self.config['validation_chunk'] if batch_size is None else batch_size
        if type(chunk) is not int or chunk<=0:raise ValueError('batch_size must be positive')
        x=_matrix(X)
        if x.shape[1]!=self.width:raise ValueError('Feature width changed')
        if not len(x):return np.empty(0,dtype=np.float64)
        if not all(bool(m._is_initialized_tensor.item()) for m in self.model[0]):
            raise RuntimeError('Uninitialized NODE prediction is forbidden')
        t=time.perf_counter()
        tx=torch.from_numpy(self.quantile.transform(x).astype('float32'))
        prediction=self._normalized_prediction(tx,chunk).astype('float64')*self.y_std+self.y_mean
        if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite NODE prediction')
        self.costs['last_prediction_seconds']=time.perf_counter()-t
        return prediction

    def resource_summary(self):
        return dict(method='B7',recipe=self.config,source_commit=SOURCE_COMMIT,
            optimizer={'name':'QHAdam','version':'1.1.0','resolved':OPTIMIZER},
            backend='CPU float32 eager',fallback=False,internal_selection='G latest-five parameter averaging',
            categories='unsupported',missingness='unsupported',context=False,transduction=False,cache=False,
            initialization_rows=getattr(self,'initialization_rows',None),
            initialization_requested=self.config['initialization_rows'],
            initialization_prefix_sha256=getattr(self,'initialization_prefix_sha256',None),
            n_fit=getattr(self,'n_fit',None),n_select=getattr(self,'n_select',None),
            executed_updates=getattr(self,'executed_updates',None),selected_step=getattr(self,'selected_step',None),
            stop_reason=getattr(self,'stop_reason',None),quality_plateau_calibrated=False,
            selection_checkpoint_available=getattr(self,'selected_step',None) is not None,
            official_selection_completed=self.config['recipe']=='year_node_shallow' and getattr(self,'selected_step',None) is not None,
            quality_selection_eligible=self.config['recipe']=='year_node_shallow' and getattr(self,'selected_step',None) is not None,
            selection_status='G_selected' if getattr(self,'selected_step',None) is not None else 'unavailable_at_budget',
            training_rows_seen=len(getattr(self,'training_row_indices',[])),
            unique_training_rows=len(set(getattr(self,'training_row_indices',[]))),
            trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else None,
            costs=copy.deepcopy(self.costs),memory={'value':None,'reason':'process peak measured by separate cost worker'},
            identity=getattr(self,'identity',None))

    def save(self,path):
        if not self.fitted:raise RuntimeError('NODE must be fitted before saving')
        path=Path(path);arrays={}
        def encode(obj):
            if isinstance(obj,(np.ndarray,torch.Tensor)):
                key=f'a{len(arrays)}';arrays[key]=obj.detach().cpu().numpy() if isinstance(obj,torch.Tensor) else obj
                return {'array':key,'tensor':isinstance(obj,torch.Tensor)}
            if isinstance(obj,dict):return {'mapping':[[encode(k),encode(v)] for k,v in obj.items()]}
            if isinstance(obj,(tuple,list)):return {'sequence':[encode(v) for v in obj]}
            return obj
        state=dict(config=self.config,identity=self.identity,width=self.width,y_mean=self.y_mean,y_std=self.y_std,
            model=_state(self.model),raw=self.raw_state,optimizer=self.optimizer.state_dict(),raw_optimizer=self.raw_optimizer_state,
            quantile_params=self.quantile.get_params(),quantiles=self.quantile.quantiles_,references=self.quantile.references_,
            summary=self.resource_summary(),history=self.report_history,training_row_indices=self.training_row_indices)
        payload=encode(state);archive=path.with_suffix('.npz')
        np.savez_compressed(archive,**arrays)
        path.write_text(json.dumps(dict(version=1,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),state=payload,metadata_sha256=_hash(payload)),allow_nan=False)+'\n')

    @classmethod
    def load(cls,path):
        path=Path(path);envelope=json.loads(path.read_text());archive=path.with_suffix('.npz')
        if (envelope['version']!=1 or hashlib.sha256(archive.read_bytes()).hexdigest()!=envelope['sha256']
                or _hash(envelope['state'])!=envelope.get('metadata_sha256')):
            raise ValueError('NODE checkpoint integrity mismatch')
        with np.load(archive,allow_pickle=False) as arrays:
            def decode(obj):
                if isinstance(obj,dict):
                    if 'array' in obj:
                        value=arrays[obj['array']].copy()
                        return torch.from_numpy(value) if obj['tensor'] else value
                    if 'mapping' in obj:return {decode(k):decode(v) for k,v in obj['mapping']}
                    if 'sequence' in obj:return [decode(v) for v in obj['sequence']]
                return obj
            s=decode(envelope['state'])
        if s['identity']!=node_identity():raise ValueError('NODE source/lock/runtime identity changed')
        m=cls(s['config']);m.identity=s['identity'];m.width=s['width'];m.y_mean=s['y_mean'];m.y_std=s['y_std']
        m.quantile=QuantileTransformer(**s['quantile_params']);m.quantile.quantiles_=s['quantiles']
        m.quantile.references_=s['references'];m.quantile.n_quantiles_=len(s['references']);m.quantile.n_features_in_=m.width
        before=torch.get_rng_state()
        try:m.model=m.build_model(m.width)
        finally:torch.set_rng_state(before)
        m.model.load_state_dict(s['model'])
        if not all(bool(layer._is_initialized_tensor.item()) for layer in m.model[0]):
            raise ValueError('Checkpoint contains uninitialized NODE')
        m.optimizer=m.build_optimizer(m.model.parameters());m.optimizer.load_state_dict(s['optimizer'])
        m.raw_state=s['raw'];m.raw_optimizer_state=s['raw_optimizer'];m.report_history=s['history'];m.costs=s['summary']['costs']
        for key in ('n_fit','n_select','executed_updates','selected_step','stop_reason','initialization_rows','initialization_prefix_sha256'):
            setattr(m,key,s['summary'][key])
        m.training_row_indices=s['training_row_indices'];m.fitted=True;m.model.eval()
        return m
