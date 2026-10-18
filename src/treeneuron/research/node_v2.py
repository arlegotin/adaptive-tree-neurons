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
from treeneuron.research.p13_simple import _read, _write, _numeric, _missing, _valid_token

_ROOT=Path(__file__).resolve().parents[3]
_THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def _array_identity(value):
    array=np.asarray(value)
    return dict(dtype=str(array.dtype),shape=list(array.shape),sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def _digest(value):return hashlib.sha256(_canonical(value).encode()).hexdigest()
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _finite(value):return type(value) in (int,float) and math.isfinite(value)
def _seed(value):
    if type(value) is not int or not 0<=value<2**32:raise ValueError('NODE requires an explicit uint32 NumPy seed')
    return value


def _verify_vendor_origins():
    for name in ('arch','odst','nn_utils','qhadam','param_conv','utils'):
        module=sys.modules.get('treeneuron.baselines._node_vendor.'+name)
        expected=_ROOT/'src/treeneuron/baselines/_node_vendor'/f'{name}.py'
        if module is not None and (getattr(module,'__file__',None) is None or Path(module.__file__).resolve()!=expected.resolve()):
            raise ValueError('Loaded NODE vendor origin differs from its source lock')


def source_identity():
    _verify_vendor_origins()
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


def _native():
    torch=importlib.import_module('torch')
    arch=importlib.import_module('treeneuron.baselines._node_vendor.arch')
    nn=importlib.import_module('treeneuron.baselines._node_vendor.nn_utils')
    qh=importlib.import_module('treeneuron.baselines._node_vendor.qhadam')
    importlib.import_module('sklearn.preprocessing')
    return torch,arch,nn,qh


def _runtime():
    """Frozen core metadata plus selected executed source and actual native images."""
    _verify_vendor_origins()
    if any(os.environ.get(name)!='1' for name in _THREADS) or not sys.dont_write_bytecode:
        raise ValueError('NODE requires one-thread environment and disabled bytecode writes')
    core=_read(_ROOT/'locks/research-v2/core.json');expected={r['name'].lower().replace('_','-'):r for r in core['distributions']}
    if platform.python_version()!=core['python'].split()[0] or platform.machine()!='arm64':raise ValueError('NODE Python/architecture differs from pin')
    actual={}
    for distribution in importlib.metadata.distributions():
        name=distribution.metadata['Name'].lower().replace('_','-')
        if name in actual:raise ValueError('Duplicate NODE dependency distribution')
        actual[name]=distribution
    if set(actual)!=set(expected):raise ValueError('NODE must use exactly the frozen core distributions')
    for name,row in expected.items():
        d=actual[name]
        if d.version!=row['version'] or str(Path(d._path).resolve())!=row['path'] or _sha(Path(d._path)/'RECORD')!=row['record_sha256']:
            raise ValueError('NODE frozen dependency RECORD/version/location changed')
    from treeneuron.research.ordinary_neural import _record_file,_runtime as images,_runtime_identity
    files=[]
    for package,distribution in (('torch','torch'),('numpy','numpy'),('scipy','scipy'),('sklearn','scikit-learn'),('pandas','pandas'),('threadpoolctl','threadpoolctl')):
        path=actual[distribution].locate_file(package+('/__init__.py' if package!='threadpoolctl' else '.py'))
        spec=importlib.util.find_spec(package)
        if spec is None or Path(spec.origin).resolve()!=Path(path).resolve():raise ValueError('NODE native import origin differs')
        module=importlib.import_module(package)
        if Path(module.__file__).resolve()!=Path(path).resolve():raise ValueError('NODE imported native module origin differs')
        files.append(_record_file(distribution,path))
    for name,distribution in (('torch._C','torch'),('torch.optim.optimizer','torch'),('torch.overrides','torch'),('torch.utils._device','torch'),
            ('numpy._core._multiarray_umath','numpy'),('sklearn.preprocessing._data','scikit-learn'),
            ('scipy.stats._distn_infrastructure','scipy'),('scipy.stats._continuous_distns','scipy'),('scipy.special._ufuncs','scipy')):
        files.append(_record_file(distribution,importlib.import_module(name).__file__))
    observed=_runtime_identity(images())
    for pool in observed['threadpools']:
        path=Path(pool['filepath']).resolve();owner=None
        for name in ('torch','numpy','scipy','scikit-learn'):
            base=Path(actual[name].locate_file('')).resolve()
            if path.is_relative_to(base):
                try:owner=_record_file(name,path);break
                except ValueError:continue
        if owner is None:raise ValueError('NODE threadpool native file lacks a core RECORD identity')
        files.append(owner)
    unique={(r['distribution'],r['path']):r for r in files}
    return dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
        executable=sys.executable,executable_sha256=_sha(sys.executable),core_sha256=_sha(_ROOT/'locks/research-v2/core.json'),
        versions={k:v.version for k,v in sorted(actual.items())},files=sorted(unique.values(),key=lambda r:(r['distribution'],r['path'])),
        native=observed,coverage='All27 versions/locations/RECORD digests; listed executed Python/native sources and observed native images/threadpools, not all core installed bytes')


def _require_native_defaults(torch):
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._device import DeviceContext
    # Match pinned get_default_device metadata lookup without its off-device
    # tensor allocation to resolve an unspecified device index.
    context=next((mode for mode in reversed(_get_current_function_mode_stack()) if isinstance(mode,DeviceContext)),None)
    if context is None:context=getattr(torch._GLOBAL_DEVICE_CONTEXT,'device_context',None)
    device=context.device if context is not None else torch.device('cpu')
    if device.type!='cpu' or torch.get_default_dtype()!=torch.float32:
        raise ValueError('NODE requires caller CPU device and float32 default dtype before native draws')


def _build_model(width,resolved):
    torch,arch,nn,_=_native();_require_native_defaults(torch)
    with torch.device('cpu'):
        return torch.nn.Sequential(arch.DenseBlock(width,resolved['trees'],num_layers=1,depth=resolved['depth'],tree_dim=3,
            flatten_output=False,choice_function=nn.entmax15,bin_function=nn.entmoid15),nn.Lambda(lambda x:x[...,0].mean(-1))).cpu().float()


def _optimizer(model,resolved):return _native()[3].QHAdam(model.parameters(),**resolved['optimizer'])
def _optimizer_step(optimizer):optimizer.step()
def _model_state(model):return {k:v.detach().clone() for k,v in model.state_dict().items()}
def _patience_reached(step,best_step,patience):return step>best_step+patience


def _average(window):
    torch=_native()[0];states=[row['model'] for row in reversed(window)];result={}
    for name,value in states[0].items():
        if value.is_floating_point():result[name]=sum(row[name] for row in states)/len(states)
        else:
            if any(not torch.equal(row[name],value) for row in states):raise ValueError('NODE rolling initialization state differs')
            result[name]=value.clone()
    return result


def _initialized(model):
    if any(int(layer._is_initialized_tensor.item())!=1 or layer._is_initialized_bool is not True for layer in model[0]):
        raise ValueError('Uninitialized NODE query is forbidden')


def _predict_normalized(model,features,chunk):
    torch=_native()[0];_initialized(model);model.eval()
    if not len(features):return np.empty(0,dtype=np.float32)
    with torch.no_grad():value=np.concatenate([model(batch).cpu().numpy() for batch in torch.from_numpy(features).split(chunk)])
    if value.dtype!=np.float32 or not np.isfinite(value).all():raise FloatingPointError('Nonfinite NODE native prediction')
    return value


def _rng(torch):return dict(numpy=np.random.get_state(),python=random.getstate(),torch=torch.get_rng_state().clone())
def _restore_rng(value,torch):
    np.random.set_state(value['numpy']);random.setstate(value['python']);torch.set_rng_state(value['torch'])


def _train(model,optimizer,F,y,G,gy,resolved,cap,trace):
    torch=_native()[0];tx=torch.from_numpy(F);ty=torch.from_numpy(y)
    indices=np.arange(len(F),dtype=np.int64);cursor=len(F);epoch=-1;window=[];reports=[];best=None;best_opt=None;best_window=[]
    best_mse=float('inf');best_step=0;step=0
    while step<cap:
        if cursor==len(F):np.random.shuffle(indices);cursor=0;epoch+=1
        batch=indices[cursor:cursor+resolved['batch_size']].copy();cursor+=len(batch)
        record=dict(step=step+1,epoch=epoch,rows=batch.tolist(),loss=None,gradient_norms={},optimizer_entered=False,applied=False,elapsed_ns=0)
        trace.append(record);started=time.perf_counter_ns()
        try:
            model.train();optimizer.zero_grad();loss=(model(tx[batch])-ty[batch]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native NODE training loss')
            record['loss']=float(loss.detach());loss.backward()
            for name,p in model.named_parameters():
                if p.grad is not None:
                    if p.grad.is_sparse or not torch.isfinite(p.grad).all():raise FloatingPointError('Invalid native NODE gradient')
                    norm=float(p.grad.norm())
                    if not math.isfinite(norm):raise FloatingPointError('Nonfinite native NODE gradient norm')
                    record['gradient_norms'][name]=norm
            record['optimizer_entered']=True;_optimizer_step(optimizer);record['applied']=True;step+=1
            if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite native NODE updated parameter')
        finally:record['elapsed_ns']=time.perf_counter_ns()-started
        if step%resolved['report_frequency']==0:
            raw=_model_state(model);window.append(dict(step=step,model=raw));window=window[-5:]
            averaged=_average(window);model.load_state_dict(averaged)
            pred=_predict_normalized(model,G,resolved['validation_chunk']);mse=float(((gy-pred)**2).mean())
            if not math.isfinite(mse) or mse<0:raise FloatingPointError('Invalid native NODE G squared loss')
            reports.append(dict(step=step,g_mse=mse,snapshot_steps=[v['step'] for v in window]))
            if mse<best_mse:
                best_mse=mse;best_step=step;best=_model_state(model);best_opt=copy.deepcopy(optimizer.state_dict());best_window=copy.deepcopy(window)
            model.load_state_dict(raw)
        if _patience_reached(step,best_step,resolved['patience']):break
    raw=_model_state(model);raw_opt=copy.deepcopy(optimizer.state_dict())
    if best is not None:model.load_state_dict(best)
    model.eval()
    return dict(initial_model=None,selected_model=best,selected_optimizer=best_opt,raw_model=raw,raw_optimizer=raw_opt,
        rolling_window=window,selected_window=best_window,reports=reports,trace=trace,
        sampler=dict(permutation=indices.copy(),cursor=cursor,epoch=epoch),rng=_rng(torch),
        summary=dict(executed_updates=step,selected_step=best_step if best is not None else None,
            stop_reason='G_patience' if _patience_reached(step,best_step,resolved['patience']) else 'max_updates',
            checkpoint_available=best is not None,selection_status='G_selected' if best is not None else 'raw_unselected_at_cap',
            quality_eligible=False,native_full_anchor_qualified=False))


class NodeV2Predictor:
    def __init__(self):self._model=None;self._state=None;self._transform=None;self._identity=None;self._config=None

    def fit(self,request):
        if not isinstance(request,FitRequest):raise ValueError('NODE requires FitRequest')
        root=Path(request.artifact_dir);root.mkdir(parents=True,exist_ok=False)
        started=time.perf_counter_ns();trace=[];phase='configuration';cfg=None;before=None;torch=None
        phase_started=started;phase_costs={}
        def enter_phase(name):
            nonlocal phase,phase_started
            now=time.perf_counter_ns();phase_costs[phase]=phase_costs.get(phase,0)+now-phase_started
            phase=name;phase_started=now
        def captured_costs():
            now=time.perf_counter_ns();costs=phase_costs.copy();costs[phase]=costs.get(phase,0)+now-phase_started
            return costs,now-started
        caller_numpy=np.random.get_state();caller_python=random.getstate()
        try:
            cfg=NodeV2Config.parse(request.config);seed=_seed(request.seed);resolved=cfg.resolved();cap=request.budget.get('max_updates')
            if (set(request.budget)!={'max_updates'} or type(cap) is not int or cap<=0
                    or cfg.state['mode']=='engineering_fixture' and cap>128):raise ValueError('Invalid NODE explicit update ceiling')
            enter_phase('source_validation');source=source_identity();enter_phase('dependencies');torch=_native()[0]
            before=_rng(torch);before['numpy']=caller_numpy;before['python']=caller_python
            _require_native_defaults(torch)
            runtime=_runtime();np.random.seed(seed);random.seed(seed);torch.random.default_generator.manual_seed(seed)
            enter_phase('preprocessing');transform=NodeTransform.fit(request.X_fit,request.y_fit,cfg.state['feature_schema'],seed)
            F=transform.transform(request.X_fit);G=transform.transform(request.X_select);y=transform.target(request.y_fit);gy=transform.target(request.y_select)
            if not len(G) or len(gy)!=len(G):raise ValueError('NODE G rows differ or are empty')
            width=F.shape[1];parameters=resolved['trees']*(3*2**resolved['depth']+width*resolved['depth']+2*resolved['depth'])
            if len(F)>20000 or len(G)>20000 or width>4096 or parameters>10000000:raise ValueError('NODE finite preallocation envelope exceeded; measured admission still pending')
            enter_phase('initialization');model=_build_model(width,resolved)
            with torch.no_grad():model(torch.from_numpy(F[:min(1000,len(F))]))
            _initialized(model)
            if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise FloatingPointError('Nonfinite native NODE initialized state')
            initial=_model_state(model);optimizer=_optimizer(model,resolved);enter_phase('training')
            state=_train(model,optimizer,F,y,G,gy,resolved,cap,trace);state['initial_model']=initial
            state.update(transform=transform.to_state(),seed=seed,max_updates=cap,n_fit=len(F),n_select=len(G),
                F_identity=dict(features=_array_identity(F),target=_array_identity(_target(request.y_fit))),
                G_identity=dict(features=_array_identity(G),target=_array_identity(_target(request.y_select))),
                initialization=dict(requested_rows=1000,actual_rows=min(1000,len(F)),ordered_indices=list(range(min(1000,len(F)))),prefix=_array_identity(F[:min(1000,len(F))])))
            state['prediction_binding']=_prediction_binding(state,cfg)
            enter_phase('identity_recheck')
            if source_identity()!=source or _runtime()!=runtime:raise ValueError('NODE source/runtime changed during fitting')
            identity=dict(source=source,runtime=runtime)
            costs,elapsed=captured_costs()
            receipt=dict(status='completed_ineligible',seed=seed,config=cfg.to_state(),resolved=resolved,phase='complete',
                elapsed_ns=elapsed,phase_costs_ns=costs,timing_scope='Through phase capture; successful receipt publication/final caller-state restoration require enclosing worker time',executed_updates=state['summary']['executed_updates'],
                trace=trace,summary=state['summary'],identity=identity)
            enter_phase('receipt_publication');_write(root/'fit-receipt.json',receipt)
        except BaseException as error:
            costs,elapsed=captured_costs()
            receipt=dict(status='failed',seed=request.seed,config=request.config,phase=phase,elapsed_ns=elapsed,phase_costs_ns=costs,
                timing_scope='Through failure capture; failure receipt publication/final caller-state restoration require enclosing worker time',
                executed_updates=sum(r['applied'] for r in trace),trace=trace,
                partial_optimizer_update_possible=any(r['optimizer_entered'] and not r['applied'] for r in trace),error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
            try:_write(root/'fit-receipt.json',receipt)
            except BaseException as publication:error.add_note(f'NODE original failure receipt publication also failed: {publication}')
            raise
        finally:
            if before is not None:_restore_rng(before,torch)
            else:np.random.set_state(caller_numpy);random.setstate(caller_python)
        self._model,self._transform,self._state,self._identity,self._config=model,transform,state,identity,cfg
        return self

    def predict(self,X,*,batch_size=None):
        if self._model is None:raise RuntimeError('NODE must be fitted')
        chunk=self._config.resolved()['validation_chunk'] if batch_size is None else batch_size
        if type(chunk) is not int or chunk<=0:raise ValueError('Invalid NODE prediction chunk')
        return self._transform.inverse(_predict_normalized(self._model,self._transform.transform(X),chunk))

    def save(self,directory):
        if self._state is None:raise RuntimeError('NODE must be fitted')
        root=Path(directory);root.mkdir(parents=True,exist_ok=False);torch=_native()[0];before=_rng(torch)
        try:
            if self._identity!=dict(source=source_identity(),runtime=_runtime()):raise ValueError('NODE fitted source/runtime changed')
            candidate,_=_validate_endpoint(self._state,self._config)
            if not _same_model(candidate.state_dict(),self._model.state_dict()):raise ValueError('NODE live predictor differs from retained selected state')
            arrays={};payload=_pack(self._state,arrays)
            np.savez(root/'arrays.npz',**arrays)
            with (root/'arrays.npz').open('rb') as stream:os.fsync(stream.fileno())
            if self._identity!=dict(source=source_identity(),runtime=_runtime()):raise ValueError('NODE source/runtime changed during save')
            manifest=dict(schema='research/v2-node-snapshot',config=self._config.to_state(),resolved=self._config.resolved(),identity=self._identity,
                payload=payload,arrays={k:_array_identity(v) for k,v in arrays.items()},array_file=dict(path='arrays.npz',sha256=_sha(root/'arrays.npz')))
            _write(root/'manifest.json',manifest)
        finally:_restore_rng(before,torch)

    @classmethod
    def load(cls,directory,*,expected_manifest_sha256):
        root=Path(directory);manifest=root/'manifest.json'
        if (root.is_symlink() or manifest.is_symlink() or not manifest.is_file() or not _digest_string(expected_manifest_sha256)
                or _sha(manifest)!=expected_manifest_sha256):raise ValueError('External NODE snapshot commitment differs')
        value=_read(manifest)
        fields={'schema','config','resolved','identity','payload','arrays','array_file'}
        if type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-node-snapshot':raise ValueError('Invalid NODE snapshot schema')
        cfg=NodeV2Config.parse(value['config'])
        if _canonical(value['resolved'])!=_canonical(cfg.resolved()):raise ValueError('NODE resolved recipe differs')
        caller_numpy=np.random.get_state();caller_python=random.getstate();torch=None;before=None
        try:
            source=source_identity();torch=_native()[0];before=_rng(torch);before['numpy']=caller_numpy;before['python']=caller_python
            identity=dict(source=source,runtime=_runtime())
            if _canonical(value['identity'])!=_canonical(identity):raise ValueError('NODE snapshot source/runtime binding differs')
            path=root/'arrays.npz';reference=value['array_file']
            if (type(reference) is not dict or set(reference)!={'path','sha256'} or reference['path']!='arrays.npz'
                    or not _digest_string(reference['sha256']) or path.is_symlink() or not path.is_file() or _sha(path)!=reference['sha256']
                    or set(p.name for p in root.iterdir())!={'manifest.json','arrays.npz'}):raise ValueError('NODE snapshot file inventory differs')
            if type(value['arrays']) is not dict:raise ValueError('Invalid NODE array declarations')
            with np.load(path,allow_pickle=False) as archive:
                if len(archive.files)!=len(set(archive.files)) or set(archive.files)!=set(value['arrays']):raise ValueError('NODE native array inventory differs')
                arrays={name:archive[name].copy() for name in archive.files}
            for name,array in arrays.items():
                if array.dtype.kind not in 'fiub' or not np.isfinite(array).all() or _array_identity(array)!=value['arrays'][name]:raise ValueError('NODE array dtype/finiteness/commitment differs')
            seen=set();state=_unpack(value['payload'],arrays,seen)
            if seen!=set(arrays):raise ValueError('Unreferenced NODE native array')
            model,transform=_validate_endpoint(state,cfg)
            if identity!=dict(source=source_identity(),runtime=_runtime()):raise ValueError('NODE source/runtime changed during restore')
            result=cls();result._model=model;result._transform=transform;result._state=state;result._identity=identity;result._config=cfg
            return result
        finally:
            if before is not None:_restore_rng(before,torch)
            else:np.random.set_state(caller_numpy);random.setstate(caller_python)

    def resource_summary(self):
        summary={} if self._state is None else copy.deepcopy(self._state['summary'])
        return dict(summary,method='P12',fitted=self._state is not None,full_workload_qualified=False,
            config=None if self._config is None else self._config.to_state(),
            encoded_width=None if self._transform is None else self._transform.state['width'],
            backend='native_cpu_float32',output_dtype='float64',output_units='original_target',
            selected_optimizer_semantics='newest raw member metadata, never optimization from averaged weights',
            recovery='prediction restoration only; no interrupted-training resume',Gate_D='pending_native_anchor_resource_and_workload_qualification')


def _pack(value,arrays):
    torch=_native()[0]
    if isinstance(value,(np.ndarray,torch.Tensor)):
        tensor=isinstance(value,torch.Tensor);array=value.detach().cpu().numpy().copy() if tensor else value.copy()
        key=f'a{len(arrays)}';arrays[key]=array
        return dict(array=key,tensor=tensor)
    if type(value) is dict:return {'mapping':[[_pack(k,arrays),_pack(value[k],arrays)] for k in sorted(value,key=lambda k:(type(k).__name__,k))]}
    if type(value) in (list,tuple):return {'tuple' if type(value) is tuple else 'list':[_pack(v,arrays) for v in value]}
    if value is None or type(value) in (str,bool,int) or type(value) is float and math.isfinite(value):return value
    raise ValueError('Unsupported/nonfinite portable NODE value')


def _unpack(value,arrays,seen):
    if type(value) is not dict:
        if value is None or type(value) in (str,bool,int) or type(value) is float and math.isfinite(value):return value
        raise ValueError('Invalid portable NODE scalar')
    fields=set(value)
    if fields=={'array','tensor'}:
        key=value['array']
        if type(key) is not str or key not in arrays or key in seen or type(value['tensor']) is not bool:raise ValueError('Invalid/duplicate NODE array reference')
        seen.add(key);array=arrays[key].copy()
        return _native()[0].from_numpy(array) if value['tensor'] else array
    if fields=={'mapping'}:
        if type(value['mapping']) is not list:raise ValueError('Invalid NODE mapping')
        result={}
        for pair in value['mapping']:
            if type(pair) is not list or len(pair)!=2:raise ValueError('Invalid NODE mapping entry')
            key=_unpack(pair[0],arrays,seen)
            if type(key) not in (str,int) or key in result:raise ValueError('Invalid/duplicate NODE mapping key')
            result[key]=_unpack(pair[1],arrays,seen)
        return result
    if fields in ({'list'},{'tuple'}):
        tag=next(iter(fields))
        if type(value[tag]) is not list:raise ValueError('Invalid NODE sequence')
        result=[_unpack(v,arrays,seen) for v in value[tag]]
        return tuple(result) if tag=='tuple' else result
    raise ValueError('Unknown portable NODE node')


def _digest_string(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _validate_array_identity(value,dtype,shape):
    if (type(value) is not dict or set(value)!={'dtype','shape','sha256'} or value['dtype']!=dtype
            or type(value['shape']) is not list or any(type(n) is not int for n in value['shape'])
            or value['shape']!=shape or not _digest_string(value['sha256'])):raise ValueError('Invalid NODE role-array identity')


def _validate_model(state,model):
    torch=_native()[0];expected=model.state_dict()
    if type(state) is not dict or set(state)!=set(expected):raise ValueError('NODE model tensor inventory differs')
    for name,value in state.items():
        target=expected[name]
        if (not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=target.dtype
                or value.shape!=target.shape or not torch.isfinite(value).all()):raise ValueError('NODE model tensor dtype/shape/finiteness differs')
        if name.endswith('_is_initialized_tensor') and int(value.item())!=1:raise ValueError('Uninitialized NODE snapshot')
        if name.endswith('bin_codes_1hot') and not torch.equal(value,target):raise ValueError('NODE leaf-code state differs from native geometry')


def _same_model(first,second):
    torch=_native()[0]
    return set(first)==set(second) and all(first[k].dtype==second[k].dtype and torch.equal(first[k],second[k]) for k in first)


def _same_optimizer(first,second):
    left={};right={}
    if _canonical(_pack(first,left))!=_canonical(_pack(second,right)):return False
    return set(left)==set(right) and all(_array_identity(left[k])==_array_identity(right[k]) for k in left)


def _validate_optimizer(value,model,resolved,age):
    torch=_native()[0];expected=_optimizer(model,resolved).state_dict();parameters=list(model.parameters())
    if (type(value) is not dict or set(value)!={'state','param_groups'} or type(value['state']) is not dict
            or _canonical(value['param_groups'])!=_canonical(expected['param_groups'])):raise ValueError('NODE optimizer recipe/group geometry differs')
    ids={i for i,p in enumerate(parameters) if p.requires_grad}
    if set(value['state'])!=ids or any(type(i) is not int for i in value['state']):raise ValueError('NODE optimizer state inventory differs')
    weights=[0.,0.]
    for _ in range(age):weights=[1.+beta*w for beta,w in zip(resolved['optimizer']['betas'],weights)]
    for i,slot in value['state'].items():
        if type(slot) is not dict or set(slot)!={'beta1_weight','beta2_weight','exp_avg','exp_avg_sq'}:raise ValueError('Invalid QHAdam slot fields')
        for key,expected_weight in zip(('beta1_weight','beta2_weight'),weights):
            if type(slot[key]) is not float or slot[key]!=expected_weight:raise ValueError('QHAdam bias-weight age differs')
        for key in ('exp_avg','exp_avg_sq'):
            a=slot[key]
            if (not isinstance(a,torch.Tensor) or a.device.type!='cpu' or a.dtype!=torch.float32
                    or a.shape!=parameters[i].shape or not torch.isfinite(a).all()
                    or key=='exp_avg_sq' and (a<0).any()):raise ValueError('Invalid native QHAdam slot tensor')


def _validate_rng_state(value):
    torch=_native()[0]
    if type(value) is not dict or set(value)!={'numpy','python','torch'}:raise ValueError('Invalid NODE terminal RNG inventory')
    n,p,t=value['numpy'],value['python'],value['torch']
    if (type(n) is not tuple or len(n)!=5 or n[0]!='MT19937' or not isinstance(n[1],np.ndarray)
            or n[1].dtype!=np.dtype('uint32') or n[1].shape!=(624,) or type(n[2]) is not int or not 0<=n[2]<=624
            or type(n[3]) is not int or n[3] not in (0,1) or not _finite(n[4])):raise ValueError('Invalid NODE NumPy RNG state')
    if (type(p) is not tuple or len(p)!=3 or p[0]!=3 or type(p[1]) is not tuple or len(p[1])!=625
            or any(type(v) is not int or not 0<=v<2**32 for v in p[1][:-1]) or not 0<=p[1][-1]<=624
            or p[2] is not None and not _finite(p[2])):raise ValueError('Invalid NODE Python RNG state')
    if not isinstance(t,torch.Tensor) or t.dtype!=torch.uint8 or t.shape!=torch.get_rng_state().shape:raise ValueError('Invalid NODE Torch RNG state')


def _validate_endpoint(state,cfg):
    torch=_native()[0];resolved=cfg.resolved()
    fields={'initial_model','selected_model','selected_optimizer','raw_model','raw_optimizer','rolling_window','selected_window',
        'reports','trace','sampler','rng','summary','transform','seed','max_updates','n_fit','n_select','F_identity','G_identity','initialization','prediction_binding'}
    if type(state) is not dict or set(state)!=fields:raise ValueError('Invalid NODE endpoint fields')
    seed=_seed(state['seed']);cap=state['max_updates'];nf=state['n_fit'];ng=state['n_select']
    if (type(cap) is not int or cap<1 or cfg.state['mode']=='engineering_fixture' and cap>128
            or type(nf) is not int or not 2<=nf<=20000 or type(ng) is not int or not 1<=ng<=20000):raise ValueError('Invalid NODE endpoint workload')
    transform=NodeTransform.from_state(state['transform']);width=transform.state['width']
    if (transform.state['seed']!=seed or transform.state['n_fit']!=nf or transform.state['schema']!=cfg.state['feature_schema']
            or width>4096 or resolved['trees']*(3*2**resolved['depth']+width*resolved['depth']+2*resolved['depth'])>10000000):raise ValueError('NODE config/transform workload differs')
    for role,rows in (('F',nf),('G',ng)):
        identity=state[role+'_identity']
        if type(identity) is not dict or set(identity)!={'features','target'}:raise ValueError('Invalid NODE role identity')
        _validate_array_identity(identity['features'],'float32',[rows,width]);_validate_array_identity(identity['target'],'float64',[rows])
    init=state['initialization'];count=min(1000,nf)
    if (type(init) is not dict or set(init)!={'requested_rows','actual_rows','ordered_indices','prefix'} or type(init['requested_rows']) is not int
            or init['requested_rows']!=1000 or type(init['actual_rows']) is not int or init['actual_rows']!=count
            or type(init['ordered_indices']) is not list or any(type(i) is not int for i in init['ordered_indices'])
            or init['ordered_indices']!=list(range(count))):raise ValueError('NODE initialization prefix differs')
    _validate_array_identity(init['prefix'],'float32',[count,width])
    if nf<=1000 and init['prefix']!=state['F_identity']['features']:raise ValueError('NODE full F initialization commitment differs')
    model=_build_model(width,resolved)
    for name in ('initial_model','raw_model'):_validate_model(state[name],model)
    summary=state['summary'];trace=state['trace'];reports=state['reports']
    if type(summary) is not dict or type(trace) is not list or not trace or len(trace)>cap or type(reports) is not list:raise ValueError('Invalid NODE training evidence')
    executed=len(trace);gradient_names={name for name,p in model.named_parameters() if p.requires_grad};cursor=0;epoch=0;seen=[]
    for step,row in enumerate(trace,1):
        if (type(row) is not dict or set(row)!={'step','epoch','rows','loss','gradient_norms','optimizer_entered','applied','elapsed_ns'}
                or type(row['step']) is not int or row['step']!=step or type(row['epoch']) is not int or row['epoch']!=epoch
                or type(row['rows']) is not list or len(row['rows'])!=min(resolved['batch_size'],nf-cursor)
                or any(type(i) is not int or not 0<=i<nf for i in row['rows']) or not _finite(row['loss']) or row['loss']<0
                or row['optimizer_entered'] is not True or row['applied'] is not True or type(row['elapsed_ns']) is not int or row['elapsed_ns']<=0
                or type(row['gradient_norms']) is not dict or set(row['gradient_norms'])!=gradient_names
                or any(not _finite(v) or v<0 for v in row['gradient_norms'].values())):raise ValueError('Invalid NODE paid row/update trace')
        seen+=row['rows'];cursor+=len(row['rows'])
        if len(set(seen))!=len(seen):raise ValueError('Duplicate NODE training row within shuffled epoch')
        if cursor==nf and step<executed:cursor=0;epoch+=1;seen=[]
    sampler=state['sampler']
    if (type(sampler) is not dict or set(sampler)!={'permutation','cursor','epoch'} or not isinstance(sampler['permutation'],np.ndarray)
            or sampler['permutation'].dtype!=np.dtype('int64') or sampler['permutation'].shape!=(nf,)
            or not np.array_equal(np.sort(sampler['permutation']),np.arange(nf,dtype=np.int64))
            or type(sampler['cursor']) is not int or sampler['cursor']!=cursor or type(sampler['epoch']) is not int or sampler['epoch']!=epoch
            or sampler['permutation'][:cursor].tolist()!=seen):raise ValueError('NODE sampler endpoint does not match paid trace')
    expected_steps=list(range(resolved['report_frequency'],executed+1,resolved['report_frequency']))
    if len(reports)!=len(expected_steps):raise ValueError('Missing or additional NODE G report')
    winner=None;best=float('inf');best_step=0;first_stop=None;report_by_step={}
    for expected_step,row in zip(expected_steps,reports):
        steps=list(range(max(resolved['report_frequency'],expected_step-4*resolved['report_frequency']),expected_step+1,resolved['report_frequency']))
        if (type(row) is not dict or set(row)!={'step','g_mse','snapshot_steps'} or type(row['step']) is not int or row['step']!=expected_step
                or not _finite(row['g_mse']) or row['g_mse']<0 or type(row['snapshot_steps']) is not list
                or any(type(i) is not int for i in row['snapshot_steps']) or row['snapshot_steps']!=steps):raise ValueError('Invalid NODE native G/window evidence')
        report_by_step[expected_step]=row
    for step in range(1,executed+1):
        row=report_by_step.get(step)
        if row is not None and row['g_mse']<best:best=row['g_mse'];best_step=step;winner=row
        if _patience_reached(step,best_step,resolved['patience']):first_stop=step;break
    if first_stop is not None and first_stop!=executed:raise ValueError('NODE trace continued past native patience')
    if first_stop is None and executed!=cap:raise ValueError('NODE stopped before its declared cap/patience')
    expected_summary=dict(executed_updates=executed,selected_step=winner['step'] if winner is not None else None,
        stop_reason='G_patience' if first_stop is not None else 'max_updates',checkpoint_available=winner is not None,
        selection_status='G_selected' if winner is not None else 'raw_unselected_at_cap',quality_eligible=False,native_full_anchor_qualified=False)
    if _canonical(summary)!=_canonical(expected_summary):raise ValueError('NODE checkpoint/eligibility/stop summary differs from evidence')
    registry={}
    for name,expected_window in (('rolling_window',reports[-1]['snapshot_steps'] if reports else []),('selected_window',winner['snapshot_steps'] if winner else [])):
        window=state[name]
        if type(window) is not list or len(window)!=len(expected_window):raise ValueError('NODE retained window inventory differs')
        for row,step in zip(window,expected_window):
            if type(row) is not dict or set(row)!={'step','model'} or type(row['step']) is not int or row['step']!=step:raise ValueError('NODE retained report-window steps differ')
            _validate_model(row['model'],model)
            if step in registry and not _same_model(registry[step],row['model']):raise ValueError('Same NODE report has conflicting retained tensors')
            registry[step]=row['model']
    if winner is None:
        if state['selected_model'] is not None or state['selected_optimizer'] is not None:raise ValueError('Fake NODE G selection before first report')
    else:
        _validate_model(state['selected_model'],model)
        if not _same_model(state['selected_model'],_average(state['selected_window'])):raise ValueError('NODE selected state differs from native rolling average')
        _validate_optimizer(state['selected_optimizer'],model,resolved,winner['step'])
    if reports and executed==reports[-1]['step'] and not _same_model(state['raw_model'],state['rolling_window'][-1]['model']):raise ValueError('NODE final raw report state differs')
    _validate_optimizer(state['raw_optimizer'],model,resolved,executed)
    if winner is not None and winner['step']==executed and not _same_optimizer(state['selected_optimizer'],state['raw_optimizer']):
        raise ValueError('NODE same-step selected/raw optimizer bytes differ')
    _validate_rng_state(state['rng'])
    if not _digest_string(state['prediction_binding']) or state['prediction_binding']!=_prediction_binding(state,cfg):
        raise ValueError('NODE retained prediction/transform/role binding differs')
    model.load_state_dict(state['selected_model'] if winner is not None else state['raw_model'])
    for layer in model[0]:layer._is_initialized_bool=True
    _initialized(model);model.eval()
    return model,transform


def _prediction_binding(state,cfg):
    arrays={}
    value=dict(config=cfg.to_state(),transform=state['transform'],F_identity=state['F_identity'],G_identity=state['G_identity'],
        seed=state['seed'],initialization=state['initialization'],selected_step=state['summary']['selected_step'],
        prediction_model=state['selected_model'] if state['selected_model'] is not None else state['raw_model'])
    payload=_pack(value,arrays)
    return _digest(dict(payload=payload,arrays={k:_array_identity(v) for k,v in arrays.items()}))
