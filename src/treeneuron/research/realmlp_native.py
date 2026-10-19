"""Pinned official RealMLP regression with an F-only category converter.

Native output normalization/clipping already fit F. Portable graph snapshots,
production workload qualification and interrupted training recovery are gated.
Author source/license bytes are under locks/research-v2/realmlp.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np

from treeneuron.research.native_trees import _schema, _frame, _category, _number, _target, _sha, _canonical
from treeneuron.research.native_trees import _write, _failure_domain, _peak_rss

_ROOT = Path(__file__).resolve().parents[3]
_THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
_IMPORT_CONTEXT = None


@dataclass(frozen=True)
class RealMLPConfig:
    mode: str
    feature_schema: dict[str, Any]
    parameters: dict[str, Any]
    max_epochs: int

    @classmethod
    def parse(cls, value):
        fields = {'schema','family','mode','feature_schema','parameters'}
        if (type(value) is not dict or value.get('schema') != 'research/v2-realmlp-config'
                or value.get('family') != 'P06'
                or value.get('mode') not in ('native_anchor','search','engineering_fixture')):
            raise ValueError('Invalid versioned native RealMLP configuration')
        fixture = value['mode'] == 'engineering_fixture'
        if set(value) != fields | ({'max_epochs'} if fixture else set()):
            raise ValueError('Unknown RealMLP setting or implicit native budget override')
        epochs = value['max_epochs'] if fixture else 256
        if fixture and (type(epochs) is not int or not 1 <= epochs <= 3):
            raise ValueError('Engineering fixtures require at most3 epochs')
        params = value['parameters']
        if type(params) is not dict or value['mode'] == 'native_anchor' and params:
            raise ValueError('Native TD anchor cannot override published settings')
        allowed = {'num_emb_type','add_front_scale','lr','p_drop','wd','plr_sigma','act','hidden_sizes'}
        for name, item in params.items():
            if name not in allowed:
                raise ValueError(f'Unknown or reserved native RealMLP setting: {name}')
            if name == 'num_emb_type':
                valid = type(item) is str and item in ('none','pbld','pl','plr')
            elif name == 'act':
                valid = type(item) is str and item in ('relu','selu','mish')
            elif name == 'add_front_scale':
                valid = type(item) is bool
            elif name == 'hidden_sizes':
                valid = type(item) is list and item in ([256]*3,[64]*5,[512]) and all(type(x) is int for x in item)
            elif name in ('p_drop','wd'):
                valid = type(item) in (int,float) and item in ((0.,.15,.3) if name == 'p_drop' else (0.,.02))
            else:
                valid = _number(item, *({'lr':(.02,.3),'plr_sigma':(.05,.5)}[name]))
            if not valid:
                raise ValueError(f'RealMLP setting outside frozen support: {name}')
        return cls(value['mode'], _schema(value['feature_schema']), json.loads(json.dumps(params)), epochs)

    def to_state(self):
        value = dict(schema='research/v2-realmlp-config',family='P06',mode=self.mode,
            feature_schema=self.feature_schema,parameters=self.parameters)
        if self.mode == 'engineering_fixture':
            value['max_epochs'] = self.max_epochs
        return json.loads(json.dumps(value))


def _source():
    files = {str(p.relative_to(_ROOT)):_sha(p) for p in sorted((_ROOT/'src/treeneuron').rglob('*.py'))}
    pins = _ROOT/'locks/research-v2/realmlp'
    lock = json.loads((pins/'source-lock.json').read_text())
    for row in lock['sources']:
        path = pins/row['local'];data = path.read_bytes()
        if (_sha(path) != row['sha256'] or hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() != row['git_blob_sha1']):
            raise ValueError('Pinned official RealMLP source bytes changed')
    for path in sorted(pins.rglob('*')):
        if path.is_file():files[str(path.relative_to(_ROOT))] = _sha(path)
    for name in ('locks/research-v2/core.json','uv.lock'):
        files[name] = _sha(_ROOT/name)
    return dict(package_sha256=_canonical(files),files=files)


def _dependencies():
    """Bind exact core, every private installed byte and actual selected imports."""
    if not sys.dont_write_bytecode:
        raise ValueError('RealMLP requires bytecode writes disabled before native imports: PYTHONDONTWRITEBYTECODE=1 or -B')
    from treeneuron.research.ordinary_neural import _record_file
    global _IMPORT_CONTEXT
    if any(os.environ.get(x) != '1' for x in _THREADS):
        raise ValueError('All numerical thread variables must be1 before native imports')
    before = os.environ.get('KMP_DUPLICATE_LIB_OK')
    if _IMPORT_CONTEXT is None and ('sklearn' in sys.modules or before is not None):
        raise ValueError('Use a fresh worker without caller OpenMP suppression or preimported sklearn')
    pins = _ROOT/'locks/research-v2/realmlp'
    core = json.loads((_ROOT/'locks/research-v2/core.json').read_text())
    wheels = json.loads((pins/'dependency-availability.json').read_text())['wheels']
    installed = json.loads((pins/'installed-files.json').read_text())
    overlay = (_ROOT/'environment/research-v2-realmlp-site').resolve()
    actual = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata['Name'].lower().replace('_','-')
        if name in actual:raise ValueError(f'Duplicate installed distribution: {name}')
        actual[name] = dist
    expected = {x['name'].lower().replace('_','-'):x for x in core['distributions']}
    extras = {x['package']:x for x in wheels}
    if set(actual) != set(expected)|set(extras):
        raise ValueError('RealMLP requires exactly the27 qualified core and15 private distributions')
    for name,row in expected.items():
        dist = actual[name]
        if (dist.version != row['version'] or str(Path(dist._path).resolve()) != row['path']
                or _sha(Path(dist._path)/'RECORD') != row['record_sha256']):
            raise ValueError(f'Frozen core distribution changed: {name}')
    owned = set(installed['ignored_installer_files'])
    owners = {}
    for name,row in extras.items():
        dist = actual[name];path = Path(dist._path).resolve()
        entries = installed['distributions'].get(path.name)
        if (dist.version != row['version'] or path.parent != overlay or entries is None
                or _sha(path/'RECORD') != entries['record_sha256']
                or _sha(path/'METADATA') != row['metadata_sha256']):
            raise ValueError('RealMLP private distribution identity/metadata/RECORD changed')
        for entry in entries['files']:
            file = (overlay/entry['path']).resolve()
            if (not file.is_relative_to(overlay) or not file.is_file()
                    or file.stat().st_size != entry['bytes'] or _sha(file) != entry['sha256']):
                raise ValueError('RealMLP private installed bytes differ from the pinned installation')
            owned.add(entry['path']);owners[file] = name
    if {str(p.relative_to(overlay)) for p in overlay.rglob('*') if p.is_file()} != owned:
        raise ValueError('Unowned/missing files in the strict RealMLP overlay')
    if importlib.util.find_spec('lightning') is not None:
        raise ValueError('Unexpected lightning umbrella import; require the qualified pytorch_lightning branch')
    imported, files = {}, []
    for package, distribution in (('torch','torch'),('numpy','numpy'),('sklearn','scikit-learn'),
            ('pytorch_lightning','pytorch-lightning'),('torchmetrics','torchmetrics'),('psutil','psutil'),
            ('pytabkit','pytabkit')):
        origin = Path(actual[distribution].locate_file(package+'/__init__.py')).resolve()
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != origin:
            raise ValueError(f'Native module origin differs from its qualified installation: {package}')
        files.append(_record_file(distribution,origin))
        module = importlib.import_module(package)
        if Path(module.__file__).resolve() != origin:
            raise ValueError('Native module selected by import escaped its qualified origin')
        imported[package] = module
    # Bind every private module actually selected by the native import stack.
    selected = {}
    for name,module in list(sys.modules.items()):
        path = getattr(module,'__file__',None)
        if path:
            path = Path(path).resolve()
            if path.is_relative_to(overlay):
                if path not in owners:raise ValueError('Imported private module lacks locked installed bytes')
                selected[str(path)] = _record_file(owners[path],path)
    files.extend(selected[k] for k in sorted(selected))
    torch = imported['torch'];torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:torch.set_num_interop_threads(1)
    files.extend([_record_file('torch',torch._C.__file__),
        _record_file('numpy',importlib.import_module('numpy._core._multiarray_umath').__file__),
        _record_file('scikit-learn',importlib.import_module('sklearn.preprocessing._data').__file__)])
    if _IMPORT_CONTEXT is None:
        _IMPORT_CONTEXT = dict(caller_KMP_DUPLICATE_LIB_OK=before,
            post_import_KMP_DUPLICATE_LIB_OK=os.environ.get('KMP_DUPLICATE_LIB_OK'),
            upstream_policy='Pinned sklearn/threadpoolctl use setdefault; adapter never sets/removes/preloads it',
            compatibility='Bounded execution under upstream suppression does not assure unsuppressed compatibility')
    elif before != _IMPORT_CONTEXT['post_import_KMP_DUPLICATE_LIB_OK']:
        raise ValueError('Native import environment changed after qualification')
    return dict(core_manifest_sha256=_sha(_ROOT/'locks/research-v2/core.json'),
        overlay_manifest_sha256=_sha(pins/'installed-files.json'),overlay=str(overlay),
        executable=sys.executable,executable_sha256=_sha(sys.executable),python=sys.version,platform=platform.platform(),
        core_distribution_count=27,overlay_distribution_count=15,bytecode_writes_disabled=True,
        package_versions={k:d.version for k,d in sorted(actual.items())},imported_files=files,
        import_environment=copy.deepcopy(_IMPORT_CONTEXT),
        validation_scope='All27 core identities/RECORD digests, every private byte and actual selected private/core/native import files')


def _runtime():
    from treeneuron.research.ordinary_neural import _runtime as inspect
    return inspect()


def native_config(config,seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Native RealMLP seed requires a coordinator collision-checked32-bit integer')
    _dependencies()
    from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Regressor
    factory = RealMLP_TD_Regressor(device='cpu',random_state=seed,n_cv=1,n_refit=0,n_repeats=1,
        n_threads=1,verbosity=0,n_epochs=config.max_epochs,batch_size=256,predict_batch_size=1024,
        n_ens=1,train_metric_name='mse',val_metric_name='rmse',use_early_stopping=False,
        **config.parameters)
    result = factory.get_config()
    # These are the pinned downstream defaults omitted by the public constructor.
    result.update(use_best_epoch=True,use_last_best_epoch=True,adjust_bs=False)
    return json.loads(json.dumps(result))


def native_frame(values,schema):
    """Keep actual category missingness for native OrdinalEncoder code zero."""
    import pandas as pd
    frame=_frame(values,schema)
    def category(value):
        missing=pd.isna(value)
        if not isinstance(missing,(bool,np.bool_)):raise ValueError('Categories must be scalar values')
        return np.nan if missing else _category(value)
    for name in schema['columns']:
        if name in schema['categorical']:
            # Object dtype preserves np.nan, rather than converting it to a token
            # or pandas.NA that this pinned sklearn boundary cannot compare.
            frame[name]=frame[name].map(category).astype(object)
        else:
            frame[name]=frame[name].astype(np.float32)
            if not np.isfinite(frame[name].to_numpy()).all():
                raise ValueError('RealMLP numeric values must remain finite in float32')
    return frame


def prepare_native_data(XF,yF,XG,yG,config,seed):
    _dependencies()
    import torch
    from pytabkit.models.data.conversion import ToDictDatasetConverter
    from pytabkit.models.data.data import DictDataset,TensorInfo
    from pytabkit.models.alg_interfaces.base import SplitIdxs
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Native RealMLP seed requires a coordinator collision-checked32-bit integer')
    if not 2 <= len(XF) <= 20000 or not 1 <= len(XG) <= 20000 or len(config.feature_schema['columns']) > 128:
        raise ValueError('Inputs exceed the finite native RealMLP safety envelope')
    F,G = native_frame(XF,config.feature_schema),native_frame(XG,config.feature_schema)
    categorical = [name in config.feature_schema['categorical'] for name in config.feature_schema['columns']]
    ftarget = _target(yF,len(F)).astype(np.float32)
    gtarget = _target(yG,len(G)).astype(np.float32)
    if not np.isfinite(ftarget).all() or not np.isfinite(gtarget).all():
        raise ValueError('RealMLP targets must remain finite in float32')
    converter = ToDictDatasetConverter(cat_features=categorical,verbosity=0)
    fdata,gdata = converter.fit_transform(F),None
    gdata = converter.transform(G)
    tensors = {name:torch.cat([fdata.tensors[name],gdata.tensors[name]],dim=0) for name in fdata.tensors}
    tensors['y'] = torch.from_numpy(np.concatenate([ftarget,gtarget])[:,None])
    infos = fdata.tensor_infos|{'y':TensorInfo(feat_shape=[1])}
    ds = DictDataset(tensors=tensors,tensor_infos=infos)
    subseed = int(np.random.RandomState(seed).randint(0,2**31-1))
    split = SplitIdxs(torch.arange(len(F))[None],torch.arange(len(F),len(F)+len(G))[None],None,
        split_seed=seed,sub_split_seeds=[subseed],split_id=0)
    return dict(dataset=ds,split=split,converter=converter,schema=config.feature_schema,
        cardinalities=fdata.tensor_infos['x_cat'].get_cat_sizes().tolist(),
        F_cat_codes=fdata.tensors['x_cat'].tolist(),G_cat_codes=gdata.tensors['x_cat'].tolist(),
        target_precision='float32',accessed_roles=['F','G'],native_seed=seed,native_subseed=subseed)


def check_raw_G(parts):
    """Reject native cached predictions before Metrics.apply can sanitize them."""
    import torch
    if not parts or any(not torch.isfinite(part).all() for part in parts):
        raise FloatingPointError('Nonfinite or absent raw G predictions before native metric evaluation')


def _fit_native(prepared,parameters):
    """Use the official trainer, adding observation-only callbacks/row tracing."""
    import torch
    import pytorch_lightning as pl
    from pytabkit.models.alg_interfaces import nn_interfaces
    from pytabkit.models.alg_interfaces.base import InterfaceResources
    from pytabkit.models.training.lightning_modules import TabNNModule
    from pytabkit.models.training.logging import StdoutLogger
    from pytabkit.models.training.metrics import Metrics,mse
    from pytabkit.models.data.data import DictDataset
    from pytabkit.models.nn_models.base import Fitter
    from pytabkit.models.training.nn_creator import NNCreator
    from treeneuron.research.realmlp_initialization import NativeInitializationTrace
    nF=prepared['split'].n_train
    n_batches=nF//min(256,nF)

    class Observer(pl.Callback):
        def __init__(self):
            self.batches=[];self.history=[];self.losses=[];self.updates=0;self.restorations=[]

        def on_fit_start(self,trainer,module):
            loader=module.train_dl
            if loader.drop_last is not True or loader.adjust_bs is not False or loader.specified_batch_size!=256:
                raise ValueError('Native RealMLP sampler differs from the pinned contract')
            old=loader.ds.get_batch
            def observe(idxs):
                values=idxs.detach().cpu().numpy().copy()
                if values.shape!=(1,min(256,nF)) or np.any(values<0) or np.any(values>=nF):
                    raise ValueError('Native training loader selected rows outside F')
                self.batches.append(values[0].tolist())
                return old(idxs)
            loader.ds.get_batch=observe
            self.restorations.append((loader.ds,old))

        def on_before_backward(self,trainer,module,loss):
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native RealMLP training loss')
            self.losses.append(float(loss.detach()))

        def on_after_backward(self,trainer,module):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.model.parameters()):
                raise FloatingPointError('Nonfinite native RealMLP gradient')

        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            if any(not torch.isfinite(p).all() for p in module.model.parameters()):
                raise FloatingPointError('Nonfinite native RealMLP updated parameter')
            self.updates+=1

        def on_validation_epoch_end(self,trainer,module):
            # Lightning invokes callback hooks before the module hook, which
            # calls Metrics.apply and replaces nonfinite predictions with zero.
            check_raw_G(module.val_preds)

        def on_validation_end(self,trainer,module):
            epoch=module.progress.epoch
            if self.updates!=epoch*n_batches or len(self.batches)!=self.updates:
                raise ValueError('Native RealMLP sampler/update observations disagree')
            prediction=module._postprocess_ens_pred(torch.cat(module.val_preds,dim=-2))
            risk=float(mse(prediction[0],module.val_dl.val_y[0]))
            native_risk=float(Metrics.apply(prediction[0],module.val_dl.val_y[0],'rmse'))
            if not all(math.isfinite(value) and value>=0 for value in (risk,native_risk)):
                raise FloatingPointError('Nonfinite native RealMLP G risk')
            self.history.append(dict(epoch=epoch,G_mse=risk,G_native_rmse=native_risk,applied_updates=self.updates,
                F_online_native_loss=float(np.mean(self.losses[-n_batches:]))))

        def restore(self):
            for ds,old in reversed(self.restorations):ds.get_batch=old
            self.restorations=[]

    observer=Observer()
    original=TabNNModule.create_callbacks
    def observed_callbacks(module):
        return [*original(module),observer]
    # Only a fresh method worker executes this scoped observation hook. All native
    # initialization/optimization/sampling/checkpoint code remains the pinned code.
    TabNNModule.create_callbacks=observed_callbacks
    interface=nn_interfaces.NNAlgInterface(**parameters)
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    initialization=NativeInitializationTrace(n_fit=nF,dataset_class=DictDataset,
        fitter_class=Fitter,creator_class=NNCreator)
    try:
        with initialization:
            interface.fit(prepared['dataset'],[prepared['split']],InterfaceResources(1,[]),
                StdoutLogger(0),[None],name='P06_native_RealMLP_fixture')
    except BaseException as error:
        visited=set(sum(observer.batches,[]))
        error.realmlp_observation=dict(F_row_batches=copy.deepcopy(observer.batches),
            applied_updates=observer.updates,history=copy.deepcopy(observer.history),
            finite_F_online_losses=list(observer.losses),fit_rows=nF,
            initialization_observation=initialization.snapshot(),
            unvisited_F_row_ids=sorted(set(range(nF))-visited))
        raise
    finally:
        observer.restore()
        TabNNModule.create_callbacks=original
        torch.backends.cuda.matmul.allow_tf32=old_tf32
    observer.initialization=initialization.snapshot()
    return interface,observer


def _graph_inspection(model):
    """Diagnostic only: enumerate native types/attributes before representation design."""
    import torch
    rows=[]
    for name,module in model.named_modules():
        values={}
        for key,value in vars(module).items():
            if key.startswith('_') or key=='training':continue
            if value is None or type(value) in (str,int,float,bool):values[key]=value
            elif callable(value):
                code=getattr(value,'__code__',None)
                values[key]=dict(callable_module=getattr(value,'__module__',None),
                    callable_name=getattr(value,'__qualname__',None),
                    bytecode_sha256=hashlib.sha256(code.co_code).hexdigest() if code else None)
            elif not isinstance(value,(torch.Tensor,torch.nn.Module)):
                values[key]=dict(opaque_type=type(value).__module__+'.'+type(value).__qualname__)
        rows.append(dict(name=name,type=type(module).__module__+'.'+type(module).__qualname__,attributes=values))
    return rows


def _validate_summary(config,summary):
    if (type(summary) is not dict or summary.get('schema')!='research/v2-realmlp-resources'
            or summary.get('family')!='P06' or summary.get('config')!=config.to_state()
            or summary.get('mode')!=config.mode or summary.get('quality_eligible') is not False
            or summary.get('checkpoint_eligible') is not False
            or summary.get('status') not in ('completed_ineligible','timeout')):
        raise ValueError('Native RealMLP mode/config/eligibility contract differs')
    epochs=config.max_epochs;rows=summary.get('fit_rows');history=summary.get('history')
    if (type(rows) is not int or not 2<=rows<=20000 or type(history) is not list or len(history)!=epochs
            or summary.get('native_epochs_executed')!=epochs or summary.get('accessed_roles')!=['F','G']):
        raise ValueError('Native declared treatment/roles did not complete')
    batches=rows//min(rows,256)
    for i,row in enumerate(history,1):
        if (type(row) is not dict or set(row)!={'epoch','G_mse','G_native_rmse','applied_updates','F_online_native_loss'}
                or row['epoch']!=i or row['applied_updates']!=i*batches
                or any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) or row[k]<0
                    for k in ('G_mse','G_native_rmse','F_online_native_loss'))):
            raise ValueError('Native risk/update history is invalid')
    best=min(row['G_native_rmse'] for row in history)
    selected=max(row['epoch'] for row in history if row['G_native_rmse']==best)
    if summary.get('selected_epoch')!=selected or summary.get('applied_updates')!=epochs*batches:
        raise ValueError('Native latest-best RMSE selection/update contract differs')
    admitted=summary.get('admitted_row_ids');omitted=summary.get('omitted_row_ids')
    if type(admitted) is not list or type(omitted) is not list or len(admitted)!=epochs or len(omitted)!=epochs:
        raise ValueError('Native sampler exposure declaration differs')
    visited=set()
    for active,missing in zip(admitted,omitted,strict=True):
        if (type(active) is not list or len(active)!=batches*min(rows,256)
                or any(type(x) is not int or not 0<=x<rows for x in active) or len(set(active))!=len(active)
                or missing!=sorted(set(range(rows))-set(active))):raise ValueError('Native F row exposure differs')
        visited.update(active)
    if summary.get('unvisited_F_row_ids')!=sorted(set(range(rows))-visited) or summary.get('cumulative_visited_rows')!=len(visited):
        raise ValueError('Native cumulative F exposure differs')


class RealMLPPredictor:
    def __init__(self):
        self._interface=None;self._summary={};self._inference_graphs=None;self._fit_attempted=False

    def fit(self,request):
        if self._fit_attempted or self._interface is not None or self._inference_graphs is not None:raise RuntimeError('Use a fresh native RealMLP predictor for every fit; snapshots are inference-only')
        self._fit_attempted=True
        started=time.perf_counter_ns();root=Path(request.artifact_dir);root.mkdir(parents=True,exist_ok=False)
        phase='input_validation'
        try:
            config=RealMLPConfig.parse(request.config)
            if type(request.seed) is not int or not 0<=request.seed<2**32:
                raise ValueError('RealMLP fit requires a coordinator collision-checked32-bit native seed')
            if (type(request.budget) is not dict or not set(request.budget)<={'max_seconds'}
                    or 'max_seconds' in request.budget and not _number(request.budget['max_seconds'],1e-9,1e12)):
                raise ValueError('Require only a positive observed max_seconds budget')
            source=_source();phase='dependency_validation';import_started=time.perf_counter_ns()
            dependencies=_dependencies();import_ns=time.perf_counter_ns()-import_started
            phase='F_preprocessing'
            parameters=native_config(config,request.seed)
            prepared=prepare_native_data(request.X_fit,request.y_fit,request.X_select,request.y_select,config,request.seed)
            nF,nG=prepared['split'].n_train,prepared['split'].n_val
            if any(c>4096 for c in prepared['cardinalities']) or sum(prepared['cardinalities'])>4096:
                raise ValueError('Native category dimensions exceed the finite4096 safety envelope')
            native_tensors=prepared['dataset'].tensors
            _write(root/'input-manifest.json',dict(schema='research/v2-realmlp-attempt',source=source,
                dependencies=dependencies,config=config.to_state(),native_parameters=parameters,
                seed=request.seed,native_subseed=prepared['native_subseed'],fit_rows=nF,select_rows=nG,
                cardinalities=prepared['cardinalities'],target_precision='float32',
                accessed_roles=['F','G'],budget=request.budget,
                F_data_sha256=_canonical({k:hashlib.sha256(v[:nF].numpy().tobytes()).hexdigest() for k,v in native_tensors.items()}),
                G_data_sha256=_canonical({k:hashlib.sha256(v[nF:].numpy().tobytes()).hexdigest() for k,v in native_tensors.items()})))
            import torch
            phase='native_training';native_started=time.perf_counter_ns()
            with torch.random.fork_rng():
                interface,observation=_fit_native(prepared,parameters)
            native_ns=time.perf_counter_ns()-native_started
            if source!=_source():raise ValueError('Scientific source changed during native RealMLP fitting')
            if len(observation.history)!=config.max_epochs:
                raise ValueError('Native fixed-epoch treatment did not finish its declared horizon')
            selected=interface.fit_params[0]['stop_epoch']['rmse']
            best=min(row['G_native_rmse'] for row in observation.history)
            expected=max(row['epoch'] for row in observation.history if row['G_native_rmse']==best)
            if selected!=expected:raise ValueError('Native RealMLP last-best selection disagrees with observed G history')
            n_batches=nF//min(nF,256)
            admitted=[sum(observation.batches[i*n_batches:(i+1)*n_batches],[]) for i in range(config.max_epochs)]
            if any(len(ids)!=len(set(ids)) for ids in admitted):raise ValueError('Native epoch duplicated an F row')
            omitted=[sorted(set(range(nF))-set(ids)) for ids in admitted]
            visited=set(sum(admitted,[]))
            phase='runtime_inspection';runtime=_runtime()
            arrays={}
            for prefix,model in (('native_model',interface.model.model),('static_preprocessing',interface.model.creator.static_model)):
                for name,tensor in model.state_dict().items():
                    value=tensor.detach().cpu().numpy().copy()
                    if not np.isfinite(value).all():raise FloatingPointError('Nonfinite native fitted state')
                    arrays[prefix+'/'+name]=value
            np.savez(root/'diagnostic-state.npz',**arrays)
            _write(root/'native-graph-inspection.json',dict(schema='research/v2-realmlp-graph-inspection',
                scope='diagnostic_only_not_a_portable_snapshot_representation',
                fitted_model=_graph_inspection(interface.model.model),
                static_preprocessing=_graph_inspection(interface.model.creator.static_model)))
            elapsed=time.perf_counter_ns()-started;timeout=elapsed>request.budget.get('max_seconds',math.inf)*1e9
            summary=dict(schema='research/v2-realmlp-resources',family='P06',mode=config.mode,
                source=source,config=config.to_state(),native_parameters=parameters,dependencies=dependencies,runtime=runtime,
                seed=request.seed,native_subseed=prepared['native_subseed'],fit_rows=nF,select_rows=nG,
                selected_epoch=selected,native_epochs_executed=len(observation.history),applied_updates=observation.updates,
                history=observation.history,admitted_row_ids=admitted,omitted_row_ids=omitted,
                epoch_rows_admitted=nF,rows_per_epoch=len(admitted[0]),cumulative_visited_rows=len(visited),
                unvisited_F_row_ids=sorted(set(range(nF))-visited),sampler='native_drop_last_true_adjust_bs_false',
                native_training_wall_ns=native_ns,dependency_import_wall_ns=import_ns,total_fit_wall_ns=elapsed,
                process_peak_rss_bytes=_peak_rss(),rss_scope='whole process lifetime peak',
                time_limit_exceeded=timeout,status='timeout' if timeout else 'completed_ineligible',
                checkpoint_eligible=False,quality_eligible=False,
                Gate_D='pending_portable_representation_full_native_anchor_resource_workload_recovery_qualification',
                portable_snapshot_qualified=False,snapshot_capability='closed_native_inference_format_requires_fixture_qualification',hard_resource_enforcement='outer harness required; max_seconds observed after fit',
                interrupted_training_resume='not_implemented',accessed_roles=['F','G'],
                initialization_observation=copy.deepcopy(observation.initialization),
                diagnostic_state_sha256=_sha(root/'diagnostic-state.npz'),
                native_graph_sha256=_sha(root/'native-graph-inspection.json'),
                target_statistics_policy='F_only_in_pinned_native_NNCreator_not_an_adapter_departure')
            phase='closed_graph_binding'
            identity=_canonical(self._snapshot_parts(interface=interface,converter=prepared['converter'],config=config)[0])
            summary['selected_inference_state_sha256']=identity
            phase='receipt_publication';_write(root/'attempt.json',summary)
            self._interface,self._summary,self._config,self._converter=interface,summary,config,prepared['converter']
            self._source_bound,self._native_identity=source,identity
            return self
        except BaseException as error:
            self._interface=None;self._inference_graphs=None
            for name in ('_config','_converter','_source_bound','_native_identity'):self.__dict__.pop(name,None)
            status,domain,retry=_failure_domain(error)
            self._summary=dict(status=status,failure_domain=domain,infrastructure_retry_permitted=retry,
                exception=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
                failure_phase=phase,total_fit_wall_ns=time.perf_counter_ns()-started,
                process_peak_rss_bytes=_peak_rss(),checkpoint_eligible=False,quality_eligible=False,
                native_observation=getattr(error,'realmlp_observation',None))
            try:_write(root/('terminal-error.json' if (root/'attempt.json').exists() else 'attempt.json'),self._summary)
            except BaseException as receipt:error.add_note(f'Failure receipt publication failed: {receipt!r}')
            raise

    def predict(self,X,*,batch_size=None):
        if self._interface is None and self._inference_graphs is None:raise RuntimeError('Native RealMLP must be fitted before prediction')
        if batch_size is not None and batch_size!=1024:
            raise ValueError('Only native1024 inference batch support is qualified in this engineering slice')
        frame=native_frame(X,self._config.feature_schema)
        data=self._converter.transform(frame)
        if not len(frame):return np.empty(0,np.float64)
        import torch
        old_tf32=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            if self._inference_graphs is None:prediction=self._interface.predict(data).detach().cpu().numpy()
            else:
                from pytabkit.models.data.data import DictDataset,ParallelDictDataLoader
                static,model=self._inference_graphs
                with torch.no_grad():
                    tensors=static.forward_tensors(data.tensors)
                    transformed=DictDataset(tensors,tensor_infos={})
                    loader=ParallelDictDataLoader(transformed,torch.arange(len(frame))[None],batch_size=1024)
                    prediction=torch.cat([model.forward_tensors(batch)['x_cont'] for batch in loader],dim=-2).numpy()
        finally:torch.backends.cuda.matmul.allow_tf32=old_tf32
        if prediction.shape!=(1,len(frame),1) or not np.isfinite(prediction).all():
            raise FloatingPointError('Native RealMLP prediction shape/finiteness mismatch')
        return prediction[0,:,0].astype(np.float64)

    def resource_summary(self):
        return copy.deepcopy(self._summary)

    def _snapshot_parts(self,*,interface=None,converter=None,config=None):
        from treeneuron.research.realmlp_snapshot import encode_graph,converter_state
        arrays={}
        interface=self._interface if interface is None else interface
        converter=self._converter if converter is None else converter
        config=self._config if config is None else config
        fitted=interface.model.model
        static=interface.model.creator.static_model
        fitted.eval();static.eval()
        graph=encode_graph(fitted,'model',arrays)
        static_graph=encode_graph(static,'static',arrays)
        payload={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
        return dict(model=graph,static=static_graph,vocabulary=converter_state(converter,config.feature_schema),payload=payload),arrays

    def save(self,path):
        if self._interface is None:raise RuntimeError('Predictor must be fitted before saving; loaded snapshots are inference-only')
        if self._source_bound!=_source():raise ValueError('Source changed after native RealMLP fitting')
        state,arrays=self._snapshot_parts()
        if _canonical(state)!=self._native_identity:raise ValueError('Fitted native prediction state changed before saving')
        _validate_summary(self._config,self._summary)
        from treeneuron.research.realmlp_snapshot import validate_recipe
        validate_recipe(state['model'],state['static'],arrays,self._summary['native_parameters'],state['vocabulary'])
        root=Path(path);root.mkdir(parents=True,exist_ok=False)
        np.savez(root/'weights.npz',**arrays)
        _write(root/'manifest.json',dict(schema='research/v2-realmlp-inference-snapshot',family='P06',
            config=self._config.to_state(),source=self._source_bound,resources=self._summary,
            state=state,weights_sha256=_sha(root/'weights.npz'),
            snapshot_scope='closed native inference-only; omitted fitter/controller objects cannot resume training'))

    @classmethod
    def load(cls,path,*,expected_manifest_sha256):
        if (type(expected_manifest_sha256) is not str or len(expected_manifest_sha256)!=64
                or any(c not in '0123456789abcdef' for c in expected_manifest_sha256)):
            raise ValueError('Loading requires an external trusted manifest SHA256')
        started=time.perf_counter_ns();root=Path(path)
        if _sha(root/'manifest.json')!=expected_manifest_sha256:raise ValueError('Snapshot differs from trusted manifest')
        value=json.loads((root/'manifest.json').read_text())
        if (set(value)!={'schema','family','config','source','resources','state','weights_sha256','snapshot_scope'}
                or value['schema']!='research/v2-realmlp-inference-snapshot' or value['family']!='P06'
                or value['source']!=_source()):raise ValueError('Snapshot schema/source binding differs')
        config=RealMLPConfig.parse(value['config']);summary=value['resources']
        _validate_summary(config,summary)
        if summary['source']!=value['source']:raise ValueError('Native fitted source differs from snapshot source')
        dependencies=_dependencies();runtime=_runtime()
        retained=summary['dependencies']
        stable_fields=('core_manifest_sha256','overlay_manifest_sha256','package_versions','executable',
            'executable_sha256','python','platform','overlay','core_distribution_count','overlay_distribution_count',
            'bytecode_writes_disabled','import_environment','validation_scope')
        if any(dependencies.get(key)!=retained.get(key) for key in stable_fields):
            raise ValueError('Native fitted executable/dependency/platform identity differs from load environment')
        selected={(row['distribution'],row['path']):row for row in dependencies['imported_files']}
        if any(selected.get((row['distribution'],row['path']))!=row for row in retained['imported_files']):
            raise ValueError('Retained native imported-file provenance differs from current authentication')
        from treeneuron.research.ordinary_neural import _runtime_identity
        if _runtime_identity(runtime)!=_runtime_identity(summary['runtime']):
            raise ValueError('Retained native runtime image/RECORD/thread identity differs from load environment')
        if native_config(config,summary['seed'])!=summary['native_parameters']:
            raise ValueError('Native resolved factory configuration differs from snapshot declaration')
        state=value['state']
        if (type(state) is not dict or set(state)!={'model','static','vocabulary','payload'}
                or _canonical(state)!=summary['selected_inference_state_sha256']
                or state['vocabulary']['schema']!=config.feature_schema):
            raise ValueError('Native selected inference state/config binding differs')
        if _sha(root/'weights.npz')!=value['weights_sha256']:raise ValueError('Native tensor archive bytes differ')
        with np.load(root/'weights.npz',allow_pickle=False) as archive:
            if set(archive.files)!=set(state['payload']) or len(archive.files)!=len(set(archive.files)):
                raise ValueError('Native tensor archive keys differ')
            arrays={k:archive[k].copy() for k in archive.files}
        for key,array in arrays.items():
            expected=dict(dtype=str(array.dtype),shape=list(array.shape),sha256=hashlib.sha256(array.tobytes()).hexdigest())
            if state['payload'][key]!=expected or array.dtype!=np.float32 or not np.isfinite(array).all():
                raise ValueError('Native tensor payload metadata differs')
        from treeneuron.research.realmlp_snapshot import decode_graph,restore_converter,validate_recipe
        used=set();static=decode_graph(state['static'],arrays,used);model=decode_graph(state['model'],arrays,used)
        if used!=set(arrays):raise ValueError('Unreferenced native tensor in archive')
        validate_recipe(state['model'],state['static'],arrays,summary['native_parameters'],state['vocabulary'])
        converter=restore_converter(state['vocabulary'])
        result=cls();result._inference_graphs=(static,model);result._converter=converter
        result._config,result._summary,result._source_bound=config,copy.deepcopy(summary),value['source']
        result._summary['load_environment']=dependencies;result._summary['load_runtime']=runtime
        result._summary['load_wall_ns']=time.perf_counter_ns()-started
        result._summary['loaded_snapshot_scope']='inference_only_training_resume_rejected'
        return result

