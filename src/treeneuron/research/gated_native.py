"""Own-family F/G-only fit boundary; engineering only until workload admission.

This slice provides fitting/prediction and retains full candidate/event evidence.
Closed inference serialization and worker registration are separate requirements;
persistent interrupted-training resume is explicitly unavailable.
"""
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from treeneuron.baselines.api import FitRequest
from .gated_config import GatedConfig
from .native_trees import _target, _peak_rss, _failure_domain
from .p13_simple import _source, _preprocessing, _features, _array_identity, _write, _hash, _sha
from .seeds import seed64

_ROOT=Path(__file__).resolve().parents[3]
_THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
_IMPORT_CONTEXT=None


def _dependencies():
    global _IMPORT_CONTEXT
    from .ordinary_neural import _record_file
    if any(os.environ.get(k)!='1' for k in _THREADS):raise ValueError('Require one numerical thread before native imports')
    before=os.environ.get('KMP_DUPLICATE_LIB_OK')
    if _IMPORT_CONTEXT is None and before is not None:raise ValueError('Use a fresh own-method worker without caller OpenMP suppression')
    lock=json.loads((_ROOT/'locks/research-v2/core.json').read_text());actual={}
    for dist in importlib.metadata.distributions():
        name=dist.metadata['Name'].lower().replace('_','-')
        if name in actual:raise ValueError('Duplicate distribution identity')
        actual[name]=dist
    expected={r['name'].lower().replace('_','-'):r for r in lock['distributions']}
    if set(expected)!=set(actual):raise ValueError('Own-method worker requires exactly the qualified core')
    for name,row in expected.items():
        dist=actual[name]
        if dist.version!=row['version'] or str(Path(dist._path).resolve())!=row['path'] or _sha(Path(dist._path)/'RECORD')!=row['record_sha256']:
            raise ValueError('Frozen core distribution identity changed')
    files=[]
    for name in ('numpy','pandas','torch','threadpoolctl'):
        module=importlib.import_module(name);files.append(_record_file(name,module.__file__))
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    files.append(_record_file('torch',torch._C.__file__))
    files.append(_record_file('numpy',importlib.import_module('numpy._core._multiarray_umath').__file__))
    if _IMPORT_CONTEXT is None:
        _IMPORT_CONTEXT=dict(caller_KMP_DUPLICATE_LIB_OK=before,post_import_KMP_DUPLICATE_LIB_OK=os.environ.get('KMP_DUPLICATE_LIB_OK'),
            policy='Pinned threadpoolctl setdefault; adapter does not set/remove/preload OpenMP suppression')
    elif before!=_IMPORT_CONTEXT['post_import_KMP_DUPLICATE_LIB_OK']:
        raise ValueError('Import environment changed after qualification')
    return dict(core_manifest_sha256=_sha(_ROOT/'locks/research-v2/core.json'),executable=sys.executable,
        executable_sha256=_sha(sys.executable),package_versions={k:d.version for k,d in sorted(actual.items())},
        imported_files=files,import_environment=copy.deepcopy(_IMPORT_CONTEXT),
        validation_scope='All core versions/locations/RECORDs and selected imported bytes; runtime images checked separately')


def _state_identity(model):return _hash({k:_array_identity(v.detach().cpu().numpy()) for k,v in model.state_dict().items()})


def _matrix(X,schema,preprocessing):
    encoded,_=_features(X,schema,preprocessing)
    if not 1<=encoded.shape[1]<=128:raise ValueError('Transformed workload exceeds the separate v2 envelope')
    converted=encoded.astype(np.float32)
    if not np.isfinite(converted).all():raise ValueError('Encoded data is not finite in training float32')
    return encoded,converted


class GatedPredictor:
    def __init__(self,family):
        if family not in ('P00','P01','P02','P05'):raise ValueError('Unknown own-method family')
        self.family=family;self._model=None;self._summary={}

    def fit(self,request):
        if self._model is not None:raise RuntimeError('A fresh own-method predictor is required for each fit')
        if not isinstance(request,FitRequest):raise ValueError('Require the F/G-only FitRequest')
        started=time.perf_counter_ns();root=Path(request.artifact_dir);root.mkdir(parents=True,exist_ok=False)
        phase='input_validation'
        try:
            config=GatedConfig.parse(self.family,request.config);recipe=config.resolved()
            if type(request.seed) is not int or not 0<=request.seed<2**63:raise ValueError('Invalid fit seed')
            if (type(request.budget) is not dict or not set(request.budget)<={'max_seconds'}
                    or 'max_seconds' in request.budget and (type(request.budget['max_seconds']) not in (int,float)
                    or not 0<request.budget['max_seconds']<=7200)):
                raise ValueError('Only the declared positive worker observation budget is accepted')
            if not 2<=len(request.X_fit)<=20000 or not 1<=len(request.X_select)<=20000 or len(config.feature_schema['columns'])>128:
                raise ValueError('Raw F/G workload exceeds the separate v2 envelope')
            yF=_target(request.y_fit,len(request.X_fit));yG=_target(request.y_select,len(request.X_select))
            source=_source();phase='dependency_validation';deps=_dependencies()
            phase='F_preprocessing';preprocessing=_preprocessing(request.X_fit,yF,config.feature_schema)
            F64,F=_matrix(request.X_fit,config.feature_schema,preprocessing)
            _,G=_matrix(request.X_select,config.feature_schema,preprocessing)
            scaled=((yF-preprocessing['target_mean'])/preprocessing['target_scale']).astype(np.float32)
            if not np.isfinite(scaled).all():raise ValueError('F target cannot be represented in float32')
            _write(root/'input-manifest.json',dict(schema='research/v2-gated-fit-input',source=source,dependencies=deps,
                config=config.to_state(),resolved_parameters=recipe,seed=request.seed,preprocessing=preprocessing,
                F_features=_array_identity(F64),G_features=_array_identity(G),F_target=_array_identity(yF),
                G_target=_array_identity(yG),accessed_roles=['F','G']))
            import torch
            from treeneuron.train import capture_rng,restore_rng
            from treeneuron.growth.score import score_units
            from treeneuron.growth.allocate import allocate
            from .gated_models import parent_model,StaticModel,insert_gates,ordinary_brackets,widen_ordinary
            from .gated_training import TrainingSession
            phase='native_training';native_start=time.perf_counter_ns();outer_rng=capture_rng()
            try:
                torch.manual_seed(request.seed);random.seed(request.seed);np.random.seed(seed64(request.seed,'gated','numpy')%2**32)
                width=recipe['width'];count=recipe['requested_count'];variant=recipe['variant']
                model=(StaticModel(F.shape[1],width,count,'centered' if variant.startswith('centered') else 'direct_leaf',
                    tau=recipe['tau'],dropout=recipe['dropout']) if self.family=='P05' else parent_model(F.shape[1],width,recipe['tau']))
                state=TrainingSession.create(model,n_rows=len(F),sampler_seed=seed64(request.seed,'gated','sampler'),
                    lr=recipe['lr'],weight_decay=recipe['weight_decay'],scheduler=recipe['scheduler'],
                    horizon=recipe['schedule_horizon'],batch_size=recipe['batch_size'])
                tF,tY,tG,tGY=map(torch.from_numpy,(F,scaled,G,yG))
                stages=[];candidate_hash=None;parent_hash=None;ordinary_width=None;preserved=None;ids=[]
                with (root/'events.jsonl').open('x') as stream:
                    stage_name='base'
                    def record(event):
                        stream.write(json.dumps(dict(event,stage=stage_name),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
                        stream.flush()
                    def stage(limit):
                        return state.fit_stage(tF,tY,tG,tGY,max_updates=limit,
                            validation_interval=recipe['validation_interval'],patience=recipe['patience'],event_sink=record,
                            target_mean=preprocessing['target_mean'],target_scale=preprocessing['target_scale'])
                    stages.append(stage(recipe['base_updates']))
                    if self.family!='P05':
                        parent_hash=_state_identity(state.model)
                        evaluation=copy.deepcopy(state.model).eval().double()
                        table=score_units(evaluation,torch.from_numpy(F64),torch.from_numpy((yF-preprocessing['target_mean'])/preprocessing['target_scale']),
                            epsilon=recipe['epsilon'],tau=recipe['tau'],adam_epsilon=recipe['adam_epsilon'],
                            batch_size=recipe['batch_size'],unit_block_size=8)
                        arrays={k:v.detach().cpu().numpy().copy() for k,v in vars(table).items() if isinstance(v,torch.Tensor)}
                        np.savez(root/'candidate-table.npz',**arrays)
                        allocation=allocate(table,count,policy='random' if self.family=='P01' else 'adam',
                            seed=seed64(request.seed,'gated','allocation'))
                        count=allocation.effective_count;ids=allocation.ids.tolist();candidate_hash=table.table_hash
                        _write(root/'allocation.json',dict(table_sha256=candidate_hash,status=list(table.status),
                            formula_version=table.formula_version,requested_count=allocation.requested_count,effective_count=count,
                            ids=ids,policy=allocation.policy,seed=allocation.seed,unchanged_contrasts=allocation.contrasts.tolist()))
                        with torch.no_grad():before=state.model(tF).clone()
                        if self.family=='P02':
                            brackets=ordinary_brackets(F.shape[1],width,count);ordinary_width=brackets[0 if variant=='lower' else 1]
                            model=widen_ordinary(state.model,state.optimizer,ordinary_width,seed=seed64(request.seed,'gated','ordinary'))
                        else:
                            insert_gates(state.model,state.optimizer,allocation.ids,allocation.contrasts.float());model=state.model
                        state.adopt_growth(model)
                        with torch.no_grad():preserved=torch.equal(state.model(tF),before)
                        if not preserved:raise ValueError('Growth changed the exact parent function')
                    stage_name='continuation';stages.append(stage(recipe['continuation_updates']))
                    stream.flush();os.fsync(stream.fileno())
                state.snapshot();model=state.model;model.eval()
                ages=[dict(name=name,birth_update=state.births[name],age=int(state.optimizer.state[p]['step']) if state.optimizer.state.get(p) else 0,
                    selected_absolute_update=state.update) for name,p in model.named_parameters()]
            finally:restore_rng(outer_rng)
            native_ns=time.perf_counter_ns()-native_start
            if _source()!=source:raise ValueError('Scientific source changed during fitting')
            from .ordinary_neural import _runtime
            runtime=_runtime();elapsed=time.perf_counter_ns()-started
            timeout=elapsed>request.budget.get('max_seconds',math.inf)*1e9
            arrays={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
            np.savez(root/'selected-state.npz',**arrays)
            summary=dict(schema='research/v2-gated-resources',family=self.family,mode=config.to_state()['mode'],
                status='timeout' if timeout else 'completed_ineligible',checkpoint_eligible=False,quality_eligible=False,Gate_D=False,
                preprocessing=preprocessing,fit_rows=len(F),select_rows=len(G),encoded_width=F.shape[1],resolved_parameters=recipe,
                parent_selected_state_sha256=parent_hash,candidate_table_sha256=candidate_hash,
                effective_count=count,selected_ids=ids,ordinary_width=ordinary_width,growth_function_preserved_exact=preserved,
                total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                stages=stages,parameter_ages=ages,selected_native_state_sha256=_state_identity(model),
                native_training_wall_ns=native_ns,total_fit_wall_ns=elapsed,process_peak_rss_bytes=_peak_rss(),
                event_stream_sha256=_sha(root/'events.jsonl'),source=source,dependencies=deps,runtime=runtime,
                accessed_roles=['F','G'],time_limit_exceeded=timeout,persistent_training_resume=False,
                hard_resource_enforcement='Containing supervisor required; max_seconds here is post-fit observation')
            _write(root/'attempt.json',summary)
            self._model,self._summary,self._config,self._preprocessing=model,summary,config,preprocessing
            self._source,self._dependencies=source,deps;self._seed=request.seed
            return self
        except BaseException as error:
            status,domain,retry=_failure_domain(error)
            self._summary=dict(status=status,failure_domain=domain,infrastructure_retry_permitted=retry,
                checkpoint_eligible=False,quality_eligible=False,failure_phase=phase,exception=type(error).__name__,
                message=str(error),traceback=traceback.format_exc(),total_fit_wall_ns=time.perf_counter_ns()-started)
            try:_write(root/'attempt.json',self._summary)
            except BaseException as receipt_error:error.add_note(f'Failure publication also failed: {receipt_error!r}')
            raise

    def predict(self,X,*,batch_size=None):
        if self._model is None:raise RuntimeError('Own-method predictor must be fitted')
        size=self._config.resolved()['batch_size'] if batch_size is None else batch_size
        if type(size) is not int or not 1<=size<=65536:raise ValueError('Invalid inference batch size')
        _,values=_matrix(X,self._config.feature_schema,self._preprocessing)
        import torch
        self._model.eval()
        with torch.no_grad():
            result=np.concatenate([self._model(torch.from_numpy(values[i:i+size])).numpy().astype(np.float64)
                for i in range(0,len(values),size)]) if len(values) else np.empty(0,dtype=np.float64)
        result=result*self._preprocessing['target_scale']+self._preprocessing['target_mean']
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite original-unit prediction')
        return result

    def resource_summary(self):return copy.deepcopy(self._summary)
