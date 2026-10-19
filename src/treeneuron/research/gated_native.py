"""Own-family F/G-only fit boundary; engineering only until workload admission.

Fitting and closed inference retain candidate/event and prediction bindings.
Worker integration and native workload admission are separate requirements;
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
from .p13_simple import (_source, _preprocessing, _features, _array_identity, _write, _hash, _sha,
    _read, _validate_preprocessing)
from .seeds import seed64

_ROOT=Path(__file__).resolve().parents[3]
_THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
_IMPORT_CONTEXT=None
_SNAPSHOT_SCOPE='Closed inference only; persistent optimizer/sampler training resume is unavailable'


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
                stages=stages,parameter_ages=ages,seed=request.seed,selected_native_state_sha256=_state_identity(model),
                native_training_wall_ns=native_ns,total_fit_wall_ns=elapsed,process_peak_rss_bytes=_peak_rss(),
                event_stream_sha256=_sha(root/'events.jsonl'),source=source,dependencies=deps,runtime=runtime,
                accessed_roles=['F','G'],time_limit_exceeded=timeout,persistent_training_resume=False,
                hard_resource_enforcement='Containing supervisor required; max_seconds here is post-fit observation')
            summary['selected_prediction_state_sha256']=_prediction_binding(config,summary)
            _validate_summary(config,summary,model)
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


    def save(self,path):
        if self._model is None:raise RuntimeError('Own-method predictor must be fitted')
        if self._source!=_source() or self._dependencies!=_dependencies():
            raise ValueError('Fitted source/runtime changed')
        _validate_preprocessing(self._preprocessing,self._config.feature_schema)
        if self._preprocessing!=self._summary['preprocessing']:
            raise ValueError('Preprocessing changed after the selected checkpoint')
        _validate_summary(self._config,self._summary,self._model)
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        state={name:t.detach().cpu().numpy().copy() for name,t in self._model.state_dict().items()}
        np.savez(path/'weights.npz',**state)
        _write(path/'manifest.json',dict(schema='research/v2-gated-prediction-snapshot',family=self.family,
            config=self._config.to_state(),seed=self._seed,source=self._source,dependencies=self._dependencies,
            preprocessing=self._preprocessing,resources=self._summary,weights_sha256=_sha(path/'weights.npz'),
            payload={k:dict(dtype=str(v.dtype),shape=list(v.shape)) for k,v in state.items()},
            snapshot_scope=_SNAPSHOT_SCOPE))

    @classmethod
    def load(cls,path,*,expected_manifest_sha256):
        import zipfile
        path=Path(path);manifest=path/'manifest.json';payload=path/'weights.npz'
        if (path.is_symlink() or manifest.is_symlink() or payload.is_symlink()
                or not manifest.is_file() or not payload.is_file() or not _digest(expected_manifest_sha256)
                or _sha(manifest)!=expected_manifest_sha256):
            raise ValueError('Require an unchanged closed snapshot and external trusted manifest SHA256')
        value=_read(manifest)
        fields={'schema','family','config','seed','source','dependencies','preprocessing','resources','weights_sha256','payload','snapshot_scope'}
        if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-gated-prediction-snapshot'
                or value['snapshot_scope']!=_SNAPSHOT_SCOPE):
            raise ValueError('Unknown closed gated snapshot schema or fields')
        config=GatedConfig.parse(value['family'],value['config'])
        source=_source();dependencies=_dependencies()
        if value['source']!=source or value['dependencies']!=dependencies:
            raise ValueError('Snapshot belongs to a different source or qualified dependency identity')
        summary=value['resources'];preprocessing=value['preprocessing']
        if (type(summary) is not dict or type(value['seed']) is not int or value['seed']!=summary.get('seed')
                or not 0<=value['seed']<2**63 or preprocessing!=summary.get('preprocessing')
                or summary.get('source')!=source or summary.get('dependencies')!=dependencies):
            raise ValueError('Snapshot differs from selected model lineage/transform')
        _validate_preprocessing(preprocessing,config.feature_schema)
        from .ordinary_neural import _runtime,_runtime_identity
        if _runtime_identity(_runtime())!=_runtime_identity(summary['runtime']):
            raise ValueError('Native runtime images or thread treatment changed')
        _validate_summary(config,summary)
        import torch
        with torch.random.fork_rng(devices=[]):model=_rebuild(config,summary)
        expected=model.state_dict()
        if _sha(payload)!=value['weights_sha256']:raise ValueError('Closed tensor payload hash differs')
        with zipfile.ZipFile(payload) as zipped:
            entries=zipped.infolist()
            if (len(entries)!=len(expected) or {e.filename for e in entries}!={k+'.npy' for k in expected}
                    or sum(e.file_size for e in entries)>64*1024**2):
                raise ValueError('Closed tensor allocation or names exceed the declared model')
        with np.load(payload,allow_pickle=False) as arrays:
            if len(arrays.files)!=len(expected) or set(arrays.files)!=set(expected) or set(value['payload'])!=set(expected):
                raise ValueError('Closed tensor names differ from the selected topology')
            tensors={}
            for name,reference in expected.items():
                a=arrays[name]
                if (a.dtype!=reference.numpy().dtype or a.shape!=tuple(reference.shape) or not np.isfinite(a).all()
                        or value['payload'][name]!=dict(dtype=str(a.dtype),shape=list(a.shape))):
                    raise ValueError('Closed tensor shape/dtype/finiteness mismatch; casting is forbidden')
                if name.endswith('grown_ids') and not np.array_equal(a,reference.numpy()):
                    raise ValueError('Closed unit identity differs from the selected topology')
                tensors[name]=torch.from_numpy(a.copy())
        model.load_state_dict(tensors,strict=True);_validate_summary(config,summary,model);model.eval()
        result=cls(config.to_state()['family'])
        result._model,result._config,result._seed=model,config,value['seed']
        result._summary,result._preprocessing=summary,preprocessing
        result._source,result._dependencies=source,dependencies
        return result


def _digest(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _prediction_binding(config,summary):
    return _hash(dict(config=config.to_state(),seed=summary['seed'],preprocessing=summary['preprocessing'],
        model=summary['selected_native_state_sha256'],effective_count=summary['effective_count'],
        selected_ids=summary['selected_ids'],ordinary_width=summary['ordinary_width']))


def _validate_summary(config,summary,model=None):
    recipe=config.resolved();family=recipe['family'];count=summary.get('effective_count');ids=summary.get('selected_ids')
    if (summary.get('schema')!='research/v2-gated-resources' or summary.get('family')!=family
            or summary.get('mode')!=config.to_state()['mode'] or summary.get('resolved_parameters')!=recipe
            or summary.get('checkpoint_eligible') is not False or summary.get('quality_eligible') is not False
            or summary.get('Gate_D') is not False or summary.get('persistent_training_resume') is not False
            or summary.get('accessed_roles')!=['F','G'] or summary.get('status') not in ('completed_ineligible','timeout')
            or type(summary.get('seed')) is not int or not 0<=summary['seed']<2**63
            or type(count) is not int or not 0<=count<=recipe['requested_count'] or type(ids) is not list
            or any(type(i) is not int or not 0<=i<recipe['width'] for i in ids) or ids!=sorted(set(ids))):
        raise ValueError('Closed recipe, roles, eligibility or unit identities differ')
    _validate_preprocessing(summary['preprocessing'],config.feature_schema)
    columns=summary['preprocessing']['columns'];d=sum(1+int(c['indicator']) if c['kind']=='numeric' else len(c['categories'])+1 for c in columns)
    if (type(summary.get('fit_rows')) is not int or not 2<=summary['fit_rows']<=20000
            or type(summary.get('select_rows')) is not int or not 1<=summary['select_rows']<=20000
            or not 1<=d<=128 or summary.get('encoded_width')!=d):raise ValueError('Closed F/G workload differs')
    base_count=recipe['width']*(d+2)+1
    if family=='P05':
        if count!=recipe['requested_count'] or ids or summary['parent_selected_state_sha256'] is not None or summary['candidate_table_sha256'] is not None:
            raise ValueError('Static snapshot cannot carry a grown-parent lineage')
        expected_count=base_count+2*count*(d+1)
    else:
        if len(ids)!=count or not _digest(summary['candidate_table_sha256']) or not _digest(summary['parent_selected_state_sha256']) or summary['growth_function_preserved_exact'] is not True:
            raise ValueError('Growth snapshot lacks its selected parent/candidate lineage')
        if family=='P02':
            from .gated_models import ordinary_brackets
            width=ordinary_brackets(d,recipe['width'],count)[0 if recipe['variant']=='lower' else 1]
            if summary['ordinary_width']!=width:raise ValueError('Ordinary capacity bracket changed')
            expected_count=width*(d+2)+1
        else:expected_count=base_count+2*count*(d+1)
    if family!='P02' and summary['ordinary_width'] is not None:raise ValueError('Unexpected ordinary topology')
    if summary.get('total_parameters')!=expected_count or summary.get('trainable_parameters')!=expected_count:
        raise ValueError('Closed parameter count differs from the selected topology')
    stages=summary.get('stages');absolute=0;birth=None
    if type(stages) is not list or len(stages)!=2:raise ValueError('Both declared training stages are required')
    for index,(stage,limit) in enumerate(zip(stages,(recipe['base_updates'],recipe['continuation_updates']))):
        if (type(stage) is not dict or set(stage)!={'selected_absolute_update','selected_stage_update','executed_stage_updates','executed_absolute_update','stop','G_mse'}
                or type(stage['executed_stage_updates']) is not int or not 1<=stage['executed_stage_updates']<=limit
                or type(stage['selected_stage_update']) is not int or not 0<=stage['selected_stage_update']<=stage['executed_stage_updates']
                or stage['selected_absolute_update']!=absolute+stage['selected_stage_update']
                or stage['executed_absolute_update']!=absolute+stage['executed_stage_updates']
                or stage['stop'] not in ('patience','max_updates') or stage['stop']=='max_updates' and stage['executed_stage_updates']!=limit
                or type(stage['G_mse']) not in (int,float) or not 0<=stage['G_mse']<float('inf')):
            raise ValueError('Invalid selected/executed stage counters or risk')
        absolute=stage['selected_absolute_update']
        if index==0:birth=absolute
    if not _digest(summary.get('selected_native_state_sha256')) or summary.get('selected_prediction_state_sha256')!=_prediction_binding(config,summary):
        raise ValueError('Prediction weights/transform/recipe binding differs')
    records=summary.get('parameter_ages')
    if type(records) is not list or not records or len({r.get('name') for r in records if type(r) is dict})!=len(records):
        raise ValueError('Missing or duplicate parameter ages')
    for row in records:
        name=row.get('name');born=(birth if family in ('P00','P01') and type(name) is str and name.startswith(('layer.gate.','layer.contrast.'))
            or family=='P02' and type(name) is str and name.startswith(('added_affine.','added_head.')) else 0)
        if (set(row)!={'name','birth_update','age','selected_absolute_update'} or type(name) is not str
                or type(row['birth_update']) is not int or row['birth_update']!=born
                or type(row['age']) is not int or row['age']!=absolute-born or row['selected_absolute_update']!=absolute):
            raise ValueError('Applied Adam age differs from selected/birth updates')
    if model is not None:
        if {r['name'] for r in records}!=set(dict(model.named_parameters())) or sum(p.numel() for p in model.parameters())!=expected_count or _state_identity(model)!=summary['selected_native_state_sha256']:
            raise ValueError('Closed/live parameters differ from the selected model')


def _rebuild(config,summary):
    import torch
    from .gated_models import parent_model,StaticModel,_WarmOrdinary
    recipe=config.resolved();family=recipe['family'];d=summary['encoded_width'];width=recipe['width']
    if family=='P05':
        return StaticModel(d,width,recipe['requested_count'],'centered' if recipe['variant'].startswith('centered') else 'direct_leaf',tau=recipe['tau'],dropout=recipe['dropout'])
    parent=parent_model(d,width,recipe['tau'])
    if family=='P02':return _WarmOrdinary(parent,summary['ordinary_width'],seed64(summary['seed'],'gated','ordinary'))
    parent.layer.grow(torch.tensor(summary['selected_ids'],dtype=torch.int64),torch.zeros((summary['effective_count'],d+1),dtype=torch.float32))
    return parent
