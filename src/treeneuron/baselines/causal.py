"""B1–B6 causal controls using the shared F/G training/recovery contract.

Factory config is finite JSON: ``experiment`` is a resolved experiment for fresh
controls; ``variant`` and variant-specific counts identify the recipe. FitRequest
config is instead explicit in-process runtime context, exactly
``{'development_data': original_DevelopmentData}``. It is not scientific config.
The original manifest/row identities are retained and supplied arrays must match
F/G. No artificial IDs, empty V partition, or label access are synthesized.
"""
from __future__ import annotations
import copy
import json
import time
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import torch
from treeneuron.artifacts import create_run
from treeneuron.baselines.api import FitRequest
from treeneuron.baselines.causal_evidence import attach_causal_evidence
from treeneuron.config import canonical_hash, canonical_json, resolve_experiment
from treeneuron.data.types import DevelopmentData, SplitManifest, array_hash
from treeneuron.growth.allocate import Allocation
from treeneuron.growth.insert import insert_growth, _validate_optimizer, _OLD
from treeneuron.growth.types import GrowthManifest, ParameterMapping
from treeneuron.models.static import StaticRegressor
from treeneuron.models.widened import WidenedRegressor
from treeneuron.train import (TrainingState, TrainingPolicy, SelectiveRegressor, UpdateSchedule,
    capture_rng, restore_rng, pack_state, unpack_state, fit_parent, continue_fit,
    validate_metadata, validate_tensor_state, _fg_hashes)


def ordinary_width_brackets(d: int, target_parameters: int) -> tuple[int, int]:
    """Nearest nonempty ordinary widths, including every scalar head parameter."""
    if (type(d) is not int or d <= 0 or type(target_parameters) is not int
            or target_parameters < d+3):
        raise ValueError('Need positive dimension and target at least one complete unit plus head')
    lower, remainder = divmod(target_parameters-1, d+2)
    return lower, lower+bool(remainder)


def _selected_parent(state, *, allow_current_boundary=False):
    boundary = isinstance(state, TrainingState) and state.stage.get('kind') == 'research_boundary'
    if boundary and not allow_current_boundary:
        raise ValueError('A current research boundary requires its explicit research widening interface')
    if (not isinstance(state, TrainingState) or type(state.model) is not SelectiveRegressor
            or state.config.model.coordinates != 'selective' or state.model.layer.growth_applied
            or not state.stage.get('complete') or state.best_snapshot is not None
            or (state.growth_metadata and not (allow_current_boundary and boundary
                    and set(state.growth_metadata) == {'research_boundary'}))):
        raise ValueError('Require canonical completed selected ordinary parent without previous event')
    _validate_optimizer(state, _OLD)
    snapshot=pack_state(state)
    validate_metadata(snapshot['metadata'],state.identity_hashes,state.device)
    validate_tensor_state(snapshot['tensors'])
    return snapshot


def widen_parent(state: TrainingState, added_units: int, seed: int) -> GrowthManifest:
    """Atomic widening with explicit whole-model slices (head included).

    Old tensors retain their exact moments/steps. Separate added incoming and
    outgoing tensors have fresh zero steps/moments. Training RNG and the
    absolute schedule do not restart. The event seed only initializes incoming
    rows. New head entries are zero. No duplication control is implemented.
    """
    return _widen_at_completed_boundary(state, added_units, seed)


def _widen_at_completed_boundary(state, added_units, seed, *, allow_current_boundary=False):
    _selected_parent(state, allow_current_boundary=allow_current_boundary)
    h,d=state.model.layer.h,state.model.layer.d
    if (type(added_units) is not int or added_units < 0 or h+added_units >95
            or type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('Invalid added units/seed or widening exceeds tested CPU width 95')
    before=capture_rng()
    try:
        torch.manual_seed(seed)
        model=WidenedRegressor(d,h+added_units,h,state.model.layer.tau)
        model.train(state.model.training)
        old,new=dict(state.model.named_parameters()),dict(model.named_parameters())
        mappings=[]
        with torch.no_grad():
            if model.head.added is not None: model.head.added.weight.zero_()
            for name,p in old.items():
                new_name=name.replace('head.','head.base.')
                q=new[new_name]; slices=tuple(slice(0,n) for n in p.shape)
                q[slices].copy_(p)
                full=tuple((0,n) for n in p.shape)
                mappings.append(ParameterMapping(name,new_name,tuple(p.shape),tuple(q.shape),full,full,
                    'identity' if p.shape==q.shape else 'slice'))
        mapped={m.old_name:m.new_name for m in mappings}
        added_names=[name for name in new if name not in mapped.values()]
        groups=[]
        for group in state.optimizer.param_groups:
            names=[mapped[n] for n in group['param_names']]
            groups.append({**{k:copy.deepcopy(v) for k,v in group.items() if k not in ('params','param_names')},
                           'param_names':names,'params':[new[n] for n in names]})
        if added_names:
            groups.append({**{k:copy.deepcopy(v) for k,v in groups[0].items() if k not in ('params','param_names')},
                           'param_names':added_names,'params':[new[n] for n in added_names]})
        optimizer=torch.optim.AdamW(groups,foreach=False)
        for name,p in old.items():
            if state.optimizer.state.get(p):
                optimizer.state[new[mapped[name]]]={k:v.clone() for k,v in state.optimizer.state[p].items()}
        for name in added_names:
            q=new[name]
            optimizer.state[q]={'step':torch.tensor(0.,dtype=torch.float32),'exp_avg':torch.zeros_like(q),'exp_avg_sq':torch.zeros_like(q)}
        schedule_state=state.scheduler.state_dict()
        if added_names: schedule_state['base_lrs'].append(state.config.training.learning_rate)
        schedule=UpdateSchedule(optimizer,**schedule_state)
        config=state.config.to_dict(); config['model'].update(h=h+added_units,coordinates='widened',parent_width=h)
        config=resolve_experiment(config)
        identity={**state.identity_hashes,'config':canonical_hash(config.to_dict())}
        manifest=GrowthManifest(1,d,h+added_units,tuple(range(h,h+added_units)),True,
            sum(p.numel() for p in old.values()),sum(p.numel() for p in new.values()),
            added_units*(d+2),tuple(old),tuple(added_names),tuple(mappings))
        backend=attach_causal_evidence(state.backend,model,config.training.batch_size)
        metadata=dict(manifest_scope='complete_regressor_including_head',parent_identity_hashes=copy.deepcopy(state.identity_hashes),
            parent_model=state.model.export_config(),widening_manifest=json.loads(canonical_json(manifest.to_dict())),
            widening_seed=seed,insertion_update=state.update,
            adam_step_policy='old steps retained; separate added tensors have step zero and zero moments')
        if 'research_boundary' in state.growth_metadata:
            metadata['research_boundary'] = copy.deepcopy(state.growth_metadata['research_boundary'])
        candidate=copy.copy(state)
        candidate.model,candidate.optimizer,candidate.scheduler=model,optimizer,schedule
        candidate.config,candidate.identity_hashes,candidate.backend=config,identity,backend
        candidate.growth_metadata=metadata
        candidate.parameter_birth_updates = {
            mapped[name]: state.parameter_birth_updates[name] for name in old}
        candidate.parameter_birth_updates.update({name: state.update for name in added_names})
        snapshot=pack_state(candidate)
        validate_metadata(snapshot['metadata'],identity,'cpu'); validate_tensor_state(snapshot['tensors'])
        for name in ('model','optimizer','scheduler','config','identity_hashes','backend','growth_metadata','parameter_birth_updates'):
            setattr(state,name,getattr(candidate,name))
        return manifest
    finally:
        restore_rng(before)


def _validate_context(data):
    """Verify original manifest commitments using F/G and stored V hashes only."""
    if not isinstance(data.manifest,SplitManifest) or data.manifest.phase!='exploratory':
        raise ValueError('Original exploratory split manifest required')
    manifest=data.manifest
    if manifest.teacher_id!=data.teacher_id or manifest.feature_names!=data.F.feature_names or data.G.feature_names!=data.F.feature_names:
        raise ValueError('Original teacher/feature provenance mismatch')
    record=manifest.to_dict();record.pop('assignment_hash');record.pop('feature_schema_hash')
    reconstructed=SplitManifest(**record)
    if reconstructed!=manifest: raise ValueError('Invalid original split identity')
    hashes=dict(manifest.hashes)
    for split in ('F','G'):
        rows=getattr(data,split)
        positions=[i for i,part in enumerate(manifest.assignments) if part==split]
        if (tuple(int(v) for v in rows.row_id)!=tuple(manifest.row_id[i] for i in positions)
                or tuple(int(v) for v in rows.group_id)!=tuple(manifest.group_id[i] for i in positions)
                or any(hashes.get(f'{split}.{name}')!=array_hash(getattr(rows,name)) for name in ('X','y'))):
            raise ValueError('F/G arrays or row/group identities differ from original manifest')
    raw=canonical_hash({'teacher_id':data.teacher_id,'hashes':manifest.hashes,'row_id':manifest.row_id,
        'feature_names':manifest.feature_names,'feature_schema_hash':manifest.feature_schema_hash})
    if raw!=manifest.dataset_id: raise ValueError('Raw development identity differs from original manifest')


def _validate_adapter_identity(method,recipe,state):
    """Labels may not silently reinterpret a saved topology or allocation."""
    if not isinstance(recipe,dict): raise ValueError('Invalid causal recipe')
    c=state.config.model
    if method=='B1':
        valid=not recipe and c.coordinates=='selective' and not state.model.layer.growth_applied
    elif method=='B2' and recipe.get('variant') in ('lower','upper'):
        valid=set(recipe)=={'variant','target_parameters'} and c.coordinates=='selective' and not state.model.layer.growth_applied
        if valid:
            widths=ordinary_width_brackets(c.d,recipe['target_parameters'])
            valid=c.h==widths[recipe['variant']=='upper']
    elif method=='B2' and recipe.get('variant')=='widen':
        valid=(set(recipe)=={'variant','added_units','seed'} and c.coordinates=='widened'
            and type(recipe['added_units']) is int and c.h-c.parent_width==recipe['added_units']
            and state.growth_metadata.get('widening_seed')==recipe['seed']
            and state.growth_metadata.get('manifest_scope')=='complete_regressor_including_head')
        if valid:
            parent_config=state.config.to_dict()
            parent_config['model'].update(h=c.parent_width,coordinates='selective',parent_width=0)
            parent_identity={**state.identity_hashes,'config':canonical_hash(parent_config)}
            valid=state.growth_metadata.get('parent_identity_hashes')==parent_identity
    elif method=='B3':
        variant=recipe.get('variant')
        valid=(variant in ('partial','full') and set(recipe)==({'variant','coordinates','count'} if variant=='partial' else {'variant','coordinates'})
            and c.coordinates in ('centered','direct_leaf') and c.coordinates==recipe.get('coordinates')
            and c.static_count==(recipe.get('count') if variant=='partial' else c.h))
    elif method in ('B4','B5','B6'):
        policy='greedy' if method=='B6' else 'random' if method=='B4' else recipe.get('variant')
        allocation=state.growth_metadata.get('allocation',{})
        valid=(set(recipe)==(set() if method in ('B4','B6') else {'variant'})
            and policy in (('greedy',) if method=='B6' else ('random',) if method=='B4' else ('gaussian','adam','sgd'))
            and c.coordinates=='selective' and allocation.get('policy')==policy
            and allocation.get('ids')==state.model.layer.grown_ids.tolist()
            and allocation.get('effective_count')==len(state.model.layer.grown_ids))
    else: valid=False
    if not valid: raise ValueError('Causal adapter recipe differs from state topology/provenance')


class CausalPredictor:
    def __init__(self, method_id, recipe, *, config=None, state=None):
        self.method_id,self.recipe,self.config,self.state=method_id,copy.deepcopy(recipe),config,state
        self.fit_nanoseconds=None
        self.run_paths=None

    def fit(self, request: FitRequest, *, on_run_created=None):
        if (not isinstance(request.config,dict) or set(request.config)!={'development_data'}
                or not isinstance(request.config['development_data'],DevelopmentData)):
            raise ValueError('FitRequest.config requires original development_data runtime context')
        data=request.config['development_data']
        _validate_context(data)
        for supplied, actual in ((request.X_fit,data.F.X),(request.y_fit,data.F.y),
                                 (request.X_select,data.G.X),(request.y_select,data.G.y)):
            if array_hash(np.asarray(supplied)) != array_hash(actual):
                raise ValueError('FitRequest F/G arrays differ from original development provenance')
        config=self.state.config if self.state is not None else self.config
        if request.seed != config.seeds.optimizer:
            raise ValueError('FitRequest seed differs from resolved optimizer seed')
        continuation=self.state is not None
        policy=TrainingPolicy.from_config(config,continuation=continuation)
        if continuation and not self.state.stage['complete']:
            policy=TrainingPolicy(**self.state.stage['policy'])
        if (set(request.budget)!={'max_updates'} or type(request.budget['max_updates']) is not int
                or request.budget['max_updates']!=policy.max_updates):
            raise ValueError('Explicit max_updates must equal the resolved stage allowance')
        if self.state is not None and self.state.data_hashes != _fg_hashes(data):
            raise ValueError('Parent F/G identities differ from supplied original provenance')
        tick=time.perf_counter_ns()
        paths=create_run(Path(request.artifact_dir),config.to_dict())
        self.run_paths=paths
        metadata=copy.deepcopy(self.state.growth_metadata) if continuation else {}
        metadata['causal_adapter']={'method_id':self.method_id,'recipe':self.recipe}
        if continuation:
            self.state.growth_metadata=copy.deepcopy(metadata)
            if self.state.best_snapshot is not None:
                self.state.best_snapshot['metadata']['growth_metadata']=copy.deepcopy(metadata)
        if on_run_created is not None:
            on_run_created(paths,metadata)
        self.state=(continue_fit(self.state,data,policy,paths) if continuation else
                    fit_parent(data,config,paths,growth_metadata=metadata))
        self.fit_nanoseconds=time.perf_counter_ns()-tick
        self.config=self.state.config
        return self

    def predict(self, X, *, batch_size=None):
        if self.state is None: raise RuntimeError('Predictor is not fitted')
        values=self.state.predict(X,batch_size=batch_size)
        if values.shape!=(len(X),) or not np.isfinite(values).all():
            raise ValueError('Predictor must return finite original-unit [N], without ensemble axes')
        return values

    def save(self, path: Path):
        if self.state is None: raise RuntimeError('Predictor is not fitted')
        from treeneuron.checkpoint import save_checkpoint
        _validate_adapter_identity(self.method_id,self.recipe,self.state)
        self.state.growth_metadata['causal_adapter']={'method_id':self.method_id,'recipe':self.recipe}
        save_checkpoint(self.state,path)

    @classmethod
    def load(cls,path: Path, *, expected_hashes):
        from treeneuron.checkpoint import load_checkpoint
        state=load_checkpoint(path,expected_hashes=expected_hashes,device='cpu')
        metadata=state.growth_metadata.get('causal_adapter')
        if not metadata or set(metadata)!={'method_id','recipe'} or metadata['method_id'] not in ('B1','B2','B3','B4','B5','B6'):
            raise ValueError('Checkpoint has no valid causal adapter identity')
        _validate_adapter_identity(metadata['method_id'],metadata['recipe'],state)
        return cls(metadata['method_id'],metadata['recipe'],config=state.config,state=state)

    def resource_summary(self):
        state=self.state
        return dict(method_id=self.method_id,recipe=self.recipe,fitted=state is not None,
            run_id=None if self.run_paths is None else self.run_paths.run_id,
            artifact_directory=None if self.run_paths is None else str(self.run_paths.directory),
            total_parameters=None if state is None else sum(p.numel() for p in state.model.parameters()),
            actual_growth_count=None if state is None else (state.model.layer.count if isinstance(state.model,StaticRegressor)
                else len(state.model.layer.grown_ids) if 'widening_manifest' not in state.growth_metadata
                else len(state.growth_metadata['widening_manifest']['grown_ids'])),
            capabilities=dict(backend='cpu',dtype='float32',execution='eager',output_units='original_target',
                internal_validation='G only',preprocessing='F only',runtime_context='original DevelopmentData required in process',
                context_persisted='original split/raw/F/G row, group and array hashes',
                duplication='deferred',id_search=False if self.method_id=='B3' else 'allocation producer owned'),
            backend=None if state is None else state.backend,
            costs={'adapter_fit_nanoseconds':self.fit_nanoseconds,
                   'boundary':'create_run plus fit/continuation; nested training costs included; factory/scoring excluded',
                   'enclosing_study_cost':'Task12 owns end-to-end timer; do not sum nested costs'},
            memory={'value':None,'reason':'adapter makes no additional memory measurement'})


def build_causal(method_id: str, config: Mapping, *, parent: TrainingState | None=None,
                 allocation: Allocation | None=None) -> CausalPredictor:
    canonical_json(config)
    cfg=dict(config)
    if method_id not in ('B1','B2','B3','B4','B5','B6'): raise ValueError('Unsupported causal method; B7 pending')
    warm=method_id in ('B1','B4','B5','B6') or (method_id=='B2' and cfg.get('variant')=='widen')
    if warm:
        allowed={'B1':set(),'B2':{'variant','added_units','seed'},'B4':set(),'B5':{'variant'},'B6':set()}[method_id]
        if set(cfg)!=allowed or parent is None: raise ValueError('Invalid warm control recipe or missing parent')
        snapshot=_selected_parent(parent)
        before=capture_rng()
        try: state=unpack_state(snapshot,expected_hashes=parent.identity_hashes,device='cpu')
        finally: restore_rng(before)
        state.history=copy.deepcopy(parent.history)
        if method_id in ('B4','B5','B6'):
            expected='greedy' if method_id=='B6' else 'random' if method_id=='B4' else cfg['variant']
            if expected not in ('random','gaussian','adam','sgd','greedy') or (method_id=='B5' and expected not in ('gaussian','adam','sgd')) or allocation is None or allocation.policy!=expected:
                raise ValueError('Allocation policy differs from declared B4/B5/B6 recipe')
            insert_growth(state,allocation)
            state.backend=attach_causal_evidence(state.backend,state.model,state.config.training.batch_size)
        else:
            if allocation is not None: raise ValueError('This control does not accept allocation')
            if method_id=='B2': widen_parent(state,cfg['added_units'],cfg['seed'])
        return CausalPredictor(method_id,cfg,state=state,config=state.config)
    if parent is not None or allocation is not None: raise ValueError('Fresh controls cannot inherit a parent/allocation')
    if 'experiment' not in cfg: raise ValueError('Fresh control requires resolved experiment')
    experiment=resolve_experiment(cfg['experiment']).to_dict()
    if experiment['backend']['device']!='cpu': raise ValueError('Only selected CPU controls are supported')
    if method_id=='B2':
        if set(cfg)!={'experiment','variant','target_parameters'} or cfg['variant'] not in ('lower','upper'):
            raise ValueError('B2 fresh requires lower/upper bracket; duplication deferred')
        widths=ordinary_width_brackets(experiment['model']['d'],cfg['target_parameters'])
        experiment['model'].update(h=widths[cfg['variant']=='upper'],coordinates='selective',static_count=0,parent_width=0)
    elif method_id=='B3':
        variant=cfg.get('variant'); expected={'experiment','variant','coordinates'} | ({'count'} if variant=='partial' else set())
        if set(cfg)!=expected or variant not in ('partial','full') or cfg['coordinates'] not in ('centered','direct_leaf'):
            raise ValueError('B3 requires partial/full and centered/direct_leaf coordinates')
        experiment['model'].update(coordinates=cfg['coordinates'],static_count=cfg.get('count',experiment['model']['h']),parent_width=0)
    else: raise ValueError('Method requires parent')
    experiment=resolve_experiment(experiment)
    if experiment.model.h>95 or (method_id=='B3' and experiment.model.h>64): raise ValueError('Control exceeds bounded CPU shape evidence')
    return CausalPredictor(method_id,{k:v for k,v in cfg.items() if k!='experiment'},config=experiment)
