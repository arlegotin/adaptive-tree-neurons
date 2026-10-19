"""Source-bound, F-scored diagnostic forks of one immutable ordinary parent."""
import copy
from dataclasses import asdict,dataclass,replace
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch

from treeneuron import train
from treeneuron.artifacts import create_run,finish_run
from treeneuron.baselines.causal import ordinary_width_brackets,widen_parent
from treeneuron.checkpoint import save_checkpoint
from treeneuron.config import canonical_hash
from treeneuron.data.types import ObservationSet,array_hash
from treeneuron.experiments.paired import _json_tree,pairing_snapshot
from treeneuron.growth.allocate import Allocation,allocate
from treeneuron.growth.insert import insert_growth
from treeneuron.growth.score import ScoreTable,score_units
from .boundaries import insert_at_current_boundary,widen_at_current_boundary
from .ledger import write_new_json
from .roles import FittingData
from .snapshots import PrefixSnapshotSink
from .treatments import reset_stream


def _parent(parent):
    if (not isinstance(parent,train.TrainingState) or type(parent.model) is not train.SelectiveRegressor
            or parent.device!='cpu' or parent.model.layer.growth_applied
            or not parent.stage.get('complete') or parent.best_snapshot is not None
            or parent.stage.get('kind') not in ('base','continuation','research_boundary')
            or (parent.growth_metadata and not (parent.stage['kind']=='research_boundary'
                    and set(parent.growth_metadata)=={'research_boundary'}))):
        raise ValueError('Require a canonical completed ordinary parent or declared current boundary')
    snapshot=train.pack_state(parent)
    train.validate_metadata(snapshot['metadata'],parent.identity_hashes,'cpu')
    return snapshot,canonical_hash(_json_tree(snapshot))


@dataclass(frozen=True)
class ScoredParent:
    table: ScoreTable
    parent_sha256: str
    table_sha256: str
    elapsed_ns: int


def score_parent(parent,F):
    """Only F is accepted; G/V/E labels cannot enter these statistics."""
    started=time.perf_counter_ns()
    _,fingerprint=_parent(parent)
    if (not isinstance(F,ObservationSet) or F.noise_free_target is not None
            or any(array_hash(getattr(F,k))!=parent.data_hashes['F.'+k] for k in ('X','y','row_id','group_id'))):
        raise ValueError('Scorer requires the exact F-only role without evaluator targets')
    table=score_units(copy.deepcopy(parent.model).double().eval(),
        torch.tensor(parent.preprocessing.transform_X(F.X),dtype=torch.float64),
        torch.tensor(parent.preprocessing.transform_y(F.y),dtype=torch.float64),
        epsilon=parent.config.growth.epsilon,tau=parent.config.model.tau,
        adam_epsilon=parent.config.training.adam_epsilon,batch_size=parent.config.training.batch_size,
        unit_block_size=parent.config.growth.block_units)
    return ScoredParent(table,fingerprint,table.table_hash,time.perf_counter_ns()-started)


@dataclass(frozen=True)
class ForkSpec:
    role: str
    requested_count: int
    allocation_seed: int
    max_updates: int
    new_stream_seed: int|None=None
    unit_ids: tuple[int,...]=()
    fixed_budget: bool=False
    selection_prefixes: tuple[int,...]=()

    def __post_init__(self):
        if (self.role not in ('keep','top','random','warm_lower','warm_upper','subset','ridge')
                or type(self.requested_count) is not int or self.requested_count<0
                or type(self.max_updates) is not int or self.max_updates<=0
                or type(self.allocation_seed) is not int or not 0<=self.allocation_seed<2**63
                or (self.new_stream_seed is not None and (type(self.new_stream_seed) is not int
                    or not 0<=self.new_stream_seed<2**63)) or type(self.fixed_budget) is not bool
                or not isinstance(self.unit_ids,tuple) or not isinstance(self.selection_prefixes,tuple)
                or any(type(i) is not int or i<0 for i in self.unit_ids)
                or list(self.unit_ids)!=sorted(set(self.unit_ids))
                or (self.role=='subset' and (not self.unit_ids or len(self.unit_ids)!=self.requested_count))
                or (self.role!='subset' and self.unit_ids)):
            raise ValueError('Invalid fixed diagnostic fork specification')


def _subset(table,ids,seed):
    rows=[table.ids.tolist().index(unit) for unit in ids]
    return Allocation(len(ids),len(ids),torch.tensor(ids,dtype=torch.int64),table.contrasts[rows].clone(),
        'greedy',seed,table.table_hash,torch.zeros(len(ids),dtype=torch.int64),
        {'shared_scored_units':len(table.ids),'standalone_contrast_units_required':0})


def fit_diagnostic_fork(parent,data,output,spec,*,scored=None,selection_trace=None,guard=lambda:None):
    """Fit one nominal fork; the outer M2 controller owns admission and recovery."""
    started=time.perf_counter_ns()
    snapshot,fingerprint=_parent(parent)
    if (not isinstance(data,FittingData) or not isinstance(spec,ForkSpec)
            or parent.identity_hashes!=train._identities(data,parent.config,parent.preprocessing)
            or parent.data_hashes!=train._fg_hashes(data)):
        raise ValueError('Diagnostic fork requires exact F/G-only parent context')
    policy=replace(train.TrainingPolicy.from_config(parent.config,continuation=True),
        max_updates=spec.max_updates,stop_by_patience=not spec.fixed_budget,
        selection_prefixes=spec.selection_prefixes)
    if policy.max_updates>parent.config.training.continuation_updates:
        raise ValueError('Fork budget exceeds declared continuation ceiling')
    if scored is None and spec.role!='keep':
        raise ValueError('Growth/ordinary bracket fork requires its common F score table')
    if scored is not None and (not isinstance(scored,ScoredParent) or scored.parent_sha256!=fingerprint):
        raise ValueError('Score binding belongs to another parent')
    if scored is not None and scored.table.table_hash!=scored.table_sha256:
        raise ValueError('Score table changed after binding')
    ridge_allocation=None
    if spec.role=='ridge':
        from .joint_allocation import SelectionTrace,selection_to_allocation
        if type(selection_trace) is not SelectionTrace:
            raise ValueError('Ridge fork requires its immutable selection trace')
        ridge_allocation=selection_to_allocation(parent,data.F,scored,selection_trace,seed=spec.allocation_seed)
        if ridge_allocation.requested_count!=spec.requested_count:
            raise ValueError('Fork count differs from selection trace count')
    elif selection_trace is not None:
        raise ValueError('Selection trace is only valid for a ridge fork')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    report=dict(schema='research/v2-diagnostic-fork',status='running',request=asdict(spec),
        resolved_policy=asdict(policy),parent_sha256=fingerprint,parent_update=parent.update,
        parent_identity_hashes=dict(parent.identity_hashes),
        table_sha256=None if scored is None else scored.table_sha256,models={},
        nominal_ids=list(spec.unit_ids),V_accessed=False,E_accessed=False,T_accessed=False)
    write_new_json(root/'request.json',report)
    if selection_trace is not None:
        write_new_json(root/'selection.json',selection_trace.to_dict())
        report['selection_trace_sha256']=selection_trace.sha256
    before=train.capture_rng()
    try:
        guard()
        eligible=None if scored is None else allocate(scored.table,len(scored.table.ids),
            policy='adam',seed=spec.allocation_seed).ids.tolist()
        effective=(0 if spec.requested_count==0 else None) if eligible is None else min(spec.requested_count,len(eligible))
        report.update(common_eligible_ids=eligible,requested_count=spec.requested_count,effective_count=effective)
        if spec.role=='subset' and any(unit not in eligible for unit in spec.unit_ids):
            report.update(status='completed_ineligible',reason='nominal subset contains a current-ineligible unit; retained without substitution')
            return report
        state=train.unpack_state(snapshot,expected_hashes=parent.identity_hashes,device='cpu')
        state.history=copy.deepcopy(parent.history)
        initial_prediction=state.predict(data.G.X)
        current=state.stage['kind']=='research_boundary'
        selected=[]
        if spec.role in ('top','random','subset','ridge'):
            allocation=(ridge_allocation if spec.role=='ridge' else _subset(scored.table,spec.unit_ids,spec.allocation_seed) if spec.role=='subset'
                else allocate(scored.table,spec.requested_count,
                    policy='adam' if spec.role=='top' else 'random',seed=spec.allocation_seed))
            (insert_at_current_boundary if current else insert_growth)(state,allocation)
            selected=allocation.ids.tolist()
        elif spec.role in ('warm_lower','warm_upper'):
            target=sum(p.numel() for p in state.model.parameters())+2*effective*(state.model.layer.d+1)
            lower,upper=ordinary_width_brackets(state.model.layer.d,target)
            width=lower if spec.role=='warm_lower' else upper
            (widen_at_current_boundary if current else widen_parent)(state,width-state.model.layer.h,spec.allocation_seed)
            report.update(ordinary_width=width,ordinary_brackets=[lower,upper],gated_parameter_target=target)
        if not np.array_equal(state.predict(data.G.X),initial_prediction):
            raise AssertionError('Diagnostic insertion changed parent predictions')
        stream=None if spec.new_stream_seed is None else reset_stream(state,seed=spec.new_stream_seed)
        state.growth_metadata={**state.growth_metadata,'research_diagnostic':dict(
            parent_sha256=fingerprint,role=spec.role,requested_count=spec.requested_count,
            selected_ids=selected,table_sha256=report['table_sha256'])}
        pairing=pairing_snapshot(state)
        pairing.update(allowance=policy.max_updates,resolved_policy=asdict(policy))
        report.update(selected_ids=selected,stream_reset=stream,initial_pairing=pairing,
            identity_hashes=dict(state.identity_hashes))
        save_checkpoint(state,root/'initial.json')
        paths=create_run(root/'runs',state.config.to_dict())
        report['run_path']=str(paths.directory.resolve())
        sink=(PrefixSnapshotSink(root/'prefixes',spec.selection_prefixes,
            trajectory_id=canonical_hash(report['request'])+'|'+fingerprint) if spec.selection_prefixes else None)
        state=train.continue_fit(state,data,policy,paths,selection_prefixes=spec.selection_prefixes,on_prefix=sink)
        finish_run(paths,{'selected_g_mse':state.stage['best_mse']},'completed')
        pointer=root/'selected.json';save_checkpoint(state,pointer)
        report.update(status='completed_eligible',selected_absolute_update=state.update,
            selected_stage_update=state.update-state.stage['start_update'],executed_stage_updates=state.stage['updates_completed'],
            selected_g_mse=state.stage['best_mse'],total_parameters=sum(p.numel() for p in state.model.parameters()),
            models={'selected':dict(path=str(pointer.resolve()),sha256=hashlib.sha256(pointer.read_bytes()).hexdigest(),
                                    loader='training_state')})
        if sink is not None:report['prefixes']=sink.finalize()
        return report
    except BaseException as error:
        report.update(status=('interrupted_recoverable' if isinstance(error,KeyboardInterrupt)
            else 'resource_exhausted' if isinstance(error,(MemoryError,OSError))
            else 'numerical_failure' if isinstance(error,FloatingPointError) else 'input_or_dependency_failure'),
            error_type=type(error).__name__,reason=str(error))
        raise
    finally:
        active_error=sys.exc_info()[1]
        report['costs']=dict(enclosing_fork_ns=time.perf_counter_ns()-started,
            shared_score_ns=None if scored is None else scored.elapsed_ns,
            standalone_total_ns=None,reason='controller must add validated parent/scoring prerequisite costs; timers are nested')
        try:
            try:
                write_new_json(root/'result.json',report)
            except BaseException as publication_error:
                if active_error is None:raise
                active_error.add_note(f'Diagnostic result publication also failed: {publication_error}')
        finally:
            train.restore_rng(before)
