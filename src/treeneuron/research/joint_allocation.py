"""F-only, parent-bound joint placement and checked native Allocation conversion."""
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import resource
import sys
import time

import numpy as np
import torch

from treeneuron.data.types import ObservationSet, array_hash
from treeneuron.growth.allocate import Allocation, allocate
from .diagnostics import ScoredParent, _parent
from .jacobian import prediction_jacobians
from .joint_selector import JointSelectionConfig, prepare_joint_problem


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def _hash(value):return hashlib.sha256(_json(value).encode()).hexdigest()


def _diagnostic_values(value):
    if isinstance(value,list):return [_diagnostic_values(v) for v in value]
    if np.isfinite(value):return value
    return 'nan' if np.isnan(value) else 'inf' if value>0 else '-inf'


@dataclass(frozen=True,eq=False)
class CandidateBlock:
    unit_id:int
    contrast:np.ndarray
    eligible:bool
    ineligibility_reason:str|None
    added_parameter_count:int

    def __post_init__(self):
        c=self.contrast
        if (type(self.unit_id) is not int or self.unit_id<0 or not isinstance(c,np.ndarray)
                or c.dtype!=np.float64 or c.ndim!=1 or len(c)<2 or not np.isfinite(c).all()
                or type(self.eligible) is not bool or type(self.added_parameter_count) is not int
                or self.added_parameter_count!=2*len(c)
                or self.eligible and self.ineligibility_reason is not None
                or not self.eligible and (type(self.ineligibility_reason) is not str or not self.ineligibility_reason)):
            raise ValueError('Finite immutable candidate geometry, eligibility and full parameter cost required')
        # A bytes-backed view cannot have its write flag re-enabled by a caller.
        object.__setattr__(self,'contrast',np.frombuffer(c.tobytes(),dtype=np.float64))


@dataclass(frozen=True)
class SelectionTrace:
    payload_json:str

    def to_dict(self):return json.loads(self.payload_json)

    @property
    def sha256(self):return hashlib.sha256(self.payload_json.encode()).hexdigest()


def _check_chain(report,config,eligible):
    order=report['selection_order'];slots=report['slots'];gain=report['predicted_gain']
    valid_number=lambda v:type(v) in (int,float) and math.isfinite(v) and v>=0
    if (type(order) is not list or any(type(i) is not int for i in order) or sorted(order)!=report['selected_ids']
            or type(slots) is not list or len(slots)!=len(order) or not valid_number(gain)
            or report['eligible_count']!=len(eligible)):
        raise ValueError('Selection trace conditional chain/count is inconsistent')
    remaining=set(eligible);total=0.
    for index,(unit,row) in enumerate(zip(order,slots)):
        if (type(row) is not dict or row.get('slot')!=index or row.get('unit_id')!=unit
                or not valid_number(row.get('marginal_gain')) or type(row.get('candidate_gains')) is not list):
            raise ValueError('Selection trace conditional winner is inconsistent')
        choices=row['candidate_gains']
        if (any(type(c) is not dict or set(c)!={'unit_id','gain'} or type(c['unit_id']) is not int
                or not valid_number(c['gain']) for c in choices)
                or len(choices)!=len(remaining) or {c['unit_id'] for c in choices}!=remaining):
            raise ValueError('Selection trace conditional candidate inventory differs')
        values={c['unit_id']:c['gain'] for c in choices};best=max(values.values())
        tolerance=config.tie_relative_tolerance*max(1.,abs(best))
        winner=min(j for j,g in values.items() if g>=best-tolerance)
        if unit!=winner or row['marginal_gain']!=values[unit]:
            raise ValueError('Selection trace conditional gain/tie rule differs')
        total+=values[unit];remaining.remove(unit)
    if not math.isclose(gain,total,rel_tol=1e-8,abs_tol=1e-10):
        raise ValueError('Selection trace total gain differs from conditional chain')


def _bound(parent,F,scored):
    _,digest=_parent(parent)
    if (type(scored) is not ScoredParent or scored.parent_sha256!=digest
            or scored.table_sha256!=scored.table.table_hash):
        raise ValueError('Exact immutable scored-parent/table binding required')
    if (not isinstance(F,ObservationSet) or F.noise_free_target is not None
            or any(array_hash(getattr(F,k))!=parent.data_hashes['F.'+k] for k in ('X','y','row_id','group_id'))):
        raise ValueError('Exact F-only observations without evaluator targets required')
    table=scored.table;layer=parent.model.layer
    if (not torch.equal(table.ids,torch.arange(layer.h,dtype=torch.int64))
            or table.contrasts.dtype!=torch.float64 or table.contrasts.device.type!='cpu'
            or table.contrasts.shape!=(layer.h,layer.d+1) or table.n_rows!=len(F.X)
            or table.tau!=layer.tau):raise ValueError('Scored parent/table geometry differs')
    return digest


def _record(candidate):
    return dict(unit_id=candidate.unit_id,contrast=candidate.contrast.tolist(),eligible=candidate.eligible,
        ineligibility_reason=candidate.ineligibility_reason,added_parameter_count=candidate.added_parameter_count)


def build_gate_blocks(parent,F,scored):
    """Retain every nominal ID, using precisely the legacy allocation guard.

    The typed ScoredParent wrapper supplies parent/table identities. A bare
    ScoreTable is insufficient to establish the originating parent.
    """
    _bound(parent,F,scored);table=scored.table
    eligible=set(allocate(table,parent.model.layer.h,policy='adam',seed=0).ids.tolist())
    return tuple(CandidateBlock(unit,table.contrasts[unit].detach().numpy(),unit in eligible,
        None if unit in eligible else 'legacy_allocation_guard:'+table.status[unit],2*(parent.model.layer.d+1))
        for unit in table.ids.tolist())


def joint_ridge_select(parent,F,candidates,config,*,scored):
    """Select IDs only; solved local parameter updates are never installed here."""
    started=time.perf_counter_ns()
    expected=build_gate_blocks(parent,F,scored)
    if (type(candidates) is not tuple or any(type(c) is not CandidateBlock for c in candidates)
            or [_record(c) for c in candidates]!=[_record(c) for c in expected]):
        raise ValueError('Candidate IDs/contrasts/eligibility differ from bound original table')
    if not isinstance(config,JointSelectionConfig):raise ValueError('Explicit immutable joint selection configuration required')
    x=parent.preprocessing.transform_X(F.X)
    y=parent.preprocessing.transform_y(F.y)
    contrasts={c.unit_id:c.contrast for c in candidates if c.eligible}
    bundle=prediction_jacobians(copy.deepcopy(parent.model).double().eval(),np.asarray(x,dtype=np.float64),contrasts)
    residual=np.asarray(y,dtype=np.float64)-bundle.prediction
    problem=prepare_joint_problem(bundle.parent,residual,bundle.gate_blocks,config)
    selected=problem.greedy(sorted(contrasts))
    eigenvalues=np.linalg.eigvalsh(problem._H)
    if len(eigenvalues) and eigenvalues[0]<=0:raise FloatingPointError('Nonpositive regularized gate Gram; no fallback')
    native_dtype=str(parent.model.layer.base.weight.dtype).removeprefix('torch.')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    report=dict(schema='research/v2-joint-selection',**selected,config=asdict(config),
        parent_sha256=scored.parent_sha256,table_sha256=scored.table_sha256,identities=copy.deepcopy(parent.identity_hashes),
        F_hashes={k:array_hash(getattr(F,k)) for k in ('X','y','row_id','group_id')},
        candidates=[_record(c) for c in candidates],candidate_sha256=_hash([_record(c) for c in candidates]),
        parameter_columns=list(bundle.parameter_columns),scaling=problem.scaling,parent_rank=problem.parent_rank,
        parent_svd_cutoff=problem.parent_svd_cutoff,parent_span_novelty_fraction=problem.novelty_fraction,
        parent_solve_relative_residual=problem.parent_solve_relative_residual,gram_asymmetry=problem.gram_asymmetry,
        gate_gram_condition_number=float(eigenvalues[-1]/eigenvalues[0]) if len(eigenvalues) else None,
        parent_objective=problem.objective([]),selected_objective=problem.objective(selected['selected_ids']),
        residual_F_MSE=float(residual@residual/len(residual)),local_update_applied=False,fallback_history=[],
        precision=dict(scoring='float64',native_parent=native_dtype,insertion_contrast=native_dtype),
        score_table_diagnostics=dict(adam_scores=_diagnostic_values(scored.table.adam_scores.tolist()),
            sgd_scores=_diagnostic_values(scored.table.sgd_scores.tolist()),
            signed_moment_eigenvalues=_diagnostic_values(scored.table.eigenvalues.tolist()),
            spectral_gaps=_diagnostic_values(scored.table.spectral_gaps.tolist()),
            nonfinite_encoding='IEEE spellings as strings; original tensor bytes bound by table_sha256'),
        cost=dict(elapsed_ns=time.perf_counter_ns()-started,upstream_scoring_ns=scored.elapsed_ns,
            jacobian_bytes=bundle.parent.nbytes+sum(b.nbytes for b in bundle.gate_blocks.values()),
            conditioned_gram_bytes=problem._H.nbytes,process_lifetime_peak_rss_bytes=peak,
            memory_scope='Whole process lifetime peak, not selector-incremental RAM',
            time_scope='Selector entry through numerical selection/diagnostics before final trace serialization; upstream scoring separate',
            standalone_parent_cost_ns=None))
    _bound(parent,F,scored)
    report['cost']['elapsed_ns']=time.perf_counter_ns()-started
    return SelectionTrace(_json(report))


def selection_to_allocation(parent,F,scored,trace,*,seed):
    """Verify the immutable selection against its unchanged native parent/table."""
    candidates=build_gate_blocks(parent,F,scored)
    if type(trace) is not SelectionTrace or type(seed) is not int or not 0<=seed<2**63:
        raise ValueError('Immutable selection trace and declared seed required')
    report=trace.to_dict();eligible={c.unit_id for c in candidates if c.eligible}
    config=JointSelectionConfig(**report['config']);ids=report['selected_ids']
    if (report['schema']!='research/v2-joint-selection' or report['parent_sha256']!=scored.parent_sha256
            or report['table_sha256']!=scored.table_sha256 or report['identities']!=parent.identity_hashes
            or report['candidate_sha256']!=_hash([_record(c) for c in candidates])
            or report['candidates']!=[_record(c) for c in candidates]
            or report['F_hashes']!={k:array_hash(getattr(F,k)) for k in ('X','y','row_id','group_id')}
            or report['requested_count']!=config.count or report['effective_count']!=min(config.count,len(eligible))
            or type(ids) is not list or any(type(i) is not int or i not in eligible for i in ids)
            or ids!=sorted(set(ids)) or len(ids)!=report['effective_count']
            or report['local_update_applied'] is not False):raise ValueError('Selection trace identity/count/candidate binding differs')
    _check_chain(report,config,eligible)
    contrasts=scored.table.contrasts[ids].clone() if ids else torch.empty((0,parent.model.layer.d+1),dtype=torch.float64)
    return Allocation(config.count,len(ids),torch.tensor(ids,dtype=torch.int64),contrasts,'joint_ridge',seed,
        scored.table_sha256,torch.zeros(len(ids),dtype=torch.int64),
        {'shared_scored_units':parent.model.layer.h,'standalone_contrast_units_required':0})
