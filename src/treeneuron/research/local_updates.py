"""Temporary F-solved nonlinear checks, without choosing a fraction on G."""
import copy
import time

import numpy as np
import torch

from treeneuron import train
from treeneuron.data.types import ObservationSet,array_hash
from .jacobian import prediction_jacobians
from .joint_allocation import selection_to_allocation
from .joint_selector import JointSelectionConfig,prepare_joint_problem


def _solve(bundle,residual,config,ids):
    """Direct augmented ridge solve with original fixed per-block RMS scales."""
    blocks={i:bundle.gate_blocks[i] for i in ids}
    problem=prepare_joint_problem(bundle.parent,residual,blocks,config)
    scales=problem.scaling;pmask=np.asarray(scales['parent']['keep_mask'],dtype=bool)
    prms=np.asarray(scales['parent']['rms'],dtype=np.float64)
    J=bundle.parent[:,pmask]/prms[pmask]
    gate_parts=[];gate_slices={};offset=0
    for unit in ids:
        metadata=scales['gate'][unit];mask=np.asarray(metadata['keep_mask'],dtype=bool)
        rms=np.asarray(metadata['rms'],dtype=np.float64)
        part=blocks[unit][:,mask]/rms[mask]
        gate_parts.append(part);gate_slices[unit]=(offset,offset+part.shape[1],mask,rms);offset+=part.shape[1]
    gates=np.column_stack(gate_parts) if gate_parts else np.empty((len(residual),0))
    retained=None
    if not config.condition_on_parent:parent_design=np.empty((len(residual),0))
    elif config.parent_ridge==0 and J.shape[1]:
        U,s,Vh=np.linalg.svd(J,full_matrices=False);keep=s>1e-10*s[0]
        parent_design=U[:,keep];retained=(s[keep],Vh[keep])
    else:parent_design=J
    npcols=parent_design.shape[1]
    design=np.column_stack([parent_design,gates])
    penalties=np.r_[np.full(npcols,config.parent_ridge),np.full(gates.shape[1],config.gate_ridge)]
    if design.shape[1]:
        augmented=np.vstack([design,np.diag(np.sqrt(len(residual)*penalties))])
        coef,_,rank,singular=np.linalg.lstsq(augmented,np.r_[residual,np.zeros(design.shape[1])],rcond=None)
        cutoff=float(max(augmented.shape)*np.finfo(np.float64).eps*float(singular[0]))
        if rank!=design.shape[1]:raise FloatingPointError('Unexpected augmented-solve rank truncation; no fallback')
    else:coef=np.empty(0);rank=0;cutoff=0.;singular=np.empty(0)
    parent_scaled=np.zeros(J.shape[1])
    if config.condition_on_parent:
        parent_scaled=(retained[1].T@(coef[:npcols]/retained[0])) if retained is not None else coef[:npcols]
    parent_delta=np.zeros(bundle.parent.shape[1]);parent_delta[pmask]=parent_scaled/prms[pmask]
    gate_delta={}
    for unit,(start,end,mask,rms) in gate_slices.items():
        delta=np.zeros(len(mask));delta[mask]=coef[npcols+start:npcols+end]/rms[mask];gate_delta[unit]=delta
    penalty=float(np.dot(penalties,coef*coef))
    objective=float(np.mean((residual-design@coef)**2)+penalty)
    if not np.isfinite(parent_delta).all() or any(not np.isfinite(v).all() for v in gate_delta.values()):
        raise FloatingPointError('Nonfinite temporary local solution; no fallback')
    if not np.isclose(objective,problem.objective(ids),rtol=1e-8,atol=1e-10):
        raise FloatingPointError('Direct local update and Schur objective disagree')
    metadata=dict(solver='numpy_lstsq_augmented',rcond=None,columns=design.shape[1],retained_rank=int(rank),
        singular_cutoff=cutoff,singular_values=singular.tolist(),
        parent_retained_rank=problem.parent_rank,parent_svd_cutoff=problem.parent_svd_cutoff,
        matched_Schur_objective_error=abs(objective-problem.objective(ids)))
    return parent_delta,gate_delta,penalty,objective,problem.objective([]),metadata


def diagnose_local_updates(parent,F,G,scored,trace):
    """Report every fixed fraction; never mutate the parent or select using G."""
    started=time.perf_counter_ns()
    allocation=selection_to_allocation(parent,F,scored,trace,seed=0)
    if (not isinstance(G,ObservationSet) or G.noise_free_target is not None
            or any(array_hash(getattr(G,k))!=parent.data_hashes['G.'+k] for k in ('X','y','row_id','group_id'))):
        raise ValueError('Exact G diagnostic role, without evaluator targets, required')
    config=JointSelectionConfig(**trace.to_dict()['config']);ids=allocation.ids.tolist()
    contrasts={unit:allocation.contrasts[row].numpy().copy() for row,unit in enumerate(ids)}
    model=copy.deepcopy(parent.model).double().eval()
    X={role:np.asarray(parent.preprocessing.transform_X(data.X),dtype=np.float64) for role,data in (('F',F),('G',G))}
    y={role:np.asarray(parent.preprocessing.transform_y(data.y),dtype=np.float64) for role,data in (('F',F),('G',G))}
    bundles={role:prediction_jacobians(model,value,contrasts) for role,value in X.items()}
    residual=y['F']-bundles['F'].prediction
    u,v,penalty,objective,parent_objective,solve=_solve(bundles['F'],residual,config,ids)
    if not np.isclose(objective,trace.to_dict()['selected_objective'],rtol=1e-8,atol=1e-10):
        raise FloatingPointError('Temporary update does not match the selected-set objective')
    tensors={role:torch.from_numpy(value.copy()) for role,value in X.items()}
    with torch.no_grad():masks={role:model.layer(value)>0 for role,value in tensors.items()}
    rng=train.capture_rng();evaluations=[]
    try:
        for fraction in (1/16,1/4,1):
            temporary=copy.deepcopy(model)
            if ids:temporary.layer.grow(allocation.ids,allocation.contrasts)
            with torch.no_grad():
                parameters=dict(temporary.named_parameters())
                for column in bundles['F'].parameter_columns:
                    delta=u[column['start']:column['stop']].reshape(column['shape'])
                    parameters[column['name']].add_(torch.from_numpy(delta.copy()),alpha=fraction)
                if ids:
                    gate=np.stack([v[unit] for unit in ids])
                    temporary.layer.gate.weight.copy_(torch.from_numpy(fraction*gate[:,:-1]))
                    temporary.layer.gate.bias.copy_(torch.from_numpy(fraction*gate[:,-1]))
                row=dict(fraction=fraction,penalty=fraction**2*penalty)
                for role,bundle in bundles.items():
                    update=bundle.parent@u
                    for unit in ids:update+=bundle.gate_blocks[unit]@v[unit]
                    actual=temporary(tensors[role]).numpy()
                    linear=bundle.prediction+fraction*update
                    changed=float(((temporary.layer(tensors[role])>0)!=masks[role]).double().mean())
                    saturation=float((torch.tanh(temporary.layer.gate(tensors[role])/(2*temporary.layer.tau)).abs()>.99).double().mean()) if ids else 0.
                    if not np.isfinite(actual).all() or not np.isfinite(linear).all():
                        raise FloatingPointError('Nonfinite temporary prediction; no fraction selection/fallback')
                    row[role]=dict(linearized_MSE=float(np.mean((y[role]-linear)**2)),
                        nonlinear_MSE=float(np.mean((y[role]-actual)**2)),
                        ReLU_mask_changed_fraction=changed,gate_saturation_fraction=saturation)
                row['predicted_penalized_F_objective']=row['F']['linearized_MSE']+row['penalty']
                evaluations.append(row)
    finally:train.restore_rng(rng)
    selection_to_allocation(parent,F,scored,trace,seed=0)
    return dict(schema='research/v2-local-update-diagnostic',trace_sha256=trace.sha256,
        identities=copy.deepcopy(parent.identity_hashes),G_hashes={k:array_hash(getattr(G,k)) for k in ('X','y','row_id','group_id')},
        fractions=[1/16,1/4,1],fraction_selected_on_G=False,local_updates_applied_to_parent=False,
        precision=dict(parent_source=str(parent.model.layer.base.weight.dtype).removeprefix('torch.'),temporary_model='float64',gate_contrasts='float64'),
        loss_units='Existing F-fitted target coordinates; no residual rescaling',
        current_parent_MSE={role:float(np.mean((y[role]-bundle.prediction)**2)) for role,bundle in bundles.items()},
        optimized_parent_objective=parent_objective,full_step_penalized_objective=objective,
        linear_solve=solve,solution=dict(parent_delta=u.tolist(),parent_parameter_columns=list(bundles['F'].parameter_columns),
            gate_delta={str(unit):value.tolist() for unit,value in v.items()}),
        penalized_local_gain=parent_objective-objective,evaluations=evaluations,
        elapsed_ns=time.perf_counter_ns()-started,
        interpretation='Penalized gain is relative to optimized local parent; raw MSE and nonlinear changes are separate diagnostics')
