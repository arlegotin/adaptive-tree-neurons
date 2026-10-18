"""Measure the gate-only first-Adam target without fitting or selecting on G."""
import copy
from dataclasses import dataclass
import time

import numpy as np
import torch

from treeneuron import train
from treeneuron.data.types import ObservationSet,array_hash,_readonly
from .diagnostics import ScoredParent,_parent
from .seeds import seed64


@dataclass(frozen=True)
class LocalCalibration:
    report: dict
    arrays: dict


@torch.enable_grad()
def calibrate_local_score(parent,F,scored,*,seed,minibatches=64,step_size=1e-7):
    """Minibatches are declared F-only samples, not the native training stream."""
    started=time.perf_counter_ns();seed64(seed,'validate')
    if (type(minibatches) is not int or minibatches<=0 or type(step_size) not in (int,float)
            or not np.isfinite(step_size) or step_size<=0):raise ValueError('Invalid local diagnostic budget')
    _,fingerprint=_parent(parent)
    if (not isinstance(scored,ScoredParent) or scored.parent_sha256!=fingerprint
            or scored.table.table_hash!=scored.table_sha256):raise ValueError('Changed parent/score binding')
    if (not isinstance(F,ObservationSet) or F.noise_free_target is not None
            or any(array_hash(getattr(F,k))!=parent.data_hashes['F.'+k] for k in ('X','y','row_id','group_id'))):
        raise ValueError('Local calibration requires exact F-only rows without evaluator targets')
    table=scored.table;rows=torch.nonzero(table.eligible,as_tuple=False).flatten()
    ids=table.ids[rows];p=parent.model.layer.d+1
    report=dict(schema='research/v2-local-calibration',status='completed_eligible',selection_labels='F_only',
        parent_sha256=fingerprint,table_sha256=scored.table_sha256,seed=seed,step_size=step_size,
        minibatches=minibatches,minibatch_policy='independent seeded samples without replacement within batch',
        loss_coordinates='F-fitted standardized target; mean squared error',
        scope='gate-only fresh-Adam first step; no claim about nonlinear joint continuation or G selection',
        units=[],gate_gradient_max_absolute_error=None,birth_contrast_gradient_max_absolute=None)
    arrays=dict(nominal_ids=table.ids.numpy(),nominal_eligible=table.eligible.numpy(),eligible_ids=ids.numpy())
    if len(ids)==0:
        report.update(status='completed_ineligible',reason='No current-eligible gate; local sensitivity unavailable',
            elapsed_ns=time.perf_counter_ns()-started)
        return LocalCalibration(report,{k:_readonly(v,v.dtype) for k,v in arrays.items()})
    before=train.capture_rng()
    try:
        model=copy.deepcopy(parent.model).double().eval()
        model.layer.grow(ids,table.contrasts[rows].clone())
        x=torch.tensor(parent.preprocessing.transform_X(F.X),dtype=torch.float64)
        y=torch.tensor(parent.preprocessing.transform_y(F.y),dtype=torch.float64)
        base_loss=torch.mean((model(x)-y)**2);base_loss.backward()
        observed=torch.cat((model.layer.gate.weight.grad,model.layer.gate.bias.grad[:,None]),dim=1).detach().clone()
        expected=table.gradients[rows]
        error=float(torch.max(torch.abs(observed-expected)))
        if error>1e-10+1e-8*float(torch.max(torch.abs(expected))):
            raise FloatingPointError('Analytic gate gradient disagrees with independent autograd')
        contrast_gradient=torch.cat((model.layer.contrast.weight.grad,model.layer.contrast.bias.grad[:,None]),dim=1)
        report.update(gate_gradient_max_absolute_error=error,
            birth_contrast_gradient_max_absolute=float(torch.max(torch.abs(contrast_gradient))),
            full_F_loss=float(base_loss.detach()))
        if torch.count_nonzero(contrast_gradient):raise FloatingPointError('Birth contrast derivative is nonzero')
        baseline=float(base_loss.detach());epsilon=table.adam_epsilon
        with torch.no_grad():
            initial_masks=model.layer(x)>0
            for j,unit in enumerate(ids.tolist()):
                direction=expected[j]/(torch.abs(expected[j])+epsilon)
                model.layer.gate.weight[j].copy_(-step_size*direction[:-1])
                model.layer.gate.bias[j].copy_(-step_size*direction[-1])
                actual=baseline-float(torch.mean((model(x)-y)**2))
                predicted=step_size*float(torch.sum(expected[j]*direction))
                masks=int(torch.count_nonzero((model.layer(x)>0)!=initial_masks))
                report['units'].append(dict(unit_id=unit,predicted_F_decrease=predicted,
                    observed_F_decrease=actual,relative_error=abs(actual-predicted)/predicted if predicted else None,
                    relu_mask_changes=masks))
                model.layer.gate.weight[j].zero_();model.layer.gate.bias[j].zero_()
        batch=min(parent.config.training.batch_size,len(F.y));rng=np.random.default_rng(seed)
        indices=np.stack([rng.choice(len(F.y),batch,replace=False) for _ in range(minibatches)])
        mini_gradients=[];decreases=[];ranks=[]
        from scipy.stats import spearmanr
        for index in indices:
            loss=torch.mean((model(x[index])-y[index])**2)
            weight,bias=torch.autograd.grad(loss,(model.layer.gate.weight,model.layer.gate.bias))
            g=torch.cat((weight,bias[:,None]),dim=1).detach()
            direction=g/(torch.abs(g)+epsilon)
            decreases.append((step_size*torch.sum(expected*direction,dim=1)).numpy())
            mini_gradients.append(g.numpy())
            mini_score=(torch.sum(g*direction,dim=1)/(2*p)).numpy();full=table.adam_scores[rows].numpy()
            ranks.append(float(spearmanr(full,mini_score).statistic) if len(ids)>1 and
                np.ptp(full)>0 and np.ptp(mini_score)>0 else None)
        arrays.update(full_gate_gradients=observed.numpy(),minibatch_row_indices=indices.astype(np.int64),
            minibatch_gate_gradients=np.stack(mini_gradients),minibatch_predicted_full_F_decreases=np.stack(decreases))
        report.update(minibatch_fraction_predicted_full_F_increase=float(np.mean(np.stack(decreases)<0)),
            minibatch_rank_correlations=ranks,undefined_rank_count=sum(r is None for r in ranks),
            elapsed_ns=time.perf_counter_ns()-started)
        if not all(np.isfinite(a).all() for a in arrays.values()):raise FloatingPointError('Nonfinite local diagnostics')
        return LocalCalibration(report,{k:_readonly(v,v.dtype) for k,v in arrays.items()})
    finally:train.restore_rng(before)
