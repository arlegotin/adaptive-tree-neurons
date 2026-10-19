"""Exact prediction Jacobians for the implemented ordinary first-growth parent.

Inputs are already in the parent's F-fitted model coordinates. The state/role
adapter owns that transformation; this helper never fits preprocessing.
"""
from dataclasses import dataclass

import numpy as np
import torch

from treeneuron.train import SelectiveRegressor
from treeneuron.models.units import SelectiveAffineLayer


@dataclass(frozen=True)
class PredictionJacobians:
    prediction:np.ndarray
    parent:np.ndarray
    gate_blocks:dict[int,np.ndarray]
    parameter_columns:tuple[dict,...]


def _array(tensor):
    value=tensor.detach().cpu().numpy().copy();value.setflags(write=False);return value


def prediction_jacobians(model,X,contrasts):
    """Differentiate all currently trainable parent parameters and birth gates.

    Preserve model parameters, existing gradients, mode, trainable flags and RNG.
    This exact version deliberately rejects other backbones and later growth.
    """
    expected=('layer.base.weight','layer.base.bias','head.weight','head.bias')
    if (type(model) is not SelectiveRegressor or type(model.layer) is not SelectiveAffineLayer
            or type(model.activation) is not torch.nn.ReLU
            or model.layer.growth_applied or model.layer.grown_ids.numel()
            or model.layer.gate is not None or model.layer.contrast is not None
            or tuple(name for name,_ in model.named_parameters())!=expected
            or model.head.in_features!=model.layer.h or model.head.out_features!=1
            or any(p.dtype!=torch.float64 or p.device.type!='cpu' or not torch.isfinite(p).all()
                for p in model.parameters())):
        raise ValueError('Exact ungrown scalar ReLU parent in CPU float64 required; no partial-parent fallback')
    if (not isinstance(X,np.ndarray) or X.dtype!=np.float64 or X.ndim!=2 or not len(X)
            or X.shape[1]!=model.layer.d or not np.isfinite(X).all()):
        raise ValueError('Finite float64 inputs in existing model coordinates required')
    if type(contrasts) is not dict:raise ValueError('Explicit original candidate contrast mapping required')
    for unit,c in contrasts.items():
        if (type(unit) is not int or not 0<=unit<model.layer.h or not isinstance(c,np.ndarray)
                or c.dtype!=np.float64 or c.shape!=(model.layer.d+1,) or not np.isfinite(c).all()):
            raise ValueError('Finite unchanged augmented float64 contrasts and valid unit IDs required')
    n=len(X);columns=[];parts=[];blocks={};offset=0
    with torch.no_grad():
        tx=torch.from_numpy(X.copy());base=model.layer.base(tx)
        mask=(base>0).to(torch.float64);factor=mask*model.head.weight[0]
        derivatives={'layer.base.weight':(factor[:,:,None]*tx[:,None,:]).reshape(n,-1),
            'layer.base.bias':factor,'head.weight':base.relu(),
            'head.bias':torch.ones((n,1),dtype=torch.float64,device='cpu')}
        for name,p in model.named_parameters():
            if not p.requires_grad:continue
            derivative=derivatives[name]
            if derivative.shape[1]!=p.numel():raise ValueError('Parent Jacobian omitted a trainable coordinate')
            columns.append(dict(name=name,shape=list(p.shape),start=offset,stop=offset+p.numel()))
            offset+=p.numel();parts.append(derivative)
        parent=torch.cat(parts,dim=1) if parts else torch.empty((n,0),dtype=torch.float64,device='cpu')
        augmented=torch.cat([tx,torch.ones((n,1),dtype=torch.float64,device='cpu')],dim=1)
        for unit in sorted(contrasts):
            c=torch.from_numpy(contrasts[unit].copy())
            block=factor[:,unit,None]*(augmented@c)[:,None]*augmented/(2.*model.layer.tau)
            blocks[unit]=_array(block)
        prediction=_array(model(tx));parent=_array(parent)
    if not np.isfinite(parent).all() or not np.isfinite(prediction).all() or any(not np.isfinite(b).all() for b in blocks.values()):
        raise FloatingPointError('Nonfinite exact prediction Jacobian; no unit eligibility fallback')
    return PredictionJacobians(prediction,parent,blocks,tuple(columns))
