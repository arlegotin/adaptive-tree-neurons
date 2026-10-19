"""Separate finite v2 model/optimizer birth core; legacy guards are unchanged.

These primitives do not qualify a training workload or predictive model. The
owning adapter must retain and validate complete stage/sampler/schedule state.
"""
import copy
import math

import torch
from torch import nn

from treeneuron.train import SelectiveRegressor


def _dimensions(d,h):
    if type(d) is not int or not 1<=d<=128 or type(h) is not int or not 1<=h<=128:
        raise ValueError('Outside the separate v2 model geometry')


def ordinary_brackets(d,h,count):
    _dimensions(d,h)
    if type(count) is not int or not 0<=count<=h:raise ValueError('Invalid effective growth count')
    numerator=h*(d+2)+2*count*(d+1)
    lower=numerator//(d+2);upper=(numerator+d+1)//(d+2)
    return lower,upper


def parent_model(d,h,tau):
    _dimensions(d,h)
    return SelectiveRegressor(d,h,tau)


def _ownership(model,optimizer):
    if (not isinstance(optimizer,torch.optim.AdamW) or optimizer.defaults.get('foreach') is not False
            or optimizer.defaults.get('amsgrad') or optimizer.defaults.get('capturable')):
        raise ValueError('Require the explicit ordinary AdamW birth policy')
    parameters=list(model.parameters());owned=[p for g in optimizer.param_groups for p in g['params']]
    ids={id(p) for p in parameters}
    if (not owned or len(owned)!=len({id(p) for p in owned}) or {id(p) for p in owned}!=ids
            or not {id(p) for p in optimizer.state}<=ids):
        raise ValueError('Optimizer ownership does not match the complete parent')
    ages=set()
    for p in parameters:
        if p.dtype!=torch.float32 or p.device.type!='cpu' or not bool(torch.isfinite(p).all()):
            raise ValueError('Native parent parameters must be finite CPU float32')
        state=optimizer.state.get(p,{})
        if not state:ages.add(0);continue
        if set(state)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Unknown Adam state')
        step=state['step']
        if (not isinstance(step,torch.Tensor) or step.shape!=() or step.dtype!=torch.float32
                or step.device.type!='cpu' or not bool(torch.isfinite(step))
                or float(step)<0 or not float(step).is_integer()):raise ValueError('Invalid Adam age')
        ages.add(int(step))
        for name in ('exp_avg','exp_avg_sq'):
            value=state[name]
            if (not isinstance(value,torch.Tensor) or value.dtype!=p.dtype or value.device!=p.device
                    or value.shape!=p.shape or not bool(torch.isfinite(value).all())
                    or name=='exp_avg_sq' and bool((value<0).any())):
                raise ValueError('Invalid Adam moment shape/dtype/sign')
    if len(ages)!=1:raise ValueError('Unfrozen parent parameter applied-update ages disagree')
    return parameters


def _register(model,optimizer,old_parameters):
    names={id(p):name for name,p in model.named_parameters()};old_ids={id(p) for p in old_parameters}
    new=[p for p in model.parameters() if id(p) not in old_ids]
    groups=list(optimizer.param_groups)
    previous_names=[copy.deepcopy(g.get('param_names')) for g in groups]
    try:
        if new:
            settings={k:copy.deepcopy(v) for k,v in groups[0].items() if k not in ('params','param_names')}
            optimizer.add_param_group(dict(settings,params=new,param_names=[names[id(p)] for p in new]))
            for p in new:
                optimizer.state[p]=dict(step=torch.tensor(0.,dtype=torch.float32),
                    exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        for group in groups:group['param_names']=[names[id(p)] for p in group['params']]
    except BaseException:
        optimizer.param_groups[:]=groups
        for p in new:optimizer.state.pop(p,None)
        for group,previous in zip(groups,previous_names):
            if previous is None:group.pop('param_names',None)
            else:group['param_names']=previous
        raise


def insert_gates(model,optimizer,ids,contrasts):
    if not isinstance(model,SelectiveRegressor) or model.layer.growth_applied:
        raise ValueError('Only the ungrown v2 selective parent can receive gates')
    old=_ownership(model,optimizer);layer=model.layer
    before_rng=torch.get_rng_state().clone();before_ids=layer.grown_ids.clone()
    try:
        manifest=layer.grow(ids,contrasts)
        _register(model,optimizer,old)
        return manifest
    except BaseException:
        layer.gate=None;layer.contrast=None;layer.grown_ids=before_ids;layer.growth_applied=False
        torch.set_rng_state(before_rng)
        raise


class _WarmOrdinary(nn.Module):
    def __init__(self,parent,width,seed):
        super().__init__();self.incumbent=parent;self.width=width
        d,h=parent.layer.d,parent.layer.h
        self.added_affine=None;self.added_head=None
        if width>h:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.added_affine=nn.Linear(d,width-h)
                self.added_head=nn.Linear(width-h,1,bias=False)
                with torch.no_grad():self.added_head.weight.zero_()
        self.train(parent.training)

    def forward(self,X):
        value=self.incumbent(X)
        return value if self.added_head is None else value+self.added_head(torch.relu(self.added_affine(X))).squeeze(-1)


def widen_ordinary(parent,optimizer,width,*,seed):
    old=_ownership(parent,optimizer)
    if (not isinstance(parent,SelectiveRegressor) or parent.layer.growth_applied
            or type(width) is not int or not parent.layer.h<=width<=256
            or type(seed) is not int or not 0<=seed<2**63):
        raise ValueError('Invalid v2 ordinary growth topology or seed')
    model=_WarmOrdinary(parent,width,seed)
    _register(model,optimizer,old)
    return model


class StaticModel(nn.Module):
    def __init__(self,d,h,count,coordinates,*,tau,dropout):
        super().__init__();_dimensions(d,h)
        if (type(count) is not int or not 0<=count<=h or coordinates not in ('centered','direct_leaf')
                or type(tau) not in (int,float) or not math.isfinite(tau) or tau<=0
                or type(dropout) not in (int,float) or dropout not in (0.,.1,.2)):
            raise ValueError('Invalid separate v2 static topology')
        self.d,self.h,self.count,self.coordinates,self.tau=d,h,count,coordinates,float(tau)
        self.ordinary=nn.Linear(d,h-count) if count<h else None
        self.gate=nn.Linear(d,count) if count else None
        self.center=nn.Linear(d,count) if count and coordinates=='centered' else None
        self.contrast=nn.Linear(d,count) if count and coordinates=='centered' else None
        self.left=nn.Linear(d,count) if count and coordinates=='direct_leaf' else None
        self.right=nn.Linear(d,count) if count and coordinates=='direct_leaf' else None
        self.dropout=nn.Dropout(float(dropout));self.head=nn.Linear(h,1)

    def forward(self,X):
        pieces=[]
        if self.count:
            if self.coordinates=='centered':
                pieces.append(self.center(X)+torch.tanh(self.gate(X)/(2*self.tau))*self.contrast(X))
            else:
                q=torch.sigmoid(self.gate(X)/self.tau)
                pieces.append(q*self.left(X)+(1-q)*self.right(X))
        if self.ordinary is not None:pieces.append(self.ordinary(X))
        return self.head(self.dropout(torch.relu(torch.cat(pieces,dim=1)))).squeeze(-1)
