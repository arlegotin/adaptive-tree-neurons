"""Exact normalized-column joint ridge geometry; no model fitting or role reads.

This numerical core does not yet supply the parent-Jacobian/Allocation adapter.
Positive ridge permits parent/gate penalty sharing; zero parent ridge measures
additional span using one fixed orthogonal parent basis.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import solve_triangular


@dataclass(frozen=True)
class JointSelectionConfig:
    count:int=16
    parent_ridge:float=.01
    gate_ridge:float=.01
    condition_on_parent:bool=True
    column_scaling:str='rms'
    solver:str='exact_float64'
    tie_relative_tolerance:float=1e-12

    def __post_init__(self):
        if (type(self.count) is not int or self.count<1 or type(self.condition_on_parent) is not bool
                or self.column_scaling!='rms' or self.solver!='exact_float64'):
            raise ValueError('Explicit positive count, boolean conditioning and exact RMS/float64 policy required')
        for name,strict in (('parent_ridge',False),('gate_ridge',True),('tie_relative_tolerance',False)):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0 or strict and value==0:
                raise ValueError('Finite nonnegative parent/tie values and positive gate ridge required')
            object.__setattr__(self,name,float(value))


def _matrix(value,n=None,*,allow_empty=False):
    if (not isinstance(value,np.ndarray) or value.dtype!=np.float64 or value.ndim!=2
            or not len(value) or n is not None and len(value)!=n or not np.isfinite(value).all()
            or not allow_empty and not value.shape[1]):
        raise ValueError('Finite aligned CPU float64 matrices required')
    return value.copy()


def _columns(matrix):
    with np.errstate(over='raise',invalid='raise'):
        rms=np.sqrt(np.mean(matrix*matrix,axis=0))
    cutoff=max(1e-15,1e-12*float(np.max(rms,initial=0.)))
    mask=rms>cutoff;normalized=matrix[:,mask]/rms[mask]
    return normalized,dict(rms=rms.tolist(),keep_mask=mask.tolist(),zero_threshold=cutoff)


def _ids(values,known):
    if (not isinstance(values,(list,tuple)) or any(type(v) is not int or v not in known for v in values)
            or len(set(values))!=len(values)):
        raise ValueError('Unique known Python integer candidate IDs required')
    return tuple(sorted(values))


def _factor(matrix,rhs):
    if not len(rhs):return np.empty((0,0)),np.empty(0)
    lower=np.linalg.cholesky(matrix)
    return lower,solve_triangular(lower,rhs,lower=True,check_finite=True)


class _JointProblem:
    def __init__(self,parent,residual,blocks,config):
        if not isinstance(config,JointSelectionConfig):raise ValueError('Immutable JointSelectionConfig required')
        self.config=config;parent=_matrix(parent,allow_empty=True);n=len(parent)
        if (not isinstance(residual,np.ndarray) or residual.dtype!=np.float64
                or residual.shape!=(n,) or not np.isfinite(residual).all()):
            raise ValueError('One finite aligned float64 residual without further standardization required')
        if type(blocks) is not dict or any(type(unit) is not int or unit<0 for unit in blocks):
            raise ValueError('Unique nonnegative integer candidate block IDs required')
        residual=residual.copy();normalized,parent_scaling=_columns(parent)
        pieces=[];self._columns={};self.scaling=dict(parent=parent_scaling,gate={});offset=0
        for unit in sorted(blocks):
            block=_matrix(blocks[unit],n);piece,scaling=_columns(block)
            self.scaling['gate'][unit]=scaling;pieces.append(piece)
            self._columns[unit]=list(range(offset,offset+piece.shape[1]));offset+=piece.shape[1]
        gates=np.concatenate(pieces,axis=1) if pieces else np.empty((n,0))
        if normalized.shape[1]:
            U,s,_=np.linalg.svd(normalized,full_matrices=False);cutoff=1e-10*float(s[0]);basis=U[:,s>cutoff]
        else:basis=np.empty((n,0));cutoff=0.
        self.parent_rank=basis.shape[1];self.parent_svd_cutoff=cutoff
        self.novelty_fraction={}
        outside=gates-basis@(basis.T@gates) if self.parent_rank<n else np.zeros_like(gates)
        for unit,columns in self._columns.items():
            norm=float(np.linalg.norm(gates[:,columns]))
            self.novelty_fraction[unit]=float(np.linalg.norm(outside[:,columns]))/norm if norm else 0.
        self.parent_solve_relative_residual=0.
        if not config.condition_on_parent or not normalized.shape[1]:
            H=gates.T@gates/n+config.gate_ridge*np.eye(offset);s=gates.T@residual/n
            q0=float(residual@residual/n)
        elif config.parent_ridge==0:
            r=residual-basis@(basis.T@residual) if self.parent_rank<n else np.zeros_like(residual)
            H=outside.T@outside/n+config.gate_ridge*np.eye(offset);s=outside.T@r/n;q0=float(r@r/n)
        else:
            A=normalized.T@normalized/n+config.parent_ridge*np.eye(normalized.shape[1])
            B=normalized.T@gates/n;d=normalized.T@residual/n
            rhs=np.column_stack([B,d]);lower=np.linalg.cholesky(A)
            solved=solve_triangular(lower.T,solve_triangular(lower,rhs,lower=True),lower=False)
            self.parent_solve_relative_residual=float(np.linalg.norm(A@solved-rhs))/max(1.,float(np.linalg.norm(rhs)))
            H=gates.T@gates/n+config.gate_ridge*np.eye(offset)-B.T@solved[:,:offset]
            s=gates.T@residual/n-B.T@solved[:,-1]
            q0=float(residual@residual/n-d@solved[:,-1])
        self.gram_asymmetry=float(np.max(np.abs(H-H.T),initial=0.))
        self._H=(H+H.T)/2.;self._s=s.copy();self._q0=q0
        if not np.isfinite(self._H).all() or not np.isfinite(s).all() or not math.isfinite(q0) or q0<-1e-10:
            raise FloatingPointError('Nonfinite/negative joint objective; no numerical fallback admitted')
        if q0<0:raise FloatingPointError('Parent objective requires explicit numerical diagnosis')
        self._H.setflags(write=False);self._s.setflags(write=False)

    def _subset(self,ids):
        ids=_ids(ids,self._columns)
        return [column for unit in ids for column in self._columns[unit]]

    def gain(self,ids):
        columns=self._subset(ids);_,z=_factor(self._H[np.ix_(columns,columns)],self._s[columns])
        gain=float(z@z)
        if not math.isfinite(gain) or gain>self._q0+1e-10*max(1.,abs(self._q0)):
            raise FloatingPointError('Joint gain contradicts the regularized objective')
        return gain

    def objective(self,ids):return self._q0-self.gain(ids)

    def singleton_top(self,eligible_ids):
        """Independent singleton ranking control, with the true selected-set gain."""
        eligible=list(_ids(eligible_ids,self._columns));effective=min(self.config.count,len(eligible))
        gains={unit:self.gain([unit]) for unit in eligible};order=[]
        for _ in range(effective):
            remaining=[j for j in eligible if j not in order];best=max(gains[j] for j in remaining)
            tolerance=self.config.tie_relative_tolerance*max(1.,abs(best))
            order.append(min(j for j in remaining if gains[j]>=best-tolerance))
        return dict(selected_ids=sorted(order),selection_order=order,requested_count=self.config.count,
            effective_count=effective,eligible_count=len(eligible),predicted_gain=self.gain(order),slots=[],
            singleton_gains=[dict(unit_id=j,gain=gains[j]) for j in eligible])

    def _conditional(self,selected,lower,z,unit):
        columns=self._columns[unit];cross=self._H[np.ix_(selected,columns)]
        projection=solve_triangular(lower,cross,lower=True) if selected else np.empty((0,len(columns)))
        H=self._H[np.ix_(columns,columns)]-projection.T@projection
        s=self._s[columns]-projection.T@z
        newborn,newz=_factor((H+H.T)/2.,s);gain=float(newz@newz)
        if not math.isfinite(gain):raise FloatingPointError('Nonfinite conditional gain')
        return gain,projection,newborn,newz

    def marginal_gain(self,ids,candidate):
        ids=_ids(ids,self._columns)
        if type(candidate) is not int or candidate not in self._columns or candidate in ids:
            raise ValueError('One remaining known candidate required')
        selected=self._subset(ids);lower,z=_factor(self._H[np.ix_(selected,selected)],self._s[selected])
        return self._conditional(selected,lower,z,candidate)[0]

    def greedy(self,eligible_ids):
        eligible=list(_ids(eligible_ids,self._columns));effective=min(self.config.count,len(eligible))
        order=[];columns=[];lower=np.empty((0,0));z=np.empty(0);slots=[]
        for slot in range(effective):
            choices={unit:self._conditional(columns,lower,z,unit) for unit in eligible if unit not in order}
            best=max(v[0] for v in choices.values());tol=self.config.tie_relative_tolerance*max(1.,abs(best))
            unit=min(unit for unit,value in choices.items() if value[0]>=best-tol)
            gain,projection,newborn,newz=choices[unit]
            diagnostics=[]
            for j,(_,project,Lj,zj) in sorted(choices.items()):
                block=self._columns[j]
                Hj=self._H[np.ix_(block,block)]-project.T@project
                sj=self._s[block]-project.T@z
                solved=solve_triangular(Lj.T,zj,lower=False) if len(zj) else np.empty(0)
                error=float(np.linalg.norm(Hj@solved-sj))/max(1.,float(np.linalg.norm(sj)))
                diagnostics.append(dict(unit_id=j,retained_columns=len(zj),relative_solve_residual=error,
                    condition_number=float(np.linalg.cond(Lj)**2) if len(zj) else None))
            slots.append(dict(slot=slot,unit_id=unit,marginal_gain=gain,
                candidate_gains=[dict(unit_id=j,gain=choices[j][0]) for j in sorted(choices)],
                numerical_diagnostics=diagnostics))
            if len(newz):
                lower=np.block([[lower,np.zeros((len(columns),len(newz)))],[projection.T,newborn]])
                z=np.r_[z,newz]
            columns+=self._columns[unit];order.append(unit)
        return dict(selected_ids=sorted(order),selection_order=order,requested_count=self.config.count,
            effective_count=effective,eligible_count=len(eligible),predicted_gain=float(z@z),slots=slots)


def prepare_joint_problem(parent_jacobian,residual,gate_blocks,config):
    """Prepare exact F-array geometry; gate columns retain their original scales."""
    return _JointProblem(parent_jacobian,residual,gate_blocks,config)
