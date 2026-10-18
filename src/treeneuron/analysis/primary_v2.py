"""Prospective teacher-level arithmetic; frozen provenance is a separate gate.

No method/data selection or protected access occurs here. A complete declared
inventory is required, including exact no-op teachers. The Student-t procedure is
primary; the whole-teacher bootstrap is a separately labeled sensitivity result.
"""
import hashlib
import math

import numpy as np
from scipy.stats import t

from treeneuron.config import canonical_hash
from treeneuron.research.ledger import TERMINAL
from treeneuron.research.seeds import seed64
from .design_v2 import TEACHER_GRID, USEFUL_MARGIN, teacher_log_benefits


def _names(values,name):
    if (not isinstance(values,list) or not values or any(type(v) is not str or not v for v in values)
            or len(set(values))!=len(values)):
        raise ValueError(f'{name} requires a nonempty unique declared order')
    return values


def holm_family(pvalues,*,planned_hypotheses,alpha=.025):
    """Adjust a complete explicit family; missing tests never shrink its size.

    Adjusted p values are not simultaneous lower confidence bounds. The eventual
    frozen analysis must identify the hypotheses and authorize claim wording.
    Strict rejection keeps equality from satisfying a strict useful-margin rule.
    """
    planned=_names(planned_hypotheses,'Multiplicity family')
    if (not isinstance(pvalues,dict) or set(pvalues)!=set(planned)
            or type(alpha) not in (float,int) or not math.isfinite(alpha) or not 0<alpha<1):
        raise ValueError('Complete explicit multiplicity family and valid alpha required')
    for value in pvalues.values():
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Every declared family p value must be finite in[0,1]')
    ordered=sorted(planned,key=lambda name:pvalues[name]);adjusted={};previous=0.
    for index,name in enumerate(ordered):
        previous=max(previous,min(1.,(len(planned)-index)*pvalues[name]));adjusted[name]=previous
    return dict(schema='research/v2-holm-family',planned_hypotheses=list(planned),
        raw_p={name:pvalues[name] for name in planned},adjusted_p={name:adjusted[name] for name in planned},
        rejected=[name for name in planned if adjusted[name]<alpha],alpha=alpha,
        ordering=ordered,family_size=len(planned),claim_authorized=False,
        interval_scope='Adjusted p values only; no simultaneous confidence interval is asserted')


def analyze_teacher_risks(planned_teacher_ids,records,*,seed,iterations=20000,chunk_size=256,
                          baseline_role='random'):
    """Analyze F-variance-normalized original-unit risks from exactly R2.

    For secondary comparisons, baseline_role names the frozen ordinary procedure;
    its risks are passed through the same R2-before-log arithmetic as random.
    No best seed, complete-case filtering, epsilon risk floor or favorable
    interval selection is permitted. This function does not verify raw metrics.
    """
    planned=_names(planned_teacher_ids,'Teacher inventory')
    if (len(planned)<2 or not isinstance(records,list) or type(seed) is not int or not 0<=seed<2**63
            or type(iterations) is not int or iterations<2 or type(chunk_size) is not int or chunk_size<1
            or type(baseline_role) is not str or not baseline_role or baseline_role=='selected'):
        raise ValueError('Explicit teacher/R2 inventory, baseline, seed and bootstrap policy required')
    seen={}
    for row in records:
        if not isinstance(row,dict):raise ValueError('Invalid teacher risk record')
        teacher=row.get('teacher_id');repeat=row.get('repetition');role=row.get('role');status=row.get('status')
        if (teacher not in planned or type(repeat) is not int or repeat not in (0,1)
                or role not in ('selected',baseline_role) or status not in TERMINAL):
            raise ValueError('Unplanned or invalid teacher/repetition/role/status')
        key=(teacher,repeat,role)
        if key in seen:raise ValueError('Duplicate teacher/repetition/role risk')
        risk=row.get('risk')
        if (risk is not None and (type(risk) not in (int,float) or not math.isfinite(risk) or risk<0)
                or status=='completed_eligible' and risk is None):
            raise ValueError('Invalid finite nonnegative risk')
        seen[key]=dict(teacher_id=teacher,repetition=repeat,role=role,status=status,risk=risk)
    expected=[(name,repeat,role) for name in planned for repeat in (0,1) for role in ('selected',baseline_role)]
    missing=[dict(teacher_id=key[0],repetition=key[1],role=key[2],
        status='absent' if key not in seen else seen[key]['status'])
        for key in expected if key not in seen or seen[key]['status']!='completed_eligible']
    base=dict(schema='research/v2-teacher-comparison',planned_teacher_ids=list(planned),teacher_count=len(planned),
        repetition_count=2,baseline_role=baseline_role,declared_cell_count=4*len(planned),
        observed_cell_count=len(seen),missing_or_failed_cells=missing,
        input_risks_sha256=canonical_hash(dict(records=[seen[key] for key in expected if key in seen])),
        teachers=[],student_t_interval=None,student_t_df=None,one_sided_margin_p=None,
        bootstrap_interval=None,bootstrap_mean_log_benefits=None,bootstrap_index_sha256=None,
        useful_margin=USEFUL_MARGIN,useful_margin_decision='unavailable',
        confirmation_count_valid=len(planned) in TEACHER_GRID,claim_authorized=False,
        protected_access_authorized=False,raw_metrics_authenticated=False,
        uncertainty_unit='Independent teacher; repetition risks averaged before log ratio',
        primary_procedure='Two-sided95% Student-t; useful lower bound strictly exceeds log(1/.98)',
        bootstrap_unit='whole teacher with both repetition risks averaged first',
        bootstrap_scope='Percentile sensitivity only; cannot replace the primary t interval')
    if missing:return dict(base,status='incomplete_predictions')
    # Retain zero-risk rows and identify undefined logarithms explicitly. They
    # cannot be converted to finite effects by an arbitrary positive constant.
    undefined=[]
    for name in planned:
        for role in ('selected',baseline_role):
            if math.fsum(seen[name,r,role]['risk']/2 for r in (0,1))==0:
                undefined.append(name);break
    if undefined:return dict(base,status='undefined_log_risk',undefined_teacher_ids=undefined)
    normalized=[{**seen[key],'role':'random' if key[2]==baseline_role else 'selected'} for key in expected]
    teachers=teacher_log_benefits(normalized);effects=np.array([row['log_benefit'] for row in teachers])
    # Shift before centering: an exactly constant representable sample must
    # remain exactly zero variance even when its ordinary floating mean differs.
    offset=effects-effects[0];center=math.fsum(offset)/len(effects)
    mean=float(effects[0])+center;centered=offset-center
    sd=math.sqrt(math.fsum(float(v)*float(v) for v in centered)/(len(effects)-1))
    se=sd/math.sqrt(len(effects));degrees=len(effects)-1
    if not all(math.isfinite(v) for v in (mean,sd,se)):raise ValueError('Nonfinite teacher inference arithmetic')
    rng=np.random.default_rng(seed64(seed,'teacher-comparison','whole-teacher-bootstrap'))
    draws=np.empty(iterations,dtype=np.float64);index_hash=hashlib.sha256()
    for start in range(0,iterations,chunk_size):
        count=min(chunk_size,iterations-start);indices=rng.integers(0,len(effects),size=(count,len(effects)),dtype=np.int64)
        index_hash.update(indices.astype('<i8',copy=False).tobytes())
        draws[start:start+count]=np.mean(centered[indices],axis=1)+mean
    base.update(teachers=teachers,mean_log_benefit=mean,median_log_benefit=float(np.median(effects)),
        teacher_signs=dict(positive=int(np.sum(effects>0)),zero=int(np.sum(effects==0)),negative=int(np.sum(effects<0))),
        teacher_log_benefit_range=[float(effects.min()),float(effects.max())],
        teacher_log_benefit_quantiles=np.quantile(effects,[.05,.95],method='linear').tolist(),
        teacher_sd=sd,standard_error=se,student_t_df=degrees,
        bootstrap_interval=np.quantile(draws,[.025,.975],method='linear').tolist(),
        bootstrap_mean_log_benefits=draws,bootstrap_index_sha256=index_hash.hexdigest(),
        bootstrap_iterations=iterations,bootstrap_seed=seed)
    if se==0:return dict(base,status='degenerate_teacher_variance',useful_margin_decision='undefined_degenerate_variance')
    critical=float(t.ppf(.975,degrees));interval=[mean-critical*se,mean+critical*se]
    p=float(t.sf((mean-USEFUL_MARGIN)/se,degrees))
    if not all(math.isfinite(v) for v in (*interval,p)):raise ValueError('Nonfinite teacher t result')
    decision='above_useful_margin' if interval[0]>USEFUL_MARGIN else (
        'below_useful_margin' if interval[1]<USEFUL_MARGIN else 'inconclusive')
    return dict(base,status='completed',student_t_interval=interval,one_sided_margin_p=p,
        useful_margin_decision=decision)
