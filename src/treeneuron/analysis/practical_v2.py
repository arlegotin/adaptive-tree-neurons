"""Practical metric/V-selection arithmetic; no protected access or claim approval.

Callers supply original-unit arrays and the variance fitted on F. These pure
functions neither open data nor authorize an evaluator. Bootstrap uncertainty,
protocol validation and full search resource accounting remain separate.
"""
import math

import numpy as np

from treeneuron.research.ledger import TERMINAL

OBSERVATIONAL=('airfoil','concrete','ccpp','parkinsons','bike')
FAMILIES=tuple(f'P{i:02}' for i in range(14))
CHAMPION=FAMILIES[2:]


def _nonnegative(value,name):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _cost(value):
    if type(value) is not int or value<0:raise ValueError('Cost must be nonnegative integer nanoseconds')
    return value


def grouped_risks(y,predictions,group_ids,*,fit_target_variance):
    y=np.asarray(y,dtype=np.float64);groups=np.asarray(group_ids,dtype=object)
    variance=_nonnegative(fit_target_variance,'F target variance')
    if (y.ndim!=1 or not len(y) or not np.isfinite(y).all() or groups.shape!=y.shape
            or variance==0 or not isinstance(predictions,dict) or not predictions):
        raise ValueError('Invalid complete grouped prediction/target/variance inputs')
    keys=[];indices={}
    for i,group in enumerate(groups.tolist()):
        if type(group) not in (str,int):raise ValueError('Group IDs must be explicit integer or string clusters')
        key=('str:' if isinstance(group,str) else 'int:')+str(group)
        if key not in indices:keys.append(key);indices[key]=[]
        indices[key].append(i)
    methods={}
    for name,values in predictions.items():
        if name not in FAMILIES:raise ValueError('Unknown practical family')
        prediction=np.asarray(values,dtype=np.float64)
        if prediction.shape!=(len(y),2) or not np.isfinite(prediction).all():
            raise ValueError('Complete paired R2 prediction arrays required')
        with np.errstate(over='raise',invalid='raise'):
            try:
                residual=prediction-y[:,None];squared=residual**2
                clustered=np.asarray([np.mean(squared[indices[key]],axis=0)/variance for key in keys])
                repeat=np.mean(clustered,axis=0);row_mse=np.mean(squared,axis=0)
                row_mae=np.mean(np.abs(residual),axis=0)
            except FloatingPointError as error:raise ValueError('Prediction error overflows metric arithmetic') from error
        methods[name]=dict(group_repeat_nmse=clustered,repeat_risk=repeat.tolist(),
            mean_repeat_risk=math.fsum(repeat)/2,row_repeat_nmse=(row_mse/variance).tolist(),
            row_repeat_rmse=np.sqrt(row_mse).tolist(),row_repeat_mae=row_mae.tolist())
    return dict(group_ids=keys,group_sizes=[len(indices[key]) for key in keys],methods=methods,
        independent_unit='held-out group/day within a fixed fitted split; repeats are paired randomizations',
        confirmed_useful_win=False)


def select_completed_trial(records,*,deadline_ns):
    deadline_ns=_cost(deadline_ns)
    if not isinstance(records,list) or len(records)>32:raise ValueError('Finite search allows at most32paid attempts')
    identities=set();seen=set();attempts=set();eligible=[];total=0;prefix=0;previous_completion=0
    for record in records:
        family=record['family_id'];repeat=record['search_repeat'];index=record['attempt_index'];status=record['status']
        if family not in FAMILIES or type(repeat) is not int or repeat not in (0,1):raise ValueError('Invalid search family/repeat')
        identities.add((family,repeat))
        if len(identities)>1:raise ValueError('Search family/repeat cannot be selected as a candidate')
        if type(index) is not int or index<0 or index>=32 or index in seen or record['attempt_id'] in attempts:
            raise ValueError('Invalid or duplicate paid attempt identity')
        seen.add(index);attempts.add(record['attempt_id'])
        if status not in TERMINAL:raise ValueError('Prefix selection requires retained terminal attempt records')
        end=_cost(record['completion_ns']);cost=_cost(record['charged_ns'])
        if end<previous_completion or cost>end-previous_completion:raise ValueError('Sequential attempt clock/cost contradiction')
        previous_completion=end;total+=cost;prefix+=min(cost,max(0,deadline_ns-(end-cost)))
        if status=='completed_eligible':
            score=_nonnegative(record['V_score'],'V normalized MSE')
            if end<=deadline_ns:eligible.append((score,index,record))
    if [r['attempt_index'] for r in records]!=list(range(len(records))):
        raise ValueError('Incomplete or unordered paid attempt inventory')
    if eligible:
        best=min(row[0] for row in eligible);tolerance=1e-12*max(1.,abs(best))
        winner=min((r for r in eligible if r[0]<=best+tolerance),key=lambda r:r[1])[2].copy()
    else:winner=None
    return dict(winner=winner,eligible_at_prefix=len(eligible),attempts_paid=len(records),
        total_charged_ns=total,prefix_charged_ns=prefix,deadline_ns=deadline_ns,
        cost_scope='Supplied charged trial intervals; enclosing coordinator overhead remains separately required')


def select_champion(family_winners):
    if not isinstance(family_winners,dict) or set(family_winners)!=set(CHAMPION):
        raise ValueError('Portfolio requires all12fixed P02-P13 family records')
    cost=0;eligible=[];repeats=set();missing=[]
    for name in CHAMPION:
        row=family_winners[name];cost+=_cost(row['total_charged_ns']);repeat=row['search_repeat']
        if type(repeat) is not int or repeat not in (0,1):raise ValueError('Invalid portfolio search repeat')
        repeats.add(repeat)
        if row['status']!='completed_eligible':missing.append(name);continue
        eligible.append((_nonnegative(row['V_score'],'V normalized MSE'),name,row))
    if len(repeats)!=1:raise ValueError('Portfolio cannot select a search repeat')
    base=dict(portfolio_charged_ns=cost,portfolio_family_count=12,equal_total_cost_comparison=False,
        search_repeat=next(iter(repeats)),missing_or_ineligible_families=missing)
    if missing:return dict(base,status='incomplete_portfolio',family_id=None,winner=None)
    best=min(row[0] for row in eligible);tol=1e-12*max(1.,abs(best))
    score,name,row=min((r for r in eligible if r[0]<=best+tol),key=lambda r:r[1])
    return dict(base,status='completed_eligible',family_id=name,winner=row.copy(),V_score=score)


def practical_benefit(dataset_risks):
    if not isinstance(dataset_risks,dict) or not set(OBSERVATIONAL)<=set(dataset_risks) or set(dataset_risks)-set(OBSERVATIONAL)-{'energy'}:
        raise ValueError('Exactly five primary observational tasks, plus optional separate Energy, required')
    effects={};repeat_effects={};degradations={};over_margin={}
    for name,row in dataset_risks.items():
        selected=np.asarray(row['selected'],dtype=np.float64);baseline=np.asarray(row['baseline'],dtype=np.float64)
        if selected.shape!=(2,) or baseline.shape!=(2,) or not np.isfinite(selected).all() or not np.isfinite(baseline).all() or np.any(selected<=0) or np.any(baseline<=0):
            raise ValueError('Paired strictly positive finite risks required for log effects; no epsilon replacement')
        s=math.fsum(selected)/2;b=math.fsum(baseline)/2
        effects[name]=math.log(b)-math.log(s)
        repeat_effects[name]=[math.log(b0)-math.log(s0) for b0,s0 in zip(baseline,selected)]
        degradations[name]=s/b-1.
        over_margin[name]=s>1.05*b
    return dict(primary_mean_log_benefit=math.fsum(effects[n] for n in OBSERVATIONAL)/5,
        repeat_mean_log_benefits=[math.fsum(repeat_effects[n][r] for n in OBSERVATIONAL)/5 for r in range(2)],
        task_log_benefits={n:effects[n] for n in OBSERVATIONAL},
        energy_log_benefit=effects.get('energy'),nonnegative_task_count=sum(effects[n]>=0 for n in OBSERVATIONAL),
        task_degradation_over_5_percent=[n for n in OBSERVATIONAL if over_margin[n]],
        task_relative_degradation={n:degradations[n] for n in OBSERVATIONAL},confirmed_useful_win=False,
        scope='Finite five-task point arithmetic; no uncertainty interval, dependence qualification or population claim')
