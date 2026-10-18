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


def _log_ratio(selected,baseline):
    selected=np.asarray(selected,dtype=np.float64);baseline=np.asarray(baseline,dtype=np.float64)
    result=np.full(selected.shape,np.nan,dtype=np.float64)
    valid=(selected>0)&(baseline>0)&np.isfinite(selected)&np.isfinite(baseline)
    result[valid]=np.log(baseline[valid])-np.log(selected[valid])
    return result


def _conditional_interval(values):
    if not np.isfinite(values).all():return None
    return np.quantile(values,[.025,.975],method='linear').tolist()


def paired_group_bootstrap(losses,*,seed,iterations=20000,chunk_size=256,bike_block_days=7):
    """Paired conditional test-sampling uncertainty on fixed fitted models.

    Every packet contains normalized per-group/day MSE arrays shaped(G,2).
    Baseline choice must already be frozen from V. One set of indices is reused
    for both methods and both repetitions. Bike uses overlapping noncircular
    calendar blocks and truncates the final block to keep exactly G days.
    """
    import hashlib
    from treeneuron.config import canonical_hash
    from treeneuron.research.seeds import seed64
    if (not isinstance(losses,dict) or not set(OBSERVATIONAL)<=set(losses)
            or set(losses)-set(OBSERVATIONAL)-{'energy'}):raise ValueError('All five fixed observational tasks required')
    if (type(iterations) is not int or iterations<2 or type(chunk_size) is not int or chunk_size<1
            or type(bike_block_days) is not int or bike_block_days not in (3,7,14)):
        raise ValueError('Invalid conditional bootstrap iterations/chunk/block')
    packets={};input_hashes={}
    for name,packet in losses.items():
        required={'group_ids','selected','baseline'}|({'day_ordinals'} if name=='bike' else set())
        if not isinstance(packet,dict) or set(packet)!=required:raise ValueError('Invalid conditional loss packet fields')
        ids=packet['group_ids'];a=np.asarray(packet['selected'],dtype=np.float64);b=np.asarray(packet['baseline'],dtype=np.float64)
        if (not isinstance(ids,list) or len(ids)<2 or any(type(x) not in (int,str) for x in ids)
                or len({(type(x),x) for x in ids})!=len(ids)
                or a.shape!=(len(ids),2) or b.shape!=a.shape or not np.isfinite(a).all()
                or not np.isfinite(b).all() or np.any(a<0) or np.any(b<0)):
            raise ValueError('Complete paired nonnegative group losses/identities required')
        identity=dict(group_ids=ids,selected_sha256=hashlib.sha256(a.astype('<f8').tobytes()).hexdigest(),
            baseline_sha256=hashlib.sha256(b.astype('<f8').tobytes()).hexdigest())
        if name=='bike':
            days=np.asarray(packet['day_ordinals'])
            if days.shape!=(len(ids),) or days.dtype.kind not in 'iu' or np.any(np.diff(days)!=1):
                raise ValueError('Bike requires actual consecutive held-out day ordinals; do not fabricate gaps')
            if len(ids)<bike_block_days:raise ValueError('Bike block exceeds held-out day count')
            identity['day_ordinals']=days.tolist()
        packets[name]=(a,b);input_hashes[name]=canonical_hash(identity)
    draws={};points={};hashes={};first={};leave_out={};units={};assumptions={}
    for name in OBSERVATIONAL:
        a,b=packets[name];n=len(a);units[name]=n
        rng=np.random.default_rng(seed64(seed,'conditional-group-bootstrap',name))
        values=np.empty(iterations,dtype=np.float64);digest=hashlib.sha256()
        points[name]=float(_log_ratio(np.mean(a),np.mean(b)))
        for offset in range(0,iterations,chunk_size):
            count=min(chunk_size,iterations-offset)
            if name=='bike':
                starts=rng.integers(0,n-bike_block_days+1,size=(count,math.ceil(n/bike_block_days)),dtype=np.int64)
                indices=(starts[:,:,None]+np.arange(bike_block_days)).reshape(count,-1)[:,:n]
            else:indices=rng.integers(0,n,size=(count,n),dtype=np.int64)
            if offset==0:first[name]=indices[0].tolist()
            digest.update(indices.astype('<i8',copy=False).tobytes())
            selected=np.mean(np.mean(a[indices],axis=1),axis=1)
            baseline=np.mean(np.mean(b[indices],axis=1),axis=1)
            values[offset:offset+count]=_log_ratio(selected,baseline)
        draws[name]=values;hashes[name]=digest.hexdigest()
        deleted=_log_ratio((np.sum(a)-np.sum(a,axis=1))/(2*(n-1)),
            (np.sum(b)-np.sum(b,axis=1))/(2*(n-1)))
        leave_out[name]=dict(effect_range=[float(np.min(deleted)),float(np.max(deleted))]
            if np.isfinite(deleted).all() else None,undefined_count=int(np.sum(~np.isfinite(deleted))))
        assumptions[name]=(['fixed_fitted_models','local_dependence_covered_by_declared_calendar_block'] if name=='bike'
            else ['fixed_fitted_models','held_out_groups_exchangeable'])
        if name=='ccpp':assumptions[name].append('unverified_exchangeability')
        if name=='parkinsons':assumptions[name].append('few_subject_clusters; resample_count_does_not_increase_units')
    aggregate=np.mean(np.stack([draws[n] for n in OBSERVATIONAL]),axis=0)
    finite_point=all(math.isfinite(points[n]) for n in OBSERVATIONAL)
    energy=None
    if 'energy' in packets:
        a,b=packets['energy'];effects=_log_ratio(np.mean(a,axis=1),np.mean(b,axis=1))
        energy=dict(bootstrap_interval=None,groups=len(a),
            descriptive_group_effect_range=[float(np.min(effects)),float(np.max(effects))] if np.isfinite(effects).all() else None,
            scope='Simulation transfer; descriptive ranges only, not independent buildings')
    undefined=int(np.sum(~np.isfinite(aggregate)))
    return dict(status='completed' if undefined==0 and finite_point else 'undefined_log_risk',
        aggregate_interval=_conditional_interval(aggregate),
        aggregate_point_log_benefit=math.fsum(points.values())/5 if finite_point else None,
        dataset_intervals={n:_conditional_interval(draws[n]) for n in OBSERVATIONAL},
        aggregate_bootstrap_log_benefits=aggregate,dataset_bootstrap_log_benefits=draws,
        undefined_aggregate_resamples=undefined,discarded_resamples=0,energy=energy,
        resampling_units=units,dataset_assumptions=assumptions,leave_one_group_out=leave_out,
        pairing_index_sha256=hashes,first_resampled_group_indices=first,input_packet_sha256=input_hashes,
        iterations=iterations,seed=seed,bike_block_days=bike_block_days,
        bike_resampling_law='overlapping noncircular calendar-day blocks, uniform starts, final truncation to G days',
        interval_scope='Conditional test sampling for fitted models on five named tasks; not retraining or dataset-population uncertainty',
        population_superiority_confirmed=False,protected_access_authorized=False)
