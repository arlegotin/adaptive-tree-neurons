"""Complete finite practical comparisons on supplied group/day loss arrays.

No files or labels are opened. Marginal bootstrap inference is an approximate
conditional location procedure, not exact randomization or population evidence.
Validation-selected champion provenance and frozen models require an evaluator.
"""
import math

import numpy as np

from .practical_v2 import (OBSERVATIONAL,FAMILIES,CHAMPION,
    paired_group_bootstrap,practical_benefit)
from .primary_v2 import holm_family


MARGIN=math.log(1/.98)
_INDIVIDUAL=FAMILIES[1:]


def bootstrap_margin_family(records,*,planned_hypotheses,margin=MARGIN,alpha=.025):
    """Strict Holm tests and compatible regions from centered bootstrap errors.

    Plus-one probabilities include ties. Inverting that same discrete law gives
    closed marginal lower bounds; applying the already declared Guilbaud Choice2
    construction preserves open rejected margin endpoints. Its coverage depends
    on valid marginal bootstrap location approximations, not on Holm alone.
    """
    planned=planned_hypotheses
    if (type(planned) is not list or not planned or any(type(n) is not str or not n for n in planned)
            or len(set(planned))!=len(planned) or type(records) is not dict or set(records)-set(planned)
            or type(margin) not in (int,float) or not math.isfinite(margin)
            or type(alpha) not in (int,float) or not math.isfinite(alpha) or not 0<alpha<1):
        raise ValueError('Explicit complete inference family, finite margin and alpha required')
    summaries={};errors={};missing=[];counts=set()
    for name in planned:
        row=records.get(name)
        if row is None:missing.append(name);continue
        if type(row) is not dict or set(row)!={'point','draws'}:raise ValueError('Exact point/bootstrap draws required')
        point=row['point'];draws=np.asarray(row['draws'],dtype=np.float64)
        if draws.ndim!=1 or len(draws)<2:raise ValueError('Complete one-dimensional bootstrap draws required')
        counts.add(len(draws))
        if point is None or not np.isfinite(draws).all():missing.append(name);continue
        if type(point) not in (int,float) or not math.isfinite(point):raise ValueError('Finite point effect required')
        with np.errstate(over='raise',invalid='raise'):
            try:offset=draws-point;threshold=point-margin
            except FloatingPointError as error:raise ValueError('Bootstrap centering overflows') from error
        if not math.isfinite(threshold) or not np.isfinite(offset).all():raise ValueError('Nonfinite bootstrap error')
        count=int(np.count_nonzero(offset>=threshold));p=(1+count)/(len(draws)+1)
        errors[name]=np.sort(offset)
        summaries[name]=dict(point=point,p=p,upper_tail_count=count,iterations=len(draws))
    if len(counts)>1:raise ValueError('All comparisons must retain the same bootstrap replication count')
    base=dict(schema='research/v2-practical-bootstrap-margin-family',planned_hypotheses=list(planned),
        family_size=len(planned),margin=margin,alpha=alpha,familywise_confidence=1-alpha,
        marginals=summaries,unavailable_hypotheses=missing,claim_authorized=False,protected_access_authorized=False,
        raw_metrics_authenticated=False,
        marginal_law='Centered bootstrap error location approximation; p=(1+count(error>=point-margin))/(B+1), ties against rejection',
        region_law='Guilbaud Choice2 with discrete inversion of the declared marginal bootstrap law',
        assumption='Valid conditional marginal bootstrap location approximation for fixed fitted models and held-out sampling units; no exact finite-sample coverage guarantee',
        endpoint_convention='Closed marginal lower endpoints; open margin only when its null is strictly rejected')
    if missing:return dict(base,status='undefined_family',holm=None,bounds=None)
    count=next(iter(counts))
    # This conservative floor guarantees every possible Holm/region level can
    # be resolved; no smaller observed rejection family rescues a short run.
    if math.ceil(alpha*(count+1)/len(planned))-2<0:
        return dict(base,status='insufficient_resampling_resolution',holm=None,bounds=None)
    adjusted=holm_family({n:summaries[n]['p'] for n in planned},planned_hypotheses=planned,alpha=alpha)
    rejected=set(adjusted['rejected']);remaining=len(planned)-len(rejected)
    def lower(name,level):
        largest_rejecting_tail=math.ceil(level*(count+1))-2
        critical=errors[name][count-largest_rejecting_tail-1]
        result=summaries[name]['point']-float(critical)
        if not math.isfinite(result):raise ValueError('Bootstrap confidence endpoint overflows')
        return result
    bounds={}
    if remaining:
        for name in planned:
            endpoint=margin if name in rejected else min(margin,lower(name,alpha/remaining))
            bounds[name]=dict(lower=endpoint,lower_endpoint_excluded=name in rejected,
                rejects_margin_null=name in rejected,strictly_sharper_than_margin=endpoint>margin)
    else:
        common=max(margin,min(lower(n,alpha) for n in planned))
        for name in planned:
            bounds[name]=dict(lower=common,lower_endpoint_excluded=common==margin,
                rejects_margin_null=True,strictly_sharper_than_margin=common>margin)
    return dict(base,status='completed',holm=adjusted,bounds=bounds,all_hypotheses_rejected=not remaining)


def compare_practical_families(losses,*,champions,seed,iterations=20000,chunk_size=256,bike_block_days=7):
    """Compare P00 with all thirteen families and the preselected V champion.

    Family arrays have shape(groups,2); champion IDs specify each repeat's
    validation choice among P02-P13. Exactly the same resampled indices serve
    every contrast and both repeats. Energy remains a separate description.
    """
    if (type(losses) is not dict or not set(OBSERVATIONAL)<=set(losses)
            or set(losses)-set(OBSERVATIONAL)-{'energy'} or type(champions) is not dict
            or set(champions)!=set(losses) or type(seed) is not int or not 0<=seed<2**63):
        raise ValueError('Complete fixed task inventory and declared validation champion choices required')
    packets={}
    for name,packet in losses.items():
        fields={'group_ids','family_losses'}|({'day_ordinals'} if name=='bike' else set())
        if type(packet) is not dict or set(packet)!=fields:raise ValueError('Exact group/family loss packet required')
        families=packet['family_losses'];ids=packet['group_ids'];choice=champions[name]
        if (type(families) is not dict or set(families)!=set(FAMILIES) or type(ids) is not list
                or len(ids)<2 or any(type(i) not in (str,int) for i in ids)
                or len(set((type(i),i) for i in ids))!=len(ids)
                or type(choice) is not list or len(choice)!=2 or any(c not in CHAMPION for c in choice)):
            raise ValueError('All14families, explicit unique groups and two external champion IDs required')
        arrays={}
        for family,values in families.items():
            array=np.asarray(values,dtype=np.float64)
            if array.shape!=(len(ids),2) or not np.isfinite(array).all() or np.any(array<0):
                raise ValueError('Complete finite nonnegative normalized group/day losses required')
            arrays[family]=array.copy()
        arrays['V_champion']=np.column_stack([arrays[choice[r]][:,r] for r in range(2)])
        packets[name]=dict(group_ids=list(ids),arrays=arrays)
        if name=='bike':packets[name]['day_ordinals']=np.asarray(packet['day_ordinals']).copy()
    comparisons={};point_risks={};index_hashes=None
    for family in (*_INDIVIDUAL,'V_champion'):
        pair={}
        for name,packet in packets.items():
            pair[name]=dict(group_ids=packet['group_ids'],selected=packet['arrays']['P00'],baseline=packet['arrays'][family])
            if name=='bike':pair[name]['day_ordinals']=packet['day_ordinals']
        result=paired_group_bootstrap(pair,seed=seed,iterations=iterations,chunk_size=chunk_size,bike_block_days=bike_block_days)
        if index_hashes is None:index_hashes=result['pairing_index_sha256']
        elif index_hashes!=result['pairing_index_sha256']:raise ValueError('Cross-method bootstrap pairing changed')
        comparisons[family]=result
        if family=='V_champion':
            point_risks={name:dict(selected=np.mean(p['arrays']['P00'],axis=0).tolist(),
                baseline=np.mean(p['arrays'][family],axis=0).tolist()) for name,p in packets.items()}
    multiplicity=bootstrap_margin_family({family:dict(point=comparisons[family]['aggregate_point_log_benefit'],
        draws=comparisons[family]['aggregate_bootstrap_log_benefits']) for family in _INDIVIDUAL},
        planned_hypotheses=list(_INDIVIDUAL))
    champion=comparisons.pop('V_champion')
    # Undefined risk stays undefined; no epsilon, deleted sample or best-seed
    # selection repairs it. Energy cannot prevent observational point reports.
    positive=all(v>0 for name in OBSERVATIONAL for row in (point_risks[name],) for k in ('selected','baseline') for v in row[k])
    point=practical_benefit({n:point_risks[n] for n in OBSERVATIONAL}) if positive else None
    interval=champion['aggregate_interval']
    both=point is not None and all(v>0 for v in point['repeat_mean_log_benefits'])
    confidence=interval is not None and interval[0]>MARGIN
    four=point is not None and point['nonnegative_task_count']>=4
    no_harm=point is not None and not point['task_degradation_over_5_percent']
    primary=bike_block_days==7
    conditions=dict(primary_seven_day_block=primary,lower_exceeds_two_percent_margin=bool(confidence),
        positive_in_both_repeats=bool(both),nonnegative_task_count=None if point is None else point['nonnegative_task_count'],
        task_degradation_over_5_percent=None if point is None else point['task_degradation_over_5_percent'],
        robustness_conditions_pass=bool(four and no_harm),statistical_useful_conditions_pass=bool(primary and confidence and both),
        consistent_useful_conditions_pass=bool(primary and confidence and both and four and no_harm))
    return dict(schema='research/v2-complete-practical-comparison-arithmetic',family_comparisons=comparisons,
        champion=champion,champion_choices={k:list(v) for k,v in champions.items()},champion_point_summary=point,
        primary_conditions=conditions,multiplicity=multiplicity,
        claim_authorized=False,protected_access_authorized=False,champion_V_selection_authenticated=False,
        native_models_authenticated=False,population_superiority_confirmed=False,
        scope='Pure conditional fixed-model five-task arithmetic. Full native/raw-model/V-choice/frozen-protocol/cost authentication and dependence sensitivities remain required; no dataset-population inference.')
