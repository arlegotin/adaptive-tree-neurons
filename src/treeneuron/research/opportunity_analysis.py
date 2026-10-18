"""Finite development screening; no test outcomes or confirmatory inference."""
import math

import numpy as np


class _Missing(ValueError):
    pass


def opportunity_summary(cases, outcomes, contract, *, teachers):
    """The coordinator supplies authenticated V-only rows from the fixed DAG.

    Every expected primary input remains required, including diagnostic head and
    ordinary capacity outcomes. No complete-case or best-repeat selection occurs.
    """
    teachers = list(teachers)
    if teachers not in (list(range(4)),list(range(8))):
        raise ValueError('Opportunity screening requires the fixed initial or once-extended teachers')
    cases = [c for c in cases if c['teacher_index'] in teachers]
    try:
        return _summarize(cases,outcomes,contract,teachers)
    except _Missing as error:
        return dict(status='irrecoverable_missing',reason=str(error),extension_required=None,
                    efficacy_claim=False,teacher_ids=teachers)


def _summarize(cases, outcomes, contract, teachers):
    if len(cases) != 112*len(teachers) or len({c['attempt_id'] for c in cases}) != len(cases):
        raise _Missing('Primary screening inventory differs from the fixed M2a/e scope')
    if set(outcomes) != {c['attempt_id'] for c in cases}:
        raise _Missing('Missing or unexpected primary screening outcomes')
    for value in outcomes.values():
        if value['status'] not in ('completed_eligible','completed_ineligible'):
            raise _Missing('A required primary input failed; no teacher can be dropped')
        if value['status'] == 'completed_eligible':
            risk = value.get('V_NMSE')
            if type(risk) not in (int,float) or not math.isfinite(risk) or risk < 0:
                raise _Missing('A primary V risk is missing, nonfinite or negative')
    index = {}
    for c in cases:
        if c['matrix'] == 'M2a':
            key = (c['matrix'],c['teacher_index'],c['repetition'],c['snapshot'],
                   c['treatment'],c['role'],c['admission'])
        else:
            key = (c['matrix'],c['teacher_index'],c['repetition'],c.get('recipe','head'))
        if key in index:
            raise _Missing('Duplicate screening coordinate')
        index[key] = outcomes[c['attempt_id']]

    def row(key):
        if key not in index:
            raise _Missing('Required screening coordinate is missing')
        return index[key]

    origins = {}
    for teacher in teachers:
        for repeat in range(2):
            for snapshot in ('actual_250','actual_1000','late_G_best'):
                keys = [('M2a',teacher,repeat,snapshot,treatment,role)
                    for treatment in ('continued_stream','new_stream_retained_adam')
                    for role in ('keep','top','random','warm_lower','warm_upper')]
                original = [row((*k,'initial' if teacher < 4 else 'primary_extension'))['status'] for k in keys]
                if snapshot == 'late_G_best':
                    if set(original) != {'completed_eligible'}:
                        raise _Missing('A late selected primary coordinate is unavailable')
                    origin = 'original'
                else:
                    fallback = [row((*k,'missing_snapshot_fallback' if teacher < 4 else
                        'extension_missing_snapshot_fallback'))['status'] for k in keys]
                    if set(original) == {'completed_eligible'} and set(fallback) == {'completed_ineligible'}:
                        origin = 'original'
                    elif set(original) == {'completed_ineligible'} and set(fallback) == {'completed_eligible'}:
                        origin = 'fixed_parent_budget_fallback'
                    else:
                        raise _Missing('Snapshot origins are inconsistent; outcomes cannot choose the acquisition')
                origins[teacher,repeat,snapshot] = origin
            for recipe in ('legacy_005','constant_001','cosine_005','wd_1e4','wd_1e3'):
                if row(('M2e_neural',teacher,repeat,recipe))['status'] != 'completed_eligible':
                    raise _Missing('An ordinary recipe is unavailable')
            if row(('M2e_head',teacher,repeat,'head'))['status'] != 'completed_eligible':
                raise _Missing('The declared head diagnostic is unavailable')

    def risk(teacher, repeat, contrast, role):
        original = origins[teacher,repeat,contrast['snapshot']] == 'original'
        admission = ('initial' if teacher < 4 else 'primary_extension') if original else (
            'missing_snapshot_fallback' if teacher < 4 else 'extension_missing_snapshot_fallback')
        return row(('M2a',teacher,repeat,contrast['snapshot'],contrast['treatment'],role,admission))['V_NMSE']

    tolerance = contract['tie_relative_tolerance']
    contrasts = []
    for contrast in contract['contrasts']:
        records = []
        for teacher in teachers:
            arm = math.fsum(risk(teacher,r,contrast,contrast['role']) for r in range(2))/2
            references = [('matched_keep',math.fsum(risk(teacher,r,contrast,'keep') for r in range(2))/2)]
            references.extend((recipe,math.fsum(row(('M2e_neural',teacher,r,recipe))['V_NMSE']
                for r in range(2))/2) for recipe in contrast['ordinary_recipes'])
            selected,reference = references[0]
            for name,value in references[1:]:
                if value < reference-tolerance*max(1.,abs(reference)):
                    selected,reference = name,value
            if reference == 0:
                raise _Missing('Zero ordinary reference risk makes relative opportunity undefined')
            records.append(dict(teacher_index=teacher,arm_V_NMSE=arm,reference_V_NMSE=reference,
                reference=selected,relative_benefit=1-arm/reference,
                origins=[origins[teacher,r,contrast['snapshot']] for r in range(2)]))
        effects = [r['relative_benefit'] for r in records]
        contrasts.append(dict(**contrast,teachers=records,mean_relative_benefit=math.fsum(effects)/len(effects),
            qualifying_teachers=sum(r['arm_V_NMSE'] <= (1-contract['minimum_relative_reduction'])*r['reference_V_NMSE']
                for r in records)))
    if not contrasts:
        raise _Missing('Fixed contrast registry is empty')
    best = contrasts[0]
    for candidate in contrasts[1:]:
        if candidate['mean_relative_benefit'] > best['mean_relative_benefit']+tolerance*max(1.,abs(best['mean_relative_benefit'])):
            best = candidate
    effects = np.asarray([r['relative_benefit'] for r in best['teachers']])
    rng = np.random.default_rng(contract['bootstrap_seed'])
    indices = rng.integers(len(teachers),size=(contract['bootstrap_iterations'],len(teachers)))
    draws = effects[indices].mean(axis=1)
    interval = np.quantile(draws,[.025,.975],method='linear').tolist()
    # Evaluate inclusive endpoints in risk space. Subtracting a rounded ratio
    # from one can move an exactly-at-margin input below its declared boundary.
    zero = np.asarray([(r['reference_V_NMSE']-r['arm_V_NMSE'])/r['reference_V_NMSE']
                       for r in best['teachers']])
    margin = np.asarray([((1-contract['minimum_relative_reduction'])*r['reference_V_NMSE']
                         -r['arm_V_NMSE'])/r['reference_V_NMSE'] for r in best['teachers']])
    centered = {name:np.quantile(values[indices].mean(axis=1),[.025,.975],method='linear').tolist()
                for name,values in (('zero',zero),('margin',margin))}
    maximum = max(c['qualifying_teachers'] for c in contrasts)
    fewer = maximum < contract['minimum_qualifying_teachers']
    uncertain = centered['zero'][0] <= 0 and centered['margin'][1] >= 0
    return dict(status='completed_eligible',contrasts=contrasts,best_contrast_id=best['contrast_id'],
        best_fixed_interval=interval,best_fixed_risk_centered_intervals=centered,
        maximum_qualifying_teachers=maximum,
        gated_growth_maximum_qualifying_teachers=max(c['qualifying_teachers'] for c in contrasts if c['role'] in ('top','random')),
        ordinary_capacity_maximum_qualifying_teachers=max(c['qualifying_teachers'] for c in contrasts if c['role'].startswith('warm_')),
        extension_required=bool(len(teachers)==4 and (fewer or uncertain)),
        extension_rule_triggered=bool(fewer or uncertain),extension_exhausted=len(teachers)==8,
        extension_triggers=dict(fewer_than_two=fewer,interval_covers_zero_and_margin=uncertain),
        efficacy_claim=False,teacher_ids=teachers,bootstrap_seed=contract['bootstrap_seed'],
        bootstrap_iterations=contract['bootstrap_iterations'],interval_scope='descriptive development; fixed arm chosen once on these teachers')
