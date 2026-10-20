"""Teacher-level M3 development description from independently audited risks.

No selection on E, repeat selection, complete-case deletion or efficacy gate.
Local-model prediction, finite remedies and full costs need separate review.
"""
import math


_ROLES=('keep','top','random','short_singleton','local_singleton','joint',
    'unconditioned','pure_span','warm_upper','fresh_upper')


def _mean(values):return math.fsum(v/len(values) for v in values)


def summarize_joint_predictions(audit):
    if (type(audit) is not dict or audit.get('schema')!='research/v2-joint-prediction-audit'
            or audit.get('status')!='completed_arithmetic' or audit.get('T_accessed') is not False
            or audit.get('efficacy_claim') is not False or type(audit.get('rows')) is not list):
        raise ValueError('Complete independent M3 development arithmetic is required')
    rows={};values={}
    for row in audit['rows']:
        p=row.get('payload',{});t=p.get('teacher_index');r=p.get('repetition');role=p.get('role')
        if (p.get('matrix')!='M3' or type(t) is not int or t not in range(16)
                or type(r) is not int or r not in (0,1) or role not in _ROLES
                or row.get('job_id')!=f'evaluate:M3/teacher-{t}/repeat-{r}/{role}'
                or row.get('status') not in ('completed_eligible','completed_ineligible') or (t,r,role) in rows):
            raise ValueError('M3 teacher/repeat/role/status inventory differs')
        rows[t,r,role]=row
        if row['status']=='completed_ineligible':
            if t<8 or 'metrics' in row:raise ValueError('An initial required fit cannot disappear or supply ineligible metrics')
            continue
        for target in ('V','E'):
            risk=row.get('metrics',{}).get('selected',{}).get(target,{}).get('normalized_mse')
            if type(risk) not in (int,float) or not math.isfinite(risk) or risk<=0:
                raise ValueError('Finite positive paired normalized risk is required')
            values[t,r,role,target]=risk
    expected={(t,r,role) for t in range(16) for r in (0,1) for role in _ROLES}
    if set(rows)!=expected:raise ValueError('Every initial and conditional M3 fit remains in the inventory')
    teachers=[]
    for t in range(16):
        status={rows[t,r,role]['status'] for r in (0,1) for role in _ROLES}
        if len(status)!=1:raise ValueError('No unpaired or complete-case teacher summaries are admitted')
        if status=={'completed_eligible'}:teachers.append(t)
    if teachers not in (list(range(8)),list(range(16))):raise ValueError('Exactly eight or sixteen ordered teachers required')
    means={(t,role,target):_mean([values[t,r,role,target] for r in (0,1)])
        for t in teachers for role in _ROLES for target in ('V','E')}
    by_role={role:{target:_mean([means[t,role,target] for t in teachers]) for target in ('V','E')}
        for role in _ROLES}
    contrasts={}
    for role in _ROLES:
        if role=='joint':continue
        result={}
        for target in ('V','E'):
            effects=[dict(teacher_index=t,reference_risk=means[t,role,target],joint_risk=means[t,'joint',target],
                log_risk_ratio=math.log(means[t,role,target])-math.log(means[t,'joint',target])) for t in teachers]
            numbers=[r['log_risk_ratio'] for r in effects]
            result[target]=dict(mean_log_risk_ratio=_mean(numbers),teacher_effects=effects,
                signs=dict(positive=sum(v>0 for v in numbers),zero=sum(v==0 for v in numbers),negative=sum(v<0 for v in numbers)))
        contrasts[role+'_over_joint']=result
    return dict(schema='research/v2-M3-development-summary',queue_plan_hash=audit['queue_plan_hash'],
        ledger_event_hash=audit['ledger_event_hash'],teachers=len(teachers),teacher_ids=teachers,repetitions=2,
        inactive_teachers=[t for t in range(16) if t not in teachers],role_means=by_role,contrasts=contrasts,
        teacher_risks=[dict(teacher_index=t,role=role,**{k:means[t,role,k] for k in ('V','E')})
            for t in teachers for role in _ROLES],
        scope='Development description; repeat-average risks before teacher log ratios; no population confirmation',
        Gate_B_decided=False,confirmatory_claim=False,T_accessed=False,
        remaining_gate_inputs=['Held-out local prediction and selection review','Complete standalone and physical resource accounting',
            'Penalty-sharing/additional-span and ordinary-control attribution','Bounded correction/admission disposition'])


def summarize_joint_local_checks(rows,*,teacher_ids):
    """Describe every saved F-solved probe, without re-solving or choosing on G.

    The caller authenticates the retained records. These are temporary joint
    parent/gate updates, not the Adam trajectory or a gate-only causal effect.
    """
    roles=('local_singleton','joint','unconditioned','pure_span');fractions=(1/16,1/4,1)
    if (type(rows) is not list or teacher_ids not in (list(range(8)),list(range(16)))):
        raise ValueError('Complete eight/sixteen-teacher local-probe inventory required')
    seen={};max_error=0.
    for row in rows:
        if type(row) is not dict:raise ValueError('Invalid local diagnostic record')
        t=row.get('teacher_index');r=row.get('repetition');role=row.get('role');d=row.get('diagnostic',{})
        key=(t,r,role)
        if (type(t) is not int or t not in teacher_ids or type(r) is not int or r not in (0,1)
                or role not in roles or key in seen or type(d) is not dict
                or d.get('fractions')!=list(fractions) or d.get('fraction_selected_on_G') is not False
                or d.get('local_updates_applied_to_parent') is not False):
            raise ValueError('Changed/unpaired local probe or a G-selected/applied fraction')
        evaluations=d.get('evaluations')
        if (type(evaluations) is not list or len(evaluations)!=3
                or [v.get('fraction') for v in evaluations]!=list(fractions)):
            raise ValueError('All three prescribed fractions must remain recorded')
        def number(value,*,positive=False,probability=False):
            if (type(value) not in (int,float) or not math.isfinite(value) or value<0
                    or positive and value==0 or probability and value>1):
                raise ValueError('Finite valid local loss/fraction/solve residual required')
            return value
        error=number(d.get('linear_solve',{}).get('matched_Schur_objective_error'))
        max_error=max(max_error,error)
        for target in ('F','G'):
            number(d.get('current_parent_MSE',{}).get(target),positive=True)
            for evaluation in evaluations:
                values=evaluation.get(target,{})
                for name in ('linearized_MSE','nonlinear_MSE'):number(values.get(name))
                for name in ('ReLU_mask_changed_fraction','gate_saturation_fraction'):
                    number(values.get(name),probability=True)
        seen[key]=d
    if set(seen)!={(t,r,k) for t in teacher_ids for r in (0,1) for k in roles}:
        raise ValueError('Every paired local-probe record must remain in the review')
    teacher_rows=[];by_role={}
    for role in roles:
        aggregates=[]
        for index,fraction in enumerate(fractions):
            grouped=[]
            for t in teacher_ids:
                data=dict(teacher_index=t,role=role,fraction=fraction)
                for target in ('F','G'):
                    base=_mean([seen[t,r,role]['current_parent_MSE'][target] for r in (0,1)])
                    values={name:_mean([seen[t,r,role]['evaluations'][index][target][name] for r in (0,1)])
                        for name in ('linearized_MSE','nonlinear_MSE','ReLU_mask_changed_fraction','gate_saturation_fraction')}
                    data[target]=dict(parent_MSE=base,**values,
                        nonlinear_relative_reduction=1-values['nonlinear_MSE']/base,
                        linear_relative_reduction=1-values['linearized_MSE']/base)
                grouped.append(data);teacher_rows.append(data)
            aggregate=dict(fraction=fraction)
            for target in ('F','G'):
                v=[d[target] for d in grouped]
                aggregate[target]={
                    'mean_teacher_nonlinear_relative_reduction':_mean([d['nonlinear_relative_reduction'] for d in v]),
                    'mean_teacher_linear_relative_reduction':_mean([d['linear_relative_reduction'] for d in v]),
                    'nonlinear_improving_teachers':sum(d['nonlinear_relative_reduction']>0 for d in v),
                    'linear_improving_teachers':sum(d['linear_relative_reduction']>0 for d in v),
                    'mean_ReLU_mask_changed_fraction':_mean([d['ReLU_mask_changed_fraction'] for d in v]),
                    'mean_gate_saturation_fraction':_mean([d['gate_saturation_fraction'] for d in v])}
            aggregates.append(aggregate)
        by_role[role]=aggregates
    return dict(schema='research/v2-M3-saved-local-review',teachers=len(teacher_ids),repetitions=2,
        local_diagnostic_count=len(rows),by_role=by_role,teacher_rows=teacher_rows,
        max_saved_direct_Schur_objective_error=max_error,Gate_B_decided=False,T_accessed=False,
        fresh_nonlinear_replay=False,record_file_authentication_performed=False,
        scope='Saved producer diagnostics only. Repetition losses averaged before teacher relative changes; all fixed fractions reported. Temporary joint parent/gate updates are separate from Adam continuation and from placement efficacy.')
