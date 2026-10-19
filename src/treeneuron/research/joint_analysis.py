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
