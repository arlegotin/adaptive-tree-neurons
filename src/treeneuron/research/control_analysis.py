"""The fixed information-advantaged M1 sensitivity screen, using V only."""
import math


def positive_sensitivity(rows):
    selected = [r for r in rows if r.get('population') == 'positive'
                and r.get('gamma') == 2. and r.get('sigma') == .1
                and r.get('role') in ('random','oracle_true')]
    expected = {(t,r,role) for t in range(4) for r in range(2) for role in ('random','oracle_true')}
    index = {}
    failure = None
    for row in selected:
        key = (row.get('teacher_index'),row.get('repetition'),row['role'])
        risk = row.get('V_NMSE')
        if (key not in expected or key in index or row.get('status') != 'completed_eligible'
                or type(risk) not in (int,float) or not math.isfinite(risk) or risk < 0):
            failure = 'Missing, failed, duplicate or invalid fixed positive-control observation'
            break
        index[key] = risk
    if set(index) != expected:
        failure = failure or 'The true-contrast oracle/random R2 observations for all four instances are required'
    records = []
    if failure is None:
        for teacher in range(4):
            random = math.fsum(index[teacher,r,'random'] for r in range(2))/2
            oracle = math.fsum(index[teacher,r,'oracle_true'] for r in range(2))/2
            if random == 0 or not math.isfinite(1-oracle/random):
                failure = 'Relative control sensitivity is undefined'
                break
            records.append(dict(instance_index=teacher,random_V_NMSE=random,oracle_V_NMSE=oracle,
                relative_benefit=1-oracle/random,qualifies=oracle<=.98*random))
    common = dict(information_advantaged=True,efficacy_claim=False,
                  coordinate=dict(gamma=2.,sigma=.1),metric='V_NMSE; mean R2 first',
                  required_instances=4,minimum_qualifying_instances=2,minimum_relative_reduction=.02)
    if failure is not None:
        return dict(status='irrecoverable_missing',sensitive=None,reason=failure,**common)
    count = sum(r['qualifies'] for r in records)
    return dict(status='completed_eligible',sensitive=count>=2,instances=records,
                qualifying_instances=count,**common)
