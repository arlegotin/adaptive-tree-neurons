"""Complete M1 development tables from independently audited retained risks."""
import math
import statistics

from treeneuron.config import canonical_hash
from .control_analysis import positive_sensitivity
from .ledger import TERMINAL


def summarize_control_audit(audit):
    if (audit.get('schema')!='research/v2-control-audit' or audit.get('T_accessed') is not False
            or audit.get('evidence_level')!='retained_raw_prediction_arithmetic'):
        raise ValueError('Requires the independent M1 arithmetic audit without protected access')
    cells=[dict(population='positive',gamma=g,sigma=s,
        roles=['keep','top','random','oracle_current','oracle_true']) for g in (.5,2.) for s in (0.,.1)]
    cells += [dict(population=p,gamma=None,sigma=None,roles=['keep','top','random']) for p in
        ('globally_affine','additive_sine_quadratic','pure_noise','random_label_permutation')]
    expected={(c['population'],c['gamma'],c['sigma'],t,r,role)
        for c in cells for t in range(4) for r in range(2) for role in c['roles']}
    index={};counts={}
    for row in audit['rows']:
        key=(row.get('population'),row.get('gamma'),row.get('sigma'),row.get('teacher_index'),
             row.get('repetition'),row.get('role'))
        if (key not in expected or key in index or type(row.get('teacher_index')) is not int
                or type(row.get('repetition')) is not int
                or row.get('status') not in TERMINAL|{'running','never_started'}):
            raise ValueError('Unexpected, duplicate or invalid M1 observation')
        if row['status']=='completed_eligible':
            for split in ('V','E'):
                value=row['metrics'][split]['normalized_mse']
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError('Invalid audited normalized risk')
            if row.get('V_NMSE')!=row['metrics']['V']['normalized_mse']:
                raise ValueError('Audit V selection risk contradicts its retained metric')
        counts[row['status']]=counts.get(row['status'],0)+1;index[key]=row
    if set(index)!=expected:raise ValueError('Missing M1 nominal observations')
    sensitivity=positive_sensitivity(audit['rows'])
    if canonical_hash(sensitivity)!=canonical_hash(audit['positive_sensitivity']):
        raise ValueError('Control sensitivity differs from retained risks')
    complete=counts.get('completed_eligible',0)==len(expected)
    if audit.get('status')!=('completed' if complete else 'incomplete_or_ineligible'):
        raise ValueError('Audit completion status contradicts its full inventory')
    reports=[]
    for cell in cells:
        prefix=(cell['population'],cell['gamma'],cell['sigma'])
        rows=[index[(*prefix,t,r,role)] for t in range(4) for r in range(2) for role in cell['roles']]
        missing=[dict(teacher_index=r['teacher_index'],repetition=r['repetition'],role=r['role'],status=r['status'])
            for r in rows if r['status']!='completed_eligible']
        report=dict(**cell,status='incomplete' if missing else 'completed',independent_base_instances=4,
            training_repeats_per_instance=2,missing_coordinates=missing,mean_risk=None,teachers=[],placement=None,
            information_advantaged_roles=['oracle_current','oracle_true'] if cell['population']=='positive' else [])
        if not missing:
            teachers=[]
            for t in range(4):
                risks={split:{role:math.fsum(index[(*prefix,t,r,role)]['metrics'][split]['normalized_mse']
                    for r in range(2))/2 for role in cell['roles']} for split in ('V','E')}
                teachers.append(dict(teacher_index=t,risk=risks))
            report['teachers']=teachers
            report['mean_risk']={split:{role:math.fsum(t['risk'][split][role] for t in teachers)/4
                for role in cell['roles']} for split in ('V','E')}
            placement={}
            for split in ('V','E'):
                effects=[];differences=[];undefined=0
                for teacher in teachers:
                    values=teacher['risk'][split];random,top=values['random'],values['top']
                    effect=math.log(random)-math.log(top) if random>0 and top>0 else None
                    teacher.setdefault('placement',{})[split]=dict(log_benefit=effect,random_minus_top=random-top)
                    differences.append(random-top)
                    if effect is None:undefined+=1
                    else:effects.append(effect)
                placement[split]=dict(mean_teacher_log_benefit=None if undefined else math.fsum(effects)/4,
                    median_teacher_log_benefit=None if undefined else statistics.median(effects),
                    mean_random_minus_top=math.fsum(differences)/4,undefined_teacher_count=undefined,
                    signs=dict(positive=sum(d>0 for d in differences),zero=sum(d==0 for d in differences),
                               negative=sum(d<0 for d in differences)))
            report['placement']=placement
        reports.append(report)
    return dict(schema='research/v2-control-development-summary',status='completed_development' if complete else 'incomplete_development',
        audit_sha256=canonical_hash(audit),planned_final_fits=len(expected),status_counts=counts,cells=reports,
        positive_sensitivity=sensitivity,positive_base_instances_reused_across_cells=True,
        uncertainty_scope='Four base instances per cell; paired repeats are not independent teachers; no confirmation intervals',
        metric_scope='V and exploratory E NMSE; mean R2 within teacher before comparisons',
        Gate_A_decided=False,efficacy_claim=False,T_accessed=False)
