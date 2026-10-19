"""Source-bound complete M3 queue; no data generation or production admission."""
from collections import Counter

from treeneuron.config import canonical_hash,canonical_json
from .config import ResearchConfig,expand
from .workqueue import make_queue_plan


def compile_joint_graph(study):
    if (study.get('schema')!='research/v2-plan' or study.get('config',{}).get('kind')!='m3_joint'
            or study.get('plan_hash')!=canonical_hash({k:v for k,v in study.items() if k!='plan_hash'})
            or canonical_json(expand(ResearchConfig.from_dict(study['config'])).to_dict())!=canonical_json(study)):
        raise ValueError('M3 source/configuration/finite study commitment differs')
    jobs=[];datasets={};initial_evaluations=[]
    finals={(r['teacher_index'],r['repetition'],r['role']):r for r in study['attempts']+study['conditional_attempts']
        if r['matrix']=='M3'}
    probes={}
    for row in study['probe_attempts']+study['conditional_attempts']:
        if row['matrix']=='M3_probes':probes.setdefault((row['teacher_index'],row['repetition']),[]).append(row)

    def add(name,kind,payload,depends=(),*,all_terminal=False):
        jobs.append(dict(job_id=name,kind=kind,payload=payload,depends_on=list(depends),
            dependency_policy='all_terminal' if all_terminal else 'all_eligible'))
        return name

    for p in study['prerequisites']:
        teacher=p['teacher_index'];repeat=p['repetition'];key=p['prerequisite_id']
        if teacher not in datasets:
            payload={k:p[k] for k in ('population','teacher_index','teacher_seed','data_seed','reference_seed','evaluation_seed')}
            payload['evaluation_rows']=study['config']['evaluation_rows']
            datasets[teacher]=add(f'data:primary/teacher-{teacher}','data',payload,
                ('extension_decision:M3',) if teacher>=8 else ())
        data=datasets[teacher];parent=add('parent:'+key,'parent',p,(data,))
        score=add('score:'+key+'/late_G_best','score',dict(snapshot='late_G_best'),(data,parent))
        placement={}
        for role in ('local_singleton','joint','unconditioned','pure_span'):
            row=finals[teacher,repeat,role]
            placement[role]=add('placement:'+row['attempt_id'],'placement',row,(data,parent,score))
            add('local_model:'+row['attempt_id'],'local_model',row,(data,parent,score,placement[role]))
        trial_jobs=[]
        for row in probes[teacher,repeat]:
            trial_jobs.append(add('probe:'+row['attempt_id'],'probe',row,(data,parent,score)))
        short=add('short_selection:'+key,'short_selection',dict(parent_key=key,snapshot='late_G_best',requested_count=16),
            (data,parent,score,*trial_jobs),all_terminal=True)
        for role in ('keep','top','random','short_singleton','local_singleton','joint','unconditioned','pure_span','warm_upper','fresh_upper'):
            row=finals[teacher,repeat,role];dependencies=[data,parent,score]
            if role in placement:dependencies.append(placement[role])
            elif role=='short_singleton':dependencies.append(short)
            fitted=add('procedure:'+row['attempt_id'],'fit',row,dependencies)
            evaluated=add('evaluate:'+row['attempt_id'],'evaluate',row,(data,fitted))
            if teacher<8:initial_evaluations.append(evaluated)
    add('extension_decision:M3','extension_decision',study['extension_contract'],initial_evaluations,all_terminal=True)
    counts=dict(Counter(row['kind'] for row in jobs))
    expected=dict(data=16,parent=32,score=32,placement=128,local_model=128,probe=2080,
        short_selection=32,fit=320,evaluate=320,extension_decision=1)
    if counts!=expected:raise ValueError('M3 complete job inventory differs from the finite design')
    identity=dict(schema='research/v2-M3-queue-identity',source_sha256=study['source_sha256'],
        study_plan_hash=study['plan_hash'],study_config=study['config'],execution_identity=study['execution_identity'],
        design_bindings=study['design_bindings'],extension_contract=study['extension_contract'],
        matrix_counts=study['matrix_counts'],job_counts=counts,execution_implemented=False,
        pending=['M3 dispatcher and measured resource qualification/admission'],T_authorized=False)
    return make_queue_plan(identity,jobs)
