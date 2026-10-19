"""Separate full-horizon engineering spine for M3 workload/resource evidence."""
import copy

from treeneuron.config import canonical_hash,canonical_json
from .config import ResearchConfig,expand
from .joint_graph import compile_joint_graph
from .joint_plan import FINAL_ROLES
from .seeds import seed64,seed32
from .workqueue import make_queue_plan
from .joint_workers import JointWorkers
from .ledger import write_new_json


def checked_target(target):
    if (target.get('schema')!='research/v2-workqueue'
            or target.get('plan_hash')!=canonical_hash({k:v for k,v in target.items() if k!='plan_hash'})
            or target.get('identity',{}).get('schema')!='research/v2-M3-queue-identity'):
        raise ValueError('Complete committed M3 target queue required')
    actual=compile_joint_graph(expand(ResearchConfig.from_dict(target['identity']['study_config'])).to_dict())
    if canonical_json(actual)!=canonical_json(target):raise ValueError('Changed complete M3 queue/source/configuration')
    return target


def workload_key(job):
    kind=job['kind'];p=job['payload']
    if kind in ('fit','evaluate','placement','local_model'):return kind+':'+p['role']
    if kind=='probe':return 'probe:keep' if p['role']=='keep' else 'probe:singleton'
    if kind in ('data','parent','score','short_selection','extension_decision'):return kind
    raise ValueError('Unknown M3 resource workload kind')


def build_joint_qualification_plan(target):
    target=checked_target(target);original={j['job_id']:j for j in target['attempts']}
    chosen=set()
    def include(name):
        if name in chosen:return
        chosen.add(name)
        for dependency in original[name]['depends_on']:include(dependency)
    for job in target['attempts']:
        p=job['payload']
        if job['kind'] in ('fit','local_model','evaluate') and (p['teacher_index'],p['repetition'])==(0,0):include(job['job_id'])
    config=target['identity']['study_config'];labels=(config['study_id'],'engineering-resource-only','primary','teacher-0')
    root=config['seed_root'];used=set()
    seeds={name+'_seed':seed64(root,*labels,name) for name in ('teacher','data','reference','evaluation','allocation','contrast')}
    seeds['candidate_sample_seed']=seed64(root,*labels,'diagnostic-inventory')
    for prefix,label in (('optimizer','optimizer'),('fresh_optimizer','fresh-ordinary-optimizer')):
        seeds[prefix+'_seed'],seeds[prefix+'_collision_counter']=seed32(root,*labels,label,used=used)
    jobs=[];mapping={}
    for original_job in target['attempts']:
        if original_job['job_id'] not in chosen:continue
        job={k:copy.deepcopy(v) for k,v in original_job.items() if k!='attempt_id'}
        for name,value in seeds.items():
            if name in job['payload']:job['payload'][name]=value
        jobs.append(job)
        mapping[job['job_id']]=dict(target_job_sha256=canonical_hash(original_job),workload_key=workload_key(original_job))
    # The decision has no training. Fixed synthetic metrics exercise its full
    #160-input shape without creating eight scientific teachers or their labels.
    fixtures=[]
    for teacher in range(8):
        for repeat in range(2):
            for role in FINAL_ROLES:
                name=f'qualification_fixture:teacher-{teacher}/repeat-{repeat}/{role}';fixtures.append(name)
                jobs.append(dict(job_id=name,kind='qualification_fixture',depends_on=[],dependency_policy='all_eligible',
                    payload=dict(teacher_index=teacher,repetition=repeat,role=role,V_NMSE=1.,
                        evidence_kind='synthetic_resource_decision_fixture')))
    barrier=copy.deepcopy(original['extension_decision:M3']);barrier.pop('attempt_id')
    barrier['depends_on']=fixtures;jobs.append(barrier)
    mapping[barrier['job_id']]=dict(target_job_sha256=canonical_hash(original[barrier['job_id']]),
        workload_key='extension_decision',evidence_kind='synthetic_resource_decision')
    maximum=sum(j['payload'].get('max_updates',0) for j in jobs if j['kind'] in ('parent','fit','probe'))
    identity=dict(schema='research/v2-M3-qualification-identity',admission='engineering_resource_qualification_only',
        scientific_admission=False,M3_production_teachers=0,T_authorized=False,
        source_sha256=target['identity']['source_sha256'],execution_identity=target['identity']['execution_identity'],
        study_config=copy.deepcopy(config),design_bindings=target['identity']['design_bindings'],
        target_plan_hash=target['plan_hash'],target_mapping=mapping,engineering_seed_overrides=seeds,
        maximum_qualification_updates=maximum,
        synthetic_exceptions=['Complete160-input extension decision uses fixed synthetic V risks, not scientific outcomes'],
        scope='One independent engineering primary-family teacher; full F/G/V/E shapes and native horizons; no scientific M3 teacher or efficacy admission')
    return make_queue_plan(identity,jobs)


class JointQualificationWorkers(JointWorkers):
    """Native spine plus explicitly synthetic decision metrics; never efficacy."""
    def __init__(self,target,qualification,*,guard=lambda:None):
        if canonical_json(build_joint_qualification_plan(target))!=canonical_json(qualification):
            raise ValueError('M3 qualification differs from its frozen native workload/seeds')
        super().__init__(qualification,guard=guard)

    def __call__(self,job,directory,dependencies,attempt):
        if job.get('job_id') not in self.jobs or canonical_hash(job)!=canonical_hash(self.jobs[job['job_id']]):
            raise ValueError('M3 qualification job differs from its immutable plan')
        self.guard()
        if job['kind']=='qualification_fixture':
            p=job['payload']
            value=dict(status='completed_eligible',evidence_kind='synthetic_resource_decision_fixture',
                metrics=dict(selected=dict(V=dict(normalized_mse=p['V_NMSE']))),
                scientific_admission=False,T_accessed=False)
            write_new_json(directory/'fixture.json',value)
        else:
            value=super().__call__(job,directory,dependencies,attempt)
            value=dict(value,evidence_kind='synthetic_resource_decision' if job['kind']=='extension_decision'
                else 'native_engineering_resource_workload',scientific_admission=False,T_accessed=False)
        self.guard();return value
