"""Shared development decisions cannot become every later model's fit cost."""
import importlib.util
import json

import pytest

from treeneuron.research.ledger import AttemptLedger
from treeneuron.research.workqueue import make_queue_plan,run_queue
from test_train import cpu_training_evidence


def api():
    assert importlib.util.find_spec('treeneuron.research.joint_costs') is not None,'Missing M3 procedure/physical cost collector'
    from treeneuron.research import joint_costs
    return joint_costs


def plan():
    jobs=[]
    def add(name,kind,depends=()):
        jobs.append(dict(job_id=name,kind=kind,depends_on=list(depends),payload=dict(matrix='M3',teacher_index=8,repetition=0,role='joint')))
    add('old_data','data');add('old_fit','fit',('old_data',));add('decision','extension_decision',('old_fit',))
    add('new_data','data',('decision',));add('parent','parent',('new_data',));add('score','score',('parent','new_data'))
    add('placement','placement',('parent','score','new_data'));add('local_diagnostic','local_model',('placement',))
    add('joint_fit','fit',('new_data','parent','score','placement'));add('evaluation','evaluate',('new_data','joint_fit'))
    return make_queue_plan(dict(schema='research/v2-M3-queue-identity',source_sha256='a'*64),jobs)


def paid_ledger(tmp_path,*,retry=False,missing=False):
    p=plan();ledger=AttemptLedger.create(tmp_path/'queue',p)
    costs=dict(old_data=2,old_fit=100,decision=10,new_data=3,parent=20,score=5,placement=7,
        local_diagnostic=11,joint_fit=30,evaluation=4)
    for job in p['attempts']:
        name=job['job_id']
        if name=='evaluation' and missing:continue
        if name=='joint_fit' and retry:
            ledger.record(name,'running',elapsed_ns=0,recovery_attempt=0)
            ledger.record(name,'interrupted_recoverable',elapsed_ns=8,recovery_attempt=0)
            ledger.record(name,'running',elapsed_ns=8,recovery_attempt=1)
        ledger.record(name,'completed_eligible',elapsed_ns=costs[name])
    return p,ledger


def test_each_extension_fit_pays_own_prerequisites_but_not_initial_cohort_or_local_diagnostic(tmp_path):
    module=api();p,ledger=paid_ledger(tmp_path)
    costs=module.describe_joint_costs(p,ledger)
    row=costs['procedures']['joint_fit'];evaluation=costs['procedures']['evaluation']
    assert row['charged_ns']==65 and evaluation['charged_ns']==69
    assert row['member_job_ids']==['joint_fit','new_data','parent','placement','score']
    assert costs['cohort_charged_job_ns']==192
    assert costs['development_only_charged_job_ns']==21
    assert costs['development_only_job_ids']==['decision','local_diagnostic']
    assert costs['artifact_inventories_authenticated'] is costs['quality_cost_claim'] is costs['Gate_B_decided'] is False


def test_retry_cumulative_cost_is_retained_once_in_every_standalone_procedure(tmp_path):
    module=api();p,ledger=paid_ledger(tmp_path,retry=True)
    result=module.describe_joint_costs(p,ledger)
    assert result['procedures']['joint_fit']['charged_ns']==65
    assert result['cohort_charged_job_ns']==192 and result['terminal_jobs']==10


def test_cost_collection_cannot_describe_a_partial_ledger_as_complete(tmp_path):
    module=api();p,ledger=paid_ledger(tmp_path,missing=True)
    with pytest.raises(ValueError):module.describe_joint_costs(p,ledger)


def test_a_non_control_dependency_on_data_cannot_be_erased_from_procedure_costs(tmp_path):
    from treeneuron.config import canonical_hash
    module=api();p,ledger=paid_ledger(tmp_path)
    for job in p['attempts']:
        if job['job_id']=='decision':job['kind']='parent'
    p['plan_hash']=canonical_hash({k:v for k,v in p.items() if k!='plan_hash'})
    ledger=AttemptLedger.create(tmp_path/'changed-but-matching-queue',p)
    for job in p['attempts']:ledger.record(job['job_id'],'completed_eligible',elapsed_ns=1)
    with pytest.raises(ValueError):module.describe_joint_costs(p,ledger)


def test_inactive_collection_authenticates_all_artifacts_and_external_endpoint(tmp_path):
    module=api();p=plan();root=tmp_path/'queue'
    def work(job,directory,dependencies,attempt):
        (directory/'fixture.txt').write_text('Metadata-only real queue, no model fit')
        return dict(status='completed_eligible')
    run_queue(p,root,work,check_identity=lambda _:None)
    last=json.loads((root/'events.jsonl').read_text().splitlines()[-1])['event_hash']
    result=module.collect_joint_costs(root,expected_plan_hash=p['plan_hash'],expected_ledger_hash=last)
    assert result['artifact_inventories_authenticated'] is True and result['terminal_jobs']==10
    assert result['cohort_charged_job_ns']>0 and result['T_accessed'] is False
    with pytest.raises(ValueError):module.collect_joint_costs(root,expected_plan_hash=p['plan_hash'],expected_ledger_hash='b'*64)
    next((root/'jobs').glob('*/attempt-0/fixture.txt')).write_text('changed old evidence')
    with pytest.raises(ValueError):module.collect_joint_costs(root,expected_plan_hash=p['plan_hash'],expected_ledger_hash=last)


def test_actual_small_native_joint_queue_charges_all_nominal_probes_without_fitting_again(tmp_path,cpu_training_evidence):
    import copy
    from test_joint_workers import joint_fixture
    from treeneuron.research.joint_workers import JointWorkers
    from treeneuron.train import source_hash
    module=api();original=joint_fixture(cpu_training_evidence);identity=copy.deepcopy(original['identity'])
    identity.update(schema='research/v2-M3-queue-identity',source_sha256=source_hash())
    p=make_queue_plan(identity,[{k:v for k,v in j.items() if k!='attempt_id'} for j in original['attempts']])
    root=tmp_path/'queue';run_queue(p,root,JointWorkers(p),check_identity=lambda _:None)
    ledger=AttemptLedger.resume(root,p);before=len(ledger.events)
    out=module.collect_joint_costs(root,expected_plan_hash=p['plan_hash'],expected_ledger_hash=ledger.events[-1]['event_hash'])
    members=out['procedures']['fit-short_singleton']['member_job_ids']
    assert {'probe-keep',*('probe-'+str(i) for i in range(8))}.issubset(members)
    assert out['artifact_inventories_authenticated'] is True and out['cohort_charged_job_ns']==ledger.charged_ns
    assert len(AttemptLedger.resume(root,p).events)==before and out['T_accessed'] is False
