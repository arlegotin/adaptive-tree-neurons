"""Qualified search requires every paid native attempt, beyond a score journal."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.qualified_search_audit_v2') is not None,'Missing complete qualified native search audit'
    from treeneuron.research import qualified_search_audit_v2
    return qualified_search_audit_v2


def test_handfilled_qualification_cannot_audit_a_search_or_open_its_inputs(tmp_path):
    module=api()
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import _inventory
    root=tmp_path/'false-qualification';root.mkdir()
    (root/'manifest.json').write_text(json.dumps(dict(native_workload_qualified=True)))
    with pytest.raises(ValueError,match='schema|registry|authority'):
        module.audit_primary_qualified_search(practical_plan=None,expected_practical_plan_hash='0'*64,
            job_id='airfoil/outer-0/repeat-0/P00',support_reference=None,
            qualification_reference=dict(root=str(root),inventory_sha256=canonical_hash(_inventory(root))),
            search_directory=tmp_path/'never-created',expected_search_plan_sha256='0'*64,
            result_path=tmp_path/'never-opened',expected_result_sha256='0'*64,expected_journal_sha256='0'*64)
    assert not (tmp_path/'never-created').exists()


@pytest.mark.parametrize('fault',[None,'omitted','extra','foreign','symlink'])
def test_attempt_inventory_requires_all_and_only_its_paid_committed_files(tmp_path,fault):
    module=api();search=tmp_path/'search';attempt=search/'attempt-000';attempt.mkdir(parents=True)
    from treeneuron.research.method_worker import _sha
    (attempt/'lost-trial.log').write_text('Actual losing attempt bytes')
    rows=[dict(path='attempt-000/lost-trial.log',sha256=_sha(attempt/'lost-trial.log'))]
    if fault=='omitted':rows=[]
    elif fault=='extra':(attempt/'uncommitted.log').write_text('not retained by journal')
    elif fault=='foreign':
        path=search/'attempt-001';path.mkdir();(path/'other.log').write_text('other candidate')
        rows=[dict(path='attempt-001/other.log',sha256=_sha(path/'other.log'))]
    elif fault=='symlink':(attempt/'alias').symlink_to(attempt/'lost-trial.log')
    if fault is None:module._exact_attempt_files(search,0,rows)
    else:
        with pytest.raises(ValueError):module._exact_attempt_files(search,0,rows)


@pytest.fixture(scope='module')
def failed_candidate(tmp_path_factory):
    cache=[]
    def get():
        if not cache:
            from test_native_trial_bridge import setup,bridge,run
            from treeneuron.research.native_trial_bridge_v2 import InspectedNativeTrialBridge
            root=tmp_path_factory.mktemp('actual-paid-failed-native')
            with pytest.MonkeyPatch.context() as patch:args,_,spec,plan=setup(root,patch)
            search=root/'search';result=run(search,plan,bridge(InspectedNativeTrialBridge,args,plan,search,rss_limit_bytes=1),max_attempts_this_invocation=1)
            assert result['records'][0]['status']=='interrupted_recoverable' and result['winner'] is None
            starts=[json.loads(line)['payload'] for line in (search/'events.jsonl').read_text().splitlines() if json.loads(line)['kind']=='attempt_started']
            cache.append(dict(search_directory=search,search_plan=plan.to_dict(),spec=spec,request=starts[0],
                prepared_root=args['prepared_root'],expected_manifest_sha256=args['expected_manifest_sha256'],outer_fold=0,
                official_schedule_reference=None,native_hpo_seed=None))
        return cache[0]
    return get


def test_actual_partial_RSS_candidate_is_audited_as_paid_failure_without_registry_authority(failed_candidate):
    module=api();row=failed_candidate()
    from treeneuron.research.workqueue import _inventory
    before=_inventory(row['search_directory']);out=module.inspect_paid_native_attempt(**row)
    assert out['attempt_evidence_authenticated'] is True
    assert out['status']=='interrupted_recoverable' and out['native_outcome']['status']=='resource_exhausted'
    assert out['inspected_V_score'] is None and out['candidate_component'] is None
    assert out['quality_eligible'] is out['native_workload_qualified'] is out['T_accessed'] is False
    assert _inventory(row['search_directory'])==before


@pytest.mark.parametrize('fault',['raw','compiled','seed','request_hash','authority','status','recovery','recovery_type'])
def test_recommitted_failed_candidate_cannot_change_proposal_or_recovery_identity(failed_candidate,fault):
    module=api();row=failed_candidate();attempt=row['search_directory']/'attempt-000'
    path=attempt/('recovery.json' if fault in ('recovery','recovery_type') else 'bridge.json');before=path.read_bytes()
    try:
        value=json.loads(before)
        if fault=='raw':value['raw_configuration']={'undeclared':True}
        elif fault=='compiled':value['compiled_configuration']={'undeclared':True}
        elif fault=='seed':value['fit_seed']+=1
        elif fault=='request_hash':value['native_request_sha256']='0'*64
        elif fault=='authority':value['quality_eligible']=True
        elif fault=='status':value['status']='numerical_failure'
        else:value['initial_attempt_index']=False if fault=='recovery_type' else 1
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):module.inspect_paid_native_attempt(**row)
    finally:path.write_bytes(before)


@pytest.mark.parametrize('fault',[None,'score','authority','registry','plan','upstream','clock'])
def test_failure_admission_packet_cannot_promote_or_hide_an_unqualified_outcome(tmp_path,fault):
    module=api();reference=dict(root=str(tmp_path/'untrusted-reference'),inventory_sha256='a'*64)
    # Pure comparison only: these literals supply no registry/candidate proof.
    outcome=dict(status='numerical_failure',candidate_component=None,inspected_V_score=None,
        expected_trial=dict(plan_hash='b'*64,source_sha256='c'*64,candidate_id='anchor-0'))
    packet=dict(schema='research/v2-qualified-native-candidate-admission',qualification_reference=reference,
        source_sha256='c'*64,plan_hash='b'*64,candidate_id='anchor-0',upstream_status='numerical_failure',
        status='numerical_failure',V_score=None,component=None,native_workload_qualified=False,quality_eligible=False,
        HPO_journal_authenticated=False,protected_access_authorized=False,T_accessed=False,
        callback_body_before_publication_ns=10,clock_scope=module._ADMISSION_CLOCK,scope=module._ADMISSION_SCOPE)
    if fault=='score':packet['V_score']=0.
    elif fault=='authority':packet['quality_eligible']=True
    elif fault=='registry':packet['qualification_reference']=dict(reference,inventory_sha256='0'*64)
    elif fault=='plan':packet['plan_hash']='0'*64
    elif fault=='upstream':packet['upstream_status']='completed_ineligible'
    elif fault=='clock':packet['callback_body_before_publication_ns']=True
    if fault is None:module._check_admission(packet,outcome,reference,charged_ns=20)
    else:
        with pytest.raises(ValueError):module._check_admission(packet,outcome,reference,charged_ns=20)


def test_actual_completed_engineering_candidate_joins_its_native_outcome_and_V_without_promoting_it(tmp_path,monkeypatch):
    module=api()
    from test_native_trial_bridge import setup,bridge,run
    from treeneuron.research.native_trial_bridge_v2 import InspectedNativeTrialBridge
    args,_,spec,plan=setup(tmp_path,monkeypatch);search=tmp_path/'search'
    run(search,plan,bridge(InspectedNativeTrialBridge,args,plan,search),max_attempts_this_invocation=1)
    request=next(json.loads(line)['payload'] for line in (search/'events.jsonl').read_text().splitlines()
        if json.loads(line)['kind']=='attempt_started')
    proof=module.inspect_paid_native_attempt(search_directory=search,search_plan=plan.to_dict(),spec=spec,request=request,
        prepared_root=args['prepared_root'],expected_manifest_sha256=args['expected_manifest_sha256'],outer_fold=0,
        official_schedule_reference=None,native_hpo_seed=None)
    assert proof['status']=='completed_ineligible' and proof['candidate_component']['candidate_evidence_authenticated'] is True
    assert proof['native_outcome']['status']=='completed_ineligible' and proof['inspected_V_score']>=0
    assert proof['quality_eligible'] is proof['native_workload_qualified'] is False


def test_actual_native_replay_RSS_failure_keeps_validation_unopened(tmp_path,monkeypatch):
    module=api()
    from test_native_generation_chain import actual
    from treeneuron.research.native_candidate_v2 import inspect_primary_native_candidate
    args,generation=actual(tmp_path,monkeypatch);out=tmp_path/'failed-inspection'
    value=inspect_primary_native_candidate(generation_arguments=args,immutable_generation_root=generation,
        output=out,interpreter=__import__('sys').executable,rss_limit_bytes=1)
    assert value['status']=='resource_exhausted' and value['V_coordinator_invoked'] is False
    proof=module._failed_inspection(out,generation,args)
    assert proof['execution_status']=='resource_exhausted' and proof['eligible_model'] is False
    path=out/'result.json';before=path.read_bytes()
    try:
        record=json.loads(before);record['V_labels_read_by_coordinator']=True;path.write_text(json.dumps(record))
        with pytest.raises(ValueError):module._failed_inspection(out,generation,args)
    finally:path.write_bytes(before)


def test_actual_second_paid_same_seed_restart_is_joined_to_the_preceding_failed_attempt(tmp_path,monkeypatch):
    module=api()
    from test_native_trial_bridge import setup,bridge,run
    from treeneuron.research.native_trial_bridge_v2 import InspectedNativeTrialBridge
    args,_,spec,plan=setup(tmp_path,monkeypatch);search=tmp_path/'search'
    callback=bridge(InspectedNativeTrialBridge,args,plan,search,rss_limit_bytes=1)
    first=run(search,plan,callback,max_attempts_this_invocation=1)
    second=run(search,plan,callback,resume=True,expected_journal_hash=first['journal_hash'])
    assert second['attempts_paid']==2 and second['stop_reason']=='recovery_exhausted' and second['winner'] is None
    request=[json.loads(line)['payload'] for line in (search/'events.jsonl').read_text().splitlines()
        if json.loads(line)['kind']=='attempt_started'][1]
    row=dict(search_directory=search,search_plan=plan.to_dict(),spec=spec,request=request,
        prepared_root=args['prepared_root'],expected_manifest_sha256=args['expected_manifest_sha256'],outer_fold=0,
        official_schedule_reference=None,native_hpo_seed=None)
    proof=module.inspect_paid_native_attempt(**row)
    assert proof['native_outcome']['recovery']==dict(resume=False,reason='same_seed_restart_without_optimizer_checkpoint')
    assert proof['status']=='interrupted_recoverable' and proof['inspected_V_score'] is None
    assert proof['quality_eligible'] is False


@pytest.mark.parametrize('preparations,minimum,valid',[([10],10,True),([0,10],10,False),([9],10,False),([0],0,True)])
def test_native_sampler_cost_is_paid_before_first_search_prefix_not_added_afterward(preparations,minimum,valid):
    module=api();assert hasattr(module,'_check_sampler_preparation'),'Missing first-session native prerequisite cost join'
    events=[dict(kind='session_started',payload=dict(invocation_preparation_ns=n)) for n in preparations]
    if valid:module._check_sampler_preparation(events,minimum)
    else:
        with pytest.raises(ValueError):module._check_sampler_preparation(events,minimum)
