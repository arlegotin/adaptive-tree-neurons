"""Unqualified fixtures exercise secondary plumbing, never a fake global gate."""
import importlib.util
import copy
import json
from pathlib import Path
import sys

import pytest

from test_primary_search_declaration_v2 import study
from test_released_transfer_v2 import toy


def api():
    assert importlib.util.find_spec('treeneuron.research.secondary_native_search_v2'), 'Missing secondary native search route'
    from treeneuron.research import secondary_native_search_v2
    return secondary_native_search_v2


@pytest.mark.parametrize('family',['P00','P13'])
def test_secondary_metadata_keeps_original_support_and_paired_seeds_without_execution(study,family):
    module=api();plan,support=study;job_id=f'airfoil/outer-1/repeat-1/{family}'
    result=module._secondary_declaration(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],job_id=job_id,support_reference=support)
    from treeneuron.config import canonical_hash
    assert result['declaration_hash']==canonical_hash({k:v for k,v in result.items() if k!='declaration_hash'})
    expected=next(j for j in plan.to_dict()['jobs'] if j['job_id']==job_id)
    search=result['search_plan']
    assert result['job']==expected and search['fit_seed']==expected['fit_seed'] and search['hpo_seed32']==expected['hpo_seed32']
    assert search['deadline_ns'] is None and search['max_attempts']==32
    assert result['native_workload_qualified'] is result['protected_access_authorized'] is result['execution_implemented'] is False
    if family=='P13':assert len(search['anchors'])+len(search['candidates'])==29
    with pytest.raises(ValueError):
        module._secondary_declaration(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],
            job_id=f'airfoil/outer-0/repeat-1/{family}',support_reference=support)


def setup(toy,tmp_path):
    module=api()
    from treeneuron.research.method_worker import scientific_source_sha256,_ROOT,_sha
    from treeneuron.research.released_transfer_v2 import _ReleasedRoles
    from treeneuron.research.search_controller_v2 import make_search_plan
    from treeneuron.research.search_supports_v2 import family_search_spec
    root,digest,legacy=toy;events=[]
    roles=_ReleasedRoles(root,expected_manifest_sha256=digest,record=lambda **kw:events.append(kw))
    schema=dict(columns=legacy.manifest['feature_names'],categorical=legacy.manifest['categorical_features'])
    spec=family_search_spec('P13',schema)
    plan=make_search_plan(identity=dict(job_id='toy-secondary/mean-ridge',family_id='P13',search_repeat=0,
        source_sha256=scientific_source_sha256(),runtime_sha256=_sha(_ROOT/'locks/research-v2/core.json'),
        support_sha256=spec['support_hash'],split_sha256=legacy.manifest['splits'][1]['split_hash']),
        fit_seed=17,hpo_seed32=91,anchors=[a['config'] for a in spec['anchors']],mode='finite',support=None,
        candidates=spec['finite_configurations'][1:],max_attempts=2,deadline_ns=None,prefixes_ns=[600000000000])
    staged=module._stage(roles,plan=plan,prepared_root=root,expected_manifest_sha256=digest,outer_fold=1,output=tmp_path/'template')
    return roles,events,plan,staged,spec


def test_real_two_candidate_secondary_search_preserves_native_proposals_and_no_promotion(toy,tmp_path):
    module=api();roles,events,plan,staged,spec=setup(toy,tmp_path)
    from treeneuron.research.search_controller_v2 import run_search
    search=tmp_path/'search'
    callback=module._SecondaryInspectedBridge(role_store=roles,plan=plan,template_path=Path(staged['request_path']),
        expected_template_sha256=staged['request_sha256'],prepared_root=toy[0],expected_manifest_sha256=toy[1],outer_fold=1,
        search_directory=search,interpreter=sys.executable)
    result=run_search(search,plan=plan,evaluate_trial=callback,invocation_preparation_ns=callback.preparation_ns)
    assert result['attempts_paid']==2 and result['winner'] is None
    assert [r['status'] for r in result['records']]==['completed_ineligible','completed_ineligible']
    for index in range(2):
        attempt=search/f'attempt-{index:03}'
        request=json.loads((attempt/'generation/input/request.json').read_text())
        packet=json.loads((attempt/'bridge.json').read_text())
        assert request['config']==spec['finite_configurations'][index] and set(request['files'])=={'F','G','V_features'}
        assert request['budget']=={'native_stopping':True} and request['fit_seed']==17
        assert packet['quality_eligible'] is packet['native_workload_qualified'] is packet['T_accessed'] is False
        expected={key:packet[key] for key in ('plan_hash','source_sha256','candidate_id','candidate_origin','raw_configuration',
            'compiled_configuration','official_schedule_reference','native_hpo_seed','fit_seed')}
        component=module._secondary_component(roles,attempt_directory=attempt,expected_trial=expected,
            prepared_root=toy[0],expected_manifest_sha256=toy[1],outer_fold=1)
        assert component['candidate_evidence_authenticated'] is True and component['inspected_V_score']==packet['inspected_V_score']
        assert component['quality_eligible'] is component['native_workload_qualified'] is component['T_accessed'] is False
    assert all(e['outer_fold']==1 for e in events) and not (toy[0]/'access.jsonl').exists()
    journal=[json.loads(line) for line in (search/'events.jsonl').read_text().splitlines()]
    starts=[event['payload'] for event in journal if event['kind']=='attempt_started']
    for index,request in enumerate(starts):
        paid=module._paid_attempt(roles,search_directory=search,search_plan=plan.to_dict(),spec=spec,request=request,
            prepared_root=toy[0],expected_manifest_sha256=toy[1],outer_fold=1,official_schedule_reference=None,native_hpo_seed=None)
        assert paid['attempt_evidence_authenticated'] is True and paid['status']=='completed_ineligible'
        assert paid['candidate_component']['generation']['outer_fold']==1
        assert paid['quality_eligible'] is paid['native_workload_qualified'] is paid['T_accessed'] is False
        if index==0:
            # Pure, untrusted companion metadata validation grants no actual
            # qualification. No loader/global barrier success is replaced.
            qref=dict(root='/unqualified-metadata-only',inventory_sha256='a'*64)
            gref=dict(path='/unqualified-metadata-only.json',sha256='b'*64)
            packet=dict(schema='research/v2-qualified-secondary-native-candidate-admission',qualification_reference=qref,
                global_barrier_reference=gref,source_sha256=paid['expected_trial']['source_sha256'],plan_hash=paid['expected_trial']['plan_hash'],
                candidate_id=paid['expected_trial']['candidate_id'],upstream_status='completed_ineligible',status='completed_eligible',
                V_score=paid['inspected_V_score'],component=paid['candidate_component'],native_workload_qualified=True,quality_eligible=True,
                HPO_journal_authenticated=False,protected_access_authorized=False,T_accessed=False,
                callback_body_before_publication_ns=paid['callback_body_ns']+1,clock_scope=module._ADMISSION_CLOCK,scope=module._ADMISSION_SCOPE)
            charged=paid['callback_body_ns']+2
            assert module._check_secondary_admission(packet,paid,qualification_reference=qref,barrier_reference=gref,charged_ns=charged) is None
            faults=[dict(global_barrier_reference=dict(gref,sha256='c'*64)),dict(V_score=True),dict(quality_eligible=False),
                dict(HPO_journal_authenticated=True),dict(status='completed_ineligible'),dict(callback_body_before_publication_ns=charged+1)]
            for fault in faults:
                changed=copy.deepcopy(packet);changed.update(fault)
                with pytest.raises(ValueError):module._check_secondary_admission(changed,paid,qualification_reference=qref,barrier_reference=gref,charged_ns=charged)


def test_secondary_template_or_bridge_cannot_substitute_primary_roles(toy,tmp_path):
    module=api();roles,events,plan,staged,spec=setup(toy,tmp_path);output=tmp_path/'forbidden'
    with pytest.raises(PermissionError):module._stage(roles,plan=plan,prepared_root=toy[0],expected_manifest_sha256=toy[1],outer_fold=0,output=output)
    assert not output.exists()
    with pytest.raises(PermissionError):
        module._SecondaryInspectedBridge(role_store=roles,plan=plan,template_path=Path(staged['request_path']),
            expected_template_sha256=staged['request_sha256'],prepared_root=toy[0],expected_manifest_sha256=toy[1],outer_fold=0,
            search_directory=tmp_path/'search',interpreter=sys.executable)
    assert not (tmp_path/'search').exists()


@pytest.mark.parametrize('entry',['factory','declaration','audit'])
def test_public_secondary_entry_refuses_false_global_qualification_before_preparation(tmp_path,monkeypatch,entry):
    module=api()
    from treeneuron.research.transfer import TransferStore
    def forbidden(*a,**kw):raise AssertionError('False qualification cannot reach role metadata or output')
    monkeypatch.setattr(TransferStore,'__init__',forbidden)
    with pytest.raises((ValueError,FileNotFoundError)):
        kwargs=dict(barrier_reference=dict(path=str(tmp_path/'absent-barrier'),sha256='a'*64),
            qualification_reference=dict(root=str(tmp_path/'absent-registry'),inventory_sha256='b'*64),job_id='energy/outer-1/repeat-0/P13')
        if entry=='factory':module.prepare_secondary_native_callback(**kwargs,output=tmp_path/'forbidden-output',search_directory=tmp_path/'forbidden-search')
        elif entry=='declaration':module.compile_secondary_search_declaration(**kwargs)
        else:module.audit_secondary_qualified_search(**kwargs,search_directory=tmp_path/'forbidden-search',expected_search_plan_sha256='c'*64,
            result_path=tmp_path/'forbidden-result.json',expected_result_sha256='d'*64,expected_journal_sha256='e'*64)
    assert list(tmp_path.iterdir())==[]
