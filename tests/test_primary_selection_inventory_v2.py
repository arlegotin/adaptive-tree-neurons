"""File-backed primary selection barrier facts, without native qualification."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from treeneuron.config import canonical_hash


def api():
    assert importlib.util.find_spec('treeneuron.research.primary_selection_v2') is not None, 'Missing immutable primary selection inventory'
    from treeneuron.research import primary_selection_v2
    return primary_selection_v2


def ref(path):return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def publish(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n');return ref(path)


def prepare(root,*,native_quality=False):
    from test_practical_workload import fixtures
    from treeneuron.research.workload import expand_practical
    from treeneuron.research.method_worker import scientific_source_sha256
    from treeneuron.research.search_controller_v2 import make_search_plan,run_search
    source=scientific_source_sha256()
    arguments=dict(native_quality=True) if native_quality else {}
    plan=expand_practical(fixtures(),support_lock_sha256='a'*64,source_sha256=source,runtime_sha256='c'*64,lock_sha256='d'*64,**arguments)
    procedure=publish(root/'procedure.json',dict(schema='explicit_unqualified_fixture_procedure',source_sha256=source))
    selections={}
    for job in (j for j in plan.to_dict()['jobs'] if j['tranche']=='primary'):
        directory=root/job['job_id'];clock=[0]
        identity=dict(job_id=job['job_id'],family_id=job['family_id'],search_repeat=job['search_repeat'],
            source_sha256=source,runtime_sha256=job['runtime_sha256'],support_sha256='e'*64,split_sha256=job['split_hash'])
        search_budget=dict(deadline_ns=None,prefixes_ns=[600000000000,1800000000000,7200000000000]) if native_quality else {}
        search_plan=make_search_plan(identity=identity,fit_seed=17,hpo_seed32=job['hpo_seed32'],
            anchors=[dict(variant='native_anchor_fixture')],mode='finite',support=None,
            candidates=[dict(variant='first'),dict(variant='tied_later')],**search_budget)
        def trial(request):
            clock[0]+=10800000000000 if native_quality else 5;index=request['attempt_index'];file=Path(request['attempt_directory'])/'model.json'
            evidence=publish(file,dict(family=job['family_id'],scope='Artificial model bytes, no native fit',index=index))
            return dict(status='numerical_failure' if index==0 else 'completed_eligible',V_score=None if index==0 else 1.-(5e-13 if index==2 else 0.),
                files=[evidence],checkpoint_files=[],reason='Fixture paid failure' if index==0 else None)
        result=run_search(directory,plan=search_plan,evaluate_trial=trial,clock_ns=lambda:clock[0])
        assert result['winner']['attempt_index']==1
        selections[job['job_id']]=dict(search_plan=ref(directory/'plan.json'),
            search_result=publish(directory/'search-result.json',result),selected_model=ref(directory/'attempt-001/model.json'))
    return plan,procedure,selections


def seal(root,plan,procedure,selections):
    return api().seal_primary_selection_inventory(plan=plan,expected_plan_hash=plan.to_dict()['plan_hash'],
        global_procedure=procedure,selections=selections,output=root/'inventory.json')


def test_full_native_quality_primary_winners_can_complete_after_every_budget_prefix(tmp_path):
    plan,procedure,selections=prepare(tmp_path,native_quality=True)
    result=seal(tmp_path,plan,procedure,selections);models=result.payload['models']
    assert len(models)==168 and all(m['winner_attempt_index']==1 and m['paid_search_ns']==32400000000000 for m in models)
    assert result.payload['secondary_release_authorized'] is result.payload['protected_access_authorized'] is False
    loaded=api().load_primary_selection_inventory(result.path,expected_file_sha256=result.file_sha256,
        plan=plan,expected_plan_hash=plan.to_dict()['plan_hash'])
    assert loaded.payload==result.payload


@pytest.mark.parametrize('mutation',['deadline_stop','false_prefix'])
def test_quality_inventory_rejects_fabricated_deadlines_or_retroactive_prefix_winners(tmp_path,mutation):
    plan,procedure,selections=prepare(tmp_path,native_quality=True);entry=selections[next(iter(selections))]
    p=Path(entry['search_result']['path']);value=json.loads(p.read_text())
    if mutation=='deadline_stop':value['stop_reason']='deadline'
    else:value['prefixes']['7200000000000']['winner']=value['winner']
    entry['search_result']=publish(p,value)
    with pytest.raises(ValueError):seal(tmp_path,plan,procedure,selections)


def test_self_rehashed_job_budget_cannot_contradict_the_primary_search_contract(tmp_path):
    from treeneuron.config import canonical_json
    from treeneuron.research.workload import PracticalPlan
    plan,procedure,selections=prepare(tmp_path);value=plan.to_dict()
    value['jobs'][0]['budget']['deadline_ns']=None
    value['plan_hash']=canonical_hash({k:v for k,v in value.items() if k!='plan_hash'})
    changed=PracticalPlan(canonical_json(value))
    with pytest.raises(ValueError):seal(tmp_path,changed,procedure,selections)


def test_all_168_primary_winners_are_sealed_with_repeats_separate_and_no_release(tmp_path,monkeypatch):
    module=api();plan,procedure,selections=prepare(tmp_path)
    import numpy as np
    def forbidden(*a,**kw):raise AssertionError('Selection inventory must not decode labels/arrays')
    monkeypatch.setattr(np,'load',forbidden)
    result=seal(tmp_path,plan,procedure,selections);payload=result.payload
    assert len(payload['models'])==168 and len({m['job_id'] for m in payload['models']})==168
    assert {m['search_repeat'] for m in payload['models']}=={0,1}
    assert all(m['winner_attempt_index']==1 and m['attempts_paid']==3 and m['paid_search_ns']==15 for m in payload['models'])
    assert all(m['outer_fold']==(4 if m['dataset']=='bike' else 0) for m in payload['models'])
    assert payload['secondary_release_authorized'] is payload['protected_access_authorized'] is payload['native_qualification_authenticated'] is False
    assert payload['model_generation_authenticated'] is payload['search_journal_authenticated'] is payload['V_arithmetic_authenticated'] is False
    assert len(payload['artifact_references'])>=168*5
    loaded=module.load_primary_selection_inventory(result.path,expected_file_sha256=result.file_sha256,
        plan=plan,expected_plan_hash=plan.to_dict()['plan_hash'])
    assert loaded.payload==payload
    payload['models'].clear();assert len(result.payload['models'])==168
    before=result.path.read_bytes()
    with pytest.raises(ValueError):seal(tmp_path,plan,procedure,selections)
    assert result.path.read_bytes()==before


@pytest.mark.parametrize('change',['missing','extra_secondary','repeat_swap','selected_loser','late_winner','fake_winner',
    'paused_search','changed_loser','source','HPO_seed','budget','bad_external_digest'])
def test_incomplete_or_mismatched_primary_selection_cannot_be_sealed(change,tmp_path):
    module=api();plan,procedure,selections=prepare(tmp_path);first=next(iter(selections));entry=selections[first]
    if change=='missing':del selections[first]
    elif change=='extra_secondary':selections['airfoil/outer-1/repeat-0/P00']=entry
    elif change=='repeat_swap':
        second=first.replace('/repeat-0/','/repeat-1/');selections[first],selections[second]=selections[second],selections[first]
    elif change=='selected_loser':entry['selected_model']=ref(Path(entry['search_plan']['path']).parent/'attempt-002/model.json')
    elif change=='bad_external_digest':entry['search_result']['sha256']='0'*64
    elif change=='changed_loser':
        p=Path(entry['search_plan']['path']).parent/'attempt-000/model.json';p.write_text('Changed retained paid failure')
    elif change in ('source','HPO_seed','budget'):
        p=Path(entry['search_plan']['path']);value=json.loads(p.read_text())
        if change=='source':value['identity']['source_sha256']='0'*64
        elif change=='HPO_seed':value['hpo_seed32']+=1
        else:value['deadline_ns']=100;value['prefixes_ns']=[]
        value['plan_hash']=canonical_hash({k:v for k,v in value.items() if k!='plan_hash'})
        entry['search_plan']=publish(p,value)
    else:
        p=Path(entry['search_result']['path']);value=json.loads(p.read_text())
        if change=='paused_search':value['stop_reason']='operator_pause'
        elif change=='fake_winner':value['winner']=value['records'][2]
        else:
            value['records'][1]['completion_ns']=7200000000001
            value['records'][2]['completion_ns']=7200000000006
            value['winner']=value['records'][1]
        entry['search_result']=publish(p,value)
    with pytest.raises(ValueError):seal(tmp_path,plan,procedure,selections)
    assert not (tmp_path/'inventory.json').exists()


@pytest.mark.parametrize('change',['model','loser','procedure','result','inventory_self_rehash','external_digest'])
def test_loading_primary_seal_rechecks_all_its_external_bytes(change,tmp_path):
    module=api();plan,procedure,selections=prepare(tmp_path);sealed=seal(tmp_path,plan,procedure,selections)
    entry=next(iter(selections.values()));digest=sealed.file_sha256
    if change=='model':Path(entry['selected_model']['path']).write_text('Changed chosen model')
    elif change=='loser':(Path(entry['search_plan']['path']).parent/'attempt-000/model.json').write_text('Changed failed trial')
    elif change=='procedure':Path(procedure['path']).write_text('Changed frozen procedure')
    elif change=='result':Path(entry['search_result']['path']).write_text('Changed selected result')
    elif change=='external_digest':digest='0'*64
    else:
        value=json.loads(sealed.path.read_text());value['payload']['secondary_release_authorized']=True
        value['inventory_hash']=canonical_hash(value['payload']);digest=publish(sealed.path,value)['sha256']
    with pytest.raises(ValueError):module.load_primary_selection_inventory(sealed.path,expected_file_sha256=digest,
        plan=plan,expected_plan_hash=plan.to_dict()['plan_hash'])


@pytest.mark.parametrize('change',['truncated_finite','wrong_anchor','wrong_candidate_id','unearned_retry','nonrecord'])
def test_terminal_search_cannot_hide_finite_candidates_or_change_proposal_lineage(change,tmp_path):
    api();plan,procedure,selections=prepare(tmp_path);entry=next(iter(selections.values()))
    path=Path(entry['search_result']['path']);value=json.loads(path.read_text())
    if change=='truncated_finite':
        value['records']=value['records'][:2];value['attempts_paid']=2;value['total_charged_ns']=10
    elif change=='wrong_anchor':value['records'][0]['config']=dict(variant='different_anchor')
    elif change=='wrong_candidate_id':value['records'][0]['candidate_id']='finite-0'
    elif change=='unearned_retry':value['records'][0]['retry_index']=1
    else:value['records'][0]=None
    entry['search_result']=publish(path,value)
    with pytest.raises(ValueError):seal(tmp_path,plan,procedure,selections)
    assert not (tmp_path/'inventory.json').exists()
