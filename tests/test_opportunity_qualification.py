"""An engineering spine cannot become scientific evidence or alter native caps."""
import copy
from collections import Counter
from pathlib import Path

import pytest


def target():
    from treeneuron.research.config import expand,load_research_config
    from treeneuron.research.opportunity_graph import compile_bounded_graph
    return compile_bounded_graph(expand(load_research_config(Path('configs/research/v2/M2-opportunity.json'))).to_dict())


def builder():
    from treeneuron.research.opportunity_qualification import build_qualification_plan
    return build_qualification_plan


def test_engineering_spine_covers_all_classes_with_fixed_native_cases_and_real_pair_search():
    from treeneuron.research.opportunity_resources import workload_manifest
    p=target();before=copy.deepcopy(p);q=builder()(p);m=workload_manifest(p)
    assert p==before and q['identity']['admission']=='engineering_resource_qualification_only'
    assert q['identity']['target_plan_hash']==p['plan_hash']
    mapping=q['identity']['target_mapping'];jobs={j['job_id']:j for j in q['attempts']}
    assert {r['workload_hash'] for r in mapping.values()}==set(m['classes'])
    kinds=Counter(j['kind'] for j in jobs.values())
    assert kinds['parent']==7 and kinds['fit']==49 and kinds['qualification_fixture']==448
    assert kinds['extension_decision']==1
    assert sum(m['jobs'][n]['maximum_updates'] for n in jobs if n in m['jobs'])==424600
    for name,j in jobs.items():
        assert set(j['depends_on'])<=set(jobs)
        if j['kind']=='qualification_fixture':continue
        if 'teacher_index' in j['payload']:assert j['payload']['teacher_index']==0
        if 'repetition' in j['payload']:assert j['payload']['repetition']==0
    pairs=[j for j in jobs.values() if j['kind']=='fit' and j['payload']['matrix']=='M2d' and len(j['payload']['unit_ids'])==2]
    assert len(pairs)==28 and sum(7 in j['payload']['unit_ids'] for j in pairs)==7
    oracle=next(j for j in jobs.values() if j['kind']=='pair_oracle')
    assert set(oracle['depends_on'])=={j['job_id'] for j in pairs}
    assert q['identity']['scientific_admission'] is False and q['identity']['T_accessed'] is False


def test_only_declared_eager_fallback_and_extension_fixture_depart_from_original_jobs():
    from treeneuron.config import canonical_hash
    p=target();q=builder()(p);original={j['job_id']:j for j in p['attempts']}
    altered=[]
    for j in q['attempts']:
        if j['kind']=='qualification_fixture':
            assert j['payload']['evidence_kind']=='synthetic_resource_fixture'
            continue
        old=original[j['job_id']]
        if j!=old:altered.append(j)
        assert q['identity']['target_mapping'][j['job_id']]['target_job_sha256']==canonical_hash(old)
        if j['kind']=='fit' and j['payload']['matrix']=='M2a':assert j['payload']['snapshot']=='late_G_best'
    assert Counter(j['kind'] for j in altered)=={'parent':1,'extension_decision':1}
    parent=next(j for j in altered if j['kind']=='parent')
    assert parent['payload']['max_updates']==8000 and parent['payload']['stop_by_patience'] is False
    assert len(parent['depends_on'])==1 and parent['depends_on'][0].startswith('data:')
    extension=next(j for j in altered if j['kind']=='extension_decision')
    assert len(extension['depends_on'])==448
    assert all(n.startswith('qualification_fixture:') for n in extension['depends_on'])
    assert q['identity']['engineering_exceptions']==['eager_fixed_budget_parent','synthetic_extension_inputs']


def test_changed_target_inventory_is_not_accepted_by_qualification_builder():
    from treeneuron.config import canonical_hash
    p=target();p['attempts'][0]['payload']['teacher_seed']+=1
    p['plan_hash']=canonical_hash({k:v for k,v in p.items() if k!='plan_hash'})
    with pytest.raises(ValueError,match='target'):
        builder()(p)


def test_synthetic_extension_runs_actual_decision_without_becoming_native_evidence(tmp_path):
    from treeneuron.research import opportunity_qualification as module
    p=target();q=builder()(p);worker=module.QualificationWorkers(p,q)
    extension=next(j for j in q['attempts'] if j['kind']=='extension_decision')
    dependencies={}
    for index,job in enumerate(j for j in q['attempts'] if j['kind']=='qualification_fixture'):
        directory=tmp_path/str(index);directory.mkdir()
        value=worker(job,directory,{},0)
        assert value['evidence_kind']=='synthetic_resource_fixture'
        dependencies[job['job_id']]=dict(value=value)
    directory=tmp_path/'decision';directory.mkdir()
    result=worker(extension,directory,dependencies,0)
    assert result['status']=='completed_eligible' and result['summary']['teacher_ids']==[0,1,2,3]
    assert result['summary']['extension_required'] is True
    assert result['evidence_kind']=='synthetic_resource_decision'
    assert result['scientific_admission'] is False and result['T_accessed'] is False
    assert len(result['synthetic_dependency_ids'])==448
    assert (directory/'decision.json').exists()


def test_engineering_worker_rejects_changed_job_and_wrong_target_mapping(tmp_path):
    from treeneuron.research import opportunity_qualification as module
    from treeneuron.config import canonical_hash
    p=target();q=builder()(p);bad=copy.deepcopy(q)
    next(iter(bad['identity']['target_mapping'].values()))['workload_hash']='forged'
    bad['plan_hash']=canonical_hash({k:v for k,v in bad.items() if k!='plan_hash'})
    with pytest.raises(ValueError,match='qualification'):
        module.QualificationWorkers(p,bad)
    worker=module.QualificationWorkers(p,q)
    j=copy.deepcopy(next(j for j in q['attempts'] if j['kind']=='qualification_fixture'))
    j['payload']['procedure_value']['status']='numerical_failure'
    with pytest.raises(ValueError,match='job'):
        worker(j,tmp_path,{},0)


def test_actual_topology_requires_full_count_and_warm_bracket_geometry():
    from treeneuron.research import opportunity_qualification as module
    expected=dict(kind='fit',shape=dict(d=16,parent_h=64),hidden_width=64,gate_count=16)
    topology=dict(kind='selective_relu_scalar',layer=dict(d=16,h=64,grown_ids=list(range(16))))
    assert module.topology_matches(topology,expected)
    topology['layer']['grown_ids']=list(range(15))
    assert not module.topology_matches(topology,expected)
    expected.update(hidden_width=95,gate_count=0)
    assert module.topology_matches(dict(kind='widened_relu_scalar',layer=dict(d=16,h=95,parent_width=64)),expected)
    assert not module.topology_matches(dict(kind='widened_relu_scalar',layer=dict(d=16,h=94,parent_width=64)),expected)


def test_partial_native_qualification_is_authenticated_and_never_admitted(tmp_path):
    from treeneuron.research import opportunity_qualification as module
    from treeneuron.research.workqueue import run_queue
    p=target();q=builder()(p);worker=module.observed_qualification_workers(p,q)
    name=next(j['job_id'] for j in q['attempts'] if j['kind']=='data' and j['payload']['population']=='primary')
    result=run_queue(q,tmp_path/'queue',worker,check_identity=lambda _:None,admitted_job_ids=[name])
    assert result['status']=='incomplete'
    report=module.collect_qualification_profiles(tmp_path/'queue',p,expected_plan_hash=q['plan_hash'])
    assert report['target_plan_hash']==p['plan_hash'] and len(report['profiles'])==1
    assert len(report['missing_workload_hashes'])==52
    assert report['scientific_admission'] is False and report['T_accessed'] is False
    assert report['profiles'][0]['provenance']['target_job_id']==name
    assert report['profiles'][0]['procedure_status']=='completed_eligible'
    assert report['profiles'][0]['provenance']['evidence_kind']=='native_engineering_resource_workload'
    from treeneuron.research.opportunity_execution import make_opportunity_admission
    with pytest.raises(ValueError,match='all53'):
        make_opportunity_admission(p,tmp_path/'queue',available_worker_seconds=604800)
    with pytest.raises(ValueError,match='commitment|qualification'):
        module.collect_qualification_profiles(tmp_path/'queue',p,expected_plan_hash='0'*64)


from test_train import cpu_training_evidence


def test_saved_topology_uses_real_checkpoint_and_preserves_cpu_rng(tmp_path,cpu_training_evidence):
    import torch
    from test_opportunity_workers import tiny_plan
    from treeneuron.research.opportunity_workers import OpportunityWorkers
    from treeneuron.research.workqueue import run_queue,authenticated_queue
    from treeneuron.research.opportunity_qualification import _saved_topology
    p=tiny_plan(cpu_training_evidence);root=tmp_path/'native'
    run_queue(p,root,OpportunityWorkers(p),check_identity=lambda _:None,
        admitted_job_ids=['data','parent','score','good'])
    with authenticated_queue(root,expected_plan_hash=p['plan_hash']) as (_,_,results):
        before=torch.get_rng_state().clone()
        descriptor=dict(kind='fit',shape=dict(d=2,parent_h=8),hidden_width=8,gate_count=2)
        actual=_saved_topology(results['good']['value'],descriptor)
        assert actual['matches_workload'] and actual['topology']['layer']['grown_ids']==[0,2]
        assert actual['total_parameters']==45 and torch.equal(before,torch.get_rng_state())
        wrong={**descriptor,'gate_count':3}
        assert _saved_topology(results['good']['value'],wrong)['matches_workload'] is False
