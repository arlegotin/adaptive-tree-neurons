"""The exact initial M2 inventory expands into a data/parent/fit/evaluation DAG."""
from pathlib import Path
from collections import Counter

import pytest


def test_initial_graph_preserves_every_nominal_case_and_shared_prerequisite():
    from treeneuron.research.config import expand,load_research_config
    from treeneuron.research.opportunity_graph import compile_initial_graph
    study=expand(load_research_config(Path('configs/research/v2/M2-opportunity.json'))).to_dict()
    graph=compile_initial_graph(study)
    rows=graph['attempts'];counts=Counter(r['kind'] for r in rows)
    assert counts==dict(data=8,parent=48,score=32,horizon_binding=8,fit=1168,head=8,
        evaluate=1184,local_diagnostic=8,pair_oracle=8)
    assert len(rows)==2472
    byid={r['job_id']:r for r in rows}
    cases={r['payload']['attempt_id'] for r in rows if r['kind'] in ('fit','head')}
    assert cases=={r['attempt_id'] for r in study['attempts']}
    assert len({r['payload']['teacher_seed'] for r in rows if r['kind']=='data'})==8
    for row in rows:
        assert all(dep in byid for dep in row['depends_on'])
        if row['kind']=='pair_oracle':
            assert row['dependency_policy']=='all_terminal'
            assert len(row['depends_on'])==28
            assert all(byid[dep]['kind']=='fit' and len(byid[dep]['payload']['unit_ids'])==2
                       for dep in row['depends_on'])
        if row['kind']=='fit' and row['payload']['matrix']=='M2c':
            assert any(byid[dep]['kind']=='horizon_binding' for dep in row['depends_on'])
    assert graph['identity']['execution_implemented'] is False
    assert graph['identity']['study_plan_hash']==study['plan_hash']
    assert graph['identity']['design_bindings']==study['design_bindings']
    assert not any('fallback' in r['job_id'] or r['payload'].get('teacher_index',0)>=4 for r in rows)


def test_graph_rejects_changed_study_bytes_before_expansion():
    from treeneuron.research.config import expand,load_research_config
    from treeneuron.research.opportunity_graph import compile_initial_graph
    study=expand(load_research_config(Path('configs/research/v2/M2-opportunity.json'))).to_dict()
    study['attempts'][0]['requested_count']=1
    with pytest.raises(ValueError,match='digest'):
        compile_initial_graph(study)


def test_bounded_graph_precommits_all_fallback_and_single_extension_work():
    from treeneuron.research.config import expand,load_research_config
    from treeneuron.research.opportunity_graph import compile_bounded_graph
    study=expand(load_research_config(Path('configs/research/v2/M2-opportunity.json'))).to_dict()
    graph=compile_bounded_graph(study);rows=graph['attempts'];byid={r['job_id']:r for r in rows}
    assert len(rows)==3853
    assert Counter(r['kind'] for r in rows)==dict(data=12,parent=104,score=88,horizon_binding=8,
        fit=1768,head=16,evaluate=1792,local_diagnostic=8,pair_oracle=8,
        fallback_gate=16,snapshot_gate=32,extension_decision=1)
    assert {r['payload']['attempt_id'] for r in rows if r['kind'] in ('fit','head')}=={
        c['attempt_id'] for c in study['attempts']+study['conditional_attempts']}
    for row in rows:
        if row['kind'] in ('fallback_gate','snapshot_gate','extension_decision'):
            assert row['dependency_policy']=='all_terminal'
        if row['kind']=='parent' and 'fixed_parent_budget_fallback' in row['job_id']:
            assert row['payload']['stop_by_patience'] is False
            assert any(byid[d]['kind']=='fallback_gate' for d in row['depends_on'])
        if row['kind']=='score' and 'fixed_parent_budget_fallback' in row['job_id']:
            assert any(byid[d]['kind']=='snapshot_gate' for d in row['depends_on'])
        if row['kind']=='data' and row['payload']['teacher_index']>=4:
            assert row['depends_on']==['extension_decision:primary']
    decision=byid['extension_decision:primary']
    assert len(decision['depends_on'])==448  #288 initial M2a/e evaluations +160 fallback slots
    assert all(byid[d]['kind']=='evaluate' for d in decision['depends_on'])
    assert all(byid[d]['payload']['matrix'] in ('M2a','M2e_neural','M2e_head') for d in decision['depends_on'])
    assert not any(r['payload'].get('teacher_index',0)>=8 for r in rows)
    assert graph['identity']['execution_implemented'] is False
