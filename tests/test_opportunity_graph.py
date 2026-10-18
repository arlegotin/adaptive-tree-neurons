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
