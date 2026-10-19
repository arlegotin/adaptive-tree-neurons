"""Full native resource spine uses different teachers from scientific M3."""
from collections import Counter
from copy import deepcopy

import pytest

from test_joint_plan import config


def target():
    from treeneuron.research.config import expand
    from treeneuron.research.joint_graph import compile_joint_graph
    return compile_joint_graph(expand(config()).to_dict())


def test_joint_qualification_keeps_full_horizons_and_separate_engineering_seeds():
    from treeneuron.research.joint_qualification import build_joint_qualification_plan,workload_key
    t=target();q=build_joint_qualification_plan(t)
    assert len(q['attempts'])==258
    assert q['identity']['admission']=='engineering_resource_qualification_only'
    assert q['identity']['scientific_admission'] is False
    assert q['identity']['maximum_qualification_updates']==107500
    assert Counter(j['kind'] for j in q['attempts'])==dict(data=1,parent=1,score=1,placement=4,
        local_model=4,probe=65,short_selection=1,fit=10,evaluate=10,qualification_fixture=160,extension_decision=1)
    actual={j['job_id']:j for j in t['attempts']}
    for j in q['attempts']:
        p=j['payload']
        if j['kind']=='data':
            assert p['teacher_seed']!=actual[j['job_id']]['payload']['teacher_seed']
            assert p['data_seed']!=actual[j['job_id']]['payload']['data_seed']
            assert p['evaluation_rows']==65536
        if j['kind']=='fit':assert p['max_updates']==8000 and p['stop_by_patience'] is False
        if j['kind']=='probe':assert p['max_updates']==300 and p['stop_by_patience'] is False
    keys={workload_key(j) for j in q['attempts'] if j['kind']!='qualification_fixture'}
    assert keys=={workload_key(j) for j in t['attempts']}
    assert len(keys)==35
    assert len({j['payload']['teacher_seed'] for j in q['attempts'] if 'teacher_seed' in j['payload']})==1


def test_qualification_rejects_a_changed_complete_scientific_queue():
    from treeneuron.config import canonical_hash
    from treeneuron.research.joint_qualification import build_joint_qualification_plan
    p=target();p['attempts'][0]['payload']['teacher_seed']+=1
    p['plan_hash']=canonical_hash({k:v for k,v in p.items() if k!='plan_hash'})
    with pytest.raises(ValueError):build_joint_qualification_plan(p)
