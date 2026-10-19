"""The M3 queue keeps every paid probe and blocks extension data until its decision."""
from collections import Counter
from copy import deepcopy

import pytest

from test_joint_plan import config


def test_complete_joint_graph_preserves_probe_costs_and_single_extension_barrier():
    from treeneuron.research.config import expand
    from treeneuron.research.joint_graph import compile_joint_graph
    study=expand(config()).to_dict();plan=compile_joint_graph(study)
    assert Counter(j['kind'] for j in plan['attempts'])==dict(data=16,parent=32,score=32,
        placement=128,local_model=128,probe=2080,short_selection=32,fit=320,evaluate=320,extension_decision=1)
    jobs={j['job_id']:j for j in plan['attempts']}
    assert len(jobs)==3089
    for j in plan['attempts']:
        p=j['payload']
        if j['kind']=='data':
            assert ('extension_decision:M3' in j['depends_on'])==(p['teacher_index']>=8)
        if j['kind']=='short_selection':
            deps=[jobs[d] for d in j['depends_on']]
            assert len([d for d in deps if d['kind']=='probe'])==65
            assert j['dependency_policy']=='all_terminal'
        if j['kind']=='fit' and p['role']=='short_singleton':
            assert any(jobs[d]['kind']=='short_selection' for d in j['depends_on'])
        if j['kind']=='placement':
            assert all(jobs[d]['kind'] not in ('probe','evaluate','short_selection') for d in j['depends_on'])
    barrier=jobs['extension_decision:M3']
    assert len(barrier['depends_on'])==160
    assert all(jobs[d]['kind']=='evaluate' and jobs[d]['payload']['teacher_index']<8 for d in barrier['depends_on'])
    assert plan['identity']['source_sha256']==study['source_sha256']
    assert plan['identity']['extension_contract']==study['extension_contract']


def test_joint_graph_cannot_be_compiled_from_changed_or_foreign_source_plan():
    from treeneuron.config import canonical_hash
    from treeneuron.research.config import expand
    from treeneuron.research.joint_graph import compile_joint_graph
    original=expand(config()).to_dict()
    for change in ('stale_source','changed_treatment'):
        p=deepcopy(original)
        if change=='stale_source':p['source_sha256']='0'*64
        else:p['attempts'][0]['stop_by_patience']=True
        p['plan_hash']=canonical_hash({k:v for k,v in p.items() if k!='plan_hash'})
        with pytest.raises(ValueError):compile_joint_graph(p)
