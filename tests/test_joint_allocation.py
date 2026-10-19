"""Exact-F selection connects to unchanged native insertion without label leaks."""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from test_train import training_case, cpu_training_evidence
from test_checkpoint import assert_nested_equal


def api():
    from treeneuron.research import joint_allocation
    return joint_allocation


def fixture(training_case):
    from treeneuron import train
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.diagnostics import score_parent
    data,config,paths=training_case;view=fitting_view(data)
    parent=train.fit_parent(view,config,paths)
    return parent,view,score_parent(parent,view.F)


def test_candidates_are_immutable_original_contrasts_and_common_legacy_eligibility(training_case):
    from treeneuron.growth.allocate import allocate
    parent,data,scored=fixture(training_case)
    candidates=api().build_gate_blocks(parent,data.F,scored)
    eligible=set(allocate(scored.table,parent.model.layer.h,policy='adam',seed=0).ids.tolist())
    assert tuple(c.unit_id for c in candidates)==tuple(range(parent.model.layer.h))
    for c in candidates:
        np.testing.assert_array_equal(c.contrast,scored.table.contrasts[c.unit_id].numpy())
        assert c.eligible==(c.unit_id in eligible) and c.added_parameter_count==6
        assert (c.ineligibility_reason is None)==c.eligible
        with pytest.raises(ValueError):c.contrast[0]=100
        with pytest.raises(ValueError):c.contrast.flags.writeable=True


def test_trace_records_full_F_geometry_all_conditional_scores_and_native_insertion(training_case):
    from treeneuron import train
    from treeneuron.research.joint_selector import JointSelectionConfig
    from treeneuron.growth.insert import insert_growth
    parent,data,scored=fixture(training_case);module=api()
    original=train.pack_state(parent);rng=train.capture_rng();pred=parent.predict(data.G.X)
    old_parameters=dict(parent.model.named_parameters())
    old_moments={name:copy.deepcopy(parent.optimizer.state[p]) for name,p in old_parameters.items()}
    candidates=module.build_gate_blocks(parent,data.F,scored)
    trace=module.joint_ridge_select(parent,data.F,candidates,JointSelectionConfig(count=2),scored=scored)
    report=trace.to_dict()
    assert report['schema']=='research/v2-joint-selection'
    assert report['parent_sha256']==scored.parent_sha256 and report['table_sha256']==scored.table_sha256
    assert report['identities']==parent.identity_hashes
    assert report['requested_count']==2 and report['effective_count']==min(2,sum(c.eligible for c in candidates))
    assert len(report['parameter_columns'])==4 and report['parent_rank']<=len(data.F.X)
    assert report['local_update_applied'] is False and report['fallback_history']==[]
    assert report['precision']==dict(scoring='float64',native_parent='float32',insertion_contrast='float32')
    assert report['cost']['elapsed_ns']>0 and report['cost']['jacobian_bytes']>0
    assert set(report['F_hashes'])=={'X','y','row_id','group_id'}
    remaining={c.unit_id for c in candidates if c.eligible}
    for slot in report['slots']:
        assert {c['unit_id'] for c in slot['candidate_gains']}==remaining
        assert {c['unit_id'] for c in slot['numerical_diagnostics']}==remaining
        for c in slot['numerical_diagnostics']:
            assert c['relative_solve_residual']<1e-10
        remaining.remove(slot['unit_id'])
    assert_nested_equal(train.pack_state(parent),original)
    assert_nested_equal(train.capture_rng(),rng)
    allocation=module.selection_to_allocation(parent,data.F,scored,trace,seed=41)
    assert allocation.policy=='joint_ridge'
    assert allocation.ids.tolist()==report['selected_ids']
    for row,unit in enumerate(allocation.ids.tolist()):
        assert torch.equal(allocation.contrasts[row],scored.table.contrasts[unit])
    insert_growth(parent,allocation)
    np.testing.assert_array_equal(parent.predict(data.G.X),pred)
    # Actual old moments remain registered; new gates/contrasts begin fresh.
    for name,p in old_parameters.items():
        assert dict(parent.model.named_parameters())[name] is p
        assert_nested_equal(parent.optimizer.state[p],old_moments[name])
    for name,p in parent.model.named_parameters():
        if name not in old_parameters:assert parent.optimizer.state[p]['step']==0


@pytest.mark.parametrize('change',['G_instead_F','row_order','oracle_targets','parent','table'])
def test_wrong_roles_and_stale_parent_or_table_cannot_be_selected(training_case,change):
    parent,data,scored=fixture(training_case);F=data.F
    if change=='G_instead_F':F=data.G
    elif change=='row_order':F=replace(F,X=F.X[::-1].copy(),y=F.y[::-1].copy(),row_id=F.row_id[::-1].copy(),group_id=F.group_id[::-1].copy())
    elif change=='oracle_targets':F=replace(F,noise_free_target=F.y.copy())
    elif change=='parent':
        with torch.no_grad():parent.model.head.bias.add_(.125)
    else:scored.table.contrasts[0,0]+=.01
    with pytest.raises(ValueError):api().build_gate_blocks(parent,F,scored)


def test_adapter_rejects_changed_directions_and_preserves_fixed_count_when_all_scores_zero(training_case):
    from treeneuron.research.joint_selector import JointSelectionConfig
    from treeneuron.research.diagnostics import ScoredParent
    parent,data,scored=fixture(training_case);module=api()
    candidates=module.build_gate_blocks(parent,data.F,scored)
    changed=list(candidates);changed[0]=replace(changed[0],contrast=changed[0].contrast.copy()+.01)
    with pytest.raises(ValueError,match='candidate|contrast'):
        module.joint_ridge_select(parent,data.F,tuple(changed),JointSelectionConfig(count=2),scored=scored)
    # Zero local blocks among admitted candidates retain common count and stable IDs.
    # A direct numerical zero-block acceptance check supplements real-F selection.
    from treeneuron.research.joint_selector import prepare_joint_problem
    problem=prepare_joint_problem(np.ones((5,1)),np.ones(5),{0:np.zeros((5,2)),1:np.zeros((5,2))},JointSelectionConfig(count=2))
    trace=problem.greedy([0,1]);assert trace['selected_ids']==[0,1] and trace['effective_count']==2
    scored.table.adam_scores.zero_();scored.table.sgd_scores.zero_()
    scored=ScoredParent(scored.table,scored.parent_sha256,scored.table.table_hash,scored.elapsed_ns)
    candidates=module.build_gate_blocks(parent,data.F,scored)
    empty=module.joint_ridge_select(parent,data.F,candidates,JointSelectionConfig(count=2),scored=scored)
    allocation=module.selection_to_allocation(parent,data.F,scored,empty,seed=41)
    assert allocation.effective_count==0 and allocation.requested_count==2 and allocation.contrasts.shape==(0,3)


def test_known_ineligible_nonfinite_diagnostics_remain_explicit_without_imputation(training_case):
    from treeneuron.research.diagnostics import ScoredParent
    from treeneuron.research.joint_selector import JointSelectionConfig
    parent,data,scored=fixture(training_case);module=api()
    table=scored.table
    table.eligible[0]=False;table.adam_scores[0]=0;table.sgd_scores[0]=0
    table.eigenvalues[0].fill_(float('nan'))
    scored=ScoredParent(table,scored.parent_sha256,table.table_hash,scored.elapsed_ns)
    candidates=module.build_gate_blocks(parent,data.F,scored)
    report=module.joint_ridge_select(parent,data.F,candidates,JointSelectionConfig(count=2),scored=scored).to_dict()
    assert candidates[0].eligible is False
    assert report['score_table_diagnostics']['signed_moment_eigenvalues'][0]==['nan']*3
    assert 0 not in report['selected_ids']


def test_selector_cost_includes_final_identity_recheck(training_case,monkeypatch):
    from treeneuron.research.joint_selector import JointSelectionConfig
    parent,data,scored=fixture(training_case);module=api()
    candidates=module.build_gate_blocks(parent,data.F,scored)
    original=module._bound;clock=[0]
    def checking(*a):
        result=original(*a);clock[0]+=10000;return result
    monkeypatch.setattr(module,'_bound',checking)
    monkeypatch.setattr(module.time,'perf_counter_ns',lambda:clock[0])
    report=module.joint_ridge_select(parent,data.F,candidates,JointSelectionConfig(count=2),scored=scored).to_dict()
    assert report['cost']['elapsed_ns']==clock[0] and clock[0]>=20000


@pytest.mark.parametrize('change',['order','winner','gain'])
def test_allocation_rejects_a_trace_with_inconsistent_conditional_chain(training_case,change):
    from treeneuron.research.joint_selector import JointSelectionConfig
    parent,data,scored=fixture(training_case);module=api()
    candidates=module.build_gate_blocks(parent,data.F,scored)
    trace=module.joint_ridge_select(parent,data.F,candidates,JointSelectionConfig(count=2),scored=scored)
    report=trace.to_dict();assert report['effective_count']>0
    if change=='order':report['selection_order']=[]
    elif change=='winner':report['slots'][0]['unit_id']=-1
    else:report['predicted_gain']=-1.
    modified=replace(trace,payload_json=json.dumps(report))
    with pytest.raises(ValueError,match='trace|conditional'):
        module.selection_to_allocation(parent,data.F,scored,modified,seed=41)
