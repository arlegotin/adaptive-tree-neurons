"""Temporary solved updates diagnose a local model; never alter placement."""
from dataclasses import replace

import numpy as np
import pytest

from test_train import training_case,cpu_training_evidence
from test_checkpoint import assert_nested_equal
from test_joint_allocation import fixture


@pytest.mark.parametrize('parent_ridge,condition',[(.01,True),(0.,True),(.01,False)])
def test_local_update_diagnostics_match_objective_and_restore_complete_parent(training_case,parent_ridge,condition):
    from treeneuron import train
    from treeneuron.research.joint_allocation import build_gate_blocks,joint_ridge_select
    from treeneuron.research.joint_selector import JointSelectionConfig
    from treeneuron.research.local_updates import diagnose_local_updates
    parent,data,scored=fixture(training_case)
    cfg=JointSelectionConfig(count=2,parent_ridge=parent_ridge,condition_on_parent=condition)
    trace=joint_ridge_select(parent,data.F,build_gate_blocks(parent,data.F,scored),cfg,scored=scored)
    before=train.pack_state(parent);rng=train.capture_rng()
    result=diagnose_local_updates(parent,data.F,data.G,scored,trace)
    report=trace.to_dict()
    np.testing.assert_allclose(result['full_step_penalized_objective'],report['selected_objective'],rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(result['optimized_parent_objective']-result['full_step_penalized_objective'],report['predicted_gain'],rtol=1e-8,atol=1e-10)
    assert result['fractions']==[1/16,1/4,1] and result['fraction_selected_on_G'] is False
    assert result['local_updates_applied_to_parent'] is False and result['trace_sha256']==trace.sha256
    assert result['loss_units']=='Existing F-fitted target coordinates; no residual rescaling'
    assert result['precision']==dict(parent_source='float32',temporary_model='float64',gate_contrasts='float64')
    for row in result['evaluations']:
        for role in ('F','G'):
            assert row[role]['linearized_MSE']>=0 and row[role]['nonlinear_MSE']>=0
            assert 0<=row[role]['ReLU_mask_changed_fraction']<=1
            assert 0<=row[role]['gate_saturation_fraction']<=1
    np.testing.assert_allclose(result['evaluations'][-1]['predicted_penalized_F_objective'],result['full_step_penalized_objective'],rtol=1e-8,atol=1e-10)
    assert result['linear_solve']['retained_rank']==result['linear_solve']['columns']
    assert result['linear_solve']['singular_cutoff']>=0
    assert result['solution']['parent_parameter_columns']==report['parameter_columns']
    assert len(result['solution']['parent_delta'])==sum(p.numel() for p in parent.model.parameters() if p.requires_grad)
    assert_nested_equal(train.pack_state(parent),before)
    assert_nested_equal(train.capture_rng(),rng)


def test_diagnostic_cannot_fit_or_substitute_G_oracle_targets(training_case):
    from treeneuron.research.joint_allocation import build_gate_blocks,joint_ridge_select
    from treeneuron.research.joint_selector import JointSelectionConfig
    from treeneuron.research.local_updates import diagnose_local_updates
    parent,data,scored=fixture(training_case)
    trace=joint_ridge_select(parent,data.F,build_gate_blocks(parent,data.F,scored),JointSelectionConfig(count=2),scored=scored)
    for wrong in (data.F,replace(data.G,noise_free_target=data.G.y.copy()),replace(data.G,y=data.G.y+.1)):
        with pytest.raises(ValueError):diagnose_local_updates(parent,data.F,wrong,scored,trace)


def test_complete_local_diagnostic_can_be_published_by_the_canonical_artifact_writer(training_case,tmp_path):
    """Numpy scalar cutoffs and integer gate keys must not break evidence publication."""
    import json
    from treeneuron.research.joint_allocation import build_gate_blocks,joint_ridge_select
    from treeneuron.research.joint_selector import JointSelectionConfig
    from treeneuron.research.local_updates import diagnose_local_updates
    from treeneuron.research.ledger import write_new_json
    parent,data,scored=fixture(training_case)
    trace=joint_ridge_select(parent,data.F,build_gate_blocks(parent,data.F,scored),JointSelectionConfig(count=2),scored=scored)
    result=diagnose_local_updates(parent,data.F,data.G,scored,trace)
    path=tmp_path/'diagnostic.json'
    write_new_json(path,result)
    restored=json.loads(path.read_text())
    assert restored==result
    assert set(restored['solution']['gate_delta'])=={str(i) for i in trace.to_dict()['selected_ids']}
