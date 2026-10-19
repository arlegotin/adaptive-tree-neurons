"""Ridge allocation enters the identical native continuation, with its own provenance."""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from test_train import training_case,cpu_training_evidence
from test_joint_allocation import fixture
from test_checkpoint import assert_nested_equal


@pytest.mark.parametrize('strategy,policy',[('conditional','joint_ridge'),('singleton','local_ridge')])
def test_bound_ridge_fork_preserves_paired_stream_and_fixed_budget(training_case,tmp_path,strategy,policy):
    from treeneuron import train
    from treeneuron.checkpoint import load_checkpoint
    from treeneuron.research.joint_allocation import build_gate_blocks,joint_ridge_select
    from treeneuron.research.joint_selector import JointSelectionConfig
    from treeneuron.research.diagnostics import ForkSpec,fit_diagnostic_fork
    parent,data,scored=fixture(training_case);before=train.pack_state(parent);rng=train.capture_rng()
    trace=joint_ridge_select(parent,data.F,build_gate_blocks(parent,data.F,scored),JointSelectionConfig(count=2),scored=scored,strategy=strategy)
    root=tmp_path/strategy
    report=fit_diagnostic_fork(parent,data,root,ForkSpec('ridge',2,71,6,fixed_budget=True),scored=scored,selection_trace=trace)
    assert report['status']=='completed_eligible' and report['executed_stage_updates']==6
    assert report['resolved_policy']['stop_by_patience'] is False
    assert report['selected_ids']==trace.to_dict()['selected_ids']
    assert report['selection_trace_sha256']==trace.sha256
    assert json.loads((root/'selection.json').read_text())==trace.to_dict()
    initial=load_checkpoint(root/'initial.json',expected_hashes=report['identity_hashes'],device='cpu')
    np.testing.assert_array_equal(initial.predict(data.G.X),parent.predict(data.G.X))
    assert initial.cursor==parent.cursor and initial.epoch==parent.epoch
    assert_nested_equal(initial.order,parent.order)
    assert initial.growth_metadata['allocation']['policy']==policy
    assert_nested_equal(train.pack_state(parent),before);assert_nested_equal(train.capture_rng(),rng)


def test_ridge_trace_cannot_be_reused_with_another_count_or_nonridge_fork(training_case,tmp_path):
    from treeneuron.research.joint_allocation import build_gate_blocks,joint_ridge_select
    from treeneuron.research.joint_selector import JointSelectionConfig
    from treeneuron.research.diagnostics import ForkSpec,fit_diagnostic_fork
    parent,data,scored=fixture(training_case)
    trace=joint_ridge_select(parent,data.F,build_gate_blocks(parent,data.F,scored),JointSelectionConfig(count=2),scored=scored)
    for role,count,selection in [('ridge',1,trace),('random',2,trace),('ridge',2,None)]:
        with pytest.raises(ValueError,match='trace|count|selection'):
            fit_diagnostic_fork(parent,data,tmp_path/f'{role}-{count}',ForkSpec(role,count,71,4),scored=scored,selection_trace=selection)


@pytest.mark.parametrize('change',['winner','gain','inventory'])
def test_singleton_control_rejects_inconsistent_trace(training_case,change):
    from treeneuron.research.joint_allocation import build_gate_blocks,joint_ridge_select,selection_to_allocation
    from treeneuron.research.joint_selector import JointSelectionConfig
    parent,data,scored=fixture(training_case)
    trace=joint_ridge_select(parent,data.F,build_gate_blocks(parent,data.F,scored),JointSelectionConfig(count=2),scored=scored,strategy='singleton')
    value=trace.to_dict()
    if change=='winner':value['selection_order'].reverse()
    elif change=='gain':value['predicted_gain']+=1
    else:value['singleton_gains'].pop()
    with pytest.raises(ValueError,match='Singleton|trace'):
        selection_to_allocation(parent,data.F,scored,replace(trace,payload_json=json.dumps(value)),seed=7)
