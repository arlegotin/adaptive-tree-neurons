"""Causal diagnostic forks preserve their exact parent and shared treatments."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from test_train import training_case,cpu_training_evidence
from test_checkpoint import assert_nested_equal


def test_F_only_score_binding_rejects_changed_table_or_parent(training_case,tmp_path):
    from treeneuron import train
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.diagnostics import score_parent,ForkSpec,fit_diagnostic_fork
    data,config,paths=training_case
    view=fitting_view(data)
    parent=train.fit_parent(view,config,paths)
    scored=score_parent(parent,view.F)
    request=ForkSpec('top',2,71,4)
    with torch.no_grad():parent.model.head.bias.add_(1)
    with pytest.raises(ValueError,match='parent'):
        fit_diagnostic_fork(parent,view,tmp_path/'changed-parent',request,scored=scored)
    with torch.no_grad():parent.model.head.bias.sub_(1)
    scored=score_parent(parent,view.F)
    scored.table.contrasts[0,0]+=1
    with pytest.raises(ValueError,match='table'):
        fit_diagnostic_fork(parent,view,tmp_path/'changed-table',request,scored=scored)


def test_matched_new_stream_growth_and_widening_preserve_parent(training_case,tmp_path):
    from treeneuron import train
    from treeneuron.checkpoint import load_checkpoint
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.diagnostics import score_parent,ForkSpec,fit_diagnostic_fork
    data,config,paths=training_case
    view=fitting_view(data)
    parent=train.fit_parent(view,config,paths)
    scored=score_parent(parent,view.F)
    expected=parent.predict(data.G.X)
    before,rng=train.pack_state(parent),train.capture_rng()
    results=[]
    for role in ('top','random','warm_lower','warm_upper'):
        root=tmp_path/role
        result=fit_diagnostic_fork(parent,view,root,ForkSpec(role,2,71,4,new_stream_seed=881),scored=scored)
        results.append(result)
        assert result['status']=='completed_eligible'
        assert result['executed_stage_updates']==4
        assert result['initial_pairing']['allowance']==4
        assert result['resolved_policy']['max_updates']==4
        assert result['resolved_policy']['stop_by_patience'] is True
        initial=load_checkpoint(root/'initial.json',expected_hashes=result['identity_hashes'],device='cpu')
        assert initial.cursor==0 and initial.epoch==0
        assert initial.update==parent.update and initial.scheduler.position==parent.scheduler.position
        np.testing.assert_array_equal(initial.predict(data.G.X),expected)
        assert result['stream_reset']['seed']==881
        assert result['costs']['standalone_total_ns'] is None
        assert_nested_equal(train.pack_state(parent),before)
        assert_nested_equal(train.capture_rng(),rng)
        result['parent_identity_hashes']['code']='changed-report-only'
        assert_nested_equal(train.pack_state(parent),before)
    assert len({r['initial_pairing']['order'] for r in results})==1
    assert len({r['initial_pairing']['sampler'] for r in results})==1
    assert results[0]['effective_count']==results[1]['effective_count']==2
    assert results[2]['total_parameters']<=results[0]['total_parameters']<=results[3]['total_parameters']


def test_unscored_keep_does_not_claim_known_common_eligibility(training_case,tmp_path):
    from treeneuron import train
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.diagnostics import ForkSpec,fit_diagnostic_fork
    data,config,paths=training_case
    view=fitting_view(data);parent=train.fit_parent(view,config,paths)
    result=fit_diagnostic_fork(parent,view,tmp_path/'keep',ForkSpec('keep',2,71,4))
    assert result['common_eligible_ids'] is None and result['effective_count'] is None


def test_actual_boundary_subset_and_fixed_prefixes(training_case,tmp_path):
    from treeneuron import train
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.boundaries import current_boundary
    from treeneuron.research.diagnostics import score_parent,ForkSpec,fit_diagnostic_fork
    data,config,paths=training_case
    view=fitting_view(data);snapshots={}
    train.fit_parent(view,config,paths,snapshot_updates=(3,),
        on_snapshot=lambda label,s:snapshots.setdefault(label,s))
    parent=current_boundary(snapshots['current'],view,requested_update=3)
    scored=score_parent(parent,view.F)
    unit=int(scored.table.ids[scored.table.eligible][0])
    result=fit_diagnostic_fork(parent,view,tmp_path/'subset',
        ForkSpec('subset',1,71,6,unit_ids=(unit,),fixed_budget=True,selection_prefixes=(0,2,6)),scored=scored)
    assert result['selected_ids']==[unit] and result['executed_stage_updates']==6
    assert result['parent_update']==3
    assert result['resolved_policy']['stop_by_patience'] is False
    assert result['resolved_policy']['selection_prefixes']==(0,2,6)
    assert [r['prefix'] for r in result['prefixes']['records']]==[0,2,6]
    assert result['prefixes']['unavailable']==[]


def test_nominal_dead_subset_is_ineligible_without_a_fitted_model(tmp_path,cpu_training_evidence,monkeypatch):
    from treeneuron import train
    from treeneuron.artifacts import create_run
    from treeneuron.config import resolve_experiment
    from treeneuron.research.controls import make_positive_control,initialize_control_state
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.diagnostics import score_parent,ForkSpec,fit_diagnostic_fork
    control=make_positive_control(instance_seed=123,data_seed=456,gamma=2.,sigma=.1)
    view=fitting_view(control.development)
    config=resolve_experiment(dict(study_id='fork-dead',model=dict(d=2,h=8),
        data=dict(n_fit=512,n_growth=256,n_validation=256),
        training=dict(base_updates=4,continuation_updates=4,validation_interval=1,patience=20),
        backend=dict(cpu_threads=1,interop_threads=1)))
    monkeypatch.setattr(train,'BACKEND_REPORT_PATH',cpu_training_evidence)
    parent=initialize_control_state(control.parameters,config,create_run(tmp_path/'parent',config.to_dict()),data=view)
    result=fit_diagnostic_fork(parent,view,tmp_path/'dead',ForkSpec('subset',1,71,4,unit_ids=(7,)),
        scored=score_parent(parent,view.F))
    assert result['status']=='completed_ineligible' and result['nominal_ids']==[7]
    assert result['models']=={} and not (tmp_path/'dead'/'runs').exists()


def test_failed_result_publication_preserves_original_failure_and_rng(training_case,tmp_path,monkeypatch):
    from treeneuron import train
    from treeneuron.research.roles import fitting_view
    from treeneuron.research import diagnostics
    from test_checkpoint import assert_nested_equal
    data,config,paths=training_case
    view=fitting_view(data);parent=train.fit_parent(view,config,paths)
    original_write=diagnostics.write_new_json
    def write(path,value):
        if path.name=='result.json':raise OSError('disk full during report')
        return original_write(path,value)
    monkeypatch.setattr(diagnostics,'write_new_json',write)
    rng=train.capture_rng()
    for error in (FloatingPointError('training failure'),KeyboardInterrupt('user interruption')):
        def fail():raise error
        with pytest.raises(type(error),match=str(error)) as caught:
            diagnostics.fit_diagnostic_fork(parent,view,tmp_path/type(error).__name__,
                diagnostics.ForkSpec('keep',0,71,4),guard=fail)
        assert any('disk full' in note for note in caught.value.__notes__)
        assert_nested_equal(train.capture_rng(),rng)
