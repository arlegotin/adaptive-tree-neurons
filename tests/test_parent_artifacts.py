"""Actual parent states remain distinct from unavailable and G-selected states."""
import json

import pytest

from test_train import training_case,cpu_training_evidence


def test_parent_artifacts_load_actual_and_selected_states_without_rng_motion(training_case,tmp_path,cpu_training_evidence,monkeypatch):
    from treeneuron import train
    from treeneuron.config import resolve_experiment
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.parent_artifacts import ParentSpec,fit_parent_artifact,load_parent_input
    from test_checkpoint import assert_nested_equal
    data,config,_=training_case;view=fitting_view(data)
    raw=config.to_dict();raw['execution']['backend_report']=str(cpu_training_evidence)
    config=resolve_experiment(raw)
    wrong=tmp_path/'wrong-global-receipt.json';monkeypatch.setattr(train,'BACKEND_REPORT_PATH',wrong)
    rng=train.capture_rng();root=tmp_path/'parent-artifact'
    record=fit_parent_artifact(view,config,root,ParentSpec((3,),fixed_budget=True))
    assert record['status']=='completed_eligible' and record['executed_updates']==6
    assert train.BACKEND_REPORT_PATH==wrong
    assert_nested_equal(train.capture_rng(),rng)
    actual=load_parent_input(root,'actual_3',view)
    assert actual.update==3 and actual.stage['kind']=='research_boundary'
    late=load_parent_input(root,'late_G_best',view)
    assert late.update==record['selected_update'] and late.stage['kind']=='base'
    assert_nested_equal(train.capture_rng(),rng)
    pointer=root/'selected.json';pointer.write_text(pointer.read_text()+' ')
    with pytest.raises(ValueError,match='commitment'):
        load_parent_input(root,'late_G_best',view)


def test_unreached_snapshot_is_unavailable_but_corruption_never_is(training_case,tmp_path,cpu_training_evidence):
    from treeneuron.config import resolve_experiment
    from treeneuron.data.types import DevelopmentData
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.parent_artifacts import ParentSpec,fit_parent_artifact,load_parent_input
    data,config,_=training_case;view=fitting_view(data)
    raw=config.to_dict();raw['training'].update(learning_rate=1e-30,patience=1,validation_interval=1)
    raw['execution']['backend_report']=str(cpu_training_evidence)
    config=resolve_experiment(raw);root=tmp_path/'stopped-parent'
    record=fit_parent_artifact(view,config,root,ParentSpec((3,6)))
    assert record['executed_updates']==1 and record['unavailable_snapshots']==[3,6]
    assert load_parent_input(root,'actual_3',view) is None
    changed=fitting_view(DevelopmentData(data.F,data.G,data.V,'changed-teacher'))
    with pytest.raises(ValueError,match='data'):
        load_parent_input(root,'actual_3',changed)
    with pytest.raises(ValueError,match='planned'):
        load_parent_input(root,'actual_4',view)
    capture=root/'snapshots'/'capture.json';capture.write_text('{}')
    with pytest.raises(ValueError,match='commitment'):
        load_parent_input(root,'actual_3',view)


def test_planted_prerequisite_is_unfitted_and_restores_backend_on_error(tmp_path,cpu_training_evidence,monkeypatch):
    from treeneuron import train
    from treeneuron.config import resolve_experiment
    from treeneuron.research.controls import make_positive_control
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.parent_artifacts import ParentSpec,fit_parent_artifact,load_parent_input
    control=make_positive_control(instance_seed=123,data_seed=456,gamma=2.,sigma=.1)
    view=fitting_view(control.development)
    config=resolve_experiment(dict(study_id='unfitted-parent',model=dict(d=2,h=8),
        data=dict(n_fit=512,n_growth=256,n_validation=256),
        training=dict(base_updates=6,continuation_updates=6,validation_interval=2,patience=20),
        backend=dict(cpu_threads=1,interop_threads=1),
        execution=dict(backend_report=str(cpu_training_evidence))))
    wrong=tmp_path/'caller-receipt.json';monkeypatch.setattr(train,'BACKEND_REPORT_PATH',wrong)
    def forbidden(*args,**kwargs):raise AssertionError('Planted prerequisite must not fit parent')
    monkeypatch.setattr(train,'fit_parent',forbidden)
    root=tmp_path/'unfitted'
    record=fit_parent_artifact(view,config,root,ParentSpec(),control=control.parameters)
    assert record['selected_update']==record['executed_updates']==0
    assert load_parent_input(root,'late_G_best',view).update==0
    assert train.BACKEND_REPORT_PATH==wrong
    def interrupt():raise KeyboardInterrupt('synthetic interruption')
    with pytest.raises(KeyboardInterrupt):
        fit_parent_artifact(view,config,tmp_path/'interrupted',ParentSpec(),control=control.parameters,guard=interrupt)
    assert train.BACKEND_REPORT_PATH==wrong
