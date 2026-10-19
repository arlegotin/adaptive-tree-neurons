"""Public M3 cannot confuse unqualified completion with native resource evidence."""
from copy import deepcopy
from pathlib import Path

import pytest

from test_train import cpu_training_evidence
from test_joint_plan import config


def study(receipt=None):
    from treeneuron.research.config import ResearchConfig,expand
    raw=config().to_dict()
    raw['base_experiment']['execution']['backend_report']=str(receipt or '/private/tmp/atn-intentionally-absent-M3-backend.json')
    return expand(ResearchConfig.from_dict(raw)).to_dict()


def test_missing_runtime_or_admission_stops_before_creating_a_scientific_directory(tmp_path):
    from treeneuron.research.joint_execution import run_joint_qualification,run_joint
    p=study()
    with pytest.raises(ValueError,match='backend|receipt'):
        run_joint_qualification(p,tmp_path/'qualification')
    assert not (tmp_path/'qualification').exists()
    with pytest.raises(ValueError,match='admission'):
        run_joint(p,tmp_path/'scientific')
    assert not (tmp_path/'scientific').exists()


def test_all_ineligible_queue_status_cannot_qualify_native_profiles(tmp_path,cpu_training_evidence):
    from treeneuron.research.joint_graph import compile_joint_graph
    from treeneuron.research.joint_qualification import build_joint_qualification_plan
    from treeneuron.research.joint_execution import collect_joint_profiles,make_joint_admission
    from treeneuron.research.workqueue import run_queue
    target=compile_joint_graph(study(cpu_training_evidence));q=build_joint_qualification_plan(target)
    root=tmp_path/'all-ineligible'
    result=run_queue(q,root,lambda *_:dict(status='completed_ineligible'),check_identity=lambda _:None)
    assert result['status']=='completed'
    with pytest.raises(ValueError,match='native|profile|qualification'):
        collect_joint_profiles(root,target)
    with pytest.raises(ValueError,match='native|profile|qualification'):
        make_joint_admission(target,root)


def test_native_qualification_reads_the_real_FittingData_feature_interface(tmp_path,cpu_training_evidence,monkeypatch):
    import numpy as np
    from treeneuron.research import joint_execution
    from treeneuron.data.types import DevelopmentData,ObservationSet
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.joint_graph import compile_joint_graph
    from treeneuron.research.joint_qualification import build_joint_qualification_plan
    from treeneuron.research.workqueue import run_queue
    def rows(n,start):
        ids=np.arange(start,start+n,dtype=np.int64)
        return ObservationSet(np.zeros((n,16)),np.zeros(n),ids,ids)
    view=fitting_view(DevelopmentData(rows(2306,0),rows(577,10000),rows(577,20000),'shape-interface-fixture'))
    target=compile_joint_graph(study(cpu_training_evidence));q=build_joint_qualification_plan(target)
    root=tmp_path/'no-models'
    def handler(job,*_):
        return dict(status='completed_eligible',artifact_path=str(tmp_path/'unused')) if job['kind']=='data' else dict(status='completed_ineligible')
    run_queue(q,root,handler,check_identity=lambda _:None)
    # Only disk acquisition is replaced; the consumer sees the real role-safe
    # object. There are deliberately no native models to qualify after shape validation.
    monkeypatch.setattr(joint_execution,'load_fitting_data',lambda _:view)
    with pytest.raises(ValueError,match='native|Native'):
        joint_execution.collect_joint_profiles(root,target)


def test_full_warm_ordinary_checkpoint_qualifies_its_actual_layer_type(tmp_path,cpu_training_evidence,monkeypatch):
    import numpy as np
    from treeneuron import train
    from treeneuron.artifacts import create_run
    from treeneuron.config import resolve_experiment
    from treeneuron.data.types import DevelopmentData,ObservationSet
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.diagnostics import ForkSpec,score_parent,fit_diagnostic_fork
    from treeneuron.research.joint_execution import _native_model
    rng=np.random.default_rng(2026101927)
    def rows(n,start):
        X=rng.normal(size=(n,16));ids=np.arange(start,start+n,dtype=np.int64)
        return ObservationSet(X,X[:,0]+.2*X[:,1]+.1*np.sin(X[:,2]),ids,ids)
    view=fitting_view(DevelopmentData(rows(2306,0),rows(577,10000),rows(577,20000),'full-width-qualification-fixture'))
    cfg=resolve_experiment(dict(study_id='full-width-qualification-fixture',model=dict(d=16,h=64),
        training=dict(base_updates=1,continuation_updates=8000,learning_rate=.005,weight_decay=.001,
            batch_size=128,validation_interval=25,patience=20,checkpoint_interval=25),
        backend=dict(cpu_threads=1,interop_threads=1)))
    monkeypatch.setattr(train,'BACKEND_REPORT_PATH',cpu_training_evidence)
    root=tmp_path/'queue';root.mkdir()
    parent=train.fit_parent(view,cfg,create_run(root/'parent',cfg.to_dict()))
    scored=score_parent(parent,view.F)
    report=fit_diagnostic_fork(parent,view,root/'warm',ForkSpec('warm_upper',16,71,8000,fixed_budget=True),scored=scored)
    assert report['ordinary_width']==95
    value=dict(report=report,artifact_path=str(root/'warm'))
    assert _native_model(value,dict(kind='fit',payload=dict(role='warm_upper',max_updates=8000)),view,root)==8000
