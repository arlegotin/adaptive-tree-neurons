"""Retained arrays have a fitting boundary separate from evaluator labels."""
import hashlib
import json

import numpy as np
import pytest

from test_train import cpu_training_evidence


def primary_request():
    from treeneuron.config import resolve_experiment
    base=resolve_experiment(dict(study_id='data-artifact',model=dict(d=2,h=4),
        data=dict(n_fit=13,n_growth=7,n_validation=5))).to_dict()
    spec=dict(population='primary',teacher_index=0,teacher_seed=123,data_seed=456,
        reference_seed=789,evaluation_seed=1011,evaluation_rows=19)
    return base,spec


def test_primary_arrays_roundtrip_without_fitting_reading_evaluator_labels(tmp_path,monkeypatch):
    from treeneuron.config import resolve_experiment
    from treeneuron.data.synthetic import make_from_config,sample_teacher
    from treeneuron.research.opportunity_data import prepare_data_artifact,load_fitting_data,load_evaluation_data
    base,spec=primary_request();root=tmp_path/'data'
    record=prepare_data_artifact(base,spec,root)
    raw=base.copy();raw['seeds']={**raw['seeds'],**{k:spec[k+'_seed'] for k in ('teacher','data','reference')}}
    config=resolve_experiment(raw);teacher,data=make_from_config(config)
    expected_E=sample_teacher(teacher,19,spec['evaluation_seed'],config.data.noise_sigma)
    forbidden={root/'evaluator.npz',root/'teacher.json'}
    original=type(root).open
    def guarded(path,*args,**kwargs):
        if path in forbidden:raise AssertionError('A fitting loader read evaluator-only material')
        return original(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(type(root),'open',guarded)
        fit=load_fitting_data(root)
        with pytest.raises(PermissionError):_ = fit.V
        assert fit.F.noise_free_target is fit.G.noise_free_target is None
        assert fit.manifest==data.manifest
        np.testing.assert_array_equal(fit.validation_features,data.V.X)
        for role in 'FG':
            for key in ('X','y','row_id','group_id'):
                np.testing.assert_array_equal(getattr(getattr(fit,role),key),getattr(getattr(data,role),key))
    restored,E=load_evaluation_data(root)
    assert record['status']=='completed_eligible' and record['T_created'] is False
    for role in 'FGV':
        for key in ('X','y','row_id','group_id','noise_free_target'):
            np.testing.assert_array_equal(getattr(getattr(restored,role),key),getattr(getattr(data,role),key))
    for key in ('X','y','row_id','group_id','noise_free_target'):
        np.testing.assert_array_equal(getattr(E,key),getattr(expected_E,key))


def test_positive_coefficients_are_retained_before_observations(tmp_path,monkeypatch):
    from treeneuron.research import controls
    from treeneuron.research.opportunity_data import prepare_data_artifact,load_fitting_data,load_control_parameters
    base,spec=primary_request();spec.update(population='positive_small',evaluation_rows=16384)
    root=tmp_path/'data';original=controls.make_positive_control
    def checked(**kwargs):
        declared=json.loads((root/'parent-coefficients.json').read_text())
        assert not (root/'fitting.npz').exists()
        result=original(**kwargs)
        np.testing.assert_array_equal(declared['base_rows'],result.base_rows)
        np.testing.assert_array_equal(declared['head'],result.head)
        assert declared['generating_ids']==[0,2] and declared['gamma']==2.
        return result
    monkeypatch.setattr(controls,'make_positive_control',checked)
    prepare_data_artifact(base,spec,root)
    fit=load_fitting_data(root);parameters=load_control_parameters(root)
    assert len(fit.F.y)==512 and fit.F.X.shape[1]==2
    assert parameters.gamma==2. and parameters.generating_ids==(0,2)
    np.testing.assert_array_equal(parameters.base_rows[0],parameters.base_rows[1])
    with pytest.raises(ValueError,match='planted'):
        primary=tmp_path/'primary';prepare_data_artifact(*primary_request(),primary)
        load_control_parameters(primary)


def test_corrupt_and_extra_label_arrays_are_rejected(tmp_path):
    from treeneuron.research.opportunity_data import prepare_data_artifact,load_fitting_data
    base,spec=primary_request();root=tmp_path/'data';prepare_data_artifact(base,spec,root)
    path=root/'fitting.npz'
    with np.load(path,allow_pickle=False) as saved:values={k:saved[k].copy() for k in saved.files}
    values['V_y']=np.ones(5)
    with path.open('wb') as stream:np.savez_compressed(stream,**values)
    metadata=json.loads((root/'result.json').read_text())
    metadata['files']['fitting.npz']=hashlib.sha256(path.read_bytes()).hexdigest()
    (root/'result.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='inventory'):
        load_fitting_data(root)


def test_unknown_or_protected_request_fails_before_output_creation(tmp_path):
    from treeneuron.research.opportunity_data import prepare_data_artifact
    base,spec=primary_request();spec['T_seed']=99
    with pytest.raises(ValueError,match='request'):
        prepare_data_artifact(base,spec,tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()


@pytest.mark.parametrize('corruption',['dataset_identity','config_seed','protected_request','V_shape'])
def test_republished_inconsistent_structured_data_is_rejected_while_loading_fit(tmp_path,corruption):
    from treeneuron.config import canonical_hash
    from treeneuron.data.types import array_hash,SplitManifest
    from treeneuron.research.opportunity_data import prepare_data_artifact,load_fitting_data
    base,spec=primary_request();root=tmp_path/'data';prepare_data_artifact(base,spec,root)
    metadata=json.loads((root/'result.json').read_text())
    def rewrite(name,value):
        path=root/name;path.write_text(json.dumps(value))
        metadata['files'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    request=json.loads((root/'request.json').read_text())
    if corruption=='config_seed':
        request['resolved_data_config']['seeds']['teacher']+=1;rewrite('request.json',request)
    elif corruption=='protected_request':
        request['spec']['T_seed']=999;metadata['request']=request['spec'];rewrite('request.json',request)
    else:
        split=json.loads((root/'split.json').read_text())
        if corruption=='dataset_identity':split['dataset_id']='0'*64
        else:
            path=root/'validation-features.npz'
            with np.load(path,allow_pickle=False) as old:arrays={k:old[k].copy() for k in old.files}
            arrays['V_X']=arrays['V_X'][:,None,:]
            with path.open('wb') as stream:np.savez_compressed(stream,**arrays)
            metadata['files'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            header=metadata['headers']['V']['arrays']['X']
            header.update(shape=list(arrays['V_X'].shape),sha256=array_hash(arrays['V_X']))
            split['hashes']=[[k,array_hash(arrays['V_X']) if k=='V.X' else v] for k,v in split['hashes']]
        manifest=SplitManifest(**{k:v for k,v in split.items() if k not in ('assignment_hash','feature_schema_hash')})
        split=manifest.to_dict();rewrite('split.json',split)
        metadata['manifest_sha256']=canonical_hash(split)
    (root/'result.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):load_fitting_data(root)


def test_retained_data_drives_native_parent_fork_and_separate_evaluator(tmp_path,cpu_training_evidence):
    from treeneuron.config import resolve_experiment
    from treeneuron.research.opportunity_data import (prepare_data_artifact,load_fitting_data,
        load_control_parameters,load_evaluation_data)
    from treeneuron.research.parent_artifacts import ParentSpec,fit_parent_artifact,load_parent_input
    from treeneuron.research.diagnostics import ForkSpec,fit_diagnostic_fork
    from treeneuron.research.runner import _metrics_and_predictions
    base,spec=primary_request();spec.update(population='positive_small',evaluation_rows=16384)
    base['training'].update(base_updates=6,continuation_updates=6,validation_interval=2,patience=20)
    base['execution']['backend_report']=str(cpu_training_evidence)
    prepare_data_artifact(base,spec,tmp_path/'data')
    config=resolve_experiment(json.loads((tmp_path/'data'/'request.json').read_text())['resolved_data_config'])
    view=load_fitting_data(tmp_path/'data')
    fit_parent_artifact(view,config,tmp_path/'parent',ParentSpec(),
        control=load_control_parameters(tmp_path/'data'))
    parent=load_parent_input(tmp_path/'parent','late_G_best',view)
    assert parent.update==0
    fork=fit_diagnostic_fork(parent,view,tmp_path/'fork',ForkSpec('keep',0,123,6))
    data,E=load_evaluation_data(tmp_path/'data')
    metrics=_metrics_and_predictions(tmp_path/'fork',fork,data,E)
    assert metrics['selected']['E']['mse']>=0 and metrics['selected']['V']['mse']>=0
    with np.load(tmp_path/'fork'/'predictions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['E_row_id'],E.row_id)
        np.testing.assert_array_equal(saved['V_y'],data.V.y)
