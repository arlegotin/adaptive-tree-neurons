"""Frozen prediction identities alone never authorize protected labels."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

from test_primary_search_declaration_v2 import study
from test_qualified_practical_models_v2 import inventory
from test_released_transfer_v2 import toy


def api():
    assert importlib.util.find_spec('treeneuron.research.practical_T_protocol_v2'), 'Missing final practical prediction protocol'
    from treeneuron.research import practical_T_protocol_v2
    return practical_T_protocol_v2


@pytest.fixture
def prediction_inputs(inventory,tmp_path):
    plan,models,primary=inventory;features={}
    for row in models:
        dataset=row['dataset'];fold=row['outer_fold'];key=f'{dataset}/outer-{fold}'
        features[key]=dict(dataset=dataset,outer_fold=fold,manifest_hash=row['manifest_hash'],split_hash=row['split_hash'],
            feature_reference=dict(path=str(tmp_path/'unqualified-data'/dataset/f'outer-{fold}/T-features.npz'),sha256='f'*64))
    return plan,models,primary,features


def test_metadata_plan_keeps_all840_models_and_four_views_without_opening_arrays(prediction_inputs):
    plan,models,primary,features=prediction_inputs
    out=api()._prediction_manifest(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],
        models=models,primary_models=primary,feature_bindings=features)
    assert out['prediction_count']==840 and out['planned_view_count']==3360 and out['unavailable_views']==2520
    assert len(out['queries'])==840 and len(out['view_bindings'])==3360
    assert len({q['query_id'] for q in out['queries']})==840
    assert {q['query_role'] for q in out['queries']}=={'T_features'}
    assert out['native_qualification_authenticated'] is out['protected_access_authorized'] is out['T_accessed'] is False


def test_same_physical_native_snapshot_reused_at_prefix_is_predicted_once(prediction_inputs):
    plan,models,primary,features=prediction_inputs
    row=next(r for r in models if r['outer_fold']!=(4 if r['dataset']=='bike' else 0))
    row['budget_view_models']['600000000000']=dict(status='retained_model_bytes',deadline_ns=600000000000,V_score=.1,
        model_binding=copy.deepcopy(row['budget_view_models']['native_quality']['model_binding']))
    out=api()._prediction_manifest(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],
        models=models,primary_models=primary,feature_bindings=features)
    assert out['prediction_count']==840 and out['unavailable_views']==2519
    bindings=[b for b in out['view_bindings'] if b['job_id']==row['job_id'] and b['query_id'] is not None]
    assert len(bindings)==2 and bindings[0]['query_id']==bindings[1]['query_id']


def test_identical_complete_snapshot_content_in_copied_root_can_share_prediction(prediction_inputs):
    plan,models,primary,features=prediction_inputs
    row=next(r for r in models if r['outer_fold']!=(4 if r['dataset']=='bike' else 0))
    binding=copy.deepcopy(row['budget_view_models']['native_quality']['model_binding'])
    root=Path(row['search_directory'])/'copied-prefix-snapshot';binding['snapshot_root']=str(root)
    binding['selected_metadata']['path']=str(root/'manifest.json')
    row['budget_view_models']['600000000000']=dict(status='retained_model_bytes',deadline_ns=600000000000,V_score=.1,model_binding=binding)
    out=api()._prediction_manifest(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],
        models=models,primary_models=primary,feature_bindings=features)
    assert out['prediction_count']==840
    binding['snapshot_inventory_sha256']='0'*64
    changed=api()._prediction_manifest(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],
        models=models,primary_models=primary,feature_bindings=features)
    assert changed['prediction_count']==841


@pytest.mark.parametrize('fault',['missing_feature','cross_split','cross_manifest','label_field','label_archive','missing_model','changed_primary'])
def test_prediction_plan_cannot_cross_roles_splits_or_frozen_models(prediction_inputs,fault):
    plan,models,primary,features=prediction_inputs
    key=f"{models[0]['dataset']}/outer-{models[0]['outer_fold']}"
    if fault=='missing_feature':features.pop(key)
    elif fault=='cross_split':features[key]['split_hash']='0'*64
    elif fault=='cross_manifest':features[key]['manifest_hash']='0'*64
    elif fault=='label_field':features[key]['T_labels_path']='/forbidden'
    elif fault=='label_archive':features[key]['feature_reference']['path']=features[key]['feature_reference']['path'].replace('T-features','T-labels')
    elif fault=='missing_model':models.pop()
    else:primary[0]['fit_seed']+=1
    with pytest.raises(ValueError):api()._prediction_manifest(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],
        models=models,primary_models=primary,feature_bindings=features)


@pytest.mark.parametrize('entry',['seal','load'])
def test_final_protocol_public_entries_load_actual_models_before_cost_readiness_or_data(tmp_path,entry):
    common=dict(models_reference=dict(path=str(tmp_path/'forbidden-models'),sha256='a'*64),
        global_barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='b'*64),
        qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='c'*64),
        preT_costs_reference=dict(path=str(tmp_path/'forbidden-cost'),sha256='d'*64),
        readiness_reference=dict(path=str(tmp_path/'forbidden-readiness'),sha256='e'*64))
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        if entry=='seal':api().seal_practical_T_protocol(**common,output=tmp_path/'protocol.json')
        else:api().load_practical_T_protocol(dict(path=str(tmp_path/'forbidden-protocol'),sha256='f'*64),**common)
    assert all(s not in str(error.value) for s in ('forbidden-models','forbidden-cost','forbidden-readiness','forbidden-protocol'))
    assert list(tmp_path.iterdir())==[]


def test_protocol_output_preserves_model_searches_and_allows_fresh_protocol_namespace(tmp_path):
    module=api();search=tmp_path/'immutable-search';search.mkdir()
    for output in (search/'new-protocol.json',Path(sys.executable).resolve().parent/'new-protocol.json'):
        with pytest.raises(ValueError):module._check_protocol_output(output,protected_paths=[str(search)])
        assert not output.exists()
    allowed=tmp_path/'protocol/research-v2/frozen-practical.json'
    assert module._check_protocol_output(allowed,protected_paths=[str(search)])==allowed
    assert not allowed.exists()


def test_unavailable_readiness_dependency_is_actionable_and_never_a_stub_capability(tmp_path,monkeypatch):
    module=api();original=module.importlib.import_module
    def unavailable(name,*args,**kwargs):
        if name=='treeneuron.research.production_readiness_v2':raise ModuleNotFoundError(name=name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(module.importlib,'import_module',unavailable)
    with pytest.raises(ValueError,match='readiness'):
        module._load_readiness(dict(path=str(tmp_path/'not-read'),sha256='a'*64),dict(root=str(tmp_path/'not-read-q'),inventory_sha256='b'*64))
    assert list(tmp_path.iterdir())==[]


def test_toy_role_binding_reads_manifest_metadata_without_any_archive_decode(toy,monkeypatch):
    module=api();root,digest,legacy=toy
    import numpy as np
    def forbidden(*args,**kwargs):raise AssertionError('Protocol assembly cannot decode a role archive')
    monkeypatch.setattr(np,'load',forbidden)
    original=Path.read_bytes
    def manifest_only(path):
        assert path.name=='manifest.json','Role archive bytes belong to the actual prior data gate, not this metadata compiler'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',manifest_only)
    data,features=module._data_bindings({'energy':dict(root=str(root),manifest_file_sha256=digest)})
    assert len(data['energy']['folds'])==5 and set(features)=={f'energy/outer-{i}' for i in range(5)}
    assert data['energy']['manifest_hash']==legacy.manifest['manifest_hash']
    assert all(f['roles']['T_features']['reference']['path'].endswith('T-features.npz') for f in data['energy']['folds'])
    assert all(f['roles']['T_labels']['reference']['path'].endswith('T-labels.npz') for f in data['energy']['folds'])
