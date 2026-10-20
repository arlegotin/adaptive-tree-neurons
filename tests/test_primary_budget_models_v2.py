"""Budget model byte joins, with no native/model-generation authority."""
import copy
import hashlib
import importlib.util
import json

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.primary_budget_models_v2'),'Missing primary budget model binding'
    from treeneuron.research import primary_budget_models_v2
    return primary_budget_models_v2


def fixture(root,*,late=False):
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import _inventory
    from treeneuron.analysis.practical_v2 import select_completed_trial
    records=[];models={};previous=0
    for i,(end,score) in enumerate([(10800_000_000_000,1.)] if late else [(300_000_000_000,3.),(1200_000_000_000,2.),(7800_000_000_000,1.)]):
        directory=root/f'model-{i}';directory.mkdir();path=directory/'manifest.json'
        path.write_text(json.dumps(dict(scope='Untrusted software model bytes',i=i)));(directory/'arrays.bin').write_bytes(bytes([i+1])*32)
        models[i]=dict(selected_metadata=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            snapshot_root=str(directory),snapshot_inventory_sha256=canonical_hash(_inventory(directory)),generation_sha256='a'*64)
        records.append(dict(family_id='P13',search_repeat=0,attempt_id=f'attempt-{i}',attempt_index=i,
            status='completed_eligible',completion_ns=end,charged_ns=end-previous,V_score=score));previous=end
    result=dict(records=records,total_charged_ns=previous,winner=select_completed_trial(records,deadline_ns=previous)['winner'],
        prefixes={str(t):select_completed_trial(records,deadline_ns=t) for t in (600_000_000_000,1800_000_000_000,7200_000_000_000)})
    return result,models


def test_every_budget_view_keeps_its_actual_completed_model_not_the_later_better_model(tmp_path):
    result,models=fixture(tmp_path);value=api().inspect_budget_model_views(search_result=result,candidate_models=models)
    views=value['views']
    assert views['native_quality']['model_binding']['attempt_index']==2
    assert views['600000000000']['model_binding']['attempt_index']==0
    assert views['1800000000000']['model_binding']['attempt_index']==1
    assert views['7200000000000']['model_binding']['attempt_index']==1
    assert views['1800000000000']['model_binding']==views['7200000000000']['model_binding']
    assert value['native_model_generation_authenticated'] is value['protected_access_authorized'] is False


def test_native_anchor_completing_after_prefix_is_unavailable_not_a_predictive_loss(tmp_path):
    result,models=fixture(tmp_path,late=True);value=api().inspect_budget_model_views(search_result=result,candidate_models=models)
    for name in ('600000000000','1800000000000','7200000000000'):
        assert value['views'][name]['status']=='unavailable_at_prefix'
        assert value['views'][name]['model_binding'] is None
    assert value['views']['native_quality']['model_binding']['attempt_index']==0


@pytest.mark.parametrize('fault',['missing_model','changed_tensor','log_as_model','wrong_inventory','changed_prefix','extra_model','boolean_index'])
def test_changed_incomplete_or_wrong_model_view_cannot_be_bound(tmp_path,fault):
    result,models=fixture(tmp_path)
    if fault=='missing_model':models.pop(0)
    elif fault=='changed_tensor':(tmp_path/'model-0/arrays.bin').write_bytes(b'changed')
    elif fault=='log_as_model':
        path=tmp_path/'model-0/stderr.log';path.write_text('not a checkpoint');models[0]['selected_metadata']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    elif fault=='wrong_inventory':models[0]['snapshot_inventory_sha256']='0'*64
    elif fault=='changed_prefix':result['prefixes']['600000000000']['winner']=copy.deepcopy(result['winner'])
    elif fault=='extra_model':models[3]=copy.deepcopy(models[2])
    else:models[False]=models.pop(0)
    with pytest.raises(ValueError):api().inspect_budget_model_views(search_result=result,candidate_models=models)


def test_native_endpoint_ties_keep_the_earlier_model_and_all_candidates_still_require_evidence(tmp_path):
    from treeneuron.analysis.practical_v2 import select_completed_trial
    result,models=fixture(tmp_path);result['records'][2]['V_score']=2.-5e-13
    result['winner']=select_completed_trial(result['records'],deadline_ns=result['total_charged_ns'])['winner']
    value=api().inspect_budget_model_views(search_result=result,candidate_models=models)
    assert value['views']['native_quality']['model_binding']['attempt_index']==1
    models.pop(2)
    with pytest.raises(ValueError):api().inspect_budget_model_views(search_result=result,candidate_models=models)
