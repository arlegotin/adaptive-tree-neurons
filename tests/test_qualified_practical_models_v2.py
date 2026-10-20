"""Complete model metadata never substitutes for genuine native/global gates."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

from test_primary_search_declaration_v2 import study


def api():
    assert importlib.util.find_spec('treeneuron.research.qualified_practical_models_v2'),'Missing complete840 model seal'
    from treeneuron.research import qualified_practical_models_v2
    return qualified_practical_models_v2


@pytest.fixture
def inventory(study,tmp_path):
    plan,_=study;rows=[]
    for job in plan.to_dict()['jobs']:
        root=tmp_path/'unqualified-models'/job['job_id'];snapshot=root/'search/attempt-000/generation/native/worker/snapshot'
        metadata=dict(path=str(snapshot/'manifest.json'),sha256='a'*64)
        binding=dict(attempt_index=0,selected_metadata=metadata,snapshot_root=str(snapshot),snapshot_inventory_sha256='b'*64,generation_sha256='c'*64)
        views={'native_quality':dict(status='retained_model_bytes',deadline_ns=None,V_score=.1,model_binding=binding)}
        for deadline in (600000000000,1800000000000,7200000000000):
            views[str(deadline)]=dict(status='unavailable_at_prefix',deadline_ns=deadline,V_score=None,model_binding=None)
        rows.append(dict(**{k:job[k] for k in ('job_id','dataset','outer_fold','family_id','search_repeat','fit_seed','split_hash','manifest_hash')},
            V_score=.1,winner_attempt_index=0,selected_metadata=metadata,snapshot_root=binding['snapshot_root'],snapshot_inventory_sha256='b'*64,
            search_directory=str(root/'search'),search_inventory_sha256='d'*64,generation_sha256='c'*64,budget_view_models=views,qualified_search_audit_sha256='e'*64))
    primary=[copy.deepcopy(r) for r in rows if r['outer_fold']==(4 if r['dataset']=='bike' else 0)]
    return plan,rows,primary


def test_untrusted840_identity_join_keeps_all_tasks_folds_repeats_and_unavailable_prefixes(inventory):
    module=api();plan,rows,primary=inventory
    out=module.inspect_practical_model_inventory(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],models=rows,primary_models=primary)
    assert out['declared_models']==840 and out['primary_models']==168 and out['secondary_models']==672
    assert out['native_qualification_authenticated'] is out['protected_access_authorized'] is out['T_accessed'] is False


@pytest.mark.parametrize('fault',['missing','duplicate','fold','seed','boolean_repeat','native_pointer','native_score','prefix_score',
    'prefix_deadline','primary_reselected','job_id_object'])
def test_full_model_metadata_refuses_missing_crossed_or_changed_fixed_identity(inventory,fault):
    module=api();plan,rows,primary=inventory
    row=next(r for r in rows if r['outer_fold']!=(4 if r['dataset']=='bike' else 0))
    if fault=='missing':rows.pop()
    elif fault=='duplicate':rows[-1]=copy.deepcopy(rows[0])
    elif fault=='fold':row['outer_fold']=0
    elif fault=='seed':row['fit_seed']+=1
    elif fault=='boolean_repeat':row['search_repeat']=False
    elif fault=='native_pointer':row['budget_view_models']['native_quality']['model_binding']['snapshot_inventory_sha256']='f'*64
    elif fault=='native_score':row['budget_view_models']['native_quality']['V_score']=.2
    elif fault=='prefix_score':row['budget_view_models']['600000000000']['V_score']=0.
    elif fault=='prefix_deadline':row['budget_view_models']['600000000000']['deadline_ns']=True
    elif fault=='job_id_object':row['job_id']={}
    else:
        chosen=next(r for r in rows if r['job_id']==primary[0]['job_id']);chosen['V_score']=.2;chosen['budget_view_models']['native_quality']['V_score']=.2
    with pytest.raises(ValueError):module.inspect_practical_model_inventory(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],models=rows,primary_models=primary)


def test_retained_prefix_can_share_native_snapshot_but_not_another_methods_model(inventory):
    module=api();plan,rows,primary=inventory
    row=next(r for r in rows if r['outer_fold']!=(4 if r['dataset']=='bike' else 0))
    row['budget_view_models']['600000000000']=dict(status='retained_model_bytes',deadline_ns=600000000000,V_score=.1,
        model_binding=copy.deepcopy(row['budget_view_models']['native_quality']['model_binding']))
    assert module.inspect_practical_model_inventory(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],models=rows,primary_models=primary)['declared_models']==840
    other=next(r for r in rows if r['family_id']!=row['family_id'])
    row['budget_view_models']['600000000000']['model_binding']=copy.deepcopy(other['budget_view_models']['native_quality']['model_binding'])
    with pytest.raises(ValueError):module.inspect_practical_model_inventory(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],models=rows,primary_models=primary)


@pytest.mark.parametrize('entry',['seal','load'])
def test_complete_model_public_entry_rejects_false_global_gate_before_models_or_output(tmp_path,monkeypatch,entry):
    module=api()
    def forbidden(**kwargs):raise AssertionError('False global qualification cannot inspect model metadata')
    monkeypatch.setattr(module,'inspect_practical_model_inventory',forbidden)
    common=dict(global_barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='a'*64),
        qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='b'*64))
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        if entry=='seal':module.seal_qualified_practical_models(**common,secondary_search_proofs={},output=tmp_path/'not-created.json')
        else:module.load_qualified_practical_models(dict(path=str(tmp_path/'forbidden-model-seal'),sha256='c'*64),**common)
    assert 'forbidden-model-seal' not in str(error.value) and list(tmp_path.iterdir())==[]


def test_pure_output_guard_preserves_resolved_interpreter_directory_without_creating_files(tmp_path):
    module=api();forbidden=Path(sys.executable).resolve().parent/'atn-uncreated-model-seal-fixture.json'
    assert not forbidden.exists()
    with pytest.raises(ValueError):module._check_model_output_location(forbidden,protected_paths=[])
    allowed=tmp_path/'not-created.json'
    assert module._check_model_output_location(allowed,protected_paths=[])==allowed
    assert not allowed.exists() and not forbidden.exists()
