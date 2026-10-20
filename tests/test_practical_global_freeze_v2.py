"""Secondary barrier metadata fixtures never qualify native model histories."""
import copy
import importlib.util

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.practical_global_freeze_v2'),'Missing secondary global primary barrier'
    from treeneuron.research import practical_global_freeze_v2
    return practical_global_freeze_v2


def rows():
    result=[]
    for dataset in ('airfoil','concrete','ccpp','parkinsons','bike','energy'):
        for repeat in (0,1):
            for i in range(14):
                family=f'P{i:02}';score=(.1 if i==1 else 1.)
                if (repeat==0 and i==3) or (repeat==1 and i==4):score=.5
                model=dict(attempt_index=0,selected_metadata=dict(path=f'/untrusted/{dataset}/{repeat}/{family}/manifest.json',sha256='a'*64))
                views={name:dict(status='retained_model_bytes',deadline_ns=None if name=='native_quality' else int(name),V_score=score,model_binding=copy.deepcopy(model)) for name in ('native_quality','600000000000','1800000000000','7200000000000')}
                result.append(dict(job_id=f'{dataset}/outer-{4 if dataset=="bike" else 0}/repeat-{repeat}/{family}',
                    dataset=dataset,outer_fold=4 if dataset=='bike' else 0,search_repeat=repeat,family_id=family,budget_view_models=views))
    return result


def test_champions_keep_both_repeats_all_twelve_external_families_and_each_view():
    value=api().inspect_primary_champion_choices(rows())
    for dataset in ('airfoil','concrete','ccpp','parkinsons','bike','energy'):
        assert value['choices'][dataset]['0']['native_quality']['family_id']=='P03'
        assert value['choices'][dataset]['1']['native_quality']['family_id']=='P04'
        assert value['choices'][dataset]['0']['600000000000']['family_id']=='P03'
    assert value['native_model_generation_authenticated'] is value['protected_access_authorized'] is False


def test_missing_budget_family_keeps_the_portfolio_unavailable_without_test_best_fallback():
    data=rows();row=next(r for r in data if r['dataset']=='airfoil' and r['search_repeat']==0 and r['family_id']=='P12')
    row['budget_view_models']['600000000000'].update(status='unavailable_at_prefix',V_score=None,model_binding=None)
    value=api().inspect_primary_champion_choices(data)['choices']['airfoil']['0']
    assert value['600000000000']['status']=='unavailable_portfolio'
    assert value['600000000000']['missing_families']==['P12']
    assert value['600000000000']['family_id'] is None
    assert value['native_quality']['family_id']=='P03'


def test_champion_numeric_ties_keep_the_lowest_family_id_separately_in_each_repeat():
    data=rows()
    for row in data:
        for view in row['budget_view_models'].values():view['V_score']=1.-(5e-13 if row['family_id']=='P13' else 0.)
    value=api().inspect_primary_champion_choices(data)
    assert value['choices']['airfoil']['0']['native_quality']['family_id']=='P02'
    assert value['choices']['airfoil']['1']['native_quality']['family_id']=='P02'


@pytest.mark.parametrize('fault',['missing','duplicate','secondary','repeat_bool','wrong_family','nan_score','extra_view'])
def test_primary_inventory_cannot_shrink_or_select_another_fold_or_seed(fault):
    data=rows()
    if fault=='missing':data.pop()
    elif fault=='duplicate':data[-1]=copy.deepcopy(data[0])
    elif fault=='secondary':data[0]['outer_fold']=1
    elif fault=='repeat_bool':data[0]['search_repeat']=False
    elif fault=='wrong_family':data[0]['family_id']='P14'
    elif fault=='nan_score':data[2]['budget_view_models']['native_quality']['V_score']=float('nan')
    else:data[0]['budget_view_models']['best_T']={}
    with pytest.raises(ValueError):api().inspect_primary_champion_choices(data)


@pytest.mark.parametrize('operation',['seal','load'])
def test_false_actual_registry_fails_before_protocol_models_or_output(tmp_path,monkeypatch,operation):
    module=api()
    from treeneuron.research import qualified_primary_models_v2
    def forbidden(*args,**kwargs):raise AssertionError('A false registry cannot reach primary models or protected roles')
    monkeypatch.setattr(qualified_primary_models_v2,'load_qualified_primary_models',forbidden)
    ref=lambda name:dict(path=str(tmp_path/name),sha256='a'*64)
    q=dict(root=str(tmp_path/'fake-registry'),inventory_sha256='b'*64)
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        if operation=='seal':module.seal_global_primary_barrier(practical_plan_reference=ref('forbidden-plan'),
            expected_practical_plan_hash='c'*64,procedure_reference=ref('forbidden-procedure'),support_reference=ref('forbidden-support'),
            qualification_reference=q,qualified_primary_models_reference=ref('forbidden-models'),output=tmp_path/'not-created.json')
        else:module.load_global_primary_barrier(ref('forbidden-freeze'),qualification_reference=q)
    assert 'forbidden-' not in str(error.value) and list(tmp_path.iterdir())==[]


def test_publication_waits_until_the_access_context_has_completed_its_exit_checks(tmp_path):
    module=api()
    assert hasattr(module,'_publish_after_validating'),'Missing validation-before-publication transaction'
    from test_transfer_access_v2 import setup,append
    from treeneuron.research.transfer_access_v2 import frozen_primary_access_prefix
    root,digest,store=setup(tmp_path);append(root);output=tmp_path/'not-published.json'
    def prepare():
        with frozen_primary_access_prefix(root,expected_manifest_sha256=digest):
            # A direct writer can ignore flock; context exit must detect it
            # BEFORE the canonical file appears. This is toy metadata only.
            append(root)
            value=dict(scope='unqualified software metadata',protected_access_authorized=False)
        return value
    with pytest.raises(ValueError):module._publish_after_validating(output,prepare)
    assert not output.exists()
