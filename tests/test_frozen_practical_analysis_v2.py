"""Fixed numeric metadata fixtures confer no native/protected authority."""
import copy
import importlib.util
import math

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.frozen_practical_analysis_v2'), 'Missing frozen practical analysis binding'
    from treeneuron.research import frozen_practical_analysis_v2
    return frozen_practical_analysis_v2


def fixture():
    from treeneuron.research.deployment_worker_v2 import _source
    from treeneuron.research.protected_journal_v2 import _hash
    names=('airfoil','concrete','ccpp','parkinsons','bike','energy');views=('native_quality','600000000000','1800000000000','7200000000000')
    queries=[];bindings=[];folds=[];data={};champions={}
    for name in names:
        data[name]=dict(folds=[]);champions[name]={str(r):{} for r in range(2)}
        for fold in range(5):
            primary=fold==(4 if name=='bike' else 0);split=_hash([name,fold]);n=21 if name=='bike' else 3
            ids=list(range(15340,15340+n)) if name=='bike' else list(range(n))
            row=dict(dataset=name,outer_fold=fold,primary=primary,split_hash=split,queries=[],group_ids=ids,
                normalized_group_losses={},arithmetic=dict(F_target_variance=1.,methods={}))
            if name=='bike':row['day_ordinals']=ids.copy()
            data[name]['folds'].append(dict(outer_fold=fold,primary=primary,split_hash=split))
            for r in range(2):
                for i in range(14):
                    family=f'P{i:02}';qid=_hash([name,fold,r,family]);job=f'{name}/outer-{fold}/repeat-{r}/{family}';model=dict(generation_sha256=qid)
                    query=dict(query_id=qid,job_id=job,dataset=name,outer_fold=fold,family_id=family,split_hash=split,model_binding=model)
                    queries.append(query);row['queries'].append(qid)
                    loss=([1.,3.] if i==0 else [20.,6.] if i==3 else [2.,8.])[r]
                    row['normalized_group_losses'][qid]=[loss]*n;row['arithmetic']['methods'][qid]=dict(normalized_risk=loss)
                    for view in views:
                        bindings.append(dict(job_id=job,dataset=name,outer_fold=fold,family_id=family,search_repeat=r,view=view,
                            status='retained_model_bytes',deadline_ns=None if view=='native_quality' else int(view),query_id=qid))
                        if primary and i==(2 if r==0 else 3):
                            champions[name][str(r)][view]=dict(status='V_selected_champion',family_id=family,job_id=job,
                                V_score=.5,model_binding=model,missing_families=[])
            folds.append(row)
    reference=dict(path='/unqualified-numeric-fixture-protocol.json',sha256='a'*64)
    manifest=dict(queries=queries,view_bindings=bindings);manifest['manifest_hash']=_hash(manifest)
    payload=dict(source_sha256=_source(),data=data,prediction_manifest=manifest,
        frozen_primary_champions=dict(choices=champions),policy=dict(analysis=dict(bootstrap_iterations=20000,bootstrap_seed=7,
            bike_primary_block_days=7,bike_sensitivity_block_days=[3,14])))
    metrics=dict(schema='research/v2-frozen-prediction-metrics',source_sha256=payload['source_sha256'],protocol_reference=reference,
        prediction_manifest_sha256=manifest['manifest_hash'],folds=folds,view_bindings=copy.deepcopy(bindings))
    return payload,reference,metrics


@pytest.mark.parametrize('entry',['seal','load'])
def test_public_analysis_requires_genuine_protocol_before_metric_or_output_files(tmp_path,entry):
    module=api();kwargs=dict(models_reference=dict(path=str(tmp_path/'forbidden-models'),sha256='a'*64),
        global_barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='b'*64),
        qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='c'*64),
        preT_costs_reference=dict(path=str(tmp_path/'forbidden-cost'),sha256='d'*64),
        readiness_reference=dict(path=str(tmp_path/'forbidden-readiness'),sha256='e'*64))
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        protocol=dict(path=str(tmp_path/'forbidden-protocol'),sha256='f'*64)
        stage=dict(path=str(tmp_path/'forbidden-stage'),sha256='0'*64)
        if entry=='seal':module.seal_protected_practical_analysis(protocol,prediction_stage_reference=stage,
            directory=tmp_path/'forbidden-study',output=tmp_path/'analysis.json',**kwargs)
        else:module.load_protected_practical_analysis(dict(path=str(tmp_path/'forbidden-analysis'),sha256='1'*64),
            protocol_reference=protocol,prediction_stage_reference=stage,directory=tmp_path/'forbidden-study',**kwargs)
    assert 'forbidden-' not in str(error.value) and list(tmp_path.iterdir())==[]


def test_all_views_and_repeats_use_the_frozen_V_champion_and_primary_folds_only():
    payload,reference,metrics=fixture();out=api()._compile_comparison_inputs(payload,reference,metrics)
    assert out['planned_views']==3360 and out['fold_count']==30
    primary=out['views']['native_quality']
    assert primary['status']=='complete' and primary['champions']['airfoil']==['P02','P03']
    packet=primary['losses']['airfoil']
    assert packet['family_losses']['P00']==[[1.,3.]]*3
    assert packet['family_losses']['P02']==[[2.,8.]]*3
    assert len(out['secondary_fold_summaries'])==24
    assert out['protected_access_authorized'] is out['native_models_authenticated'] is False


def test_frozen_analysis_averages_R2_risks_before_log_and_reuses_pairing_for_every_family():
    payload,reference,metrics=fixture();module=api()
    result=module._analyze_fixed_metrics(payload,reference,metrics)
    main=result['native_quality']['primary_seven_day']
    assert main['champion']['aggregate_point_log_benefit']==pytest.approx(math.log(2))
    assert main['family_comparisons']['P02']['aggregate_point_log_benefit']==pytest.approx(math.log(2.5))
    assert main['champion']['iterations']==20000
    assert main['multiplicity']['holm']['family_size']==13
    for family in main['family_comparisons'].values():assert family['pairing_index_sha256']==main['champion']['pairing_index_sha256']
    assert main['champion']['energy']['bootstrap_interval'] is None
    assert set(result['native_quality']['bike_sensitivities'])=={'3','14'}
    assert all(v['inferential_claim_authorized'] is False for v in result['descriptive_prefixes'].values())
    assert result['claim_authorized'] is result['raw_model_provenance_authenticated'] is False


@pytest.mark.parametrize('fault',['missing_view','duplicate_view','wrong_query','wrong_split','wrong_metric_query','wrong_champion_model','wrong_champion_job','changed_source','wrong_metric_mean'])
def test_compiler_rejects_missing_or_substituted_frozen_bindings(fault):
    payload,reference,metrics=fixture()
    if fault=='missing_view':metrics['view_bindings'].pop()
    elif fault=='duplicate_view':
        payload['prediction_manifest']['view_bindings'].append(copy.deepcopy(payload['prediction_manifest']['view_bindings'][0]));metrics['view_bindings']=copy.deepcopy(payload['prediction_manifest']['view_bindings'])
    elif fault=='wrong_query':metrics['folds'][0]['normalized_group_losses']['unknown-query']=[1.,1.,1.]
    elif fault=='wrong_split':metrics['folds'][0]['split_hash']='0'*64
    elif fault=='wrong_metric_query':metrics['folds'][0]['queries'].pop()
    elif fault=='wrong_champion_model':payload['frozen_primary_champions']['choices']['airfoil']['0']['native_quality']['model_binding']={}
    elif fault=='wrong_champion_job':payload['frozen_primary_champions']['choices']['airfoil']['0']['native_quality']['job_id']='different-job'
    elif fault=='changed_source':metrics['source_sha256']='0'*64
    else:metrics['folds'][0]['arithmetic']['methods'][metrics['folds'][0]['queries'][0]]['normalized_risk']=999.
    with pytest.raises(ValueError):api()._compile_comparison_inputs(payload,reference,metrics)


def test_unavailable_native_or_prefix_view_blocks_its_complete_comparison_without_imputation():
    payload,reference,metrics=fixture()
    for view in ('native_quality','600000000000'):
        row=next(v for v in payload['prediction_manifest']['view_bindings'] if v['dataset']=='airfoil' and v['outer_fold']==0 and v['family_id']=='P13' and v['search_repeat']==0 and v['view']==view)
        row.update(query_id=None,status='unavailable_no_eligible_model' if view=='native_quality' else 'unavailable_at_prefix')
        payload['frozen_primary_champions']['choices']['airfoil']['0'][view]=dict(status='unavailable_portfolio',family_id=None,
            job_id=None,V_score=None,model_binding=None,missing_families=['P13'])
    metrics['view_bindings']=copy.deepcopy(payload['prediction_manifest']['view_bindings'])
    out=api()._analyze_fixed_metrics(payload,reference,metrics)
    assert out['native_quality']['status']=='incomplete_primary_predictions'
    assert out['native_quality']['primary_seven_day'] is None
    assert len(out['native_quality']['missing'])==1
    assert out['descriptive_prefixes']['600000000000']['status']=='incomplete_primary_predictions'
    assert out['descriptive_prefixes']['1800000000000']['point_summary']['primary_mean_log_benefit']==pytest.approx(math.log(2))


def test_secondary_results_cannot_change_primary_effects_or_champion_selection():
    payload,reference,metrics=fixture();module=api();before=module._compile_comparison_inputs(payload,reference,metrics)
    for fold in metrics['folds']:
        if fold['primary']:continue
        for qid in fold['queries']:
            fold['normalized_group_losses'][qid]=[1000.]*len(fold['group_ids'])
            fold['arithmetic']['methods'][qid]['normalized_risk']=1000.
    after=module._compile_comparison_inputs(payload,reference,metrics)
    assert before['views']==after['views']
    assert before['secondary_fold_summaries']!=after['secondary_fold_summaries']


def test_a_missing_external_family_cannot_retain_a_false_complete_portfolio_champion():
    payload,reference,metrics=fixture()
    row=next(v for v in payload['prediction_manifest']['view_bindings'] if v['dataset']=='airfoil' and v['outer_fold']==0
        and v['family_id']=='P13' and v['search_repeat']==0 and v['view']=='native_quality')
    row.update(query_id=None,status='unavailable_no_eligible_model');metrics['view_bindings']=copy.deepcopy(payload['prediction_manifest']['view_bindings'])
    with pytest.raises(ValueError,match='portfolio'):api()._compile_comparison_inputs(payload,reference,metrics)


def test_zero_Energy_risk_does_not_suppress_five_observational_prefix_points():
    payload,reference,metrics=fixture()
    for row in metrics['folds']:
        if row['dataset']!='energy':continue
        for qid in row['queries']:
            row['normalized_group_losses'][qid]=[0.]*len(row['group_ids']);row['arithmetic']['methods'][qid]['normalized_risk']=0.
    out=api()._analyze_fixed_metrics(payload,reference,metrics)
    prefix=out['descriptive_prefixes']['600000000000']['point_summary']
    assert prefix is not None and prefix['primary_mean_log_benefit']==pytest.approx(math.log(2))
    assert prefix['energy_log_benefit'] is None


def test_undefined_F_variance_is_preserved_as_missing_normalization_instead_of_zero():
    payload,reference,metrics=fixture();row=metrics['folds'][0];row['arithmetic']['F_target_variance']=0.
    for qid in row['queries']:
        row['normalized_group_losses'][qid]=None;row['arithmetic']['methods'][qid]['normalized_risk']=None
    result=api()._analyze_fixed_metrics(payload,reference,metrics)
    assert result['native_quality']['status']=='incomplete_primary_predictions'
    assert len(result['native_quality']['missing'])==28
    assert {r['status'] for r in result['native_quality']['missing']}=={'undefined_F_variance'}
