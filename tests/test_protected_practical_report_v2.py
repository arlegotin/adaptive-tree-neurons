"""Unqualified numeric report compilation is software evidence only."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.practical_report_v2'),'Missing actual protected practical report'
    from treeneuron.research import practical_report_v2
    return practical_report_v2


def fixture():
    from test_frozen_practical_analysis_v2 import fixture as numeric_fixture
    from treeneuron.research.frozen_practical_analysis_v2 import _analyze_fixed_metrics
    from treeneuron.research.deployment_worker_v2 import _source
    payload,reference,metrics=numeric_fixture()
    from treeneuron.research.protected_journal_v2 import _hash
    for binding in payload['prediction_manifest']['view_bindings']:
        if binding['view']!='native_quality':binding.update(status='unavailable_at_prefix',query_id=None)
    payload['prediction_manifest']['manifest_hash']=_hash({k:v for k,v in payload['prediction_manifest'].items() if k!='manifest_hash'})
    metrics['prediction_manifest_sha256']=payload['prediction_manifest']['manifest_hash'];metrics['view_bindings']=copy.deepcopy(payload['prediction_manifest']['view_bindings'])
    for name,rows in payload['frozen_primary_champions']['choices'].items():
        for repeat,views in rows.items():
            for view in list(views):
                if view!='native_quality':views[view]=dict(status='unavailable_portfolio',family_id=None,job_id=None,model_binding=None,V_score=None,missing_families=[f'P{i:02}' for i in range(2,14)])
    for fold in metrics['folds']:
        for values in fold['arithmetic']['methods'].values():
            risk=values['normalized_risk'];values.update(primary_MSE=risk,row_RMSE=risk**.5,row_MAE=risk**.5)
    # Rebind after adding the separate original-unit reporting fields.
    analysis=_analyze_fixed_metrics(payload,reference,metrics)
    identity=dict(pinned_commit='1'*40,execution_sha256='2'*64)
    attempts=[dict(attempt_id='known/attempt-000',status='numerical_failure'),dict(attempt_id='known/attempt-001',status='completed_eligible')]
    costs=dict(source_sha256=_source(),physical_owned_execution_ns=100,standalone_training_plus_all_frozen_queries_ns={'known':70})
    return payload,reference,metrics,analysis,identity,attempts,costs


def compile_fixture():
    m=api();p,r,metrics,analysis,identity,attempts,costs=fixture()
    return m._compile_practical_report(p,r,metrics,analysis,identity,attempts,costs,attempt_resolutions={})


def test_canonical_primary_is_only_140_native_observational_rows_but_all_views_remain():
    result=compile_fixture();audit=result['canonical_audit']
    assert len(audit['planned_prediction_ids'])==len(audit['predictions'])==140
    assert all('energy' not in row['task_id'] for row in audit['predictions'])
    assert len(result['supplemental_views'])==3360 and result['native_model_count']==840
    assert len(result['secondary_fold_summaries'])==24
    assert all(row['decision']!='passed' for row in audit['comparisons'])
    assert audit['evidence_scope']=='software' and audit['provenance']['protected_protocol_authenticated'] is False
    assert audit['source']==dict(commit='1'*40,package_sha256='2'*64)


def test_missing_descriptive_prefix_does_not_remove_eligible_native_primary_models():
    result=compile_fixture()
    assert sum(v['status']=='unavailable_at_prefix' for v in result['supplemental_views'])==2520
    assert all(p['status']=='completed_eligible' for p in result['canonical_audit']['predictions'])
    from treeneuron.analysis.report_v2 import render_report
    text=render_report(result['canonical_audit'])
    assert 'known/attempt-000' in text and 'numerical_failure' in text and 'No protected confirmation' in text


def test_unqualified_compilation_is_deterministic_and_cannot_authenticate_cost_or_population():
    first=compile_fixture();second=compile_fixture();assert first==second
    assert first['claim_authorized'] is first['population_superiority_confirmed'] is False
    assert first['whole_training_procedure_cost_authenticated'] is False
    assert first['canonical_audit']['costs']['physical_elapsed_ns']==100
    assert first['canonical_audit']['costs']['standalone_attributed_ns']==70


def test_public_report_loads_actual_gates_before_saved_records_or_output(tmp_path):
    m=api();fake=dict(path=str(tmp_path/'forbidden'),sha256='a'*64)
    kwargs={k:fake for k in ('programme_reference','analysis_reference','raw_audit_reference','prediction_stage_reference','costs_reference',
        'models_reference','global_barrier_reference','preT_costs_reference','readiness_reference')}
    kwargs['qualification_reference']=dict(root=str(tmp_path/'false-q'),inventory_sha256='b'*64)
    with pytest.raises((ValueError,FileNotFoundError)):
        m.write_protected_practical_report(fake,directory=tmp_path/'study',output=tmp_path/'report.md',**kwargs)
    assert list(tmp_path.iterdir())==[]


def test_changed_metric_or_analysis_binding_cannot_enter_report_tables():
    m=api();p,r,metrics,analysis,identity,attempts,costs=fixture()
    metrics['folds'][0]['arithmetic']['methods'][metrics['folds'][0]['queries'][0]]['normalized_risk']=1000.
    with pytest.raises(ValueError):m._compile_practical_report(p,r,metrics,analysis,identity,attempts,costs,attempt_resolutions={})
