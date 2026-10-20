"""Bounded geometry exercises never stand in for native quality training."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_native_qualification_uncapped import prepared_config


def api():
    assert importlib.util.find_spec('treeneuron.research.support_probes_v2') is not None, 'Missing native support workload probes'
    from treeneuron.research import support_probes_v2
    return support_probes_v2


SCHEMA = dict(columns=['x', 'kind'], categorical=['kind'])


def test_catalog_exercises_largest_declared_geometries_without_private_imports_or_quality_authority():
    module = api()
    catalogs = {f'P{i:02}': module.support_probe_cases(f'P{i:02}', SCHEMA) for i in range(14)}
    assert all(catalogs.values())
    for family, rows in catalogs.items():
        assert len({row['case_id'] for row in rows}) == len(rows)
        assert all(row['native_config']['family'] == family for row in rows)
        assert all(row['quality_eligible'] is False and row['native_workload_qualified'] is False for row in rows)
        assert rows == module.support_probe_cases(family, SCHEMA)
    assert all(r['native_config']['parameters']['width'] == 128 for r in catalogs['P02'])
    assert {r['native_config']['variant'] for r in catalogs['P02']} == {'lower', 'upper'}
    assert {r['native_config']['parameters']['batch_size'] for r in catalogs['P02']} == {128, 256}
    assert any(r['native_config']['parameters']['hidden_dims'] == [512]*3 for r in catalogs['P03'])
    assert any(r['native_config']['parameters']['width'] == 512 and r['native_config']['parameters']['blocks'] == 4 for r in catalogs['P04'])
    embedded = next(r for r in catalogs['P07'] if r['native_config']['variant'] == 'piecewise')['native_config']['parameters']
    assert (embedded['d_block'], embedded['n_blocks'], embedded['n_bins'], embedded['d_embedding']) == (1024, 4, 128, 32)
    assert catalogs['P12'][0]['native_config']['parameters'] == dict(trees=2048, depth=6, lr_multiplier=1)
    assert catalogs['P13'][0]['native_config']['parameters']['n_knots'] == 16


@pytest.mark.parametrize('family', [f'P{i:02}' for i in range(14)])
def test_probe_retains_native_shape_parameters_and_has_explicit_short_nonquality_budget(family):
    module = api()
    for case in module.support_probe_cases(family, SCHEMA):
        native = copy.deepcopy(case['native_config'])
        probe = module.engineering_probe_configuration(native, fit_rows=2306)
        assert native == case['native_config']
        assert probe['configuration']['parameters'] == native['parameters']
        assert probe['configuration']['feature_schema'] == native['feature_schema']
        assert probe['full_native_training'] is False
        if family == 'P12':
            assert probe['configuration'] == native and probe['budget'] == {'max_updates': 6}
        elif family == 'P13':
            assert probe['configuration'] == native
        else:
            assert probe['configuration']['mode'] == 'engineering_fixture'
        assert probe['native_workload_qualified'] is probe['quality_eligible'] is False


def plan_for(module, tmp_path, family='P13'):
    _, value = prepared_config(tmp_path/'input')
    ref = value['datasets']['energy']
    return module.plan_support_probes(prepared_root=ref['root'], expected_manifest_sha256=ref['manifest_sha256'], family=family)


def test_actual_simple_workload_closes_finite_model_and_reload_without_any_V_or_T_labels(tmp_path):
    module = api(); plan = plan_for(module, tmp_path)
    # The maximum spline geometry needs more rows than this tiny fixture; use
    # the explicitly retained ridge geometry case, never silently replace it.
    case = next(c for c in plan['cases'] if c['native_config']['estimator'] == 'ridge')
    result = module.run_support_probe(plan=plan, case_id=case['case_id'], output=tmp_path/'probe', interpreter=sys.executable)
    assert result['status'] == 'completed_ineligible' and result['snapshot_reload_exact'] is True
    assert result['query_predictions_finite'] is True and result['source_sha256'] == plan['source_sha256']
    assert result['V_labels_read'] is result['T_accessed'] is result['native_workload_qualified'] is result['quality_eligible'] is False
    process = json.loads((tmp_path/'probe/process/process.json').read_text())
    assert process['timeout_ns'] is None and process['child_reaped'] is True
    assert result['elapsed_ns'] > 0 and result['process_receipt_sha256']
    events = list(map(json.loads, (Path(plan['prepared_root'])/'access.jsonl').read_text().splitlines()))
    assert all(event['roles'] == ['F_labels', 'G_labels', 'V_features'] for event in events)
    assert json.loads((tmp_path/'probe/result.json').read_text()) == result


def test_actual_full_width_ordinary_growth_probe_does_not_shrink_to_the_old_tiny_fixture(tmp_path):
    module = api(); plan = plan_for(module, tmp_path, 'P02')
    case = next(c for c in plan['cases'] if c['native_config']['variant'] == 'upper')
    result = module.run_support_probe(plan=plan, case_id=case['case_id'], output=tmp_path/'probe', interpreter=sys.executable)
    assert result['status'] == 'completed_ineligible'
    summary = json.loads((tmp_path/'probe/worker/terminal.json').read_text())['adapter_resources']
    assert summary['resolved_parameters']['width'] == 128
    assert summary['ordinary_width'] > 128 and summary['growth_function_preserved_exact'] is True
    assert summary['quality_eligible'] is False
    assert result['full_native_training'] is result['native_workload_qualified'] is False


@pytest.mark.parametrize('change', ['source', 'case', 'manifest', 'output'])
def test_changed_plan_or_forbidden_output_rejects_before_worker_or_role_access(tmp_path, change):
    module = api(); plan = plan_for(module, tmp_path)
    root = Path(plan['prepared_root']); output = tmp_path/'bad'; case = plan['cases'][0]['case_id']
    if change == 'source': plan['source_sha256'] = '0'*64
    elif change == 'case': plan['cases'][0]['native_config']['parameters']['n_knots'] = 4
    elif change == 'manifest': (root/'manifest.json').write_text('{}')
    else: output = root/'forbidden'
    before = (root/'access.jsonl').read_bytes() if (root/'access.jsonl').exists() else b''
    with pytest.raises(ValueError): module.run_support_probe(plan=plan, case_id=case, output=output, interpreter=sys.executable)
    assert not output.exists()
    assert ((root/'access.jsonl').read_bytes() if (root/'access.jsonl').exists() else b'') == before


def test_owned_RSS_stop_retains_failed_geometry_without_claiming_it_passed(tmp_path):
    module = api(); plan = plan_for(module, tmp_path)
    case = next(c for c in plan['cases'] if c['native_config']['estimator'] == 'ridge')
    result = module.run_support_probe(plan=plan, case_id=case['case_id'], output=tmp_path/'stopped', interpreter=sys.executable, rss_limit_bytes=1)
    assert result['status'] == 'resource_exhausted'
    assert result['snapshot_reload_exact'] is result['query_predictions_finite'] is False
    assert result['quality_eligible'] is result['native_workload_qualified'] is False


def test_late_original_role_change_invalidates_the_published_success_flags(tmp_path, monkeypatch):
    module = api(); plan = plan_for(module, tmp_path)
    case = next(c for c in plan['cases'] if c['native_config']['estimator'] == 'ridge')
    original = module.EngineeringTrialBridge._worker_terminal
    def changed_after_model_closed(*args, **kwargs):
        terminal = original(*args, **kwargs)
        assert terminal['status'] == 'completed_ineligible'
        with (Path(plan['prepared_root'])/'outer-0/G.npz').open('ab') as stream:
            stream.write(b'controlled late role-byte drift')
        return terminal
    monkeypatch.setattr(module.EngineeringTrialBridge, '_worker_terminal', changed_after_model_closed)
    with pytest.raises(ValueError):
        module.run_support_probe(plan=plan, case_id=case['case_id'], output=tmp_path/'changed', interpreter=sys.executable)
    saved = json.loads((tmp_path/'changed/result.json').read_text())
    assert saved['status'] == 'input_or_dependency_failure'
    assert saved['snapshot_reload_exact'] is saved['query_predictions_finite'] is False
    assert saved['native_workload_qualified'] is False
