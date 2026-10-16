"""End-to-end causal pairing, retained controls and artifact-only rebuilding."""
import copy
import json
import weakref
from dataclasses import replace

import pytest
import torch

from test_train import cpu_training_evidence
from treeneuron.config import resolve_experiment


def small_config(root, receipt):
    return resolve_experiment({'study_id': 'paired-contract',
        'model': {'d': 2, 'h': 4},
        'data': {'population': 'conditional_slope_2d', 'n_fit': 16, 'n_growth': 8, 'n_validation': 8},
        'training': {'batch_size': 8, 'base_updates': 2, 'continuation_updates': 2,
                     'validation_interval': 1, 'patience': 20},
        'backend': {'cpu_threads': 1},
        'execution': {'backend_report': str(receipt), 'artifacts_root': str(root)}})


@pytest.fixture(scope='module')
def study(tmp_path_factory, cpu_training_evidence):
    from treeneuron.experiments import paired
    config = small_config(tmp_path_factory.mktemp('paired'), cpu_training_evidence)
    original = paired.load_checkpoint
    refs = []
    def sequential_load(*args, **kwargs):
        assert all(ref() is None for ref in refs), 'previous parent fork still live'
        state = original(*args, **kwargs)
        refs.append(weakref.ref(state))
        return state
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(paired, 'load_checkpoint', sequential_load)
        directory = paired.run_paired(config)
    assert len(refs) >= 6
    return directory


def test_complete_study_retains_all_branches_and_equal_budgets(study):
    from treeneuron.analysis.exploratory import summarize_exploratory
    report = summarize_exploratory(study)
    assert report['status'] == 'completed'
    methods = report['methods']
    assert {'parent', 'keep', 'sham', 'top', 'random', 'gaussian', 'mean', 'ridge',
            'static_partial_centered', 'static_full_direct_leaf', 'ordinary_lower',
            'ordinary_upper', 'widen_lower', 'widen_upper'} <= methods.keys()
    assert methods['top']['allocation']['requested_count'] == methods['random']['allocation']['requested_count'] == 16
    assert methods['top']['allocation']['effective_count'] == methods['random']['allocation']['effective_count']
    assert report['pairing_audit']['passed']
    assert methods['static_full_direct_leaf']['declared_total_updates'] == 4
    assert methods['top']['declared_total_updates'] == 4
    assert report['paired_differences']['sham_minus_keep']['G']['mse'] == 0
    assert report['paired_differences']['sham_minus_keep']['V']['max_prediction_abs'] == 0
    assert report['costs']['total_study_ns'] > report['costs']['shared_scoring_ns'] > 0
    assert report['costs']['standalone_random_scoring_ns'] is None
    assert report['tiny_gate_probe']['table_predicted_decrease'] > 0
    assert report['tiny_gate_probe']['table_gradient_max_abs'] < 1e-10
    for name, method in methods.items():
        assert method['status'] == 'completed', name
        assert method['enclosing_ns'] > 0
        assert method['metrics']['V']['mse'] >= 0


def test_pairing_audit_rejects_sampler_cursor_mutation(study):
    from treeneuron.experiments.paired import audit_pairing
    report = json.loads((study / 'study.json').read_text())
    report['methods']['random']['initial_pairing']['cursor'] += 1
    with pytest.raises(ValueError, match='pairing'):
        audit_pairing(report)


def test_shared_per_id_contrasts_and_executed_row_streams(study):
    report = json.loads((study / 'study.json').read_text())
    table = json.loads((study / 'score-table.json').read_text())
    for name in ('top', 'random'):
        method = report['methods'][name]
        allocation = method['allocation']
        for unit, contrast in zip(allocation['ids'], allocation['contrasts_float64']):
            assert contrast == table['contrasts'][unit]
        assert method['initial_pairing'] == report['parent_pairing']
    top = report['methods']['top']['executed_row_ids']
    random = report['methods']['random']['executed_row_ids']
    assert top[:min(len(top), len(random))] == random[:min(len(top), len(random))]


def test_exploration_never_loads_T(tmp_path, cpu_training_evidence, monkeypatch):
    from treeneuron.experiments.paired import run_paired
    from treeneuron.data import types
    def forbidden(*args, **kwargs):
        raise AssertionError('Protected T loader called')
    monkeypatch.setattr(types.DevelopmentData, 'T', property(forbidden), raising=False)
    path = run_paired(small_config(tmp_path, cpu_training_evidence))
    assert json.loads((path / 'study.json').read_text())['test_labels_accessed'] is False


def test_zero_score_teacher_is_retained(tmp_path, cpu_training_evidence, monkeypatch):
    from treeneuron.experiments import paired
    actual_score = paired.score_units
    def zero_residual_teacher(model, X, y, **kwargs):
        # Exact zero residual teacher on this F: real scorer, no invented table.
        with torch.no_grad(): target = model(X).clone()
        table = actual_score(model, X, target, **kwargs)
        assert not table.eligible.any()
        return table
    monkeypatch.setattr(paired, 'score_units', zero_residual_teacher)
    path = paired.run_paired(small_config(tmp_path, cpu_training_evidence))
    report = json.loads((path / 'study.json').read_text())
    top, random = [report['methods'][name]['allocation'] for name in ('top', 'random')]
    assert top['requested_count'] == random['requested_count'] == 16
    assert top['effective_count'] == random['effective_count'] == 0
    assert report['status'] == 'completed'


def test_summary_detects_modified_prediction_artifact(study, tmp_path):
    import shutil
    from treeneuron.analysis.exploratory import summarize_exploratory
    shutil.copytree(study, tmp_path / 'copy')
    copied = tmp_path / 'copy'
    target = copied / 'predictions' / 'top-G.json'
    target.write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        summarize_exploratory(copied)


def test_failed_fork_retains_completed_losers_and_terminal_cost(tmp_path, cpu_training_evidence, monkeypatch):
    from treeneuron.experiments import paired
    build = paired.build_causal
    def fail_top(method, recipe, **kwargs):
        if method == 'B5' and recipe.get('variant') == 'adam':
            raise RuntimeError('intentional factory failure')
        return build(method, recipe, **kwargs)
    monkeypatch.setattr(paired, 'build_causal', fail_top)
    with pytest.raises(RuntimeError, match='intentional factory failure'):
        paired.run_paired(small_config(tmp_path, cpu_training_evidence))
    report = json.loads(next(tmp_path.glob('*/study.json')).read_text())
    assert report['status'] == 'failed'
    assert report['methods']['keep']['status'] == 'completed'
    assert report['methods']['sham']['status'] == 'completed'
    assert report['methods']['top']['status'] == 'failed'
    assert report['methods']['top']['enclosing_ns'] > 0
    assert report['failures'][0]['message'] == 'intentional factory failure'
