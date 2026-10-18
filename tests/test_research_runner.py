"""Research contracts protect pairing, exact snapshots and planted controls."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_train import training_case, cpu_training_evidence, assert_state_equal


def test_seed_namespace_matches_canonical_handoff_and_rejects_ambiguous_labels():
    from treeneuron.research.seeds import seed64
    assert seed64(2026101801, 'M0', 'primary', 'teacher', '0') == 23290880342537040
    assert seed64(1, 'ab', 'c') != seed64(1, 'a', 'bc')
    with pytest.raises(ValueError):
        seed64(True, 'teacher')
    with pytest.raises(ValueError):
        seed64(1, 3)


def config_dict():
    return {'schema': 'research/v2', 'study_id': 'fixture', 'kind': 'm0_reference',
            'seed_root': 2026101801, 'teachers': 4, 'repetitions': 2,
            'evaluation_rows': 65536, 'base_experiment': {
                'study_id': 'fixture', 'model': {'d': 16, 'h': 64},
                'training': {'base_updates': 8000, 'continuation_updates': 8000,
                             'learning_rate': .005, 'batch_size': 128,
                             'validation_interval': 25, 'patience': 20},
                'backend': {'cpu_threads': 1, 'interop_threads': 1}}}


def test_strict_config_plan_freezes_role_counts_and_pairing():
    from treeneuron.research.config import ResearchConfig, expand
    raw = config_dict()
    config = ResearchConfig.from_dict(raw)
    raw['teachers'] = 99
    plan = expand(config)
    assert len(plan.attempts) == 64
    assert plan.probe_trials == 520
    assert len({a['teacher_seed'] for a in plan.attempts}) == 4
    first = [a for a in plan.attempts if a['teacher_index'] == 0]
    assert len({a['data_seed'] for a in first}) == 1
    for repeat in (0, 1):
        paired = [a for a in first if a['repetition'] == repeat]
        assert len({a['optimizer_seed'] for a in paired}) == 1
    assert len({a['optimizer_seed'] for a in first}) == 2
    for field, value in [('surprise', 1), ('kind', 'unregistered'), ('teachers', True)]:
        bad = config_dict()
        bad[field] = value
        with pytest.raises(ValueError):
            ResearchConfig.from_dict(bad)
    bad = config_dict()
    bad['base_experiment']['training']['patience_typo'] = 20
    with pytest.raises(ValueError):
        ResearchConfig.from_dict(bad)


def test_plan_cli_does_not_fit_or_open_labels(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps(config_dict()))
    output = tmp_path/'plan.json'
    result = subprocess.run([sys.executable, '-m', 'treeneuron.research.cli', 'plan',
                             '--config', str(config), '--output', str(output)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    value = json.loads(output.read_text())
    assert len(value['attempts']) == 64
    assert value['protected_labels_accessed'] is False
    again = subprocess.run(result.args, text=True, capture_output=True)
    assert again.returncode != 0
    assert json.loads(output.read_text()) == value


def test_attempt_ledger_retains_failures_and_rejects_identity_drift(tmp_path):
    from treeneuron.research.ledger import AttemptLedger
    plan = {'plan_hash': 'a'*64, 'attempts': [{'attempt_id': 'one'}, {'attempt_id': 'two'}]}
    ledger = AttemptLedger.create(tmp_path/'study', plan)
    ledger.record('one', 'running', elapsed_ns=0)
    ledger.record('one', 'numerical_failure', elapsed_ns=123, reason='nonfinite')
    resumed = AttemptLedger.resume(tmp_path/'study', plan)
    assert resumed.pending_ids == ('two',)
    assert resumed.charged_ns == 123
    with pytest.raises(ValueError):
        resumed.record('one', 'completed_eligible', elapsed_ns=124)
    with pytest.raises(ValueError):
        AttemptLedger.resume(tmp_path/'study', {**plan, 'plan_hash': 'b'*64})
    with pytest.raises(ValueError):
        AttemptLedger.create(tmp_path/'study', plan)


def test_exact_snapshot_is_actual_update_and_observer_cannot_change_state(training_case, tmp_path):
    from treeneuron import train
    from treeneuron.artifacts import create_run
    data, config, paths = training_case
    baseline = train.fit_parent(data, config, paths)
    seen = []

    def observe(label, snapshot):
        seen.append((label, snapshot['metadata']['update']))
        snapshot['tensors']['model']['head.bias'].fill_(999)
        np.random.random(9)
        torch.rand(9)

    observed = train.fit_parent(data, config, create_run(tmp_path/'observed', config.to_dict()),
                                snapshot_updates=(1, 3), on_snapshot=observe)
    assert seen[:2] == [('current', 1), ('current', 3)]
    assert seen[-1] == ('selected', observed.update)
    assert_state_equal(observed, baseline)


def test_planted_control_preserves_raw_predictions_and_dead_candidate():
    from treeneuron.research.controls import make_positive_control, transport_parent
    from treeneuron.data.transforms import FittedTransform
    control = make_positive_control(instance_seed=123, data_seed=456, gamma=2., sigma=.1)
    data = control.development
    transform = FittedTransform.fit(data.F.X, data.F.y)
    parent = transport_parent(control, transform, teacher=False)
    teacher = transport_parent(control, transform, teacher=True)
    x = np.asarray([[.25, -.5], [-.75, .4], [0, 0]], dtype=np.float64)
    xt = torch.as_tensor(transform.transform_X(x), dtype=torch.float64)
    with torch.no_grad():
        parent_pred = transform.inverse_y(parent(xt).numpy())
        teacher_pred = transform.inverse_y(teacher(xt).numpy())
    np.testing.assert_allclose(parent_pred, control.raw_predict(x, teacher=False), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(teacher_pred, control.raw_predict(x, teacher=True), atol=1e-12, rtol=1e-12)
    assert control.generating_ids == (0, 2)
    assert np.array_equal(control.base_rows[0], control.base_rows[1])
    assert control.head[0] == control.head[1]
    assert bool((parent.layer.base(xt)[:, 7] < 0).all())
    assert not parent.layer.growth_applied
    assert len(data.F.y) == 512 and len(data.G.y) == 256 and len(data.V.y) == 256


def test_fit_worker_view_never_exposes_validation_targets(training_case):
    from treeneuron.research.roles import fitting_view
    data, _, _ = training_case
    view = fitting_view(data)
    assert view.F is data.F and view.G is data.G
    np.testing.assert_array_equal(view.validation_features, data.V.X)
    with pytest.raises(PermissionError):
        _ = view.V
    with pytest.raises(PermissionError):
        _ = view.T


def test_runner_executes_small_finite_fixture_with_raw_evidence(tmp_path, cpu_training_evidence):
    from treeneuron.research.config import ResearchConfig, expand
    from treeneuron.research.runner import run
    raw = config_dict()
    raw.update(teachers=1, repetitions=1, evaluation_rows=8)
    raw['base_experiment'].update(model={'d': 2, 'h': 4}, growth={'m_requested': 2},
        data={'n_fit': 13, 'n_growth': 7, 'n_validation': 5},
        training={'base_updates': 6, 'continuation_updates': 6, 'learning_rate': .02,
                  'batch_size': 4, 'validation_interval': 2, 'patience': 20},
        execution={'backend_report': str(cpu_training_evidence)},
        pilot={'probe_updates': 2})
    plan = expand(ResearchConfig.from_dict(raw))
    result = run(plan, tmp_path/'study')
    assert result['status'] == 'completed'
    assert result['completed_final_fits'] == 8
    assert result['protected_labels_accessed'] is False
    assert (tmp_path/'study'/'events.jsonl').exists()
    predictions = list((tmp_path/'study').rglob('predictions.npz'))
    assert len(predictions) == 1
    with np.load(predictions[0], allow_pickle=False) as saved:
        assert saved['V_y'].shape == (5,) and saved['E_y'].shape == (8,)
        assert saved['top_V'].shape == (5,)
    with pytest.raises(ValueError):
        run(plan, tmp_path/'study')


def test_control_plan_keeps_four_base_instances_across_signal_coordinates():
    from treeneuron.research.config import ResearchConfig, expand
    raw = config_dict()
    raw.update(kind='m1_controls', study_id='M1-controls')
    raw['base_experiment']['study_id'] = 'M1-controls'
    plan = expand(ResearchConfig.from_dict(raw))
    positive = [a for a in plan.attempts if a['population'] == 'positive']
    assert len(positive) == 160
    assert len(plan.attempts) == 256
    assert len({a['teacher_seed'] for a in positive}) == 4
    assert len({a['cell'] for a in positive}) == 4


def test_planted_parent_is_unfitted_with_age_zero(tmp_path, cpu_training_evidence, monkeypatch):
    from treeneuron.research.controls import make_positive_control, initialize_control_state
    from treeneuron.artifacts import create_run
    from treeneuron.config import resolve_experiment
    from treeneuron import train
    control = make_positive_control(instance_seed=123, data_seed=456, gamma=2., sigma=.1)
    config = resolve_experiment({'study_id': 'planted-birth', 'model': {'d': 2, 'h': 8},
        'data': {'n_fit': 512, 'n_growth': 256, 'n_validation': 256},
        'training': {'base_updates': 4, 'continuation_updates': 4,
                     'validation_interval': 1, 'patience': 20},
        'backend': {'cpu_threads': 1, 'interop_threads': 1}})
    monkeypatch.setattr(train, 'BACKEND_REPORT_PATH', cpu_training_evidence)
    from treeneuron.research.roles import fitting_view
    parent = initialize_control_state(control.parameters, config, create_run(tmp_path/'parent', config.to_dict()),
                                      data=fitting_view(control.development))
    assert parent.update == 0 and not parent.optimizer.state
    assert not parent.model.layer.growth_applied
    np.testing.assert_allclose(parent.predict(control.development.G.X),
        control.raw_predict(control.development.G.X, teacher=False), atol=2e-6, rtol=2e-6)
    assert parent.stage['complete'] and parent.best_snapshot is None
    assert parent.parameter_birth_updates == dict.fromkeys(dict(parent.model.named_parameters()), 0)


def test_native_planted_forks_retain_all_oracles_and_raw_score_table(tmp_path, cpu_training_evidence):
    from treeneuron.research.controls import make_positive_control
    from treeneuron.research.control_runner import fit_control_pair
    from treeneuron.config import resolve_experiment
    from treeneuron.checkpoint import load_checkpoint
    control = make_positive_control(instance_seed=123, data_seed=456, gamma=2., sigma=.1)
    config = resolve_experiment({'study_id': 'control-path', 'model': {'d': 2, 'h': 8},
        'growth': {'m_requested': 2},
        'data': {'n_fit': 512, 'n_growth': 256, 'n_validation': 256},
        'training': {'base_updates': 4, 'continuation_updates': 4,
                     'validation_interval': 1, 'patience': 20},
        'backend': {'cpu_threads': 1, 'interop_threads': 1},
        'execution': {'backend_report': str(cpu_training_evidence)}})
    roles = ('keep', 'top', 'random', 'oracle_current', 'oracle_true')
    root = tmp_path/'control'
    from treeneuron.research.roles import fitting_view
    result = fit_control_pair(config, fitting_view(control.development), root, roles, control=control.parameters)
    assert set(result['methods']) == set(roles)
    assert set(result['models']) == set(roles)
    with np.load(root/'score-table.npz', allow_pickle=False) as saved:
        assert saved['eligible'].shape == (8,)
        assert not saved['eligible'][7]
        assert saved['eigenvalues'].shape == (8, 3)
    for role in roles:
        assert result['methods'][role]['executed_stage_updates'] == 4


def test_production_matrix_requires_measured_resource_admission_before_creating_output(tmp_path):
    from treeneuron.research.config import ResearchConfig, expand
    from treeneuron.research.runner import run
    plan = expand(ResearchConfig.from_dict(config_dict()))
    with pytest.raises(ValueError, match='resource admission'):
        run(plan, tmp_path/'unadmitted')
    assert not (tmp_path/'unadmitted').exists()


def test_new_stream_immediately_replaces_permutation_and_retains_adam(training_case):
    from treeneuron import train
    from treeneuron.research.treatments import reset_stream
    data, config, paths = training_case
    state = train.fit_parent(data, config, paths)
    old_order, old_cursor = state.order.clone(), state.cursor
    optimizer = copy.deepcopy(state.optimizer.state_dict())
    scheduler = state.scheduler.state_dict()
    update = state.update
    event = reset_stream(state, seed=77123)
    generator = torch.Generator().manual_seed(77123)
    assert torch.equal(state.order, torch.randperm(13, generator=generator))
    assert state.cursor == 0 and state.update == update
    assert state.scheduler.state_dict() == scheduler
    for key, value in optimizer['state'].items():
        for field, expected in value.items():
            torch.testing.assert_close(state.optimizer.state_dict()['state'][key][field], expected, rtol=0, atol=0)
    assert event['discarded_cursor'] == old_cursor
    assert event['discarded_order'] == old_order.tolist()
    assert state.growth_metadata['stream_phases'][-1] == event


def test_new_horizon_grids_are_nested_and_reject_historical_off_grid_endpoint():
    from treeneuron.research.snapshots import nested_grid
    a, b, c = map(nested_grid, (300, 1200, 8000))
    assert set(a) < set(b) < set(c)
    assert a[0] == 0 and a[-1] == 300
    with pytest.raises(ValueError):
        nested_grid(320)


def test_interrupted_ledger_resume_preserves_cost_and_committed_attempt(tmp_path):
    from treeneuron.research.ledger import AttemptLedger
    plan = {'plan_hash': 'a'*64, 'attempts': [{'attempt_id': 'one'}]}
    ledger = AttemptLedger.create(tmp_path/'study', plan)
    ledger.record('one', 'running', elapsed_ns=0)
    ledger.record('one', 'interrupted_recoverable', elapsed_ns=123)
    resumed = AttemptLedger.resume(tmp_path/'study', plan)
    assert resumed.pending_ids == ('one',)
    resumed.record('one', 'running', elapsed_ns=123, recovery_attempt=1)
    resumed.record('one', 'completed_eligible', elapsed_ns=200)
    assert resumed.charged_ns == 200
    assert resumed.events[1]['status'] == 'interrupted_recoverable'


def test_recovery_ledger_cannot_reset_the_one_retry_allowance(tmp_path):
    from treeneuron.research.ledger import AttemptLedger
    plan = {'plan_hash': 'a'*64, 'attempts': [{'attempt_id': 'one'}]}
    ledger = AttemptLedger.create(tmp_path/'study', plan)
    ledger.record('one', 'running', elapsed_ns=0)
    ledger.record('one', 'interrupted_recoverable', elapsed_ns=123)
    ledger.record('one', 'running', elapsed_ns=123, recovery_attempt=1)
    ledger.record('one', 'interrupted_recoverable', elapsed_ns=200)
    with pytest.raises(ValueError, match='retry|rerun'):
        ledger.record('one', 'running', elapsed_ns=200, recovery_attempt=1)


def test_study_lease_refuses_concurrent_execution_and_releases_on_error(tmp_path):
    from treeneuron.research.ledger import StudyLease
    root = tmp_path/'study'
    root.mkdir()
    with pytest.raises(RuntimeError):
        with StudyLease(root):
            with pytest.raises(ValueError, match='active|lease'):
                with StudyLease(root):
                    pytest.fail('concurrent lease acquired')
            raise RuntimeError('interruption')
    with StudyLease(root):
        assert (root/'worker-lease.json').exists()


def test_physical_cost_summary_distinguishes_invocation_and_cumulative_time():
    from treeneuron.research.runner import physical_cost_summary
    got = physical_cost_summary([{'invocation_elapsed_ns': 11}, {'invocation_elapsed_ns': 7}], 13)
    assert got == {'invocation_elapsed_ns': 13, 'cumulative_physical_elapsed_ns': 31}


def test_control_coordinator_passes_only_restricted_worker_inputs(tmp_path, cpu_training_evidence, monkeypatch):
    from treeneuron.research.config import ResearchConfig, expand
    from treeneuron.research.runner import run
    from treeneuron.research import control_runner
    class Checked(RuntimeError):
        pass
    def inspect(config, data, root, roles, *, control, **kwargs):
        with pytest.raises(PermissionError):
            _ = data.V
        assert not hasattr(control, 'development')
        assert not hasattr(control, 'evaluation')
        assert data.F.noise_free_target is None
        raise Checked('worker interface verified')
    monkeypatch.setattr(control_runner, 'fit_control_pair', inspect)
    raw = config_dict()
    raw.update(kind='m1_controls', teachers=1, repetitions=1, evaluation_rows=8)
    raw['base_experiment'].update(model={'d': 2, 'h': 4},
        data={'n_fit': 13, 'n_growth': 7, 'n_validation': 5},
        training={'base_updates': 4, 'continuation_updates': 4, 'validation_interval': 1, 'patience': 20},
        execution={'backend_report': str(cpu_training_evidence)})
    with pytest.raises(Checked):
        run(expand(ResearchConfig.from_dict(raw)), tmp_path/'study')


def test_admission_rejects_unverified_asserted_measurements(tmp_path):
    from treeneuron.research.config import ResearchConfig, expand
    from treeneuron.research.resources import validate_resource_admission
    from treeneuron.config import canonical_hash
    raw = config_dict()
    path = tmp_path/'admission.json'
    raw['resource_admission'] = str(path)
    plan = expand(ResearchConfig.from_dict(raw)).to_dict()
    record = dict(schema='research/v2-resource-admission', source_sha256=plan['source_sha256'],
        config_hash=canonical_hash(plan['config']), measurements='trust me',
        forecast_worker_seconds_p50=1, forecast_worker_seconds_p90=1,
        forecast_remaining_bytes=1, free_disk_margin_bytes=1, peak_rss_bytes=1,
        available_worker_seconds=1)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='admission|measurement'):
        validate_resource_admission(plan, tmp_path/'study')
