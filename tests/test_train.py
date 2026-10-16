"""Training must select complete G states and preserve paired update streams."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from treeneuron.artifacts import create_run
from treeneuron.config import resolve_experiment
from treeneuron.data.types import DevelopmentData, ObservationSet


@pytest.fixture(scope='session')
def cpu_training_evidence(tmp_path_factory):
    # Actual numerical producer evidence, independent of an ignored local doctor
    # artifact or native MPS access. A one-batch probe measures the exact whole
    # physical=logical recipe and makes no batch-selection comparison claim.
    from treeneuron.backend import probe_backend, select_backend
    probe = probe_backend('cpu', seed=1602030, batch_sizes=(128,))
    assert probe['status'] == 'passed', probe.get('reason')
    probes = {'cpu': probe}
    report = {'backend_probes': {'probes': probes, 'comparison_blocks': [],
                                'decision': select_backend(probes)}}
    path = tmp_path_factory.mktemp('training-evidence') / 'report.json'
    path.write_text(json.dumps(report))
    return path


@pytest.fixture
def training_case(tmp_path, monkeypatch, cpu_training_evidence):
    from treeneuron import train
    monkeypatch.setattr(train, 'BACKEND_REPORT_PATH', cpu_training_evidence)
    config = resolve_experiment({'study_id': 'training-contract',
        'model': {'d': 2, 'h': 4},
        'data': {'n_fit': 13, 'n_growth': 7, 'n_validation': 5},
        'training': {'batch_size': 4, 'base_updates': 6, 'continuation_updates': 6,
                     'validation_interval': 2, 'patience': 20, 'learning_rate': .02,
                     'scheduler': 'cosine'},
        'backend': {'cpu_threads': 1}})
    rng = np.random.default_rng(83)
    def rows(n, start):
        x = rng.normal(size=(n, 2))
        ids = np.arange(start, start+n, dtype=np.int64)
        return ObservationSet(x, 7 + 3*x[:, 0] - x[:, 1], ids, ids)
    data = DevelopmentData(rows(13, 0), rows(7, 100), rows(5, 200), 'test-teacher')
    return data, config, create_run(tmp_path, config.to_dict())


def events(paths):
    return [json.loads(line) for line in (paths.directory / 'events.jsonl').read_text().splitlines()]


def assert_state_equal(a, b):
    assert (a.update, a.epoch, a.cursor) == (b.update, b.epoch, b.cursor)
    assert a.scheduler.state_dict() == b.scheduler.state_dict()
    torch.testing.assert_close(a.order, b.order, rtol=0, atol=0)
    torch.testing.assert_close(a.sampler_generator.get_state(), b.sampler_generator.get_state(), rtol=0, atol=0)
    for name, tensor in a.model.state_dict().items():
        torch.testing.assert_close(tensor, b.model.state_dict()[name], rtol=0, atol=0)
    def equal(x, y):
        if isinstance(x, torch.Tensor):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x: equal(x[k], y[k])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for i, j in zip(x, y): equal(i, j)
        else:
            assert x == y
    equal(a.optimizer.state_dict(), b.optimizer.state_dict())
    equal(a.rng_states, b.rng_states)


def test_parent_predictions_original_units_and_no_validation_access(training_case):
    from treeneuron.train import fit_parent
    data, config, paths = training_case
    class Forbidden:
        def __getattribute__(self, name):
            raise AssertionError('V must never be read during training')
    object.__setattr__(data, 'V', Forbidden())
    state = fit_parent(data, config, paths)
    x = torch.tensor(state.preprocessing.transform_X(data.G.X), dtype=torch.float32)
    with torch.no_grad():
        expected = state.preprocessing.inverse_y(state.model(x).numpy())
    np.testing.assert_array_equal(state.predict(data.G.X), expected)
    assert state.predict(data.G.X).shape == (7,)
    assert np.mean(state.predict(data.G.X)) > 4
    evaluations = [e for e in events(paths) if e['type'] == 'evaluation']
    assert [e['stage_update'] for e in evaluations] == [0, 2, 4, 6]
    assert all(e['split'] == 'G' for e in evaluations)
    manifest = json.loads((paths.directory / 'manifest.json').read_text())
    assert manifest['identity']['status'] == 'running'
    assert manifest['data']['split_hash'] == data.manifest.assignment_hash
    assert manifest['method']['trainable_parameters'] == 17
    assert manifest['method']['backend']['actual']['device'] == 'cpu'
    assert manifest['method']['serialized_bytes'] > 0
    assert manifest['costs']['base']['nanoseconds'] > 0
    assert manifest['data']['test_labels_accessed'] is False


def test_earliest_tie_keeps_complete_update_zero_for_parent_and_continuation(training_case):
    from treeneuron.train import fit_parent, continue_fit, TrainingPolicy
    data, config, paths = training_case
    # ReLU is inactive for the only varying feature; constant prediction is exact
    # at update zero and weight updates cannot improve zero risk.
    from treeneuron import train
    original = train.SelectiveRegressor
    class ConstantStart(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            with torch.no_grad():
                for p in self.parameters(): p.zero_()
    # Initialization seam only: real loss/evaluation/update/select/checkpoint path.
    from unittest.mock import patch
    rows = lambda r: ObservationSet(r.X, np.full(len(r.y), 8.), r.row_id, r.group_id)
    data = DevelopmentData(rows(data.F), rows(data.G), rows(data.V), data.teacher_id)
    with patch.object(train, 'SelectiveRegressor', ConstantStart):
        state = fit_parent(data, config, paths)
    assert state.update == 0 and state.cursor == 0
    assert state.optimizer.state_dict()['state'] == {}
    assert state.scheduler.position == 0
    before = copy.deepcopy(state.rng_states)
    child = create_run(paths.directory.parent, config.to_dict())
    continued = continue_fit(state, data, TrainingPolicy(4, 4, 4, 2, 20), child)
    assert continued.update == 0 and continued.cursor == 0
    assert continued.optimizer.state_dict()['state'] == {}
    assert continued.rng_states['python'] == before['python']
    assert [e['stage_update'] for e in events(child) if e['type'] == 'evaluation'] == [0, 2, 4]
    assert continued.history[-1]['type'] == 'selection'
    assert continued.history[-1]['executed_updates'] == 4


def test_interrupted_non_epoch_resume_matches_uninterrupted(training_case, monkeypatch):
    from treeneuron import train
    from treeneuron.checkpoint import load_checkpoint
    data, config, paths = training_case
    uninterrupted = train.fit_parent(data, config, paths)
    attempt = create_run(paths.directory.parent, config.to_dict())
    append = train.append_event
    def interrupt(paths, event):
        append(paths, event)
        if event['type'] == 'checkpoint' and event['update'] == 2:
            raise KeyboardInterrupt('controlled complete-update interruption')
    monkeypatch.setattr(train, 'append_event', interrupt)
    with pytest.raises(KeyboardInterrupt):
        train.fit_parent(data, config, attempt)
    monkeypatch.setattr(train, 'append_event', append)
    resumed = load_checkpoint(attempt.directory / 'checkpoints' / 'last.json',
                              expected_hashes=uninterrupted.identity_hashes, device='cpu')
    assert resumed.update == 2 and resumed.cursor == 8 and resumed.epoch == 0
    retry = create_run(paths.directory.parent, config.to_dict())
    resumed = train.continue_fit(resumed, data, train.TrainingPolicy(4, 4, 6, 2, 20), retry)
    assert_state_equal(resumed, uninterrupted)
    np.testing.assert_array_equal(resumed.predict(data.G.X), uninterrupted.predict(data.G.X))
    assert resumed.next_row_ids(data.F).tolist() == uninterrupted.next_row_ids(data.F).tolist()
    assert json.loads((attempt.directory / 'manifest.json').read_text())['identity']['status'] == 'interrupted'


def test_nonfinite_failure_is_logged_without_completed_record(training_case, monkeypatch):
    from treeneuron import train
    data, config, paths = training_case
    original = train.SelectiveRegressor.forward
    def nonfinite(self, x):
        return original(self, x) * float('nan') if self.training else original(self, x)
    monkeypatch.setattr(train.SelectiveRegressor, 'forward', nonfinite)
    with pytest.raises((ValueError, FloatingPointError), match='[Nn]onfinite|finite'):
        train.fit_parent(data, config, paths)
    assert events(paths)[-1]['type'] == 'failure'
    manifest = json.loads((paths.directory / 'manifest.json').read_text())
    assert manifest['identity']['status'] == 'failed'
    assert manifest['costs']['base']['nanoseconds'] > 0
    assert manifest['method']['serialized_bytes'] > 0


@pytest.mark.parametrize('changes,match', [
    ({'model': {'depth': 2}}, 'depth'),
    ({'training': {'base_updates': None}}, 'base_updates'),
    ({'training': {'optimizer': 'sgd'}}, 'sgd'),
])
def test_unsupported_or_unresolved_config_rejected(training_case, changes, match):
    from treeneuron.train import fit_parent
    data, config, paths = training_case
    value = config.to_dict()
    for section, fields in changes.items(): value[section].update(fields)
    config = resolve_experiment(value)
    paths = create_run(paths.directory.parent, config.to_dict())
    with pytest.raises(ValueError, match=match): fit_parent(data, config, paths)


def test_reject_unverified_accumulation():
    from treeneuron.train import TrainingPolicy
    with pytest.raises(ValueError, match='physical'):
        TrainingPolicy(128, 64, 10, 2, 2)


def test_nonzero_g_best_restores_moments_and_all_executed_diagnostics(training_case, monkeypatch):
    from treeneuron import train
    from treeneuron.checkpoint import load_checkpoint
    data, config, paths = training_case
    value = config.to_dict()
    value['training']['scheduler'] = 'constant'
    config = resolve_experiment(value)
    first = create_run(paths.directory.parent, config.to_dict())
    append = train.append_event
    def stop_after_two(paths, event):
        append(paths, event)
        if event['type'] == 'checkpoint' and event['update'] == 2: raise KeyboardInterrupt()
    monkeypatch.setattr(train, 'append_event', stop_after_two)
    with pytest.raises(KeyboardInterrupt): train.fit_parent(data, config, first)
    # Read safe metadata for the expected identity; the public load revalidates it.
    pointer = json.loads((first.directory / 'checkpoints/last.json').read_text())
    header = json.loads((first.directory / 'checkpoints' / pointer['metadata']).read_text())
    reference = load_checkpoint(first.directory / 'checkpoints/last.json',
        expected_hashes=header['state']['identity_hashes'], device='cpu')
    g = ObservationSet(data.G.X, reference.predict(data.G.X), data.G.row_id, data.G.group_id)
    selected_data = DevelopmentData(data.F, g, data.V, data.teacher_id)
    monkeypatch.setattr(train, 'append_event', append)
    attempt = create_run(paths.directory.parent, config.to_dict())
    selected = train.fit_parent(selected_data, config, attempt)
    assert selected.update == 2
    assert_state_equal(selected, reference)
    assert selected.history[-1]['executed_updates'] == 6
    assert len([e for e in selected.history if e['type'] == 'update']) == 6
    best = load_checkpoint(attempt.directory / 'checkpoints/best.json',
                          expected_hashes=selected.identity_hashes, device='cpu')
    assert_state_equal(best, selected)


def test_continuation_keeps_current_schedule_and_adam_recipe(training_case):
    from treeneuron.train import fit_parent, continue_fit, TrainingPolicy
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    position = state.update
    current_lr = state.optimizer.param_groups[0]['lr']
    child = create_run(paths.directory.parent, config.to_dict())
    continued = continue_fit(state, data, TrainingPolicy.from_config(config, continuation=True), child)
    updates = [e for e in events(child) if e['type'] == 'update']
    assert updates[0]['update'] == position + 1
    assert updates[0]['learning_rate'] == current_lr
    assert continued.scheduler.horizon == 12
    group = continued.optimizer.param_groups[0]
    assert group['foreach'] is False
    assert group['eps'] == 1e-8 and group['betas'] == (.9, .999)
    assert events(child)[1]['stage_update'] == 0


def test_scalar_wrapper_consumable_by_cpu64_scorer(training_case):
    from treeneuron.train import fit_parent
    from treeneuron.growth.score import score_units
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    score = score_units(copy.deepcopy(state.model).double().eval(),
        torch.tensor(state.preprocessing.transform_X(data.F.X), dtype=torch.float64),
        torch.tensor(state.preprocessing.transform_y(data.F.y), dtype=torch.float64),
        epsilon=.05, tau=1., adam_epsilon=1e-8, batch_size=4, unit_block_size=2)
    assert score.ids.tolist() == [0, 1, 2, 3]


def test_invalid_policy_records_failure_and_does_not_advance_parent(training_case):
    from treeneuron.train import fit_parent, continue_fit, TrainingPolicy
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    update = state.update
    child = create_run(paths.directory.parent, config.to_dict())
    with pytest.raises(ValueError, match='paired'):
        continue_fit(state, data, TrainingPolicy(2, 2, 6, 2, 20), child)
    assert state.update == update
    assert events(child)[-1]['type'] == 'failure'


def test_nonfinite_optimizer_state_is_failed_before_publication(training_case, monkeypatch):
    from treeneuron.train import fit_parent
    data, config, paths = training_case
    step = torch.optim.AdamW.step
    def corrupt(self, *args, **kwargs):
        result = step(self, *args, **kwargs)
        next(iter(self.state.values()))['exp_avg'].fill_(float('inf'))
        return result
    monkeypatch.setattr(torch.optim.AdamW, 'step', corrupt)
    with pytest.raises(FloatingPointError, match='optimizer'):
        fit_parent(data, config, paths)
    pointer = paths.directory / 'checkpoints/last.json'
    descriptor = json.loads((pointer.parent / json.loads(pointer.read_text())['metadata']).read_text())
    assert descriptor['state']['update'] == 0
    assert events(paths)[-1]['type'] == 'failure'


def test_completed_last_recovery_selects_best_before_new_continuation(training_case):
    from treeneuron.train import fit_parent, continue_fit, TrainingPolicy
    from treeneuron.checkpoint import load_checkpoint
    data, config, paths = training_case
    # Select an initial exact function, then let F deliberately train away from it.
    from treeneuron.train import SelectiveRegressor
    from treeneuron.data.transforms import FittedTransform
    torch.manual_seed(config.seeds.optimizer)
    initial = SelectiveRegressor(2, 4).eval()
    transform = FittedTransform.fit(data.F.X, data.F.y)
    with torch.no_grad():
        x = torch.tensor(transform.transform_X(data.G.X), dtype=torch.float32)
        y = transform.inverse_y(initial(x).numpy())
    data = DevelopmentData(data.F, ObservationSet(data.G.X, y, data.G.row_id, data.G.group_id), data.V, data.teacher_id)
    selected = fit_parent(data, config, paths)
    assert selected.update == 0
    last = load_checkpoint(paths.directory / 'checkpoints/last.json',
                           expected_hashes=selected.identity_hashes, device='cpu')
    assert last.update == 6
    child = create_run(paths.directory.parent, config.to_dict())
    continued = continue_fit(last, data, TrainingPolicy.from_config(config, continuation=True), child)
    assert continued.update == 0
    assert continued.optimizer.state_dict()['state'] == {}
    assert [e['update'] for e in events(child) if e['type'] == 'evaluation'][0] == 0
