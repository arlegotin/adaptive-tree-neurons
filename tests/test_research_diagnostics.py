"""Additional WP02 diagnostics distinguish replay from exhausted capacity."""
import copy

import numpy as np
import pytest

from test_train import training_case, cpu_training_evidence


def trajectory(*, selected=2, last=6, start=0, interval=2):
    rows = [{'type': 'stage_started', 'update': start, 'stage': 'base'}]
    for step in range(start, last+1):
        if step > start:
            rows.append(dict(type='update', update=step, stage_update=step-start,
                loss_standardized_mse=1/(step+1), learning_rate=.005,
                rows=2, row_ids=[step % 3, (step+1) % 3]))
        if (step-start) % interval == 0:
            rows.append(dict(type='evaluation', update=step, stage_update=step-start,
                split='G', metrics={'mse': 1 if step == selected else 2}))
    rows.append(dict(type='selection', selected_update=selected,
        last_update=last, executed_updates=last-start))
    return rows


def test_replay_requires_patience_premises_and_exact_native_suffix():
    from treeneuron.research.replay import replay_evidence
    parent = trajectory()
    keep = trajectory(start=2, selected=2)
    proof = replay_evidence(parent, keep, validation_interval=2, patience=2,
                            parent_max_updates=8)
    assert proof['classification'] == 'verified_patience_tail_replay'
    assert proof['matching_updates'] == 4
    assert proof['matching_G_evaluations'] == 3
    assert proof['keep_selected_stage_update'] == 0
    assert proof['training_headroom_exhausted'] is False
    changed = copy.deepcopy(keep)
    next(e for e in changed if e['type'] == 'update')['row_ids'][0] = 999
    failed = replay_evidence(parent, changed, validation_interval=2, patience=2,
                             parent_max_updates=8)
    assert failed['classification'] == 'replay_mismatch'
    assert failed['mismatches'][0]['field'] == 'row_ids'
    capped = trajectory(last=4, selected=4)
    control = replay_evidence(capped, trajectory(start=4, selected=4, last=8),
        validation_interval=2, patience=2, parent_max_updates=4)
    assert control['classification'] == 'not_replay_guaranteed'
    assert control['parent_stop'] == 'cap'


def test_replay_missing_rows_is_incomplete_not_a_proof():
    from treeneuron.research.replay import replay_evidence
    parent, keep = trajectory(), trajectory(start=2, selected=2)
    del next(e for e in keep if e['type'] == 'update')['row_ids']
    result = replay_evidence(parent, keep, validation_interval=2, patience=2,
                             parent_max_updates=8)
    assert result['classification'] == 'incomplete_evidence'


@pytest.mark.parametrize('corruption', ['duplicate', 'reverse', 'wrong_start', 'extra_update',
                                      'wrong_count', 'missing_metrics', 'wrong_best'])
def test_replay_rejects_contradictory_or_incomplete_raw_streams(corruption):
    from treeneuron.research.replay import replay_evidence
    parent, keep = trajectory(), trajectory(start=2, selected=2)
    if corruption == 'duplicate':
        duplicate = copy.deepcopy(next(e for e in keep if e['type'] == 'update'))
        duplicate['loss_standardized_mse'] = 999
        keep.insert(2, duplicate)
    elif corruption == 'reverse':
        keep.reverse()
    elif corruption == 'wrong_start':
        keep[0]['update'] = 999
    elif corruption == 'extra_update':
        extra = copy.deepcopy(next(e for e in keep if e['type'] == 'update'))
        extra['update'] = 7
        keep.insert(-1, extra)
    elif corruption == 'wrong_count':
        keep[-1]['executed_updates'] = 999
    elif corruption == 'missing_metrics':
        del next(e for e in parent if e['type'] == 'evaluation')['metrics']
    else:
        keep[-1]['selected_update'] = 4
    result = replay_evidence(parent, keep, validation_interval=2, patience=2, parent_max_updates=8)
    assert result['classification'] == 'incomplete_evidence'


def test_replay_counts_matching_G_independently_from_training_row_mismatches():
    from treeneuron.research.replay import replay_evidence
    parent, keep = trajectory(), trajectory(start=2, selected=2)
    next(e for e in keep if e.get('type') == 'update' and e['update'] == 4)['row_ids'][0] = 999
    result = replay_evidence(parent, keep, validation_interval=2, patience=2, parent_max_updates=8)
    assert result['classification'] == 'replay_mismatch'
    assert result['matching_G_evaluations'] == 3 and result['matching_updates'] == 3


def test_replay_observes_real_native_patience_tail(training_case, tmp_path, monkeypatch):
    import json
    from treeneuron import train
    from treeneuron.artifacts import create_run
    from treeneuron.config import resolve_experiment
    from treeneuron.research.replay import replay_evidence
    data, config, _ = training_case
    raw = config.to_dict()
    raw['training'].update(base_updates=6, continuation_updates=6,
                           validation_interval=2, patience=2)
    config = resolve_experiment(raw)
    # Constant G risk forces strict-best update0; F updates remain native.
    monkeypatch.setattr(train, 'regression_metrics', lambda *args, **kwargs: {'mse': 1.})
    parent_paths = create_run(tmp_path/'parent', config.to_dict())
    parent = train.fit_parent(data, config, parent_paths)
    selected_update = parent.update
    keep_paths = create_run(tmp_path/'keep', config.to_dict())
    keep = train.continue_fit(parent, data, train.TrainingPolicy.from_config(config, continuation=True), keep_paths)
    events = lambda p: [json.loads(line) for line in (p.directory/'events.jsonl').read_text().splitlines()]
    proof = replay_evidence(events(parent_paths), events(keep_paths), validation_interval=2,
                            patience=2, parent_max_updates=6)
    assert proof['classification'] == 'verified_patience_tail_replay'
    assert proof['matching_updates'] == 4 and keep.update == selected_update == 0


def test_seed32_collision_resolution_records_smallest_counter(monkeypatch):
    from treeneuron.research import seeds
    monkeypatch.setattr(seeds, 'seed64', lambda root, *labels: 17+int(labels[-1]))
    used = {17, 18}
    assert seeds.seed32(1, 'x', used=used) == (19, 2)
    assert used == {17, 18, 19}


def test_zero_score_and_duplicate_ties_keep_common_inventory_and_count():
    import torch
    from treeneuron.train import SelectiveRegressor
    from treeneuron.growth.score import score_units
    from treeneuron.growth.allocate import allocate
    model = SelectiveRegressor(2, 4).double().eval()
    with torch.no_grad():
        model.layer.base.weight.fill_(1.)
        model.layer.base.bias.fill_(2.)
        model.head.weight.fill_(1.)
        model.head.bias.fill_(0.)
    x = torch.tensor([[.2, .3], [-.2, .1], [.5, .5]], dtype=torch.float64)
    score = lambda: score_units(model, x, torch.zeros(3, dtype=torch.float64),
        epsilon=.05, tau=1., adam_epsilon=1e-8, batch_size=3, unit_block_size=4)
    table = score()
    assert table.eligible.tolist() == [True]*4
    assert allocate(table, 2, policy='adam', seed=7).ids.tolist() == [0, 1]
    assert allocate(table, 8, policy='random', seed=7).effective_count == 4
    with torch.no_grad():
        model.head.weight.zero_()
    zero = score()
    assert zero.status == ('zero_score',)*4
    for policy in ('adam', 'random'):
        result = allocate(zero, 2, policy=policy, seed=7)
        assert result.requested_count == 2 and result.effective_count == 0
        assert result.ids.tolist() == []


def test_custom_prevalence_and_covariance_are_separate_serialized_populations():
    from treeneuron.research.stress import (StressTeacher, make_stress_teacher,
                                           sample_stress, evaluate_stress)
    from treeneuron.data.synthetic import make_teacher
    with pytest.raises(ValueError, match='Rare'):
        make_teacher(123, 4, regime='rare', gate_prevalences=(.05,))
    teacher = make_stress_teacher(123, 4, prevalence=.05, condition_target=10.)
    assert teacher.kind == 'custom_prevalence_covariance'
    assert teacher.base.prevalences == (.05,)*4
    assert np.linalg.cond(teacher.population_covariance) == pytest.approx(10., rel=1e-12)
    restored = StressTeacher.from_dict(teacher.to_dict())
    assert restored.teacher_id == teacher.teacher_id
    rows = sample_stress(teacher, 19, 456, sigma=0.)
    again = sample_stress(restored, 19, 456, sigma=0.)
    np.testing.assert_array_equal(rows.X, again.X)
    np.testing.assert_array_equal(rows.y, again.y)
    np.testing.assert_allclose(evaluate_stress(teacher, rows.X), rows.y, atol=1e-13, rtol=1e-13)
    assert teacher.teacher_id in rows.stream_id
    assert not rows.X.flags.writeable
    other = make_stress_teacher(123, 4, prevalence=.2, condition_target=10.)
    assert other.teacher_id != teacher.teacher_id
    assert np.array_equal(other.input_map, teacher.input_map)


def test_stress_nuisance_coordinates_remain_irrelevant_and_bad_specs_fail():
    from treeneuron.research.stress import make_stress_teacher, sample_stress, evaluate_stress
    teacher = make_stress_teacher(55, 4, nuisance_dimensions=2, condition_target=10.)
    rows = sample_stress(teacher, 7, 88)
    changed = rows.X.copy()
    changed[:, -2:] += 100
    np.testing.assert_array_equal(evaluate_stress(teacher, rows.X), evaluate_stress(teacher, changed))
    for bad in ({'prevalence': .01}, {'condition_target': 2.}, {'condition_target': True}):
        with pytest.raises(ValueError):
            make_stress_teacher(55, 4, **bad)
