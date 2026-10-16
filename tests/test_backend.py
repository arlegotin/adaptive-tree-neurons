"""Executed kernels, reference errors, complete-method selection and gate integrity."""
import copy
import json
import os
import subprocess

import pytest
import torch


@pytest.fixture(scope='module', params=['cpu', 'mps'])
def backend_report(request):
    from treeneuron.doctor import probe_backend

    rng, threads, dtype = torch.random.get_rng_state(), torch.get_num_threads(), torch.get_default_dtype()
    environment = dict(os.environ)
    report = probe_backend(request.param, seed=1602030, batch_sizes=(16, 32, 64, 128))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads and torch.get_default_dtype() == dtype
    assert all(os.environ.get(key) == value for key, value in environment.items())
    assert set(os.environ) == set(environment)
    if report['status'] == 'unavailable':
        assert request.param == 'mps'
        assert report['reason']
        pytest.skip(report['reason'])
    return report


def test_actual_float32_kernels_and_resume_against_cpu64(backend_report):
    from treeneuron.backend import REQUIRED_OPERATIONS, validate_backend_report

    validate_backend_report(backend_report)
    assert backend_report['status'] == 'passed'
    assert backend_report['fallback_disabled_before_import'] is True
    assert backend_report['environment']['dtype'] == 'torch.float32'
    assert backend_report['reference']['dtype'] == 'torch.float64'
    assert backend_report['reference']['device'] == 'cpu'
    assert set(backend_report['operations']) == set(REQUIRED_OPERATIONS)
    for op in backend_report['operations'].values():
        assert op['status'] == 'passed'
        assert len(op['trials']) == 9
        assert all(trial['max_normalized_error'] <= 1 for trial in op['trials'])
    assert backend_report['operations']['serialization']['max_abs'] == 0
    assert backend_report['operations']['resumed_update']['max_abs'] == 0


def test_doubled_physical_batches_keep_logical_updates_and_score_cpu64(backend_report):
    calibration = backend_report['calibration']
    assert calibration['logical_batch_size'] == 128
    assert calibration['n_rows'] == 2306
    assert calibration['score_boundary']['device'] == 'cpu'
    assert calibration['score_boundary']['dtype'] == 'torch.float64'
    for phase in ('base_epoch', 'grown_epoch', 'score'):
        measurements = calibration[phase]
        assert [m['batch_size'] for m in measurements] == [16, 32, 64, 128]
        assert all(len(m['samples_ns']) >= 5 for m in measurements)
        best = max(m['rows_per_second'] for m in measurements)
        expected = min(m['batch_size'] for m in measurements if m['rows_per_second'] >= .95 * best)
        assert calibration['selected_batches'][phase] == expected
    assert calibration['accumulation_max_abs'] < 1e-5
    assert backend_report['resources_before']['initial_reservation_bytes'] == 9 * 1024**3
    json.dumps(backend_report, allow_nan=False)


def test_unsupported_indispensable_operation_selects_whole_cpu(backend_report):
    from treeneuron.backend import select_backend

    cpu = copy.deepcopy(backend_report)
    cpu['device'] = 'cpu'
    other = copy.deepcopy(backend_report)
    other['device'] = 'mps'
    other['status'] = 'failed'
    other['operations']['index_copy']['status'] = 'failed'
    other['reason'] = 'deliberately unsupported index_copy'
    decision = select_backend({'cpu': cpu, 'mps': other})
    assert decision['selected_backend'] == 'cpu'
    assert decision['scoring_backend'] == 'cpu_float64'
    with pytest.raises(ValueError, match='CPU'):
        select_backend({'cpu': other})


def test_status_only_backend_report_is_not_kernel_evidence():
    from treeneuron.backend import validate_backend_report

    with pytest.raises(ValueError):
        validate_backend_report({'status': 'passed', 'device': 'cpu', 'mps_available': True})


def test_probe_preserves_parent_rng_dtype_threads_environment(backend_report):
    # The fixture uses a child; repeated direct calls must preserve parent state too.
    from treeneuron.doctor import probe_backend

    rng, threads, dtype = torch.random.get_rng_state(), torch.get_num_threads(), torch.get_default_dtype()
    env = dict(os.environ)
    report = probe_backend('invalid', seed=1, batch_sizes=(32, 64))
    assert report['status'] == 'failed'
    assert 'Unsupported device' in report['reason']
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads and torch.get_default_dtype() == dtype
    assert all(os.environ.get(key) == value for key, value in env.items())
    assert set(os.environ) == set(env)


def test_failed_diagnostic_preserves_sanitized_stdout(monkeypatch):
    from treeneuron import doctor

    monkeypatch.setattr(doctor.subprocess, 'run', lambda args, **kw:
                        subprocess.CompletedProcess(args, 1, 'permission denied\nSerial Number: SECRET', ''))
    report = doctor._probe(['diagnostic'])
    assert report['returncode'] == 1 and report['value'] is None
    assert 'permission denied' in report['reason']
    assert 'SECRET' not in json.dumps(report)


def test_whole_method_cost_is_directly_timed(backend_report):
    calibration = backend_report['calibration']
    complete = calibration['whole_method']
    assert calibration['whole_method_ns'] == complete['median_ns']
    assert len(complete['samples_ns']) >= 5
    assert calibration['whole_method_boundary'] == 'base epoch -> CPU64 copy/score -> matched growth -> grown epoch; fresh optimizer per phase'


@pytest.mark.parametrize('mutation', ['operation_error', 'missing_operation', 'batch_choice', 'fallback', 'hash', 'lock', 'environment'])
def test_corrupted_real_evidence_is_rejected(backend_report, mutation):
    from treeneuron.backend import validate_backend_report

    report = copy.deepcopy(backend_report)
    if mutation == 'operation_error':
        report['operations']['index_copy']['trials'][0]['max_normalized_error'] = 2
    elif mutation == 'missing_operation':
        del report['operations']['adamw_step']
    elif mutation == 'batch_choice':
        report['calibration']['selected_batches']['base_epoch'] = 999
    elif mutation == 'fallback':
        report['fallback_disabled_before_import'] = False
    elif mutation == 'hash':
        report['implementation_sha256'] = '0' * 64
    elif mutation == 'lock':
        report['lock_sha256'] = '0' * 64
    else:
        report['environment_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        validate_backend_report(report)


def test_resources_separate_current_rss_peak_rss_and_overlap(backend_report):
    for phase in ('resources_before', 'resources_after'):
        resources = backend_report[phase]
        if resources['rss_bytes']['value'] is None:
            assert resources['rss_bytes']['reason']
        else:
            assert resources['rss_bytes']['value'] > 0
            assert resources['peak_rss_bytes'] >= resources['rss_bytes']['value']
        assert resources['unified_memory_counters_overlap'] is True
        assert resources['swap_used_bytes']['value'] is None or resources['swap_used_bytes']['value'] >= 0
    assert isinstance(backend_report['resource_deltas']['swap_used_bytes'], (int, type(None)))


def test_optimized_python_still_rejects_failed_numeric_evidence(backend_report, tmp_path):
    import sys
    report = copy.deepcopy(backend_report)
    report['operations']['index_copy']['trials'][0]['max_normalized_error'] = 2
    artifact = tmp_path / 'corrupt.json'
    artifact.write_text(json.dumps(report))
    result = subprocess.run([sys.executable, '-O', '-c',
        'import json,sys; from treeneuron.backend import validate_backend_report; '
        'validate_backend_report(json.load(open(sys.argv[1])))', str(artifact)],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Invalid actual backend' in result.stderr


def test_resource_failure_retains_observations_and_unexecuted_operations():
    import sys
    code = '''import sys
import treeneuron.backend as b
def stop(_):
    raise RuntimeError('simulated resource pressure')
b._guard = stop
sys.argv = ['probe', '--device', 'cpu', '--seed', '1', '--batches', '64,128']
b.main()
'''
    result = subprocess.run([sys.executable, '-c', code],
        env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0'), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'failed' and 'simulated resource pressure' in report['reason']
    assert report['resources_before']['initial_reservation_bytes'] == 9 * 1024**3
    assert all(operation['status'] == 'not_run' for operation in report['operations'].values())


def test_balanced_backend_selection_uses_all_complete_path_blocks(backend_report, monkeypatch):
    from treeneuron import backend

    calls = []
    factors = [1, 4, 4, 9]  # CPU median5 vs MPS median4: final CPU block alone would mislead.
    def measured(device, **kwargs):
        report = copy.deepcopy(backend_report)
        report['device'] = device
        factor = factors[len(calls)]
        calls.append(device)
        whole = report['calibration']['whole_method']
        whole['samples_ns'] = [t * factor for t in whole['samples_ns']]
        whole['median_ns'] *= factor
        report['calibration']['whole_method_ns'] *= factor
        return report
    monkeypatch.setattr(backend, 'probe_backend', measured)
    aggregate = backend.collect_backend_report()
    assert calls == ['cpu', 'mps', 'mps', 'cpu']
    assert aggregate['decision']['selected_backend'] == 'mps'
    assert len(aggregate['comparison_blocks']) == 4
    assert aggregate['decision']['comparison_order'] == calls


def test_indispensable_kernel_exception_cannot_silently_mix_backends():
    import sys
    code = """import sys
import treeneuron.backend as b
original = b._run
def unsupported(*args, **kwargs):
    import torch
    def fail(*a, **kw):
        raise NotImplementedError('deliberately unsupported index_copy')
    torch.Tensor.index_copy = fail
    return original(*args, **kwargs)
b._run = unsupported
sys.argv = ['probe', '--device', 'cpu', '--seed', '1', '--batches', '64,128']
b.main()
"""
    result = subprocess.run([sys.executable, '-c', code],
        env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0'), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'failed'
    assert report['fallback_disabled_before_import'] is True
    assert report['operations']['index_copy']['status'] == 'failed'
    assert 'deliberately unsupported index_copy' in report['reason']
    assert report['operations']['layer_output']['status'] == 'not_run'


def test_guard_reserves_measured_first_use_runtime_overhead():
    from treeneuron.backend import _guard

    # Measured native first-use MPS RSS growth was ~250MiB, exceeding128MiB.
    resources = {'observations': {'pressure_level': {'value': '1'}},
                 'pressure_headroom_bytes': {'value': 9 * 1024**3 + 256 * 1024**2}}
    with pytest.raises(RuntimeError, match='reserve'):
        _guard(resources)
    resources['pressure_headroom_bytes']['value'] = 9 * 1024**3 + 512 * 1024**2
    _guard(resources)


@pytest.mark.parametrize('labels', [
    [1000, 2000, 4000, 8000], [False, True, True, True],
    [16., 32., 64., 128.], [16, 16, 32, 64], [3, 6, 12, 24],
    [32, 64, 128, 256], [0, 0, 0, 0], [-16, -32, -64, -128],
])
def test_consistent_out_of_contract_batch_labels_cannot_open_gate(backend_report, labels):
    from treeneuron.backend import select_backend, validate_backend_report

    report = copy.deepcopy(backend_report)
    original = report['batch_sizes']
    mapping = dict(zip(original, labels))
    report['batch_sizes'] = labels
    calibration = report['calibration']
    for phase in ('base_epoch', 'grown_epoch', 'score'):
        for row in calibration[phase]:
            row['batch_size'] = mapping[row['batch_size']]
        calibration['selected_batches'][phase] = mapping[calibration['selected_batches'][phase]]
    if 'physical_batches' in calibration['whole_method']:
        calibration['whole_method']['physical_batches'] = dict(calibration['selected_batches'])
    # A batch regression must fail without relying on unrelated source drift.
    assert report['implementation_sha256'] == backend_report['implementation_sha256']
    with pytest.raises(ValueError):
        validate_backend_report(report)
    with pytest.raises(ValueError, match='CPU'):
        select_backend({'cpu': dict(report, device='cpu')})


@pytest.mark.parametrize('location,value', [
    ('batch_list', None), ('batch_list', []), ('batch_list', '16,32,64,128'),
    ('stage_row', 128.), ('stage_row', True), ('stage_row', 8000),
    ('selected', 128.), ('selected', True), ('selected', 8000),
    ('logical', 128.), ('logical', True),
    ('complete_physical', 128.), ('complete_physical', True), ('complete_physical', 8000),
    ('complete_logical', 128.), ('complete_logical', True), ('complete_logical', 256),
    ('complete_sentinel', 8000),
])
def test_batch_contract_applies_at_each_artifact_boundary(backend_report, location, value):
    from treeneuron.backend import validate_backend_report

    report = copy.deepcopy(backend_report)
    calibration = report['calibration']
    if location == 'batch_list':
        report['batch_sizes'] = value
    elif location == 'stage_row':
        calibration['base_epoch'][-1]['batch_size'] = value
    elif location == 'selected':
        calibration['selected_batches']['base_epoch'] = value
    elif location == 'logical':
        calibration['logical_batch_size'] = value
    elif location == 'complete_physical':
        calibration['whole_method']['physical_batches'] = dict(calibration['selected_batches'], base_epoch=value)
    elif location == 'complete_logical':
        calibration['whole_method']['logical_batch_size'] = value
    else:
        calibration['whole_method']['batch_size'] = value
    with pytest.raises(ValueError):
        validate_backend_report(report)


@pytest.mark.parametrize('labels', [None, [], '16,32,64,128', {'batch': 16},
    [1000, 2000, 4000, 8000], [False, True], [64., 128.], [64, 64], [3, 6], [128, 256]])
def test_live_probe_rejects_malformed_physical_batches(labels):
    from treeneuron.backend import probe_backend

    report = probe_backend('cpu', seed=1602030, batch_sizes=labels)
    assert report['status'] == 'failed'
    assert 'batch' in report['reason'].lower()


@pytest.mark.parametrize('location,recipe', [
    ('selected_batches', None), ('selected_batches', {}),
    ('selected_batches', {'base_epoch': 128, 'grown_epoch': 128, 'score': 128, 'extra': 128}),
    ('physical_batches', None), ('physical_batches', {}),
    ('physical_batches', {'base_epoch': 128, 'grown_epoch': 128, 'score': 128, 'extra': 128}),
])
def test_selected_and_complete_recipes_require_exact_phase_keys(backend_report, location, recipe):
    from treeneuron.backend import validate_backend_report

    report = copy.deepcopy(backend_report)
    target = report['calibration'] if location == 'selected_batches' else report['calibration']['whole_method']
    target[location] = recipe
    with pytest.raises(ValueError):
        validate_backend_report(report)
