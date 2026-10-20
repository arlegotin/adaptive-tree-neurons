"""Reopen actual geometry evidence without granting native-quality authority."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from test_native_support_probes import plan_for
from treeneuron.config import canonical_hash
from treeneuron.research.method_worker import _sha
from treeneuron.research.workqueue import _inventory


def api():
    from treeneuron.research import support_probes_v2
    assert hasattr(support_probes_v2, 'load_support_probe'), 'Missing closed support-geometry evidence reader'
    return support_probes_v2


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')


def reference(root):
    return dict(root=str(root), inventory_sha256=canonical_hash(_inventory(root)))


def prepare(tmp_path, family='P13', *, stopped=False, extra_pythonpath=()):
    module = api(); plan = plan_for(module, tmp_path, family)
    case = next(c for c in plan['cases'] if
        c['native_config'].get('estimator') == 'ridge') if family == 'P13' else plan['cases'][0]
    output = (tmp_path/'probe').resolve()
    module.run_support_probe(plan=plan, case_id=case['case_id'], output=output,
        interpreter=sys.executable, rss_limit_bytes=1 if stopped else 4*1024**3,
        extra_pythonpath=extra_pythonpath)
    return module, plan, case, output


def read(module, plan, case, root, ref=None):
    return module.load_support_probe(ref or reference(root),
        prepared_root=plan['prepared_root'], expected_manifest_sha256=plan['manifest_sha256'],
        expected_family=plan['family_id'], expected_case_id=case['case_id'])


def update_terminal(root, terminal):
    write(root/'worker/terminal.json', terminal)
    result = json.loads((root/'result.json').read_text())
    result['worker_terminal_sha256'] = _sha(root/'worker/terminal.json')
    write(root/'result.json', result)


def test_actual_ridge_proof_reopens_rows_and_managed_receipts_without_fit_predict_or_V_labels(tmp_path, monkeypatch):
    module, plan, case, root = prepare(tmp_path)
    before = _inventory(root); prepared = Path(plan['prepared_root'])
    access = (prepared/'access.jsonl').read_text().splitlines()
    original = Path.read_bytes
    forbidden_paths = {prepared/'outer-0/V.npz', prepared/'outer-0/T.npz'}
    def bytes_without_held_out_labels(path):
        assert path not in forbidden_paths, 'Evidence reader opened held-out labels'
        return original(path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Evidence reader must not launch fitting or inference')
    monkeypatch.setattr(Path, 'read_bytes', bytes_without_held_out_labels)
    monkeypatch.setattr(module, 'supervise_process', forbidden)
    result = read(module, plan, case, root)
    assert result['status'] == 'authenticated_short_geometry_evidence'
    assert result['family_id'] == 'P13' and result['case_id'] == case['case_id']
    actual_rows = {}
    for role in ('F', 'G', 'V_features'):
        with np.load(root/'input'/f'{role}.npz', allow_pickle=False) as saved:
            actual_rows[role] = len(saved['row_id'])
    assert result['complete_input_rows'] == actual_rows
    assert result['query_predictions_finite'] is result['managed_snapshot_reload_record_checked'] is True
    for key in ('full_native_training', 'native_workload_qualified', 'quality_eligible',
            'independent_new_inference', 'HPO_journal_authenticated', 'protected_access_authorized',
            'V_labels_read', 'T_accessed'):
        assert result[key] is False
    assert _inventory(root) == before
    new_events = (prepared/'access.jsonl').read_text().splitlines()[len(access):]
    assert new_events and all(json.loads(e)['roles'] == ['F_labels', 'G_labels', 'V_features'] for e in new_events)


def test_actual_wide_ordinary_probe_stays_explicitly_short_and_unqualified(tmp_path):
    module, plan, case, root = prepare(tmp_path, 'P02')
    result = read(module, plan, case, root)
    assert result['native_configuration'] == case['native_config']
    assert result['native_configuration']['parameters']['width'] == 128
    assert result['engineering_configuration']['parameters']['width'] == 128
    assert result['engineering_configuration']['base_updates'] == 2
    assert result['all_training_rows_visited_authenticated'] is False
    assert result['full_native_training'] is result['native_workload_qualified'] is False


def test_actual_full_NODE_geometry_retains_explicit_checkpoint_and_short_budget_path(tmp_path):
    module, plan, case, root = prepare(tmp_path, 'P12')
    result = read(module, plan, case, root)
    assert result['native_configuration']['parameters']['trees'] == 2048
    assert result['native_configuration']['parameters']['depth'] == 6
    assert result['budget'] == dict(max_updates=(result['complete_input_rows']['F']+511)//512+1)
    assert (root/'training-checkpoints').is_dir()
    assert result['full_native_training'] is result['quality_eligible'] is False


def test_actual_linear_tree_geometry_reopens_its_state_json_contract(tmp_path):
    overlay = os.environ.get('ATN_TREE_OVERLAY')
    if not overlay: pytest.skip('Requires the separately pinned native tree overlay')
    module, plan, case, root = prepare(tmp_path, 'P11', extra_pythonpath=(Path(overlay),))
    result = read(module, plan, case, root)
    assert (root/'worker/snapshot/state.json').is_file()
    assert not (root/'worker/snapshot/manifest.json').exists()
    assert result['engineering_configuration']['max_rounds'] == 2
    assert result['full_native_training'] is result['quality_eligible'] is False


@pytest.mark.parametrize('change', ['source', 'case', 'geometry', 'budget', 'authority',
    'argv', 'threads', 'exit', 'deadline', 'process_authority', 'rss_peak',
    'row_ids', 'nonfinite', 'snapshot_pointer', 'snapshot_sidecar', 'worker_request_pointer'])
def test_recommitted_but_semantically_wrong_geometry_proof_is_rejected(tmp_path, change):
    module, plan, case, root = prepare(tmp_path)
    result = json.loads((root/'result.json').read_text())
    process = json.loads((root/'process/process.json').read_text())
    terminal = json.loads((root/'worker/terminal.json').read_text())
    if change == 'source':
        result['source_sha256'] = '0'*64
    elif change == 'case':
        result['case_id'] = 'P13-geometry-99'
    elif change == 'geometry':
        result['probe']['configuration']['parameters']['alpha'] = 1.
    elif change == 'budget':
        result['probe']['budget'] = dict(max_updates=1)
    elif change == 'authority':
        result['native_workload_qualified'] = True
    elif change in ('argv', 'threads', 'exit', 'deadline', 'process_authority', 'rss_peak'):
        if change == 'argv': process['argv'][-1] = str(root/'different-worker')
        elif change == 'threads': process['environment_overrides']['OMP_NUM_THREADS'] = '2'
        elif change == 'exit': process['returncode'] = 9
        elif change == 'deadline': process['timeout_ns'] = 1000000000
        elif change == 'process_authority': process['protected_access_authorized'] = True
        else: process['observed_peak_root_rss_bytes'] = 1
        write(root/'process/process.json', process)
        result['process_receipt_sha256'] = _sha(root/'process/process.json')
    else:
        if change == 'snapshot_pointer':
            terminal['snapshot_commitment']['sha256'] = '0'*64
        elif change == 'snapshot_sidecar':
            terminal['snapshot_manifest_sha256'] = '0'*64
        elif change == 'worker_request_pointer':
            terminal['request_path'] = str(root/'different-request.json')
        else:
            path = root/'worker/predictions.npz'
            with np.load(path, allow_pickle=False) as saved:
                arrays = {k:saved[k].copy() for k in saved.files}
            if change == 'row_ids': arrays['row_id'] = arrays['row_id'][::-1]
            else: arrays['prediction'][0] = np.nan
            np.savez(path, **arrays)
            for row in terminal['files']:
                if row['path'] == 'predictions.npz':
                    row.update(sha256=_sha(path), bytes=path.stat().st_size)
        update_terminal(root, terminal)
        result['worker_terminal_sha256'] = _sha(root/'worker/terminal.json')
    write(root/'result.json', result)
    with pytest.raises(ValueError): read(module, plan, case, root)


def test_stale_external_inventory_or_changed_original_roles_are_rejected(tmp_path):
    module, plan, case, root = prepare(tmp_path)
    ref = reference(root)
    with (root/'worker/predictions.npz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError): read(module, plan, case, root, ref)
    # Also fails when someone recommits the outer inventory instead of repairing
    # the original worker/role evidence.
    with pytest.raises(ValueError): read(module, plan, case, root)


def test_late_original_role_change_cannot_return_checked_flags(tmp_path, monkeypatch):
    module, plan, case, root = prepare(tmp_path)
    original = module.EngineeringTrialBridge._worker_terminal
    def after_terminal(*args, **kwargs):
        terminal = original(*args, **kwargs)
        with (Path(plan['prepared_root'])/'outer-0/G.npz').open('ab') as stream:
            stream.write(b'controlled late original-role change')
        return terminal
    monkeypatch.setattr(module.EngineeringTrialBridge, '_worker_terminal', after_terminal)
    with pytest.raises(ValueError): read(module, plan, case, root)


def test_owned_resource_stop_is_retained_but_cannot_be_a_passed_geometry_proof(tmp_path):
    module, plan, case, root = prepare(tmp_path, stopped=True)
    assert json.loads((root/'result.json').read_text())['status'] == 'resource_exhausted'
    with pytest.raises(ValueError): read(module, plan, case, root)


def test_final_zero_RSS_exit_sample_is_valid_when_positive_observations_are_retained(tmp_path):
    module, plan, case, root = prepare(tmp_path)
    process_path = root/'process/process.json'
    process = json.loads(process_path.read_text())
    samples_path = root/'process/resource-samples.jsonl'
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    # /bin/ps can report zero for an exiting worker before poll observes exit.
    # Preserve its genuine positive samples and add that allowed transition.
    assert samples[-1]['elapsed_ns'] < process['elapsed_ns']-1
    samples.append(dict(elapsed_ns=process['elapsed_ns']-1, root_rss_bytes=0))
    samples_path.write_text(''.join(json.dumps(s)+'\n' for s in samples))
    process['rss_sample_count'] = len(samples)
    for ref in process['files']:
        if ref['path'] == 'resource-samples.jsonl':
            ref.update(sha256=_sha(samples_path), bytes=samples_path.stat().st_size)
    write(process_path, process)
    result = json.loads((root/'result.json').read_text())
    result['process_receipt_sha256'] = _sha(process_path)
    write(root/'result.json', result)
    assert read(module, plan, case, root)['status'] == 'authenticated_short_geometry_evidence'


def test_mutating_the_callers_reference_does_not_change_the_authenticated_returned_reference(tmp_path, monkeypatch):
    module, plan, case, root = prepare(tmp_path)
    ref = reference(root); committed = dict(ref)
    original = module.EngineeringTrialBridge._worker_terminal
    def mutate_reference_after_input_authentication(*args, **kwargs):
        result = original(*args, **kwargs)
        ref['inventory_sha256'] = '0'*64
        return result
    monkeypatch.setattr(module.EngineeringTrialBridge, '_worker_terminal', mutate_reference_after_input_authentication)
    assert read(module, plan, case, root, ref)['reference'] == committed
