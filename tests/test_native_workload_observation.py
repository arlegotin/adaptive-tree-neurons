"""Recorded native procedures need workload checks beyond exit/reload flags."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from treeneuron.config import canonical_hash
from test_native_generation_chain import actual,rebind_native


def api():
    assert importlib.util.find_spec('treeneuron.research.native_workload_v2') is not None, 'Missing native workload observation'
    from treeneuron.research import native_workload_v2
    return native_workload_v2


def actual_estimator(tmp_path,monkeypatch,estimator):
    args,attempt=actual(tmp_path,monkeypatch)
    if estimator=='mean':return args,attempt
    from treeneuron.research.native_fit_v2 import run_native_fit
    variant=tmp_path/estimator;inputs=variant/'input';inputs.mkdir(parents=True)
    request=json.loads(args['request_path'].read_text());request['job_id']='workload-fixture/'+estimator
    request['config']['estimator']=estimator
    request['config']['parameters']=dict(alpha=1.,**({'n_knots':4} if estimator=='spline' else {}))
    for ref in request['files'].values():shutil.copyfile(args['request_path'].parent/ref['path'],inputs/ref['path'])
    path=inputs/'request.json';path.write_text(json.dumps(request));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    native=run_native_fit(path,expected_request_sha256=digest,output=variant/'native',interpreter=sys.executable)
    assert native['status']=='completed_ineligible'
    return dict(args,request_path=path,expected_request_sha256=digest,native_reference=native['recovery_reference'],
        expected_configuration=request['config']),variant


@pytest.mark.parametrize('estimator',['mean','ridge','spline'])
def test_actual_closed_form_workload_checks_full_F_identity_and_process_observations_without_fit_or_V_labels(tmp_path,monkeypatch,estimator):
    module=api();args,attempt=actual_estimator(tmp_path,monkeypatch,estimator)
    from treeneuron.research.p13_simple import P13Predictor
    def forbidden(*a,**kw):raise AssertionError('Read-only workload observation must not fit, predict or solve')
    monkeypatch.setattr(P13Predictor,'fit',forbidden);monkeypatch.setattr(P13Predictor,'predict',forbidden)
    monkeypatch.setattr(np.linalg,'lstsq',forbidden)
    result=module.inspect_native_workload(**args)
    with np.load(args['request_path'].parent/'F.npz',allow_pickle=False) as saved:nF=len(saved['y'])
    assert result['status']=='checked_recorded_native_workload' and result['family_id']=='P13'
    assert result['input_rows']['F']==nF and result['training']['F_context_rows']==nF
    assert result['training']['procedure']=='closed_form' and result['training']['iterative_gradients']=='not_applicable'
    assert result['training']['finite_coefficient_payload_checked'] is True
    assert result['process_observation_recomputed'] is True
    assert result['native_workload_qualified'] is result['quality_eligible'] is result['T_accessed'] is False
    access=[json.loads(s) for s in (Path(args['prepared_root'])/'access.jsonl').read_text().splitlines()]
    assert access and all(x['roles']==['F_labels','G_labels','V_features'] for x in access)


def test_self_recommitted_understated_RSS_peak_cannot_supply_a_workload_observation(tmp_path,monkeypatch):
    module=api();args,attempt=actual(tmp_path,monkeypatch)
    process_path=attempt/'native/process/process.json';process=json.loads(process_path.read_text())
    assert process['observed_peak_root_rss_bytes']>1
    process['observed_peak_root_rss_bytes']-=1;process_path.write_text(json.dumps(process))
    native=attempt/'native/result.json';value=json.loads(native.read_text())
    value['process_receipt_sha256']=hashlib.sha256(process_path.read_bytes()).hexdigest();native.write_text(json.dumps(value))
    rebind_native(args,attempt)
    with pytest.raises(ValueError,match='RSS'):module.inspect_native_workload(**args)


def test_proposal_capture_survives_a_late_change_to_the_callers_mutable_configuration(tmp_path,monkeypatch):
    module=api();args,attempt=actual(tmp_path,monkeypatch);original=module.audit_native_generation
    expected=canonical_hash(copy.deepcopy(args['expected_configuration']));calls=0
    def observe_then_change_caller(**values):
        nonlocal calls
        result=original(**values);calls+=1
        if calls==1:args['expected_configuration']['estimator']='not-a-native-estimator'
        return result
    monkeypatch.setattr(module,'audit_native_generation',observe_then_change_caller)
    result=module.inspect_native_workload(**args)
    assert result['configuration_sha256']==expected
    assert args['expected_configuration']['estimator']=='not-a-native-estimator'


def test_unknown_family_rejects_before_reading_any_training_artifacts(tmp_path,monkeypatch):
    module=api()
    def forbidden(**kwargs):raise AssertionError('Unknown family must fail before generation/data access')
    monkeypatch.setattr(module,'audit_native_generation',forbidden)
    with pytest.raises(ValueError,match='family'):
        module.inspect_native_workload(request_path=tmp_path/'absent',expected_request_sha256='0'*64,
            native_reference={},prepared_root=tmp_path/'absent-data',expected_manifest_sha256='0'*64,
            outer_fold=0,expected_fit_seed=17,expected_configuration={'family':'P14'})


def recommit_closed_records(args,attempt):
    """Controlled content faults remain consistent at the outer byte level."""
    root=attempt/'native';worker=root/'worker'
    terminal=json.loads((worker/'terminal.json').read_text())
    for ref in terminal['files']:
        p=worker/ref['path'];ref.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    metadata=worker/terminal['snapshot_commitment']['path'];digest=hashlib.sha256(metadata.read_bytes()).hexdigest()
    terminal['snapshot_commitment']['sha256']=digest
    terminal['snapshot_state_sha256' if metadata.name=='state.json' else 'snapshot_manifest_sha256']=digest
    (worker/'terminal.json').write_text(json.dumps(terminal))
    process=json.loads((root/'process/process.json').read_text())
    for ref in process['files']:
        p=root/'process'/ref['path'];ref.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (root/'process/process.json').write_text(json.dumps(process))
    value=json.loads((root/'result.json').read_text())
    value['worker_terminal_sha256']=hashlib.sha256((worker/'terminal.json').read_bytes()).hexdigest()
    value['process_receipt_sha256']=hashlib.sha256((root/'process/process.json').read_bytes()).hexdigest()
    (root/'result.json').write_text(json.dumps(value));rebind_native(args,attempt)


@pytest.mark.parametrize('fault',['nonfinite_coefficient','false_array_shape','boolean_basis_width','boolean_array_shape'])
def test_self_recommitted_bad_closed_form_payload_is_not_a_checked_workload(tmp_path,monkeypatch,fault):
    module=api();args,attempt=actual(tmp_path,monkeypatch)
    root=attempt/'native/worker/snapshot';metadata=json.loads((root/'manifest.json').read_text())
    if fault=='false_array_shape':metadata['arrays']['coefficients']['shape']=[metadata['basis_width']+1]
    elif fault=='boolean_basis_width':metadata['basis_width']=True
    elif fault=='boolean_array_shape':metadata['arrays']['coefficients']['shape']=[True]
    else:
        from treeneuron.research.p13_simple import _array_identity
        with np.load(root/'arrays.npz',allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
        arrays['coefficients'][0]=np.nan;np.savez(root/'arrays.npz',**arrays)
        metadata['array_file']['sha256']=hashlib.sha256((root/'arrays.npz').read_bytes()).hexdigest()
        metadata['arrays']['coefficients']=_array_identity(arrays['coefficients'])
    (root/'manifest.json').write_text(json.dumps(metadata));recommit_closed_records(args,attempt)
    with pytest.raises(ValueError,match='array|coefficient|payload|geometry'):module.inspect_native_workload(**args)


def test_real_exit_transition_zero_RSS_sample_does_not_invalidate_positive_observation(tmp_path,monkeypatch):
    module=api();args,attempt=actual(tmp_path,monkeypatch)
    root=attempt/'native/process';value=json.loads((root/'process.json').read_text())
    samples=root/'resource-samples.jsonl';last=json.loads(samples.read_text().splitlines()[-1])
    assert value['elapsed_ns']-last['elapsed_ns']>1
    with samples.open('a') as stream:stream.write(json.dumps(dict(elapsed_ns=value['elapsed_ns']-1,root_rss_bytes=0))+'\n')
    value['rss_sample_count']+=1;(root/'process.json').write_text(json.dumps(value));recommit_closed_records(args,attempt)
    result=module.inspect_native_workload(**args)
    assert result['process_observation']['sample_count']==value['rss_sample_count']
    assert result['process_observation']['observed_peak_root_rss_bytes']>0


def test_self_recommitted_unowned_process_start_is_rejected(tmp_path,monkeypatch):
    module=api();args,attempt=actual(tmp_path,monkeypatch)
    path=attempt/'native/process/spawned.json';value=json.loads(path.read_text());value['new_session']=False
    path.write_text(json.dumps(value));recommit_closed_records(args,attempt)
    with pytest.raises(ValueError,match='process'):module.inspect_native_workload(**args)
