"""Owned cold negative integration: never a successful native qualifier."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


def api():
    from treeneuron.research import program_process_v2
    return program_process_v2


def arguments(root):
    return dict(programme_reference=dict(path=str(root/'forbidden-programme'),sha256='a'*64),
        protocol_reference=dict(path=str(root/'forbidden-protocol'),sha256='b'*64),
        models_reference=dict(path=str(root/'forbidden-models'),sha256='c'*64),
        global_barrier_reference=dict(path=str(root/'forbidden-global'),sha256='d'*64),
        qualification_reference=dict(root=str(root/'false-registry'),inventory_sha256='e'*64),
        preT_costs_reference=dict(path=str(root/'forbidden-costs'),sha256='f'*64),
        readiness_reference=dict(path=str(root/'forbidden-readiness'),sha256='1'*64),
        directory=str(root/'forbidden-programme-root'),max_step_starts=1)


@pytest.mark.parametrize('limit',[None,2])
def test_confirmation_worker_preserves_cli_request_after_known_metadata_programme_returns(tmp_path,monkeypatch,limit):
    # Exercise the real main/metadata/checksum/history code, with only the
    # ownership context and expensive protected operation replaced.  The
    # callback writes seven known inert JSON files, grants no qualification,
    # and never reads labels, fits or predicts a model.
    from contextlib import nullcontext
    from test_confirm_programme_v2 import fixture,known_operations
    from treeneuron.research import cooperative_process_v2,primary_search_worker_v2
    programme,plan,reference=fixture(tmp_path/'known')
    execute,resolve=known_operations(plan);module=api()
    def known_run(bound_reference,**kwargs):
        assert bound_reference==reference
        assert kwargs['protocol_reference']==plan['protocol_reference']
        return programme._run_fixed_steps(reference,plan,execute_step=execute,
            resolve_completed=resolve,guard=lambda:None,max_step_starts=kwargs['max_step_starts'])
    monkeypatch.setattr(programme,'run_practical_confirmation',known_run)
    monkeypatch.setattr(cooperative_process_v2,'cooperative_worker',lambda identity:nullcontext())
    for key,value in primary_search_worker_v2._episode_environment().items():
        if value is None:monkeypatch.delenv(key,raising=False)
        else:monkeypatch.setenv(key,value)
    process=tmp_path/'known-process';process.mkdir()
    monkeypatch.setenv('ATN_SEARCH_CONTROL_ROOT',str(process))
    request=module._make_request('confirmation',dict(programme_reference=reference,
        protocol_reference=plan['protocol_reference'],directory=plan['directory'],
        max_step_starts=limit,**plan['inputs']),time.perf_counter_ns())
    path=tmp_path/'request.json';path.write_text(json.dumps(request))
    assert module.main(['--request',str(path),'--expected-request-sha256',hashlib.sha256(path.read_bytes()).hexdigest()])==0
    outcome=json.loads((process/'body-outcome.json').read_bytes())
    assert outcome['status']=='program_returned' and outcome['error_type'] is None
    assert outcome['program_progress']['status']==('completed' if limit is None else 'operator_pause')
    assert len(outcome['program_invocation_references'])==1
    assert outcome['native_qualification_authenticated'] is outcome['protected_access_authorized'] is False


def test_confirmation_request_names_only_fixed_programme_and_frozen_inputs(tmp_path):
    m=api();value=m._make_request('confirmation',arguments(tmp_path),time.perf_counter_ns())
    assert value['tranche']=='confirmation' and value['arguments']['max_step_starts']==1
    assert not tmp_path.exists() or list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('fault',['labels','unknown','boolean_limit','other_tranche'])
def test_confirmation_request_does_not_accept_role_paths_or_unbounded_stage_selection(tmp_path,fault):
    m=api();value=m._make_request('confirmation',arguments(tmp_path),time.perf_counter_ns())
    if fault=='labels':value['arguments']['T_labels_path']=str(tmp_path/'labels')
    elif fault=='unknown':value['arguments']['selector']='different-model'
    elif fault=='boolean_limit':value['arguments']['max_step_starts']=True
    else:value['tranche']='confirmation-retry-with-new-seed'
    with pytest.raises(ValueError):m._validate_request(value)


def test_public_confirmation_observer_requires_actual_registry_before_any_output(tmp_path):
    m=api()
    with pytest.raises((ValueError,FileNotFoundError)) as caught:
        m.run_program_process(tranche='confirmation',arguments=arguments(tmp_path),output=tmp_path/'observer',
            admission_identity=dict(purpose='software-only'),admission_check=lambda:dict(admitted=True))
    assert 'forbidden-' not in str(caught.value) and list(tmp_path.iterdir())==[]


@pytest.fixture(scope='module')
def episodes(tmp_path_factory):
    m=api();root=tmp_path_factory.mktemp('cold-confirmation-process');rows={}
    for admitted in (False,True):
        output=root/('granted' if admitted else 'denied')
        request=m._make_request('confirmation',arguments(root),time.perf_counter_ns())
        result=m._run_program_envelope(request=request,output=output,admission_identity=dict(purpose='local-negative-fixture-only'),
            admission_check=lambda:dict(admitted=admitted),root_rss_pause_bytes=1024**3)
        rows[admitted]=(output,request,result)
    return rows


def test_granted_cold_worker_stops_at_genuine_native_gate_without_programme_or_target_read(episodes):
    root,request,result=episodes[True]
    assert result['child_started'] and result['worker_protocol_closed'] and result['returncode']==1
    body=json.loads((root/'process/body-outcome.json').read_text())
    assert body['status']=='program_failed' and body['program_invocation_references']==[]
    assert body['T_accessed'] is None and body['protected_access_authorized'] is False
    assert 'forbidden-' not in body['traceback'] and not Path(request['arguments']['directory']).exists()


def test_denied_confirmation_starts_no_worker_and_preserves_explicit_no_access(episodes):
    root,request,result=episodes[False]
    assert result['child_started'] is False and not (root/'process/body-outcome.json').exists()
    proof=api().inspect_program_process(result['envelope_reference'])
    assert proof['T_accessed'] is False and proof['protected_access_authorized'] is False


def test_cold_confirmation_readback_is_readonly_and_does_not_certify_target_access_from_clock(episodes,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('Read-only cold envelope check started a process')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    root,request,result=episodes[True];proof=api().inspect_program_process(result['envelope_reference'])
    assert proof['T_accessed'] is None and proof['whole_training_procedure_cost_authenticated'] is False
    assert proof['elapsed_ns']>0 and proof['worker_protocol_closed']


def test_confirmation_failure_marker_reopens_exact_original_invocation_without_qualifying_it(tmp_path):
    from test_confirm_programme_v2 import fixture,known_operations
    module,plan,reference=fixture(tmp_path/'known');execute,resolve=known_operations(plan)
    def fail(step):raise FloatingPointError('Known metadata failure only')
    with pytest.raises(FloatingPointError) as caught:module._run_fixed_steps(reference,plan,execute_step=fail,resolve_completed=resolve,guard=lambda:None)
    args=dict(plan['inputs'],programme_reference=reference,protocol_reference=plan['protocol_reference'],directory=plan['directory'],max_step_starts=1)
    m=api();request=m._make_request('confirmation',args,time.perf_counter_ns())
    refs=m._failed_program_invocation_references(caught.value,request)
    assert refs==[caught.value.confirmation_invocation_reference]
    assert m._failed_program_invocation_references(ValueError('before programme'),request)==[]


def test_unowned_confirmation_cli_cannot_run_the_public_programme(tmp_path):
    m=api();value=m._make_request('confirmation',arguments(tmp_path),time.perf_counter_ns())
    path=tmp_path/'request.json';path.write_text(json.dumps(value)+'\n');sha=hashlib.sha256(path.read_bytes()).hexdigest()
    result=subprocess.run([sys.executable,'-P','-B','-m','treeneuron.research.program_process_v2','--request',str(path),
        '--expected-request-sha256',sha],capture_output=True,text=True,timeout=15)
    assert result.returncode!=0 and 'forbidden-' not in result.stderr
    assert not Path(value['arguments']['directory']).exists()


def test_rehashed_owned_body_cannot_substitute_unrelated_confirmation_invocation(episodes,tmp_path):
    root,request,result=episodes[True];body_path=root/'process/body-outcome.json';original=body_path.read_bytes()
    receipt_path=root/'process/process.json';old_receipt=receipt_path.read_bytes();envelope_path=Path(result['envelope_reference']['path']);old_envelope=envelope_path.read_bytes()
    def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    marker=tmp_path/'unrelated-end.json';marker.write_text('{"not_this_programme":true}\n')
    try:
        body=json.loads(original);body['program_invocation_references']=[ref(marker)];body_path.write_text(json.dumps(body)+'\n')
        receipt=json.loads(old_receipt)
        for entry in receipt['files']:
            path=root/'process'/entry['path'];entry.update(bytes=path.stat().st_size,sha256=ref(path)['sha256'])
        receipt_path.write_text(json.dumps(receipt)+'\n');envelope=json.loads(old_envelope);envelope['body_reference']=ref(body_path)
        envelope['process_reference']['process_receipt_sha256']=ref(receipt_path)['sha256'];envelope_path.write_text(json.dumps(envelope)+'\n')
        with pytest.raises((ValueError,FileNotFoundError)):api().inspect_program_process(ref(envelope_path))
    finally:body_path.write_bytes(original);receipt_path.write_bytes(old_receipt);envelope_path.write_bytes(old_envelope)
