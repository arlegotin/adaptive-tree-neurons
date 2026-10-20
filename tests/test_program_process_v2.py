"""Owned outer clocks never substitute for actual program qualification."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.program_process_v2'), 'Missing owned program process envelope'
    from treeneuron.research import program_process_v2
    return program_process_v2


def arguments(root,tranche):
    value=dict(qualification_reference=dict(root=str(root/'false-registry'),inventory_sha256='a'*64),
        resource_forecast_reference=dict(path=str(root/'forbidden-forecast'),sha256='b'*64),
        directory=str(root/'uncreated-program'),rss_limit_bytes=4*1024**3,
        expected_journal_sha256=None,max_episode_starts=1)
    if tranche=='primary':value.update(practical_plan_reference=dict(path=str(root/'forbidden-plan'),sha256='c'*64),
        expected_practical_plan_hash='d'*64,support_reference=dict(path=str(root/'forbidden-support'),sha256='e'*64),
        procedure_reference=dict(path=str(root/'forbidden-procedure'),sha256='f'*64))
    else:value['barrier_reference']=dict(path=str(root/'forbidden-global'),sha256='c'*64)
    return value


def test_public_runner_rejects_false_registry_before_control_or_program_output(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        api().run_program_process(tranche='primary',arguments=arguments(tmp_path,'primary'),output=tmp_path/'control',
            admission_identity=dict(policy='software'),admission_check=lambda:dict(admitted=True))
    assert 'forbidden-plan' not in str(error.value) and list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('fault',['extra','tranche','source','rss_boolean','limit_boolean','relative','labels','journal'])
def test_strict_program_request_has_no_undeclared_role_or_clock_authority(tmp_path,fault):
    module=api();request=module._make_request('primary',arguments(tmp_path,'primary'),time.perf_counter_ns())
    if fault=='extra':request['quality_eligible']=True
    elif fault=='tranche':request['tranche']='test'
    elif fault=='source':request['source_sha256']=True
    elif fault=='rss_boolean':request['arguments']['rss_limit_bytes']=True
    elif fault=='limit_boolean':request['arguments']['max_episode_starts']=True
    elif fault=='relative':request['arguments']['directory']='relative'
    elif fault=='labels':request['arguments']['V_labels_path']=str(tmp_path/'labels')
    else:request['arguments']['expected_journal_sha256']=True
    with pytest.raises(ValueError):module._validate_request(request)
    assert list(tmp_path.iterdir())==[]


@pytest.fixture(scope='module')
def episodes(tmp_path_factory):
    module=api();root=tmp_path_factory.mktemp('owned-program-process');rows={}
    for tranche,admitted in [('primary',True),('secondary',True),('primary',False)]:
        name=tranche if admitted else 'denied';output=root/name
        origin=time.perf_counter_ns();request=module._make_request(tranche,arguments(root,tranche),origin)
        result=module._run_program_envelope(request=request,output=output,
            admission_identity=dict(policy='software-owned-false-qualification'),admission_check=lambda:dict(admitted=admitted),
            root_rss_pause_bytes=512*1024**2)
        rows[name]=dict(root=output,result=result,request=request)
    return rows


@pytest.mark.parametrize('tranche',['primary','secondary'])
def test_actual_cold_program_worker_records_failed_gate_before_any_plan_or_data(episodes,tranche):
    row=episodes[tranche];value=row['result'];root=row['root']
    assert value['returncode']==1 and value['worker_protocol_closed'] is True
    body=json.loads((root/'process/body-outcome.json').read_bytes())
    assert body['status']=='program_failed' and body['program_invocation_references']==[]
    assert 'forbidden-' not in body['traceback']
    assert not Path(row['request']['arguments']['directory']).exists()
    assert body['T_accessed'] is body['protected_access_authorized'] is body['whole_training_procedure_cost_authenticated'] is False


def test_closed_envelope_reader_includes_worker_return_and_starts_no_process(episodes,monkeypatch):
    module=api()
    def forbidden(*a,**kw):raise AssertionError('Closed reader cannot spawn or probe saved PIDs')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    for row in episodes.values():
        proof=module.inspect_program_process(row['result']['envelope_reference'])
        assert proof['elapsed_ns']==proof['end_ns']-proof['origin_ns']>0
        assert proof['nested_program_closure_authenticated'] is proof['whole_training_procedure_cost_authenticated'] is False
        assert proof['protected_access_authorized'] is proof['T_accessed'] is False
    assert episodes['denied']['result']['child_started'] is False


@pytest.mark.parametrize('fault',['hash','source','domain','origin','authority','request','command'])
def test_closed_envelope_rejects_altered_operational_evidence(episodes,tmp_path,fault):
    module=api();row=episodes['primary'];reference=copy.deepcopy(row['result']['envelope_reference'])
    path=Path(reference['path']);before=path.read_bytes();value=json.loads(before)
    if fault=='source':value['source_sha256']='0'*64
    elif fault=='domain':value['clock_domain']['boot_sha256']=True
    elif fault=='origin':value['origin_ns']+=1
    elif fault=='authority':value['whole_training_procedure_cost_authenticated']=True
    elif fault=='request':value['request_reference']['sha256']='0'*64
    elif fault=='command':value['argv']=['/bin/echo']
    raw=json.dumps(value).encode();path.write_bytes(raw)
    reference['sha256']='0'*64 if fault=='hash' else hashlib.sha256(raw).hexdigest()
    try:
        with pytest.raises(ValueError):module.inspect_program_process(reference)
    finally:path.write_bytes(before)


def test_cold_module_does_not_import_numerical_or_search_stack():
    api()
    code="import treeneuron.research.program_process_v2; import sys; assert not {'torch','numpy','yaml','pandas','optuna'}.intersection(sys.modules)"
    result=subprocess.run([sys.executable,'-P','-B','-c',code],capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr


def test_outer_control_paths_preserve_program_working_proof_and_interpreter_roots(tmp_path):
    module=api();request=module._make_request('primary',arguments(tmp_path,'primary'),time.perf_counter_ns())
    working=tmp_path/'data';proof=tmp_path/'qualification';working.mkdir();proof.mkdir()
    for output in [working/'control',proof/'control',Path(request['arguments']['directory'])/'control',
            Path(sys.prefix)/'control',Path(sys.executable).resolve().parent/'control']:
        with pytest.raises(ValueError):module._operational_location(output,request,working_roots=[str(working)],proof_roots=[str(proof)])
        assert not output.exists()
    accepted=tmp_path/'separate-control'
    assert module._operational_location(accepted,request,working_roots=[str(working)],proof_roots=[str(proof)])==accepted
    assert not accepted.exists()


def test_changed_source_rejects_private_composition_before_operational_output(tmp_path):
    module=api();request=module._make_request('primary',arguments(tmp_path,'primary'),time.perf_counter_ns())
    request['source_sha256']='0'*64
    with pytest.raises(ValueError):module._run_program_envelope(request=request,output=tmp_path/'control',
        admission_identity=dict(policy='software'),admission_check=lambda:dict(admitted=True))
    assert list(tmp_path.iterdir())==[]
