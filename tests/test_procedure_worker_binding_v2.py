"""Program requests require semantic declarations without weakening old schemas."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from test_primary_resource_binding_v2 import request


def program_request(root):
    value=request(root);value['schema']='research/v2-primary-program-episode-request'
    value['procedure_reference']=dict(path=str(root/'procedure.json'),sha256='1'*64)
    return value


def test_program_request_binds_procedure_without_opening_it_or_granting_authority(tmp_path):
    from treeneuron.research.primary_search_worker_v2 import _validate_request,_job_binding,primary_worker_identity
    value=program_request(tmp_path)
    assert _validate_request(value)==value
    assert _job_binding(value)['procedure_reference']==value['procedure_reference']
    path=tmp_path/'request.json';raw=json.dumps(value).encode();path.write_bytes(raw)
    assert primary_worker_identity(path,hashlib.sha256(raw).hexdigest())['source_sha256']=='a'*64
    assert not (tmp_path/'procedure.json').exists() and not (tmp_path/'search').exists()


@pytest.mark.parametrize('fault',['missing_procedure','missing_resources','relative_procedure','boolean_hash','extra','legacy_schema'])
def test_program_semantics_cannot_fall_back_to_ungated_or_legacy_requests(tmp_path,fault):
    from treeneuron.research.primary_search_worker_v2 import _validate_request
    value=program_request(tmp_path)
    if fault=='missing_procedure':value.pop('procedure_reference')
    elif fault=='missing_resources':value.pop('resource_forecast_reference')
    elif fault=='relative_procedure':value['procedure_reference']['path']='procedure.json'
    elif fault=='boolean_hash':value['procedure_reference']['sha256']=True
    elif fault=='extra':value['production_ready']=True
    else:value['schema']='research/v2-primary-search-resource-episode-request'
    with pytest.raises(ValueError):_validate_request(value)


@pytest.mark.parametrize('operation',['preflight','caller','worker'])
def test_actual_registry_is_required_before_any_procedure_plan_or_output(tmp_path,monkeypatch,operation):
    from treeneuron.research import practical_procedure_v2
    from treeneuron.research.primary_caller_v2 import run_primary_search_episode
    from treeneuron.research.primary_invocation_v2 import prepare_primary_invocation
    from treeneuron.research.primary_search_worker_v2 import _prepare_qualified_episode
    def forbidden(*args,**kwargs):raise AssertionError('False actual qualification must fail before semantic or data reads')
    monkeypatch.setattr(practical_procedure_v2,'load_practical_procedure',forbidden)
    value=program_request(tmp_path)
    args={k:value[k] for k in ('practical_plan_reference','expected_practical_plan_hash','job_id','support_reference','qualification_reference','resource_forecast_reference','procedure_reference')}
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        if operation=='worker':_prepare_qualified_episode(request=value,output=tmp_path/'worker')
        elif operation=='caller':run_primary_search_episode(**args,episode_root=tmp_path/'episode',search_directory=tmp_path/'search')
        else:prepare_primary_invocation(**args,request_path=tmp_path/'request.json',worker_output=tmp_path/'worker',control_output=tmp_path/'control',search_directory=tmp_path/'search')
    assert 'procedure.json' not in str(error.value) and list(tmp_path.iterdir())==[]


def test_actual_cold_program_worker_rejects_false_qualification_before_procedure_data_or_output(tmp_path):
    from treeneuron.research.primary_search_worker_v2 import primary_worker_identity,_episode_environment
    from treeneuron.research.deployment_worker_v2 import _source
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process
    value=program_request(tmp_path);value['source_sha256']=_source()
    path=tmp_path/'request.json';path.write_text(json.dumps(value));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    result=supervise_cooperative_process(argv=[sys.executable,'-P','-B','-m','treeneuron.research.primary_search_worker_v2',
        '--request',str(path),'--expected-request-sha256',sha,'--output',str(tmp_path/'worker')],
        cwd=str(Path(__file__).resolve().parents[1]),env_overrides=_episode_environment(),output=tmp_path/'process',
        identity=primary_worker_identity(path,sha),admission_identity=dict(policy='software-negative'),admission_check=lambda:dict(admitted=True),rss_pause_bytes=512*1024**2)
    assert result['returncode']==1 and result['worker_protocol_closed']
    outcome=json.loads((tmp_path/'process/body-outcome.json').read_bytes())
    assert outcome['status']=='body_failed' and 'procedure.json' not in outcome['reason']
    assert not (tmp_path/'worker').exists() and not (tmp_path/'search').exists()
    assert outcome['protected_access_authorized'] is outcome['T_accessed'] is False
