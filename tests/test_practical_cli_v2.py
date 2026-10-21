"""Practical commands preserve real public gates and immutable metadata."""
import hashlib
import importlib.util
import json

import pytest

from test_primary_search_declaration_v2 import study


def ref(path):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def control(tmp_path,operation,arguments,**changes):
    from treeneuron.research.deployment_worker_v2 import _source
    path=tmp_path/'command.json'
    path.write_text(json.dumps(dict(schema='research/v2-practical-command',
        source_sha256=_source(),operation=operation,arguments=arguments,**changes)))
    return path


def api():
    assert importlib.util.find_spec('treeneuron.research.practical_cli_v2'), 'Missing practical command boundary'
    from treeneuron.research import practical_cli_v2
    return practical_cli_v2


def test_cli_support_lock_is_real_immutable_and_planning_confers_no_authority(tmp_path):
    from treeneuron.research.cli import main
    from treeneuron.research.practical_support_lock_v2 import load_practical_support_lock
    command=control(tmp_path,'support_lock',{});path=tmp_path/'support.json';plan=tmp_path/'plan.json'
    assert main(['plan','--config',str(command),'--output',str(plan)])==0
    value=json.loads(plan.read_bytes())
    assert value['operation']=='support_lock' and value['config_reference']==ref(command)
    assert value['execution_authorized'] is value['protected_access_authorized'] is False
    assert not path.exists()
    assert main(['run','--config',str(command),'--output',str(path)])==0
    saved=path.read_bytes();loaded=load_practical_support_lock(ref(path))
    assert len(loaded['families'])==6 and loaded['native_workload_qualified'] is False
    assert main(['run','--config',str(command),'--output',str(path)])==2
    assert path.read_bytes()==saved


def test_cli_procedure_keeps_native_completion_policy_and_all840_jobs(study,tmp_path):
    from treeneuron.research.cli import main
    from treeneuron.research.practical_procedure_v2 import load_practical_procedure
    plan,support=study;plan_path=tmp_path/'workload.json';plan_path.write_text(plan.resolved_json)
    command=control(tmp_path,'procedure',dict(practical_plan_reference=ref(plan_path),
        expected_practical_plan_hash=plan.to_dict()['plan_hash'],support_reference=support))
    output=tmp_path/'procedure.json'
    assert main(['freeze','--config',str(command),'--output',str(output)])==0
    value=load_practical_procedure(ref(output),practical_plan=plan,
        expected_practical_plan_hash=plan.to_dict()['plan_hash'],support_reference=support)
    assert value['policy']['search']['training_wall_time_limit_ns'] is None
    assert value['policy']['program']['primary_jobs']==168
    assert value['policy']['program']['secondary_jobs']==672
    assert value['protected_access_authorized'] is False


@pytest.mark.parametrize('fault',['unknown_operation','unknown_argument','output_override','changed_source','unknown_field','duplicate_field'])
def test_invalid_practical_controls_reject_before_creating_outputs(tmp_path,fault):
    from treeneuron.research.cli import main
    command=control(tmp_path,'support_lock',{});value=json.loads(command.read_bytes())
    if fault=='unknown_operation':value['operation']='os.system'
    elif fault=='unknown_argument':value['arguments']['labels']='T-labels.npz'
    elif fault=='output_override':value['arguments']['output']=str(tmp_path/'foreign')
    elif fault=='changed_source':value['source_sha256']='0'*64
    elif fault=='unknown_field':value['protected_access_authorized']=True
    command.write_text(json.dumps(value))
    if fault=='duplicate_field':command.write_text(command.read_text()[:-1]+',"operation":"support_lock"}')
    output=tmp_path/'never-created.json'
    assert main(['run','--config',str(command),'--output',str(output)])==2
    assert not output.exists() and not (tmp_path/'foreign').exists()


def test_native_control_cannot_open_foreign_plan_or_publish_before_actual_qualification(tmp_path,capsys):
    from treeneuron.research.cli import main
    command=control(tmp_path,'resource_forecast',dict(
        qualification_reference=dict(root=str(tmp_path/'unqualified'),inventory_sha256='b'*64),
        practical_plan_reference=dict(path=str(tmp_path/'forbidden-plan.json'),sha256='c'*64),
        expected_practical_plan_hash='d'*64,
        support_reference=dict(path=str(tmp_path/'forbidden-support.json'),sha256='e'*64)))
    output=tmp_path/'never-created.json'
    assert main(['run','--config',str(command),'--output',str(output)])==2
    assert not output.exists()
    assert 'forbidden-plan' not in capsys.readouterr().err


def test_command_planner_cannot_publish_into_scientific_source(tmp_path):
    from treeneuron.research.deployment_worker_v2 import _ROOT
    command=control(tmp_path,'support_lock',{})
    with pytest.raises(ValueError,match='output|executable|preserve'):
        api().plan_practical_command(command,output=_ROOT/'src/never-practical-output.json')
    assert not (_ROOT/'src/never-practical-output.json').exists()


def test_practical_plan_cli_expands_all_jobs_without_loading_any_role_arrays(tmp_path):
    from test_practical_workload import fixtures
    from treeneuron.research.cli import main
    from treeneuron.research.practical_support_lock_v2 import seal_practical_support_lock
    from treeneuron.research.qualify import qualification_locks_sha256
    support=seal_practical_support_lock(tmp_path/'support.json');references={}
    for name,value in fixtures().items():
        path=tmp_path/(name+'-manifest.json');path.write_text(json.dumps(value));references[name]=ref(path)
    command=control(tmp_path,'practical_plan',dict(data_manifest_references=references,
        support_reference=support,lock_sha256=qualification_locks_sha256()))
    output=tmp_path/'workload.json'
    assert main(['run','--config',str(command),'--output',str(output)])==0
    value=json.loads(output.read_bytes())
    assert len(value['jobs'])==840 and value['tranches']==dict(primary_jobs=168,secondary_jobs=672)
    assert all(row['budget']['deadline_ns'] is None for row in value['jobs'])
    assert value['protected_access_authorized'] is value['execution_implemented'] is False


def test_practical_control_cannot_write_into_runtime_tree(tmp_path):
    from treeneuron.research.deployment_worker_v2 import _ROOT
    command=control(tmp_path,'support_lock',{})
    with pytest.raises(ValueError,match='preserve|executable|runtime'):
        api().plan_practical_command(command,output=_ROOT/'environment/never-practical-output.json')


def test_confirm_bootstrap_metadata_cannot_grant_native_authority_or_create_study(tmp_path):
    from treeneuron.config import canonical_hash
    from treeneuron.research.deployment_worker_v2 import _source
    module=api()
    assert hasattr(module,'confirm_practical_protocol'),'Missing actual-protocol confirmation CLI'
    fake=dict(path=str(tmp_path/'forbidden-models.json'),sha256='a'*64)
    payload=dict(source_sha256=_source(),models_reference=fake,global_barrier_reference=fake,
        preT_costs_reference=fake,readiness_reference=fake,
        qualification_reference=dict(root=str(tmp_path/'unqualified'),inventory_sha256='b'*64))
    path=tmp_path/'protocol.json';path.write_text(json.dumps(dict(
        schema='research/v2-final-practical-prediction-protocol',payload=payload,payload_hash=canonical_hash(payload))))
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module.confirm_practical_protocol(path,output=tmp_path/'never-study')
    assert 'forbidden-models' not in str(error.value)
    assert not (tmp_path/'never-study').exists()


def test_report_cli_requires_genuine_frozen_inputs_before_any_publication(tmp_path):
    module=api()
    assert hasattr(module,'report_practical_study'),'Missing protected practical report CLI'
    root=tmp_path/'study';root.mkdir();fake=dict(path=str(tmp_path/'forbidden-input.json'),sha256='a'*64)
    value=dict(schema='research/v2-practical-publication',protocol_reference=fake,programme_reference=fake,
        analysis_reference=fake,raw_audit_reference=fake,prediction_stage_reference=fake,costs_reference=fake,
        directory=str(root/'protected-study'),models_reference=fake,global_barrier_reference=fake,
        preT_costs_reference=fake,readiness_reference=fake,
        qualification_reference=dict(root=str(tmp_path/'unqualified'),inventory_sha256='b'*64))
    (root/'publication.json').write_text(json.dumps(value))
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module.report_practical_study(root,output=tmp_path/'never-report.md')
    assert 'forbidden-input' not in str(error.value) and not (tmp_path/'never-report.md').exists()


def test_confirmation_resume_detects_actual_closed_history_without_starting_another_invocation(tmp_path):
    from test_confirm_programme_v2 import fixture,known_operations
    programme,plan,reference=fixture(tmp_path/'known');execute,resolve=known_operations(plan);module=api()
    assert hasattr(module,'_programme_complete'),'Missing read-only fixed-history resume check'
    assert module._programme_complete(reference,plan) is False
    programme._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None,max_step_starts=2)
    assert module._programme_complete(reference,plan) is False
    programme._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None)
    assert module._programme_complete(reference,plan) is True
    before={str(p):p.read_bytes() for p in (tmp_path/'known/programme').rglob('*.json')}
    assert module._programme_complete(reference,plan) is True
    assert before=={str(p):p.read_bytes() for p in (tmp_path/'known/programme').rglob('*.json')}


@pytest.mark.parametrize('status,expected',[('completed',0),('paused',3),('blocked',4),('preparation_failed',4),
    ('software_verification_failed',4),('resource_exhausted',3),('interrupted_recoverable',3)])
def test_native_command_exit_status_distinguishes_completion_from_safe_pause_and_failure(status,expected):
    module=api()
    assert hasattr(module,'practical_command_exit_code'),'Missing operational exit status handling'
    assert module.practical_command_exit_code(dict(status=status))==expected


@pytest.mark.parametrize('operation',['production_verification','readiness'])
def test_cli_plans_required_production_gates_without_treating_metadata_as_authority(tmp_path,operation):
    import sys
    from treeneuron.research.cli import main
    if operation=='production_verification':arguments=dict(profile='reporting',interpreter=sys.executable)
    else:
        arguments=dict(qualification_reference=dict(root=str(tmp_path/'unqualified'),inventory_sha256='b'*64),
            verification_references={},review_reference=dict(path=str(tmp_path/'absent-review'),sha256='c'*64),pinned_commit='d'*40)
    command=control(tmp_path,operation,arguments);output=tmp_path/'metadata-plan.json'
    assert main(['plan','--config',str(command),'--output',str(output)])==0
    value=json.loads(output.read_bytes())
    assert value['operation']==operation and value['arguments']==arguments
    assert value['execution_authorized'] is value['protected_access_authorized'] is False
    assert not (tmp_path/'unqualified').exists() and not (tmp_path/'absent-review').exists()
