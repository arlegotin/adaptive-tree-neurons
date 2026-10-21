"""Fixed programme orchestration fixtures grant no native/test authority."""
import importlib.util
import json
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.confirm_v2'),'Missing fixed protected practical programme'
    from treeneuron.research import confirm_v2
    return confirm_v2


def fixture(root):
    from treeneuron.research.deployment_worker_v2 import _source
    from treeneuron.research.protected_journal_v2 import _hash
    module=api();root.mkdir(parents=True,exist_ok=True)
    protocol=root/'known-protocol.json';protocol.write_text(json.dumps(dict(source_sha256=_source()))+'\n')
    import hashlib
    reference=dict(path=str(protocol),sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())
    inputs={k:reference for k in ('models_reference','global_barrier_reference','preT_costs_reference','readiness_reference')}
    inputs['qualification_reference']=dict(root=str(root/'unqualified-registry'),inventory_sha256='a'*64)
    plan=module._make_fixed_programme(dict(source_sha256=_source()),reference,inputs,directory=root/'known-study',output=root/'programme',rss_limit_bytes=1024**3)
    (root/'programme').mkdir();path=root/'programme/programme.json';path.write_text(json.dumps(plan,sort_keys=True)+'\n')
    return module,plan,dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def known_operations(plan):
    from treeneuron.research.protected_journal_v2 import _write_new
    import hashlib
    root=Path(plan['directory'])
    def resolve(step):
        path=root/f'{step}-known.json'
        if not path.exists():return None
        value=json.loads(path.read_text())
        if value!={'step':step,'native_qualified':False}:raise ValueError('Known fixture output changed')
        return dict(status='completed',evidence_references=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
    def execute(step):
        path=root/f'{step}-known.json';_write_new(path,dict(step=step,native_qualified=False));return resolve(step)
    return execute,resolve


def test_fixed_plan_does_not_admit_new_method_search_or_protected_query_variants(tmp_path):
    module,plan,reference=fixture(tmp_path)
    assert plan['steps']==['native_timing_plan','predictions','prediction_stage','metrics','analysis','raw_audit','native_measurements']
    assert plan['rss_limit_bytes']==1024**3 and plan['pause_after_ns'] is None
    assert plan['native_qualification_authenticated'] is plan['protected_access_authorized'] is False


def test_public_fake_native_qualification_stops_before_programme_output(tmp_path):
    module=api();fake=dict(path=str(tmp_path/'forbidden'),sha256='a'*64)
    kwargs={k:fake for k in ('models_reference','global_barrier_reference','preT_costs_reference','readiness_reference')}
    kwargs['qualification_reference']=dict(root=str(tmp_path/'false-q'),inventory_sha256='b'*64)
    with pytest.raises((ValueError,FileNotFoundError)):
        module.seal_practical_confirmation_programme(fake,directory=tmp_path/'study',output=tmp_path/'programme',**kwargs)
    with pytest.raises((ValueError,FileNotFoundError)):
        module.run_practical_confirmation(fake,protocol_reference=fake,**kwargs)
    assert list(tmp_path.iterdir())==[]


def test_prefix_resume_reopens_completed_proofs_without_executing_them_again(tmp_path):
    module,plan,reference=fixture(tmp_path);execute,resolve=known_operations(plan)
    first=module._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None,max_step_starts=2)
    assert first['status']=='operator_pause' and first['completed_steps']==plan['steps'][:2]
    originals={p:p.read_bytes() for p in Path(plan['directory']).glob('*-known.json')}
    second=module._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None)
    assert second['status']=='completed' and second['completed_steps']==plan['steps']
    assert all(p.read_bytes()==b for p,b in originals.items())
    assert len(list((Path(plan['directory'])/'calls').rglob('end.json')))==7
    module._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None)
    assert len(list((Path(plan['directory'])/'calls').rglob('end.json')))==7
    assert second['protected_access_authorized'] is second['native_qualification_authenticated'] is False


def test_partial_step_stops_later_operations_and_retains_each_paid_call(tmp_path):
    module,plan,reference=fixture(tmp_path);execute,resolve=known_operations(plan)
    def partial(step):
        if step=='predictions':return dict(status='paused',evidence_references=[])
        return execute(step)
    first=module._run_fixed_steps(reference,plan,execute_step=partial,resolve_completed=resolve,guard=lambda:None)
    assert first['status']=='operator_pause' and first['completed_steps']==['native_timing_plan']
    assert not (Path(plan['directory'])/'metrics-known.json').exists()
    second=module._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None)
    assert second['status']=='completed'
    calls=list((Path(plan['directory'])/'calls').rglob('end.json'))
    assert len(calls)==8 and all(json.loads(p.read_text())['elapsed_ns']>0 for p in calls)


def test_failed_operation_closes_cost_record_before_propagating_and_never_runs_later_step(tmp_path):
    module,plan,reference=fixture(tmp_path);execute,resolve=known_operations(plan)
    def fail(step):raise FloatingPointError('Declared software fault')
    with pytest.raises(FloatingPointError):module._run_fixed_steps(reference,plan,execute_step=fail,resolve_completed=resolve,guard=lambda:None)
    ends=list((Path(plan['directory'])/'calls').rglob('end.json'));assert len(ends)==1
    value=json.loads(ends[0].read_text());assert value['error']['type']=='FloatingPointError' and value['elapsed_ns']>0
    assert not list(Path(plan['directory']).glob('*-known.json'))


def test_orphan_step_start_is_not_silently_rerun_or_zero_charged(tmp_path):
    module,plan,reference=fixture(tmp_path);execute,resolve=known_operations(plan)
    module._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None,max_step_starts=1)
    end=next((Path(plan['directory'])/'calls').rglob('end.json'));end.unlink()
    with pytest.raises(ValueError):module._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None)
    assert len(list(Path(plan['directory']).glob('*-known.json')))==1


def test_changed_programme_commitment_rejects_before_any_operation(tmp_path):
    module,plan,reference=fixture(tmp_path);execute,resolve=known_operations(plan)
    Path(reference['path']).write_bytes(Path(reference['path']).read_bytes()+b' ')
    with pytest.raises(ValueError):module._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None)
    assert not (Path(plan['directory'])/'calls').exists()


def test_final_actual_context_failure_is_retained_in_programme_invocation(tmp_path):
    module,plan,reference=fixture(tmp_path);execute,resolve=known_operations(plan);calls=0
    def guard():
        nonlocal calls
        calls+=1
        if calls==2:raise ValueError('Actual context changed')
    with pytest.raises(ValueError):module._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=guard,max_step_starts=1)
    ends=list((Path(plan['directory'])/'invocations').glob('*-end.json'));assert len(ends)==1
    value=json.loads(ends[0].read_text());assert value['error']['type']=='ValueError'


def test_missing_invocation_end_blocks_resume_before_any_new_step(tmp_path):
    module,plan,reference=fixture(tmp_path);execute,resolve=known_operations(plan)
    module._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None,max_step_starts=1)
    next((Path(plan['directory'])/'invocations').glob('*-end.json')).unlink()
    with pytest.raises(ValueError):module._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None)
    assert len(list(Path(plan['directory']).glob('*-known.json')))==1


def test_semantically_wrong_rehashed_call_cannot_be_adopted_on_resume(tmp_path):
    from treeneuron.research.protected_journal_v2 import _hash
    import hashlib
    module,plan,reference=fixture(tmp_path);execute,resolve=known_operations(plan)
    module._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None,max_step_starts=1)
    path=next((Path(plan['directory'])/'calls').rglob('start.json'));value=json.loads(path.read_text());value['source_sha256']='0'*64;path.write_text(json.dumps(value)+'\n')
    end=path.parent/'end.json';value=json.loads(end.read_text());value['start_reference']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();end.write_text(json.dumps(value)+'\n')
    with pytest.raises(ValueError):module._run_fixed_steps(reference,plan,execute_step=execute,resolve_completed=resolve,guard=lambda:None)
    assert len(list(Path(plan['directory']).glob('*-known.json')))==1


def test_protected_owner_binds_one_programme_before_timing_or_prediction(tmp_path):
    from test_protected_journal_v2 import create,ready,event,ref
    from treeneuron.research.deployment_worker_v2 import _source
    proto=tmp_path/'protocol.json';proto.write_text('{}\n')
    metadata=dict(protocol_reference=ref(proto),source_sha256=_source(),prediction_manifest_sha256='c'*64,
        queries=[dict(query_id=c*64) for c in ('a','b')],assets=[dict(dataset='toy',raw_sha256='d'*64,target='target')])
    m,root,owners,binding=create(tmp_path,metadata);ready(m,root)
    path=tmp_path/'programme.json';path.write_text('{"known_fixture":true}\n')
    event(m,root,'programme_frozen',programme_reference=ref(path));state=m._read_metadata_study(root)
    assert state['programme_reference']==ref(path) and state['phase']=='owners_ready'
    assert state['protected_access_authorized'] is state['native_qualification_authenticated'] is False
    with pytest.raises(ValueError):event(m,root,'programme_frozen',programme_reference=ref(path))
    path.write_text('{"changed":true}\n')
    with pytest.raises(ValueError):m._read_metadata_study(root)


@pytest.mark.parametrize('existing',['timing','query'])
def test_programme_cannot_be_claimed_after_other_protected_work_started(tmp_path,existing):
    from test_protected_journal_v2 import create,ready,event,start,ref
    from treeneuron.research.deployment_worker_v2 import _source
    proto=tmp_path/'protocol.json';proto.write_text('{}\n')
    metadata=dict(protocol_reference=ref(proto),source_sha256=_source(),prediction_manifest_sha256='c'*64,
        queries=[dict(query_id=c*64) for c in ('a','b')],assets=[dict(dataset='toy',raw_sha256='d'*64,target='target')])
    m,root,owners,binding=create(tmp_path,metadata);ready(m,root)
    if existing=='timing':
        path=tmp_path/'deployment-plan.json';path.write_text('{}\n');event(m,root,'deployment_plan_frozen',plan_reference=ref(path))
    else:start(m,root)
    with pytest.raises(ValueError):event(m,root,'programme_frozen',programme_reference=dict(path=str(tmp_path/'programme.json'),sha256='f'*64))
