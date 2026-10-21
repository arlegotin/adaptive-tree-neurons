"""Real local measurements remain false-authority fixtures, never native gates."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from test_native_deployment_worker_v2 import measurement_fixture,ref


def api():
    assert importlib.util.find_spec('treeneuron.research.native_deployment_execution_v2'), 'Missing original-model timing controller'
    from treeneuron.research import native_deployment_execution_v2
    return native_deployment_execution_v2


def fixture(root):
    from treeneuron.research.deployment_worker_v2 import _source
    from treeneuron.research.protected_journal_v2 import _hash
    path,value,_=measurement_fixture(root/'known');request=ref(path)
    models=[dict(job_id='known-model',request_reference=request)]
    jobs=[]
    for i,role in enumerate(('selected','baseline')):
        jobs.append(dict(job_id=f'known-pair/block-00/{role}',comparison_id='known-pair',dataset='airfoil',outer_fold=0,
            search_repeat=0,block_id=0,role=role,order='selected_first',model_job_id='known-model',request_reference=request,
            depends_on=[] if not i else [jobs[-1]['job_id']],dependency_policy='all_terminal'))
    payload=dict(source_sha256=_source(),schedule=dict(source_sha256=_source(),models=models,jobs=jobs,
        comparisons=[dict(comparison_id='known-pair',blocks=30)],worker_concurrency=1))
    target=root/'known-plan.json';target.write_text(json.dumps(payload)+'\n')
    return payload,ref(target)


def make(root,rss=1024**3):
    module=api();payload,reference=fixture(root)
    return module,module._make_execution_queue(reference,payload,rss_limit_bytes=rss)


def allow(*args):return dict(admitted=True,reason='known-fixture-only')


def test_public_fake_qualification_stops_before_timing_output_or_worker(tmp_path):
    module=api();output=tmp_path/'forbidden';fake=dict(path=str(tmp_path/'missing.json'),sha256='a'*64)
    kwargs={k:fake for k in ('models_reference','global_barrier_reference','qualification_reference','preT_costs_reference','readiness_reference')}
    for method in (module.run_native_deployment_measurements,module.collect_native_deployment_measurements):
        with pytest.raises((ValueError,FileNotFoundError,PermissionError)):
            method(fake,protocol_reference=fake,directory=tmp_path/'study',output=output,**kwargs)
    assert not output.exists()


def test_queue_binds_original_requests_serial_order_and_no_wall_cap(tmp_path):
    module,queue=make(tmp_path)
    assert [j['payload']['role'] for j in queue['attempts']]==['selected','baseline']
    assert queue['attempts'][1]['depends_on']==[queue['attempts'][0]['job_id']]
    policy=queue['identity']
    assert policy['pause_after_ns'] is None and policy['worker_concurrency']==1
    assert policy['required_memory_proxy_bytes']==1879048192
    assert policy['rss_limit_bytes']==1024**3 and policy['required_disk_proxy_bytes']==536870912
    assert policy['native_models_authenticated'] is policy['cost_claim_authorized'] is False


def test_known_owned_pair_prefix_resume_never_remeasures_committed_job(tmp_path):
    module,queue=make(tmp_path);output=tmp_path/'execution'
    first=module._run_measurement_queue(queue,output,admission_check=allow,max_jobs_this_invocation=1)
    assert first['status']=='paused_before_job'
    before=list(output.rglob('measurement.json'));assert len(before)==1
    first_bytes=before[0].read_bytes()
    second=module._run_measurement_queue(queue,output,resume=True,admission_check=allow)
    assert second['status']=='completed'
    assert len(list(output.rglob('measurement.json')))==2 and before[0].read_bytes()==first_bytes
    collected=module._collect_measurement_queue(queue,output)
    assert collected['completed_measurements']==2
    assert collected['complete_30_block_comparisons']==0
    assert collected['comparisons']['known-pair']['resident_blocks'] is None
    assert collected['comparisons']['known-pair']['missing_blocks']==list(range(1,30))
    assert collected['native_models_authenticated'] is collected['cost_claim_authorized'] is collected['T_accessed'] is False
    assert collected['sum_full_executor_call_ns']>collected['sum_owned_root_ns']>0
    assert collected['sum_queue_invocation_ns']>=collected['sum_full_executor_call_ns']
    module._run_measurement_queue(queue,output,resume=True,admission_check=allow)
    assert len(list(output.rglob('measurement.json')))==2


def test_pre_start_memory_denial_can_resume_without_consuming_measurement_retry(tmp_path):
    module,queue=make(tmp_path);output=tmp_path/'execution'
    denied=module._run_measurement_queue(queue,output,admission_check=lambda *args:dict(admitted=False,reason='other processes need RAM'))
    assert denied['status']=='paused_before_job' and not list(output.rglob('started.json'))
    assert (output/'events.jsonl').read_bytes()==b''
    module._run_measurement_queue(queue,output,resume=True,admission_check=allow,max_jobs_this_invocation=1)
    assert len(list(output.rglob('measurement.json')))==1
    assert all('attempt-0' in str(p) for p in output.rglob('measurement.json'))


def test_template_mutation_blocks_resume_before_another_measurement(tmp_path):
    module,queue=make(tmp_path);output=tmp_path/'execution'
    module._run_measurement_queue(queue,output,admission_check=allow,max_jobs_this_invocation=1)
    template=Path(queue['identity']['measurement_plan_reference']['path'])
    template.write_bytes(template.read_bytes()+b' ')
    with pytest.raises(ValueError):module._run_measurement_queue(queue,output,resume=True,admission_check=allow)
    assert len(list(output.rglob('measurement.json')))==1


def test_incomplete_failed_member_never_becomes_a_complete_case_timing_result(tmp_path):
    module,queue=make(tmp_path);output=tmp_path/'execution'
    def broken(**kwargs):raise FloatingPointError('known fixture missing measurement')
    result=module._run_measurement_queue(queue,output,admission_check=allow,execute_episode=broken)
    assert result['status']=='completed_with_failures'
    found=module._collect_measurement_queue(queue,output)
    assert found['completed_measurements']==0 and len(found['missing_members'])==2
    assert found['comparisons']['known-pair']['resident_blocks'] is None
    assert found['sum_full_executor_call_ns']>0
    assert found['all_attempt_costs_observed'] is True


def test_resume_after_ambiguous_executed_member_does_not_resample(tmp_path):
    module,queue=make(tmp_path);output=tmp_path/'execution'
    def interrupted(**kwargs):raise KeyboardInterrupt('known interruption after unknown progress')
    with pytest.raises(KeyboardInterrupt):module._run_measurement_queue(queue,output,admission_check=allow,execute_episode=interrupted)
    def forbidden(**kwargs):pytest.fail('Ambiguous measurement was silently resampled')
    result=module._run_measurement_queue(queue,output,resume=True,admission_check=allow,execute_episode=forbidden,max_jobs_this_invocation=1)
    assert result['statuses'][queue['attempts'][0]['job_id']]=='irrecoverable_missing'
    assert not list(output.rglob('measurement.json'))


def test_readonly_collection_does_not_call_predictor_again(tmp_path,monkeypatch):
    module,queue=make(tmp_path);output=tmp_path/'execution'
    module._run_measurement_queue(queue,output,admission_check=allow,max_jobs_this_invocation=1)
    from treeneuron.research.p13_simple import P13Predictor
    def forbidden(*args,**kwargs):pytest.fail('Read-only collection called the model')
    monkeypatch.setattr(P13Predictor,'predict',forbidden)
    assert module._collect_measurement_queue(queue,output)['completed_measurements']==1


@pytest.mark.parametrize('limit',[True,0,17*1024**3])
def test_invalid_resource_policy_rejects_before_queue_publication(tmp_path,limit):
    module=api();payload,reference=fixture(tmp_path)
    with pytest.raises(ValueError):module._make_execution_queue(reference,payload,rss_limit_bytes=limit)


def test_closed_measurement_recovers_without_a_second_model_process(tmp_path):
    module,queue=make(tmp_path);output=tmp_path/'execution'
    def closed_then_interrupted(**kwargs):
        module._execute_owned_measurement(**kwargs)
        raise KeyboardInterrupt('closure published before coordinator interruption')
    with pytest.raises(KeyboardInterrupt):
        module._run_measurement_queue(queue,output,admission_check=allow,execute_episode=closed_then_interrupted)
    assert len(list(output.rglob('measurement.json')))==1
    def forbidden(**kwargs):pytest.fail('Recovery remeasured an already closed original')
    result=module._run_measurement_queue(queue,output,resume=True,admission_check=allow,execute_episode=forbidden,max_jobs_this_invocation=1)
    assert result['statuses'][queue['attempts'][0]['job_id']]=='completed_ineligible'
    assert len(list(output.rglob('measurement.json')))==1
    found=module._collect_measurement_queue(queue,output)
    assert found['completed_measurements']==1 and found['all_attempt_costs_observed'] is True


@pytest.mark.parametrize('fault',['clock_negative','foreign_member','wrong_scope','unknown_field'])
def test_validly_rehashed_semantically_wrong_call_is_rejected(tmp_path,fault):
    module,queue=make(tmp_path);output=tmp_path/'execution'
    module._run_measurement_queue(queue,output,admission_check=allow,max_jobs_this_invocation=1)
    call=next(output.rglob('call-clock.json'));value=json.loads(call.read_text())
    if fault=='clock_negative':value['elapsed_ns']=-1
    elif fault=='foreign_member':value['job_id']=queue['attempts'][1]['job_id']
    elif fault=='wrong_scope':value['clock_scope']='Only selected samples'
    else:value['native_override']=True
    call.write_text(json.dumps(value)+'\n')
    with pytest.raises(ValueError):module._read_call(ref(call),queue,queue['attempts'][0],call.parent)


def test_failed_final_context_guard_is_recorded_as_failure_not_a_clean_completion(tmp_path):
    module,queue=make(tmp_path);output=tmp_path/'execution';calls=0
    def guard():
        nonlocal calls
        calls+=1
        if calls>1:raise ValueError('Actual frozen context changed')
    with pytest.raises(ValueError,match='Actual frozen context changed'):
        module._run_measurement_queue(queue,output,admission_check=allow,max_jobs_this_invocation=1,guard=guard)
    ends=sorted(output.glob('context-*.json'))
    assert ends, 'Final context outcome was not retained alongside the inner queue receipt'
    last=json.loads(ends[-1].read_text())
    assert last['status']=='interrupted_or_failed' and last['error']['type']=='ValueError'
    assert len(list(output.rglob('measurement.json')))==1
