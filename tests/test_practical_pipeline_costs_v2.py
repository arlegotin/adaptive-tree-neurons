"""Supplied interval arithmetic never grants actual pipeline cost authority."""
import copy
import importlib.util
from pathlib import Path

import pytest

from test_practical_preT_costs_v2 import domain


def api():
    assert importlib.util.find_spec('treeneuron.research.practical_pipeline_costs_v2'),'Missing actual practical pipeline cost join'
    from treeneuron.research import practical_pipeline_costs_v2
    return practical_pipeline_costs_v2


def intervals(second_boot='b'):
    second=300 if second_boot=='b' else 10
    frames=[dict(frame_id='frame-a',clock_domain=domain('b'),origin_ns=100,end_ns=200,elapsed_ns=100,invocation_ids=['inv-a']),
        dict(frame_id='frame-b',clock_domain=domain(second_boot),origin_ns=second,end_ns=second+50,elapsed_ns=50,invocation_ids=['inv-a','inv-b']),
        dict(frame_id='denied-frame',clock_domain=domain('b'),origin_ns=500,end_ns=510,elapsed_ns=10,invocation_ids=[])]
    inv=[dict(invocation_id='inv-a',clock_domain=domain('b'),origin_ns=105,end_ns=195,elapsed_ns=90),
        dict(invocation_id='inv-b',clock_domain=domain(second_boot),origin_ns=second+5,end_ns=second+45,elapsed_ns=40)]
    names=['native_timing_plan','predictions','prediction_stage','metrics','analysis','raw_audit','native_measurements'];calls=[]
    for i,step in enumerate(names):
        offset=110+5*i if i<4 else second+10+5*(i-4)
        calls.append(dict(call_id=f'call-{i}',step=step,invocation_id='inv-a' if i<4 else 'inv-b',
            clock_domain=domain('b' if i<4 else second_boot),origin_ns=offset,end_ns=offset+5,elapsed_ns=5))
    return frames,inv,calls


def test_sum_closed_frames_counts_denial_and_never_adds_nested_clocks():
    result=api()._summarize_postT_intervals(*intervals())
    assert result['physical_worker_ns']==160 and result['nested_invocation_ns']==130
    assert sum(result['full_step_call_ns'].values())==35 and result['common_programme_overhead_ns']==125
    assert result['observed_calendar_span_ns']==410 and result['observed_idle_ns']==250
    assert result['whole_training_procedure_cost_authenticated'] is result['protected_access_authorized'] is False


def test_different_observed_boots_allow_duration_sum_without_calendar_comparison():
    result=api()._summarize_postT_intervals(*intervals('c'))
    assert result['physical_worker_ns']==160 and result['observed_clock_domain_count']==2
    assert result['observed_calendar_span_ns'] is result['observed_idle_ns'] is None


@pytest.mark.parametrize('fault',['missing_step','duplicate_call','unknown_step','overlap','unlinked','different_boot','negative','extra','unknown_domain'])
def test_incomplete_or_unroutable_interval_metadata_cannot_certify_complete_cost(fault):
    f,i,c=intervals()
    if fault=='missing_step':c.pop()
    elif fault=='duplicate_call':c.append(copy.deepcopy(c[0]))
    elif fault=='unknown_step':c[0]['step']='different-training-treatment'
    elif fault=='overlap':f[2].update(origin_ns=110,end_ns=120)
    elif fault=='unlinked':f[0]['invocation_ids']=[]
    elif fault=='different_boot':c[0]['clock_domain']=domain('c')
    elif fault=='negative':c[0]['elapsed_ns']=-1
    elif fault=='extra':f[0]['whole_cost']=True
    else:f[0]['clock_domain'].update(status='unknown',boot_sha256=None,reason='unavailable')
    with pytest.raises(ValueError):api()._summarize_postT_intervals(f,i,c)


def test_public_cost_seal_requires_actual_full_frozen_gates_before_output(tmp_path):
    m=api();fake=dict(path=str(tmp_path/'forbidden'),sha256='a'*64)
    kwargs={k:fake for k in ('programme_reference','analysis_reference','raw_audit_reference','prediction_stage_reference',
        'models_reference','global_barrier_reference','preT_costs_reference','readiness_reference')}
    kwargs['qualification_reference']=dict(root=str(tmp_path/'false-q'),inventory_sha256='b'*64)
    with pytest.raises((ValueError,FileNotFoundError)):
        m.seal_practical_pipeline_costs(fake,directory=tmp_path/'study',output=tmp_path/'costs.json',**kwargs)
    assert list(tmp_path.iterdir())==[]


def archived_negative_frames(tmp_path):
    from test_confirm_programme_v2 import fixture
    from treeneuron.research import program_process_v2 as process
    import time
    _,plan,reference=fixture(tmp_path/'known')
    assert 'observer_directory' in plan,'Fixed programme does not bind its complete observer archive'
    root=Path(plan['observer_directory'])
    args=dict(plan['inputs'],programme_reference=reference,protocol_reference=plan['protocol_reference'],directory=plan['directory'],max_step_starts=1)
    for index,admitted in enumerate((False,True)):
        request=process._make_request('confirmation',args,time.perf_counter_ns());output=root/f'episode-{index:04}'
        with process._confirmation_observer_slot(plan,output):
            process._run_program_envelope(request=request,output=output,admission_identity=dict(purpose='known-negative-cost-fixture'),
                admission_check=lambda:dict(admitted=admitted),root_rss_pause_bytes=1024**3)
    return plan,reference,root


def test_discovery_includes_denied_and_failed_owned_frames_without_a_caller_filter(tmp_path):
    module=api();plan,reference,root=archived_negative_frames(tmp_path)
    result=module._discover_confirmation_envelopes(reference,plan)
    assert len(result['frames'])==2
    assert result['frames'][0]['proof']['child_started'] is False
    assert result['frames'][1]['proof']['child_started'] is True
    assert result['frames'][1]['proof']['returncode']==1
    assert all(f['proof']['elapsed_ns']>0 for f in result['frames'])
    assert result['whole_training_procedure_cost_authenticated'] is False


def test_missing_last_frame_terminal_cannot_silently_reduce_paid_physical_cost(tmp_path):
    module=api();plan,reference,root=archived_negative_frames(tmp_path)
    path=root/'episode-0001/envelope.json';original=path.read_bytes();path.unlink()
    try:
        with pytest.raises(ValueError):module._discover_confirmation_envelopes(reference,plan)
    finally:path.write_bytes(original)


def test_confirmation_cannot_escape_or_reuse_its_serial_observer_slot(tmp_path):
    api();plan,reference,root=archived_negative_frames(tmp_path)
    from treeneuron.research.program_process_v2 import _confirmation_observer_slot
    for path in (tmp_path/'untracked/episode-0002',root/'episode-0000',root/'episode-0003'):
        with pytest.raises(ValueError):
            with _confirmation_observer_slot(plan,path):pytest.fail('Non-next observer slot was granted')
    assert not (tmp_path/'untracked').exists()


def test_query_attribution_charges_failed_calls_and_reused_views_once():
    m=api();assert hasattr(m,'_attribute_query_calls'),'Missing query cost attribution'
    queries=[dict(query_id='q1',job_id='A'),dict(query_id='q2',job_id='A'),dict(query_id='q3',job_id='B')]
    calls=[dict(call_id='first-failed',query_id='q1',elapsed_ns=5),dict(call_id='first-complete',query_id='q1',elapsed_ns=10),
        dict(call_id='other-view',query_id='q2',elapsed_ns=7),dict(call_id='second-model',query_id='q3',elapsed_ns=3)]
    result=m._attribute_query_calls(queries,calls)
    assert result['per_job_ns']=={'A':22,'B':3} and result['per_query_ns']=={'q1':15,'q2':7,'q3':3}
    assert result['whole_training_procedure_cost_authenticated'] is False
    with pytest.raises(ValueError):m._attribute_query_calls(queries,calls+[calls[0]])
    with pytest.raises(ValueError):m._attribute_query_calls(queries,calls[:-1])


def test_unfinished_owned_archive_cannot_be_promoted_to_complete_pipeline_cost(tmp_path):
    m=api();assert hasattr(m,'_collect_postT_programme'),'Missing full programme collection'
    plan,reference,root=archived_negative_frames(tmp_path)
    with pytest.raises(ValueError):m._collect_postT_programme(reference,plan)


def test_public_pipeline_cost_loader_repeats_actual_native_gates_before_saved_true_flags(tmp_path):
    m=api();assert hasattr(m,'load_practical_pipeline_costs'),'Missing public pipeline cost loader'
    fake=dict(path=str(tmp_path/'forbidden'),sha256='a'*64)
    kwargs={k:fake for k in ('programme_reference','analysis_reference','raw_audit_reference','prediction_stage_reference',
        'models_reference','global_barrier_reference','preT_costs_reference','readiness_reference')}
    kwargs['qualification_reference']=dict(root=str(tmp_path/'false-q'),inventory_sha256='b'*64)
    with pytest.raises((ValueError,FileNotFoundError)):
        m.load_practical_pipeline_costs(fake,protocol_reference=fake,directory=tmp_path/'study',**kwargs)
    assert list(tmp_path.iterdir())==[]


def test_cost_publication_cannot_mutate_programme_or_observer_evidence(tmp_path):
    m=api();assert hasattr(m,'_cost_output_location'),'Missing cost proof-path guard'
    from test_confirm_programme_v2 import fixture
    _,plan,reference=fixture(tmp_path/'known')
    for output in (Path(plan['directory'])/'cost.json',Path(plan['observer_directory'])/'cost.json'):
        with pytest.raises(ValueError):m._cost_output_location(output,plan,reference)
        assert not output.exists()
    out=tmp_path/'final-cost.json';assert m._cost_output_location(out,plan,reference)==out


def test_unknown_outer_episode_artifact_cannot_hide_a_partial_attempt(tmp_path):
    m=api();plan,reference,root=archived_negative_frames(tmp_path)
    (root/'episode-0000/unreported-failure.json').write_text('{}\n')
    with pytest.raises(ValueError):m._discover_confirmation_envelopes(reference,plan)


def test_overlapping_programme_invocations_cannot_be_added_as_serial_work():
    m=api();frames,inv,calls=intervals()
    frames[0]['invocation_ids'].append('inv-b')
    inv[1].update(clock_domain=domain('b'),origin_ns=145,end_ns=185)
    for offset,call in enumerate(calls[4:]):call.update(clock_domain=domain('b'),origin_ns=150+5*offset,end_ns=155+5*offset)
    with pytest.raises(ValueError):m._summarize_postT_intervals(frames,inv,calls)
