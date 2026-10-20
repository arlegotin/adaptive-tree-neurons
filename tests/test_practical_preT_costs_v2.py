"""Cost arithmetic and false-gate fixtures confer no actual native success."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from test_primary_search_declaration_v2 import study
from test_primary_program_v2 import declared,no_child_episode


def api():
    assert importlib.util.find_spec('treeneuron.research.practical_preT_costs_v2'), 'Missing pre-T practical execution-cost join'
    from treeneuron.research import practical_preT_costs_v2
    return practical_preT_costs_v2


def domain(boot):
    return dict(schema='research/v2-monotonic-clock-domain',clock='perf_counter_ns',system='Linux',
        host_sha256='a'*64,boot_sha256=boot*64,status='observed',reason=None,
        clock_implementation='clock_gettime(CLOCK_MONOTONIC)')


def observations(second_boot='b'):
    second_origin=300 if second_boot=='b' else 10
    outer=[dict(envelope_id='first',clock_domain=domain('b'),origin_ns=100,end_ns=200,elapsed_ns=100),
        dict(envelope_id='second',clock_domain=domain(second_boot),origin_ns=second_origin,end_ns=second_origin+50,elapsed_ns=50)]
    calls=[dict(call_id='one',job_id='method-A',envelope_id='first',clock_domain=domain('b'),origin_ns=110,end_ns=150,elapsed_ns=40,caller_elapsed_ns=30),
        dict(call_id='two',job_id='method-B',envelope_id='second',clock_domain=domain(second_boot),
            origin_ns=second_origin+10,end_ns=second_origin+40,elapsed_ns=30,caller_elapsed_ns=25)]
    return outer,calls


def test_containing_clock_and_inner_caller_are_never_added_twice():
    outer,calls=observations()
    value=api()._summarize_intervals(outer,calls,expected_job_ids=['method-A','method-B'],controller_paid_ns={'method-A':35,'method-B':29})
    assert value['physical_worker_ns']==150
    assert value['standalone_full_call_ns']=={'method-A':40,'method-B':30}
    assert value['known_inner_caller_ns']==55
    assert value['common_program_overhead_ns']==80
    assert value['observed_calendar_span_ns']==250 and value['observed_idle_ns']==100
    assert value['whole_training_procedure_cost_authenticated'] is value['protected_access_authorized'] is False


def test_reboot_allows_elapsed_sum_without_inventing_calendar_wall_or_idle():
    outer,calls=observations(second_boot='c')
    value=api()._summarize_intervals(outer,calls,expected_job_ids=['method-A','method-B'],controller_paid_ns={'method-A':35,'method-B':29})
    assert value['physical_worker_ns']==150 and value['common_program_overhead_ns']==80
    assert value['observed_calendar_span_ns'] is value['observed_idle_ns'] is None
    assert value['observed_clock_domain_count']==2


@pytest.mark.parametrize('fault',['duplicate_outer','duplicate_call','missing_job','unknown_domain','foreign_domain','escape',
    'overlap_calls','overlap_outer','boolean_clock','changed_elapsed','shorter_than_caller','unpaid_controller'])
def test_ambiguous_or_incomplete_clock_routing_cannot_pass_cost_join(fault):
    outer,calls=observations();jobs=['method-A','method-B'];paid={'method-A':35,'method-B':29}
    if fault=='duplicate_outer':outer.append(copy.deepcopy(outer[0]))
    elif fault=='duplicate_call':calls.append(copy.deepcopy(calls[0]))
    elif fault=='missing_job':jobs.append('method-C')
    elif fault=='unknown_domain':calls[0]['clock_domain'].update(status='unknown',boot_sha256=None,reason='kernel_boot_identity_unavailable')
    elif fault=='foreign_domain':calls[0]['clock_domain']=domain('c')
    elif fault=='escape':calls[0].update(origin_ns=90,end_ns=130)
    elif fault=='overlap_calls':calls.append(dict(calls[0],call_id='three',origin_ns=120,end_ns=160))
    elif fault=='overlap_outer':outer[1].update(origin_ns=150,end_ns=200)
    elif fault=='boolean_clock':calls[0]['origin_ns']=True
    elif fault=='changed_elapsed':calls[0]['elapsed_ns']=39
    elif fault=='shorter_than_caller':calls[0]['caller_elapsed_ns']=41
    else:paid['method-A']=41
    with pytest.raises(ValueError):api()._summarize_intervals(outer,calls,expected_job_ids=jobs,controller_paid_ns=paid)


def test_public_seal_requires_real_complete_models_before_cost_inputs_or_output(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        api().seal_practical_preT_costs(models_reference=dict(path=str(tmp_path/'forbidden-models'),sha256='a'*64),
            global_barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='b'*64),
            qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='c'*64),
            program_bundles={'primary':{'forbidden':'must not be read'}},output=tmp_path/'cost.json')
    assert 'forbidden-models' not in str(error.value) and list(tmp_path.iterdir())==[]


def test_public_cost_readback_reopens_real_models_before_cost_record(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        api().load_practical_preT_costs(dict(path=str(tmp_path/'forbidden-cost'),sha256='a'*64),
            models_reference=dict(path=str(tmp_path/'forbidden-models'),sha256='a'*64),
            global_barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='b'*64),
            qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='c'*64))
    assert 'forbidden-cost' not in str(error.value) and 'forbidden-models' not in str(error.value)
    assert list(tmp_path.iterdir())==[]


def test_private_file_collector_reopens_actual_partial_queue_without_native_authority(declared):
    module=api()
    from treeneuron.research.primary_program_v2 import _run_program_queue
    program,args=declared
    result=_run_program_queue(program,execute_episode=no_child_episode,audit_terminal=lambda **kw:None)
    p=args['directory']/'plan.json';reference=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    collected=module._collect_program(reference,expected_journal_sha256=result['journal_sha256'])
    assert collected['completed_jobs']==0 and len(collected['calls'])==len(collected['invocations'])==1
    assert collected['calls'][0]['job_id']==program['jobs'][0]['job_id']
    assert collected['calls'][0]['caller_elapsed_ns']==result['known_observed_caller_ns']
    assert collected['controller_paid_ns']=={}
    assert collected['native_qualification_authenticated'] is collected['whole_training_procedure_cost_authenticated'] is False
    call=next((args['directory']/'call-clocks').rglob('*.json'));before=call.read_bytes();call.write_bytes(before+b' ')
    try:
        with pytest.raises(ValueError):module._collect_program(reference,expected_journal_sha256=result['journal_sha256'])
    finally:call.write_bytes(before)


def test_unclosed_queue_is_not_a_complete_cost_collection(declared):
    module=api()
    from treeneuron.research.primary_program_v2 import _run_program_queue
    program,args=declared
    def interrupted(**kw):raise RuntimeError('fixture: no caller output')
    with pytest.raises(RuntimeError):_run_program_queue(program,execute_episode=interrupted,audit_terminal=lambda **kw:None)
    p=args['directory']/'plan.json';journal=args['directory']/'events.jsonl'
    with pytest.raises((ValueError,PermissionError)):module._collect_program(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
        expected_journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest())
