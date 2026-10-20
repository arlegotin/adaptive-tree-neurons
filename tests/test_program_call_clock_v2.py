"""A method call includes its final return; queue bookkeeping stays separate."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import pytest

from test_primary_search_declaration_v2 import study
from test_primary_program_v2 import declared,no_child_episode


def api():
    assert importlib.util.find_spec('treeneuron.research.program_call_clock_v2'), 'Missing full program call clock'
    from treeneuron.research import program_call_clock_v2
    return program_call_clock_v2


@pytest.fixture
def clock(declared):
    api()
    from treeneuron.research.primary_program_v2 import _run_program_queue
    program,args=declared
    def delayed_return(**kw):
        result=no_child_episode(**kw)
        time.sleep(.025)
        return result
    result=_run_program_queue(program,execute_episode=delayed_return,audit_terminal=lambda **kw:None)
    path=next((args['directory']/'call-clocks').rglob('*.json'))
    ref=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return program,args,result,path,ref


def test_full_call_clock_includes_caller_publication_and_actual_return(clock,monkeypatch):
    module=api();program,args,result,path,ref=clock
    value=json.loads(path.read_bytes());caller=json.loads(Path(value['caller_reference']['path']).read_bytes())
    def forbidden(*a,**kw):raise AssertionError('Full call reader cannot spawn or inspect saved PIDs')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    proof=module.inspect_program_call_clock(ref,program=program)
    assert value['end_ns']-caller['end_ns']>=25_000_000
    assert proof['elapsed_ns']>caller['elapsed_ns']>0
    assert proof['elapsed_ns']==proof['end_ns']-proof['origin_ns']
    assert proof['caller_boundary_checked'] is True
    assert proof['whole_training_procedure_cost_authenticated'] is proof['protected_access_authorized'] is False
    assert result['known_observed_caller_ns']==caller['elapsed_ns']


def test_raised_driver_after_closed_caller_keeps_full_clock_and_exact_recovery(declared,monkeypatch):
    module=api()
    from treeneuron.research.primary_program_v2 import _run_program_queue
    program,args=declared
    def interrupted(**kw):no_child_episode(**kw);raise RuntimeError('driver stopped after caller return')
    with pytest.raises(RuntimeError):_run_program_queue(program,execute_episode=interrupted,audit_terminal=lambda **kw:None)
    path=next((args['directory']/'call-clocks').rglob('*.json'))
    ref=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    proof=module.inspect_program_call_clock(ref,program=program)
    assert proof['disposition']=='raised' and proof['error_type']=='RuntimeError' and proof['caller_boundary_checked']
    journal=args['directory']/'events.jsonl';before=journal.read_bytes()
    def forbidden(*a,**kw):raise AssertionError('Recovery cannot replace a closed caller')
    import treeneuron.research.execution_clock_v2 as execution_clock
    saved_domain=copy.deepcopy(proof['clock_domain'])
    monkeypatch.setattr(execution_clock,'observe_clock_domain',lambda:copy.deepcopy(saved_domain))
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    result=_run_program_queue(program,execute_episode=forbidden,audit_terminal=forbidden,
        expected_journal_sha256=hashlib.sha256(before).hexdigest())
    assert result['status']=='resource_pause' and result['completed_jobs']==0
    assert path.read_bytes() and hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
    assert len(list((args['directory']/'call-clocks').rglob('*.json')))==1


def test_failure_before_caller_exists_never_invents_a_closed_boundary(declared):
    module=api()
    from treeneuron.research.primary_program_v2 import _run_program_queue
    program,args=declared
    def failed(**kw):raise RuntimeError('fixture: before caller output')
    with pytest.raises(RuntimeError):_run_program_queue(program,execute_episode=failed,audit_terminal=lambda **kw:None)
    path=next((args['directory']/'call-clocks').rglob('*.json'));value=json.loads(path.read_bytes())
    proof=module.inspect_program_call_clock(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),program=program)
    assert value['caller_reference'] is None and proof['caller_boundary_checked'] is False
    assert proof['disposition']=='raised' and proof['elapsed_ns']>0
    assert proof['whole_training_procedure_cost_authenticated'] is False


@pytest.mark.parametrize('fault',['hash','source','domain','bounds','origin_boolean','caller','job','authority'])
def test_full_call_reader_rejects_changed_source_clocks_and_caller_binding(clock,fault):
    module=api();program,args,result,path,ref=clock;before=path.read_bytes();value=json.loads(before)
    if fault=='source':value['source_sha256']='0'*64
    elif fault=='domain':value['clock_domain']['status']='qualified'
    elif fault=='bounds':value['end_ns']+=1
    elif fault=='origin_boolean':value['origin_ns']=True
    elif fault=='caller':value['caller_reference']['sha256']='0'*64
    elif fault=='job':value['job_id']='foreign'
    elif fault=='authority':value['whole_training_procedure_cost_authenticated']=True
    raw=json.dumps(value).encode();path.write_bytes(raw)
    pin=dict(ref,sha256='0'*64 if fault=='hash' else hashlib.sha256(raw).hexdigest())
    try:
        with pytest.raises(ValueError):module.inspect_program_call_clock(pin,program=program)
    finally:path.write_bytes(before)
