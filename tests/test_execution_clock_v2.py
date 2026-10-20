"""Closed clock records retain reboot boundaries without granting cost authority."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from test_primary_search_declaration_v2 import study
from test_primary_program_v2 import declared, no_child_episode


def api():
    assert importlib.util.find_spec('treeneuron.research.execution_clock_v2'), 'Missing explicit execution clock records'
    from treeneuron.research import execution_clock_v2
    return execution_clock_v2


@pytest.mark.parametrize('system,raw,want',[
    ('Linux','5a1aaf79-5043-49bb-9ac4-91b0be98f930\n','linux-boot-id:5a1aaf79-5043-49bb-9ac4-91b0be98f930'),
    ('Darwin','{ sec = 1780000000, usec = 42 } Fri May 29\n','darwin-boottime:1780000000:42'),
])
def test_kernel_identity_ignores_only_display_formatting(system,raw,want):
    assert api()._parse_boot_identity(system,raw)==want


@pytest.mark.parametrize('system,raw',[
    ('Linux',''),('Linux','not-a-boot-id'),('Darwin','{ sec = 0, usec = 2 }'),
    ('Darwin','{ sec = 1780000000, usec = 1000000 }'),('Darwin','garbage'),('Other','anything'),
])
def test_invalid_kernel_identity_cannot_become_a_shared_epoch(system,raw):
    with pytest.raises(ValueError):api()._parse_boot_identity(system,raw)


def test_real_domain_is_well_formed_and_unknown_probe_stays_unknown(monkeypatch):
    module=api();real=module.observe_clock_domain()
    module.validate_clock_domain(real)
    assert real['status'] in ('observed','unknown')
    if real['status']=='observed':assert module.same_clock_domain(real,copy.deepcopy(real))
    import platform
    monkeypatch.setattr(platform,'system',lambda:'Darwin')
    def unavailable(*args,**kwargs):raise OSError('software fixture: kernel query unavailable')
    monkeypatch.setattr(subprocess,'run',unavailable)
    unknown=module.observe_clock_domain()
    assert unknown['status']=='unknown' and unknown['boot_sha256'] is None
    assert unknown['reason']=='kernel_boot_identity_unavailable'
    assert not module.same_clock_domain(unknown,unknown)
    assert not module.same_clock_domain(real,unknown)


@pytest.mark.parametrize('key,value',[
    ('schema','wrong'),('clock','time_ns'),('system',True),('host_sha256','g'*64),
    ('boot_sha256',True),('status','qualified'),('reason','invented'),('clock_implementation',''),
])
def test_domain_shape_is_strict_and_never_uses_truthiness(key,value):
    module=api();domain=dict(schema='research/v2-monotonic-clock-domain',clock='perf_counter_ns',
        system='Linux',host_sha256='a'*64,boot_sha256='b'*64,status='observed',reason=None,
        clock_implementation='clock_gettime(CLOCK_MONOTONIC)')
    domain[key]=value
    with pytest.raises(ValueError):module.validate_clock_domain(domain)


def test_different_observed_boots_cannot_share_absolute_timestamps():
    module=api();first=dict(schema='research/v2-monotonic-clock-domain',clock='perf_counter_ns',
        system='Linux',host_sha256='a'*64,boot_sha256='b'*64,status='observed',reason=None,
        clock_implementation='clock_gettime(CLOCK_MONOTONIC)')
    second=dict(first,boot_sha256='c'*64)
    assert not module.same_clock_domain(first,second)


def test_import_and_observation_do_not_import_numerical_or_yaml_stacks():
    api()
    code="from treeneuron.research.execution_clock_v2 import observe_clock_domain; import sys; observe_clock_domain(); assert not {'torch','numpy','yaml','pandas','scipy'}.intersection(sys.modules)"
    completed=subprocess.run([sys.executable,'-P','-B','-c',code],capture_output=True,text=True,check=False)
    assert completed.returncode==0,completed.stderr


def test_closed_no_child_caller_binds_domain_to_start_and_survives_current_boot_change(tmp_path,monkeypatch):
    module=api()
    from treeneuron.research import primary_caller_v2
    from test_primary_caller_v2 import prepared
    root=tmp_path/'episode'
    result=primary_caller_v2._run_prepared_episode(prepared=prepared(root,tmp_path/'unused.py'),episode_root=root,
        identity=dict(fixture='clock-record'),admission_identity=dict(policy='software-no-child'),
        admission_check=lambda:dict(admitted=False),invocation_origin_ns=time.perf_counter_ns(),rss_limit_bytes=256*1024**2)
    reference=result['caller_reference'];path=Path(reference['path']);raw=path.read_bytes();value=json.loads(raw)
    start=json.loads((root/'caller-start.json').read_bytes())
    assert value['clock_domain']==start['clock_domain']
    assert value['elapsed_ns']==value['end_ns']-value['origin_ns']>0
    def forbidden():raise AssertionError('Stored closed cost must not depend on the current boot')
    monkeypatch.setattr(module,'observe_clock_domain',forbidden)
    proof=primary_caller_v2.inspect_caller_boundary(reference,process_reference=result['process_reference'])
    assert proof['clock_domain']==value['clock_domain']
    assert proof['origin_ns']==value['origin_ns'] and proof['end_ns']==value['end_ns']
    assert proof['whole_training_procedure_cost_authenticated'] is proof['protected_access_authorized'] is False
    for fault in ('domain','bounds','source','hash'):
        changed=copy.deepcopy(value)
        if fault=='domain':changed['clock_domain']['host_sha256']='c'*64
        elif fault=='bounds':changed['end_ns']+=1
        elif fault=='source':changed['caller_source_sha256']='d'*64
        replacement=json.dumps(changed).encode();path.write_bytes(replacement)
        pin=dict(reference,sha256=hashlib.sha256(replacement).hexdigest() if fault!='hash' else '0'*64)
        try:
            with pytest.raises(ValueError):primary_caller_v2.inspect_caller_boundary(pin,process_reference=result['process_reference'])
        finally:path.write_bytes(raw)


def test_program_invocation_reader_checks_closed_intervals_and_all_byte_pins(declared,monkeypatch):
    module=api()
    from treeneuron.research import primary_program_v2
    program,args=declared
    result=primary_program_v2._run_program_queue(program,execute_episode=no_child_episode,audit_terminal=lambda **kw:None)
    root=args['directory'];path=next((root/'invocations').glob('*.json'));raw=path.read_bytes();value=json.loads(raw)
    ref=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    program_reference=ref(root/'plan.json');reference=ref(path)
    assert value['elapsed_ns']==value['end_ns']-value['origin_ns']>=result['known_observed_caller_ns']>0
    def forbidden():raise AssertionError('Read-only clock inspection cannot run a process')
    monkeypatch.setattr(module,'observe_clock_domain',forbidden)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:forbidden())
    proof=module.inspect_program_invocation_clock(reference,program_reference=program_reference)
    assert proof['origin_ns']==value['origin_ns'] and proof['end_ns']==value['end_ns']
    assert proof['clock_domain']==value['clock_domain']
    assert proof['whole_training_procedure_cost_authenticated'] is proof['protected_access_authorized'] is False
    for fault in ('end','origin','domain','source','hash','authority','result','foreign_program'):
        changed=copy.deepcopy(value)
        if fault=='end':changed['end_ns']+=1
        elif fault=='origin':changed['origin_ns']=True
        elif fault=='domain':changed['clock_domain']['status']='qualified'
        elif fault=='source':changed['source_sha256']='c'*64
        elif fault=='authority':changed['whole_training_procedure_cost_authenticated']=True
        elif fault=='result':changed['result']['program_hash']='e'*64
        replacement=json.dumps(changed).encode();path.write_bytes(replacement)
        pin=dict(reference,sha256=hashlib.sha256(replacement).hexdigest() if fault!='hash' else '0'*64)
        program_pin=dict(program_reference,sha256='f'*64) if fault=='foreign_program' else program_reference
        try:
            with pytest.raises(ValueError):module.inspect_program_invocation_clock(pin,program_reference=program_pin)
        finally:path.write_bytes(raw)


def test_resumed_program_keeps_prior_closed_clock_readable_after_journal_grows(declared):
    module=api()
    from treeneuron.research import primary_program_v2
    program,args=declared
    first=primary_program_v2._run_program_queue(program,execute_episode=no_child_episode,audit_terminal=lambda **kw:None)
    root=args['directory'];path=next((root/'invocations').glob('*.json'))
    ref=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    reference=ref(path);plan=ref(root/'plan.json')
    before=module.inspect_program_invocation_clock(reference,program_reference=plan)
    second=primary_program_v2._run_program_queue(program,execute_episode=no_child_episode,audit_terminal=lambda **kw:None,
        expected_journal_sha256=first['journal_sha256'])
    assert second['journal_sha256']!=first['journal_sha256']
    after=module.inspect_program_invocation_clock(reference,program_reference=plan)
    assert before==after and after['journal_sha256']==first['journal_sha256']


def test_failed_driver_records_known_interval_without_inventing_nested_closure(declared):
    module=api()
    from treeneuron.research import primary_program_v2
    program,args=declared
    def interrupted(**kw):raise RuntimeError('fixture: no caller closure')
    with pytest.raises(RuntimeError):primary_program_v2._run_program_queue(program,execute_episode=interrupted,audit_terminal=lambda **kw:None)
    root=args['directory'];path=next((root/'invocations').glob('*.json'));value=json.loads(path.read_bytes())
    ref=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    proof=module.inspect_program_invocation_clock(ref(path),program_reference=ref(root/'plan.json'))
    assert proof['elapsed_ns']>0 and value['error']['type']=='RuntimeError' and value['result'] is None
    assert proof['whole_training_procedure_cost_authenticated'] is False
