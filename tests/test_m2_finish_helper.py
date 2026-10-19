"""Read-only delayed collection cannot consume active or partial outcomes."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def api():
    path=Path(__file__).resolve().parents[1]/'docs/research/v2/scripts/finish_m2.py'
    spec=importlib.util.spec_from_file_location('m2_finish_helper',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def setup(tmp_path):
    module=api();actors=[]
    for name in ('producer','auditor'):
        root=tmp_path/name;(root/'src/treeneuron').mkdir(parents=True)
        (root/'src/treeneuron/__init__.py').write_text('# isolated source identity fixture\n')
        actors.append(dict(root=str(root),source_sha256=module.package_digest(root)))
    study=tmp_path/'study';study.mkdir();(study/'worker-lease.json').write_text('{"pid":123}\n')
    plan=dict(schema='research/v2-workqueue',attempts=[dict(job_id='first'),dict(job_id='second')])
    plan['plan_hash']=module.canonical_hash(plan)
    (study/'plan.json').write_text(json.dumps(plan));(study/'events.jsonl').write_text('')
    control=tmp_path/'control.json';control.write_text('{}')
    producer_config=Path(actors[0]['root'])/'configs/research/v2/M2-opportunity.json'
    producer_config.parent.mkdir(parents=True);producer_config.write_text('{}')
    auditor_helper=Path(actors[1]['root'])/'docs/research/v2/scripts/audit_m2_predictions.py'
    auditor_helper.parent.mkdir(parents=True);auditor_helper.write_text('# isolated helper fixture\n')
    config=dict(schema='research/v2-M2-delayed-collection',study=str(study),producer=actors[0],auditor=actors[1],
        expected_queue_hash=plan['plan_hash'],producer_config='configs/research/v2/M2-opportunity.json',
        producer_config_sha256=hashlib.sha256(producer_config.read_bytes()).hexdigest(),
        auditor_helper_sha256=hashlib.sha256(auditor_helper.read_bytes()).hexdigest(),
        control_audit=str(control),control_audit_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),
        control_plan_hash='0'*64,interpreter='/usr/bin/python3')
    return module,config,study


def events(module,study,statuses):
    previous='0'*64;records=[]
    for name,status in statuses:
        event=dict(attempt_id=name,status=status,previous_hash=previous)
        event['event_hash']=module.canonical_hash(event);previous=event['event_hash'];records.append(event)
    (study/'events.jsonl').write_text(''.join(json.dumps(record)+'\n' for record in records))
    return previous


def test_active_queue_is_not_read_and_lease_metadata_stays_identical(tmp_path):
    module,config,study=setup(tmp_path);before=(study/'worker-lease.json').read_bytes()
    (study/'events.jsonl').write_text('unreadable partial events')
    with (study/'worker-lease.json').open('r') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert module.probe(config)['status']=='producer_active'
    assert (study/'worker-lease.json').read_bytes()==before


def test_inactive_partial_and_failure_inventory_cannot_enter_collection(tmp_path):
    module,config,study=setup(tmp_path)
    assert module.probe(config)['status']=='incomplete_queue'
    events(module,study,[('first','completed_eligible'),('second','numerical_failure')])
    assert module.probe(config)['status']=='failed_queue'
    called=[]
    with pytest.raises(ValueError):module.collect(config,tmp_path/'output',run_stage=lambda *args,**kwargs:called.append(args))
    assert called==[] and not (tmp_path/'output').exists()


@pytest.mark.parametrize('change',['producer','auditor','control','plan','ledger','producer_config','auditor_helper'])
def test_byte_or_commitment_drift_fails_before_any_stage(tmp_path,change):
    module,config,study=setup(tmp_path)
    events(module,study,[('first','completed_eligible'),('second','completed_ineligible')])
    if change in ('producer','auditor'):
        (Path(config[change]['root'])/'src/treeneuron/__init__.py').write_text('# changed')
    elif change=='control':Path(config['control_audit']).write_text('{"changed":true}')
    elif change=='producer_config':(Path(config['producer']['root'])/config['producer_config']).write_text('{"changed":true}')
    elif change=='auditor_helper':(Path(config['auditor']['root'])/'docs/research/v2/scripts/audit_m2_predictions.py').write_text('# changed')
    elif change=='plan':(study/'plan.json').write_text('{}')
    else:
        text=(study/'events.jsonl').read_text().replace('completed_eligible','completed_ineligible')
        (study/'events.jsonl').write_text(text)
    with pytest.raises(ValueError):module.probe(config)


def test_ready_collection_calls_only_frozen_collection_then_independent_audit_once(tmp_path):
    module,config,study=setup(tmp_path)
    endpoint=events(module,study,[('first','completed_eligible'),('second','completed_ineligible')])
    calls=[]
    def run(stage,**kw):
        calls.append((stage,kw));return dict(status='completed',ledger_event_hash=endpoint)
    result=module.collect(config,tmp_path/'output',run_stage=run)
    assert [stage for stage,kw in calls]==['producer_collection','independent_prediction_audit']
    assert calls[0][1]['actor']==config['producer'] and calls[1][1]['actor']==config['auditor']
    assert all(kw['endpoint']==endpoint for stage,kw in calls)
    assert result['status']=='collected_and_audited' and result['T_accessed'] is False
    assert (tmp_path/'output/terminal.json').is_file()
    with pytest.raises(ValueError):module.collect(config,tmp_path/'output',run_stage=run)
    assert len(calls)==2


def test_collection_failure_keeps_error_and_does_not_run_audit_or_retry(tmp_path):
    module,config,study=setup(tmp_path)
    events(module,study,[('first','completed_eligible'),('second','completed_ineligible')])
    calls=[]
    def fail(stage,**kw):calls.append(stage);raise RuntimeError('controlled collector failure')
    with pytest.raises(RuntimeError,match='controlled'):module.collect(config,tmp_path/'output',run_stage=fail)
    result=json.loads((tmp_path/'output/terminal.json').read_text())
    assert calls==['producer_collection'] and result['status']=='postprocessing_failed'
    assert result['T_accessed'] is False and result['failure']['message']=='controlled collector failure'


def test_collector_holds_shared_lease_for_both_stages(tmp_path):
    module,config,study=setup(tmp_path)
    endpoint=events(module,study,[('first','completed_eligible'),('second','completed_ineligible')])
    def run(stage,**kw):
        with (study/'worker-lease.json').open('r') as stream:
            with pytest.raises(BlockingIOError):fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return dict(ledger_event_hash=endpoint)
    assert module.collect(config,tmp_path/'output',run_stage=run)['status']=='collected_and_audited'


def test_collector_different_endpoint_stops_before_audit(tmp_path):
    module,config,study=setup(tmp_path)
    events(module,study,[('first','completed_eligible'),('second','completed_ineligible')])
    calls=[]
    def run(stage,**kw):calls.append(stage);return dict(ledger_event_hash='0'*64)
    with pytest.raises(ValueError,match='different ledger'):module.collect(config,tmp_path/'output',run_stage=run)
    assert calls==['producer_collection']


def test_parent_path_escape_cannot_create_outputs_inside_study(tmp_path):
    module,config,study=setup(tmp_path)
    endpoint=events(module,study,[('first','completed_eligible'),('second','completed_ineligible')])
    path=tmp_path/'alias';path.symlink_to(study,target_is_directory=True)
    with pytest.raises(ValueError):
        module.collect(config,path/'new',run_stage=lambda *a,**k:dict(ledger_event_hash=endpoint))
    assert not (study/'new').exists()


def test_real_cli_waits_on_partial_without_starting_numeric_child(tmp_path):
    module,config,study=setup(tmp_path)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    helper=Path(module.__file__)
    result=subprocess.run([sys.executable,'-B',str(helper),'--config',str(path),
        '--expected-config-sha256',hashlib.sha256(path.read_bytes()).hexdigest(),
        '--output',str(tmp_path/'out'),'--watch-record',str(tmp_path/'watch'),
        '--poll-seconds','.01','--max-wait-seconds','.03'],capture_output=True,text=True)
    assert result.returncode==3,result.stderr
    receipt=json.loads((tmp_path/'watch/terminal.json').read_text())
    assert receipt['result']['status']=='watch_window_expired' and receipt['failure'] is None
    assert not (tmp_path/'out').exists()


def test_real_sequential_children_on_artificial_actors(tmp_path):
    module,config,study=setup(tmp_path)
    endpoint=events(module,study,[('first','completed_eligible'),('second','completed_ineligible')])
    producer=Path(config['producer']['root']);package=producer/'src/treeneuron/research'
    package.mkdir();(package/'__init__.py').write_text('')
    (producer/'torch.py').write_text('def set_num_threads(n): assert n==1\ndef set_num_interop_threads(n): assert n==1\n')
    (package/'config.py').write_text('''def load_research_config(path): return path
class Fixture:
    def to_dict(self): return {}
def expand(value): return Fixture()
''')
    (package/'opportunity_collection.py').write_text('def collect_opportunity(*a,**kw):\n'
        f'    return dict(ledger_event_hash={endpoint!r},T_accessed=False)\n')
    (package/'ledger.py').write_text('import json\ndef write_new_json(path,value):\n'
        '    with path.open("x") as stream: json.dump(value,stream)\n')
    helper=Path(config['auditor']['root'])/'docs/research/v2/scripts/audit_m2_predictions.py'
    helper.write_text('''import json,sys
from pathlib import Path
args=sys.argv;out=Path(args[args.index('--output')+1]);out.mkdir()
endpoint=args[args.index('--expected-ledger-hash')+1]
(out/'audit.json').write_text(json.dumps(dict(status='completed_arithmetic',ledger_event_hash=endpoint,T_accessed=False)))
''')
    config['auditor_helper_sha256']=module.sha(helper)
    config['producer']['source_sha256']=module.package_digest(producer)
    config['interpreter']=sys.executable
    result=module.collect(config,tmp_path/'output')
    assert result['status']=='collected_and_audited'
    for stage in ('producer_collection','independent_prediction_audit'):
        receipt=json.loads((tmp_path/'output'/stage/'terminal.json').read_text())
        assert receipt['exit_code']==0 and receipt['reaped'] is True
