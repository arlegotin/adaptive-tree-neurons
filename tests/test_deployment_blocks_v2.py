"""Paired process-block fixtures; no quality bound or deployment claim."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_deployment_worker_v2 import prepare,publish


def api():
    assert importlib.util.find_spec('treeneuron.research.deployment_blocks_v2') is not None, 'Missing paired deployment coordinator'
    from treeneuron.research import deployment_blocks_v2
    return deployment_blocks_v2


def requests(root):
    values={}
    for role in ('selected','baseline'):
        directory=root/role;directory.mkdir();path,value,digest=prepare(directory)
        values[role]=dict(path=str(path),sha256=digest)
    return values


def plan(root):
    return api().make_deployment_plan(requests(root),seed=17,interpreter=sys.executable,
        timeout_ns=60_000_000_000,rss_limit_bytes=1024**3)


def test_thirty_block_order_matches_existing_paired_cost_arithmetic_and_never_selects_a_seed(tmp_path):
    module=api();value=plan(tmp_path);jobs=value['attempts']
    assert len(jobs)==60 and value['identity']['block_count']==30
    rows=[]
    for index in range(30):
        pair=jobs[index*2:index*2+2]
        assert {j['payload']['role'] for j in pair}=={'selected','baseline'}
        assert all(j['payload']['block_id']==index for j in pair)
        assert pair[1]['depends_on']==[pair[0]['job_id']] and pair[1]['dependency_policy']=='all_terminal'
        rows.append(dict(block_id=index,order='selected_first' if pair[0]['payload']['role']=='selected' else 'baseline_first',
            selected_ns=500,baseline_ns=1000))
    from treeneuron.analysis.cost_v2 import paired_timing_savings
    result=paired_timing_savings(rows,seed=17,quality_risk_ratio_upper=1.01,iterations=20)
    assert result['block_count']==30 and result['cost_claim_authorized'] is False
    assert value['identity']['cost_claim_authorized'] is False


@pytest.mark.parametrize('change',['query','warmups','seed_bool','missing_role','changed_plan'])
def test_changed_pairing_or_execution_contract_cannot_start_a_queue(change,tmp_path):
    module=api();refs=requests(tmp_path);kwargs=dict(seed=17,interpreter=sys.executable,
        timeout_ns=60_000_000_000,rss_limit_bytes=1024**3)
    if change in ('query','warmups'):
        path=Path(refs['baseline']['path']);value=json.loads(path.read_text())
        if change=='query':
            query=path.parent/'query.npz';query.write_bytes(query.read_bytes()+b'identity differs')
            value['query']['sha256']=hashlib.sha256(query.read_bytes()).hexdigest()
        else:value['warmup_calls']=2
        refs['baseline']['sha256']=publish(path,value)
    elif change=='seed_bool':kwargs['seed']=True
    elif change=='missing_role':refs.pop('baseline')
    if change=='changed_plan':
        value=module.make_deployment_plan(refs,**kwargs);value['identity']['seed']=99
        with pytest.raises(ValueError):module.run_deployment_blocks(value,tmp_path/'queue')
        assert not (tmp_path/'queue').exists()
    else:
        with pytest.raises(ValueError):module.make_deployment_plan(refs,**kwargs)


def test_live_shortage_pauses_before_any_owned_child_and_retains_the_fixed_plan(tmp_path,monkeypatch):
    module=api();value=plan(tmp_path)
    monkeypatch.setattr(module,'live_resources',lambda _:dict(free_disk_bytes=10**12,available_memory_bytes=0))
    def forbidden(**kwargs):raise AssertionError('Live admission must precede child start')
    monkeypatch.setattr(module,'supervise_process',forbidden)
    result=module.run_deployment_blocks(value,tmp_path/'queue')
    assert result['status']=='paused_before_job' and len(result['statuses'])==0
    assert json.loads((tmp_path/'queue/plan.json').read_text())['plan_hash']==value['plan_hash']
    assert not list((tmp_path/'queue').rglob('process.json'))


def test_actual_thirty_paired_owned_process_blocks_keep_cold_and_resident_costs_separate(tmp_path):
    module=api();value=plan(tmp_path);result=module.run_deployment_blocks(value,tmp_path/'queue')
    assert result['status']=='completed',result
    report=module.collect_deployment_blocks(tmp_path/'queue',expected_plan_hash=value['plan_hash'])
    assert report['status']=='complete_engineering_measurements' and report['missing_members']==[]
    assert report['completed_processes']==60 and report['all_children_reaped']
    assert report['cost_claim_authorized'] is report['dedicated_timing_qualified'] is False
    for size in ('1','1024'):
        blocks=report['resident_blocks'][size];assert len(blocks)==30
        assert all(b['selected_ns']>0 and b['baseline_ns']>0 for b in blocks)
    assert len(report['cold_process_blocks'])==len(report['model_load_blocks'])==30
    assert report['sum_disjoint_process_enclosing_ns']<=report['charged_queue_job_ns']
    for p in (tmp_path/'queue').rglob('process.json'):
        record=json.loads(p.read_text());assert record['child_reaped'] and record['returncode']==0
        assert record['environment_overrides']['KMP_DUPLICATE_LIB_OK'] is None


def test_paused_partial_blocks_are_reported_without_complete_case_inference(tmp_path,monkeypatch):
    module=api();value=plan(tmp_path);calls=0
    def probe(_):
        nonlocal calls
        calls+=1
        return dict(free_disk_bytes=10**12,available_memory_bytes=10**12 if calls<=2 else 0)
    monkeypatch.setattr(module,'live_resources',probe)
    result=module.run_deployment_blocks(value,tmp_path/'queue');assert result['status']=='paused_before_job'
    report=module.collect_deployment_blocks(tmp_path/'queue',expected_plan_hash=value['plan_hash'])
    assert report['status']=='incomplete_measurements' and len(report['missing_members'])==58
    assert report['resident_blocks'] is None and report['cold_process_blocks'] is None
    assert report['completed_processes']==2 and report['cost_claim_authorized'] is False


def test_nonzero_exit_cannot_be_promoted_by_an_earlier_success_terminal(tmp_path,monkeypatch):
    module=api();value=plan(tmp_path);calls=0
    def probe(_):
        nonlocal calls
        calls+=1
        return dict(free_disk_bytes=10**12,available_memory_bytes=10**12 if calls==1 else 0)
    monkeypatch.setattr(module,'live_resources',probe)
    def supervisor(**kwargs):
        # Explicit simulated process receipt: this test starts no child.
        output=Path(kwargs['output']);output.mkdir(parents=True)
        worker=output.parent/'worker';worker.mkdir()
        (worker/'terminal.json').write_text(json.dumps(dict(status='completed_ineligible')))
        result=dict(execution_status='exited',returncode=17,child_started=True,child_reaped=True,
            identity=kwargs['identity'],elapsed_ns=1,fixture_only=True)
        (output/'process.json').write_text(json.dumps(result));return result
    monkeypatch.setattr(module,'supervise_process',supervisor)
    result=module.run_deployment_blocks(value,tmp_path/'queue')
    assert result['status']=='paused_before_job'
    assert list(result['statuses'].values())==['input_or_dependency_failure']


def test_orphan_running_member_retains_status_and_unknown_cleanup_instead_of_becoming_absent(tmp_path,monkeypatch):
    module=api();value=plan(tmp_path)
    monkeypatch.setattr(module,'live_resources',lambda _:dict(free_disk_bytes=10**12,available_memory_bytes=0))
    module.run_deployment_blocks(value,tmp_path/'queue')
    from treeneuron.research.ledger import AttemptLedger
    ledger=AttemptLedger.resume(tmp_path/'queue',value)
    ledger.record(value['attempts'][0]['job_id'],'running',elapsed_ns=1,recovery_attempt=0,started_utc_ns=1)
    report=module.collect_deployment_blocks(tmp_path/'queue',expected_plan_hash=value['plan_hash'])
    assert report['missing_members'][0]['status']=='running'
    assert report['all_children_reaped'] is None and report['resident_blocks'] is None
