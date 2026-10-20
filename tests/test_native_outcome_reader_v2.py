"""Paid native failures must retain their actual process/request disposition."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.native_outcome_reader_v2') is not None,'Missing native paid outcome reader'
    from treeneuron.research.native_outcome_reader_v2 import inspect_native_fit_outcome
    return inspect_native_fit_outcome


@pytest.fixture(scope='module')
def outcomes(tmp_path_factory):
    cached={}
    def get():
        if not cached:
            from test_method_worker import request_fixture,write_request
            from treeneuron.research.native_fit_v2 import run_native_fit
            root=tmp_path_factory.mktemp('native-outcomes')
            path,value=request_fixture(root/'input');value['budget']={'native_stopping':True}
            digest=write_request(path,value)
            for name,limit in (('mean',1024**3),('rss',1)):
                result=run_native_fit(path,expected_request_sha256=digest,output=root/name,
                    interpreter=sys.executable,rss_limit_bytes=limit)
                assert result['status']==('completed_ineligible' if name=='mean' else 'resource_exhausted')
                cached[name]=dict(request_path=path,expected_request_sha256=digest,native_reference=result['recovery_reference'])
        return cached
    return get


@pytest.mark.parametrize('name,status',[('mean','completed_ineligible'),('rss','resource_exhausted')])
def test_real_native_outcomes_reopen_without_running_a_worker_or_granting_quality(outcomes,name,status,monkeypatch):
    reader=api();row=outcomes()[name]
    from treeneuron.research.workqueue import _inventory
    from treeneuron.research import process_supervision_v2
    before=_inventory(Path(row['native_reference']['root']))
    def forbidden(*a,**kw):raise AssertionError('Closed outcome reader must not launch a worker')
    monkeypatch.setattr(process_supervision_v2,'supervise_process',forbidden)
    proof=reader(**row)
    assert proof['native_outcome_authenticated'] is True and proof['status']==status
    assert proof['native_workload_qualified'] is proof['quality_eligible'] is proof['T_accessed'] is False
    assert proof['recovery_lineage_authenticated'] is proof['whole_process_cost_authenticated'] is False
    assert proof['process']['process_closure_authenticated'] is True
    assert _inventory(Path(row['native_reference']['root']))==before


@pytest.mark.parametrize('fault',['reference','request_hash','source','fit_seed','status','authority',
    'process_hash','child_reaped','worker_hash','clock','unknown_field'])
def test_recommitted_bad_native_result_cannot_override_actual_managed_outcome(outcomes,fault):
    reader=api();row=copy.deepcopy(outcomes()['mean']);root=Path(row['native_reference']['root']);path=root/'result.json';before=path.read_bytes()
    try:
        value=json.loads(before)
        if fault=='reference':row['native_reference']['request_sha256']='0'*64
        elif fault=='request_hash':row['expected_request_sha256']='0'*64
        elif fault=='source':value['source_sha256']='0'*64
        elif fault=='fit_seed':value['fit_seed']+=1
        elif fault=='status':value['status']='resource_exhausted'
        elif fault=='authority':value['quality_eligible']=True
        elif fault=='process_hash':value['process_receipt_sha256']='0'*64
        elif fault=='child_reaped':value['child_reaped']=False
        elif fault=='worker_hash':value['worker_terminal_sha256']='0'*64
        elif fault=='clock':value['elapsed_before_recovery_inventory_ns']=0
        else:value['extra']='unpaid replacement'
        if fault not in ('reference','request_hash'):
            path.write_text(json.dumps(value))
            from treeneuron.config import canonical_hash
            from treeneuron.research.workqueue import _inventory
            row['native_reference']['inventory_sha256']=canonical_hash(_inventory(root))
        with pytest.raises(ValueError):reader(**row)
    finally:path.write_bytes(before)


def test_native_reader_rejects_a_recommitted_changed_process_command_even_when_other_links_match(outcomes):
    reader=api();row=copy.deepcopy(outcomes()['mean']);root=Path(row['native_reference']['root'])
    paths=[root/'result.json',root/'process/process.json',root/'process/started.json'];before={p:p.read_bytes() for p in paths}
    try:
        from treeneuron.research.method_worker import _sha
        from treeneuron.config import canonical_hash
        from treeneuron.research.workqueue import _inventory
        process=json.loads(before[paths[1]]);start=json.loads(before[paths[2]])
        process['argv'][3]='unmanaged.worker';start['argv']=process['argv'];paths[2].write_text(json.dumps(start))
        for ref in process['files']:
            p=root/'process'/ref['path'];ref.update(sha256=_sha(p),bytes=p.stat().st_size)
        paths[1].write_text(json.dumps(process));result=json.loads(before[paths[0]])
        result['process_receipt_sha256']=_sha(paths[1]);paths[0].write_text(json.dumps(result))
        row['native_reference']['inventory_sha256']=canonical_hash(_inventory(root))
        with pytest.raises(ValueError):reader(**row)
    finally:
        for p,raw in before.items():p.write_bytes(raw)


@pytest.mark.parametrize('fault',['clock_scope','pythonpath_type','worker_status'])
def test_typed_invocation_and_known_worker_disposition_required_after_recommit(outcomes,fault):
    reader=api();row=copy.deepcopy(outcomes()['mean']);root=Path(row['native_reference']['root'])
    paths=[root/'result.json',root/'process/process.json',root/'process/started.json',root/'worker/terminal.json']
    before={p:p.read_bytes() for p in paths}
    try:
        from treeneuron.research.method_worker import _sha
        from treeneuron.config import canonical_hash
        from treeneuron.research.workqueue import _inventory
        value=json.loads(before[paths[0]])
        if fault=='clock_scope':value['clock_scope']=None
        elif fault=='worker_status':
            terminal=json.loads(before[paths[3]]);terminal['status']='invented_disposition';paths[3].write_text(json.dumps(terminal))
            value.update(status='invented_disposition',worker_terminal_sha256=_sha(paths[3]))
            process=json.loads(before[paths[1]]);process['returncode']=2;paths[1].write_text(json.dumps(process))
            value['process_receipt_sha256']=_sha(paths[1])
        else:
            process=json.loads(before[paths[1]]);started=json.loads(before[paths[2]])
            process['environment_overrides']['PYTHONPATH']=[];started['environment_overrides']['PYTHONPATH']=[]
            paths[2].write_text(json.dumps(started))
            for ref in process['files']:
                p=root/'process'/ref['path'];ref.update(bytes=p.stat().st_size,sha256=_sha(p))
            paths[1].write_text(json.dumps(process));value['process_receipt_sha256']=_sha(paths[1])
        paths[0].write_text(json.dumps(value));row['native_reference']['inventory_sha256']=canonical_hash(_inventory(root))
        with pytest.raises(ValueError):reader(**row)
    finally:
        for p,raw in before.items():p.write_bytes(raw)
