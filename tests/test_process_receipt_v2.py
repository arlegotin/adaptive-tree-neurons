"""Operational failures remain paid closed evidence, never predictive losses."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.process_receipt_v2') is not None,'Missing closed failed/successful process reader'
    from treeneuron.research import process_receipt_v2
    return process_receipt_v2


def reference(root):
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import _inventory
    return dict(root=str(root),inventory_sha256=canonical_hash(_inventory(root)))


@pytest.fixture(scope='module')
def processes(tmp_path_factory):
    cached={}
    def get():
        if not cached:
            from treeneuron.research.process_supervision_v2 import supervise_process
            root=tmp_path_factory.mktemp('closed-process-outcomes');bad=root/'nonexecutable';bad.write_text('not executable')
            for name,argv,limit,timeout in (
                ('normal',[sys.executable,'-c','import time; print("ok",flush=True); time.sleep(.1)'],64*1024**2,None),
                ('nonzero',[sys.executable,'-c','raise SystemExit(7)'],64*1024**2,None),
                ('rss',[sys.executable,'-c','import time; time.sleep(5)'],1,None),
                ('timeout',[sys.executable,'-c','import time; time.sleep(5)'],64*1024**2,500_000_000),
                ('not_started',[str(bad)],64*1024**2,None)):
                path=root/name;env={'PYTHONDONTWRITEBYTECODE':'1'};identity=dict(fixture=name)
                result=supervise_process(argv=argv,cwd=root,env_overrides=env,output=path,timeout_ns=timeout,
                    rss_limit_bytes=limit,poll_interval_ns=10_000_000,identity=identity)
                cached[name]=dict(reference=reference(path),expected_argv=argv,expected_cwd=str(root),
                    expected_environment_overrides=env,expected_identity=identity,actual=result)
        return cached
    return get


@pytest.mark.parametrize('name,status,code',[('normal','exited',0),('nonzero','exited',7),
    ('rss','resource_exhausted',None),('timeout','timeout',None),('not_started','input_or_dependency_failure',None)])
def test_actual_closed_outcomes_keep_operational_status_without_model_authority(processes,name,status,code):
    module=api();row=processes()[name];actual=row['actual']
    proof=module.inspect_closed_process(**{k:v for k,v in row.items() if k!='actual'})
    assert proof['process_closure_authenticated'] is True and proof['execution_status']==status
    if code is not None:assert proof['returncode']==code
    assert proof['elapsed_ns']>0
    if name=='rss':assert proof['observed_peak_root_rss_bytes']>actual['rss_limit_bytes']
    if name=='not_started':assert proof['child_started'] is False and proof['returncode'] is None
    assert proof['eligible_model'] is proof['native_recipe_runtime_qualified'] is proof['protected_access_authorized'] is False


@pytest.mark.parametrize('fault',['unreaped','rss_count','authority','argv','overshoot','termination',
    'status_type','reason_type','file_row_type'])
def test_recommitted_bad_closed_process_records_reject(processes,fault):
    module=api();row=processes()['normal'];root=Path(row['reference']['root']);path=root/'process.json';start=root/'started.json'
    before={p:p.read_bytes() for p in (path,start)}
    try:
        value=json.loads(before[path])
        if fault=='unreaped':value['child_reaped']=False
        elif fault=='rss_count':value['rss_sample_count']+=1
        elif fault=='authority':value['eligible_model']=True
        elif fault=='overshoot':value['deadline_overshoot_ns']=1
        elif fault=='termination':value['termination']['kill_sent']=True
        elif fault=='status_type':value['execution_status']=[]
        elif fault=='reason_type':value['stop_reason']=[]
        elif fault=='file_row_type':value['files'][0]=None
        else:
            value['argv'][0]='/bin/echo';started=json.loads(before[start]);started['argv']=value['argv'];start.write_text(json.dumps(started))
            import hashlib
            for ref in value['files']:
                p=root/ref['path'];ref.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        path.write_text(json.dumps(value));arguments={k:v for k,v in row.items() if k!='actual'};arguments['reference']=reference(root)
        with pytest.raises(ValueError):module.inspect_closed_process(**arguments)
    finally:
        for p,raw in before.items():p.write_bytes(raw)
        assert reference(root)==row['reference']
