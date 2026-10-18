"""Pauses leave an immutable full queue and no fictitious failed attempts."""
import json

import pytest

from treeneuron.research.ledger import AttemptLedger
from treeneuron.research.workqueue import make_queue_plan, run_queue


def plan():
    return make_queue_plan({'study':'tranche-fixture'},[
        dict(job_id='parent',kind='parent',depends_on=[],payload={}),
        dict(job_id='growth',kind='fit',depends_on=['parent'],payload={}),
        dict(job_id='ordinary',kind='fit',depends_on=['parent'],payload={}),
        dict(job_id='compare',kind='evaluate',depends_on=['growth','ordinary'],payload={}),
    ])


def handler(seen):
    def execute(job,directory,dependencies,attempt):
        seen.append((job['job_id'],attempt))
        (directory/'prediction.json').write_text(json.dumps({'prediction':len(dependencies)+1}))
        return dict(status='completed_eligible',prediction=len(dependencies)+1)
    return execute


def test_dependency_closed_tranche_keeps_full_plan_and_next_jobs_untouched(tmp_path):
    p=plan();seen=[];root=tmp_path/'queue'
    result=run_queue(p,root,handler(seen),check_identity=lambda _:None,admitted_job_ids=['growth','parent'])
    assert seen==[('parent',0),('growth',0)] and result['status']=='incomplete'
    assert json.loads((root/'plan.json').read_text())==p
    assert {e['attempt_id'] for e in AttemptLedger.resume(root,p).events}=={'parent','growth'}
    assert result['admitted_job_ids']==['parent','growth']
    result=run_queue(p,root,handler(seen),check_identity=lambda _:None,resume=True)
    assert result['status']=='completed' and seen==[('parent',0),('growth',0),('ordinary',0),('compare',0)]


@pytest.mark.parametrize('ids',[[],['missing'],['parent','parent'],['growth'],['parent','compare'],['parent',7],'parent'])
def test_invalid_tranche_fails_before_creating_study(tmp_path,ids):
    with pytest.raises(ValueError,match='tranche'):
        run_queue(plan(),tmp_path/'queue',handler([]),check_identity=lambda _:None,admitted_job_ids=ids)
    assert not (tmp_path/'queue').exists()


def test_resource_denial_has_no_attempt_or_retry_and_resume_preserves_outputs(tmp_path):
    p=plan();seen=[];root=tmp_path/'queue';checks=[]
    def resource(job,events):
        checks.append((job['job_id'],len(events)))
        return dict(admitted=job['job_id']=='parent',reason='Live free disk changed',free_bytes=1)
    result=run_queue(p,root,handler(seen),check_identity=lambda _:None,before_start=resource)
    assert result['status']=='paused_before_job' and result['paused_job_id']=='growth'
    ledger=AttemptLedger.resume(root,p)
    assert len(ledger.events)==2 and {e['attempt_id'] for e in ledger.events}=={'parent'}
    assert checks==[('parent',0),('growth',2)]
    assert result['preflight_checks'][-1]['decision']['admitted'] is False
    parent_result=ledger.events[-1]['result'];original=__import__('pathlib').Path(parent_result['path']).read_bytes()
    result=run_queue(p,root,handler(seen),check_identity=lambda _:None,resume=True,
        before_start=lambda job,events:dict(admitted=True,reason='Full next transaction now fits'))
    assert result['status']=='completed' and all(attempt==0 for _,attempt in seen)
    assert __import__('pathlib').Path(parent_result['path']).read_bytes()==original
    unpaused=tmp_path/'unpaused';run_queue(p,unpaused,handler([]),check_identity=lambda _:None)
    a=sorted(x.read_bytes() for x in root.rglob('prediction.json'))
    b=sorted(x.read_bytes() for x in unpaused.rglob('prediction.json'))
    assert a==b


@pytest.mark.parametrize('decision',[None,True,{'admitted':1},{'admitted':False,'x':float('nan')}])
def test_resource_policy_is_strict_and_cannot_begin_an_unapproved_attempt(tmp_path,decision):
    p=plan();root=tmp_path/'queue'
    with pytest.raises(ValueError,match='preflight'):
        run_queue(p,root,handler([]),check_identity=lambda _:None,before_start=lambda *_:decision)
    assert not AttemptLedger.resume(root,p).events
    receipt=json.loads((root/'invocation-0.json').read_text())
    assert receipt['status']=='interrupted_or_failed'
