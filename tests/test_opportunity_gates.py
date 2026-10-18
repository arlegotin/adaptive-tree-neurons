"""Missing actual updates can admit fallback; failures and existing states cannot."""
import copy
import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def parent(tmp_path):
    counter=0
    def build(status='completed_eligible',missing=(1000,)):
        nonlocal counter
        root=tmp_path/str(counter);counter+=1;root.mkdir()
        captured=sorted({250,1000}-set(missing))
        capture=dict(requested=[250,1000],captured=captured,unavailable=list(missing),
            records=[dict(label='current',update=u) for u in captured])
        path=root/'capture.json';path.write_text(json.dumps(capture))
        return dict(value=dict(status=status,artifact_path=str(root),report=dict(
            request=dict(snapshot_updates=[250,1000],fixed_budget=False),
            unavailable_snapshots=list(missing),executed_updates=500 if missing else 8000,
            capture=dict(path='capture.json',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))))
    return build


def test_fallback_admits_only_declared_unreached_updates_and_retains_origin(parent):
    from treeneuron.research.opportunity_gates import fallback_gate,snapshot_gate
    job=dict(payload=dict(original_parent='parent:original',snapshot_updates=[250,1000]))
    dependencies={'parent:original':parent()}
    admitted=fallback_gate(job,dependencies)
    assert admitted['status']=='completed_eligible'
    assert admitted['unavailable_snapshots']==[1000]
    assert admitted['original_parent_job_id']=='parent:original'
    original=copy.deepcopy(dependencies)
    for update,expected in ((250,'completed_ineligible'),(1000,'completed_eligible')):
        score=dict(value=dict(status='completed_eligible' if update==250 else 'completed_ineligible'))
        gate=dict(payload=dict(original_parent='parent:original',original_score='score:original',
            fallback_gate='fallback_gate:original',snapshot=f'actual_{update}'))
        values={**dependencies,'score:original':score,'fallback_gate:original':dict(value=admitted)}
        result=snapshot_gate(gate,values)
        assert result['status']==expected
    assert dependencies==original


def test_failure_and_corruption_are_never_permission_for_fallback(parent):
    from treeneuron.research.opportunity_gates import fallback_gate,snapshot_gate
    job=dict(payload=dict(original_parent='parent:original',snapshot_updates=[250,1000]))
    for status in ('numerical_failure','input_or_dependency_failure','irrecoverable_missing'):
        assert fallback_gate(job,{'parent:original':parent(status)})['status']=='irrecoverable_missing'
    assert fallback_gate(job,{'parent:original':parent(missing=())})['status']=='completed_ineligible'
    assert fallback_gate(job,{'parent:original':parent('completed_ineligible')})['status']=='completed_ineligible'
    broken=parent();broken['value']['report']['unavailable_snapshots']=[999]
    with pytest.raises(ValueError,match='snapshot'):fallback_gate(job,{'parent:original':broken})
    admitted=fallback_gate(job,{'parent:original':parent()})
    gate=dict(payload=dict(original_parent='parent:original',original_score='score:original',
        fallback_gate='fallback_gate:original',snapshot='actual_1000'))
    for status in ('numerical_failure','completed_eligible'):
        values={'parent:original':parent(),'score:original':dict(value=dict(status=status)),
            'fallback_gate:original':dict(value=admitted)}
        if status=='numerical_failure':assert snapshot_gate(gate,values)['status']=='irrecoverable_missing'
        else:
            with pytest.raises(ValueError,match='snapshot'):snapshot_gate(gate,values)


def test_gate_worker_integrates_without_fitting_or_label_access(tmp_path,parent):
    from treeneuron.research.opportunity_workers import OpportunityWorkers
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    rows=[dict(job_id='parent:original',kind='parent',payload={},depends_on=[]),
        dict(job_id='fallback',kind='fallback_gate',payload=dict(original_parent='parent:original',
            snapshot_updates=[250,1000]),depends_on=['parent:original'],dependency_policy='all_terminal'),
        dict(job_id='would-fit',kind='fixture',payload={},depends_on=['fallback'])]
    queue=make_queue_plan(dict(study_config={}),rows);worker=OpportunityWorkers(queue);called=[]
    def dispatch(job,path,dependencies,attempt):
        if job['job_id']=='parent:original':return parent(missing=())['value']
        if job['kind']=='fallback_gate':return worker(job,path,dependencies,attempt)
        called.append(job['job_id']);return dict(status='completed_eligible')
    result=run_queue(queue,tmp_path/'queue',dispatch,check_identity=lambda identity:None)
    assert result['status']=='completed' and not called
    assert result['statuses']['fallback']=='completed_ineligible'
    assert result['statuses']['would-fit']=='completed_ineligible'


def test_fallback_rejects_semantic_unreached_claim_and_changed_capture(parent):
    from treeneuron.research.opportunity_gates import fallback_gate
    job=dict(payload=dict(original_parent='parent:original',snapshot_updates=[250,1000]))
    contradicted=parent();contradicted['value']['report']['executed_updates']=8000
    with pytest.raises(ValueError,match='snapshot|capture'):
        fallback_gate(job,{'parent:original':contradicted})
    changed=parent();root=Path(changed['value']['artifact_path'])
    path=root/'capture.json';capture=json.loads(path.read_text());capture['unavailable']=[]
    path.write_text(json.dumps(capture))
    changed['value']['report']['capture']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='snapshot|capture'):
        fallback_gate(job,{'parent:original':changed})
    corrupt=parent();corrupt['value']['report']['capture']['sha256']='0'*64
    with pytest.raises(ValueError,match='commitment'):
        fallback_gate(job,{'parent:original':corrupt})
