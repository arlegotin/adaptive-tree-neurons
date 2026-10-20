"""Real native trials must compose the declared proposal, generation and V chain."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from treeneuron.config import canonical_hash
from test_native_generation_chain import actual,sha


def api():
    assert importlib.util.find_spec('treeneuron.research.native_trial_bridge_v2') is not None,'Missing native inspected trial bridge'
    from treeneuron.research.native_trial_bridge_v2 import InspectedNativeTrialBridge
    return InspectedNativeTrialBridge


def setup(tmp_path,monkeypatch,*,family='P13'):
    from treeneuron.research.search_supports_v2 import family_search_spec
    from treeneuron.research.search_controller_v2 import make_search_plan
    args,old=actual(tmp_path,monkeypatch)
    request=json.loads(args['request_path'].read_text())
    assert json.loads((old/'native/result.json').read_text())['status']=='completed_ineligible'
    spec=family_search_spec(family,request['config']['feature_schema'])
    if family!='P13':
        request.update(family_id=family,config=spec['anchors'][0]['config'])
        args['request_path'].write_text(json.dumps(request,sort_keys=True)+'\n')
        args['expected_request_sha256']=sha(args['request_path'])
    plan=make_search_plan(identity=dict(job_id='fixture/'+family,family_id=family,search_repeat=0,
        source_sha256=request['scientific_source_sha256'],runtime_sha256='f'*64,
        support_sha256=spec['support_hash'],split_sha256=request['observation_identity']['split_sha256']),
        fit_seed=17,hpo_seed32=91,anchors=[a['config'] for a in spec['anchors']],
        mode=spec['search_kind'],support=spec['support'],
        candidates=spec['finite_configurations'][1:] if family=='P13' else [],
        max_attempts=2,deadline_ns=None,prefixes_ns=[600_000_000_000])
    return args,old,spec,plan


def bridge(cls,args,plan,root,**kwargs):
    return cls(plan=plan,template_path=args['request_path'],expected_template_sha256=sha(args['request_path']),
        prepared_root=args['prepared_root'],expected_manifest_sha256=args['expected_manifest_sha256'],outer_fold=0,
        search_directory=root,interpreter=sys.executable,**kwargs)


def run(root,plan,callback,**kwargs):
    from treeneuron.research.search_controller_v2 import run_search
    return run_search(root,plan=plan,evaluate_trial=callback,invocation_preparation_ns=callback.preparation_ns,**kwargs)


def test_two_actual_native_models_keep_distinct_inspected_scores_without_quality_promotion(tmp_path,monkeypatch):
    cls=api();args,old,spec,plan=setup(tmp_path,monkeypatch)
    from treeneuron.research.workqueue import _inventory
    before=_inventory(old);root=tmp_path/'search';callback=bridge(cls,args,plan,root)
    out=run(root,plan,callback)
    assert out['stop_reason']=='attempt_limit' and out['attempts_paid']==2
    assert [r['status'] for r in out['records']]==['completed_ineligible']*2
    assert out['winner'] is None and all(r['V_score'] is None for r in out['records'])
    for i in range(2):
        attempt=root/f'attempt-{i:03}';packet=json.loads((attempt/'bridge.json').read_text())
        inspection=json.loads((attempt/'inspection/result.json').read_text())
        request=json.loads((attempt/'generation/input/request.json').read_text())
        assert packet['inspected_V_score']==inspection['validation']['score']>=0
        assert packet['quality_eligible'] is packet['native_workload_qualified'] is packet['T_accessed'] is False
        assert inspection['replay']['prediction_reload_exact'] is True
        assert request['budget']=={'native_stopping':True} and request['fit_seed']==17
        expected=spec['anchors'][0]['config'] if i==0 else spec['finite_configurations'][1]
        assert request['config']==expected
        assert packet['callback_body_ns']>=packet['inspection_span_ns']>0
        import os
        staged=json.loads((attempt/'generation/input/materialization.json').read_text())
        template=json.loads(args['request_path'].read_text())
        assert len(staged['files'])==3
        for role,ref in template['files'].items():
            assert os.path.samefile(args['request_path'].parent/ref['path'],attempt/'generation/input'/(role+'.npz'))
    assert _inventory(old)==before


@pytest.mark.parametrize('change',['source','support','split','seed','secondary'])
def test_wrong_native_bridge_bindings_fail_before_any_search_directory(tmp_path,monkeypatch,change):
    cls=api();args,old,spec,plan=setup(tmp_path,monkeypatch)
    from treeneuron.research.search_controller_v2 import make_search_plan
    p=plan.to_dict();values={k:p[k] for k in ('identity','fit_seed','hpo_seed32','anchors','mode','support','candidates','max_attempts','deadline_ns','prefixes_ns')}
    if change in ('source','support','split'):values['identity'][change+'_sha256']='0'*64
    elif change=='seed':values['fit_seed']=18
    else:args['outer_fold']=1
    plan=make_search_plan(**values);root=tmp_path/'search'
    with pytest.raises((ValueError,PermissionError)):
        cls(plan=plan,template_path=args['request_path'],expected_template_sha256=sha(args['request_path']),
            prepared_root=args['prepared_root'],expected_manifest_sha256=args['expected_manifest_sha256'],
            outer_fold=args['outer_fold'],search_directory=root,interpreter=sys.executable)
    assert not root.exists()


def test_bad_raw_TPE_coordinate_fails_before_staging_or_native_fitting(tmp_path,monkeypatch):
    cls=api();args,old,spec,plan=setup(tmp_path,monkeypatch,family='P00')
    root=tmp_path/'search';callback=bridge(cls,args,plan,root);attempt=root/'attempt-000';attempt.mkdir(parents=True)
    def forbidden(*a,**kw):raise AssertionError('Invalid proposal must not fit')
    from treeneuron.research import native_trial_bridge_v2 as module
    monkeypatch.setattr(module,'run_native_fit',forbidden)
    with pytest.raises(ValueError):callback(dict(candidate_id='tpe-0',candidate_origin='tpe',config={'unpaid':1},
        retry_index=0,checkpoint_files=[],attempt_index=0,fit_seed=17,budget_remaining_ns=None,
        attempt_directory=str(attempt),search_directory=str(root)))
    assert not list(attempt.iterdir())


def test_real_RSS_stop_recovers_same_candidate_seed_and_request_in_fresh_attempt(tmp_path,monkeypatch):
    cls=api();args,old,spec,plan=setup(tmp_path,monkeypatch);root=tmp_path/'search'
    from treeneuron.research.workqueue import _inventory
    first=run(root,plan,bridge(cls,args,plan,root,rss_limit_bytes=1))
    assert first['stop_reason']=='interrupted_recoverable' and first['attempts_paid']==1
    journal=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    interrupted=next(e['payload']['result'] for e in journal if e['kind']=='attempt_finished')
    assert first['records'][0]['status']=='interrupted_recoverable' and interrupted['checkpoint_files']
    before=_inventory(root/'attempt-000')
    second=run(root,plan,bridge(cls,args,plan,root),resume=True,expected_journal_hash=first['journal_hash'])
    assert second['stop_reason']=='attempt_limit' and second['attempts_paid']==2 and second['winner'] is None
    left,right=second['records'];assert left['config']==right['config'] and left['fit_seed']==right['fit_seed']==17
    assert right['retry_index']==1 and right['status']=='completed_ineligible'
    assert _inventory(root/'attempt-000')==before
    requests=[json.loads((root/f'attempt-{i:03}/generation/input/request.json').read_text()) for i in range(2)]
    assert requests[0]==requests[1]
    packet=json.loads((root/'attempt-001/generation/native-fit-result.json').read_text())
    assert packet['recovery']['reason']=='same_seed_restart_without_optimizer_checkpoint'
