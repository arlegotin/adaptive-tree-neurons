"""Actual engineering workers inside the immutable paid search queue."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

from test_method_worker import request_fixture,sha
from treeneuron.config import canonical_hash


def api():
    from treeneuron.research import trial_bridge_v2
    return trial_bridge_v2


def setup(tmp_path):
    from treeneuron.research.search_controller_v2 import make_search_plan
    path,template=request_fixture(tmp_path/'input')
    mean=template['config'];bad=copy.deepcopy(mean);bad.update(estimator='ridge',parameters=dict(alpha=-1.))
    ridge=copy.deepcopy(mean);ridge.update(estimator='ridge',parameters=dict(alpha=1.))
    spline=copy.deepcopy(mean);spline.update(estimator='spline',parameters=dict(alpha=1.,n_knots=4))
    identity=dict(job_id='engineering/P13',family_id='P13',search_repeat=0,
        source_sha256=template['scientific_source_sha256'],
        runtime_sha256=canonical_hash({'fixture':'explicitly_unqualified_runtime'}),
        support_sha256=canonical_hash(dict(configurations=[mean,bad,ridge,spline])),split_sha256=template['observation_identity']['split_sha256'])
    plan=make_search_plan(identity=identity,fit_seed=17,hpo_seed32=19,anchors=[mean],
        mode='finite',support=None,candidates=[bad,ridge,spline],max_attempts=4,
        deadline_ns=30000000000,prefixes_ns=[])
    return path,template,plan


@pytest.mark.parametrize('change',['template_digest','source_identity','fit_seed'])
def test_invalid_bridge_binding_rejects_before_creating_search_or_starting_worker(change,tmp_path):
    assert hasattr(api(),'EngineeringTrialBridge')
    path,template,plan=setup(tmp_path);expected=sha(path)
    if change=='template_digest':expected='0'*64
    else:
        from treeneuron.research.search_controller_v2 import make_search_plan
        p=plan.to_dict();keys=('identity','fit_seed','hpo_seed32','anchors','mode','support','candidates','max_attempts','deadline_ns','prefixes_ns')
        values={k:p[k] for k in keys}
        if change=='source_identity':values['identity']['source_sha256']='0'*64
        else:values['fit_seed']=18
        plan=make_search_plan(**values)
    with pytest.raises(ValueError):
        api().EngineeringTrialBridge(plan=plan,template_path=path,expected_template_sha256=expected,
            search_directory=tmp_path/'search',interpreter=sys.executable)
    assert not (tmp_path/'search').exists()


def test_capped_engineering_bridge_cannot_silently_substitute_for_native_quality(tmp_path):
    from treeneuron.research.search_controller_v2 import make_search_plan
    path,template,legacy=setup(tmp_path);p=legacy.to_dict()
    keys=('identity','fit_seed','hpo_seed32','anchors','mode','support','candidates','max_attempts','deadline_ns','prefixes_ns')
    values={k:p[k] for k in keys};values['deadline_ns']=None
    quality=make_search_plan(**values)
    with pytest.raises(ValueError,match='native.quality'):
        api().EngineeringTrialBridge(plan=quality,template_path=path,expected_template_sha256=sha(path),
            search_directory=tmp_path/'search',interpreter=sys.executable)
    assert not (tmp_path/'search').exists()


def test_actual_supervised_paid_search_retains_failures_and_exact_clean_resume(tmp_path):
    assert hasattr(api(),'EngineeringTrialBridge')
    from treeneuron.research.search_controller_v2 import run_search
    path,template,plan=setup(tmp_path);search=tmp_path/'search';started=time.perf_counter_ns()
    bridge=api().EngineeringTrialBridge(plan=plan,template_path=path,expected_template_sha256=sha(path),
        search_directory=search,interpreter=sys.executable)
    first=run_search(search,plan=plan,evaluate_trial=bridge,max_attempts_this_invocation=2,
        invocation_preparation_ns=bridge.preparation_ns)
    assert first['stop_reason']=='operator_pause' and first['attempts_paid']==2
    again=api().EngineeringTrialBridge(plan=plan,template_path=path,expected_template_sha256=sha(path),
        search_directory=search,interpreter=sys.executable)
    result=run_search(search,plan=plan,evaluate_trial=again,resume=True,
        expected_journal_hash=first['journal_hash'],invocation_preparation_ns=again.preparation_ns)
    outer_elapsed=time.perf_counter_ns()-started
    assert result['stop_reason']=='attempt_limit' and result['attempts_paid']==4 and result['winner'] is None
    assert [r['status'] for r in result['records']]==['completed_ineligible','input_or_dependency_failure','completed_ineligible','completed_ineligible']
    assert all(r['fit_seed']==17 and r['V_score'] is None for r in result['records'])
    planned=plan.to_dict();assert [r['config'] for r in result['records']]==planned['anchors']+planned['candidates']
    process_ns=0;process_digests=[]
    for i,row in enumerate(result['records']):
        attempt=search/f'attempt-{i:03}'
        refs={r['path']:r['sha256'] for r in row['files']}
        actual={p.relative_to(search).as_posix() for p in attempt.rglob('*') if p.is_file()}
        assert set(refs)==actual
        for name,digest in refs.items():assert sha(search/name)==digest
        request=json.loads((attempt/'input/request.json').read_text())
        assert request['config']==row['config'] and request['fit_seed']==17
        assert request['scientific_source_sha256']==template['scientific_source_sha256']
        process=json.loads((attempt/'process/process.json').read_text());process_ns+=process['elapsed_ns']
        process_digests.append(sha(attempt/'process/process.json'))
        assert process['child_reaped'] and process['eligible_model'] is False
        worker=json.loads((attempt/'worker/terminal.json').read_text())
        assert worker['quality_eligible'] is False and worker['T_accessed'] is False
        assert worker['accessed_roles']==['F_labels','G_labels','V_features']
        material=json.loads((attempt/'input/materialization.json').read_text())
        assert len(material['files'])==3
        for ref in material['files']:
            assert ref['method'] in ('hard_link','verified_copy') and sha(attempt/'input'/ref['path'])==ref['sha256']
    assert result['total_charged_ns']>=process_ns+bridge.preparation_ns+again.preparation_ns
    assert result['protected_access_authorized'] is False and result['native_worker_qualified'] is False
    proof=dict(result=result,source_sha256=template['scientific_source_sha256'],
        process_receipt_sha256=process_digests,initial_journal_hash=first['journal_hash'],
        disjoint_process_body_ns=process_ns,external_two_invocation_span_ns=outer_elapsed,
        constructor_preparation_ns=[bridge.preparation_ns,again.preparation_ns],
        fixture_generation_excluded=True,quality_eligible=False,practical_searches=0,T_accessed=False)
    (tmp_path/'proof.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
