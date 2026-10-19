"""Raw TPE/native config provenance inside supervised engineering searches."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_method_worker import request_fixture,sha,write_request
from treeneuron.config import canonical_hash


def api():
    assert importlib.util.find_spec('treeneuron.research.compiled_trial_bridge_v2') is not None, 'Missing typed supervised TPE bridge'
    from treeneuron.research import compiled_trial_bridge_v2
    return compiled_trial_bridge_v2


def setup(root):
    from treeneuron.research.search_supports_v2 import family_search_spec
    from treeneuron.research.search_controller_v2 import make_search_plan
    path,template=request_fixture(root/'input');family='P00'
    spec=family_search_spec(family,template['config']['feature_schema'])
    template['family_id']=family;template['config']=spec['anchors'][0]['config'];write_request(path,template)
    identity=dict(job_id='engineering/typed-P00',family_id=family,search_repeat=0,
        source_sha256=template['scientific_source_sha256'],runtime_sha256=canonical_hash(dict(fixture='unqualified_runtime')),
        support_sha256=spec['support_hash'],split_sha256=template['observation_identity']['split_sha256'])
    plan=make_search_plan(identity=identity,fit_seed=17,hpo_seed32=73,anchors=[a['config'] for a in spec['anchors']],
        mode='tpe',support=spec['support'],candidates=[],max_attempts=2,deadline_ns=120_000_000_000,prefixes_ns=[])
    return path,template,spec,plan


@pytest.mark.parametrize('kind',['support_hash','support','anchors','family_mode'])
def test_changed_or_unrelated_support_cannot_construct_a_typed_bridge(kind,tmp_path):
    module=api();path,template,spec,plan=setup(tmp_path)
    from treeneuron.research.search_controller_v2 import make_search_plan
    p=plan.to_dict();args={k:p[k] for k in ('identity','fit_seed','hpo_seed32','anchors','mode','support','candidates','max_attempts','deadline_ns','prefixes_ns')}
    if kind=='support_hash':args['identity']['support_sha256']='0'*64
    elif kind=='support':args['support']['width']['choices']=[64]
    elif kind=='anchors':args['anchors'][0]['parameters']={'lr':.001}
    else:args.update(mode='finite',support=None)
    plan=make_search_plan(**args)
    with pytest.raises(ValueError):module.CompiledEngineeringTrialBridge(plan=plan,template_path=path,
        expected_template_sha256=sha(path),search_directory=tmp_path/'search',interpreter=sys.executable)
    assert not (tmp_path/'search').exists()


def test_invalid_raw_proposal_never_starts_a_native_worker(tmp_path,monkeypatch):
    module=api();path,template,spec,plan=setup(tmp_path);search=tmp_path/'search';attempt=search/'attempt-000';attempt.mkdir(parents=True)
    bridge=module.CompiledEngineeringTrialBridge(plan=plan,template_path=path,expected_template_sha256=sha(path),
        search_directory=search,interpreter=sys.executable)
    def forbidden(*args,**kwargs):raise AssertionError('Invalid proposal must not fit')
    monkeypatch.setattr(module.EngineeringTrialBridge,'__call__',forbidden)
    request=dict(candidate_origin='tpe',candidate_id='tpe-0',retry_index=0,checkpoint_files=[],config={'surprise':1},
        attempt_index=0,fit_seed=17,budget_remaining_ns=1_000_000_000,
        attempt_directory=str(attempt.resolve()),search_directory=str(search.resolve()))
    with pytest.raises(ValueError):bridge(request)
    assert not list(attempt.iterdir())


def test_compilation_exhausting_the_remaining_budget_records_timeout_without_a_child(tmp_path,monkeypatch):
    module=api();path,template,spec,plan=setup(tmp_path);search=tmp_path/'search';attempt=search/'attempt-000';attempt.mkdir(parents=True)
    bridge=module.CompiledEngineeringTrialBridge(plan=plan,template_path=path,expected_template_sha256=sha(path),
        search_directory=search,interpreter=sys.executable)
    def forbidden(*args,**kwargs):raise AssertionError('An exhausted trial cannot start a native worker')
    monkeypatch.setattr(module.EngineeringTrialBridge,'__call__',forbidden)
    result=bridge(dict(candidate_origin='anchor',candidate_id='anchor-0',retry_index=0,checkpoint_files=[],
        config=template['config'],attempt_index=0,fit_seed=17,budget_remaining_ns=1,
        attempt_directory=str(attempt.resolve()),search_directory=str(search.resolve())))
    assert result['status']=='timeout' and result['V_score'] is None
    trace=json.loads((attempt/'translation.json').read_text())
    assert trace['remaining_worker_budget_ns']==0 and trace['compilation_ns']>1
    assert [f['path'] for f in result['files']]==['attempt-000/translation.json']


def test_actual_compiled_TPE_worker_keeps_raw_history_exact_resume_and_paid_translation(tmp_path):
    module=api()
    from treeneuron.research.opportunity_execution import live_resources
    live=live_resources(tmp_path)
    (tmp_path/'admission.json').write_text(json.dumps(live,sort_keys=True)+'\n')
    if live['available_memory_bytes'] is None or live['available_memory_bytes']<7*1024**3//4 or live['free_disk_bytes']<512*1024**2:
        pytest.skip('Retained actual live admission below the typed-worker fixture floor')
    path,template,spec,plan=setup(tmp_path);search=tmp_path/'search'
    from treeneuron.research.search_controller_v2 import run_search
    def bridge():return module.CompiledEngineeringTrialBridge(plan=plan,template_path=path,
        expected_template_sha256=sha(path),search_directory=search,interpreter=sys.executable,rss_limit_bytes=1024**3)
    first_bridge=bridge();first=run_search(search,plan=plan,evaluate_trial=first_bridge,max_attempts_this_invocation=1,
        invocation_preparation_ns=first_bridge.preparation_ns)
    assert first['stop_reason']=='operator_pause' and first['attempts_paid']==1
    second_bridge=bridge();result=run_search(search,plan=plan,evaluate_trial=second_bridge,resume=True,
        expected_journal_hash=first['journal_hash'],invocation_preparation_ns=second_bridge.preparation_ns)
    assert result['attempts_paid']==2 and result['winner'] is None
    assert [r['candidate_origin'] for r in result['records']]==['anchor','tpe']
    assert [r['status'] for r in result['records']]==['completed_ineligible']*2
    for i,row in enumerate(result['records']):
        attempt=search/f'attempt-{i:03}';trace=json.loads((attempt/'translation.json').read_text())
        worker_request=json.loads((attempt/'input/request.json').read_text())
        assert trace['raw_configuration']==row['config'] and trace['compiled_configuration']==worker_request['config']
        assert trace['family_support_hash']==spec['support_hash'] and trace['compilation_ns']>0
        assert trace['remaining_worker_budget_ns']<trace['remaining_budget_at_entry_ns']
        assert worker_request['budget']['max_seconds']*1e9<=trace['remaining_worker_budget_ns']
        assert row['V_score'] is None and row['fit_seed']==17
        files={f['path']:f['sha256'] for f in row['files']}
        assert f'attempt-{i:03}/translation.json' in files
        for relative,digest in files.items():assert sha(search/relative)==digest
        process=json.loads((attempt/'process/process.json').read_text());assert process['child_reaped'] and process['returncode']==0
        terminal=json.loads((attempt/'worker/terminal.json').read_text());assert terminal['quality_eligible'] is terminal['T_accessed'] is False
    assert 'width' in result['records'][1]['config'] and 'family' not in result['records'][1]['config']
    (tmp_path/'proof.json').write_text(json.dumps(dict(result=result,initial_journal_hash=first['journal_hash'],
        source_sha256=template['scientific_source_sha256'],quality_eligible=False,V_labels_read=False,T_accessed=False),sort_keys=True,indent=2)+'\n')
