"""Cached resource evidence needs actual requalification, not a saved true flag."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def api():
    from treeneuron.research import native_resource_forecast_v2
    assert hasattr(native_resource_forecast_v2,'load_native_practical_resource_forecast'), 'Missing actual-qualified resource forecast reader'
    return native_resource_forecast_v2


def test_cached_forecast_reader_requires_actual_registry_before_opening_cache(tmp_path):
    module=api()
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module.load_native_practical_resource_forecast(dict(path=str(tmp_path/'forbidden-cache'),sha256='a'*64),
            practical_plan=None,expected_practical_plan_hash='b'*64,support_reference=dict(path=str(tmp_path/'forbidden-support'),sha256='c'*64),
            qualification_reference=dict(root=str(tmp_path/'fake-registry'),inventory_sha256='d'*64))
    assert 'forbidden-cache' not in str(error.value) and 'forbidden-support' not in str(error.value)
    assert list(tmp_path.iterdir())==[]


def test_cached_metadata_match_excludes_only_preparation_clock_not_measurements():
    module=api()
    from treeneuron.config import canonical_hash
    actual=dict(schema='software-only',forecast_preparation_ns=20,observations=[dict(process_elapsed_ns=123)],native_qualification_authenticated=False)
    actual['forecast_hash']=canonical_hash(actual)
    saved=copy.deepcopy(actual);saved['forecast_preparation_ns']=10
    saved['forecast_hash']=canonical_hash({k:v for k,v in saved.items() if k!='forecast_hash'})
    assert module._match_cached_forecast(saved,actual) is None
    changed=copy.deepcopy(saved);changed['observations'][0]['process_elapsed_ns']=124
    changed['forecast_hash']=canonical_hash({k:v for k,v in changed.items() if k!='forecast_hash'})
    with pytest.raises(ValueError):module._match_cached_forecast(changed,actual)


@pytest.mark.parametrize('fault',['boolean_clock','extra','authority','bad_hash'])
def test_cached_metadata_cannot_forge_current_measured_or_authority_fields(fault):
    module=api()
    from treeneuron.config import canonical_hash
    actual=dict(forecast_preparation_ns=20,observations=[],native_qualification_authenticated=False);actual['forecast_hash']=canonical_hash(actual)
    saved=copy.deepcopy(actual)
    if fault=='boolean_clock':saved['forecast_preparation_ns']=True
    elif fault=='extra':saved['unexpected']=True
    elif fault=='authority':saved['native_qualification_authenticated']=True
    saved['forecast_hash']='0'*64 if fault=='bad_hash' else canonical_hash({k:v for k,v in saved.items() if k!='forecast_hash'})
    with pytest.raises(ValueError):module._match_cached_forecast(saved,actual)


def request(tmp_path):
    return dict(schema='research/v2-primary-search-resource-episode-request',source_sha256='a'*64,
        practical_plan_reference=dict(path=str(tmp_path/'plan'),sha256='b'*64),expected_practical_plan_hash='c'*64,
        job_id='airfoil/outer-0/repeat-0/P13',support_reference=dict(path=str(tmp_path/'support'),sha256='d'*64),
        qualification_reference=dict(root=str(tmp_path/'registry'),inventory_sha256='e'*64),search_directory=str(tmp_path/'search'),
        rss_limit_bytes=4*1024**3,previous_episodes=[],resource_forecast_reference=dict(path=str(tmp_path/'forecast'),sha256='f'*64))


def test_resource_episode_bootstrap_is_metadata_only_and_binds_its_exact_cache(tmp_path):
    from treeneuron.research.primary_search_worker_v2 import primary_worker_identity
    value=request(tmp_path);path=tmp_path/'request.json';path.write_text(json.dumps(value));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    identity=primary_worker_identity(path,sha)
    assert identity['request_sha256']==sha and identity['source_sha256']=='a'*64
    assert not (tmp_path/'forecast').exists() and not (tmp_path/'search').exists()


@pytest.mark.parametrize('fault',['missing','relative','wrong_schema'])
def test_resource_episode_schema_cannot_silently_fall_back_to_ungated_execution(tmp_path,fault):
    from treeneuron.research.primary_search_worker_v2 import primary_worker_identity
    value=request(tmp_path)
    if fault=='missing':value.pop('resource_forecast_reference')
    elif fault=='relative':value['resource_forecast_reference']['path']='relative-cache'
    else:value['schema']='research/v2-primary-search-episode-request'
    path=tmp_path/'request.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError):primary_worker_identity(path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_real_host_gate_uses_bound_job_forecast_and_reports_shortage(tmp_path):
    from treeneuron.research import primary_search_worker_v2
    assert hasattr(primary_search_worker_v2,'_make_resource_gate'), 'Missing real per-trial forecast/host composition'
    from test_native_resource_forecast_v2 import study
    # Use the undecorated pure fixture generator. No actual native registry or
    # resource authority is supplied or mocked by this metadata-only test.
    plan,observations=study.__wrapped__(tmp_path)
    from treeneuron.research.native_resource_forecast_v2 import _forecast_from_observations
    forecast=_forecast_from_observations(plan,plan.to_dict()['plan_hash'],observations,free_disk_margin_bytes=1024**3)
    job=next(j for j in plan.to_dict()['jobs'] if j['job_id']=='airfoil/outer-0/repeat-0/P13')
    context=dict(forecast=forecast,job=job,reference=dict(path=str(tmp_path/'not-created-cache'),sha256='a'*64))
    gate,identity=primary_search_worker_v2._make_resource_gate(context,search_directory=tmp_path/'search',
        observe=lambda path:dict(free_disk_bytes=0,available_memory_bytes=None,observed_utc_ns=1,memory_basis='software fixture'))
    result=gate()
    assert result['admitted'] is False and result['reason']=='insufficient_live_disk'
    assert identity['forecast_hash']==forecast['forecast_hash'] and result['job_id']==job['job_id']
    assert result['native_qualification_authenticated'] is False and not (tmp_path/'search').exists()


def test_actual_cold_controller_observes_host_and_closes_before_fitting_on_shortage(tmp_path):
    """Synthetic envelope deliberately denies real host; no qualification fake."""
    from test_native_resource_forecast_v2 import study
    from treeneuron.config import canonical_hash
    from treeneuron.research.native_resource_forecast_v2 import _forecast_from_observations
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process,inspect_cooperative_process
    plan,observations=study.__wrapped__(tmp_path)
    forecast=_forecast_from_observations(plan,plan.to_dict()['plan_hash'],observations,free_disk_margin_bytes=1024**3)
    job=next(j for j in plan.to_dict()['jobs'] if j['job_id']=='airfoil/outer-0/repeat-0/P13')
    forecast['envelopes']['airfoil/P13']['transaction_reserve_proxy_bytes']=10**30
    forecast['forecast_hash']=canonical_hash({k:v for k,v in forecast.items() if k!='forecast_hash'})
    context=dict(forecast=forecast,job=job,reference=dict(path=str(tmp_path/'unused-unqualified-cache'),sha256='a'*64))
    (tmp_path/'context.json').write_text(json.dumps(context))
    script=tmp_path/'worker.py';script.write_text('''
import os,json
from pathlib import Path
from treeneuron.research.cooperative_process_v2 import cooperative_worker
with cooperative_worker(identity=dict(fixture='cold-resource-denial')):
    from treeneuron.research.search_controller_v2 import make_search_plan
    from treeneuron.research.primary_search_worker_v2 import _run_prepared_search
    root=Path(os.environ['ATN_SEARCH_CONTROL_ROOT']).parent
    context=json.loads((root/'context.json').read_text());job=context['job']
    plan=make_search_plan(identity=dict(job_id=job['job_id'],family_id=job['family_id'],search_repeat=job['search_repeat'],source_sha256=job['source_sha256'],runtime_sha256=job['runtime_sha256'],support_sha256='c'*64,split_sha256=job['split_hash']),fit_seed=job['fit_seed'],hpo_seed32=job['hpo_seed32'],anchors=[dict(constant=0)],mode='finite',support=None,candidates=[],max_attempts=32,deadline_ns=None,prefixes_ns=[])
    def forbidden(request):raise AssertionError('Resource denial cannot start fitting')
    _run_prepared_search(plan=plan,callback=forbidden,search_directory=root/'search',output=root/'worker',previous_outer_tail_ns=0,resume=False,expected_journal_hash=None,resource_context=context)
''')
    receipt=supervise_cooperative_process(argv=[sys.executable,'-P','-B',str(script)],cwd=str(tmp_path),
        env_overrides=dict(PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'),PYTHONDONTWRITEBYTECODE='1'),
        output=tmp_path/'control',identity=dict(fixture='cold-resource-denial'),admission_identity=dict(policy='software-only'),
        admission_check=lambda:dict(admitted=True),rss_pause_bytes=768*1024**2)
    assert receipt['returncode']==0
    saved=json.loads((tmp_path/'worker/search-result.json').read_text())
    assert saved['stop_reason']=='resource_pause' and saved['attempts_paid']==0
    events=[json.loads(line) for line in (tmp_path/'search/events.jsonl').read_text().splitlines()]
    event=next(e for e in events if e['kind']=='trial_admission')
    assert event['payload']['decision']['live_observation']['observed_utc_ns']>0
    assert event['payload']['decision']['admitted'] is False
    sha=lambda name:hashlib.sha256((tmp_path/'control'/name).read_bytes()).hexdigest()
    proof=inspect_cooperative_process(root=tmp_path/'control',expected_receipt_sha256=sha('process.json'),expected_started_sha256=sha('started.json'))
    assert proof['worker_protocol_closed'] and proof['root_process_clock_authenticated']
    assert proof['whole_training_procedure_cost_authenticated'] is False
