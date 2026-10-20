"""Official RealMLP suggestions stay isolated from labels and generic TPE."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

from treeneuron.config import canonical_hash
from test_native_generation_chain import actual,sha


def api():
    assert importlib.util.find_spec('treeneuron.research.native_schedule_v2') is not None,'Missing owned native schedule worker'
    from treeneuron.research import native_schedule_v2
    return native_schedule_v2


def overlay():return Path(__file__).resolve().parents[1]/'environment/research-v2-realmlp-site'


def prepare(module,tmp_path,schema=None,**kw):
    return module.prepare_realmlp_native_schedule(feature_schema=schema or dict(columns=['x'],categorical=[]),
        hpo_seed=901,output=tmp_path/'schedule',interpreter=sys.executable,
        extra_pythonpath=[overlay()],**kw)


def test_actual_official_schedule_and_primary_native_fit_share_the_exact_paid_configurations(tmp_path,monkeypatch):
    module=api();args,old=actual(tmp_path,monkeypatch)
    from treeneuron.research.native_trial_bridge_v2 import InspectedNativeTrialBridge
    from treeneuron.research.search_supports_v2 import family_search_spec
    from treeneuron.research.search_controller_v2 import make_search_plan,run_search
    request=json.loads(args['request_path'].read_text());schema=request['config']['feature_schema']
    started=time.perf_counter_ns();prepared=prepare(module,tmp_path,schema)
    sampler_elapsed=time.perf_counter_ns()-started
    assert prepared['status']=='completed_native_schedule'
    before=canonical_hash(module._inventory(tmp_path/'schedule'))
    proof=module.load_realmlp_native_schedule(prepared['reference'],expected_hpo_seed=901,expected_feature_schema=schema)
    assert len(proof['configurations'])==32 and proof['native_fits']==0 and proof['labels_read']==[]
    assert proof['paid_preparation_ns']>0 and proof['protected_access_authorized'] is False
    spec=family_search_spec('P06',schema);request.update(family_id='P06',config=spec['anchors'][0]['config'])
    args['request_path'].write_text(json.dumps(request,sort_keys=True)+'\n')
    plan=make_search_plan(identity=dict(job_id='fixture/P06',family_id='P06',search_repeat=0,
        source_sha256=request['scientific_source_sha256'],runtime_sha256='e'*64,
        support_sha256=spec['support_hash'],split_sha256=request['observation_identity']['split_sha256']),
        fit_seed=17,hpo_seed32=81,anchors=proof['configurations'][:1],mode='finite',support=None,
        candidates=proof['configurations'][1:],max_attempts=1,deadline_ns=None,prefixes_ns=[])
    root=tmp_path/'search'
    callback=InspectedNativeTrialBridge(plan=plan,template_path=args['request_path'],expected_template_sha256=sha(args['request_path']),
        prepared_root=args['prepared_root'],expected_manifest_sha256=args['expected_manifest_sha256'],outer_fold=0,
        search_directory=root,interpreter=sys.executable,extra_pythonpath=[overlay()],
        official_schedule_reference=prepared['reference'],native_hpo_seed=901)
    assert callback.native_sampler_minimum_prerequisite_ns==proof['paid_preparation_ns']<sampler_elapsed
    result=run_search(root,plan=plan,evaluate_trial=callback,
        invocation_preparation_ns=callback.preparation_ns+sampler_elapsed)
    assert result['stop_reason']=='attempt_limit' and result['records'][0]['status']=='completed_ineligible'
    assert result['winner'] is None
    inspection=json.loads((root/'attempt-000/inspection/result.json').read_text())
    assert inspection['replay']['prediction_reload_exact'] is True and inspection['validation']['score']>=0
    assert canonical_hash(module._inventory(tmp_path/'schedule'))==before
    for changed in ('seed','schema','bytes'):
        reference=copy.deepcopy(prepared['reference']);seed=901;changed_schema=copy.deepcopy(schema)
        if changed=='seed':seed=902
        elif changed=='schema':changed_schema['columns']=['different']
        else:reference['inventory_sha256']='0'*64
        with pytest.raises(ValueError):module.load_realmlp_native_schedule(reference,expected_hpo_seed=seed,expected_feature_schema=changed_schema)


def test_schedule_RSS_stop_is_paid_and_reaped_without_model_fits_or_label_access(tmp_path):
    module=api();result=prepare(module,tmp_path,rss_limit_bytes=1)
    assert result['status']=='resource_exhausted' and result['process']['child_reaped'] is True
    assert result['reference'] is None and result['native_fits']==0 and result['labels_read']==[]


@pytest.mark.parametrize('change',['source','seed','labels','unknown'])
def test_bad_schedule_request_rejects_before_importing_the_private_native_sampler(tmp_path,change):
    module=api()
    from treeneuron.research.deployment_worker_v2 import _source
    value=dict(schema='research/v2-native-schedule-request',scientific_source_sha256=_source(),
        feature_schema=dict(columns=['x'],categorical=[]),hpo_seed=901)
    if change=='source':value['scientific_source_sha256']='0'*64
    elif change=='seed':value['hpo_seed']=False
    elif change=='labels':value['feature_schema']['y']=[1.]
    else:value['files']={'F':'forbidden'}
    path=tmp_path/'request.json';path.write_text(json.dumps(value,sort_keys=True)+'\n')
    with pytest.raises(ValueError):module.checked_request(path,sha(path))
    assert 'pytabkit' not in sys.modules
