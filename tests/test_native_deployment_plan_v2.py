"""Timing metadata/FSM fixtures carry no native or protected authority."""
import copy
import importlib.util
from pathlib import Path

import pytest

from test_protected_journal_v2 import metadata,create,ready,event,start,commit,ref


def api():
    assert importlib.util.find_spec('treeneuron.research.native_deployment_plan_v2'), 'Missing native timing plan binding'
    from treeneuron.research import native_deployment_plan_v2
    return native_deployment_plan_v2


def test_generic_plan_witness_freezes_once_before_labels_without_authority(tmp_path,metadata):
    m,root,owners,_=create(tmp_path,metadata);ready(m,root)
    plan=tmp_path/'deployment-plan.json';plan.write_text('{"known_metadata_fixture":true}\n')
    event(m,root,'deployment_plan_frozen',plan_reference=ref(plan))
    state=m._read_metadata_study(root)
    assert state['phase']=='owners_ready' and state['deployment_plan_reference']==ref(plan)
    assert state['protected_access_authorized'] is state['native_qualification_authenticated'] is state['T_accessed'] is False
    with pytest.raises(ValueError):event(m,root,'deployment_plan_frozen',plan_reference=ref(plan))


@pytest.mark.parametrize('phase',['prepared','active_query','labels_started','metrics_committed'])
def test_timing_plan_cannot_be_frozen_during_or_after_forbidden_stage(tmp_path,metadata,phase):
    m,root,owners,_=create(tmp_path,metadata)
    if phase!='prepared':ready(m,root)
    if phase=='active_query':start(m,root)
    if phase in ('labels_started','metrics_committed'):
        for q in ('a','b'):start(m,root,q);commit(m,root,tmp_path,q)
        event(m,root,'all_predictions_committed');event(m,root,'labels_started')
        if phase=='metrics_committed':
            marker=tmp_path/'metrics.json';marker.write_text('{}\n');event(m,root,'metrics_committed',metrics_reference=ref(marker),elapsed_ns=7)
    # Must fail on phase before trying to open this nonexistent plan.
    with pytest.raises(ValueError):event(m,root,'deployment_plan_frozen',plan_reference=dict(path=str(tmp_path/'deployment-plan.json'),sha256='f'*64))


def test_changed_witnessed_plan_bytes_invalidate_the_history(tmp_path,metadata):
    m,root,owners,_=create(tmp_path,metadata);ready(m,root)
    plan=tmp_path/'deployment-plan.json';plan.write_text('{}\n');event(m,root,'deployment_plan_frozen',plan_reference=ref(plan))
    plan.write_text('{"changed":true}\n')
    with pytest.raises(ValueError):m._read_metadata_study(root)


def test_later_private_timing_witness_preserves_the_completed_owned_prediction_proof(tmp_path):
    from test_protected_prediction_gate_v2 import local_complete_stage
    from treeneuron.research.protected_prediction_gate_v2 import _inspect_complete_stage
    from treeneuron.research import protected_journal_v2 as journal
    payload,reference,root,seen=local_complete_stage(tmp_path)
    before=_inspect_complete_stage(protocol_payload=payload,protocol_reference=reference,directory=root,owner_root=tmp_path/'owners')
    plan=tmp_path/'deployment-plan.json';plan.write_text('{"known_metadata_fixture":true}\n')
    journal._append_metadata_event(root,kind='deployment_plan_frozen',details=dict(plan_reference=ref(plan)))
    after=_inspect_complete_stage(protocol_payload=payload,protocol_reference=reference,directory=root,owner_root=tmp_path/'owners')
    assert before==after and len(seen)==2
    assert after['protected_access_authorized'] is after['T_accessed'] is False


@pytest.mark.parametrize('entry',['seal','load'])
def test_public_timing_plan_requires_actual_protocol_before_output_or_F_reads(tmp_path,entry):
    module=api();kwargs=dict(models_reference=dict(path=str(tmp_path/'forbidden-models'),sha256='a'*64),
        global_barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='b'*64),
        qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='c'*64),
        preT_costs_reference=dict(path=str(tmp_path/'forbidden-cost'),sha256='d'*64),
        readiness_reference=dict(path=str(tmp_path/'forbidden-readiness'),sha256='e'*64))
    protocol=dict(path=str(tmp_path/'forbidden-protocol'),sha256='f'*64)
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        if entry=='seal':module.seal_native_deployment_plan(protocol,directory=tmp_path/'forbidden-study',output=tmp_path/'plan',**kwargs)
        else:module.load_native_deployment_plan(dict(path=str(tmp_path/'forbidden-plan'),sha256='0'*64),protocol_reference=protocol,
            directory=tmp_path/'forbidden-study',**kwargs)
    assert 'forbidden-' not in str(error.value) and list(tmp_path.iterdir())==[]


def schedule_fixture():
    from test_frozen_practical_analysis_v2 import fixture
    from treeneuron.research.protected_journal_v2 import _hash
    payload,reference,_=fixture();requests={}
    for v in payload['prediction_manifest']['view_bindings']:
        if v['view']=='native_quality' and v['outer_fold']==(4 if v['dataset']=='bike' else 0):
            requests[v['job_id']]=dict(path=f"/unqualified-timing-requests/{v['job_id']}/request.json",sha256=_hash(v['job_id']))
    return payload,reference,requests


def test_metadata_compiler_retains_all_primary_models_and_exact_serial_paired_workload():
    payload,reference,requests=schedule_fixture();out=api()._compile_timing_schedule(payload,reference,requests)
    assert len(out['models'])==168 and len(out['comparisons'])==156 and len(out['jobs'])==9360
    first=out['comparisons'][0]
    assert first['dataset']=='airfoil' and first['search_repeat']==0 and first['baseline_family']=='P01'
    assert first['selected_job_id']=='airfoil/outer-0/repeat-0/P00'
    for index in range(0,len(out['jobs']),2):
        pair=out['jobs'][index:index+2]
        assert {j['role'] for j in pair}=={'selected','baseline'} and pair[0]['block_id']==pair[1]['block_id']
        assert pair[1]['depends_on']==[pair[0]['job_id']]
        assert all(j['outer_fold']==(4 if j['dataset']=='bike' else 0) for j in pair)
    assert out['native_models_authenticated'] is out['execution_authorized'] is out['T_accessed'] is False


@pytest.mark.parametrize('fault',['missing_view','duplicate_view','query','wrong_fold','missing_request','foreign_request'])
def test_metadata_timing_plan_cannot_drop_or_substitute_frozen_models(fault):
    module=api();payload,reference,requests=schedule_fixture()
    v=next(v for v in payload['prediction_manifest']['view_bindings'] if v['dataset']=='airfoil' and v['outer_fold']==0 and v['view']=='native_quality')
    if fault=='missing_view':v.update(query_id=None,status='unavailable_no_eligible_model')
    elif fault=='duplicate_view':payload['prediction_manifest']['view_bindings'].append(copy.deepcopy(v))
    elif fault=='query':v['query_id']='0'*64
    elif fault=='wrong_fold':v['outer_fold']=1
    elif fault=='missing_request':requests.pop(v['job_id'])
    else:requests['foreign']=requests[v['job_id']]
    with pytest.raises(ValueError):module._compile_timing_schedule(payload,reference,requests)


def test_order_is_fixed_by_protocol_identity_and_never_selected_from_outcomes():
    module=api();payload,reference,requests=schedule_fixture()
    a=module._compile_timing_schedule(payload,reference,requests);b=module._compile_timing_schedule(payload,reference,requests)
    c=module._compile_timing_schedule(payload,dict(reference,sha256='b'*64),requests)
    assert a==b and a['comparisons'][0]['orders']!=c['comparisons'][0]['orders']
