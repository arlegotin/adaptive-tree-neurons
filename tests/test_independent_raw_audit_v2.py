"""Independent arithmetic and owned known-target fixtures, never native authority."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.independent_raw_audit_v2'), 'Missing independent raw protected audit'
    from treeneuron.research import independent_raw_audit_v2
    return independent_raw_audit_v2


def arrays():
    return (dict(y=np.array([0.,2.]),row_id=np.array([100,101],dtype=np.int64)),
        dict(y=np.zeros(3),row_id=np.arange(3,dtype=np.int64)),
        {'query-a':dict(prediction=np.array([1.,1.,3.],dtype=np.float32),
            row_id=np.arange(3,dtype=np.int64),group_id=np.array([10,10,20],dtype=np.int64))})


def test_independent_numpy_arithmetic_uses_original_units_and_equal_groups():
    F,labels,predictions=arrays();result=api()._independent_fold_arrays(F=F,labels=labels,predictions=predictions,temporal=False)
    assert result['F_target_variance']==1.
    assert result['methods']['query-a']['primary_MSE']==5.
    assert result['methods']['query-a']['row_MSE']==pytest.approx(11/3)
    assert result['normalized_group_losses']['query-a']==[1.,9.]


@pytest.mark.parametrize('fault',['rows','overlap','nan','target_dtype','prediction_dtype','group_geometry'])
def test_independent_reconstruction_rejects_invalid_or_misaligned_arrays(fault):
    F,labels,predictions=arrays()
    if fault=='rows':labels['row_id']=labels['row_id'][::-1]
    elif fault=='overlap':F['row_id'][0]=0
    elif fault=='nan':labels['y'][0]=np.nan
    elif fault=='target_dtype':labels['y']=labels['y'].astype(np.float32)
    elif fault=='prediction_dtype':predictions['query-a']['prediction']=np.array([1,1,3])
    else:predictions['query-b']={**predictions['query-a'],'group_id':np.array([1,2,3],dtype=np.int64)}
    with pytest.raises(ValueError):api()._independent_fold_arrays(F=F,labels=labels,predictions=predictions,temporal=False)


def test_independent_day_loss_preserves_missing_hours_and_zero_F_variance():
    F,labels,predictions=arrays();times=np.array(['2012-01-01T01','2012-01-01T23','2012-01-02T04'],dtype='datetime64[h]')
    predictions['query-a'].update(timestamps=times,group_id=times.astype('datetime64[D]').astype(np.int64))
    result=api()._independent_fold_arrays(F=F,labels=labels,predictions=predictions,temporal=True)
    assert result['group_ids']==[15340,15341] and result['methods']['query-a']['normalized_risk']==5.
    F['y'][:]=3.
    result=api()._independent_fold_arrays(F=F,labels=labels,predictions=predictions,temporal=True)
    assert result['methods']['query-a']['normalized_risk'] is None and result['normalized_group_losses']['query-a'] is None


def test_public_audit_requires_actual_gates_before_output_or_target_paths(tmp_path):
    module=api();kwargs=dict(models_reference=dict(path=str(tmp_path/'forbidden-models'),sha256='a'*64),
        global_barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='b'*64),
        qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='c'*64),
        preT_costs_reference=dict(path=str(tmp_path/'forbidden-cost'),sha256='d'*64),
        readiness_reference=dict(path=str(tmp_path/'forbidden-readiness'),sha256='e'*64))
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module.audit_protected_raw_predictions(dict(path=str(tmp_path/'forbidden-protocol'),sha256='f'*64),
            analysis_reference=dict(path=str(tmp_path/'forbidden-analysis'),sha256='1'*64),
            prediction_stage_reference=dict(path=str(tmp_path/'forbidden-stage'),sha256='0'*64),
            directory=tmp_path/'forbidden-study',output=tmp_path/'audit',**kwargs)
    assert 'forbidden-' not in str(error.value) and list(tmp_path.iterdir())==[]


def prepared(tmp_path,*,producer_fault=False,monkeypatch=None,bad_policy=False):
    from test_protected_label_evaluator_v2 import complete_fixture,run_fixture
    from treeneuron.research import protected_label_evaluator_v2 as scorer
    from treeneuron.research.practical_procedure_v2 import _policy
    def augment(payload):
        payload['policy']=_policy('known-fixture-only')
        if bad_policy:payload['policy']['information']['protected_audit']['configuration_selection']=True
    payload,reference,study,seen=complete_fixture(tmp_path,augment=augment)
    if producer_fault:
        original=scorer._score_fold_arrays
        def wrong(**kwargs):
            result=original(**kwargs)
            for metrics in result['arithmetic']['methods'].values():metrics['primary_MSE']+=1.
            return result
        monkeypatch.setattr(scorer,'_score_fold_arrays',wrong)
    result=run_fixture(scorer,payload,reference,study,tmp_path)
    return payload,reference,study,result


def audit_fixture(module,tmp_path,prepared_fixture):
    payload,reference,study,result=prepared_fixture
    return module._audit_raw_files(protocol_payload=payload,protocol_reference=reference,directory=study,
        owner_root=tmp_path/'owners',output=tmp_path/'raw-audit',guard=lambda:None)


def test_owned_known_fixture_audit_uses_separate_math_and_explicit_auditor_accesses(tmp_path,monkeypatch):
    module=api();data=prepared(tmp_path)
    from treeneuron.research import protected_label_evaluator_v2 as scorer
    from treeneuron.analysis import prediction_arithmetic_v2 as producer
    from treeneuron.research.p13_simple import P13Predictor
    def forbidden(*a,**kw):raise AssertionError('Independent raw audit cannot use producer arithmetic or model load/fit/predict')
    monkeypatch.setattr(scorer,'_score_fold_arrays',forbidden);monkeypatch.setattr(producer,'audit_prediction_arrays',forbidden)
    for name in ('load','fit','predict'):monkeypatch.setattr(P13Predictor,name,forbidden)
    result=audit_fixture(module,tmp_path,data)
    assert result['status']=='matched_raw_metrics' and result['query_count']==2
    assert result['max_absolute_difference']<1e-10
    assert result['native_models_authenticated'] is result['protected_access_authorized'] is result['T_accessed'] is False
    starts=list((tmp_path/'raw-audit/accesses').glob('*-start.json'))
    assert len(starts)==2 and {json.loads(p.read_text())['actor'] for p in starts}=={'auditor'}


def test_independent_audit_detects_real_wrong_producer_arithmetic_despite_valid_file_commits(tmp_path,monkeypatch):
    module=api();data=prepared(tmp_path,producer_fault=True,monkeypatch=monkeypatch)
    with pytest.raises(ValueError,match='reconstruction|metric'):
        audit_fixture(module,tmp_path,data)
    assert (tmp_path/'raw-audit/terminal.json').is_file()
    assert json.loads((tmp_path/'raw-audit/terminal.json').read_text())['status']=='audit_failure'


def test_audit_without_explicit_fixed_verification_policy_cannot_read_labels(tmp_path):
    module=api();F,labels,predictions=arrays()
    from treeneuron.research.deployment_worker_v2 import _source
    payload=dict(source_sha256=_source(),policy=dict(information=dict(protected_audit=dict(configuration_selection=True))))
    with pytest.raises(ValueError,match='policy'):
        module._check_audit_policy(payload)
    assert list(tmp_path.iterdir())==[]


def test_main_audit_rejects_tuning_policy_before_output_or_role_read(tmp_path,monkeypatch):
    module=api();data=prepared(tmp_path,bad_policy=True)
    from treeneuron.research import protected_label_evaluator_v2 as scorer
    def forbidden(*a,**kw):raise AssertionError('Forbidden audit policy cannot read another role')
    monkeypatch.setattr(scorer,'_read_role',forbidden)
    with pytest.raises(ValueError,match='policy'):audit_fixture(module,tmp_path,data)
    assert not (tmp_path/'raw-audit').exists()
