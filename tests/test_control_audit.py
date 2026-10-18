"""Retained prediction arithmetic is audited separately from model replay."""
from pathlib import Path
import hashlib
import math

import numpy as np
import pytest


def fixture(tmp_path):
    from treeneuron.config import canonical_hash
    from treeneuron.research.ledger import AttemptLedger
    root=tmp_path/'study'
    case=dict(attempt_id='positive-gamma-2.0-sigma-0.1/teacher-0/repeat-0/random',
        cell='positive-gamma-2.0-sigma-0.1',population='positive',gamma=2.,sigma=.1,
        teacher_index=0,repetition=0,role='random')
    plan=dict(schema='research/v2-plan',source_sha256='a'*64,config=dict(kind='m1_controls'),attempts=[case])
    plan['plan_hash']=canonical_hash(plan)
    ledger=AttemptLedger.create(root,plan);(root/'worker-lease.json').touch()
    pair=root/case['cell']/'teacher-0/repeat-0-attempt-0';(pair/'models').mkdir(parents=True)
    data=pair.parent/'data-repeat-0-attempt-0';data.mkdir()
    arrays=dict(F_y=np.array([0.,2.]),V_y=np.array([1.,3.]),E_y=np.array([2.,4.,6.]),
        V_row_id=np.array(['v0','v1']),E_row_id=np.array(['e0','e1','e2']))
    np.savez_compressed(data/'observations.npz',**arrays)
    np.savez_compressed(pair/'predictions.npz',**{k:v for k,v in arrays.items() if k!='F_y'},
        random_V=np.array([1.,2.]),random_E=np.array([2.,3.,4.]))
    pointer=pair/'models/random.json';pointer.write_text('retained model pointer bytes')
    metrics=dict(V=dict(n=2,mse=.5,normalized_mse=.5,rmse=math.sqrt(.5),mae=.5,
        absolute_error_p95=.95,max_absolute_error=1.),E=dict(n=3,mse=5/3,normalized_mse=5/3,
        rmse=math.sqrt(5/3),mae=1.,absolute_error_p95=1.9,max_absolute_error=2.))
    ledger.record(case['attempt_id'],'running',elapsed_ns=0,recovery_attempt=0)
    ledger.record(case['attempt_id'],'completed_eligible',elapsed_ns=15,metrics=metrics,
        retained_model=dict(path=str(pointer),sha256=hashlib.sha256(pointer.read_bytes()).hexdigest()))
    return root,plan,ledger,pair,data


def test_raw_arithmetic_authenticates_rows_variance_status_and_honest_level(tmp_path):
    from treeneuron.research.control_audit import audit_control_study
    root,plan,ledger,_,_=fixture(tmp_path)
    result=audit_control_study(root,expected_plan_hash=plan['plan_hash'])
    assert result['status']=='completed' and result['audited_predictions']==2
    assert result['rows'][0]['V_NMSE']==.5 and result['rows'][0]['metrics']['E']['mse']==5/3
    assert result['evidence_level']=='retained_raw_prediction_arithmetic'
    assert result['model_predictions_reexecuted'] is False
    assert result['positive_sensitivity']['sensitive'] is None
    assert result['charged_standalone_ns']==15 and result['metric_comparisons']==14
    with pytest.raises(ValueError,match='plan'):
        audit_control_study(root,expected_plan_hash='f'*64)


def test_raw_prediction_or_row_corruption_cannot_reuse_saved_metrics(tmp_path):
    from treeneuron.research.control_audit import audit_control_study
    root,plan,_,pair,data=fixture(tmp_path)
    path=pair/'predictions.npz'
    with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
    arrays['random_E'][0]+=1
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='metric'):
        audit_control_study(root,expected_plan_hash=plan['plan_hash'])
    arrays['random_E'][0]-=1;arrays['E_row_id']=arrays['E_row_id'][::-1]
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='row'):
        audit_control_study(root,expected_plan_hash=plan['plan_hash'])


def test_audit_rejects_active_worker_and_retains_missing_attempt(tmp_path):
    from treeneuron.config import canonical_hash
    from treeneuron.research.control_audit import audit_control_study
    from treeneuron.research.ledger import StudyLease,AttemptLedger
    root,plan,_,_,_=fixture(tmp_path)
    with StudyLease(root):
        with pytest.raises(PermissionError,match='active'):
            audit_control_study(root,expected_plan_hash=plan['plan_hash'])
    missing=dict(plan['attempts'][0],attempt_id='missing',role='oracle_true')
    # A separate immutable fixture plan declares a never-started method.
    other=tmp_path/'empty';plan=dict(plan,attempts=[missing]);plan.pop('plan_hash');plan['plan_hash']=canonical_hash(plan)
    AttemptLedger.create(other,plan);(other/'worker-lease.json').touch()
    result=audit_control_study(other,expected_plan_hash=plan['plan_hash'])
    assert result['status']=='incomplete_or_ineligible' and result['rows'][0]['status']=='never_started'
    assert result['rows'][0].get('V_NMSE') is None


@pytest.mark.parametrize('ids',[np.array(['v0']),np.array(['v0','v0']),
    np.array([['v0'],['v1']]),np.array([0.,1.])])
def test_equal_but_invalid_row_identifiers_cannot_authenticate_alignment(tmp_path,ids):
    from treeneuron.research.control_audit import audit_control_study
    root,plan,_,pair,data=fixture(tmp_path)
    for path in (data/'observations.npz',pair/'predictions.npz'):
        with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
        arrays['V_row_id']=ids;np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='row'):
        audit_control_study(root,expected_plan_hash=plan['plan_hash'])
