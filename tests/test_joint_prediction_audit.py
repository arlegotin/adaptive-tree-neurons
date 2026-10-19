"""M3 evidence reuses independent arithmetic without inheriting M2 gate claims."""
import copy
import math

import pytest

from test_opportunity_prediction_audit import fixture
from test_train import cpu_training_evidence


def test_M3_retained_predictions_are_reconstructed_under_their_own_evidence_schema(tmp_path):
    from treeneuron.research import opportunity_prediction_audit as module
    root,plan,last=fixture(tmp_path,kind='m3_joint')
    out=module.audit_joint_predictions(root,expected_plan_hash=plan['plan_hash'],expected_ledger_hash=last)
    assert out['schema']=='research/v2-joint-prediction-audit'
    assert out['audited_prediction_vectors']==2 and out['maximum_absolute_metric_difference']<1e-14
    assert out['rows'][0]['metrics']['selected']['E']['mse']==pytest.approx(5/3)
    assert out['Gate_B_decided'] is False and out['efficacy_claim'] is False and out['T_accessed'] is False


@pytest.mark.parametrize('mutation',['metric','rows'])
def test_M3_cannot_authenticate_wrong_prediction_arithmetic_or_alignment(tmp_path,mutation):
    from treeneuron.research import opportunity_prediction_audit as module
    root,plan,last=fixture(tmp_path,kind='m3_joint',mutation=mutation)
    with pytest.raises(ValueError):
        module.audit_joint_predictions(root,expected_plan_hash=plan['plan_hash'],expected_ledger_hash=last)


def test_M3_audit_retains_missing_evaluations_and_rejects_an_M2_queue(tmp_path):
    from treeneuron.research import opportunity_prediction_audit as module
    root,plan,last=fixture(tmp_path/'missing',kind='m3_joint',missing=True)
    out=module.audit_joint_predictions(root,expected_plan_hash=plan['plan_hash'],expected_ledger_hash=last)
    assert out['status']=='incomplete_or_failed' and out['rows'][1]['status']=='never_started'
    root,plan,last=fixture(tmp_path/'foreign')
    with pytest.raises(ValueError):
        module.audit_joint_predictions(root,expected_plan_hash=plan['plan_hash'],expected_ledger_hash=last)


def test_independent_audit_reads_real_native_M3_fixture_predictions(tmp_path,cpu_training_evidence):
    import json
    from test_joint_workers import joint_fixture
    from treeneuron.research.joint_workers import JointWorkers
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    from treeneuron.research.opportunity_prediction_audit import audit_joint_predictions
    from treeneuron.train import source_hash
    original=joint_fixture(cpu_training_evidence)
    identity=copy.deepcopy(original['identity']);identity['study_config']['kind']='m3_joint'
    identity['source_sha256']=source_hash()
    plan=make_queue_plan(identity,[{k:v for k,v in j.items() if k!='attempt_id'} for j in original['attempts']])
    root=tmp_path/'native';run_queue(plan,root,JointWorkers(plan),check_identity=lambda _:None)
    endpoint=json.loads((root/'events.jsonl').read_text().splitlines()[-1])['event_hash']
    result=audit_joint_predictions(root,expected_plan_hash=plan['plan_hash'],expected_ledger_hash=endpoint)
    assert result['status']=='completed_arithmetic' and result['audited_evaluation_jobs']==10
    assert result['audited_prediction_vectors']>=20 and result['maximum_absolute_metric_difference']<1e-12
    assert result['model_predictions_reexecuted'] is False and result['T_accessed'] is False


ROLES=('keep','top','random','short_singleton','local_singleton','joint',
    'unconditioned','pure_span','warm_upper','fresh_upper')


def audited_table(extended=False):
    rows=[]
    for teacher in range(16):
        for repeat in range(2):
            for role in ROLES:
                row=dict(job_id=f'evaluate:M3/teacher-{teacher}/repeat-{repeat}/{role}',
                    payload=dict(matrix='M3',teacher_index=teacher,repetition=repeat,role=role),
                    status='completed_eligible' if teacher<8 or extended else 'completed_ineligible')
                if row['status']=='completed_eligible':
                    risk=(1. if repeat==0 else 9.) if role=='random' else 1. if role=='joint' else 2.
                    row['metrics']={'selected':{'V':{'normalized_mse':risk},'E':{'normalized_mse':risk*.8}}}
                rows.append(row)
    return dict(schema='research/v2-joint-prediction-audit',status='completed_arithmetic',rows=rows,
        queue_plan_hash='a'*64,ledger_event_hash='b'*64,T_accessed=False,efficacy_claim=False)


@pytest.mark.parametrize('extended',[False,True])
def test_M3_summary_averages_both_repeats_before_log_ratio_and_preserves_skipped_teachers(extended):
    from treeneuron.research import joint_analysis as module
    result=module.summarize_joint_predictions(audited_table(extended))
    assert result['teachers']==(16 if extended else 8) and result['repetitions']==2
    assert result['role_means']['random']['V']==5. and result['role_means']['joint']['V']==1.
    contrast=result['contrasts']['random_over_joint']['V']
    assert contrast['mean_log_risk_ratio']==pytest.approx(math.log(5))
    assert all(r['log_risk_ratio']==pytest.approx(math.log(5)) for r in contrast['teacher_effects'])
    assert contrast['signs']=={'positive':result['teachers'],'zero':0,'negative':0}
    assert result['inactive_teachers']==([] if extended else list(range(8,16)))
    assert result['Gate_B_decided'] is False and result['confirmatory_claim'] is False


@pytest.mark.parametrize('mutation',['missing','duplicate','failed','risk','unpaired','foreign_role','wrong_schema'])
def test_incomplete_or_malformed_M3_evidence_cannot_become_a_complete_teacher_summary(mutation):
    from treeneuron.research import joint_analysis as module
    table=copy.deepcopy(audited_table())
    if mutation=='missing':table['rows'].pop(0)
    elif mutation=='duplicate':table['rows'].append(copy.deepcopy(table['rows'][0]))
    elif mutation=='failed':table['rows'][0]['status']='numerical_failure'
    elif mutation=='risk':table['rows'][0]['metrics']['selected']['V']['normalized_mse']=float('nan')
    elif mutation=='unpaired':table['rows'][0]['payload']['repetition']=2
    elif mutation=='foreign_role':table['rows'][0]['payload']['role']='new_unregistered_arm'
    else:table['schema']='research/v2-opportunity-prediction-audit'
    with pytest.raises(ValueError):module.summarize_joint_predictions(table)
