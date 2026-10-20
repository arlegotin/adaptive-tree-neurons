"""Known fixture labels exercise scoring; no true public qualification is faked."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.protected_label_evaluator_v2'), 'Missing once-only frozen label evaluator'
    from treeneuron.research import protected_label_evaluator_v2
    return protected_label_evaluator_v2


def test_public_evaluator_requires_genuine_protocol_before_stage_study_or_labels(tmp_path):
    module=api()
    refs=dict(models_reference=dict(path=str(tmp_path/'forbidden-models'),sha256='a'*64),
        global_barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='b'*64),
        qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='c'*64),
        preT_costs_reference=dict(path=str(tmp_path/'forbidden-costs'),sha256='d'*64),
        readiness_reference=dict(path=str(tmp_path/'forbidden-readiness'),sha256='e'*64))
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module.evaluate_protected_labels(dict(path=str(tmp_path/'forbidden-protocol'),sha256='f'*64),
            prediction_stage_reference=dict(path=str(tmp_path/'forbidden-stage'),sha256='0'*64),
            directory=tmp_path/'forbidden-study',**refs)
    assert 'forbidden-' not in str(error.value)
    assert list(tmp_path.iterdir())==[]


def arrays():
    F=dict(y=np.array([0.,2.]),row_id=np.array([100,101],dtype=np.int64))
    target=dict(y=np.zeros(3),row_id=np.arange(3,dtype=np.int64))
    prediction=dict(prediction=np.array([1.,1.,3.],dtype=np.float32),
        row_id=np.arange(3,dtype=np.int64),group_id=np.array([10,10,20],dtype=np.int64))
    return F,target,{'query-a':prediction}


def test_original_unit_scoring_weights_groups_then_rows_and_uses_F_ddof_zero():
    F,target,predictions=arrays()
    out=api()._score_fold_arrays(F=F,labels=target,predictions=predictions,temporal=False)
    assert out['arithmetic']['F_target_variance']==1.
    metrics=out['arithmetic']['methods']['query-a']
    assert metrics['primary_MSE']==5. and metrics['normalized_risk']==5.
    assert metrics['row_MSE']==pytest.approx(11/3)
    assert out['group_ids']==[10,20]
    assert out['normalized_group_losses']=={'query-a':[1.,9.]}
    assert out['arithmetic']['protected_access_authenticated'] is False


@pytest.mark.parametrize('fault',['rows','overlap','nan','dtype','prediction_group','prediction_rows','extra_label'])
def test_scoring_rejects_misaligned_leaky_or_invalid_numeric_roles(fault):
    F,target,predictions=arrays()
    if fault=='rows':target['row_id']=target['row_id'][::-1]
    elif fault=='overlap':F['row_id'][0]=0
    elif fault=='nan':target['y'][0]=np.nan
    elif fault=='dtype':target['y']=target['y'].astype(np.float32)
    elif fault=='prediction_group':predictions['query-b']={**predictions['query-a'],'group_id':np.arange(3,dtype=np.int64)}
    elif fault=='prediction_rows':predictions['query-a']['row_id']=np.array([1,0,2],dtype=np.int64)
    else:target['extra']=np.ones(3)
    with pytest.raises(ValueError):api()._score_fold_arrays(F=F,labels=target,predictions=predictions,temporal=False)


def test_temporal_scoring_uses_actual_days_and_keeps_missing_hours():
    F,target,predictions=arrays();predictions['query-a']['timestamps']=np.array(['2012-01-01T01','2012-01-01T23','2012-01-02T04'],dtype='datetime64[h]')
    predictions['query-a']['group_id']=predictions['query-a']['timestamps'].astype('datetime64[D]').astype(np.int64)
    out=api()._score_fold_arrays(F=F,labels=target,predictions=predictions,temporal=True)
    assert out['arithmetic']['grouping']=='day'
    assert out['day_ordinals']==[15340,15341]
    assert out['normalized_group_losses']['query-a']==[1.,9.]


def complete_fixture(tmp_path,*,augment=None):
    from test_protected_prediction_queue_v2 import local_protocol,owned_fixture_executor
    from test_protected_query_worker_v2 import ref
    from treeneuron.research.protected_prediction_queue_v2 import _run_prediction_queue
    payload,reference=local_protocol(tmp_path);fold=payload['data']['toy']['folds'][0]
    root=Path(payload['data']['toy']['root']);F=root/'outer-0/F.npz';target=root/'protected/outer-0/T-labels.npz';target.parent.mkdir(parents=True)
    with np.load(root/'outer-0/T-features.npz',allow_pickle=False) as archive:rows=archive['row_id'].copy()
    np.savez(F,y=np.array([0.,2.]),row_id=np.array([100000,100001],dtype=np.int64),X=np.zeros((2,4)),group_id=np.array([100000,100001],dtype=np.int64))
    np.savez(target,y=np.zeros(len(rows)),row_id=rows)
    fold.update(primary=True);fold['roles'].update(F=dict(reference=ref(F),bytes=F.stat().st_size),
        T_labels=dict(reference=ref(target),bytes=target.stat().st_size))
    payload['prediction_manifest']['view_bindings']=[dict(job_id=q['job_id'],dataset='toy',outer_fold=0,family_id='P13',
        search_repeat=r,view='native_quality',status='retained_model_bytes',deadline_ns=None,query_id=q['query_id'])
        for r,q in enumerate(payload['prediction_manifest']['queries'])]
    if augment is not None:augment(payload)
    path=Path(reference['path']);path.write_text(json.dumps(payload,sort_keys=True)+'\n');reference=ref(path)
    seen=[];study=tmp_path/'study'
    result=_run_prediction_queue(protocol_payload=payload,protocol_reference=reference,directory=study,owner_root=tmp_path/'owners',
        execute_episode=owned_fixture_executor(seen),guard=lambda:None,rss_limit_bytes=1024**3)
    assert result['status']=='all_predictions_committed',result
    return payload,reference,study,seen


def run_fixture(module,payload,reference,study,tmp_path,guard=lambda:None):
    return module._evaluate_label_stage(protocol_payload=payload,protocol_reference=reference,directory=study,
        owner_root=tmp_path/'owners',guard=guard)


def test_private_owned_fixture_scores_once_and_idempotent_replay_opens_no_labels(tmp_path,monkeypatch):
    module=api();payload,reference,study,seen=complete_fixture(tmp_path)
    out=run_fixture(module,payload,reference,study,tmp_path)
    assert out['status']=='metrics_committed' and out['label_roles_read_this_call']==1
    assert out['native_qualification_authenticated'] is out['protected_access_authorized'] is out['T_accessed'] is False
    completion=json.loads(Path(out['metrics_reference']['path']).read_text())
    saved=json.loads(Path(completion['metrics_reference']['path']).read_text())
    assert len(saved['folds'])==1 and len(saved['folds'][0]['arithmetic']['methods'])==2
    assert len(saved['view_bindings'])==2 and len(seen)==2
    from treeneuron.research.protected_journal_v2 import _read_metadata_study
    assert _read_metadata_study(study)['phase']=='metrics_committed'
    def forbidden(*a,**kw):raise AssertionError('Committed readback must not reopen label arrays')
    monkeypatch.setattr(module,'_read_role',forbidden)
    again=run_fixture(module,payload,reference,study,tmp_path)
    assert again['metrics_reference']==out['metrics_reference'] and again['label_roles_read_this_call']==0


def test_label_access_begins_only_after_full_guard_and_failed_access_is_retained(tmp_path):
    module=api();payload,reference,study,seen=complete_fixture(tmp_path)
    target=Path(payload['data']['toy']['folds'][0]['roles']['T_labels']['reference']['path'])
    target.write_bytes(b'changed')
    with pytest.raises(ValueError):run_fixture(module,payload,reference,study,tmp_path)
    from treeneuron.research.protected_journal_v2 import _read_metadata_study
    assert _read_metadata_study(study)['phase']=='labels_started'
    ends=list((study/'labels/accesses').glob('*-end.json'))
    assert ends and any(json.loads(p.read_text())['status']=='failed' for p in ends)
    with pytest.raises(ValueError,match='ambiguous|incomplete|closed'):
        run_fixture(module,payload,reference,study,tmp_path)


def test_final_genuine_guard_failure_does_not_commit_metrics_or_allow_re_evaluation(tmp_path):
    module=api();payload,reference,study,seen=complete_fixture(tmp_path);calls=[]
    def guard():
        calls.append(1)
        if len(calls)==2:raise ValueError('Fixture changed final context')
    with pytest.raises(ValueError,match='changed final'):run_fixture(module,payload,reference,study,tmp_path,guard)
    from treeneuron.research.protected_journal_v2 import _read_metadata_study
    assert _read_metadata_study(study)['phase']=='labels_started'
    recovered=run_fixture(module,payload,reference,study,tmp_path)
    assert recovered['status']=='metrics_committed' and recovered['label_roles_read_this_call']==0


def test_ambiguous_started_stage_cannot_reset_or_read_labels(tmp_path,monkeypatch):
    module=api();payload,reference,study,seen=complete_fixture(tmp_path)
    from treeneuron.research.protected_journal_v2 import _append_metadata_event
    _append_metadata_event(study,kind='labels_started',details={})
    def forbidden(*a,**kw):raise AssertionError('Ambiguous stage cannot reopen labels')
    monkeypatch.setattr(module,'_read_role',forbidden)
    with pytest.raises(ValueError,match='ambiguous|incomplete|closed'):run_fixture(module,payload,reference,study,tmp_path)


def test_failed_full_guard_prevents_label_stage_or_array_reads(tmp_path,monkeypatch):
    module=api();payload,reference,study,seen=complete_fixture(tmp_path)
    def forbidden(*a,**kw):raise AssertionError('No arrays before full guard')
    monkeypatch.setattr(module,'_read_role',forbidden)
    def guard():raise ValueError('Fixture full context denied')
    with pytest.raises(ValueError,match='full context denied'):run_fixture(module,payload,reference,study,tmp_path,guard)
    from treeneuron.research.protected_journal_v2 import _read_metadata_study
    assert _read_metadata_study(study)['phase']=='all_predictions_committed'
    assert not (study/'labels').exists()


def test_role_reader_rejects_changed_source_before_any_label_or_output_io(tmp_path):
    module=api();case=dict(dataset='toy',outer_fold=0,temporal=False,
        T_labels=dict(reference=dict(path=str(tmp_path/'forbidden-labels'),sha256='a'*64),bytes=10))
    with pytest.raises(ValueError,match='source'):
        module._read_role(case,'T_labels',root=tmp_path/'output',source_sha256='0'*64,
            protocol_sha256='b'*64,genesis_sha256='c'*64,index=0)
    assert list(tmp_path.iterdir())==[]


def test_exact_target_archive_rejects_extra_supervised_fields():
    import io
    stream=io.BytesIO();np.savez(stream,y=np.array([1.,2.]),row_id=np.array([1,2]),proxy_target=np.array([3.,4.]))
    with pytest.raises(ValueError,match='extra fields'):api()._npz(stream.getvalue(),['y','row_id'])


def test_label_metric_stage_does_not_load_fit_or_predict_a_model(tmp_path,monkeypatch):
    module=api();payload,reference,study,seen=complete_fixture(tmp_path)
    from treeneuron.research.p13_simple import P13Predictor
    def forbidden(*a,**kw):raise AssertionError('Label scoring cannot load, fit or predict a model')
    for name in ('load','fit','predict'):monkeypatch.setattr(P13Predictor,name,forbidden)
    result=run_fixture(module,payload,reference,study,tmp_path)
    assert result['status']=='metrics_committed' and len(seen)==2


def test_changed_committed_metric_bytes_cannot_be_reissued_or_re_evaluated(tmp_path,monkeypatch):
    module=api();payload,reference,study,seen=complete_fixture(tmp_path)
    result=run_fixture(module,payload,reference,study,tmp_path)
    complete=json.loads(Path(result['metrics_reference']['path']).read_text())
    with Path(complete['metrics_reference']['path']).open('ab') as stream:stream.write(b'changed')
    def forbidden(*a,**kw):raise AssertionError('Changed output cannot trigger another label read')
    monkeypatch.setattr(module,'_read_role',forbidden)
    with pytest.raises(ValueError,match='bytes changed'):run_fixture(module,payload,reference,study,tmp_path)
