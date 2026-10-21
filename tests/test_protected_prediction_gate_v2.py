"""Complete prediction evidence is a gate, never a caller-supplied label flag."""
import importlib.util
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.protected_prediction_gate_v2'),'Missing complete protected prediction gate'
    from treeneuron.research import protected_prediction_gate_v2
    return protected_prediction_gate_v2


@pytest.mark.parametrize('entry',['seal','load'])
def test_public_complete_stage_requires_genuine_protocol_before_output_or_study(tmp_path,entry):
    module=api();refs=dict(models_reference=dict(path=str(tmp_path/'forbidden-models'),sha256='a'*64),
        global_barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='b'*64),
        qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='c'*64),
        preT_costs_reference=dict(path=str(tmp_path/'forbidden-cost'),sha256='d'*64),
        readiness_reference=dict(path=str(tmp_path/'forbidden-readiness'),sha256='e'*64))
    protocol=dict(path=str(tmp_path/'forbidden-protocol'),sha256='f'*64)
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        if entry=='seal':
            module.seal_protected_prediction_stage(protocol,directory=tmp_path/'forbidden-study',output=tmp_path/'stage.json',**refs)
        else:
            module.load_protected_prediction_stage(dict(path=str(tmp_path/'forbidden-stage'),sha256='0'*64),
                protocol_reference=protocol,directory=tmp_path/'forbidden-study',**refs)
    assert all(name not in str(error.value) for name in ('forbidden-models','forbidden-protocol','forbidden-stage','forbidden-cost','forbidden-study'))
    assert list(tmp_path.iterdir())==[]


def local_complete_stage(tmp_path):
    from test_protected_prediction_queue_v2 import local_protocol,owned_fixture_executor
    from treeneuron.research.protected_prediction_queue_v2 import _run_prediction_queue
    payload,reference=local_protocol(tmp_path);seen=[];root=tmp_path/'study'
    result=_run_prediction_queue(protocol_payload=payload,protocol_reference=reference,directory=root,owner_root=tmp_path/'owners',
        execute_episode=owned_fixture_executor(seen),guard=lambda:None,rss_limit_bytes=1024**3)
    assert result['status']=='all_predictions_committed',result
    return payload,reference,root,seen


def test_private_complete_stage_recomputes_all_commits_and_closed_invocations_without_labels(tmp_path):
    module=api();payload,reference,root,seen=local_complete_stage(tmp_path)
    out=module._inspect_complete_stage(protocol_payload=payload,protocol_reference=reference,directory=root,owner_root=tmp_path/'owners')
    assert out['query_count']==2 and out['completed_query_count']==2
    assert out['invocation_count']==1 and out['call_count']==2
    assert out['all_prediction_evidence_checked'] is True
    assert out['native_qualification_authenticated'] is out['protected_access_authorized'] is out['T_accessed'] is False
    assert out['whole_training_procedure_cost_authenticated'] is False
    assert len(seen)==2


@pytest.mark.parametrize('fault',['missing_end','missing_start','partial_end','unknown_end_field','wrong_genesis','wrong_duration',
    'missing_call','duplicated_call','wrong_call_hash','orphan_call'])
def test_complete_stage_rejects_missing_or_inconsistent_invocation_evidence(tmp_path,fault):
    import json
    module=api();payload,reference,root,seen=local_complete_stage(tmp_path)
    end=next((root/'invocations').glob('*-end.json'));start=next((root/'invocations').glob('*-start.json'))
    value=json.loads(end.read_text())
    if fault=='missing_end':end.unlink()
    elif fault=='missing_start':start.unlink()
    elif fault=='partial_end':end.write_bytes(end.read_bytes()[:-4])
    elif fault=='unknown_end_field':value['extra']=True
    elif fault=='wrong_genesis':value['genesis_sha256']='0'*64
    elif fault=='wrong_duration':value['elapsed_ns']+=1
    elif fault=='missing_call':value['call_references'].pop()
    elif fault=='duplicated_call':value['call_references'].append(value['call_references'][0])
    elif fault=='wrong_call_hash':value['call_references'][0]['sha256']='0'*64
    elif fault=='orphan_call':
        orphan=root/'queries'/('0'*64)/'episode-0000/call-clock.json';orphan.parent.mkdir(parents=True);orphan.write_text('{}\n')
    if fault not in ('missing_end','missing_start','partial_end','orphan_call'):end.write_text(json.dumps(value)+'\n')
    with pytest.raises((ValueError,FileNotFoundError)):
        module._inspect_complete_stage(protocol_payload=payload,protocol_reference=reference,directory=root,owner_root=tmp_path/'owners')


def test_stage_output_preserves_models_data_proofs_study_and_owner_archive(tmp_path):
    module=api();study=tmp_path/'study';owners=tmp_path/'owners';data=tmp_path/'data';model=tmp_path/'model';proof=tmp_path/'proof'
    inspection=dict(study_directory=str(study),owner_root=str(owners),invocation_references=[])
    payload=dict(data={'toy':dict(root=str(data))},prediction_manifest=dict(queries=[dict(model_binding=dict(snapshot_root=str(model)))]))
    assert hasattr(module,'_check_stage_output'),'Missing complete-stage output guard'
    for output in (study/'stage.json',owners/'stage.json',data/'stage.json',model/'stage.json',proof/'stage.json'):
        with pytest.raises(ValueError):
            module._check_stage_output(output,inspection=inspection,protocol_payload=payload,protected_roots=[str(proof)],input_files=[])
        assert not output.exists()
    assert module._check_stage_output(tmp_path/'release/stage.json',inspection=inspection,protocol_payload=payload,
        protected_roots=[str(proof)],input_files=[])==tmp_path/'release/stage.json'


@pytest.mark.parametrize('kind',['directory','markdown'])
def test_publication_guard_accepts_actual_programme_and_report_output_types_without_weakening_inputs(tmp_path,kind):
    module=api();study=tmp_path/'protected-study';owners=tmp_path/'owners';proof=tmp_path/'model-proof'
    inspection=dict(study_directory=str(study),owner_root=str(owners),invocation_references=[])
    payload=dict(data={},prediction_manifest=dict(queries=[]))
    kwargs=dict(inspection=inspection,protocol_payload=payload,protected_roots=[str(proof)],input_files=[])
    target=tmp_path/('programme' if kind=='directory' else 'report.md')
    assert module._check_stage_output(target,kind=kind,**kwargs)==target
    if kind=='directory':target.mkdir()
    else:target.write_text('unchanged report')
    with pytest.raises(ValueError):module._check_stage_output(target,kind=kind,**kwargs)
    if kind=='directory':
        assert module._check_stage_output(target,kind=kind,fresh=False,**kwargs)==target
    for root in (study,owners,proof):
        with pytest.raises(ValueError):
            module._check_stage_output(root/target.name,kind=kind,fresh=False,**kwargs)
    with pytest.raises(ValueError):module._check_stage_output(target,**kwargs)


def test_readonly_complete_gate_never_loads_fits_or_predicts_again(tmp_path,monkeypatch):
    module=api();payload,reference,root,seen=local_complete_stage(tmp_path)
    from treeneuron.research.p13_simple import P13Predictor
    def forbidden(*a,**kw):raise AssertionError('Complete stage readback must not fit/load/predict a model')
    monkeypatch.setattr(P13Predictor,'fit',forbidden);monkeypatch.setattr(P13Predictor,'load',forbidden);monkeypatch.setattr(P13Predictor,'predict',forbidden)
    result=module._inspect_complete_stage(protocol_payload=payload,protocol_reference=reference,directory=root,owner_root=tmp_path/'owners')
    assert result['all_prediction_evidence_checked'] is True and result['labels_read'] is False


def test_prediction_stage_is_stable_after_later_private_metadata_metrics_event(tmp_path):
    module=api();payload,reference,root,seen=local_complete_stage(tmp_path)
    before=module._inspect_complete_stage(protocol_payload=payload,protocol_reference=reference,directory=root,owner_root=tmp_path/'owners')
    from treeneuron.research import protected_journal_v2 as journal
    journal._append_metadata_event(root,kind='labels_started',details={})
    target=tmp_path/'untrusted-metric-marker.json';target.write_text('{"no_labels_read":true}\n')
    import hashlib
    ref=dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    journal._append_metadata_event(root,kind='metrics_committed',details=dict(metrics_reference=ref,elapsed_ns=11))
    after=module._inspect_complete_stage(protocol_payload=payload,protocol_reference=reference,directory=root,owner_root=tmp_path/'owners')
    assert before==after
    assert after['protected_access_authorized'] is after['T_accessed'] is False


def test_invocation_bytes_changed_during_readback_are_not_sealed(tmp_path,monkeypatch):
    module=api();payload,reference,root,seen=local_complete_stage(tmp_path)
    original=module._invocations
    def changing(*args,**kwargs):
        result=original(*args,**kwargs)
        path=Path(result[0][0]['path'])
        with path.open('ab') as stream:stream.write(b'changed')
        return result
    monkeypatch.setattr(module,'_invocations',changing)
    with pytest.raises(ValueError):
        module._inspect_complete_stage(protocol_payload=payload,protocol_reference=reference,directory=root,owner_root=tmp_path/'owners')
