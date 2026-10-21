"""Readonly known-target evidence is not a native qualification capability."""
import hashlib
import json
from pathlib import Path

import pytest

from test_independent_raw_audit_v2 import prepared,audit_fixture


def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def api():
    from treeneuron.research import independent_raw_audit_v2 as module
    assert hasattr(module,'_inspect_raw_audit_evidence'),'Missing independent audit evidence reader'
    assert hasattr(module,'load_protected_raw_audit'),'Missing genuine bound-audit loader'
    return module


def context(tmp_path):
    module=api();payload,reference,study,result=prepared(tmp_path)
    audit_fixture(module,tmp_path,(payload,reference,study,result))
    from treeneuron.research.protected_prediction_gate_v2 import _inspect_complete_stage
    inspection=_inspect_complete_stage(protocol_payload=payload,protocol_reference=reference,directory=study,owner_root=tmp_path/'owners')
    completion=ref(study/'labels/completion.json');closed=json.loads((study/'labels/completion.json').read_text())
    metrics=json.loads(Path(closed['metrics_reference']['path']).read_text())
    return module,dict(protocol_payload=payload,protocol_reference=reference,inspection=inspection,metrics=metrics,
        completion_reference=completion,raw_reference=ref(tmp_path/'raw-audit/audit.json'),terminal_reference=ref(tmp_path/'raw-audit/terminal.json'))


def test_public_bound_audit_loader_never_uses_supplied_true_flags_before_actual_gates(tmp_path):
    module=api();fake=dict(path=str(tmp_path/'forbidden.json'),sha256='a'*64)
    with pytest.raises((ValueError,FileNotFoundError)):
        module.load_protected_raw_audit(fake,protocol_reference=fake,analysis_reference=fake,prediction_stage_reference=fake,
            directory=tmp_path/'forbidden-study',models_reference=fake,global_barrier_reference=fake,
            qualification_reference=dict(root=str(tmp_path/'false-q'),inventory_sha256='b'*64),preT_costs_reference=fake,readiness_reference=fake)
    assert list(tmp_path.iterdir())==[]


def test_readonly_reader_checks_exact_accesses_without_target_decode_or_model_operation(tmp_path,monkeypatch):
    module,kwargs=context(tmp_path)
    from treeneuron.research import protected_label_evaluator_v2 as scorer
    from treeneuron.research.p13_simple import P13Predictor
    def forbidden(*args,**kw):pytest.fail('Readback decoded targets, reran audit or called a model')
    monkeypatch.setattr(scorer,'_read_role',forbidden);monkeypatch.setattr(module,'_independent_fold_arrays',forbidden)
    for name in ('load','fit','predict'):monkeypatch.setattr(P13Predictor,name,forbidden)
    before={p:p.read_bytes() for p in (tmp_path/'raw-audit').rglob('*') if p.is_file()}
    result=module._inspect_raw_audit_evidence(**kwargs)
    assert result['raw_arithmetic_evidence_checked'] is True and result['numeric_comparisons']==35
    assert result['max_absolute_difference']==0. and result['auditor_role_reads']==2
    assert result['native_models_authenticated'] is result['T_accessed'] is result['claim_authorized'] is False
    assert all(p.read_bytes()==raw for p,raw in before.items())


@pytest.mark.parametrize('fault',['wrong_metric','source','metrics_hash','unknown_field','access_actor','access_role','access_clock','terminal_status','terminal_time','orphan_access','orphan_file','missing_reconstruction'])
def test_rehashed_but_invalid_audit_evidence_fails_closed(tmp_path,fault):
    module,kwargs=context(tmp_path);root=tmp_path/'raw-audit';raw=json.loads((root/'audit.json').read_text());terminal=json.loads((root/'terminal.json').read_text())
    if fault=='wrong_metric':
        next(iter(raw['reconstructions'][0]['methods'].values()))['primary_MSE']+=1.
    elif fault=='source':raw['source_sha256']='0'*64
    elif fault=='metrics_hash':raw['metric_inputs_sha256']='1'*64
    elif fault=='unknown_field':raw['native_override']=True
    elif fault.startswith('access_'):
        end_path=root/'accesses/0000-end.json';end=json.loads(end_path.read_text());start_path=Path(end['start_reference']['path']);start=json.loads(start_path.read_text())
        if fault=='access_actor':start['actor']='evaluator'
        elif fault=='access_role':start['role']='V_labels'
        else:end['elapsed_ns']=-1
        start_path.write_text(json.dumps(start)+'\n');end['start_reference']=ref(start_path);end_path.write_text(json.dumps(end)+'\n');raw['access_references'][0]=ref(end_path)
    elif fault=='terminal_status':terminal['status']='audit_failure'
    elif fault=='terminal_time':terminal['elapsed_ns']=-1
    elif fault=='orphan_access':(root/'accesses/forgotten-start.json').write_text('{}\n')
    elif fault=='orphan_file':(root/'forgotten-failure.json').write_text('{}\n')
    else:raw['reconstructions']=[]
    (root/'audit.json').write_text(json.dumps(raw)+'\n');terminal['audit_reference']=ref(root/'audit.json');(root/'terminal.json').write_text(json.dumps(terminal)+'\n')
    kwargs.update(raw_reference=ref(root/'audit.json'),terminal_reference=ref(root/'terminal.json'))
    with pytest.raises(ValueError):module._inspect_raw_audit_evidence(**kwargs)


def test_actual_metric_consumer_requires_a_witnessed_plan_reference_not_a_phase_flag(tmp_path):
    from treeneuron.research import native_deployment_plan_v2 as plans
    assert hasattr(plans,'_required_plan_reference'),'Missing timing-plan consistency validator'
    with pytest.raises(ValueError):plans._required_plan_reference(dict(phase='metrics_committed',deployment_plan_reference=None))
    plan=tmp_path/'deployment-plan.json';plan.write_text('{}\n');reference=ref(plan)
    assert plans._required_plan_reference(dict(phase='metrics_committed',deployment_plan_reference=reference))==reference
    plan.write_text('{"changed":true}\n')
    with pytest.raises(ValueError):plans._required_plan_reference(dict(phase='metrics_committed',deployment_plan_reference=reference))
