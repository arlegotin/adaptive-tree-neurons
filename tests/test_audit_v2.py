"""Saved-array audit fixtures, never protected observations or efficacy runs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from test_protocol import repository,git
from treeneuron.protocol import execution_snapshot


def api():
    assert importlib.util.find_spec('treeneuron.analysis.audit_v2') is not None, 'Missing raw development artifact auditor'
    from treeneuron.analysis import audit_v2
    return audit_v2


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(root,value):
    path=root/'artifacts/study/input.json';path.write_text(json.dumps(value))
    return path,sha(path)


def prepare(root):
    project=Path(__file__).resolve().parents[1]
    for name in ('audit_v2.py','prediction_arithmetic_v2.py','report_v2.py'):
        relative='src/treeneuron/analysis/'+name;p=root/relative;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(project/relative,p)
    git(root,'add','.');git(root,'commit','-qm','artificial committed audit source')
    commit=git(root,'rev-parse','HEAD');snapshot=execution_snapshot(root,commit)
    digest=hashlib.sha256()
    for p in sorted((root/'src/treeneuron').rglob('*.py')):
        digest.update(p.relative_to(root/'src/treeneuron').as_posix().encode());digest.update(p.read_bytes())
    target=root/'artifacts/study';target.mkdir(parents=True)
    np.savez(target/'arrays.npz',F_y=np.array([1.,3.]),y=np.array([0.,0.,0.]),
        row_id=np.array([1,2,3],dtype=np.int64),group_id=np.array([5,5,9],dtype=np.int64))
    for name,values in [('chosen',[1.,0.,0.]),('random',[0.,0.,2.])]:
        np.savez(target/(name+'.npz'),values=np.array(values),row_id=np.array([1,2,3],dtype=np.int64))
    def ref(name):return dict(path='artifacts/study/'+name,sha256=sha(target/name))
    methods=[dict(method_id=n,prediction_id='task/r0/'+n,status='completed_eligible',prediction=ref(n+'.npz'))
        for n in ('chosen','random')]
    value=dict(schema='research/v2-development-audit-input',study_id='saved-array-fixture',status='estimation',
        source=dict(commit=commit,package_sha256=digest.hexdigest(),execution_snapshot=snapshot),
        independent_units='One artificial grouped task; no teacher population inference.',
        cases=[dict(case_id='task/r0/E',task_id='task',repetition=0,role='E',grouping='group',arrays=ref('arrays.npz'),methods=methods)],
        attempts=[dict(attempt_id='anchor',status='completed_eligible'),dict(attempt_id='failed',status='numerical_failure')],
        limitations=['Artificial arrays only.'])
    path,digest=publish(root,value)
    return value,path,digest


def run(root,path,digest):
    return api().audit_development_study(path,expected_manifest_sha256=digest,repository_root=root)


@pytest.mark.parametrize('status',['completed','estimation','retired','blocked'])
def test_raw_files_rebuild_original_unit_group_metrics_and_canonical_report(status,repository):
    api();value,path,_=prepare(repository);value['status']=status;path,digest=publish(repository,value)
    result=run(repository,path,digest)
    assert result['status']==status and result['evidence_scope']=='development'
    rows={p['method_id']:p for p in result['predictions']}
    assert rows['chosen']['normalized_risk']==rows['chosen']['primary_MSE']==.25
    assert rows['random']['normalized_risk']==rows['random']['primary_MSE']==2.
    assert rows['chosen']['row_RMSE']==pytest.approx(np.sqrt(1/3))
    assert result['provenance']==dict(source_authenticated=True,raw_predictions_authenticated=True,protected_protocol_authenticated=False)
    assert result['comparisons']==[] and result['protocol_sha256'] is None
    assert result['costs']['physical_elapsed_ns'] is result['costs']['standalone_attributed_ns'] is None
    from treeneuron.analysis.report_v2 import render_report
    text=render_report(result);assert text==render_report(run(repository,path,digest))
    assert 'numerical_failure' in text and 'No protected confirmation' in text and 'Artificial arrays only.' in text
    assert 'passed' not in text


def test_missing_prediction_keeps_its_status_and_never_acquires_a_risk(repository):
    api();value,path,_=prepare(repository);method=value['cases'][0]['methods'][1]
    method.update(status='irrecoverable_missing',prediction=None);value['status']='blocked'
    path,digest=publish(repository,value);result=run(repository,path,digest)
    assert len(result['planned_prediction_ids'])==len(result['predictions'])==2
    row=result['predictions'][1]
    assert row['status']=='irrecoverable_missing' and row['normalized_risk'] is row['primary_MSE'] is None
    from treeneuron.analysis.report_v2 import render_report
    assert 'Incomplete planned predictions' in render_report(result)


@pytest.mark.parametrize('kind',['manifest','prediction_bytes','row_order','extra_target','nonfinite','object_payload','source','source_package','symlink','symlink_parent','traversal','duplicate_case','duplicate_method','duplicate_prediction_id','failed_with_prediction','unplanned_summary'])
def test_drift_alignment_payload_and_inventory_errors_are_rejected(kind,repository):
    api();value,path,digest=prepare(repository);case=value['cases'][0];target=repository/'artifacts/study/chosen.npz'
    if kind=='manifest':path.write_text(path.read_text()+' ')
    elif kind=='prediction_bytes':target.write_bytes(target.read_bytes()+b'changed')
    elif kind in ('row_order','extra_target','nonfinite','object_payload'):
        rows=np.array([3,2,1] if kind=='row_order' else [1,2,3],dtype=np.int64)
        values=np.array([1.,float('nan'),0.] if kind=='nonfinite' else [1.,0.,0.])
        if kind=='object_payload':values=np.array([object(),object(),object()])
        arrays=dict(values=values,row_id=rows)
        if kind=='extra_target':arrays['y']=np.zeros(3)
        np.savez(target,**arrays);case['methods'][0]['prediction']['sha256']=sha(target)
    elif kind=='source':(repository/'src/model.py').write_text('# source drift\n')
    elif kind=='source_package':value['source']['package_sha256']='0'*64
    elif kind=='symlink':
        alias=target.with_name('alias.npz');alias.symlink_to(target);case['methods'][0]['prediction']['path']='artifacts/study/alias.npz'
    elif kind=='symlink_parent':
        alias=target.parent.with_name('alias');alias.symlink_to(target.parent,target_is_directory=True)
        case['methods'][0]['prediction']['path']='artifacts/alias/chosen.npz'
    elif kind=='traversal':case['methods'][0]['prediction']['path']='artifacts/study/../study/chosen.npz'
    elif kind=='duplicate_case':value['cases'].append(copy.deepcopy(case))
    elif kind=='duplicate_method':case['methods'].append(copy.deepcopy(case['methods'][0]))
    elif kind=='duplicate_prediction_id':case['methods'][1]['prediction_id']=case['methods'][0]['prediction_id']
    elif kind=='failed_with_prediction':case['methods'][0]['status']='timeout'
    else:value['comparisons']=[dict(decision='passed',estimate=999)]
    if kind not in ('manifest','prediction_bytes'):path,digest=publish(repository,value)
    with pytest.raises(ValueError):run(repository,path,digest)


@pytest.mark.parametrize('kind',['T_role','confirmation_scope','supplied_authority'])
def test_protected_or_claim_promotion_fails_before_any_array_is_opened(kind,repository,monkeypatch):
    api();value,path,_=prepare(repository)
    if kind=='T_role':value['cases'][0]['role']='T'
    elif kind=='confirmation_scope':value['evidence_scope']='confirmation'
    else:value['protected_access_authorized']=True
    path,digest=publish(repository,value)
    def forbidden(*args,**kwargs):raise AssertionError('Target/prediction decoder must remain untouched')
    monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ValueError):run(repository,path,digest)


def test_blocked_without_predictions_does_not_need_a_fictitious_confirmation_directory(repository):
    api();value,path,_=prepare(repository);value.update(status='blocked',cases=[])
    path,digest=publish(repository,value);result=run(repository,path,digest)
    assert result['predictions']==result['planned_prediction_ids']==[]
    assert result['provenance']['raw_predictions_authenticated'] is False
    from treeneuron.analysis.report_v2 import render_report
    assert 'blocked' in render_report(result)


def test_input_drift_during_decode_cannot_return_a_current_artifact_attestation(repository,monkeypatch):
    module=api();value,path,digest=prepare(repository)
    target=repository/'artifacts/study/chosen.npz'
    # Mutate only after the prediction file itself has been checked/decoded.
    arithmetic=module.prediction_arithmetic_v2.audit_prediction_arrays
    def after_decode(**kwargs):
        target.write_bytes(target.read_bytes()+b'changed after decode')
        return arithmetic(**kwargs)
    monkeypatch.setattr(module.prediction_arithmetic_v2,'audit_prediction_arrays',after_decode)
    with pytest.raises(ValueError):run(repository,path,digest)


def test_development_report_cli_rebuilds_raw_metrics_and_refuses_to_overwrite(repository):
    api();value,path,digest=prepare(repository);path=path.rename(path.with_name('development-manifest.json'))
    from treeneuron.research.cli import main
    output=repository/'artifacts/study/report.md'
    argv=['report','--study',str(path.parent),'--output',str(output),
        '--repository-root',str(repository),'--development-manifest-sha256',digest]
    assert main(argv)==0
    before=output.read_bytes();assert b'0.25' in before and b'No protected confirmation' in before
    assert main(argv)==2 and output.read_bytes()==before


@pytest.mark.parametrize('kind',['wrong_hash','protected_role','execution_output'])
def test_development_report_cli_cannot_publish_unqualified_or_source_mutating_output(kind,repository):
    api();value,path,digest=prepare(repository)
    if kind=='protected_role':value['cases'][0]['role']='T';path,digest=publish(repository,value)
    path=path.rename(path.with_name('development-manifest.json'))
    from treeneuron.research.cli import main
    output=repository/('src/report.py' if kind=='execution_output' else 'artifacts/study/result.md')
    assert main(['report','--study',str(path.parent),'--output',str(output),'--repository-root',str(repository),
        '--development-manifest-sha256','0'*64 if kind=='wrong_hash' else digest])==2
    assert not output.exists()
