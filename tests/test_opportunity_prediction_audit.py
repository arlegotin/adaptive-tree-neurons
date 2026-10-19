"""M2 raw arithmetic is reconstructed without model/producer metric loading."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def api():
    from treeneuron.research import opportunity_prediction_audit
    return opportunity_prediction_audit


def digest(value):
    from treeneuron.data.types import array_hash
    return array_hash(value)


def fixture(tmp_path,mutation=None,missing=False,prefix=False,kind='m2_opportunity'):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    request=dict(population='primary',teacher_index=0)
    jobs=[dict(job_id='data:primary/teacher-0',kind='data',payload=request,depends_on=[]),
        dict(job_id='procedure:toy',kind='fit',payload={'matrix':'M2a'},depends_on=['data:primary/teacher-0']),
        dict(job_id='evaluate:toy',kind='evaluate',payload={'matrix':'M2a'},depends_on=['data:primary/teacher-0','procedure:toy'])]
    if missing:jobs.append(dict(job_id='evaluate:missing',kind='evaluate',payload={},depends_on=['data:primary/teacher-0','procedure:toy']))
    plan=make_queue_plan(dict(source_sha256='a'*64,study_config={'kind':kind}),jobs)
    root=tmp_path/'queue'
    arrays={}
    for index,(role,y) in enumerate([('F',[0.,2.]),('G',[1.]),('V',[1.,3.]),('E',[2.,4.,6.])]):
        n=len(y);arrays.update({role+'_X':np.arange(n,dtype=np.float64)[:,None],role+'_y':np.array(y),
            role+'_row_id':np.arange(index*10,index*10+n,dtype=np.int64),
            role+'_group_id':np.arange(index*10,index*10+n,dtype=np.int64),role+'_noise_free_target':np.array(y)-.25})
    def handler(job,directory,deps,*_):
        if job['kind']=='data':
            artifact=directory/'artifact';artifact.mkdir()
            paths={}
            for name,keys in [('fitting.npz',[r+'_'+k for r in 'FG' for k in ('X','y','row_id','group_id')]),
                ('validation-features.npz',['V_X','V_row_id','V_group_id']),
                ('evaluator.npz',['V_y',*['E_'+k for k in ('X','y','row_id','group_id')],*[r+'_noise_free_target' for r in 'FGVE']])]:
                np.savez(artifact/name,**{k:arrays[k] for k in keys})
                paths[name]=hashlib.sha256((artifact/name).read_bytes()).hexdigest()
            headers={r:dict(stream_id='toy-'+r,feature_names=['x'],arrays={k:dict(dtype=str(arrays[r+'_'+k].dtype),shape=list(arrays[r+'_'+k].shape),
                sha256=digest(arrays[r+'_'+k])) for k in ('X','y','row_id','group_id','noise_free_target')}) for r in 'FGVE'}
            from treeneuron.config import canonical_hash
            split=dict(teacher_id='toy',phase='exploratory',feature_names=['x'],
                row_id=[int(i) for r in 'FGV' for i in arrays[r+'_row_id']],
                group_id=[int(i) for r in 'FGV' for i in arrays[r+'_group_id']],
                assignments=[r for r in 'FGV' for _ in arrays[r+'_row_id']],
                hashes=[[r+'.'+k,digest(arrays[r+'_'+k])] for r in 'FGV' for k in ('X','y')])
            for name,value in [('request.json',{'spec':request}),('teacher.json',{'teacher_id':'toy'}),('split.json',split)]:
                (artifact/name).write_text(json.dumps(value));paths[name]=hashlib.sha256((artifact/name).read_bytes()).hexdigest()
            if mutation=='data_header':headers['F']['arrays']['y']['sha256']='0'*64
            metadata=dict(schema='research/v2-data-artifact',status='completed_eligible',source_sha256='a'*64,
                request=request,teacher_id='toy',headers=headers,files=paths,T_created=False,manifest_sha256=canonical_hash(split))
            (artifact/'result.json').write_text(json.dumps(metadata))
            return dict(status='completed_eligible',artifact_path=str(artifact),teacher_id='toy')
        if job['kind']=='fit':
            pointer=directory/'placeholder-model.json';pointer.write_text('{"artificial":true}')
            report={'models':{'selected':dict(path=str(pointer),sha256=hashlib.sha256(pointer.read_bytes()).hexdigest())}}
            if prefix:report['prefixes']={'records':[{'prefix':300}]}
            return dict(status='completed_eligible',artifact_path=str(directory),report=report)
        values={k:v for k,v in arrays.items() if k.startswith(('V_','E_')) and not k.endswith(('X','noise_free_target'))}
        values.update(selected_V=np.array([1.,2.]),selected_E=np.array([2.,3.,4.]))
        if mutation=='rows':values['V_row_id']=values['V_row_id'][::-1]
        if mutation=='targets':values['E_y']=values['E_y']+.1
        if mutation=='groups':values['V_group_id']=values['V_group_id']+1
        if mutation=='missing_vector':del values['selected_E']
        if mutation=='extra_vector':values['unreported_V']=values['selected_V']
        metrics={'selected':{'V':dict(n=2,mse=.5,normalized_mse=.5,rmse=math.sqrt(.5),mae=.5,
            absolute_error_p95=.95,max_absolute_error=1.,noise_free_mse=.3125),
            'E':dict(n=3,mse=5/3,normalized_mse=5/3,rmse=math.sqrt(5/3),mae=1.,
            absolute_error_p95=1.9,max_absolute_error=2.,noise_free_mse=(.0625+.5625+3.0625)/3)}}
        for value in metrics['selected'].values():value.update(normalized_mse_reason=None,group_macro_mse=None,worst_group_mse=None,group_metrics=None)
        if prefix:
            values.update(prefix_300_V=values['selected_V'],prefix_300_E=values['selected_E'])
            metrics['prefix_300']=metrics['selected']
        if mutation=='metric':metrics['selected']['V']['mse']=.4
        if mutation=='missing_model':metrics['other']=metrics.pop('selected')
        path=directory/'predictions.npz';np.savez(path,**values)
        return dict(status='completed_eligible',metrics=metrics,fit_target_variance=1.1 if mutation=='variance' else 1.,
            predictions_path=str(path),T_accessed=False,E_exploratory=True)
    run_queue(plan,root,handler,check_identity=lambda _:None,
        admitted_job_ids=[j['job_id'] for j in jobs if j['job_id']!='evaluate:missing'])
    last=json.loads((root/'events.jsonl').read_text().splitlines()[-1])['event_hash']
    return root,plan,last


def test_reconstructs_every_vector_and_reports_its_actual_evidence_level(tmp_path):
    root,plan,last=fixture(tmp_path)
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    out=api().audit_opportunity_predictions(root,expected_plan_hash=plan['plan_hash'],expected_ledger_hash=last)
    assert out['status']=='completed_arithmetic' and out['audited_prediction_vectors']==2
    assert out['scalar_metric_comparisons']==24 and out['maximum_absolute_metric_difference']<1e-14
    assert out['rows'][0]['metrics']['selected']['E']['mse']==pytest.approx(5/3)
    assert out['producer_source_sha256']=='a'*64
    assert out['model_predictions_reexecuted'] is False and out['checkpoint_tensor_chain_authenticated'] is False
    assert out['Gate_A_decided'] is False and out['T_accessed'] is False
    assert before=={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mutation',['rows','targets','groups','missing_vector','extra_vector','data_header','metric','variance','missing_model'])
def test_authenticated_but_semantically_wrong_producer_outputs_cannot_pass(tmp_path,mutation):
    root,plan,last=fixture(tmp_path,mutation=mutation)
    with pytest.raises(ValueError):
        api().audit_opportunity_predictions(root,expected_plan_hash=plan['plan_hash'],expected_ledger_hash=last)


def test_missing_planned_evaluation_is_retained_and_blocks_complete_arithmetic_inventory(tmp_path):
    root,plan,last=fixture(tmp_path,missing=True)
    out=api().audit_opportunity_predictions(root,expected_plan_hash=plan['plan_hash'],expected_ledger_hash=last)
    assert out['status']=='incomplete_or_failed' and out['planned_evaluation_jobs']==2
    assert out['rows'][1]['job_id']=='evaluate:missing' and out['rows'][1]['status']=='never_started'
    assert 'metrics' not in out['rows'][1]


def test_external_plan_and_ledger_endpoints_are_required(tmp_path):
    root,plan,last=fixture(tmp_path)
    for external_plan,external_ledger in [('0'*64,last),(plan['plan_hash'],'0'*64)]:
        with pytest.raises(ValueError):
            api().audit_opportunity_predictions(root,expected_plan_hash=external_plan,expected_ledger_hash=external_ledger)


def test_active_queue_is_rejected_before_any_predictions_are_opened(tmp_path,monkeypatch):
    from treeneuron.research.ledger import StudyLease
    root,plan,last=fixture(tmp_path)
    def fail(*a,**k):raise AssertionError('Active prediction file opened')
    monkeypatch.setattr(np,'load',fail)
    with StudyLease(root):
        with pytest.raises(PermissionError,match='active'):
            api().audit_opportunity_predictions(root,expected_plan_hash=plan['plan_hash'],expected_ledger_hash=last)


def test_metric_reference_divides_before_sum_and_preserves_undefined_normalization():
    result=api().prediction_metrics(np.zeros(2),np.full(2,1e154),0.,np.zeros(2))
    assert math.isfinite(result['mse']) and result['mse']==1e308
    assert result['normalized_mse'] is None and result['noise_free_mse']==1e308


def test_nested_horizon_predictions_are_audited_as_well_as_the_selected_model(tmp_path):
    root,plan,last=fixture(tmp_path,prefix=True)
    out=api().audit_opportunity_predictions(root,expected_plan_hash=plan['plan_hash'],expected_ledger_hash=last)
    assert out['audited_prediction_vectors']==4 and out['scalar_metric_comparisons']==48
    assert set(out['rows'][0]['metrics'])=={'selected','prefix_300'}


def test_auditor_import_does_not_load_torch_sklearn_or_producer_model_loaders():
    root=Path(__file__).resolve().parents[1]
    env=os.environ|dict(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
    program='import sys; import treeneuron.research.opportunity_prediction_audit; assert not set(("torch","sklearn","treeneuron.research.opportunity_workers","treeneuron.research.opportunity_data","treeneuron.evaluation.metrics")) & set(sys.modules)'
    child=subprocess.run([sys.executable,'-B','-c',program],cwd=root,env=env,capture_output=True,text=True,timeout=15)
    assert child.returncode==0,child.stderr
