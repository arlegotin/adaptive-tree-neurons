import fcntl,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from treeneuron.research.deployment_worker_v2 import _source,_THREADS
from treeneuron.research.ledger import write_new_json
from treeneuron.research.validation_scoring_v2 import score_primary_validation_prediction
from treeneuron.research.transfer import TransferStore
root=Path(__file__).resolve().parent;project=Path.cwd();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected='acfe5bd73e1866ceba6d8c3065a4bf7a17a1c931332a1f4f52a3dff223eef250'
assert sha(root/'plan.json')==expected
plan=json.loads((root/'plan.json').read_text());source=_source();assert source==plan['source_sha256']
write_new_json(root/'invocation.json',dict(argv=list(sys.argv),cwd=str(project),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=source,operator_sha256=sha(__file__),plan_sha256=expected,environment_overrides={k:os.environ.get(k) for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE')}))
started=time.perf_counter_ns();error=None;rows=[]
try:
    for ref in plan['prerequisites']:assert sha(ref['path'])==ref['sha256']
    for case in plan['fixed_cases']:
        if case['status']!='native_completed_and_audited_ready_for_primary_V':
            rows.append(dict(cell_id=case['cell_id'],dataset=case['dataset'],status='pending_native_completion',V_score=None));continue
        native=Path(case['native_root'])
        with (native/'worker-lease.json').open('rb') as lease:
            fcntl.flock(lease.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
            for ref in case['evidence']:assert sha(ref['path'])==ref['sha256']
            terminal=json.loads((native/'attempt-000/worker/terminal.json').read_text())
            assert terminal['scientific_source_sha256']==case['source_sha256'] and terminal['fit_seed']==17
            assert terminal['status']=='completed_ineligible' and terminal['accessed_roles']==['F_labels','G_labels','V_features']
            assert terminal.get('V_labels_read',False) is False
            for ref in terminal['files']:
                p=native/'attempt-000/worker'/ref['path'];assert sha(p)==ref['sha256'] and p.stat().st_size==ref['bytes']
            score=score_primary_validation_prediction(root=case['prepared_root'],outer_fold=case['outer_fold'],expected_manifest_sha256=case['expected_manifest_sha256'],prediction_path=case['prediction_path'],expected_prediction_sha256=case['expected_prediction_sha256'])
            store=TransferStore(case['prepared_root']);view=store.development(case['outer_fold']);V=store.coordinator_validation(case['outer_fold'])
            with np.load(case['prediction_path'],allow_pickle=False) as payload:
                pred=payload['prediction'].astype('float64');assert np.array_equal(payload['row_id'],V.row_id) and np.array_equal(payload['group_id'],V.group_id)
            F=[float(v) for v in view.F.y];anchor=F[0];center=math.fsum(v-anchor for v in F)/len(F);variance=math.fsum((v-anchor-center)**2 for v in F)/len(F)
            differences=[float(p)-float(y) for p,y in zip(pred,V.y)];square=[v*v for v in differences];absolute=[abs(v) for v in differences]
            groups={}
            for group,s,a in zip(V.group_id,square,absolute):
                pair=groups.setdefault(int(group),([],[]));pair[0].append(s);pair[1].append(a)
            group_mse=math.fsum(math.fsum(v[0])/len(v[0]) for v in groups.values())/len(groups)
            group_mae=math.fsum(math.fsum(v[1])/len(v[1]) for v in groups.values())/len(groups)
            row_mse=math.fsum(square)/len(square)
            metrics=dict(normalized_risk=group_mse/variance,primary_MSE=group_mse,primary_RMSE=math.sqrt(group_mse),primary_MAE=group_mae,row_MSE=row_mse,row_RMSE=math.sqrt(row_mse),row_MAE=math.fsum(absolute)/len(absolute))
            discrepancy=max(abs(metrics[k]-score['metrics'][k]) for k in metrics);variance_discrepancy=abs(variance-score['F_target_variance'])
            assert discrepancy<=1e-12 and variance_discrepancy<=1e-12
            for ref in case['evidence']+score['sources']:assert sha(ref['path'])==ref['sha256']
            rows.append(dict(cell_id=case['cell_id'],dataset=case['dataset'],status='completed_native_development_score',V_score=score['score'],metrics=metrics,validation=score,independent_metric_comparisons=len(metrics),independent_max_absolute_discrepancy=discrepancy,F_variance_discrepancy=variance_discrepancy,V_groups=len(groups),V_rows=len(V.y),quality_eligible=False))
    assert sha(root/'plan.json')==expected and _source()==source
    write_new_json(root/'scores.json',dict(schema='research/v2-native-development-appendix',plan_sha256=expected,rows=rows,original_survey_unchanged=True,quality_eligible=False,T_accessed=False,new_fits=0,scope='Fixed newly completed native anchor development V; no tuned-family or protected comparison'))
except BaseException as caught:
    error=dict(type=type(caught).__name__,reason=str(caught),traceback=traceback.format_exc());raise
finally:
    write_new_json(root/'result.json',dict(status='completed' if error is None else 'failed',error=error,elapsed_ns=time.perf_counter_ns()-started,scored=sum(r['status']=='completed_native_development_score' for r in rows),pending=sum(r['status']=='pending_native_completion' for r in rows),new_fits=0,T_accessed=False))
print(json.dumps([dict(cell_id=r['cell_id'],status=r['status'],V_score=r['V_score']) for r in rows]))
