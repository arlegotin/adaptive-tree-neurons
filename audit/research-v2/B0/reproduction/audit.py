"""Independent B0 raw-array arithmetic; never fit, infer or alter a producer."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
from treeneuron.config import canonical_hash
from treeneuron.research.workqueue import authenticated_queue,_inventory
from treeneuron.research.opportunity_data import load_evaluation_data
from treeneuron.research.method_worker import scientific_source_sha256
from treeneuron.research.ledger import write_new_json


def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def load(path):return json.loads(Path(path).read_text())

def raw_metrics(prediction,observations,variance):
 values=np.array(prediction,dtype=np.float64,copy=True)
 if values.shape!=observations.y.shape or not np.isfinite(values).all():raise ValueError('Invalid raw prediction vector')
 error=values-observations.y
 mse=float(np.mean(np.square(error),dtype=np.float64))
 result=dict(MSE=mse,NMSE=mse/variance,RMSE=float(np.sqrt(mse)),MAE=float(np.mean(np.abs(error),dtype=np.float64)))
 if observations.noise_free_target is not None:result['prediction_to_noiseless_NMSE']=float(np.mean(np.square(values-observations.noise_free_target),dtype=np.float64))/variance
 return result


def pick(rows):
 best=min(r['V_NMSE'] for r in rows);tolerance=1e-12*max(1.,abs(best))
 return min((r for r in rows if r['V_NMSE']<=best+tolerance),key=lambda r:r['order'])


def analyze(rows,configs):
 if len(rows)!=72 or len({(r['teacher'],r['repeat'],r['config_id']) for r in rows})!=72:raise ValueError('Incomplete/duplicate72-case matrix')
 winners=[];teachers=[]
 for teacher in range(4):
  selected={}
  for repeat in range(2):
   group=[r for r in rows if r['teacher']==teacher and r['repeat']==repeat]
   if {r['config_id'] for r in group}!={c['config_id'] for c in configs}:raise ValueError('Missing planned native configuration')
   for name in ('one_layer','backbone'):
    winner=pick([r for r in group if r['group']==name]);selected[name,repeat]=winner
    winners.append(dict(teacher=teacher,repeat=repeat,group=name,config_id=winner['config_id'],V_NMSE=winner['V_NMSE'],E_NMSE=winner['E_NMSE']))
  ordinary=float(np.mean([selected['one_layer',r]['E_NMSE'] for r in range(2)],dtype=np.float64))
  backbone=float(np.mean([selected['backbone',r]['E_NMSE'] for r in range(2)],dtype=np.float64))
  if min(ordinary,backbone)<=0:raise ValueError('Nonpositive risk cannot supply a finite log comparison')
  teachers.append(dict(teacher=teacher,one_layer_E_NMSE=ordinary,backbone_E_NMSE=backbone,
   log_risk_benefit=math.log(ordinary/backbone),relative_reduction=1-backbone/ordinary,at_least_5_percent=backbone<=.95*ordinary))
 trigger=sum(t['at_least_5_percent'] for t in teachers)>=3
 return dict(config_means=[dict(config_id=c['config_id'],group=c['group'],E_NMSE=float(np.mean([r['E_NMSE'] for r in rows if r['config_id']==c['config_id']],dtype=np.float64)),
   V_NMSE=float(np.mean([r['V_NMSE'] for r in rows if r['config_id']==c['config_id']],dtype=np.float64))) for c in configs],
  V_selected=winners,teacher_effects=teachers,independent_teachers=4,paired_repetitions=2,screen_trigger=trigger,
  qualifying_teachers=sum(t['at_least_5_percent'] for t in teachers),
  decision='Measure the competent standardized ordinary control and residual structure in a separately fixed diagnostic; M6 still not admitted' if trigger else
   'Specified native-family ceiling screen negative/inconclusive; no added B0 configurations/teachers or automatic M6 admission',
  confirmation=False,quality_eligible=False,T_accessed=False)


def selfcheck():
 from types import SimpleNamespace
 target=SimpleNamespace(y=np.array([1.,3.]),noise_free_target=None)
 assert raw_metrics(np.array([2.,2.]),target,1.)==dict(MSE=1.,NMSE=1.,RMSE=1.,MAE=1.)
 # V must pick the worse E point when that is what its own criterion chooses.
 assert pick([dict(order=0,V_NMSE=1.,E_NMSE=10.),dict(order=1,V_NMSE=2.,E_NMSE=1.)])['E_NMSE']==10.
 assert pick([dict(order=1,V_NMSE=1.,E_NMSE=1.),dict(order=0,V_NMSE=1.+5e-13,E_NMSE=2.)])['order']==0
 configs=[dict(config_id=f'O{i}',group='one_layer') for i in range(5)]+[dict(config_id=f'B{i}',group='backbone') for i in range(4)]
 rows=[]
 for teacher in range(4):
  for repeat in range(2):
   for order,c in enumerate(configs):
    ordinary=(1.,9.)[repeat];backbone=1. if teacher<3 else ordinary*1.1
    rows.append(dict(teacher=teacher,repeat=repeat,order=order,**c,V_NMSE=float(order if c['group']=='one_layer' else order-5),E_NMSE=ordinary if c['group']=='one_layer' else backbone))
 result=analyze(rows,configs)
 assert result['screen_trigger'] is True and result['qualifying_teachers']==3
 assert result['teacher_effects'][0]['log_risk_benefit']==math.log(5.)
 try:analyze(rows[:-1],configs)
 except ValueError:pass
 else:raise AssertionError('Incomplete matrix accepted')
 return dict(status='analytic_arithmetic_selection_and_teacher_aggregation_checks_passed',actual_models_or_outcomes_read=False)


def audit(study,output):
 started=time.perf_counter_ns();plan=load(study/'study-plan.json');source=scientific_source_sha256()
 if source!=plan['identity']['source_sha256']:raise ValueError('Audit must use the held producer data reader; no silent source migration')
 if output.exists():raise ValueError('Fresh independent audit output required')
 output.mkdir(parents=True);max_difference=0.;metrics_checked=0;predictions_checked=0;rows=[];raw_references=[]
 with authenticated_queue(study/'queue',expected_plan_hash=plan['plan_hash']) as (saved,ledger,results):
  if saved!=plan or len(results)!=76 or ledger.pending_ids:raise ValueError('Complete76-job queue required; no complete-case substitution')
  if any(results[j['job_id']]['value']['status']!=('completed_eligible' if j['kind']=='data' else 'completed_ineligible') for j in plan['attempts']):raise ValueError('Missing or failed planned method blocks complete B0 screen')
  datasets={}
  for teacher in range(4):
   value=results[f'data:{teacher}']['value'];artifact=Path(value['artifact_path'])
   data,E=load_evaluation_data(artifact);variance=float(np.var(data.F.y,ddof=0,dtype=np.float64))
   if not np.isfinite(variance) or variance<=0:raise ValueError('Invalid independent F variance')
   datasets[teacher]=(value,data,E,variance)
  for job in plan['attempts']:
   if job['kind']!='native_fit':continue
   record=results[job['job_id']]['value'];p=job['payload'];data_value,data,E,variance=datasets[p['teacher_index']]
   if (record['config_id']!=p['config_id'] or record['fit_seed']!=p['fit_seed'] or record['configuration']!=p['config']
       or record['teacher_id']!=data.teacher_id or record['full_native_stopping'] is not True or record['quality_eligible'] is not False or record['T_accessed'] is not False):raise ValueError('Native case identity/recipe/scope mismatch')
   # The marker authenticates the whole closed job, including all retained files.
   result_event=next(e for e in reversed(ledger.events) if e['attempt_id']==job['job_id'] and e['status']!='running')
   directory=Path(result_event['result']['path']).parent
   if 'control' in directory.relative_to(study/'queue').parts:raise ValueError('Failure marker cannot supply completed prediction')
   native=Path(record['native_reference']['root']);request=load(directory/'input/request.json')
   if canonical_hash(_inventory(native))!=record['native_reference']['inventory_sha256']:raise ValueError('Original native container differs')
   process=load(native/'process/process.json');terminal=load(native/'worker/terminal.json')
   if (request['config']!=p['config'] or request['fit_seed']!=p['fit_seed'] or request['scientific_source_sha256']!=source
       or request['budget']!={'native_stopping':True} or set(request['files'])!={'F','G','V_features'}
       or process['timeout_ns'] is not None or process['child_reaped'] is not True or process['returncode']!=0
       or terminal['snapshot_reload_exact'] is not True or terminal['budget_resolution']['policy']!='native_stopping'
       or terminal['accessed_roles']!=['F_labels','G_labels','V_features']):raise ValueError('Native source/roles/stopping/reload provenance differs')
   for role in ('F','G','V_features'):
    original=Path(data_value['native_inputs'])/(role+'.npz')
    if sha(original)!=request['files'][role]['sha256'] or sha(directory/'input'/request['files'][role]['path'])!=sha(original):raise ValueError('Original/staged fitting observations differ')
   query_request=load(directory/'query-input/request.json');query_result=load(directory/'query/result.json');query_terminal=load(directory/'query/worker/terminal.json')
   if (query_request['scientific_source_sha256']!=source or query_request['chunk_rows']!=1024 or query_result['status']!='completed_ineligible'
       or query_result['prediction_rows']!=68419 or query_terminal['fitting_performed'] is not False or query_terminal['labels_read'] is not False
       or query_request['query']['sha256']!=sha(Path(data_value['native_inputs'])/'E-F-G-features.npz')):raise ValueError('Closed feature-only evaluation provenance differs')
   computed={}
   with np.load(native/'worker/predictions.npz',allow_pickle=False) as v:
    if set(v.files)!={'prediction','row_id','group_id'} or not np.array_equal(v['row_id'],data.V.row_id) or not np.array_equal(v['group_id'],data.V.group_id):raise ValueError('V vector alignment differs')
    computed['V']=raw_metrics(v['prediction'],data.V,variance)
   with np.load(directory/'query/worker/predictions.npz',allow_pickle=False) as pred:
    if set(pred.files)!={'prediction','row_id','group_id'}:raise ValueError('Unexpected evaluated prediction fields')
    for role,observation in [('E',E),('F',data.F),('G',data.G)]:
     begin,end=data_value['spans'][role]
     if not np.array_equal(pred['row_id'][begin:end],observation.row_id) or not np.array_equal(pred['group_id'][begin:end],observation.group_id):raise ValueError('E/F/G vector alignment differs')
     computed[role]=raw_metrics(pred['prediction'][begin:end],observation,variance)
   for role,values in computed.items():
    predictions_checked+=1
    if set(values)!=set(record['metrics'][role]):raise ValueError('Metric inventory differs')
    for name,value in values.items():
     expected=record['metrics'][role][name];difference=abs(value-expected);max_difference=max(max_difference,difference);metrics_checked+=1
     if difference>1e-12*max(1.,abs(value),abs(expected)):raise ValueError('Independent numerical metric mismatch')
   if abs(variance-record['F_variance'])>1e-12*max(1.,variance):raise ValueError('F variance differs')
   raw_references.extend([dict(path=str(native/'worker/predictions.npz'),sha256=sha(native/'worker/predictions.npz')),
    dict(path=str(directory/'query/worker/predictions.npz'),sha256=sha(directory/'query/worker/predictions.npz'))])
   rows.append(dict(teacher=p['teacher_index'],repeat=p['repetition'],config_id=p['config_id'],group=p['group'],order=p['order'],
    V_NMSE=computed['V']['NMSE'],E_NMSE=computed['E']['NMSE'],metrics=computed,
    native_and_query_span_ns=record['native_call_span_ns']+record['query_call_span_ns']))
  analysis=analyze(rows,plan['identity']['configs'])
  for ref in raw_references:
   if sha(ref['path'])!=ref['sha256']:raise ValueError('Late raw prediction drift')
  result=dict(schema='research/v2-B0-independent-development-audit',status='complete_raw_arithmetic_checked',plan_hash=plan['plan_hash'],source_sha256=source,
   operator_sha256=sha(__file__),raw_prediction_vectors=predictions_checked,metric_comparisons=metrics_checked,max_abs_metric_difference=max_difference,
   rows=rows,analysis=analysis,raw_references=raw_references,charged_job_ns=ledger.charged_ns,independent_training_reexecution=False,
   independent_new_inference=False,source_receipts_trusted=True,full_native_suite_qualification=False,quality_eligible=False,T_accessed=False,
   scope='Available source-bound raw prediction/observation arithmetic and fixed development screen, not retraining, fresh inference, protected confirmation or mandatory-suite superiority',
   elapsed_ns=time.perf_counter_ns()-started)
  write_new_json(output/'analysis.json',result)
 print(json.dumps(dict(status=result['status'],raw_vectors=predictions_checked,metric_comparisons=metrics_checked,max_abs_difference=max_difference,
  screen_trigger=analysis['screen_trigger'],qualifying_teachers=analysis['qualifying_teachers'])),flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--selfcheck',action='store_true');p.add_argument('--study',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
 if args.selfcheck:print(json.dumps(selfcheck()));return
 if args.study is None or args.output is None:raise ValueError('Exact study and fresh audit output required')
 audit(args.study.resolve(),args.output.resolve())
if __name__=='__main__':main()
