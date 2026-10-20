"""Fixed B0 diagnostic; producer/core and this operator are separately pinned."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
from treeneuron.config import canonical_hash
from treeneuron.data.types import array_hash
from treeneuron.research.method_worker import _ROOT,scientific_source_sha256
from treeneuron.research.ledger import write_new_json
from treeneuron.research.workqueue import make_queue_plan,run_queue,_inventory
from treeneuron.research.opportunity_data import prepare_data_artifact,load_fitting_data,load_evaluation_data,_metadata,_file
from treeneuron.research.search_supports_v2 import family_search_spec,_config
from treeneuron.research.seeds import seed64,seed32
from treeneuron.research.qualify import qualification_locks_sha256
from treeneuron.research.immutable_inputs import stage_immutable_file
from treeneuron.research.native_fit_v2 import run_native_fit
from treeneuron.research.native_query_v2 import predict_native_query
from treeneuron.research.native_verification_v2 import load_native_implementation_verification
from treeneuron.research.trial_bridge_v2 import EngineeringTrialBridge
from treeneuron.research.opportunity_execution import live_resources


def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save_arrays(path,**arrays):
 with Path(path).open('xb') as stream:
  np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())

def configs():
 schema=dict(columns=[f'x{i}' for i in range(16)],categorical=[])
 p03=family_search_spec('P03',schema)
 entries=[dict(config_id='O0_h64_historical',group='one_layer',config=p03['anchors'][1]['config'])]
 for width in (64,128,256,512):
  entries.append(dict(config_id=f'O{len(entries)}_h{width}_native_optimizer',group='one_layer',
   config=_config('P03',schema,dict(hidden_dims=[width],activation='ReLU',dropout=.1,lr=.001,weight_decay=.01,batch_size=128),mode='search')))
 for name,family,variant in [('MLP_D','P03',None),('ResNet_D','P04',None),('RealMLP_TD','P06',None),('TabM_piecewise','P07','piecewise')]:
  anchors=family_search_spec(family,schema)['anchors']
  cfg=next(a['config'] for a in anchors if variant is None or a['config'].get('variant')==variant)
  entries.append(dict(config_id='B'+str(len(entries)-5)+'_'+name,group='backbone',config=cfg))
 assert len(entries)==9 and sum(e['group']=='one_layer' for e in entries)==5
 assert len({canonical_hash(e['config']) for e in entries})==9
 return entries


def create_plan(root,repo):
 endpoint=read(root/'preparation/software/endpoint.json')
 for proof in endpoint['proofs']:
  load_native_implementation_verification(proof['reference'],expected_profile=proof['profile'])
 if {p['profile'] for p in endpoint['proofs']}!={'ordinary_neural','realmlp','tabm'}:raise ValueError('Missing fixed native software profiles')
 old=repo/'artifacts/research-v2/M3-joint-53bd029/queue/plan.json'
 base=read(old)['identity']['study_config']['base_experiment']
 if (base['model']['d'],base['data']['n_fit'],base['data']['n_growth'],base['data']['n_validation'],base['data']['noise_sigma'])!=(16,2306,577,577,.1):raise ValueError('Primary defaults differ')
 exclusion=repo/'protocol/research-v2/exclusions/current-known-exclusions.json';index=read(exclusion)
 seed_ref=index['teacher_seed_registry'];id_ref=index['teacher_ID_candidates']
 if sha(repo/seed_ref['path'])!=seed_ref['sha256'] or sha(repo/id_ref['path'])!=id_ref['sha256']:raise ValueError('Known development exclusions changed')
 known=set(read(repo/seed_ref['path'])['teacher_seeds']);used64=set(known);used32={s for s in known if 0<=s<2**32}
 entries=configs();specs=[];jobs=[];fit_seeds=[]
 for teacher in range(4):
  spec=dict(population='primary',teacher_index=teacher,evaluation_rows=65536)
  for role in ('teacher','reference','data','evaluation'):
   seed=seed64(2026101801,'B0','primary','teacher',str(teacher),role)
   if seed in used64:raise ValueError('Independent teacher/reference/observation seed collision; no data generated')
   used64.add(seed);spec[role+'_seed']=seed
  specs.append(spec)
  jobs.append(dict(job_id=f'data:{teacher}',kind='data',depends_on=[],payload=spec))
  for repeat in range(2):
   seed,counter=seed32(2026101801,'B0','primary','teacher',str(teacher),'repeat',str(repeat),'fit',used=used32)
   fit_seeds.append(dict(teacher_index=teacher,repetition=repeat,fit_seed=seed,collision_counter=counter))
 for pair in fit_seeds:
  for order,entry in enumerate(entries):
   jobs.append(dict(job_id=f'fit:{pair["teacher_index"]}:{pair["repetition"]}:{entry["config_id"]}',kind='native_fit',
    depends_on=[f'data:{pair["teacher_index"]}'],payload=dict(**pair,**entry,order=order)))
 assert len(jobs)==76 and sum(j['kind']=='native_fit' for j in jobs)==72
 decision=repo/'docs/research/v2/B0-native-primary-backbones.md'
 identity=dict(schema='research/v2-B0-native-primary-diagnostic',source_sha256=scientific_source_sha256(),
  producer_root=str(_ROOT),source_commit='80afc0d',locks_sha256=qualification_locks_sha256(),
  operator_path=str(Path(__file__).resolve()),operator_sha256=sha(__file__),
  decision=dict(path=str(decision),sha256=sha(decision)),base_experiment=base,configs=entries,teacher_specs=specs,fit_seeds=fit_seeds,
  software_endpoint=dict(path=str(root/'preparation/software/endpoint.json'),sha256=sha(root/'preparation/software/endpoint.json')),
  workload_forecast=dict(path=str(root/'preparation/workload-forecast.json'),sha256=sha(root/'preparation/workload-forecast.json')),
  known_seeds=dict(path=str(repo/seed_ref['path']),sha256=seed_ref['sha256']),known_ID_candidates=dict(path=str(repo/id_ref['path']),sha256=id_ref['sha256']),
  rss_limit_bytes=4*1024**3,memory_proxy_floor_bytes=8*1024**3,disk_floor_bytes=4*1024**3,
  protected_access_authorized=False,quality_eligible=False,study_scope='Fixed native baseline development, not M3 extension, HPO, final mandatory-suite qualification or confirmation')
 plan=make_queue_plan(identity,jobs);write_new_json(root/'study-plan.json',plan)
 return plan


def check(identity):
 if identity['producer_root']!=str(_ROOT) or identity['source_sha256']!=scientific_source_sha256() or identity['locks_sha256']!=qualification_locks_sha256():raise ValueError('Held B0 code/locks changed')
 if sha(identity['operator_path'])!=identity['operator_sha256']:raise ValueError('Frozen B0 operator changed')
 for key in ('decision','software_endpoint','workload_forecast','known_seeds','known_ID_candidates'):
  ref=identity[key]
  if sha(ref['path'])!=ref['sha256']:raise ValueError('Frozen B0 prerequisite changed: '+key)


def overlays(family):
 return (_ROOT/'environment/research-v2-realmlp-site',) if family=='P06' else (_ROOT/'environment/research-v2-tabm-site',) if family=='P07' else ()


def metrics(pred,rows,variance):
 errors=np.asarray(pred,dtype=np.float64)-rows.y
 if errors.shape!=rows.y.shape or not np.isfinite(errors).all():raise FloatingPointError('Invalid complete native diagnostic predictions')
 mse=math.fsum(float(e)*float(e) for e in errors)/len(errors)
 result=dict(MSE=mse,NMSE=mse/variance,RMSE=math.sqrt(mse),MAE=math.fsum(abs(float(e)) for e in errors)/len(errors))
 if rows.noise_free_target is not None:
  error=np.asarray(pred,dtype=np.float64)-rows.noise_free_target
  result['prediction_to_noiseless_NMSE']=math.fsum(float(e)*float(e) for e in error)/len(error)/variance
 return result


class Worker:
 def __init__(self,plan):self.plan=plan;self.identity=plan['identity']
 def __call__(self,job,directory,dependencies,attempt):
  if job['kind']=='data':return self.data(job,Path(directory))
  if job['kind']=='native_fit':return self.fit(job,Path(directory),dependencies,attempt)
  raise ValueError('Unknown B0 planned job kind')
 def data(self,job,directory):
  artifact=directory/'artifact';record=prepare_data_artifact(self.identity['base_experiment'],job['payload'],artifact)
  if record['teacher_id'] in read(self.identity['known_ID_candidates']['path'])['teacher_ids']:raise ValueError('Known teacher ID collision; retain original failed cell')
  fitting=load_fitting_data(artifact);_,metadata,_=_metadata(artifact)
  with np.load(_file(artifact,metadata,'validation-features.npz'),allow_pickle=False) as saved:
   V={k:saved['V_'+k].copy() for k in ('X','row_id','group_id')}
  with np.load(_file(artifact,metadata,'evaluator.npz'),allow_pickle=False) as saved:
   E={k:saved['E_'+k].copy() for k in ('X','row_id','group_id')}
  for key,value in E.items():
   if array_hash(value)!=metadata['headers']['E']['arrays'][key]['sha256']:raise ValueError('Exploratory feature export differs')
  inputs=directory/'native-inputs';inputs.mkdir()
  for role in ('F','G'):
   rows=getattr(fitting,role);save_arrays(inputs/(role+'.npz'),**{k:getattr(rows,k) for k in ('X','y','row_id','group_id')})
  save_arrays(inputs/'V_features.npz',**V)
  # E first: all65536 E rows receive exactly64 complete1024-row chunks.
  spans={};pieces={k:[] for k in ('X','row_id','group_id')};cursor=0
  for role,arrays in [('E',E),('F',{k:getattr(fitting.F,k) for k in pieces}),('G',{k:getattr(fitting.G,k) for k in pieces})]:
   n=len(arrays['row_id']);spans[role]=[cursor,cursor+n];cursor+=n
   for key in pieces:pieces[key].append(arrays[key])
  save_arrays(inputs/'E-F-G-features.npz',**{key:np.concatenate(values) for key,values in pieces.items()})
  return dict(status='completed_eligible',artifact_path=str(artifact),native_inputs=str(inputs),teacher_id=record['teacher_id'],
   spans=spans,exported_feature_rows=cursor,E_labels_decoded_by_export=False,T_created=False)
 def fit(self,job,directory,dependencies,attempt):
  data=next(iter(dependencies.values()))['value'];p=job['payload'];family=p['config']['family']
  input_root=directory/'input';input_root.mkdir();files={}
  for role in ('F','G','V_features'):
   original=Path(data['native_inputs'])/(role+'.npz');target=input_root/original.name
   stage_immutable_file(original,target,expected_sha256=sha(original));files[role]=dict(path=target.name,sha256=sha(original))
  data_meta=read(Path(data['artifact_path'])/'result.json')
  request=dict(schema='research/v2-method-worker-request',purpose='engineering_qualification',job_id='B0/'+job['job_id'],
   family_id=family,fit_seed=p['fit_seed'],scientific_source_sha256=self.identity['source_sha256'],config=p['config'],budget=dict(native_stopping=True),
   files=files,observation_identity=dict(dataset='B0_primary_'+data['teacher_id'],split_kind='grouped',split_sha256=data_meta['manifest_sha256']))
  path=input_root/'request.json';write_new_json(path,request);request_sha=sha(path)
  old=directory.parent/'attempt-0';reuse=None;recovery=None
  if attempt==1 and (old/'native-call.json').is_file():
   previous=read(old/'native-call.json')
   if previous['request_sha256']!=request_sha:raise ValueError('Recovery changed native request')
   if previous['status']=='completed_ineligible':reuse=previous
   elif previous['status'] in ('resource_exhausted','interrupted_recoverable'):recovery=previous['recovery_reference']
  started=time.perf_counter_ns()
  native=reuse or run_native_fit(path,expected_request_sha256=request_sha,output=directory/'native',interpreter=sys.executable,
   extra_pythonpath=overlays(family),rss_limit_bytes=self.identity['rss_limit_bytes'],recovery_reference=recovery)
  native_span=time.perf_counter_ns()-started;write_new_json(directory/'native-call.json',native)
  if native['status']!='completed_ineligible':
   if native['status']=='resource_exhausted':raise MemoryError('Owned native fit RSS stop; retain and recover same request/seed once')
   if native['status']=='interrupted_recoverable':raise KeyboardInterrupt('Owned native fit interrupted; retain same queue')
   return dict(status=native['status'],native=native,quality_eligible=False)
  native_root=Path(native['recovery_reference']['root'])
  if canonical_hash(_inventory(native_root))!=native['recovery_reference']['inventory_sha256']:raise ValueError('Closed native fit changed')
  terminal=EngineeringTrialBridge._worker_terminal(native_root/'worker',request,request_sha)
  if terminal['snapshot_reload_exact'] is not True or terminal['budget_resolution']['policy']!='native_stopping':raise ValueError('Incomplete native stopping/state')
  query_root=directory/'query-input';query_root.mkdir();snapshot=query_root/'snapshot';snapshot.mkdir()
  for original in (native_root/'worker/snapshot').rglob('*'):
   target=snapshot/original.relative_to(native_root/'worker/snapshot')
   if original.is_dir():target.mkdir(exist_ok=True)
   else:stage_immutable_file(original,target,expected_sha256=sha(original))
  original=Path(data['native_inputs'])/'E-F-G-features.npz'
  stage_immutable_file(original,query_root/'query.npz',expected_sha256=sha(original))
  query_request=dict(schema='research/v2-native-query-request',purpose='engineering_qualification',family_id=family,
   scientific_source_sha256=self.identity['source_sha256'],feature_schema=p['config']['feature_schema'],
   snapshot=dict(path='snapshot',metadata_sha256=sha(snapshot/'manifest.json')),query=dict(path='query.npz',sha256=sha(original)),chunk_rows=1024)
  query_path=query_root/'request.json';write_new_json(query_path,query_request);started=time.perf_counter_ns()
  prediction=predict_native_query(query_path,expected_request_sha256=sha(query_path),output=directory/'query',interpreter=sys.executable,
   extra_pythonpath=overlays(family),rss_limit_bytes=self.identity['rss_limit_bytes'])
  query_span=time.perf_counter_ns()-started
  if prediction['status']=='resource_exhausted':raise MemoryError('Owned inference RSS stop; retain closed fit for one identical inference retry')
  if prediction['status']!='completed_ineligible':return dict(status=prediction['status'],prediction=prediction,quality_eligible=False)
  write_new_json(directory/'coordinator-label-access.json',dict(roles=['F_labels','G_labels','V_labels','E_labels'],
   phase='B0_development',teacher_id=data['teacher_id'],fit_worker_labels=['F','G'],T_accessed=False))
  development,E=load_evaluation_data(data['artifact_path']);start=time.perf_counter_ns()
  mean=math.fsum(float(y) for y in development.F.y)/len(development.F.y)
  variance=math.fsum((float(y)-mean)**2 for y in development.F.y)/len(development.F.y)
  values={}
  with np.load(native_root/'worker/predictions.npz',allow_pickle=False) as saved:
   if not np.array_equal(saved['row_id'],development.V.row_id) or not np.array_equal(saved['group_id'],development.V.group_id):raise ValueError('Native V row identities differ')
   values['V']=metrics(saved['prediction'],development.V,variance)
  with np.load(directory/'query/worker/predictions.npz',allow_pickle=False) as saved:
   for role,rows in [('E',E),('F',development.F),('G',development.G)]:
    begin,end=data['spans'][role]
    if not np.array_equal(saved['row_id'][begin:end],rows.row_id) or not np.array_equal(saved['group_id'][begin:end],rows.group_id):raise ValueError('Complete query row identities differ')
    values[role]=metrics(saved['prediction'][begin:end],rows,variance)
  if canonical_hash(_inventory(native_root))!=native['recovery_reference']['inventory_sha256']:raise ValueError('Native generation changed during evaluation')
  result=dict(status='completed_ineligible',teacher_index=p['teacher_index'],repetition=p['repetition'],teacher_id=data['teacher_id'],
   config_id=p['config_id'],group=p['group'],family=family,fit_seed=p['fit_seed'],configuration=p['config'],metrics=values,F_variance=variance,
   native_reference=native['recovery_reference'],cached_completed_fit=reuse is not None,native_call_span_ns=native_span,
   native_original_callback_ns=native['callback_elapsed_ns'],query_call_span_ns=query_span,coordinator_arithmetic_ns=time.perf_counter_ns()-start,
   full_native_stopping=True,quality_eligible=False,production_selection_authorized=False,T_accessed=False,
   scope='Actual native baseline development execution and complete V/E arithmetic; formal practical-suite qualification is separate')
  write_new_json(directory/'metrics.json',result)
  print(json.dumps({key:result[key] for key in ('teacher_index','repetition','config_id','status')},sort_keys=True),flush=True)
  return result


def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);parser.add_argument('--repo',required=True,type=Path)
 parser.add_argument('--study',required=True,type=Path);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 root=args.study.resolve();repo=args.repo.resolve()
 if args.action=='plan':
  plan=create_plan(root,repo);print(json.dumps(dict(jobs=len(plan['attempts']),native_fits=72,data_jobs=4,plan_sha256=plan['plan_hash'])));return
 plan=read(root/'study-plan.json');check(plan['identity'])
 admission_log=root/'resource-admission.jsonl'
 def admission(job,events):
  observed=live_resources(root);identity=plan['identity'];decision=dict(admitted=observed['available_memory_bytes']>=identity['memory_proxy_floor_bytes'] and observed['free_disk_bytes']>=identity['disk_floor_bytes'],observed=observed,
   memory_proxy_not_guaranteed=True,one_new_worker=True,job_id=job['job_id'])
  with admission_log.open('a') as stream:stream.write(json.dumps(decision,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
  return decision
 result=run_queue(plan,root/'queue',Worker(plan),check_identity=check,resume=args.resume,before_start=admission)
 endpoint=root/f'endpoint-{len(list(root.glob("endpoint-*.json")))}.json';write_new_json(endpoint,result)
 print(json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
