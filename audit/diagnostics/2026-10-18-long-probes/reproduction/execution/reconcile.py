"""Independent saved-state pairing plus reviewed production analysis/twice-render."""
import time,os,json,hashlib,resource,traceback,subprocess,math
from pathlib import Path
START=time.perf_counter_ns()
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.checkpoint import load_checkpoint
from treeneuron.config import canonical_hash
from treeneuron.experiments.paired import _json_tree
from treeneuron.data.types import array_hash
from treeneuron.pilot import probe_diagnostic as production
ROOT=Path.cwd();C=ROOT/'artifacts/long-probes-20261018/cohort';OUT=ROOT/'artifacts/long-probes-20261018/audit';ENV=ROOT/'environment/long-probes-20261018/audit'
report={'status':'running','failures':[],'pairing':[],'maximum_numeric_difference':0.0,'comparison_values':0,'renderings':[]}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save():(OUT/'reconciliation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
def compare(a,b,label):
 if isinstance(a,dict):
  for k,v in a.items():compare(v,b[k],label+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),label
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,label+'/'+str(i))
 elif isinstance(a,(int,float)) and not isinstance(a,bool):
  delta=abs(a-b);report['maximum_numeric_difference']=max(report['maximum_numeric_difference'],delta);report['comparison_values']+=1;assert delta<1e-14,(label,delta)
 else:assert a==b,(label,a,b)
awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
try:
 independent=read(OUT/'independent.json');assert independent['status']=='completed'
 assert read(OUT/'replay-v2.json')['status']=='completed'
 assert train.source_hash()=='089b03d13e1950b8e0c1ff54a147a121cf78db66acaaa2dcedd9ebf29afc3543'
 ledger=read(C/'members.json')
 for member in ledger['members']:
  root=C/member['pair_path'];pair=read(root/'pair.json');pointer=root/'runs'/pair['parent_run_id']/'checkpoints/best.json';p=read(pointer);meta=read(pointer.parent/p['metadata']);parent=load_checkpoint(pointer,expected_hashes=meta['state']['identity_hashes'],device='cpu');packed=train.pack_state(parent);state_hash=canonical_hash(_json_tree(packed))
  schedule=parent.scheduler.state_dict();assert len(set(schedule['base_lrs']))==1;schedule['base_lrs']=schedule['base_lrs'][:1]
  initial={'update':parent.update,'epoch':parent.epoch,'cursor':parent.cursor,'order':array_hash(parent.order.numpy()),'sampler':array_hash(parent.sampler_generator.get_state().numpy()),'rng':canonical_hash(_json_tree(parent.rng_states)),'schedule':schedule,'logical_batch':parent.config.training.batch_size,'allowance':parent.config.training.continuation_updates,'validation_interval':parent.config.training.validation_interval,'patience':parent.config.training.patience,'checkpoint_interval':parent.config.training.checkpoint_interval}
  assert pair['parent_pairing']==initial
  score=read(root/'score-table.json');contrasts=dict(zip(score['ids'],score['contrasts']));warm=[(role,m) for role,m in pair['methods'].items() if m.get('parent_run_id')];assert len(warm)==6
  refrows=None;row_counts={};first_losses={}
  with np.load(C/member['observations_path'],allow_pickle=False) as observations:
   F_X=observations['F_X'];F_y=observations['F_y'];F_rows=observations['F_row_id']
  batch_indices=parent.order[parent.cursor:min(parent.cursor+parent.config.training.batch_size,len(parent.order))]
  xx=torch.tensor(parent.preprocessing.transform_X(F_X),dtype=torch.float32);yy=torch.tensor(parent.preprocessing.transform_y(F_y),dtype=torch.float32)
  with torch.no_grad():expected_first_loss=float(torch.square(parent.model(xx.index_select(0,batch_indices))-yy.index_select(0,batch_indices)).mean())
  for role,m in warm:
   assert m['parent_run_id']==pair['parent_run_id'] and m['parent_checkpoint_sha256']==pair['parent_checkpoint_sha256'] and m['initial_pairing']==initial
   stage=m['stages'][-1];stage_root=Path(stage['path']);stage_root=stage_root if stage_root.is_absolute() else root/stage_root
   ev=[json.loads(l) for l in (stage_root/'events.jsonl').read_text().splitlines()];rows=[v['row_ids'] for v in ev if v['type']=='update'];hashes=[array_hash(np.asarray(v,dtype=np.int64)) for v in rows]
   assert hashes==m['executed_row_ids']==stage['row_hashes'];assert len(rows)==stage['executed_updates'];row_counts[role]=len(rows)
   if role in ('keep','top','random','greedy','greedy_long'):
    first=next(v for v in ev if v['type']=='update');assert first['row_ids']==F_rows[batch_indices.numpy()].tolist();assert first['loss_standardized_mse']==expected_first_loss,(role,first['loss_standardized_mse'],expected_first_loss);first_losses[role]=first['loss_standardized_mse']
   if role=='keep':refrows=rows
   if role in ('top','random','greedy','greedy_long'):
    allocation=m['allocation'];assert allocation['table_hash']==score['table_hash']
    assert allocation['contrasts_float64']==[contrasts[u] for u in allocation['ids']]
  assert refrows is not None
  for role,m in warm:
   stage=m['stages'][-1];stage_root=Path(stage['path']);stage_root=stage_root if stage_root.is_absolute() else root/stage_root
   rows=[v['row_ids'] for line in (stage_root/'events.jsonl').read_text().splitlines() if (v:=json.loads(line))['type']=='update'];n=min(len(rows),len(refrows));assert rows[:n]==refrows[:n]
  searches=[]
  for key in ('greedy_report','greedy_long_report'):
   g=read(pair[key]);assert g['parent_state_sha256']==state_hash
   assert g['parent_identity_hashes']==parent.identity_hashes
   searches.append(g)
  assert pair['methods']['greedy_long']['original_parent_audit']['complete_state_sha256']==state_hash
  report['pairing'].append({'teacher_index':member['teacher_index'],'repetition':member['repetition'],'parent_state_sha256':state_hash,'parent_checkpoint_path':str(pointer.relative_to(ROOT)),'parent_checkpoint_sha256':sha(pointer),'warm_forks':len(warm),'raw_update_rows':row_counts,'exact_original_parent_stream':True,'same_score_and_original_contrasts':True,'parent_expected_first_training_loss':expected_first_loss,'actual_first_losses':first_losses});save()
 print('independent original-parent/warm-fork/raw-row audit exact:16 pairs/96 warm forks',flush=True)
 tick=time.perf_counter_ns();result=production.analyze(C);report['analyze_ns']=time.perf_counter_ns()-tick
 compare(independent['methods'],result['methods'],'all9 methods/all4 metrics');compare(independent['differences'],result['differences'],'all4 paired differences/teacher means/repeats/paired-bootstrap/t sensitivity')
 for independent_row,reviewed_row in zip(independent['pairs'],result['records']):
  compare(independent_row['risks'],reviewed_row['risks'],'pair risks');compare(independent_row['F_variance'],reviewed_row['F_variance'],'F variance')
 assert result['complete_cohort'] and result['completed_repetitions']==16 and result['independent_teachers']==8
 snapshot=sha(C/'analysis.json')
 for name in ('report-1.md','report-2.md'):
  tick=time.perf_counter_ns();destination=OUT/name;production.report(C,destination);assert sha(C/'analysis.json')==snapshot
  report['renderings'].append({'path':str(destination.relative_to(ROOT)),'sha256':sha(destination),'bytes':destination.stat().st_size,'elapsed_ns':time.perf_counter_ns()-tick});save()
 assert (OUT/'report-1.md').read_bytes()==(OUT/'report-2.md').read_bytes()
 report['report_bytes_identical']=True;report['analysis_sha256']=snapshot
 report['status']='completed'
except BaseException:report['status']='failed';report['failures'].append(traceback.format_exc());raise
finally:
 awake.terminate();awake.wait();report.update(elapsed_ns=time.perf_counter_ns()-START,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,script_sha256=sha(__file__));save()
