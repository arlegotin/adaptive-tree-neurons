"""Prospectively fixed teacher0/R0: both complete rankings, durable last restore."""
import time,os,json,hashlib,resource,traceback,copy,subprocess
from pathlib import Path
from dataclasses import fields
START=time.perf_counter_ns()
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.checkpoint import load_checkpoint
from treeneuron.artifacts import create_run,finish_run
from treeneuron.growth.score import ScoreTable
from treeneuron.baselines.greedy import run_greedy_probes
from treeneuron.pilot.calibrate import teacher_inputs,repetition_config
from treeneuron.config import resolve_experiment,canonical_hash
from treeneuron.data.types import array_hash
from treeneuron.experiments.paired import _json_tree
ROOT=Path.cwd(); C=ROOT/'artifacts/long-probes-20261018/cohort';REF=C/'teacher-0/repetition-0';OUT=ROOT/'artifacts/long-probes-20261018/audit';ENV=ROOT/'environment/long-probes-20261018/audit'
report={'status':'running','failures':[],'comparisons':[],'checkpoint_inputs':[],'locator_normalization':[],'replayed_rankings':[]}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save():(OUT/'replay.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
def eq(a,b,label=''):
 if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu()),label
 elif isinstance(a,np.ndarray):assert a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),label
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),(label,a.keys(),b.keys())
  for k in a:eq(a[k],b[k],label+'/'+str(k))
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b),label
  for i,(x,y) in enumerate(zip(a,b)):eq(x,y,label+'/'+str(i))
 else:assert a==b,(label,a,b)
def load(p):
 ptr=read(p);meta=p.parent/ptr['metadata'];m=read(meta);payload=p.parent/m['payload']
 report['checkpoint_inputs'] += [{'path':str(x.relative_to(ROOT)),'sha256':sha(x),'bytes':x.stat().st_size} for x in (p,meta,payload)]
 return load_checkpoint(p,expected_hashes=m['state']['identity_hashes'],device='cpu')
def normalized(meta):
 meta=copy.deepcopy(meta)
 if 'greedy_probe' in meta.get('growth_metadata',{}):
  for key in ('coordinator_run_id','parent_checkpoint_sha256'):meta['growth_metadata']['greedy_probe'][key]='<verified-run-locator>'
 return meta
def compare(a,b,label,data):
 x=train.pack_state(a);y=train.pack_state(b);eq(x['tensors'],y['tensors'],label+'/all_tensors');eq(normalized(x['metadata']),normalized(y['metadata']),label+'/all_metadata')
 eq(a.next_row_ids(data.F),b.next_row_ids(data.F),label+'/next_rows');eq(a.predict(data.G.X),b.predict(data.G.X),label+'/G_predictions')
 ha=[{k:v for k,v in event.items() if not (event['type']=='checkpoint' and k=='sha256')} for event in a.history]
 hb=[{k:v for k,v in event.items() if not (event['type']=='checkpoint' and k=='sha256')} for event in b.history]
 eq(ha,hb,label+'/history except descriptor locators')
 if a.best_snapshot is None or b.best_snapshot is None:eq(a.best_snapshot,b.best_snapshot,label+'/best_snapshot_presence')
 else:eq(a.best_snapshot['tensors'],b.best_snapshot['tensors'],label+'/best_snapshot_tensors');eq(normalized(a.best_snapshot['metadata']),normalized(b.best_snapshot['metadata']),label+'/best_snapshot_metadata')
 report['comparisons'].append({'name':label,'exact':True,'maximum_numeric_difference':0,'update':a.update,'epoch':a.epoch,'cursor':a.cursor,'next_row_ids_sha256':array_hash(a.next_row_ids(data.F)),'G_predictions_sha256':array_hash(a.predict(data.G.X)),'tensor_groups':list(x['tensors']),'metadata_fields':list(x['metadata']),'history_original_sha256':canonical_hash(a.history),'history_replay_sha256':canonical_hash(b.history),'history_normalization':'Only checkpoint.sha256 run-dependent descriptor locator; complete ordered update IDs commitments/losses/evaluations unchanged.'});save()
def events(p):return [json.loads(l) for l in (p/'events.jsonl').read_text().splitlines()]
awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
try:
 assert train.source_hash()=='089b03d13e1950b8e0c1ff54a147a121cf78db66acaaa2dcedd9ebf29afc3543'
 assert read(OUT/'independent.json')['status']=='completed'
 spec=read(C/'study-lock.json')['spec']; cfg=spec['base_config'];cfg['seeds'].update(spec['study_seeds']);cfg['pilot'].update(evaluation_rows=spec['evaluation_rows'],probe_updates=spec['short_probe_updates']);cfg=resolve_experiment(cfg)
 teacher,data,e,tc=teacher_inputs(cfg,0);cfg=repetition_config(tc,0);pair=read(REF/'pair.json');parent_path=REF/'runs'/pair['parent_run_id']/'checkpoints/best.json';parent=load(parent_path)
 report['original_parent']={'path':str(parent_path.relative_to(ROOT)),'sha256':sha(parent_path),'complete_state_sha256':canonical_hash(_json_tree(train.pack_state(parent))),'update':parent.update}
 for mode,key,allowance in [('fixed_updates','greedy_report',320),('patience','greedy_long_report',8000)]:
  original_path=Path(pair[key]);original=read(original_path)
  eq(original['parent_state_sha256'],report['original_parent']['complete_state_sha256'],'original parent complete state')
  table_values={}
  with np.load(original_path.parent/original['score_artifact']['path'],allow_pickle=False) as z:
   for field in fields(ScoreTable):
    a=z[field.name]
    table_values[field.name]=tuple(a.tolist()) if field.name=='status' else a.item() if a.ndim==0 else torch.from_numpy(a.copy())
  table=ScoreTable(**table_values);assert table.table_hash==original['score_hash']
  p=create_run(OUT/'ranking-replays',cfg.to_dict());tick=time.perf_counter_ns()
  allocation=run_greedy_probes(load(parent_path),table,data,probe_updates=allowance,seed=cfg.seeds.allocation,paths=p,stopping_mode=mode)
  fresh_path=p.directory/'greedy.json';fresh=read(fresh_path);finish_run(p,{'complete':fresh['complete']},'completed')
  for k in ('allocation','eligible_ids','effective_count','improvements','keep_g_mse','probe_updates','complete','stopping_mode','parent_state_sha256','policy','parent_identity_hashes'):eq(original[k],fresh[k],mode+'/'+k)
  # The parent descriptor references new serialized filenames. Its state is independently compared.
  compare(load(original_path.parent/'parent.json'),load(p.directory/'parent.json'),mode+'/parent',data)
  report['locator_normalization'].append({'mode':mode,'fields':['growth_metadata.greedy_probe.coordinator_run_id','growth_metadata.greedy_probe.parent_checkpoint_sha256'],'original_coordinator_run_id':original_path.parent.name,'replay_coordinator_run_id':p.run_id,'original_parent_descriptor_sha256':original['parent_checkpoint_sha256'],'replay_parent_descriptor_sha256':fresh['parent_checkpoint_sha256'],'justification':'Only run identity and descriptor locators; parent complete tensors/metadata/history independently exact. Learned values, RNG, sampler and all stage state are compared unchanged.'})
  totals={'updates':0,'evaluations':0,'checkpoint_events':0}
  for old,new in zip(original['trials'],fresh['trials']):
   for k in old:
    if k not in ('run_id','path','nanoseconds'):eq(old[k],new[k],mode+'/'+old['candidate']+'/'+k)
   a=original_path.parent/old['path'];b=p.directory/new['path'];ea=events(a);eb=events(b)
   for kind in ('update','evaluation'):
    aa=[x for x in ea if x['type']==kind];bb=[x for x in eb if x['type']==kind];eq(aa,bb,mode+'/'+old['candidate']+'/'+kind);totals['updates' if kind=='update' else 'evaluations']+=len(aa)
   ca=[x for x in ea if x['type']=='checkpoint'];cb=[x for x in eb if x['type']=='checkpoint'];eq([{k:v for k,v in x.items() if k!='sha256'} for x in ca],[{k:v for k,v in x.items() if k!='sha256'} for x in cb],mode+'/checkpoint events without descriptor hash');totals['checkpoint_events']+=len(ca)
   compare(load(a/'checkpoints/best.json'),load(b/'checkpoints/best.json'),mode+'/'+old['candidate']+'/best',data)
  report['replayed_rankings'].append({'mode':mode,'original':str(original_path.relative_to(ROOT)),'replay':str(fresh_path.relative_to(ROOT)),'trials':len(fresh['trials']),'elapsed_ns':time.perf_counter_ns()-tick,**totals});save();print('ranking exact',mode,totals,flush=True)
  if mode=='patience':
   chosen=next((i for i,t in enumerate(original['trials']) if t['unit_id'] is not None),0);old=original['trials'][chosen];new=fresh['trials'][chosen]
   a=original_path.parent/old['path']/'checkpoints/last.json';b=p.directory/new['path']/'checkpoints/last.json';compare(load(a),load(b),'prospective-first-eligible-long-last-restore',data)
   report['restore']={'candidate':old['candidate'],'original':str(a.relative_to(ROOT)),'replay':str(b.relative_to(ROOT)),'actual_durable_last':True,'all_adam_states_groups_ages_scheduler_rng_sampler_order_cursor_next_rows_G_exact':True}
 report.update(status='completed',complete_rankings=2,complete_trials=130,automatic_retries=0,new_efficacy_units=0)
except BaseException:report['status']='failed';report['failures'].append(traceback.format_exc());raise
finally:
 awake.terminate();awake.wait();report.update(elapsed_ns=time.perf_counter_ns()-START,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,script_sha256=sha(__file__));save()
