"""Independent NPZ risk arithmetic before production analyze/report; held source."""
import os,time,resource,json,hashlib,traceback,math,subprocess
from pathlib import Path
START=time.perf_counter_ns()
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from scipy.stats import t
from treeneuron import train
from treeneuron.config import resolve_experiment
from treeneuron.pilot.calibrate import teacher_inputs
ROOT=Path.cwd(); ENV=ROOT/'environment/long-probes-20261018/audit'; OUT=ROOT/'artifacts/long-probes-20261018/audit'; C=ROOT/'artifacts/long-probes-20261018/cohort'
R=['keep','top','random','greedy','greedy_long','widen_upper','ordinary_lower','ordinary_upper','static_partial']
D={'random_minus_greedy_long':('random','greedy_long'),'greedy_short_minus_greedy_long':('greedy','greedy_long'),'ordinary_upper_minus_greedy_long':('ordinary_upper','greedy_long'),'random_minus_top':('random','top')}
report={'status':'running','failures':[],'pairs':[],'teachers':[]}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(): (OUT/'independent.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
def equal(a,b,label):assert a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),label
awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
try:
 assert train.source_hash()=='089b03d13e1950b8e0c1ff54a147a121cf78db66acaaa2dcedd9ebf29afc3543'
 inv=ROOT/'environment/long-probes-20261018/cohort-inventory.json'
 assert sha(inv)=='900b5cfeaa6d3d67cc12a7b895506fea6951c83bed76e79b195921a5cf79a5fd'
 assert read(inv.with_name('cohort-inventory-seal.json'))=={'sha256':sha(inv)}
 entries=read(inv)['artifacts']
 for item in entries:assert sha(ROOT/item['path'])==item['sha256'],item['path']
 report['inventory']={'path':str(inv.relative_to(ROOT)),'sha256':sha(inv),'checked_files':len(entries),'bytes':sum((ROOT/i['path']).stat().st_size for i in entries)}
 lock=read(C/'study-lock.json'); spec=lock['spec']; cfg=spec['base_config'];cfg['seeds'].update(spec['study_seeds']);cfg['pilot'].update(evaluation_rows=spec['evaluation_rows'],probe_updates=spec['short_probe_updates']);cfg=resolve_experiment(cfg)
 members=read(C/'members.json')['members']; assert len(members)==16
 for i in range(8):
  teacher,data,e,tc=teacher_inputs(cfg,i)
  with np.load(C/f'teacher-{i}/observations.npz',allow_pickle=False) as z:
   for k,v in [('E_X',e.X),('E_y',e.y),('E_rows',e.row_id)]+[(f'{p}_{k}',getattr(getattr(data,p),k)) for p in 'FG' for k in ('X','y','row_id')]:equal(z[k],v,k)
  variance=float(np.mean((data.F.y-np.mean(data.F.y))**2)); assert variance>0
  pairs=[]
  for j in range(2):
   member=members[2*i+j];assert (member['teacher_index'],member['repetition'])==(i,j)
   p=C/member['pair_path']; pair=read(p/'pair.json'); assert pair['teacher_id']==teacher.teacher_id
   risks={}
   with np.load(p/'E-predictions.npz',allow_pickle=False) as z:
    assert set(z.files)==set(R)|{'y','row_id'};equal(z['y'],e.y,'E_y');equal(z['row_id'],e.row_id,'E_rows')
    for r in R:
     pred=z[r];assert pred.shape==e.y.shape and np.isfinite(pred).all()
     residual=pred-e.y;mse=float(np.mean(residual*residual));risks[r]={'mse':mse,'rmse':math.sqrt(mse),'mae':float(np.mean(abs(residual))),'normalized_mse':mse/variance}
   effects={k:risks[a]['normalized_mse']-risks[b]['normalized_mse'] for k,(a,b) in D.items()}
   item={'teacher_index':i,'repetition':j,'teacher_id':teacher.teacher_id,'F_variance':variance,'risks':risks,'effects':effects};pairs.append(item);report['pairs'].append(item)
  teacher_record={'teacher_index':i,'risks':{r:{k:float(np.mean([p['risks'][r][k] for p in pairs])) for k in risks[r]} for r in R},'effects':{k:float(np.mean([p['effects'][k] for p in pairs])) for k in D},'within_teacher_repeat_sd':{k:float(np.std([p['effects'][k] for p in pairs],ddof=1)) for k in D},'within_teacher_method_sd':{r:float(np.std([p['risks'][r]['normalized_mse'] for p in pairs],ddof=1)) for r in R}}
  report['teachers'].append(teacher_record);save();print('independent teacher',i,flush=True)
 report['methods']={r:{k:float(np.mean([v['risks'][r][k] for v in report['teachers']])) for k in risks[r]} for r in R}
 matrix=np.asarray([[v['effects'][k] for k in D] for v in report['teachers']]);rng=np.random.default_rng(261031106);samples=np.empty((512000,4))
 for start in range(0,512000,10000):
  stop=min(start+10000,512000); indices=rng.integers(0,8,size=(stop-start,8));samples[start:stop]=matrix[indices].mean(axis=1)
 bounds=np.percentile(samples,[2.5,97.5],axis=0);report['differences']={}
 for col,key in enumerate(D):
  values=matrix[:,col];reps=np.asarray([p['effects'][key] for p in report['pairs']]);mean=float(values.mean());se=float(values.std(ddof=1)/math.sqrt(8));delta=float(t.ppf(.975,7))*se
  report['differences'][key]={'mean':mean,'median':float(np.median(values)),'teacher_effects':values.tolist(),'repetition_effects':reps.tolist(),'repetition_sd':float(reps.std(ddof=1)),'percentile_95':bounds[:,col].tolist(),'student_t_95':[mean-delta,mean+delta],'zero_teachers':int(sum(abs(values)<1e-14)),'negative_teachers':int(sum(values< -1e-14))}
 report.update(status='completed',independent_counts={'teachers':8,'pairs':16,'roles':144,'E_rows_per_role':262144,'prediction_values':144*262144},bootstrap={'iterations':512000,'seed':261031106,'confidence':.95},production_analysis_invoked=False)
except BaseException:report['status']='failed';report['failures'].append(traceback.format_exc());raise
finally:
 awake.terminate();awake.wait();report.update(elapsed_ns=time.perf_counter_ns()-START,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,script_sha256=sha(__file__));save()
