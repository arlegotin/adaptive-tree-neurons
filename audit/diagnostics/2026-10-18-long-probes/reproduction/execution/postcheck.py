"""Final read-only preservation verification and complete audit inventory."""
import time,json,hashlib,resource,subprocess,shutil,os,traceback
from pathlib import Path
START=time.perf_counter_ns();ROOT=Path.cwd();ENV=ROOT/'environment/long-probes-20261018/audit';OUT=ROOT/'artifacts/long-probes-20261018/audit'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
report={'status':'running','failures':[]}
try:
 for name in ('independent.json','replay-v2.json','reconciliation-v2.json'):assert read(OUT/name)['status']=='completed',name
 inventory_path=ROOT/'environment/long-probes-20261018/cohort-inventory.json'
 assert sha(inventory_path)=='900b5cfeaa6d3d67cc12a7b895506fea6951c83bed76e79b195921a5cf79a5fd'
 assert read(inventory_path.with_name('cohort-inventory-seal.json'))=={'sha256':sha(inventory_path)}
 original=read(inventory_path)
 for entry in original['artifacts']:assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
 preflight=read(ROOT/'environment/long-probes-20261018/cohort-preflight.json')
 for name,digest in preflight['tracked_sha256'].items():assert sha(ROOT/name)==digest,name
 for name,digest in preflight['prior_evidence_sha256'].items():assert sha(ROOT/name)==digest,name
 report.update(original_cohort_inventory_files_verified=len(original['artifacts']),original_cohort_bytes_verified=sum((ROOT/e['path']).stat().st_size for e in original['artifacts']),tracked_files_unchanged=len(preflight['tracked_sha256']),prior_cost_evidence_unchanged=len(preflight['prior_evidence_sha256']),held_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256='089b03d13e1950b8e0c1ff54a147a121cf78db66acaaa2dcedd9ebf29afc3543',only_derived_analysis_added=True)
 assert report['held_HEAD']=='5e46ce1230256c37339714c38adfa9324167f6dd'
 native={'free_disk_bytes':shutil.disk_usage(ROOT).free}
 for label,argv in [('memory_pressure',['memory_pressure','-Q']),('vm_stat',['vm_stat'])]:
  p=subprocess.run(argv,capture_output=True,text=True);native[label]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};assert p.returncode==0
 report['final_resources']=native;report['status']='completed'
except BaseException:report['status']='failed';report['failures'].append(traceback.format_exc());raise
finally:
 report.update(elapsed_ns=time.perf_counter_ns()-START,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,script_sha256=sha(__file__))
 write(OUT/'preservation.json',report)
print(json.dumps({'status':report['status'],'verified_original_files':report.get('original_cohort_inventory_files_verified'),'held_HEAD':report.get('held_HEAD')}),flush=True)
