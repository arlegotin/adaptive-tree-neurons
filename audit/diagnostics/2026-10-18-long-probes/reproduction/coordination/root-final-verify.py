"""Root read-only verification of completed diagnostic/auditor evidence."""
import hashlib,json,time,subprocess
from pathlib import Path
ROOT=Path.cwd(); S=ROOT/'.superpowers/sdd/2026-10-18-long-probe-allocation-diagnostic'; start=time.perf_counter_ns()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
checks=[]
for rel in ['environment/long-probes-20261018/cost-only-inventory.json','environment/long-probes-20261018/cohort-inventory.json','environment/long-probes-20261018/audit/inventory.json']:
 inv=read(ROOT/rel); rows=inv.get('artifacts',inv.get('files',[])); assert rows,rel
 for r in rows:
  p=ROOT/r['path']; assert p.stat().st_size==r['bytes'],p
  assert sha(p)==r['sha256'],p
 checks.append({'path':rel,'sha256':sha(ROOT/rel),'files':len(rows),'bytes':sum(r['bytes'] for r in rows)})
for rel,field in [('source-freeze.json','files'),('historical-preservation.json','historical_files')]:
 obj=read(S/rel)
 for p,v in obj[field].items():
  assert sha(ROOT/p)==v['sha256'],p
checks.append({'held_source_and_historical_files':True})
for rel,seal in [('environment/long-probes-20261018/cohort-inventory.json','environment/long-probes-20261018/cohort-inventory-seal.json'),('environment/long-probes-20261018/audit/inventory.json','environment/long-probes-20261018/audit/inventory-seal.json'),('pilot/long-probes-20261018/resource-decision.json','pilot/long-probes-20261018/resource-decision-seal.json'),('protocol/diagnostics/2026-10-18-long-probes-lock.json','protocol/diagnostics/2026-10-18-long-probes-lock-seal.json')]:
 if not (ROOT/seal).exists():
  candidates=list((ROOT/rel).parent.glob(Path(rel).stem+'*seal*'))
  assert len(candidates)==1,(seal,candidates);seal=str(candidates[0].relative_to(ROOT))
 assert sha(ROOT/rel)==read(ROOT/seal)['sha256'],rel
A=ROOT/'artifacts/long-probes-20261018/audit'
ind=read(A/'independent.json'); rep=read(A/'replay-v2.json'); rec=read(A/'reconciliation-v2.json'); pres=read(A/'preservation.json')
for a in [ind,rep,rec,pres]:assert a['status']=='completed' and not a.get('failures'),a.get('failures')
assert rep['complete_trials']==130 and rep['complete_rankings']==2 and len(rep['comparisons'])==133
assert rep['automatic_retries']==0 and rep['new_efficacy_units']==0
assert all(v['exact'] for v in rep['comparisons'])
assert rep['short_replay_reused_read_only']['new_short_numerical_calls']==0
assert rec['report_bytes_identical'] and len(rec['renderings'])==2
assert (A/'report-1.md').read_bytes()==(A/'report-2.md').read_bytes()
for v in rec['renderings']:assert sha(ROOT/v['path'])==v['sha256']
analysis=read(ROOT/'artifacts/long-probes-20261018/cohort/analysis.json')
assert analysis['complete_cohort'] and analysis['independent_teachers']==8 and analysis['completed_repetitions']==16
for k,values in ind['methods'].items():
 for metric,v in values.items():assert abs(v-analysis['methods'][k][metric])<1e-14
for k,values in ind['differences'].items():
 for metric in ['mean','median','percentile_95','student_t_95','teacher_effects','repetition_effects']:
  a,b=values[metric],analysis['differences'][k][metric]
  if isinstance(a,list):assert len(a)==len(b) and all(abs(x-y)<1e-14 for x,y in zip(a,b))
  else:assert abs(a-b)<1e-14
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='5e46ce1230256c37339714c38adfa9324167f6dd'
result={'status':'passed','inventory_checks':checks,'source_and_history_unchanged':True,'original_and_audit_hashes_sizes_seals_exact':True,'independent_and_production_results_reconciled':True,'exact_replayed_rankings':2,'exact_replayed_trials':130,'exact_complete_states':133,'renderings_identical':True,'report_sha256':sha(A/'report-1.md'),'new_efficacy_units':0,'automatic_numerical_retries':0,'read_only_verification_ns':time.perf_counter_ns()-start}
with (S/'root-final-verification.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(result,indent=2))
