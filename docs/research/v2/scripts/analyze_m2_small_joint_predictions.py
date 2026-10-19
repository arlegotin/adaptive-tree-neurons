"""Fixed WP04 selectors on existing M2d F geometry, then join audited outcomes.

Development reanalysis only: four existing planted bases/R2/legacy treatment.
"""
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from treeneuron.config import canonical_json
from treeneuron.research.joint_selector import JointSelectionConfig,prepare_joint_problem
from treeneuron.research.ledger import write_new_json
from treeneuron.train import source_hash

root,output=map(Path,sys.argv[1:]);started=time.perf_counter_ns()
export=json.loads((root/'export.json').read_text());assert export['status']=='completed_eligible'
assert len(export['cases'])==8 and export['new_fits']==0
output.mkdir(parents=True,exist_ok=False)
source=source_hash();write_new_json(output/'request.json',dict(source_sha256=source,export_sha256=hashlib.sha256((root/'export.json').read_bytes()).hexdigest(),fixed_ridges=[.01,.01],parent_span_ridge=0.,purpose='Small-oracle development prediction check; no fitting/new teacher/T'))
# Lock every F-only choice before reading finite training outcomes.
choices={};geometry_reports={};max_dense_error=0.
def columns(a):
 rms=np.linalg.norm(a,axis=0)/np.sqrt(len(a));keep=rms>max(1e-15,1e-12*float(np.max(rms,initial=0.)))
 return a[:,keep]/rms[keep]
def dense(J,r,blocks,ids,cfg):
 A=columns(J) if cfg.condition_on_parent else np.empty((len(r),0))
 if cfg.parent_ridge==0 and A.shape[1]:
  u,s,_=np.linalg.svd(A,full_matrices=False);A=u[:,s>1e-10*s[0]]
 B=np.column_stack([columns(blocks[j]) for j in ids]) if ids else np.empty((len(r),0))
 design=np.column_stack([A,B]);penalty=np.r_[np.full(A.shape[1],cfg.parent_ridge),np.full(B.shape[1],cfg.gate_ridge)]
 aug=np.vstack([design/np.sqrt(len(r)),np.diag(np.sqrt(penalty))]);rhs=np.r_[r/np.sqrt(len(r)),np.zeros(design.shape[1])]
 coef=np.linalg.lstsq(aug,rhs,rcond=None)[0];error=aug@coef-rhs
 return float(error@error)
for case in export['cases']:
 name=case['case_id'];path=root/name/'F-geometry.npz'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==case['geometry_sha256']
 with np.load(path,allow_pickle=False) as archive:
  J=archive['parent_J'].copy();r=archive['residual'].copy();blocks={j:archive['gate_'+str(j)].copy() for j in case['eligible_ids']}
 reports={};choice=dict(cheap=case['current_top'],random=case['matched_random'])
 for method,cfg in [('joint',JointSelectionConfig(count=2)),('unconditioned',JointSelectionConfig(count=2,condition_on_parent=False)),('pure_span',JointSelectionConfig(count=2,parent_ridge=0.))]:
  problem=prepare_joint_problem(J,r,blocks,cfg);selection=problem.greedy(sorted(blocks))
  choice[method]=selection['selected_ids'];predictions=[]
  for count in (0,1,2):
   for ids in itertools.combinations(sorted(blocks),count):
    reference=dense(J,r,blocks,ids,cfg);error=abs(reference-problem.objective(ids));max_dense_error=max(max_dense_error,error)
    np.testing.assert_allclose(problem.objective(ids),reference,rtol=1e-8,atol=1e-10)
    predictions.append(dict(unit_ids=list(ids),gain=problem.gain(ids),singleton_sum=sum(problem.gain([j]) for j in ids),objective=problem.objective(ids)))
  reports[method]=dict(config=asdict(cfg),selection=selection,parent_rank=problem.parent_rank,parent_span_novelty_fraction={str(k):v for k,v in problem.novelty_fraction.items()},scaling=dict(parent=problem.scaling['parent'],gate={str(k):v for k,v in problem.scaling['gate'].items()}),all_subset_predictions=predictions)
  if method=='joint':
   local=problem.singleton_top(sorted(blocks));choice['local_singleton']=local['selected_ids'];reports['local_singleton']=local
 choices[name]=choice;geometry_reports[name]=reports
write_new_json(output/'F-only-selections.json',dict(schema='research/v2-small-F-selections',source_sha256=source,choices=choices,geometry_reports=geometry_reports,max_dense_objective_error=max_dense_error,finite_outcomes_opened=False))
rows=[];correlations=[];nominal=0;fitted=0
for case in export['cases']:
 name=case['case_id'];outcomes=json.loads((root/name/'outcomes.json').read_text())['records'];nominal+=len(outcomes)
 fitted+=sum(row['status']=='completed_eligible' for row in outcomes)
 lookup={tuple(row['unit_ids']):row for row in outcomes};keep=lookup[()]
 eligible=case['eligible_ids'];e=len(eligible);count=min(2,e)
 assert all(lookup[ids]['status']=='completed_eligible' for ids in itertools.combinations(eligible,2))
 def ordered(records):
  remaining=list(records);order=[]
  while remaining:
   best=min(row['selected_g_mse'] for row in remaining)
   winner=min((row for row in remaining if row['selected_g_mse']<=best+1e-12*max(1.,abs(best))),key=lambda row:tuple(row['unit_ids']))
   order.append(winner);remaining.remove(winner)
  return order
 singles=[lookup[(j,)] for j in eligible];pairs=[lookup[ids] for ids in itertools.combinations(eligible,2)]
 selected={**choices[name],'G_singleton':sorted(row['unit_ids'][0] for row in ordered(singles)[:count]),'G_pair_oracle':ordered(pairs)[0]['unit_ids']}
 for method,ids in selected.items():
  record=lookup[tuple(ids)]
  assert record['status']=='completed_eligible'
  rows.append(dict(case_id=name,teacher_index=case['teacher_index'],repetition=case['repetition'],method=method,unit_ids=ids,selected_g_mse=record['selected_g_mse'],V_NMSE=record['metrics']['V']['normalized_mse'],E_NMSE=record['metrics']['E']['normalized_mse'],procedure_job_id=record['procedure_job_id'],standalone_cost=record['standalone_cost']))
 for method in ('joint','unconditioned','pure_span'):
  predictions=geometry_reports[name][method]['all_subset_predictions']
  for size,label in ((1,'singleton'),(2,'pair')):
   pred={tuple(p['unit_ids']):p['gain'] for p in predictions if len(p['unit_ids'])==size}
   x=np.array(list(pred.values()));y=np.array([keep['selected_g_mse']-lookup[ids]['selected_g_mse'] for ids in pred])
   defined=len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0
   correlations.append(dict(case_id=name,method=method,subset_type=label,count=len(x),Spearman=float(spearmanr(x,y).statistic) if defined else None,undefined_reason=None if defined else 'constant predicted or finite selected gain',stage_zero_or_nonpositive=int(np.count_nonzero(y<=0))))
prediction_ranks=[]
for teacher in range(4):
 cases=[case for case in export['cases'] if case['teacher_index']==teacher]
 common=set.intersection(*(set(case['eligible_ids']) for case in cases))
 pairs=list(itertools.combinations(sorted(common),2));joint=[];singleton=[];actual=[]
 for ids in pairs:
  jp=[];sp=[];vp=[]
  for case in cases:
   name=case['case_id']
   pred=next(p for p in geometry_reports[name]['joint']['all_subset_predictions'] if p['unit_ids']==list(ids))
   outcomes=json.loads((root/name/'outcomes.json').read_text())['records']
   keep=next(p for p in outcomes if p['unit_ids']==[])
   fitted_pair=next(p for p in outcomes if p['unit_ids']==list(ids))
   jp.append(pred['gain']);sp.append(pred['singleton_sum'])
   vp.append(keep['metrics']['V']['normalized_mse']-fitted_pair['metrics']['V']['normalized_mse'])
  joint.append(float(np.mean(jp)));singleton.append(float(np.mean(sp)));actual.append(float(np.mean(vp)))
 defined=len(pairs)>1 and all(np.ptp(v)>0 for v in (joint,singleton,actual))
 cj=float(spearmanr(joint,actual).statistic) if defined else None
 cs=float(spearmanr(singleton,actual).statistic) if defined else None
 prediction_ranks.append(dict(teacher_index=teacher,pair_count=len(pairs),joint_Spearman=cj,
     independent_singleton_sum_Spearman=cs,joint_better=cj>cs if defined else None,
     pair_ids=[list(p) for p in pairs],joint_gains=joint,singleton_sums=singleton,mean_R2_finite_V_gains=actual,
     undefined_reason=None if defined else 'Constant predicted or finite pair gains'))
defined=[r for r in prediction_ranks if r['joint_better'] is not None]
prediction_screen=dict(defined_teachers=len(defined),joint_better_teachers=sum(r['joint_better'] for r in defined),
    passed=len(defined)>=2 and sum(r['joint_better'] for r in defined)>len(defined)/2,
    scope='Existing spec held-out small-oracle prediction screen only; not full Gate B or fresh M3 evidence')
means={};teacher_means=[]
for method in sorted({row['method'] for row in rows}):
 for teacher in range(4):
  chosen=[row for row in rows if row['method']==method and row['teacher_index']==teacher];assert len(chosen)==2
  teacher_means.append(dict(method=method,teacher_index=teacher,V_NMSE=float(np.mean([r['V_NMSE'] for r in chosen])),E_NMSE=float(np.mean([r['E_NMSE'] for r in chosen]))))
 selected=[r for r in teacher_means if r['method']==method]
 means[method]={key:float(np.mean([r[key] for r in selected])) for key in ('V_NMSE','E_NMSE')}
write_new_json(output/'analysis.json',dict(schema='research/v2-small-joint-development-analysis',source_sha256=source,export_sha256=hashlib.sha256((root/'export.json').read_bytes()).hexdigest(),status='completed',independent_planted_bases=4,repetitions=2,nominal_old_fit_entries=nominal,eligible_old_fits=fitted,new_fits=0,primary_M3_teachers=0,rows=rows,teacher_means=teacher_means,means=means,correlations=correlations,teacher_pair_prediction_ranks=prediction_ranks,prediction_screen=prediction_screen,max_dense_objective_error=max_dense_error,elapsed_ns=time.perf_counter_ns()-started,T_accessed=False,Gate_B_decision=False,limitations=['Known development data, not independent validation','Planted h8/m2 with legacy WD0/patience, different from locked M3 treatment','Pair oracle uses G and pays every finite pair fit; no practical efficiency inference','Only audited producer scalar outcomes joined here; independent raw prediction arithmetic was completed in M2 audit']))
print(json.dumps(dict(means=means,prediction_screen=prediction_screen,max_dense_objective_error=max_dense_error,nominal=nominal,eligible=fitted)),flush=True)
