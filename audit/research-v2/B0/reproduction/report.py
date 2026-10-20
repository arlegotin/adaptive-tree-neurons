"""Rebuild B0 development prose and tables from its authenticated arithmetic."""
from pathlib import Path
import argparse,csv,hashlib,json

def main():
 p=argparse.ArgumentParser();p.add_argument('--analysis',required=True,type=Path);p.add_argument('--expected-sha256',required=True)
 p.add_argument('--output',required=True,type=Path);p.add_argument('--tables',required=True,type=Path);args=p.parse_args()
 raw=args.analysis.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=args.expected_sha256:raise ValueError('Canonical independent analysis changed')
 result=json.loads(raw);a=result['analysis']
 if result['status']!='complete_raw_arithmetic_checked' or a['independent_teachers']!=4 or a['paired_repetitions']!=2 or len(result['rows'])!=72:raise ValueError('Complete B0 audit required')
 if args.output.exists() or args.tables.exists():raise ValueError('Fresh report/table outputs required')
 args.tables.mkdir(parents=True)
 with (args.tables/'cases.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['teacher','repeat','config_id','group','V_NMSE','E_NMSE']);writer.writeheader()
  for row in result['rows']:writer.writerow({k:row[k] for k in writer.fieldnames})
 with (args.tables/'teacher-effects.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(a['teacher_effects'][0]));writer.writeheader();writer.writerows(a['teacher_effects'])
 with (args.tables/'V-selected.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(a['V_selected'][0]));writer.writeheader();writer.writerows(a['V_selected'])
 report=['# Native backbones on four fresh primary synthetic problems','']
 if a['screen_trigger']:
  report+=['The predeclared development screen passed. This permits the separately fixed competent-ordinary/residual-structure diagnostic; it does not admit M6 or establish a gated-model benefit.','']
 else:
  report+=['The stronger-backbone screen was **negative: 0 of 4 teachers met the 5% improvement rule**, against a required 3 of 4. The small 64-unit ordinary coordinate won the validation-selected one-layer family in all eight repetitions, and RealMLP-TD won the backbone family. The ordinary procedure had lower exploratory E error on all four teacher averages. No B0 expansion or residual-growth branch is admitted.','']
 report+=['This is development evidence on the named primary d16 synthetic construction. It is not a new placement result, a full tuning benchmark, a general ranking of neural methods or protected confirmation. M3/Gate B remains negative. The main grown model is not an arm of this screen.','',
  'Exactly four fresh teachers, two paired repeats and nine fixed recipes produced 72 complete native fits; all 76 queue jobs completed without failure/recovery. F2306/G577/V577/E65536 and sigma 0.1 were fixed before generation. Models fit F, used G for native stopping, and the two finite families chose configurations on V only. No seed was selected. E is permanently development information.','',
  '## Complete configuration means','',
  'Original-unit MSE divided by the teacher’s F variance (ddof=0), averaged over its two repeats and four teachers; lower is better.','',
  '| Configuration | Family | V NMSE | E NMSE |','|---|---|---:|---:|']
 for row in a['config_means']:report.append(f"| {row['config_id']} | {row['group']} | {row['V_NMSE']:.6f} | {row['E_NMSE']:.6f} |")
 report+=['',
  'O0 uses [64]/ReLU/LR .005/WD 0/dropout 0 under native P03 quantile preprocessing and stopping. O1–O4 use widths 64/128/256/512 with LR .001/WD .01/dropout .1. Thus O0 versus O1 changes an optimization/regularization bundle; this does not isolate learning rate, dropout or decay. MLP-D, ResNet-D, RealMLP-TD and embedded TabM retain their published native anchors, preprocessing, losses and schedules. No short engineering recipe substituted for native fitting.','',
  '## Validation-selected family comparison','',
  'Choose one configuration inside each family on V for each repeat, average the two E risks within teacher, then compare. Five one-layer and four backbone fits are paid per teacher/repeat; this is not equal portfolio cost. The teacher, not 72 fits or 262,144 E rows across teachers, is the independent unit.','',
  '| Teacher | Selected one-layer E NMSE | Selected backbone E NMSE | Backbone error change | Meets 5% reduction |','|---|---:|---:|---:|---|']
 for row in a['teacher_effects']:
  report.append(f"| {row['teacher']} | {row['one_layer_E_NMSE']:.6f} | {row['backbone_E_NMSE']:.6f} | {-100*row['relative_reduction']:+.1f}% | {'Yes' if row['at_least_5_percent'] else 'No'} |")
 report+=['',
  'The negative screen supplies no measured one-layer-family ceiling requiring a stronger parent in this regime. It does not prove that all tuned/backbone alternatives fail. In particular, the historical competent F-standardized continuation used in M3 was not directly rerun on these teachers, and the native one-layer grid uses quantile preprocessing. The predeclared standardized-ordinary and residual-structure safeguards remain prerequisites for any separately justified M6 proposal; this screen did not open that follow-up.','',
  'The earlier real-data anchor survey favored existing methods over P00 on all five observational tasks. That remains separate evidence. Do not pool its tasks/teachers with B0 or compare B0 and M3 means as paired before/after improvements. Four development teachers give limited population precision; no confirmatory interval, power or state-of-the-art claim is made.','',
  '## Execution and independent checks','',
  f"The queue charged {result['charged_job_ns']/1e9:.2f} seconds of enclosing job time (about {result['charged_job_ns']/60e9:.2f} minutes), including data/preparation/native/query work inside those jobs. Outer imports/publication and the earlier software qualification are separate. Another native NODE study and unrelated processes competed for resources; these are not dedicated-hardware or deployment timings.",'',
  f"Independent NumPy arithmetic checked {result['raw_prediction_vectors']} aligned F/G/V/E vectors and {result['metric_comparisons']} metrics against the producer’s fsum calculations; maximum absolute metric discrepancy was {result['max_abs_metric_difference']:.3g}. All 72 F variances also agreed within the declared numerical tolerance. The read-only queue lease, source/config/seed/role/native stopping and closed-state records were authenticated. Original inference/training receipts are trusted: no independent retraining or new model inference is claimed.",'',
  'The producer core is commit 80afc0d (the full commit is in its reconciliation record), package `'+result['source_sha256']+'`; the plan hash is `'+result['plan_hash']+'`. The independently checked raw arithmetic hash is `'+args.expected_sha256+'`. Native worker quality/registry flags remain false: this is a controlled development screen while final common-source/support/practical qualification is unfinished. No protected T was created or read.','',
  'All four new teacher IDs/seeds are excluded from future confirmation. Local raw evidence and separately hashed operators are retained at `artifacts/research-v2/WP07/B0-native-primary-backbones-80afc0d/`. Rebuild with its read-only `audit.py` and this `report.py`; full arrays/checkpoints are retained execution assets, not promised in a clean checkout. [Declaration](../docs/research/v2/B0-native-primary-backbones.md), [execution](../docs/research/v2/B0-execution.md).','']
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text('\n'.join(report))
 (args.tables/'summary.json').write_text(json.dumps(dict(analysis_sha256=args.expected_sha256,analysis=a,scope=result['scope'],quality_eligible=False,T_accessed=False),sort_keys=True,indent=2)+'\n')
 print(json.dumps(dict(report=str(args.output),configurations=9,teacher_rows=4,case_rows=72,screen_trigger=a['screen_trigger'])))
if __name__=='__main__':main()
