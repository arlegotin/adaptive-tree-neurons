"""Inspect the actual small CLI study, without loading any checkpoint."""
import hashlib,json,sys
from pathlib import Path
from treeneuron.config import canonical_hash
from treeneuron.train import source_hash

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
root=Path(sys.argv[1]);study=read(root/'study.json');manifest=read(root/'manifest.json');split=read(root/'splits.json');teacher=read(root/'teacher.json')
assert study['status']=='completed' and len(study['methods'])==17
assert study['provenance']['source_sha256']==source_hash()
assert manifest['method']['scope']=='coordinator'
assert manifest['data']['raw_hash']==split['dataset_id']
assert manifest['data']['split_hash']==study['data']['split_hash']
assert manifest['method']['backend']['actual']['device']=='cpu'
assert manifest['resources']['rss_bytes']==study['resources']['process_peak_rss_bytes']
assert manifest['costs']['enclosing']['nanoseconds']==study['costs']['total_study_ns']
assert manifest['identity']['score_hash']==read(root/'score-table.json')['table_hash']
checked=[('manifest.json',sha(root/'manifest.json'))]
for name,m in study['methods'].items():
 for i,st in enumerate(m['stages']):
  p=root/st['path']/'manifest.json';c=read(p)
  assert c['identity']['status']=='completed' and c['identity']['coordinator_run_id']==root.name
  assert c['method']['scope']=='single_model' and c['method']['total_parameters']==m['total_parameters']
  if m.get('parent_run_id'):assert c['identity']['parent_run_id']==m['parent_run_id']
  elif i:assert c['identity']['parent_run_id']==m['stages'][i-1]['run_id']
  if 'allocation' in m:
   assert c['identity']['score_hash']==m['allocation']['table_hash']
   assert c['identity']['growth_hash']==canonical_hash(m['growth_metadata'])
  if i==len(m['stages'])-1:
   assert c['method']['prediction_artifacts']==m['predictions']
   assert read(p.parent/'metrics.json')==m['metrics']
   for link in m['predictions'].values():assert sha(root/link['path'])==link['sha256']
  checked.append((str(p.relative_to(root)),sha(p)))
for path,_ in checked:
 c=read(root/path)
 assert c['data']['boundaries']['V']=='descriptive_only' and c['data']['test_labels_accessed'] is False
 for part,n in [('F',16),('G',8),('V',8)]:assert c['data']['counts'][part]=={'rows':n,'groups':n}
 for k in ('reference_mean','reference_std','reference_rows','reference_seed'):assert c['data']['teacher_normalization'][k]==teacher[k]
 assert c['data']['teacher_normalization']['target_normalization']=='none'
 assert c['resources']['pressure']['value'] is None and c['resources']['swap_delta_bytes']['value'] is None
events=[read_line for read_line in (json.loads(line) for line in (root/'events.jsonl').read_text().splitlines())]
registrations=[event for event in events if event['type']=='child_registered']
assert len(registrations)==len(checked)-1
for name,method in study['methods'].items():
 diagnostics=method['diagnostics']
 assert diagnostics['population']=='F' and diagnostics['role']=='diagnostic_only'
 assert diagnostics['input_norms']['raw']['max']>=0
 assert diagnostics['synthetic_regime_occupancy']['rows']==16
 if diagnostics['gate']['status']=='observed':
  assert diagnostics['gate']['variance']>=0
  assert len(diagnostics['gate']['quantiles'])==5
  assert diagnostics['gate']['mean_g_one_minus_g']>=0
  assert diagnostics['leaf_disagreement']['rms']>=0
 else:
  assert diagnostics['gate']['value'] is None and diagnostics['leaf_disagreement']['value'] is None
 for i,stage in enumerate(method['stages']):
  assert any(event['method']==name and event['stage_index']==i and event['run_id']==stage['run_id'] for event in registrations)
  child=read(root/stage['path']/'manifest.json')
  assert child['identity']['growth_hash']==canonical_hash(method.get('growth_metadata',{}))
  pointer=read(root/stage['path']/'checkpoints'/'best.json')
  descriptor=read(root/stage['path']/'checkpoints'/pointer['metadata'])
  assert descriptor['state']['growth_metadata']==method.get('growth_metadata',{})
assert study['execution_semantics']['inventory']=='fixed_causal_comparison'
table=read(root/'score-table.json')
assert table['export_schema_version']==1
assert 'unavailable_diagnostics' in table
out={'schema_version':1,'purpose':'Small engineering integration; not an added predictive comparison','path':str(root),'study_sha256':sha(root/'study.json'),'provenance':study['provenance'],'config_hash':study['config_hash'],'status':study['status'],'method_count':17,'manifests_verified':dict(checked),'manifest_count':len(checked),'pairing_audit':study['pairing_audit'],'total_study_ns':study['costs']['total_study_ns'],'source_unchanged_during_audit':True,'registered_children':len(registrations),'diagnostic_methods_verified':len(study['methods']),'score_export_schema':table['export_schema_version']}
Path('environment/core-final-fix-small-audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'passed','manifest_count':len(checked),'study_sha256':out['study_sha256'],'source':source_hash()}))
