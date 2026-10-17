"""Prespecified 8000-draw saved-pair diagnostics; separate noncanonical lineage."""
import time
local_started=time.perf_counter_ns()
import hashlib,json,os,resource,shutil,subprocess
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.config import load_experiment,canonical_hash
from treeneuron.pilot.design import summarize_pilot,choose_design,calibration_status
from treeneuron.artifacts import _atomic_json

base=Path('environment/h1-v2/user-pause-continuation');attempt=base/'attempt-2'
config=load_experiment('environment/h1-v2/scientific-ready/scientific-config.yaml')
readout=json.loads((attempt/'readout.json').read_text());assert readout['status']=='completed' and len(readout['pairs'])==2
root=Path(readout['artifact_root']);out=root/'joined-diagnostics';out.mkdir(exist_ok=False)
started=readout['started_perf_ns']
def elapsed():return (time.perf_counter_ns()-started)/1e9
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,value):_atomic_json(path,value,replace=True)
def observation(boundary):
    p=subprocess.run(['memory_pressure'],capture_output=True,text=True)
    return dict(boundary=boundary,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),aggregate_elapsed_seconds=elapsed(),
        saved_process_seconds=(time.perf_counter_ns()-local_started)/1e9,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        disk=dict(zip(('total','used','free'),shutil.disk_usage('.'))),memory_pressure=dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
manifest=dict(scope='Exploratory joined saved-E diagnostics only; original timeout and missing canonical second runtime batch retained; never freeze eligible',
    original_status='failed',original_attempt='artifacts/h1-v2/scientific-pilot/pilot-fe9efe02afc2444f94bd4aa6511cb76d',
    original_inventory_sha256=readout['original_inventory_sha256'],original_raw_sha256=readout['original_raw_sha256'],
    continuation_manifest_sha256=sha(root/'continuation-manifest.json'),script_sha256=sha(Path(__file__)),
    canonical_freeze_eligible=False,numerical_experiments=0,resampling=dict(iterations=8000,seed=261022106),
    source_package_sha256=train.source_hash(),full_config_key=canonical_hash(config.to_dict()),source_execution_sha256=readout['source']['execution_sha256'],
    scientific_recipe_sha256=readout['scientific_recipe_sha256'],runtime_receipt_sha256=readout['receipt_sha256'],observations=[],failures=[])
manifest['observations'].append(observation('before saved inputs'))
save(out/'analysis-manifest.json',manifest)
try:
    assert train.source_hash()=='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c'
    assert manifest['full_config_key']=='9152a59bb3180da35f780b44a78d9eee59f7efa3e1af9677e4acf307a1bbb073'
    assert elapsed()<7363.856268376
    inputs=readout['original_members']+[dict(teacher_index=15,repetition=x['repetition'],path=x['path'],sha256=x['sha256']) for x in readout['pairs']]
    assert {(x['teacher_index'],x['repetition']) for x in inputs}=={(i,j) for i in range(16) for j in range(2)} and len(inputs)==32
    records=[];pair_means=[]
    for member in inputs:
        path=Path(member['path']);assert sha(path)==member['sha256']
        # One file decoded at a time; no original duplicated raw ledger is loaded.
        record=json.loads(path.read_text())
        assert record['status']=='completed' and not record['failures'] and record['source_sha256']==manifest['source_package_sha256']
        assert record['recipe_hash']==manifest['full_config_key'] and record['repetition']==member['repetition']
        assert set(record['methods'])=={'keep','top','random','greedy','widen_upper','ordinary_lower','ordinary_upper','static_partial'}
        assert set(record['controls'])=={'B1','B2','B3','B4','B5','B6'}
        records.append(record)
        pair_means.append(dict(member,teacher_id=record['teacher_id'],optimizer_seed=record['optimizer_seed'],
            h1_mean=float(np.mean(record['h1_rows'])),static_mean=float(np.mean(record['static_rows'])),evaluation_rows=len(record['h1_rows'])))
    assert len({r['teacher_id'] for r in records})==16
    manifest['observations'].append(observation('after saved input decode'))
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=6448447488
    tick=time.perf_counter_ns()
    summary=summarize_pilot(records,batch_runtimes=[406.949557520875],bootstrap_iterations=8000,seed=261022106,purpose='scientific')
    manifest['summarize_seconds']=(time.perf_counter_ns()-tick)/1e9
    accepted,reasons=calibration_status(summary,config.pilot.endpoint_tolerance)
    save(out/'joined-summary-diagnostic.json',dict(scope=manifest['scope'],canonical_freeze_eligible=False,summary=asdict(summary),
        calibration_accepted=accepted,calibration_reasons=reasons,missing_canonical_second_runtime_batch=True,original_producer_status='failed'))
    tick=time.perf_counter_ns();decision=choose_design(summary,json.loads(Path(config.pilot.constraints_path).read_text()))
    manifest['choose_design_seconds']=(time.perf_counter_ns()-tick)/1e9
    save(out/'joined-design-diagnostic.json',dict(scope=manifest['scope'],canonical_freeze_eligible=False,design=decision.to_dict(),
        original_producer_status='failed',missing_canonical_second_runtime_batch=True))
    teachers=[]
    for i in range(16):
        pair=pair_means[2*i:2*i+2];assert all(p['teacher_index']==i for p in pair)
        static=np.asarray([p['static_mean'] for p in pair]);mean=float(static.mean())
        teachers.append(dict(index=i,teacher_id=pair[0]['teacher_id'],h1_mean=summary.teacher_means[i],static_mean=mean,
            within_teacher_H1_variance=summary.diagnostics['within_teacher_variances'][i],
            within_teacher_static_mean_variance=float(static.var(ddof=1)),paired_row_H1_variance=summary.diagnostics['paired_row_variances'][i],
            evaluation_rows=pair[0]['evaluation_rows'],repetitions=2,exact_pair_means=pair))
    # Reuse already retained old risk readout, read only the two new E files.
    old=json.loads(Path('environment/h1-v2/genuine-pilot/terminal-summary.json').read_text())
    risks=[dict(teacher_index=a['teacher_index'],repetition=a['repetition'],normalized_E_risks=a['normalized_E_risks']) for a in old['attempts'] if a['status']=='completed']
    with np.load(Path(manifest['original_attempt'])/'teacher-15/observations.npz',allow_pickle=False) as data:
        variance=float(np.var(data['F_y']))
    for member in readout['pairs']:
        with np.load(Path(member['path']).parent/'E-predictions.npz',allow_pickle=False) as p:
            y=p['y'];row=dict(teacher_index=15,repetition=member['repetition'],normalized_E_risks={role:float(np.mean((p[role]-y)**2)/variance) for role in records[-1]['methods']})
            assert abs(row['normalized_E_risks']['random']-row['normalized_E_risks']['top']-member['h1_mean'])<1e-12
            assert abs(row['normalized_E_risks']['ordinary_upper']-row['normalized_E_risks']['static_partial']-member['static_mean'])<1e-12
            risks.append(row)
    assert len(risks)==32
    save(out/'joined-scalars-and-risks.json',dict(scope=manifest['scope'],teachers=teachers,pair_risks=risks,
        method_means={role:float(np.mean([x['normalized_E_risks'][role] for x in risks])) for role in records[-1]['methods']}))
    save(root/'joined-input-members.json',dict(scope=manifest['scope'],members=inputs,counts=dict(teachers=16,repetitions=2,pairs=32,roles=256,B6_probes=2080),
        original_member_count=30,new_member_count=2,original_raw_status='failed',canonical_freeze_eligible=False))
    manifest['status']='completed'
    print(json.dumps(dict(scope=manifest['scope'],h1_mean=summary.h1_mean,h1_interval=summary.h1_interval,static_mean=summary.static_mean,
        endpoint_mc_error=summary.endpoint_mc_error,sd_interval=summary.sd_interval,calibration_accepted=accepted,reasons=reasons,branch=decision.branch)),flush=True)
except BaseException as error:
    manifest['status']='failed';manifest['failures'].append(f'{type(error).__name__}: {error}');raise
finally:
    manifest['observations'].append(observation('final saved diagnostics'));manifest['aggregate_elapsed_seconds']=elapsed()
    manifest['saved_process_seconds']=(time.perf_counter_ns()-local_started)/1e9
    manifest['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save(out/'analysis-manifest.json',manifest)
