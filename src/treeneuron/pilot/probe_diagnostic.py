"""Opt-in exploratory short/long allocation diagnostic, separate from calibration.

E is saved after the fitting graph has completed. The fixed cohort and all
resource decisions depend on execution evidence, never on E prediction errors.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.stats import t as student_t

from treeneuron import train
from treeneuron.artifacts import _atomic_json
from treeneuron.backend import lock_hash, runtime_hash
from treeneuron.config import canonical_hash, resolve_experiment
from treeneuron.experiments.paired import audit_pairing, file_hash, write_json
from treeneuron.locators import locate
from .calibrate import teacher_inputs, repetition_config, run_repetition, _runtime_preflight

METHODS = ('keep','top','random','greedy','greedy_long','widen_upper',
           'ordinary_lower','ordinary_upper','static_partial')
DIFFERENCES = {'random_minus_greedy_long':('random','greedy_long'),
    'greedy_short_minus_greedy_long':('greedy','greedy_long'),
    'ordinary_upper_minus_greedy_long':('ordinary_upper','greedy_long'),
    'random_minus_top':('random','top')}
SEEDS = {'teacher','data','optimizer','allocation','contrast','reference'}
_SPEC_KEYS = {'schema_version','base_config','teachers','repetitions','short_probe_updates',
    'long_probe_updates','evaluation_rows','cost_seeds','study_seeds','bootstrap','resources'}
METRICS = ['mse','rmse','mae','normalized_mse']
POLICIES = dict(purpose='exploratory allocation diagnostic; not variance calibration or confirmation',
    stopping_modes={'greedy':'fixed_updates','greedy_long':'patience'},
    selection='F fits; G selects/ranks; V unused; E evaluates completed procedures; T absent',
    failure='retain ordered incomplete cohort; zero retries; no E-dependent selection or expansion',
    aggregation='repetitions within teacher, then teachers equally; zero and negative effects retained',
    bootstrap='512000 whole-teacher paired resamples; percentile 95%; Student-t sensitivity descriptive')


def _keys(value, keys, label):
    if not isinstance(value,dict) or set(value)!=set(keys):
        raise ValueError(f'Invalid {label} keys')


def _positive(value, label):
    if type(value) is not int or value<=0: raise ValueError(f'{label} must be a positive integer')


def _validate_spec(spec):
    _keys(spec,_SPEC_KEYS,'spec')
    if type(spec['schema_version']) is not int or spec['schema_version']!=1:
        raise ValueError('Unsupported diagnostic spec version')
    for key in ('teachers','repetitions','short_probe_updates','long_probe_updates','evaluation_rows'):
        _positive(spec[key],key)
    for name in ('cost_seeds','study_seeds'):
        _keys(spec[name],SEEDS,name)
        if any(type(v) is not int or v<0 for v in spec[name].values()): raise ValueError('Invalid seeds')
    if set(spec['cost_seeds'].values())&set(spec['study_seeds'].values()):
        raise ValueError('Cost and study seed namespaces must be disjoint')
    _keys(spec['bootstrap'],('iterations','seed','confidence'),'bootstrap')
    if spec['bootstrap']!={'iterations':512000,'seed':261031106,'confidence':.95}:
        raise ValueError('Fixed diagnostic bootstrap required')
    _keys(spec['resources'],('safety_factor','disk_reserve_bytes','rss_cap_bytes',
        'severe_available_percent','automatic_retries'),'resources')
    if spec['resources']!=dict(safety_factor=1.5,disk_reserve_bytes=10*1024**3,
            rss_cap_bytes=6*1024**3,severe_available_percent=10,automatic_retries=0):
        raise ValueError('Fixed diagnostic resource policy required')
    config=resolve_experiment(spec['base_config'])
    if (config.pilot is None or config.pilot.purpose!='engineering' or config.data.n_test is not None
            or config.model.coordinates!='selective' or config.model.depth!=1
            or config.backend.device!='cpu' or config.backend.cpu_threads!=1
            or config.backend.interop_threads!=1):
        raise ValueError('Diagnostic requires an engineering-only CPU1 selective parent config without T')
    if not 0<spec['short_probe_updates']<=spec['long_probe_updates']<=config.training.continuation_updates:
        raise ValueError('Invalid short/long probe allowances')
    return copy.deepcopy(spec)


def _read_json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=unique)


def load_spec(path: Path) -> dict:
    path=Path(path)
    spec=_read_json(path)
    if isinstance(spec,dict) and isinstance(spec.get('base_config'),str):
        spec['base_config']=_read_json((path.parent/spec['base_config']).resolve())
    if isinstance(spec,dict) and isinstance(spec.get('base_config'),dict):
        spec['base_config']=resolve_experiment(spec['base_config']).to_dict()
    return _validate_spec(spec)


def _config(spec,phase):
    value=copy.deepcopy(spec['base_config'])
    value['seeds'].update(spec[phase+'_seeds'])
    value['pilot'].update(evaluation_rows=spec['evaluation_rows'],probe_updates=spec['short_probe_updates'])
    return resolve_experiment(value)


def _git(*args):
    return subprocess.check_output(['git',*args],text=True).strip()


def _provenance(spec):
    receipt=locate(_config(spec,'study').execution.backend_report,producer=True)
    return dict(source=train.source_hash(),lock=lock_hash(),runtime=runtime_hash(),
        receipt=file_hash(receipt),commit=_git('rev-parse','HEAD'),execution=canonical_hash(spec))


def _same_execution(expected,current):
    # A documented evidence-only commit between cost and cohort is permitted.
    # HEAD is retained in each record; all actual execution identities must match.
    if {k:v for k,v in expected.items() if k!='commit'}!={k:v for k,v in current.items() if k!='commit'}:
        raise ValueError('Diagnostic execution source/config/lock/runtime/receipt drift')


@contextmanager
def _awake():
    process=None
    if sys.platform=='darwin':
        process=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try: yield
    finally:
        if process is not None:
            process.terminate(); process.wait()


def _resource_snapshot(path):
    rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    value=dict(free_disk_bytes=shutil.disk_usage(path).free,peak_rss_bytes=rss,
        available_percent=None,swapouts=None,counter_reason=None)
    try:
        if sys.platform=='darwin':
            pressure=subprocess.check_output(['memory_pressure','-Q'],text=True)
            vm=subprocess.check_output(['vm_stat'],text=True)
            available=re.search(r'free percentage:\s*(\d+)%',pressure)
            swapouts=re.search(r'Swapouts:\s*(\d+)',vm)
            if available is None or swapouts is None: raise ValueError('Cannot parse native counters')
            value.update(available_percent=int(available[1]),swapouts=int(swapouts[1]))
        else:
            memory=Path('/proc/meminfo').read_text()
            counters=Path('/proc/vmstat').read_text()
            available=int(re.search(r'MemAvailable:\s*(\d+)',memory)[1])
            total=int(re.search(r'MemTotal:\s*(\d+)',memory)[1])
            value.update(available_percent=100*available/total,
                swapouts=int(re.search(r'^pswpout (\d+)',counters,re.M)[1]))
    except (OSError,ValueError,TypeError,subprocess.SubprocessError) as error:
        value['counter_reason']=f'{type(error).__name__}: {error}'
    return value


def _resource_guard(output,policy,ledger,projected_bytes=0):
    previous=None
    def check():
        nonlocal previous
        observation=_resource_snapshot(output)
        observation['elapsed_ns']=time.perf_counter_ns()-ledger.get('started_ns',time.perf_counter_ns())
        ledger['resource_observations'].append(observation)
        before=previous; previous=observation['swapouts']
        expected=len(ledger.get('expected_members',[]))
        completed=sum(entry.get('status')=='completed' for entry in ledger.get('members',[]))
        remaining=projected_bytes if not expected else math.ceil(projected_bytes*(expected-completed)/expected)
        if observation['free_disk_bytes']<remaining+policy['disk_reserve_bytes']:
            raise RuntimeError('Diagnostic disk reserve/projection guard failed')
        if observation['peak_rss_bytes']>policy['rss_cap_bytes']:
            raise RuntimeError('Diagnostic process RSS cap exceeded')
        if (observation['available_percent'] is not None
                and observation['available_percent']<=policy['severe_available_percent']
                and before is not None and previous is not None and previous>before):
            raise RuntimeError('Diagnostic severe memory pressure with new swapouts')
    return check


def _safe(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Artifact path escapes diagnostic root')
    return path


def _seal(path):
    write_json(path.with_name(path.stem+'-seal.json'),dict(sha256=file_hash(path)))


def _read_sealed(path):
    if _read_json(path.with_name(path.stem+'-seal.json'))!={'sha256':file_hash(path)}:
        raise ValueError(f'{path.name} hash mismatch')
    return _read_json(path)


def _observations(root,index,teacher,data,e):
    directory=root/f'teacher-{index}'
    directory.mkdir()
    write_json(directory/'teacher.json',teacher.to_dict())
    write_json(directory/'splits.json',data.manifest.to_dict())
    path=directory/'observations.npz'
    np.savez_compressed(path,E_X=e.X,E_y=e.y,E_rows=e.row_id,
        **{f'{part}_{key}':getattr(getattr(data,part),key) for part in 'FG' for key in ('X','y','row_id')})
    return path


def _bind_member(root,entry,pair):
    pair_root=_safe(root,entry['pair_path'])
    # Pair/stage events, checkpoints, curves, score tables and both full rankings
    # stay in place. The cumulative ledger contains references, never E row arrays.
    artifacts={str(p.relative_to(pair_root)):file_hash(p) for p in sorted(pair_root.rglob('*')) if p.is_file()}
    write_json(pair_root/'artifacts.json',artifacts)
    entry.update(pair_sha256=file_hash(pair_root/'pair.json'),
        predictions_sha256=file_hash(pair_root/'E-predictions.npz'),
        artifacts_sha256=file_hash(pair_root/'artifacts.json'),
        costs=copy.deepcopy(pair['costs']), status='completed')


def _run_members(spec,phase,output,provenance,decision=None):
    started=time.perf_counter_ns()
    teachers=1 if phase=='cost' else spec['teachers']
    repetitions=1 if phase=='cost' else spec['repetitions']
    expected=[dict(teacher_index=i,repetition=j) for i in range(teachers) for j in range(repetitions)]
    ledger=dict(schema_version=1,phase=phase,status='running',spec=spec,provenance=provenance,
        expected_members=expected,members=[],failures=[],resource_observations=[],started_ns=started,
        costs=dict(aggregation='one enclosing clock; nested clocks are never added'),
        policies=POLICIES,metrics=METRICS)
    if decision is not None: ledger['resource_decision']=decision
    _atomic_json(output/'members.json',ledger,replace=True)
    guard=_resource_guard(output,spec['resources'],ledger,
        0 if decision is None else decision['forecast']['retained_bytes'])
    try:
        with _awake():
            guard()
            ledger['runtime']=_runtime_preflight(_config(spec,phase))
            for index in range(teachers):
                guard()
                teacher,data,e,tc=teacher_inputs(_config(spec,phase),index)
                obs=_observations(output,index,teacher,data,e)
                for repetition in range(repetitions):
                    entry=dict(teacher_index=index,repetition=repetition,teacher_id=data.teacher_id,
                        status='running',pair_path=f'teacher-{index}/repetition-{repetition}',
                        observations_path=str(obs.relative_to(output)),observations_sha256=file_hash(obs),
                        teacher_sha256=file_hash(obs.parent/'teacher.json'),splits_sha256=file_hash(obs.parent/'splits.json'))
                    ledger['members'].append(entry)
                    _atomic_json(output/'members.json',ledger,replace=True)
                    tick=time.perf_counter_ns()
                    try:
                        guard()
                        pair=run_repetition(repetition_config(tc,repetition),data,e,output/entry['pair_path'],
                            canonical_hash(spec),reference_probe_updates=spec['long_probe_updates'],resource_guard=guard)
                        if pair['status']!='completed' or set(pair['methods'])!=set(METHODS):
                            raise ValueError('Incomplete nine-role diagnostic comparison')
                        _same_execution(provenance,_provenance(spec))
                        _bind_member(output,entry,pair)
                    except BaseException as error:
                        entry.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                            failure=f'{type(error).__name__}: {error}')
                        pair_root=output/entry['pair_path']
                        if pair_root.exists():
                            try:
                                failed={str(p.relative_to(pair_root)):file_hash(p) for p in sorted(pair_root.rglob('*')) if p.is_file()}
                                write_json(pair_root/'failed-artifacts.json',failed)
                                entry['failed_artifacts_sha256']=file_hash(pair_root/'failed-artifacts.json')
                            except BaseException as publication:
                                error.add_note(f'Failed-artifact binding also failed: {publication}')
                        raise
                    finally:
                        entry['enclosing_ns']=time.perf_counter_ns()-tick
                        _atomic_json(output/'members.json',ledger,replace=True)
                    guard()
            ledger['status']='completed'
    except BaseException as error:
        ledger['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        ledger['failures'].append(f'{type(error).__name__}: {error}')
        raise
    finally:
        ledger['enclosing_ns']=time.perf_counter_ns()-started
        ledger['costs']['boundary']='generation, fresh parent, score, both exhaustive searches, nine final roles, E evaluation, artifact hashing and publication; nested clocks excluded from sums'
        ledger['peak_rss_bytes']=max([o['peak_rss_bytes'] for o in ledger['resource_observations']] or [0])
        ledger['retained_bytes']=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        _atomic_json(output/'members.json',ledger,replace=True)
        _seal(output/'members.json')
    return ledger


def measure(spec: dict, output: Path) -> dict:
    spec=_validate_spec(spec); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    provenance=_provenance(spec)
    lock=dict(schema_version=1,phase='cost',spec=spec,provenance=provenance,policies=POLICIES,metrics=METRICS,wrapper_arguments=dict(command='measure',output=str(output),reference_probe_updates=spec['long_probe_updates']))
    write_json(output/'study-lock.json',lock); _seal(output/'study-lock.json')
    ledger=_run_members(spec,'cost',output,provenance)
    comparisons=spec['teachers']*spec['repetitions']
    safety=spec['resources']['safety_factor']
    forecast=dict(comparisons=comparisons,safety_factor=safety,
        seconds=ledger['enclosing_ns']/1e9*comparisons*safety,
        retained_bytes=math.ceil(ledger['retained_bytes']*comparisons*safety),
        uncertainty='One full engineering comparison only; runtime, eligibility, stopping and retained size vary across problems. This is not a time kill or a precision guarantee.')
    decision=dict(schema_version=1,spec_sha256=canonical_hash(spec),provenance=provenance,
        measurement_path=str(output.resolve()),measurement_sha256=file_hash(output/'members.json'),
        measurement_lock_sha256=file_hash(output/'study-lock.json'),forecast=forecast,
        selection_basis='Measured resources only; no E accuracy, effect signs or G ranking changes select budgets/cohort')
    observation=_resource_snapshot(output)
    decision['resources']=observation
    decision['allowed']=(observation['free_disk_bytes']>=forecast['retained_bytes']+spec['resources']['disk_reserve_bytes']
        and observation['peak_rss_bytes']<=spec['resources']['rss_cap_bytes'])
    write_json(output/'resource-decision.json',decision); _seal(output/'resource-decision.json')
    return ledger


def run(spec: dict, resource_decision: Path, output: Path) -> dict:
    spec=_validate_spec(spec); decision=_read_sealed(Path(resource_decision))
    if decision['spec_sha256']!=canonical_hash(spec): raise ValueError('Resource decision spec drift')
    if decision['allowed'] is not True: raise ValueError('Resource decision did not allow fixed cohort')
    measured=Path(decision['measurement_path'])
    if (file_hash(measured/'members.json')!=decision['measurement_sha256']
            or file_hash(measured/'study-lock.json')!=decision['measurement_lock_sha256']):
        raise ValueError('Resource decision measurement hash mismatch')
    measurement=_read_sealed(measured/'members.json')
    if measurement['phase']!='cost' or measurement['status']!='completed' or len(measurement['members'])!=1:
        raise ValueError('Resource decision requires a complete separate cost comparison')
    provenance=_provenance(spec); _same_execution(decision['provenance'],provenance)
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    lock=dict(schema_version=1,phase='study',spec=spec,provenance=provenance,policies=POLICIES,
        metrics=METRICS,wrapper_arguments=dict(command='run',output=str(output),resource_decision=str(Path(resource_decision).resolve()),reference_probe_updates=spec['long_probe_updates']),resource_decision_sha256=file_hash(Path(resource_decision)),resource_decision=decision)
    write_json(output/'study-lock.json',lock); _seal(output/'study-lock.json')
    return _run_members(spec,'study',output,provenance,decision)


def _audit_pair(root,pair):
    if pair['status']!='completed' or set(pair['methods'])!=set(METHODS):
        raise ValueError('Require a complete nine-role fitting graph')
    pairing=audit_pairing(pair)
    score=_read_json(root/'score-table.json')
    contrasts={unit:q for unit,q in zip(score['ids'],score['contrasts'])}
    allocations=[pair['methods'][name]['allocation'] for name in ('top','random','greedy','greedy_long')]
    reference=allocations[0]
    for allocation in allocations:
        for key in ('requested_count','effective_count','table_hash','seed'):
            if allocation[key]!=reference[key]: raise ValueError('Allocation pairing mismatch')
        if allocation['table_hash']!=score['table_hash']: raise ValueError('Original score binding mismatch')
        if any(q!=contrasts[unit] for unit,q in zip(allocation['ids'],allocation['contrasts_float64'])):
            raise ValueError('Original per-ID contrast mismatch')
    probes=[]
    for name,key,mode in (('greedy','greedy_report','fixed_updates'),('greedy_long','greedy_long_report','patience')):
        path=Path(pair[key])
        if not path.is_absolute(): path=root/path
        if not path.resolve().is_relative_to(root.resolve()): raise ValueError('Probe report outside pair')
        probe=_read_json(path)
        if (probe['complete'] is not True or probe['stopping_mode']!=mode
                or probe['not_attempted'] or probe['score_hash']!=score['table_hash']):
            raise ValueError('Incomplete/misbound probe ranking')
        ids=[v['unit_id'] for v in probe['trials'] if v['unit_id'] is not None]
        if sorted(ids)!=sorted(probe['eligible_ids']) or len(ids)!=len(set(ids)):
            raise ValueError('Incomplete probe candidate coverage')
        if sum(v['unit_id'] is None for v in probe['trials'])!=1: raise ValueError('Missing continued keep')
        for trial in probe['trials']:
            if trial['outcome']!='completed' or not np.isfinite(trial['g_mse']): raise ValueError('Failed/nonfinite probe')
            if trial['stopping_reason'] not in ('ceiling','patience'): raise ValueError('Missing completion evidence')
            if mode=='fixed_updates' and trial['executed_updates']!=probe['probe_updates']:
                raise ValueError('Short probe allowance mismatch')
        allocation=pair['methods'][name]['allocation']
        if allocation['ids']!=probe['allocation']['ids']: raise ValueError('Joint allocation differs from ranking')
        probes.append(probe)
    if (pair['methods']['greedy_long']['original_parent_audit']['passed'] is not True
            or pair['methods']['greedy_long']['original_parent_audit']['complete_state_sha256']!=probes[0]['parent_state_sha256']):
        raise ValueError('Long final fork original-parent state binding mismatch')
    if probes[0]['parent_state_sha256']!=probes[1]['parent_state_sha256']:
        raise ValueError('Independent rankings did not start at identical original parent')
    keep=pair['methods']['keep']['executed_row_ids']
    # Independently compare raw trial row streams across the two searches and
    # against each final fork's original-parent prefix.
    streams=[]
    for probe,key in zip(probes,('greedy_report','greedy_long_report')):
        directory=Path(pair[key]).parent
        if not directory.is_absolute(): directory=root/directory
        by_id={}
        for trial in probe['trials']:
            events=[json.loads(line) for line in (directory/trial['path']/'events.jsonl').read_text().splitlines()]
            rows=[event['row_ids'] for event in events if event['type']=='update']
            if len(rows)!=trial['executed_updates']: raise ValueError('Probe raw executed row count mismatch')
            from treeneuron.data.types import array_hash
            hashes=[array_hash(np.asarray(row,dtype=np.int64)) for row in rows]
            if hashes[:min(len(hashes),len(keep))]!=keep[:min(len(hashes),len(keep))]:
                raise ValueError('Probe/final original-parent row stream mismatch')
            by_id[trial['unit_id']]=hashes
        streams.append(by_id)
    for unit in streams[0]:
        a,b=streams[0][unit],streams[1][unit]
        if a[:min(len(a),len(b))]!=b[:min(len(a),len(b))]: raise ValueError('Short/long row prefix mismatch')
    return dict(pairing=pairing,selected_ids={name:pair['methods'][name]['allocation']['ids'] for name in ('top','random','greedy','greedy_long')},
        overlap=len(set(allocations[2]['ids'])&set(allocations[3]['ids'])),
        rank_changes=[unit for unit in probes[0]['eligible_ids'] if
            sorted(probes[0]['improvements'],key=lambda u:(-probes[0]['improvements'][u],int(u))).index(str(unit))!=
            sorted(probes[1]['improvements'],key=lambda u:(-probes[1]['improvements'][u],int(u))).index(str(unit))],
        searches={name:dict(improvements=probe['improvements'],
            trials=[{key:trial[key] for key in ('unit_id','g_mse','allowance','executed_updates','selected_stage_updates','stopping_reason','hit_ceiling')} for trial in probe['trials']],
            nanoseconds=probe['nanoseconds']) for name,probe in zip(('short','long'),probes)})


def _read_member(output,entry):
    root=_safe(output,entry['pair_path'])
    if file_hash(root/'pair.json')!=entry['pair_sha256']: raise ValueError('Pair hash mismatch')
    if file_hash(root/'artifacts.json')!=entry['artifacts_sha256']: raise ValueError('Pair artifact manifest mismatch')
    for relative,digest in _read_json(root/'artifacts.json').items():
        if file_hash(_safe(root,relative))!=digest: raise ValueError('Retained artifact hash mismatch')
    obs=_safe(output,entry['observations_path'])
    if file_hash(obs)!=entry['observations_sha256']: raise ValueError('Original observations hash mismatch')
    for name,key in (('teacher.json','teacher_sha256'),('splits.json','splits_sha256')):
        if file_hash(obs.parent/name)!=entry[key]: raise ValueError('Teacher/split identity mismatch')
    pair=_read_json(root/'pair.json')
    if pair['teacher_id']!=entry['teacher_id']: raise ValueError('Pair teacher mismatch')
    if file_hash(root/'E-predictions.npz')!=entry['predictions_sha256'] or pair['E_predictions_sha256']!=entry['predictions_sha256']:
        raise ValueError('Prediction hash mismatch')
    return root,pair,obs


def _bootstrap(matrix,spec):
    n=matrix.shape[0]; rng=np.random.default_rng(spec['seed'])
    samples=np.empty((spec['iterations'],matrix.shape[1]),dtype=np.float64)
    for start in range(0,len(samples),8192):
        stop=min(start+8192,len(samples))
        indices=rng.integers(0,n,size=(stop-start,n))
        samples[start:stop]=matrix[indices].mean(axis=1)
    return np.quantile(samples,[.025,.975],axis=0)


def analyze(output: Path) -> dict:
    output=Path(output); lock=_read_sealed(output/'study-lock.json')
    spec=_validate_spec(lock['spec']); _same_execution(lock['provenance'],_provenance(spec))
    ledger=_read_sealed(output/'members.json')
    if ledger['phase']!='study' or lock['phase']!='study': raise ValueError('Cost accuracy cannot enter study analysis')
    if ledger['spec']!=spec or ledger['provenance']!=lock['provenance']: raise ValueError('Ledger/lock drift')
    expected=[dict(teacher_index=i,repetition=j) for i in range(spec['teachers']) for j in range(spec['repetitions'])]
    if ledger['expected_members']!=expected: raise ValueError('Fixed cohort membership drift')
    actual=[{key:entry[key] for key in ('teacher_index','repetition')} for entry in ledger['members']]
    if actual!=expected[:len(actual)]: raise ValueError('Members not an ordered fixed cohort prefix')
    records=[]; originals={}
    for entry in ledger['members']:
        if entry['status']!='completed': continue
        root,pair,obs_path=_read_member(output,entry)
        index=entry['teacher_index']
        if index not in originals:
            teacher,data,e,tc=teacher_inputs(_config(spec,'study'),index)
            originals[index]=(data,e,tc)
            if teacher.teacher_id!=entry['teacher_id']: raise ValueError('Original generated teacher identity mismatch')
        data,e,tc=originals[index]
        if (pair['source_sha256']!=lock['provenance']['source']
                or pair['recipe_hash']!=canonical_hash(spec)
                or pair['data_hash']!=data.manifest.assignment_hash
                or pair['optimizer_seed']!=repetition_config(tc,entry['repetition']).seeds.optimizer):
            raise ValueError('Pair source/recipe/data/optimizer identity mismatch')
        with np.load(obs_path,allow_pickle=False) as obs:
            for name,expected_array in [('E_X',e.X),('E_y',e.y),('E_rows',e.row_id)]+[
                (f'{part}_{key}',getattr(getattr(data,part),key)) for part in 'FG' for key in ('X','y','row_id')]:
                if not np.array_equal(obs[name],expected_array): raise ValueError('Original F/E observations differ')
            scale=float(np.var(obs['F_y'],ddof=0))
            if not np.isfinite(scale) or scale<=0: raise ValueError('Invalid original F variance')
        risks={}
        with np.load(root/'E-predictions.npz',allow_pickle=False) as predictions:
            if set(predictions.files)!=set(METHODS)|{'y','row_id'}: raise ValueError('Missing/unexpected saved predictions')
            if not np.array_equal(predictions['y'],e.y) or not np.array_equal(predictions['row_id'],e.row_id):
                raise ValueError('Saved E targets/row IDs differ from originals')
            for name in METHODS:
                prediction=predictions[name]
                if prediction.shape!=e.y.shape or not np.isfinite(prediction).all(): raise ValueError('Invalid saved prediction')
                error=prediction-e.y; mse=float(np.mean(error**2))
                risks[name]=dict(mse=mse,rmse=math.sqrt(mse),mae=float(np.mean(np.abs(error))),normalized_mse=mse/scale)
        diagnostics=_audit_pair(root,pair)
        records.append(dict(teacher_index=index,repetition=entry['repetition'],F_variance=scale,
            risks=risks,diagnostics=diagnostics,costs=entry['costs']))
    if not records: raise ValueError('No complete repetitions for descriptive analysis')
    teachers=sorted({r['teacher_index'] for r in records})
    methods={name:{metric:float(np.mean([np.mean([r['risks'][name][metric] for r in records if r['teacher_index']==i]) for i in teachers])) for metric in METRICS} for name in METHODS}
    differences={}; matrix=[]
    for key,(control,grown) in DIFFERENCES.items():
        values=[r['risks'][control]['normalized_mse']-r['risks'][grown]['normalized_mse'] for r in records]
        effects=[float(np.mean([v for v,r in zip(values,records) if r['teacher_index']==i])) for i in teachers]
        matrix.append(effects)
        n=len(effects); mean=float(np.mean(effects)); se=float(np.std(effects,ddof=1)/math.sqrt(n)) if n>1 else None
        differences[key]=dict(mean=mean,median=float(np.median(effects)),teacher_effects=effects,
            repetition_effects=values,repetition_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
            zero_teachers=sum(abs(v)<1e-14 for v in effects),negative_teachers=sum(v < -1e-14 for v in effects),
            student_t_95=None if se is None else [mean-float(student_t.ppf(.975,n-1))*se,mean+float(student_t.ppf(.975,n-1))*se])
    intervals=_bootstrap(np.asarray(matrix).T,spec['bootstrap'])
    for column,key in enumerate(DIFFERENCES): differences[key]['percentile_95']=intervals[:,column].tolist()
    complete=(ledger['status']=='completed' and len(records)==len(expected) and len(ledger['members'])==len(expected))
    result=dict(schema_version=1,complete_cohort=complete,status=ledger['status'],independent_teachers=len(teachers),
        teacher_indices=teachers,completed_repetitions=len(records),expected_teachers=spec['teachers'],
        expected_repetitions=spec['teachers']*spec['repetitions'],methods=methods,differences=differences,
        records=records,bootstrap=spec['bootstrap'],source_provenance=lock['provenance'],
        study_lock_sha256=file_hash(output/'study-lock.json'),members_sha256=file_hash(output/'members.json'),
        costs=dict(enclosing_ns=ledger['enclosing_ns'],retained_bytes=ledger['retained_bytes'],peak_rss_bytes=ledger['peak_rss_bytes'],
            short_search_ns=sum(r['costs'].get('probe_ns',0) for r in records),
            long_search_ns=sum(r['costs'].get('long_probe_ns',0) for r in records)),
        limitations=['Exploratory diagnostic; no superiority, equivalence, nominal-coverage or powered-confirmation claim.',
            'Restricted single-unit greedy ranking; no oracle or full ANT reproduction; validation reuse and finite stopping.',
            'Short endpoint at 320 and long 25-update grid/own endpoint are different complete procedures, not nested evaluated checkpoints.',
            'Additional search is fully charged; this is not an equal-cost practical comparison.',
            'Repetitions and bootstrap draws are not additional independent teachers.'])
    _atomic_json(output/'analysis.json',result,replace=True)
    return result


def report(output: Path, destination: Path) -> dict:
    result=analyze(output)
    lines=['# Longer-probe allocation diagnostic','',
        ('Complete fixed cohort.' if result['complete_cohort'] else 'Incomplete fixed cohort: descriptive partial evidence only; complete-result claim blocked.'),'',
        f"Independent teachers: {result['independent_teachers']}/{result['expected_teachers']}; completed repetitions: {result['completed_repetitions']}/{result['expected_repetitions']}.",
        '','Repetitions are averaged within teacher, then teachers equally. Positive differences favor the named grown method. All zero and negative effects are retained.','',
        '| Method | Original MSE | RMSE | MAE | F-normalized MSE |','|---|---:|---:|---:|---:|']
    for name,metrics in result['methods'].items(): lines.append('| '+name+' | '+' | '.join(f'{metrics[key]:.9g}' for key in METRICS)+' |')
    lines+=['','| Paired difference | Mean | Median | Teacher bootstrap 95% | Student-t sensitivity |','|---|---:|---:|---|---|']
    for name,value in result['differences'].items():
        interval=', '.join(f'{v:.9g}' for v in value['percentile_95'])
        sensitivity='unavailable' if value['student_t_95'] is None else ', '.join(f'{v:.9g}' for v in value['student_t_95'])
        lines.append(f"| {name} | {value['mean']:.9g} | {value['median']:.9g} | [{interval}] | {sensitivity} |")
    lines+=['','Teacher effects (in locked teacher order):','']
    for name,value in result['differences'].items():
        lines.append(f"- {name}: "+', '.join(f'{v:.9g}' for v in value['teacher_effects'])+f"; repetition SD {value['repetition_sd']}; zero teachers {value['zero_teachers']}; negative teachers {value['negative_teachers']}.")
    lines+=['',f"Bootstrap: {result['bootstrap']['iterations']} paired whole-teacher draws, seed {result['bootstrap']['seed']}; percentile 95% intervals are descriptive.",'',
        f"Full cohort enclosing seconds: {result['costs']['enclosing_ns']/1e9:.6f}; short search seconds: {result['costs']['short_search_ns']/1e9:.6f}; extra long search seconds: {result['costs']['long_search_ns']/1e9:.6f}. Search clocks nest inside the full comparison.",
        f"Retained bytes: {result['costs']['retained_bytes']}; peak RSS bytes: {result['costs']['peak_rss_bytes']}.",'']
    for record in result['records']:
        lines.append(f"Teacher {record['teacher_index']}, repetition {record['repetition']}: F variance {record['F_variance']:.9g}.")
        diagnostic=record['diagnostics']
        if diagnostic:
            for name,ids in diagnostic['selected_ids'].items(): lines.append(f'- {name} selected IDs: {ids}.')
            lines.append(f"- Short/long overlap: {diagnostic['overlap']}; changed rank IDs: {diagnostic['rank_changes']}.")
            for name,search in diagnostic['searches'].items():
                lines.append(f'- {name} candidate G improvements: '+', '.join(f'{unit}: {value:.9g}' for unit,value in search['improvements'].items())+'.')
                for trial in search['trials']:
                    lines.append(f"  - Candidate {trial['unit_id'] if trial['unit_id'] is not None else 'keep'}: G MSE {trial['g_mse']:.9g}; allowance {trial['allowance']}; executed {trial['executed_updates']}; selected {trial['selected_stage_updates']}; {trial['stopping_reason']}; ceiling {trial['hit_ceiling']}.")
        lines.append('')
    lines+=['Interpretation: assess longer selection against random and ordinary capacity together. A component-only opportunity supports studying a cheaper approximation; unclear gain supports investigating timing/training or narrowing this allocation procedure. These descriptive results authorize no automatic follow-up cohort.','']
    lines.extend('- '+item for item in result['limitations'])
    lines+=['',f"Study lock SHA-256: {result['study_lock_sha256']}",f"Member ledger SHA-256: {result['members_sha256']}",'']
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as stream: stream.write('\n'.join(lines))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('measure','run'):
        command=commands.add_parser(name)
        command.add_argument('--spec',type=Path,required=True)
        command.add_argument('--output',type=Path,required=True)
        if name=='run': command.add_argument('--resource-decision',type=Path,required=True)
    command=commands.add_parser('analyze'); command.add_argument('--output',type=Path,required=True)
    command=commands.add_parser('report'); command.add_argument('--output',type=Path,required=True)
    command.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='measure': result=measure(load_spec(args.spec),args.output)
    elif args.command=='run': result=run(load_spec(args.spec),args.resource_decision,args.output)
    elif args.command=='analyze': result=analyze(args.output)
    else: result=report(args.output,args.destination)
    print(json.dumps({'status':result['status'],'output':str(args.output)},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
