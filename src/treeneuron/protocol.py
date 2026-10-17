"""Scientific identity, staged readiness and immutable synthetic H1 freezes.

These are procedural safeguards for cooperative callers, not encryption against
an owner who can edit the Python process, repository, or evidence files.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .artifacts import _atomic_json, _utc
from .locators import locate, relocated
from .config import canonical_hash, canonical_json, resolve_experiment
from .pilot.design import DesignDecision, summarize_pilot, choose_design

EVALUATION_METHODS=('keep','top','random','greedy','widen_upper','ordinary_lower',
                    'ordinary_upper','static_partial')


def _keys(value, expected, name):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f'{name}: expected exactly {sorted(expected)}')


def _read(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    try:
        value = json.loads(Path(path).read_text(), object_pairs_hook=unique)
        canonical_json(value)
        return value
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Cannot read canonical evidence: {path}') from error


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(root, *args):
    try:
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError('Missing committed source/review identity') from error


def _relative(root, name):
    if type(name) is not str or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Evidence paths must be repository-relative without traversal')
    root = locate(root)
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Evidence path escapes repository or is a symlink')
    return path


def _reference(root, value):
    _keys(value, ('path', 'sha256'), 'file reference')
    path = _relative(root, value['path'])
    if not path.is_file() or _hash(path) != value['sha256']:
        raise ValueError(f'Input hash drift: {value["path"]}')
    return path


def _ref(root, path):
    return dict(path=str(Path(path).resolve().relative_to(root.resolve())), sha256=_hash(path))


def _execution(name):
    p = Path(name)
    if p.parts[0] in ('docs', 'audit', 'artifacts', 'environment', 'pilot', 'protocol', '.superpowers'):
        return False
    return (p.parts[0] in ('src', 'tests', 'configs', 'analysis', 'scripts', 'locks')
            or p.suffix in ('.py', '.sh', '.ipynb', '.lock')
            or p.name in ('pyproject.toml', 'requirements.txt', 'uv.lock'))


def execution_snapshot(root: Path, pinned_commit: str) -> dict:
    """Hash committed execution/scientific files; docs/generated outputs excluded.

    The pinned tree, index and working tree must agree for every covered file.
    Untracked (even ignored) overrides in execution directories are rejected.
    """
    root = Path(root).resolve()
    pinned = _git(root, 'rev-parse', f'{pinned_commit}^{{commit}}').decode().strip()
    checkout = _git(root, 'rev-parse', 'HEAD').decode().strip()
    tracked = set(_git(root, 'ls-files', '-z').decode().split('\0')) - {''}
    head = set(_git(root, 'ls-tree', '-r', '--name-only', pinned).decode().splitlines())
    selected = {p for p in tracked if _execution(p)}
    if selected != {p for p in head if _execution(p)}:
        raise ValueError('Execution file set drift from pinned commit')
    untracked = set(_git(root, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')) - {''}
    # Git ignored files cannot override imports/configuration unnoticed.
    for directory in ('src', 'configs', 'analysis', 'scripts', 'locks'):
        for p in (root / directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                if str(p.relative_to(root)) not in tracked: untracked.add(str(p.relative_to(root)))
    for p in root.iterdir():
        if p.is_file() and p.name not in tracked and _execution(p.name): untracked.add(p.name)
    if any(_execution(p) for p in untracked): raise ValueError('Untracked executable/scientific input')
    dirty = set(_git(root, 'diff', '--name-only').decode().splitlines())
    dirty.update(_git(root, 'diff', '--cached', '--name-only', 'HEAD').decode().splitlines())
    if dirty & selected: raise ValueError('Dirty executable/scientific inputs')
    files = {}
    for name in sorted(selected):
        path = _relative(root, name)
        if not path.is_file(): raise ValueError('Missing execution input')
        digest = _hash(path)
        if digest != hashlib.sha256(_git(root, 'show', f'{pinned}:{name}')).hexdigest():
            raise ValueError(f'Execution drift: {name}')
        files[name] = digest
    if not files: raise ValueError('Empty execution identity')
    return dict(pinned_commit=pinned, checkout_commit=checkout, files=files,
                execution_sha256=canonical_hash(files))


def _validate_runtime(config, value):
    from .backend import validate_backend_report, select_backend
    probes = value['backend_probes']
    decision = select_backend(probes['probes'], comparison_blocks=probes['comparison_blocks'])
    if decision != probes['decision'] or decision['selected_backend'] != config.backend.device:
        raise ValueError('Runtime backend decision drift')
    report = probes['probes'][config.backend.device]
    validate_backend_report(report)
    if (config.backend.cpu_threads != 1 or config.backend.interop_threads != 1
            or report['environment']['threads'] != 1 or report['environment'].get('interop_threads') != 1):
        raise ValueError('Scientific pilot requires measured explicit 1/1 runtime')


def recipe_hash(config) -> str:
    """Execution recipe shared by exploratory resource probes and scientific pilot."""
    value=config.to_dict()
    recipe={k:value[k] for k in ('model','training','growth','data','backend')}
    if config.pilot is not None:
        recipe['pilot_workload']={k:value['pilot'][k] for k in (
            'repetitions','evaluation_rows','probe_updates','bootstrap_iterations')}
    return canonical_hash(recipe)


def validate_pilot_readiness(config, path: Path, *, root: Path | None = None) -> dict:
    """PRE-pilot gate: no genuine pilot rows, final K or T commitment required."""
    root = Path(root or _git(Path.cwd(), 'rev-parse', '--show-toplevel').decode().strip()).resolve()
    config = resolve_experiment(config.to_dict())
    if config.pilot is None or config.pilot.purpose != 'scientific':
        raise ValueError('Scientific pilot config required')
    value = _read(path)
    _keys(value, ('schema_version', 'pinned_commit', 'execution_sha256', 'reviews',
                 'config_sha256', 'runtime', 'measurement', 'constraints', 'seed_plan'), 'readiness')
    if value['schema_version'] != 1: raise ValueError('Unknown readiness schema')
    snapshot = execution_snapshot(root, value['pinned_commit'])
    if value['execution_sha256'] != snapshot['execution_sha256']:
        raise ValueError('Readiness source drift')
    if canonical_hash(config.to_dict()) != value['config_sha256']:
        raise ValueError('Readiness config drift')
    _keys(value['reviews'], ('9', '10', '13'), 'review acceptance')
    for task, reference in value['reviews'].items():
        review_path = _reference(root, reference); review = _read(review_path)
        _keys(review, ('task', 'verdict', 'implementation_commit', 'execution_sha256', 'files', 'artifacts'), 'review')
        if review['task'] != task or review['verdict'] != 'accepted': raise ValueError('Review not accepted')
        if _git(root, 'show', f'HEAD:{reference["path"]}') != review_path.read_bytes():
            raise ValueError('Review must be committed unchanged')
        if not isinstance(review['artifacts'],list) or not review['artifacts']:
            raise ValueError('Committed review artifacts required')
        for artifact in review['artifacts']:
            reviewed=_reference(root,artifact)
            if _git(root,'show',f'HEAD:{artifact["path"]}')!=reviewed.read_bytes():
                raise ValueError('Review artifact must be committed unchanged')
        commit=review['implementation_commit']
        _git(root,'merge-base','--is-ancestor',commit,'HEAD')
        names=_git(root,'ls-tree','-r','--name-only',commit).decode().splitlines()
        original={name:hashlib.sha256(_git(root,'show',f'{commit}:{name}')).hexdigest()
                  for name in names if _execution(name)}
        if canonical_hash(original)!=review['execution_sha256'] or not review['files']:
            raise ValueError('Reviewed implementation identity differs')
        for name,digest in review['files'].items():
            if (name not in original or name not in snapshot['files']
                    or type(digest) is not str or len(digest)!=64
                    or original[name]!=digest or snapshot['files'][name]!=digest):
                raise ValueError('Reviewed implementation file drift')
    runtime_path=_reference(root,value['runtime'])
    if _relative(root,config.execution.backend_report)!=runtime_path:
        raise ValueError('Runtime receipt differs from execution config')
    _validate_runtime(config, _read(runtime_path))
    if config.pilot.constraints_path is None or _relative(root, config.pilot.constraints_path) != _reference(root, value['constraints']):
        raise ValueError('Pilot constraints binding differs')
    expected = dict(namespace='exploratory/pilot', seeds=config.to_dict()['seeds'],
                    max_teachers=config.pilot.max_teachers, repetitions=config.pilot.repetitions)
    if value['seed_plan'] != expected: raise ValueError('Independent exploratory seed plan differs')
    measured = _read(_reference(root, value['measurement']))
    _keys(measured, ('measured_seconds_per_teacher','measured_bytes_per_teacher','peak_memory_bytes',
        'machine_memory_bytes','reserved_memory_bytes','free_disk_bytes','max_seconds','max_storage_bytes',
        'retries','failure_policy','source_sha256','samples'), 'measured resource forecast')
    if measured['source_sha256'] != snapshot['execution_sha256'] or measured['failure_policy'] != 'retain_and_block':
        raise ValueError('Measured source/failure policy differs')
    for key, number in measured.items():
        if key in ('source_sha256','failure_policy','samples'): continue
        if type(number) not in (int,float) or (number < 0 if key=='retries' else number <= 0):
            raise ValueError('Positive measured resource bounds required')
    if type(measured['retries']) is not int: raise ValueError('Finite integer retry count required')
    if not isinstance(measured['samples'],list) or not measured['samples']:
        raise ValueError('Raw measured resource samples required')
    samples=[]
    for reference in measured['samples']:
        sample=_read(_reference(root,reference))
        _keys(sample,('status','teacher_count','enclosing_seconds','retained_bytes','peak_memory_bytes',
            'recipe_sha256','runtime_sha256','source_sha256'),'resource sample')
        if (sample['status']!='completed' or sample['recipe_sha256']!=recipe_hash(config)
                or sample['runtime_sha256']!=value['runtime']['sha256']
                or sample['source_sha256']!=snapshot['execution_sha256']):
            raise ValueError('Measured sample recipe/runtime/source differs')
        if type(sample['teacher_count']) is not int or sample['teacher_count']<1 or any(
                type(sample[k]) not in (int,float) or sample[k]<=0
                for k in ('enclosing_seconds','retained_bytes','peak_memory_bytes')):
            raise ValueError('Positive measured samples required')
        samples.append(sample)
    if (measured['measured_seconds_per_teacher']!=max(x['enclosing_seconds']/x['teacher_count'] for x in samples)
            or measured['measured_bytes_per_teacher']!=max(x['retained_bytes']/x['teacher_count'] for x in samples)
            or measured['peak_memory_bytes']!=max(x['peak_memory_bytes'] for x in samples)):
        raise ValueError('Resource forecast must derive from raw measured samples')
    count = config.pilot.max_teachers
    forecast = dict(teachers=count, seconds=count*measured['measured_seconds_per_teacher']*(1+measured['retries']),
                    storage_bytes=count*measured['measured_bytes_per_teacher']*(1+measured['retries']))
    if (forecast['seconds'] > min(config.pilot.max_seconds, measured['max_seconds'])
            or forecast['storage_bytes'] > min(measured['max_storage_bytes'], measured['free_disk_bytes'])
            or measured['peak_memory_bytes'] > measured['machine_memory_bytes']-measured['reserved_memory_bytes']):
        raise ValueError('Pilot forecast exceeds measured resource caps')
    if config.pilot.initial_teachers < 8 or config.pilot.repetitions < 2:
        raise ValueError('Scientific pilot requires at least eight teachers and two repetitions')
    return dict(readiness_sha256=_hash(path), snapshot=snapshot, forecast=forecast)


def pilot_seal(directory: Path) -> dict:
    """Seal raw pilot and all retained child/data files, excluding derived reports."""
    directory=Path(directory)
    excluded={'raw-seal.json','summary.json','design_decision.json','resource_forecast.json','design_decision.md'}
    files={str(p.relative_to(directory)):_hash(p) for p in sorted(directory.rglob('*'))
           if p.is_file() and not (p.parent==directory and p.name in excluded)}
    raw=_read(directory/'raw.json')
    return dict(raw_sha256=_hash(directory/'raw.json'),
                source_sha256=raw['provenance']['source_sha256'],files=files)


def _validate_pilot_graph(record, directory, config):
    """Trace the existing pilot producer's roles to selected retained checkpoints.

    Descriptor/payload hashes and JSON/event provenance are read without loading
    tensor state, predicting, or consuming any evaluation labels. Probe losers
    belong to the coordinator's graph, not the top-level child registration list.
    """
    from .checkpoint import _descriptor
    from .data.types import array_hash
    from .experiments.paired import audit_pairing
    import numpy as np

    directory=Path(directory).resolve()
    def contained(base, value):
        path=Path(value)
        path=(path if path.is_absolute() else base/path).resolve()
        if not path.is_relative_to(base.resolve()): raise ValueError('Pilot control path escapes enclosing run')
        return path

    def completed(path, run_id):
        manifest=_read(path/'manifest.json')
        identity=manifest['identity']
        if (identity['run_id']!=run_id or identity['status']!='completed'
                or identity['phase']!='exploratory' or manifest['data']['test_labels_accessed'] is not False):
            raise ValueError('Incomplete or protected pilot control/probe')
        return manifest

    def selected(path, manifest):
        method=manifest['method']
        if method['weight_checkpoint']!='checkpoints/best.json':
            raise ValueError('Pilot control lacks selected checkpoint')
        pointer,descriptor,_=_descriptor(path/'checkpoints/best.json')
        state=descriptor['state']; stage=state['stage']; identities=state['identity_hashes']
        if (pointer['sha256']!=method['weight_hash'] or stage['complete'] is not True
                or descriptor['best_state'] is not None or state['config']!=method['resolved_config']
                or canonical_hash(state['config'])!=manifest['identity']['config_hash']
                or identities['code']!=manifest['identity']['code_revision']
                or identities['lock']!=manifest['identity']['lock_hash']
                or identities['split']!=record['data_hash']
                or any(identities[k]!=manifest['data'][v] for k,v in
                    (('raw','raw_hash'),('split','split_hash'),('transform','transform_hash')))):
            raise ValueError('Pilot selected checkpoint identity/completion differs')
        events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
        updates=[e for e in events if e['type']=='update']
        selections=[e for e in events if e['type']=='selection']
        if (len(selections)!=1 or selections[0]['selected_update']!=state['update']
                or selections[0]['executed_updates']!=stage['updates_completed']
                or len(updates)!=stage['updates_completed'] or selections[0]['g_mse']!=stage['best_mse']):
            raise ValueError('Pilot checkpoint differs from completed stage selection')
        return state,events,updates

    try:
        children={}
        for child in record['child_runs']:
            # Top-level child locators are emitted relative to the producer CWD;
            # stage/trial locators below are relative to their enclosing run.
            path=contained(directory,locate(child['path'], producer=True).resolve()); run_id=child['run_id']
            if run_id in children or path in [entry[0] for entry in children.values()]:
                raise ValueError('Duplicate pilot control registration')
            children[run_id]=(path,completed(path,run_id),child['growth_metadata'])
        parent_id=record['parent_run_id']; parent_path,parent_manifest,parent_growth=children[parent_id]
        parent,_,_=selected(parent_path,parent_manifest)
        if (parent['identity_hashes']!=record['parent_identity_hashes']
                or parent['identity_hashes']['code']!=record['source_sha256']
                or _hash(parent_path/'checkpoints/best.json')!=record['parent_checkpoint_sha256']
                or parent['growth_metadata'] or parent_growth or parent['stage']['kind']!='base'
                or parent['stage']['policy']['max_updates']!=config.training.base_updates
                or parent['update']!=record['parent_pairing']['update']):
            raise ValueError('Pilot shared parent checkpoint differs')
        used={parent_id}
        for name,method in record['methods'].items():
            warm=name in ('keep','top','random','greedy','widen_upper')
            stages=method['stages']; recipe=method['recipe']
            variant=dict(top='adam',widen_upper='widen',ordinary_lower='lower',ordinary_upper='upper',static_partial='partial')
            if (len(stages)!=(1 if warm else 2)
                    or (name in variant and recipe.get('variant')!=variant[name])
                    or (name=='static_partial' and recipe.get('coordinates')!='centered')
                    or (name in ('keep','random','greedy') and recipe)):
                raise ValueError('Pilot control stage/recipe differs from role')
            adapter=dict(method_id=method['method_id'],recipe={k:v for k,v in recipe.items() if k!='experiment'})
            if warm and (method['parent_run_id']!=parent_id
                    or method['parent_checkpoint_sha256']!=record['parent_checkpoint_sha256']):
                raise ValueError('Pilot control shared parent linkage differs')
            previous=parent if warm else None
            for index,stage in enumerate(stages):
                run_id=stage['run_id']
                if run_id in used: raise ValueError('Pilot control stage reused by another role')
                path,manifest,growth=children[run_id]; used.add(run_id)
                if contained(directory,stage['path'])!=path:
                    raise ValueError('Pilot control stage/run path differs')
                state,events,updates=selected(path,manifest)
                expected_allowance=config.training.continuation_updates if warm or index else config.training.base_updates
                if (any(state['identity_hashes'][k]!=parent['identity_hashes'][k]
                        for k in ('code','lock','raw','split','transform','runtime'))
                        or state['growth_metadata'].get('causal_adapter')!=adapter or growth.get('causal_adapter')!=adapter
                        or state['stage']['kind']!=('continuation' if warm or index else 'base')
                        or state['stage']['policy']['max_updates']!=expected_allowance
                        or stage['max_updates']!=expected_allowance
                        or stage['checkpoint_interval']!=state['stage']['policy']['checkpoint_interval']
                        or stage['selected_update']!=state['update']
                        or stage['executed_updates']!=state['stage']['updates_completed']
                        or stage['evaluations']!=[e for e in events if e['type']=='evaluation']
                        or stage['training_loss']!=[dict(update=e['update'],value=e['loss_standardized_mse']) for e in updates]
                        or stage['row_hashes']!=[array_hash(np.array(e['row_ids'],dtype=np.int64)) for e in updates]):
                    raise ValueError('Pilot control stage/checkpoint provenance differs')
                if previous is not None and (state['stage']['start_update']!=previous['update']
                        or any(state['identity_hashes'][k]!=previous['identity_hashes'][k]
                               for k in ('code','lock','raw','split','transform','runtime'))):
                    raise ValueError('Pilot control continuation linkage differs')
                previous=state
            if (method['selected_g_mse']!=state['stage']['best_mse']
                    or method['executed_row_ids']!=stages[-1]['row_hashes']):
                raise ValueError('Pilot control selected result differs')
        audit_pairing(record)
        greedy_path=contained(directory,locate(record['greedy_report'], producer=True).resolve()); coordinator_path=greedy_path.parent
        coordinator_ids=[key for key,value in children.items() if value[0]==coordinator_path]
        if len(coordinator_ids)!=1: raise ValueError('Pilot probe coordinator missing')
        coordinator_id=coordinator_ids[0]
        if coordinator_id in used or children[coordinator_id][1]['method']['scope']!='coordinator':
            raise ValueError('Pilot probe coordinator role differs')
        used.add(coordinator_id)
        if used!=set(children): raise ValueError('Unlinked pilot control child')
        greedy=_read(greedy_path)
        parent_pointer,parent_descriptor,_=_descriptor(coordinator_path/'parent.json')
        allocation=record['methods']['greedy']['allocation']
        if (greedy['complete'] is not True or greedy['method']!='B6' or greedy['not_attempted']
                or greedy['selection_split']!='G' or greedy['V_accessed'] is not False or greedy['T_accessed'] is not False
                or greedy['probe_updates']!=config.pilot.probe_updates
                or greedy['parent_checkpoint_sha256']!=parent_pointer['sha256']
                or parent_descriptor['state']!=parent or greedy['parent_identity_hashes']!=parent['identity_hashes']
                or any(greedy['allocation'][k]!=allocation[k] for k in ('ids','table_hash','requested_count','effective_count'))):
            raise ValueError('Pilot probe graph/parent/allocation differs')
        score=greedy['score_artifact']
        if _hash(contained(coordinator_path,score['path']))!=score['sha256']:
            raise ValueError('Pilot probe score artifact differs')
        expected={'keep':None,**{f'unit_{unit}':unit for unit in greedy['eligible_ids']}}
        if (len(expected)!=len(greedy['trials']) or {t['candidate'] for t in greedy['trials']}!=set(expected)):
            raise ValueError('Pilot probe trials missing or duplicated')
        probe_paths=set(); probe_ids=set()
        for trial in greedy['trials']:
            path=contained(coordinator_path,trial['path'])
            if path in probe_paths or trial['run_id'] in probe_ids|used:
                raise ValueError('Pilot probe run reused')
            probe_paths.add(path); probe_ids.add(trial['run_id'])
            manifest=completed(path,trial['run_id']); state,_,_=selected(path,manifest)
            metadata=state['growth_metadata']['greedy_probe']
            if (trial['unit_id']!=expected[trial['candidate']] or trial['method_id']!='B6'
                    or trial['outcome']!='completed' or trial['executed_updates']!=config.pilot.probe_updates
                    or state['stage']['policy']['max_updates']!=config.pilot.probe_updates
                    or state['stage']['kind']!='continuation' or state['stage']['start_update']!=parent['update']
                    or any(state['identity_hashes'][k]!=parent['identity_hashes'][k]
                           for k in ('code','lock','raw','split','transform','runtime'))
                    or trial['executed_updates']!=state['stage']['updates_completed']
                    or trial['selected_update']!=state['update'] or trial['g_mse']!=state['stage']['best_mse']
                    or manifest['identity']['coordinator_run_id']!=coordinator_id
                    or metadata['coordinator_run_id']!=coordinator_id or metadata['candidate']!=trial['candidate']
                    or metadata['parent_checkpoint_sha256']!=greedy['parent_checkpoint_sha256']
                    or metadata['score_hash']!=greedy['score_hash']):
                raise ValueError('Pilot probe selected checkpoint linkage differs')
    except (KeyError,TypeError,OSError) as error:
        raise ValueError('Missing or invalid pilot control stage/checkpoint evidence') from error


def eligible_design(sources: Mapping[str, Any]) -> DesignDecision:
    """POST-pilot gate: recompute numeric design from sealed completed raw evidence."""
    from .train import source_hash
    from .data.types import array_hash
    import numpy as np
    _keys(sources, ('root','pinned_commit','readiness','raw','raw_seal','scientific','cohort'), 'sources')
    root = locate(sources['root']).resolve()
    raw_path = _reference(root, sources['raw']); raw = _read(raw_path)
    seal_path=_reference(root,sources['raw_seal'])
    if seal_path!=raw_path.with_name('raw-seal.json'):
        raise ValueError('Raw seal location differs')
    seal=_read(seal_path)
    if seal!=pilot_seal(raw_path.parent):
        raise ValueError('Input hash drift in sealed raw pilot/predictions')
    if raw['provenance']['source_sha256'] != source_hash(): raise ValueError('Pilot package source drift')
    if raw['status'] != 'completed' or raw['purpose'] != 'scientific' or raw['failed_pairs'] != 0:
        raise ValueError('Completed genuine pilot required')
    config = resolve_experiment(raw['config'])
    readiness_path = _reference(root, sources['readiness'])
    ready = validate_pilot_readiness(config, readiness_path, root=root)
    if sources['pinned_commit'] != ready['snapshot']['pinned_commit'] or raw.get('readiness') != sources['readiness']:
        raise ValueError('Pilot readiness/source binding differs')
    readiness = _read(readiness_path)
    if _read(_reference(root, readiness['constraints'])) != raw['constraints']:
        raise ValueError('Measured constraints drift')
    policies=raw['constraints'].get('policies',{})
    required_policies=dict(parent_updates=config.training.base_updates,
        continuation_updates=config.training.continuation_updates,patience=config.training.patience,
        validation_interval=config.training.validation_interval,logical_batch=config.training.batch_size,
        physical_batch=config.training.batch_size,probe_updates=config.pilot.probe_updates,
        bootstrap_iterations=config.pilot.bootstrap_iterations,hpo_seconds=0.,hpo_trials=0)
    if any(policies.get(k)!=v for k,v in required_policies.items()):
        raise ValueError('Numeric policies differ from completed common recipe')
    records = raw['accepted_records']
    if len(raw['attempts']) != len(records) or any(a['status'] != 'completed' for a in raw['attempts']):
        raise ValueError('Unresolved pilot attempts')
    for attempt, record in zip(raw['attempts'], records):
        pair_path = raw_path.parent / f'teacher-{attempt["teacher_index"]}' / f'repetition-{attempt["repetition"]}' / 'pair.json'
        if attempt['record'] != record or _read(pair_path) != record:
            raise ValueError('Raw/pair ledger disagreement')
        if (record['source_sha256'] != source_hash() or record['recipe_hash'] != canonical_hash(raw['config'])
                or record['V_accessed'] is not False or record['T_accessed'] is not False
                or record['enclosing_ns'] <= 0 or record['retained_bytes'] <= 0 or not record['child_runs']):
            raise ValueError('Invalid pilot pair provenance/cost/access')
        expected_methods=dict(keep='B1',top='B5',random='B4',greedy='B6',widen_upper='B2',
            ordinary_lower='B2',ordinary_upper='B2',static_partial='B3')
        methods=record.get('methods',{})
        if set(methods)!=set(expected_methods) or any(
                methods[k].get('status')!='completed' or methods[k].get('method_id')!=v
                for k,v in expected_methods.items()):
            raise ValueError('Actual completed control methods missing')
        _validate_pilot_graph(record,pair_path.parent,config)
        prediction_path=pair_path.with_name('E-predictions.npz')
        if not prediction_path.is_file() or _hash(prediction_path)!=record.get('E_predictions_sha256'):
            raise ValueError('Pilot prediction artifact hash differs')
        with np.load(pair_path.parent.parent/'observations.npz',allow_pickle=False) as data, np.load(
                prediction_path,allow_pickle=False) as predictions:
            if set(predictions.files)!=set(expected_methods)|{'y','row_id'}:
                raise ValueError('Pilot prediction methods differ')
            if (not np.array_equal(predictions['y'],data['E_y'])
                    or not np.array_equal(predictions['row_id'],data['E_rows'])
                    or data['E_y'].shape!=(config.pilot.evaluation_rows,)
                    or data['F_y'].shape!=(config.data.n_fit,)):
                raise ValueError('Pilot E/F observation identity differs')
            evaluation_hash=canonical_hash(dict(X=array_hash(data['E_X']),y=array_hash(data['E_y']),
                                                rows=array_hash(data['E_rows'])))
            if evaluation_hash!=record['evaluation_hash']: raise ValueError('Pilot E data identity differs')
            variance=float(np.var(data['F_y']))
            if not np.isfinite(variance) or variance<=0: raise ValueError('Pilot F variance undefined')
            losses={}
            for name in expected_methods:
                values=predictions[name]
                if values.shape!=data['E_y'].shape or not np.isfinite(values).all():
                    raise ValueError('Invalid retained pilot prediction rows')
                losses[name]=(values-data['E_y'])**2/variance
            if (not np.array_equal(losses['random']-losses['top'],record['h1_rows'])
                    or not np.array_equal(losses['ordinary_upper']-losses['static_partial'],record['static_rows'])):
                raise ValueError('Effect rows differ from retained predictions')
    counts=raw['batch_completed_counts']
    unique_teachers={r['teacher_id'] for r in records}
    if (not counts or counts[0]!=config.pilot.initial_teachers or counts[-1]!=len(unique_teachers)
            or len(counts)!=len(raw['batch_runtimes'])
            or any(b!=2*a for a,b in zip(counts,counts[1:]))
            or counts[-1]>config.pilot.max_teachers):
        raise ValueError('Completed batch identities differ from pilot')
    summary = summarize_pilot(records,batch_runtimes=raw['batch_runtimes'],
        bootstrap_iterations=config.pilot.bootstrap_iterations,seed=config.pilot.bootstrap_seed,purpose='scientific')
    summary = replace(summary, evidence_hashes=dict(summary.evidence_hashes,raw=_hash(raw_path),
        source=source_hash(),config=canonical_hash(raw['config'])))
    decision = choose_design(summary, raw['constraints'])
    if decision.branch not in ('powered','estimation'): raise ValueError('Numeric pilot design remains blocked')
    return decision


def publish_design(design: DesignDecision, sources: Mapping[str, Any], markdown: Path) -> None:
    """Publish a new human report and strict adjacent JSON companion, exclusively."""
    actual = eligible_design(sources)
    if canonical_hash(actual.to_dict()) != canonical_hash(design.to_dict()):
        raise ValueError('Hand-filled design differs from sealed evidence')
    markdown = Path(markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    text = ('# Pilot design decision\n\n'+f'Branch: {design.branch}. K: {design.K}. '
            f'Repetitions: {design.repetitions}. n_T: {design.n_T}.\n\n'
            'Scientific freeze requires the separately verified confirmation commitments.\n')
    with markdown.open('x') as stream: stream.write(text)
    _atomic_json(markdown.with_suffix('.canonical.json'), dict(schema_version=1,
        markdown_sha256=_hash(markdown), design=design.to_dict(), sources=dict(sources)))


def load_design(path: Path) -> tuple[DesignDecision, dict]:
    companion = Path(path).with_suffix('.canonical.json')
    if not companion.is_file(): raise ValueError('Missing strict canonical design companion')
    value = _read(companion)
    _keys(value, ('schema_version','markdown_sha256','design','sources'), 'design companion')
    if value['schema_version'] != 1 or value['markdown_sha256'] != _hash(path):
        raise ValueError('Markdown binding mismatch')
    actual = eligible_design(value['sources'])
    if canonical_hash(actual.to_dict()) != canonical_hash(value['design']):
        raise ValueError('Canonical design differs from sealed pilot')
    return actual, value['sources']


def confirmation_seeds(study_id: str, cohort_id: str, root_seed: int,
                       unit_index: int, repetitions: int) -> dict:
    """Commit independent teacher/data/optimizer/T seeds without generating labels."""
    import re
    if any(type(s) is not str or re.fullmatch(r'[A-Za-z0-9_-]+',s) is None
           or s == 'exploratory' for s in (study_id,cohort_id)):
        raise ValueError('Invalid confirmation study/cohort namespace')
    if any(type(n) is not int or n < 0 for n in (root_seed,unit_index,repetitions)) or repetitions < 2:
        raise ValueError('Invalid confirmation seed/count')
    def seed(name):
        value=canonical_hash(dict(domain='confirmatory/v1',study=study_id,cohort=cohort_id,
            root_seed=root_seed,unit=unit_index,stream=name))
        return int(value[:16],16) & ((1<<63)-1)
    return dict(teacher=seed('teacher'),reference=seed('reference'),data=seed('FGV'),test=seed('T'),
                optimizers=[seed(f'optimizer/{i}') for i in range(repetitions)])


def _manifest(value):
    from .data.types import SplitManifest
    raw=dict(value)
    assignment=raw.pop('assignment_hash',None); schema=raw.pop('feature_schema_hash',None)
    try: manifest=SplitManifest(**raw)
    except (TypeError,KeyError) as error: raise ValueError('Invalid split manifest') from error
    if manifest.assignment_hash!=assignment or manifest.feature_schema_hash!=schema:
        raise ValueError('Split manifest hash drift')
    return manifest


def confirmation_policy(scientific):
    """Versioned descriptive gate diagnostics; explicit optional Holm families.

    Legacy freezes resolve the same v1 policy through their bound source code.
    Newly prepared cohorts publish this object explicitly before the freeze.
    """
    default=dict(schema_version=1,gate_threshold=.5,
        regime_slices='each_teacher_gate_below_half_and_at_or_above_half',
        regime_claim='descriptive_only',holm_families=[])
    value=scientific.get('confirmation_policy',default)
    _keys(value,default,'confirmation policy')
    if any(value[k]!=default[k] for k in default if k!='holm_families'):
        raise ValueError('Unsupported confirmation diagnostic policy')
    families=value['holm_families']; used=set()
    if not isinstance(families,list): raise ValueError('Explicit Holm families required')
    for family in families:
        if not isinstance(family,list) or not family: raise ValueError('Empty Holm family')
        for contrast in family:
            if not isinstance(contrast,str) or contrast.count(':')!=1: raise ValueError('Invalid contrast')
            left,right=contrast.split(':')
            if left not in EVALUATION_METHODS or right not in EVALUATION_METHODS or left==right or contrast in used:
                raise ValueError('Unsupported or overlapping Holm contrast')
            used.add(contrast)
    return value


def preparation_receipt(root,cohort):
    """Optional bound measured preparation work; absence means unmeasured."""
    import math
    if 'preparation' not in cohort: return None
    value=_read(_reference(root,cohort['preparation']))
    _keys(value,('schema_version','units_sha256','raw_sha256','seconds','retained_bytes','boundary'),'preparation receipt')
    if (value['schema_version']!=1 or value['units_sha256']!=canonical_hash({'units':cohort['units']})
            or type(value['seconds']) not in (int,float) or not math.isfinite(value['seconds']) or value['seconds']<0
            or type(value['retained_bytes']) is not int or value['retained_bytes']<0 or not value['boundary']):
        raise ValueError('Preparation receipt differs from cohort or measured work')
    return value


def _scientific_sources(sources, design):
    from .data.synthetic import TeacherSpec
    root=locate(sources['root']).resolve()
    scientific=sources['scientific']
    expected=dict(hypothesis='H1-A',contrast=['B4','B5'],population='primary',coordinate='selective',
        independent_unit='teacher',exclusions=[],metric='squared_error_over_F_variance',
        grouping='row',preprocessing='F_only',refit=False,evaluation_methods=list(EVALUATION_METHODS))
    _keys(scientific, (*expected,'analysis',*(('confirmation_policy',) if 'confirmation_policy' in scientific else ())), 'synthetic scientific specification')
    confirmation_policy(scientific)
    if any(scientific[k]!=v for k,v in expected.items()):
        raise ValueError('Unsupported scientific study; transfer requires a separate freeze')
    _reference(root,scientific['analysis'])
    cohort=sources['cohort']
    _keys(cohort,('study_id','cohort_id','root_seed','units',*(('preparation',) if 'preparation' in cohort else ())), 'confirmation cohort')
    preparation=preparation_receipt(root,cohort)
    if preparation is not None and preparation['raw_sha256']!=sources['raw']['sha256']:
        raise ValueError('Preparation receipt differs from source design')
    raw=_read(_reference(root,sources['raw']))
    data_config=raw['config']['data']
    if data_config['regime']=='rare' and tuple(data_config['gate_prevalences']) not in ((.1,.5),(.01,)):
        raise ValueError('Rare generation recipe fixes prevalence at .01')
    if data_config['population']!=scientific['population']:
        raise ValueError('Pilot population differs from scientific population')
    if cohort['study_id']!=raw['config']['study_id'] or len(cohort['units'])!=design.K:
        raise ValueError('Confirmation study/count differs from design')
    pilot_teachers={r['teacher_id'] for r in raw['accepted_records']}
    teachers=set(); run_ids=set(); unit_ids=set(); seed_values=set()
    for i,unit in enumerate(cohort['units']):
        _keys(unit,('unit_id','seeds','teacher','fit_targets','manifest','run_ids',*(('development',) if 'development' in unit else ())), 'confirmation unit')
        if 'development' in unit:
            from .evaluation.confirm import load_development
            load_development(root,unit)
        expected_seeds=confirmation_seeds(cohort['study_id'],cohort['cohort_id'],cohort['root_seed'],i,design.repetitions)
        if unit['seeds']!=expected_seeds: raise ValueError('Confirmation seed commitment differs')
        values=[expected_seeds[k] for k in ('teacher','reference','data','test')]+expected_seeds['optimizers']
        if seed_values.intersection(values) or len(set(values))!=len(values): raise ValueError('Reused confirmation seed')
        seed_values.update(values)
        teacher=TeacherSpec.from_dict(_read(_reference(root,unit['teacher'])))
        manifest=_manifest(unit['manifest'])
        prevalence_options=(.01,) if data_config['regime']=='rare' else tuple(data_config['gate_prevalences'])
        gate_options=tuple(data_config['gate_types'])
        if (teacher.regime!=data_config['regime']
                or teacher.nuisance_dimensions!=data_config['nuisance_dimensions']
                or teacher.reference_rows!=data_config['reference_rows']
                or len(teacher.u)!=data_config['components']
                or teacher.prevalence_options!=prevalence_options
                or teacher.gate_type_options!=gate_options
                or teacher.gate_type_probabilities!=tuple(1/len(gate_options) for _ in gate_options)
                or not set(teacher.prevalences)<=set(prevalence_options)
                or not set(teacher.gate_types)<=set(gate_options)):
            raise ValueError('Teacher generation recipe differs from configured population')
        if any(manifest.assignments.count(part)!=data_config[key] for part,key in
                (('F','n_fit'),('G','n_growth'),('V','n_validation'))):
            raise ValueError('Development count differs from configured recipe')
        from .evaluation.protected import _test_ids
        expected_rows=_test_ids(f'confirmatory/{cohort["study_id"]}/{cohort["cohort_id"]}/T',
                                expected_seeds['test'],design.n_T)
        if expected_rows!=tuple(row for row,part in zip(manifest.row_id,manifest.assignments) if part=='T'):
            raise ValueError('T rows differ from seed commitment')
        fit_variance(root,unit)
        if (teacher.teacher_id in teachers|pilot_teachers or teacher.seed!=expected_seeds['teacher']
                or teacher.reference_seed!=expected_seeds['reference']
                or teacher.d!=raw['config']['model']['d'] or manifest.teacher_id!=teacher.teacher_id
                or manifest.phase!='confirmatory' or manifest.assignments.count('T')!=design.n_T
                or not set('FGVT')<=set(manifest.assignments)):
            raise ValueError('Teacher/split/count commitment differs or reuses pilot')
        teachers.add(teacher.teacher_id)
        if unit['unit_id'] in unit_ids or not unit['unit_id'] or len(unit['run_ids'])!=design.repetitions:
            raise ValueError('Invalid unit/repetition commitment')
        unit_ids.add(unit['unit_id'])
        for run in unit['run_ids']:
            if type(run) is not str or not run or '/' in run or run in ('.','..') or run in run_ids:
                raise ValueError('Invalid/reused confirmation run identity')
            run_ids.add(run)
    return dict(study_id=cohort['study_id'],cohort_id=cohort['cohort_id'])


def fit_variance(root: Path, unit: Mapping) -> float:
    """Verify retained F labels against the frozen split before normalizing risk."""
    import numpy as np
    from .data.types import array_hash
    with _reference(root,unit['fit_targets']).open('rb') as stream:
        values=np.load(stream,allow_pickle=False)
    manifest=_manifest(unit['manifest'])
    if (values.shape!=(manifest.assignments.count('F'),) or not np.isfinite(values).all()
            or dict(manifest.hashes).get('F.y')!=array_hash(values)):
        raise ValueError('F target evidence differs from frozen development split')
    variance=float(np.var(values,ddof=0))
    if variance<=0: raise ValueError('Degenerate F variance cannot support normalized H1 risk')
    return variance


@dataclass(frozen=True)
class FrozenProtocol:
    """An identity reference, never an access grant; every use revalidates disk."""
    path: Path
    protocol_hash: str
    pinned_commit: str
    checkout_commit: str


def _archive(path):
    return path.parent/'history'


def freeze(design: DesignDecision, sources: Mapping[str, Any], output: Path) -> FrozenProtocol:
    """Freeze only reconstructed numeric evidence and complete H1 commitments."""
    actual=eligible_design(sources)
    if canonical_hash(design.to_dict())!=canonical_hash(actual.to_dict()):
        raise ValueError('Design is draft, incomplete or differs from sealed pilot')
    identity=_scientific_sources(sources,design)
    root=locate(sources['root']).resolve()
    snapshot=execution_snapshot(root,sources['pinned_commit'])
    output=Path(output).resolve()
    if output.parent != root/'protocol': raise ValueError('Frozen protocols must use repository protocol directory')
    history=_archive(output)
    if history.exists():
        for path in history.glob('*.json'):
            previous=_read(path)['payload']['sources']['cohort']
            if (previous['study_id']==identity['study_id'] or previous['cohort_id']==identity['cohort_id']
                    or previous['root_seed']==sources['cohort']['root_seed']):
                raise ValueError('Reused confirmation study/namespace/root seed')
    payload=dict(schema_version=1,utc=_utc(),study=identity,design=design.to_dict(),
                 sources=dict(sources),snapshot=snapshot)
    digest=canonical_hash(payload)
    envelope=dict(protocol_hash=digest,payload=payload)
    output.parent.mkdir(parents=True,exist_ok=True); history.mkdir(exist_ok=True)
    # Archive first: a crash can burn a namespace but cannot silently reuse one.
    _atomic_json(history/f'{digest}.json',envelope)
    _atomic_json(output,envelope)
    return FrozenProtocol(output,digest,snapshot['pinned_commit'],snapshot['checkout_commit'])


def validate_frozen(path: Path, current: Mapping[str, Any]) -> FrozenProtocol:
    """Validate original hashes, current source and all retained scientific inputs."""
    _keys(current,('root',),'current checkout')
    path=Path(path).resolve(); value=_read(path)
    _keys(value,('protocol_hash','payload'),'frozen protocol')
    payload=value['payload']; digest=canonical_hash(payload)
    if digest!=value['protocol_hash'] or _read(_archive(path)/f'{digest}.json')!=value:
        raise ValueError('Frozen protocol/history hash drift')
    root = Path(current['root']).resolve()
    if path.parent != root/'protocol':
        raise ValueError('Frozen protocol must belong to current checkout protocol directory')
    with relocated(payload['sources']['root'], root):
        return _validate_frozen_at(path, value, root)


def _validate_frozen_at(path, value, root):
    payload=value['payload']; digest=value['protocol_hash']
    for amendment in (path.parent/'amendments').glob('*.json'):
        if _read(amendment).get('protocol_hash')==digest: raise ValueError('Scientific amendment requires fresh confirmation')
    design=eligible_design(payload['sources'])
    if canonical_hash(design.to_dict())!=canonical_hash(payload['design']): raise ValueError('Frozen design drift')
    _scientific_sources(payload['sources'],design)
    snapshot=execution_snapshot(root,payload['snapshot']['pinned_commit'])
    if snapshot['execution_sha256']!=payload['snapshot']['execution_sha256']:
        raise ValueError('Frozen execution digest drift')
    return FrozenProtocol(path,digest,snapshot['pinned_commit'],snapshot['checkout_commit'])


def amend(protocol: FrozenProtocol, *, reason: str, replacement_study_id: str,
          archive: Path | None = None) -> Path:
    """Append a dated scientific amendment; revoke old access without rewriting results."""
    value=_read(protocol.path)
    if canonical_hash(value['payload'])!=protocol.protocol_hash: raise ValueError('Protocol hash drift')
    if not reason.strip() or not replacement_study_id.strip() or replacement_study_id==value['payload']['study']['study_id']:
        raise ValueError('Amendment requires reason and a fresh study identity')
    archive_record=dict(eligibility='unavailable_no_archive_supplied',archive_sha256=None)
    if archive is not None:
        from .analysis.audit import _read_archive
        _,manifest,digest,_,payload=_read_archive(archive)
        if manifest['protocol_hash']!=protocol.protocol_hash or payload!=value['payload']:
            raise ValueError('Archive belongs to a different original protocol')
        archive_record=dict(eligibility='sealed_saved_evidence_only',archive_sha256=digest)
    directory=protocol.path.parent/'amendments'; directory.mkdir(exist_ok=True)
    record=dict(protocol_hash=protocol.protocol_hash,utc=_utc(),reason=reason,
        replacement_study_id=replacement_study_id,old_results='exploratory',fresh_confirmation_required=True,
        archive=archive_record)
    path=directory/f'{canonical_hash(record)}.json'; _atomic_json(path,record)
    return path
