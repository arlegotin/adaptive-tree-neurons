"""Fixed-cohort audits and prospective, saved-evidence-only archive reanalysis.

Archive sealing requires valid live source before amendment. The copied inventory
preserves the exact access ledger at sealing; later live evidence-read appends do
not change it. An archive is never a live protocol or a training capability.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import math
import pickle
import shutil
import time
import uuid

import numpy as np

from ..artifacts import _atomic_json
from ..config import canonical_hash
from ..locators import relocated, locate
from ..protocol import (FrozenProtocol, _read, _hash, _reference, _relative, _manifest,
                        validate_frozen, execution_snapshot, fit_variance,
                        preparation_receipt, EVALUATION_METHODS, _git, _keys)
from ..evaluation import protected
from .primary import analyze_primary, summarize_cohort

_ERRORS = (ValueError, KeyError, TypeError, OSError, IndexError, RuntimeError,
           AttributeError, EOFError, OverflowError, pickle.UnpicklingError)


def conclude(primary, reasons):
    """Completeness and validity precede the sign of the scientific result."""
    if reasons:
        return dict(status='blocked', interpretation='blocked', superiority=False,
                    decision='revise', reason='Evidence integrity or accounting failed')
    if primary is None or primary['status'] != 'completed':
        return dict(status='incomplete', interpretation='incomplete', superiority=False,
                    decision='continue', reason='Finish every frozen unit; retain all failures')
    summary = primary['primary']
    if summary['status'] != 'supported':
        return dict(status='completed', interpretation=summary['status'], superiority=False,
                    decision='narrow', reason='Completed cohort; inferential precision/support remains limited')
    return dict(status='completed', interpretation=summary['interpretation'],
                superiority=summary['superiority'], decision='replicate' if summary['superiority'] else 'narrow',
                reason='H1 placement component only; practical advantage is not established')


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid measured {name}')
    return value


def _account(payload, ledger, root, study_dir):
    """Reconstruct all attempts and disjoint work, never sum nested clocks."""
    if ledger.get('schema_version') != 1 or ledger.get('study') != 'synthetic':
        raise ValueError('Unsupported confirmation ledger schema/study')
    if ledger.get('status') not in ('running','interrupted','completed','incomplete','resource_limited'):
        raise ValueError('Unknown confirmation ledger status')
    if ledger['status']=='completed' and any(r['status']!='completed' for r in ledger['runs']):
        raise ValueError('A completed ledger requires every frozen run completed')
    from ..evaluation.confirm import SEED_POLICY
    if ledger.get('seed_policy') != SEED_POLICY:
        raise ValueError('Confirmation seed policy differs')
    expected = [(u['unit_id'], j, rid) for u in payload['sources']['cohort']['units']
                for j, rid in enumerate(u['run_ids'])]
    if [(r['unit_id'], r['repetition'], r['run_id']) for r in ledger['runs']] != expected:
        raise ValueError('Run ledger omits or changes frozen cohort')
    max_attempts = 1 + payload['design']['statistical_policy']['retries']
    attempts = []; unknown = bool(ledger.get('unmeasured_interruption', False))
    declared_paths = set()
    for run in ledger['runs']:
        previous = None
        if len(run['attempts']) > max_attempts:
            raise ValueError('Attempt count exceeds frozen retry allowance')
        if run['status'] not in ('pending', 'running', 'interrupted', 'failed', 'completed'):
            raise ValueError('Unknown run outcome')
        for index, attempt in enumerate(run['attempts']):
            if attempt['index'] != index or attempt['status'] not in ('running','interrupted','failed','completed'):
                raise ValueError('Invalid attempt sequence/outcome')
            phase = attempt['phase']; path = _relative(root, attempt['path']).resolve()
            if not path.is_relative_to(study_dir/run['run_id']):
                raise ValueError('Attempt path escapes declared run')
            declared_paths.add(path)
            if phase == 'evaluation':
                if previous is None or attempt['path'] != previous['path'] or attempt.get('pair') != previous.get('pair'):
                    raise ValueError('Evaluation resume changed selected fitting evidence')
            elif phase != 'fitting_and_evaluation' or path != study_dir/run['run_id']/f'attempt-{index}':
                raise ValueError('Invalid fitting attempt boundary/path')
            if previous is not None and previous['status'] == 'completed':
                raise ValueError('Attempt follows completed immutable evidence')
            if attempt['seconds'] is None or attempt.get('duration_incomplete', False):
                unknown = True
            else:
                _number(attempt['seconds'], 'attempt seconds')
            if attempt['status'] in ('failed','interrupted') and not attempt.get('error'):
                raise ValueError('Failure classification missing')
            if 'pair' in attempt:
                pair_path = _reference(root, attempt['pair'])
                if pair_path.resolve() != path/'pair.json':
                    raise ValueError('Attempt pair path differs')
            attempts.append(dict(run_id=run['run_id'], **attempt))
            previous = attempt
        if run['status'] == 'completed':
            if len([a for a in run['attempts'] if a['status']=='completed']) != 1:
                raise ValueError('Completed run has ambiguous attempt evidence')
            output = _reference(root, run['predictions'])
            if output.resolve() != study_dir.parent.parent/'access'/ledger['protocol_hash']/'runs'/run['run_id']/'predictions.json':
                raise ValueError('Completed prediction locator differs')
    for path in study_dir.glob('*/attempt-*'):
        if path.is_dir() and path.resolve() not in declared_paths:
            raise ValueError('Unaccounted fitting attempt directory')
    invocations = ledger.get('invocations')
    if not isinstance(invocations, list) or not invocations:
        raise ValueError('Missing measured execution invocations')
    seconds = sum(_number(i['seconds'], 'invocation seconds') for i in invocations)
    # All attempt timers are nested in active invocations. A hard interruption
    # can lose an invocation clock, so its unknown duration is not a zero bound.
    attempt_seconds=math.fsum(a['seconds'] for a in attempts if a['seconds'] is not None)
    if not unknown and attempt_seconds>seconds and not math.isclose(attempt_seconds,seconds,rel_tol=1e-12,abs_tol=1e-9):
        raise ValueError('Attempt durations exceed enclosing invocation work')
    preparation = preparation_receipt(root, payload['sources']['cohort'])
    prep = preparation['seconds'] if preparation else None
    combined = prep + seconds if prep is not None and not unknown else None
    forecast = payload['design']['forecast']; recorded = ledger['costs']
    expected_cost = dict(seconds=seconds, preparation_seconds=prep, combined_seconds=combined,
                         forecast_seconds=forecast['seconds'],
                         seconds_deviation=combined-forecast['seconds'] if combined is not None else None,
                         forecast_storage_bytes=forecast['storage_bytes'],hard_interrupt_cost_unknown=unknown)
    if any(recorded.get(k) != v for k,v in expected_cost.items()):
        raise ValueError('Recorded costs differ from disjoint measured receipts')
    for name in ('aggregation','boundary','forecast_boundary'):
        if not recorded.get(name):
            raise ValueError(f'Missing timing {name}')
    retained = _number(recorded['retained_bytes'], 'retained bytes')
    if recorded.get('storage_deviation') != (retained-forecast['storage_bytes'] if preparation else None):
        raise ValueError('Storage forecast deviation differs')
    if combined is None and not recorded.get('combined_reason'):
        raise ValueError('Unknown combined work needs a reason')
    return dict(attempts=attempts, attempted_runs=sum(bool(r['attempts']) for r in ledger['runs']),
                failed_attempts=sum(a['status']=='failed' for a in attempts),
                interrupted_attempts=sum(a['status']=='interrupted' for a in attempts),
                costs=dict(expected_cost, retained_bytes_at_execution=retained,
                    combined_reason=recorded.get('combined_reason'),
                    aggregation=recorded['aggregation'], boundary=recorded['boundary'],
                    storage_note='Execution storage was sampled before its final ledger write; current bytes include later analyses'))


def _validate_saved(saved, unit, payload, variance):
    """Check producer scalar metrics against original-unit rows and frozen IDs."""
    manifest = _manifest(unit['manifest'])
    ids = [r for r,part in zip(manifest.row_id, manifest.assignments) if part=='T']
    if saved['row_id'] != ids or set(saved['predictions']) != set(EVALUATION_METHODS):
        raise ValueError('Saved rows/method inventory differs from frozen commitment')
    y = np.asarray(saved['observations']['y'], dtype=np.float64)
    if y.shape != (len(ids),) or not np.isfinite(y).all():
        raise ValueError('Invalid original-unit target rows')
    if saved['observations']['group_id'] != ids:
        raise ValueError('Saved group identity differs')
    target = np.asarray(saved['observations']['noise_free_target'], dtype=np.float64)
    if target.shape != y.shape or not np.isfinite(target).all():
        raise ValueError('Invalid saved noise-free targets')
    teacher = _read(_reference(locate(payload['sources']['root']),unit['teacher']))
    masks = saved['observations']['regime_masks']
    expected = {f'gate_{i}_{label}' for i in range(len(teacher['u']))
                for label in ('below_half', 'at_or_above_half')}
    if set(masks) != expected:
        raise ValueError('Missing declared descriptive regime masks')
    for i in range(len(teacher['u'])):
        a,b = (masks[f'gate_{i}_{label}'] for label in ('below_half','at_or_above_half'))
        if len(a)!=len(ids) or len(b)!=len(ids) or any(type(x) is not bool for x in a+b) or any(x==z for x,z in zip(a,b)):
            raise ValueError('Invalid complementary descriptive masks')
    if set(saved['metrics']) != set(EVALUATION_METHODS):
        raise ValueError('Missing producer metric terms')
    for role,values in saved['predictions'].items():
        prediction = np.asarray(values,dtype=np.float64)
        if prediction.shape != y.shape or not np.isfinite(prediction).all():
            raise ValueError('Invalid original-unit predictions')
        mse = float(np.mean((prediction-y)**2))
        if saved['metrics'][role] != dict(mse=mse,normalized_risk=mse/variance):
            raise ValueError('Manual metric mismatch against saved original-unit rows')



def _audit_pair_inventory(root, pair, directory):
    """Read every selected/last/probe checkpoint with ordinary strict guards.

    Loaded states are discarded and process RNG is restored; no update, fit,
    prediction or test generation occurs. This routine is never used by archives.
    """
    from ..checkpoint import load_checkpoint, _descriptor
    from ..train import capture_rng, restore_rng, source_hash
    from ..backend import lock_hash, runtime_hash
    if pair['source_sha256']!=source_hash():
        raise ValueError('Confirmation package source identity differs')
    graph_paths={locate(c['path'],producer=True).resolve() for c in pair['child_runs']}
    greedy_path=locate(pair['greedy_report'],producer=True)
    greedy=_read(greedy_path)
    graph_paths.update((greedy_path.parent/t['path']).resolve() for t in greedy['trials'])
    counts={}; clocks=[]; checkpoints=0
    for path in sorted(graph_paths):
        if not path.is_relative_to(Path(directory).resolve()): raise ValueError('Checkpoint graph path escapes attempt')
        manifest=_read(path/'manifest.json'); method=manifest['method']
        if method['scope']=='coordinator': continue
        terms=method.get('weight_terms')
        if not isinstance(terms,dict) or set(terms)!={'layer','head'} or any(type(v) is not int or v<0 for v in terms.values()):
            raise ValueError('Missing or invalid model weight terms')
        if sum(terms.values())!=method['total_parameters'] or not 0<=method['trainable_parameters']<=method['total_parameters']:
            raise ValueError('Model parameter terms disagree')
        counts[str(path.relative_to(directory))]=dict(weight_terms=terms,total_parameters=method['total_parameters'],
            trainable_parameters=method['trainable_parameters'])
        costs=manifest['costs']
        expected={'search','preprocess','base','score','probe','continuation','validation',
                  'first_prediction','warm_prediction','acquisition','enclosing'}
        if set(costs)!=expected: raise ValueError('Missing declared cost terms')
        for name,value in costs.items():
            if not value.get('boundary'): raise ValueError('Missing timing boundary')
            if value['status']=='unmeasured':
                if value['nanoseconds'] is not None or not value['reason']: raise ValueError('Unmeasured cost needs null and reason')
            elif value['status'] in ('measured','estimated'):
                _number(value['nanoseconds'],'nested nanoseconds')
                if value['reason'] is not None: raise ValueError('Measured cost has inconsistent reason')
            else: raise ValueError('Unknown timing classification')
        clocks.append(dict(path=str(path.relative_to(directory)),costs=costs))
        for pointer in ('best.json','last.json'):
            checkpoint=path/'checkpoints'/pointer
            _,descriptor,_=_descriptor(checkpoint)
            identities=descriptor['state']['identity_hashes']
            if identities['code']!=source_hash() or identities['lock']!=lock_hash() or identities['runtime']!=runtime_hash():
                raise ValueError('Checkpoint source/lock/runtime differs')
            before=capture_rng()
            try:
                state=load_checkpoint(checkpoint,expected_hashes=identities,device=descriptor['state']['device'])
                actual=sum(t.numel() for t in state.model.parameters())
                # last and G-selected best share the role topology and count.
                if actual!=method['total_parameters']: raise ValueError('Checkpoint parameter count differs')
            finally:
                restore_rng(before)
            checkpoints+=1
    return dict(selected_models=len(pair['models']),probe_candidates=len(greedy['trials']),
                checkpoint_pointers=checkpoints,model_counts=counts,nested_costs=clocks,
                effective_growth=pair['capacity']['effective_count'],
                aggregation='Diagnostic nested clocks only; never added to active invocation total')

def _limits():
    return dict(hypothesis='H1-A placement only', practical_advantage='not established',
                strongest_models='H3 mandatory practical assets/adapters and numeric transfer design not evaluated',
                energy='simulation separate; no Energy measurements in H1 tables',
                track_c='external references excluded from local measured tables',
                novelty='R12 full text retrieved in WP00; bounded mechanism difference is provisional, not proof of novelty; revisit closest complete mechanisms at actual claim boundary',
                reproduction='Actual locked clean reproduction and scientific report execution remain a separate post-cohort gate')


def audit(study_dir: Path, protocol: FrozenProtocol) -> dict[str, Any]:
    """Validate live evidence and independently reconstruct the full frozen cohort."""
    started=time.perf_counter_ns(); work_id=uuid.uuid4().hex
    study_dir=Path(study_dir).resolve()
    work_enabled=isinstance(protocol,FrozenProtocol) and study_dir==(protocol.path.resolve().parent/'confirmation'/protocol.protocol_hash) and study_dir.is_dir()
    before=set((study_dir/'analysis-work').glob('*.json')) if work_enabled else set()
    if work_enabled:
        markers=study_dir/'audit-starts'; markers.mkdir(exist_ok=True)
        _atomic_json(markers/f'{work_id}.json',dict(schema_version=1,protocol_hash=protocol.protocol_hash))
    result = dict(schema_version=1,mode='live',valid=False,status='blocked',reasons=[],
                  protocol_hash=getattr(protocol,'protocol_hash',None),primary=None,accounting=None,limits=_limits())
    try:
        if not isinstance(protocol,FrozenProtocol):
            raise ValueError('Live FrozenProtocol required; archive is not an access capability')
        root = protocol.path.resolve().parent.parent
        checked = validate_frozen(protocol.path, {'root':str(root)})
        if checked.protocol_hash != protocol.protocol_hash:
            raise ValueError('Protocol identity differs')
        payload = _read(protocol.path)['payload']; study_dir=Path(study_dir).resolve()
        if study_dir != protocol.path.parent/'confirmation'/protocol.protocol_hash:
            raise ValueError('Study path differs from frozen protocol')
        result['identity']=dict(original_root=payload['sources']['root'],current_root=str(root),
            pinned_commit=checked.pinned_commit,checkout_commit=checked.checkout_commit,
            execution_sha256=payload['snapshot']['execution_sha256'])
        with relocated(payload['sources']['root'], root):
            ledger = _read(study_dir/'confirmation.json')
            if ledger['protocol_hash'] != protocol.protocol_hash:
                raise ValueError('Confirmation protocol identity differs')
            result['accounting'] = _account(payload, ledger, root, study_dir)
            events = protected.access_events(protocol)
            declared = {rid:u for u in payload['sources']['cohort']['units'] for rid in u['run_ids']}
            if any(e.get('run_id') not in declared for e in events):
                raise ValueError('Access ledger contains undeclared run')
            result['accounting']['graphs']=[]
            for run in ledger['runs']:
                if run['status'] == 'completed':
                    attempt=next(a for a in run['attempts'] if a['status']=='completed')
                    pair_path=_reference(root,attempt['pair'])
                    graph=_audit_pair_inventory(root,_read(pair_path),pair_path.parent)
                    result['accounting']['graphs'].append(dict(run_id=run['run_id'],**graph))
                    saved = protected.load_evaluation(protocol,run_id=run['run_id'])
                    _validate_saved(saved,declared[run['run_id']],payload,fit_variance(root,declared[run['run_id']]))
            result['primary'] = analyze_primary(study_dir,protocol)
            for run in result['primary']['runs']:
                if run['status']=='invalid': result['reasons'].append(f'{run["run_id"]}: {run["reason"]}')
    except _ERRORS as error:
        result['reasons'].append(f'{type(error).__name__}: {error}')
    result['conclusion']=conclude(result['primary'],result['reasons'])
    result['status']=result['conclusion']['status']
    result['valid']=result['status']=='completed' and not result['reasons']
    result['audit_sha256']=canonical_hash(result)
    if work_enabled:
        _finish_audit_work(study_dir, result, work_id, started, before)
    return result


def seal_archive(study_dir: Path, protocol: FrozenProtocol, output: Path) -> Path:
    """Copy validated evidence before amendment/drift; never seal a blocked audit."""
    result = audit(study_dir,protocol)
    if result['reasons'] or result['primary'] is None:
        raise ValueError('Archive sealing requires valid live evidence: '+ '; '.join(result['reasons']))
    copy_started=time.perf_counter_ns()
    root=protocol.path.resolve().parent.parent; output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    payload=_read(protocol.path)['payload']; paths=set()
    def collect(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'}:
                paths.add(_reference(root,value))
            else:
                for item in value.values(): collect(item)
        elif isinstance(value,list):
            for item in value: collect(item)
    collect(payload['sources'])
    for name in payload['snapshot']['files']: paths.add(_relative(root,name))
    readiness=_read(_reference(root,payload['sources']['readiness'])); collect(readiness)
    for reference in readiness['reviews'].values(): collect(_read(_reference(root,reference)))
    bases=[Path(study_dir),protected._directory(protocol),
           _reference(root,payload['sources']['raw']).parent]
    for unit in payload['sources']['cohort']['units']: bases.append(_reference(root,unit['teacher']).parent)
    for base in bases:
        if output.is_relative_to(base.resolve()): raise ValueError('Archive output cannot be inside its input inventory')
        paths.update(p for p in base.rglob('*') if p.is_file())
    paths.add(protocol.path); paths.add(protocol.path.parent/'history'/f'{protocol.protocol_hash}.json')
    files={}
    for path in sorted(paths):
        name=str(path.relative_to(root)); safe=_relative(root,name)
        if safe.is_symlink(): raise ValueError('Archive inventory cannot contain symlinks')
        files[name]=_hash(safe)
    output.mkdir(parents=True)
    for name,digest in files.items():
        destination=output/'files'/name; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,destination)
        if _hash(destination)!=digest: raise ValueError('Evidence changed during archive copy')
    manifest=dict(schema_version=1,mode='sealed-saved-evidence-only',protocol_hash=protocol.protocol_hash,
        protocol_path=str(protocol.path.relative_to(root)),study_path=str(Path(study_dir).relative_to(root)),
        files=files,live_audit=result,authority='Prospective live validation; immutable copied access-ledger prefix; no new T, fit, predict, or resume')
    manifest['archive_sha256']=canonical_hash(manifest)
    _atomic_json(output/'archive.json',manifest)
    _atomic_json(output/'copy-work.json',dict(schema_version=1,archive_sha256=manifest['archive_sha256'],
        seconds=(time.perf_counter_ns()-copy_started)/1e9,
        boundary='inventory and copying after live audit; excludes final copy-work receipt',
        aggregation='disjoint from live audit; do not count copied historical work again'))
    return output/'archive.json'


def _read_archive(bundle):
    """Verify the prospective copied inventory without granting live authority."""
    bundle=Path(bundle)
    directory=bundle.parent if bundle.is_file() else bundle
    manifest=_read(bundle if bundle.is_file() else bundle/'archive.json')
    _keys(manifest,('schema_version','mode','protocol_hash','protocol_path','study_path',
        'files','live_audit','authority','archive_sha256'),'archive')
    digest=manifest['archive_sha256']
    if (canonical_hash({k:v for k,v in manifest.items() if k!='archive_sha256'})!=digest
            or manifest['schema_version']!=1 or manifest['mode']!='sealed-saved-evidence-only'):
        raise ValueError('Archived inventory identity differs')
    root=directory/'files'
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if not isinstance(manifest['files'],dict) or actual!=set(manifest['files']):
        raise ValueError('Archived inventory missing or extra files')
    for name,checksum in manifest['files'].items():
        if _hash(_relative(root,name))!=checksum: raise ValueError(f'Archived evidence hash drift: {name}')
    frozen=_read(_relative(root,manifest['protocol_path'])); payload=frozen['payload']
    if canonical_hash(payload)!=manifest['protocol_hash'] or frozen['protocol_hash']!=manifest['protocol_hash']:
        raise ValueError('Archived frozen identity differs')
    history=_read(_relative(root,str(Path(manifest['protocol_path']).parent/'history'/f'{manifest["protocol_hash"]}.json')))
    if history!=frozen: raise ValueError('Archived frozen history differs')
    if manifest['study_path']!=str(Path(manifest['protocol_path']).parent/'confirmation'/manifest['protocol_hash']):
        raise ValueError('Archived study path differs')
    original=manifest['live_audit']; original_hash=original['audit_sha256']
    if (canonical_hash({k:v for k,v in original.items() if k!='audit_sha256'})!=original_hash
            or original['reasons'] or original['mode']!='live'
            or original['protocol_hash']!=manifest['protocol_hash']):
        raise ValueError('Archive lacks valid original live audit')
    return directory,manifest,digest,root,payload


def audit_archive(bundle: Path, *, analysis_root: Path | None = None) -> dict[str, Any]:
    """Recompute only sealed observations; all archived claims are exploratory."""
    from ..backend import ROOT
    from ..train import source_hash
    started=time.perf_counter_ns()
    result=dict(schema_version=1,mode='archived-saved-evidence/v1',valid=False,status='blocked',
                reasons=[],primary=None,accounting=None,limits=_limits())
    try:
        directory,manifest,digest,root,payload=_read_archive(bundle)
        original=manifest['live_audit']
        current=Path(analysis_root or ROOT).resolve()
        snapshot=execution_snapshot(current,_git(current,'rev-parse','HEAD').decode().strip())
        result.update(protocol_hash=manifest['protocol_hash'],archive_sha256=digest,
            identity=dict(original=original['identity'],analysis_execution_sha256=snapshot['execution_sha256'],
                          analysis_commit=snapshot['checkout_commit'],analysis_package_sha256=source_hash()))
        with relocated(payload['sources']['root'],root):
            study=_relative(root,manifest['study_path']); ledger=_read(study/'confirmation.json')
            result['accounting']=_account(payload,ledger,root,study)
            result['accounting']['graphs']=original['accounting']['graphs']
            runs=[]; predictions={}
            lookup={r['run_id']:r for r in ledger['runs']}
            access=root/Path(manifest['protocol_path']).parent/'access'/manifest['protocol_hash']
            for unit in payload['sources']['cohort']['units']:
                variance=fit_variance(root,unit)
                for rid in unit['run_ids']:
                    entry=lookup[rid]
                    run=dict(run_id=rid,unit_id=unit['unit_id'],teacher_id=unit['manifest']['teacher_id'],
                             f_target_variance=variance,status='incomplete')
                    if entry['status']=='completed':
                        saved=protected._read_sealed_rows(access,root,payload,manifest['protocol_hash'],rid)
                        _validate_saved(saved,unit,payload,variance)
                        predictions[rid]=saved; run['status']='completed'
                    else: run['reason']='Declared run incomplete'
                    runs.append(run)
            result['primary']=summarize_cohort(payload,ledger,runs,predictions,manifest['protocol_hash'])
            # Original live validation attests checkpoint/graph integrity; every byte
            # in that graph was copied and independently rehashed above.
            if result['primary']!= {k:v for k,v in original['primary'].items() if k!='report_sha256'}:
                raise ValueError('Archived raw-row reconstruction differs from sealed live analysis')
    except _ERRORS as error:
        result['reasons'].append(f'{type(error).__name__}: {error}')
    result['conclusion']=conclude(result['primary'],result['reasons'])
    if not result['reasons']:
        result['conclusion'].update(superiority=False,interpretation='exploratory',decision='replicate',
            reason='Archived saved-evidence analysis; fresh confirmation required after amendment/source drift')
        result['valid']=result['primary']['status']=='completed'
        result['status']='exploratory' if result['valid'] else 'incomplete'
        result['conclusion']['status']=result['status']
        result['primary']['primary']['superiority']=False
        result['primary']['claim']='exploratory saved-evidence only'
    result['audit_sha256']=canonical_hash(result)
    if not result['reasons']:
        work=directory/'archive-work'; work.mkdir(exist_ok=True)
        _atomic_json(work/f'{uuid.uuid4().hex}.json',dict(schema_version=1,archive_sha256=digest,
            audit_sha256=result['audit_sha256'],seconds=(time.perf_counter_ns()-started)/1e9,
            boundary='copied inventory, original access and raw-row reanalysis; excludes final work receipt',
            aggregation='new archive analysis invocation only; copied historical clocks are not new work'))
    return result


def _finish_audit_work(study, result, work_id, started, previous_analysis):
    """Record outer audit work once; primary-analysis receipts inside it are nested."""
    work=study/'audit-work'; work.mkdir(exist_ok=True)
    nested=sorted(set((study/'analysis-work').glob('*.json'))-previous_analysis)
    references=[dict(path=str(p.relative_to(study)),sha256=_hash(p)) for p in nested]
    outer=0.; linked=set(r['path'] for r in references); completed=set()
    unknown=False
    try:
        for path in work.glob('*.json'):
            receipt=_read(path); completed.add(path.stem)
            if receipt['schema_version']!=1 or receipt['protocol_hash']!=result['protocol_hash']:
                raise ValueError('Audit work identity differs')
            outer+=_number(receipt['seconds'],'audit seconds')
            for reference in receipt['nested_analysis_receipts']:
                _reference(study,reference)
                if reference['path'] in linked: raise ValueError('Analysis work charged to multiple audits')
                linked.add(reference['path'])
        analysis=[]
        for path in (study/'analysis-work').glob('*.json'):
            receipt=_read(path)
            if receipt['schema_version']!=1 or receipt['protocol_hash']!=result['protocol_hash']:
                raise ValueError('Analysis work identity differs')
            seconds=_number(receipt['seconds'],'analysis seconds')
            report=_read(study/f'primary-{receipt["report_sha256"]}.json')
            if canonical_hash({k:v for k,v in report.items() if k!='report_sha256'})!=receipt['report_sha256']:
                raise ValueError('Analysis work report identity differs')
            analysis.append((receipt['previous_analysis_seconds'],seconds,path))
        previous=0.; standalone=0.
        for prior,seconds,path in sorted(analysis):
            if prior!=previous: raise ValueError('Analysis work sequence differs')
            previous+=seconds
            if str(path.relative_to(study)) not in linked: standalone+=seconds
        unknown=any(p.stem not in completed|{work_id} for p in (study/'audit-starts').glob('*.json'))
    except _ERRORS as error:
        result['reasons'].append(f'Work receipt integrity: {error}')
        result['conclusion']=conclude(result['primary'],result['reasons'])
        result['valid']=False; result['status']='blocked'
        result.pop('audit_sha256'); result['audit_sha256']=canonical_hash(result)
        unknown=True; standalone=0.
    base=(result.get('accounting') or {}).get('costs',{}).get('combined_seconds')
    elapsed=(time.perf_counter_ns()-started)/1e9
    destination=study/f'audit-{result["audit_sha256"]}.json'
    if not destination.exists(): _atomic_json(destination,result)
    _atomic_json(work/f'{work_id}.json',dict(schema_version=1,protocol_hash=result['protocol_hash'],
        audit_sha256=result['audit_sha256'],seconds=elapsed,nested_analysis_receipts=references,
        total_study_work_seconds=base+outer+standalone+elapsed if base is not None and not unknown else None,
        total_reason=None if base is not None and not unknown else 'Missing preparation/execution or interrupted/invalid audit work',
        boundary='entire audit including nested primary reconstruction; excludes final deterministic snapshot and work receipt',
        aggregation='preparation/execution plus outer audits plus standalone analyses; nested analysis excluded'))
