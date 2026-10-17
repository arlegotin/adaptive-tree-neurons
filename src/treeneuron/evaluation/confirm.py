"""Fixed synthetic cohort preparation and serial protected confirmation.

Preparation creates F/G/V only. FrozenProtocol is revalidated before execution;
none of these APIs grants a fitting routine access to protected targets.
"""
from __future__ import annotations

import copy
from pathlib import Path
import resource
from ..locators import located, locate, producer_directory
import sys
import time
import numpy as np

from ..artifacts import _atomic_json
from ..config import canonical_hash, resolve_experiment
from ..data.synthetic import make_from_config
from ..data.types import DevelopmentData, ObservationSet
from ..protocol import (FrozenProtocol, EVALUATION_METHODS, _read, _ref, _reference,
    _manifest, _hash, eligible_design, confirmation_seeds, fit_variance, validate_frozen,
    _scientific_sources, _validate_pilot_graph, confirmation_policy, preparation_receipt)
from ..pilot.calibrate import fit_repetition, _runtime_preflight
from . import protected

SEED_POLICY='confirmation-streams/v1: optimizer=committed%2**32; allocation/contrast=sha256(full-optimizer,domain)[:16]&((1<<63)-1)'


def repetition_config(config, seeds, repetition):
    """Map nominal commitments once to the existing training RNG interfaces."""
    if type(repetition) is not int or not 0<=repetition<len(seeds['optimizers']):
        raise ValueError('Invalid committed repetition')
    value=config.to_dict()
    for name in ('teacher','reference','data'): value['seeds'][name]=seeds[name]
    nominal=seeds['optimizers'][repetition]
    value['seeds']['optimizer']=nominal % 2**32
    for name in ('allocation','contrast'):
        value['seeds'][name]=int(canonical_hash(dict(seed=nominal,domain=f'confirmation-streams/v1/{name}'))[:16],16)&((1<<63)-1)
    # Generic generators/training remain F/G/V only. Confirmation provenance is
    # in the frozen unit, never granted by a direct config namespace/phase flag.
    return resolve_experiment(value)


def prepare_cohort(design, sources, output: Path, *, cohort_id: str, root_seed: int) -> dict:
    """Create immutable commitments after numeric design, before freeze or fitting.

    Return a fresh sources mapping for publish_design. Existing evidence is never
    overwritten. Only F/G/V arrays and teachers are generated; T has row IDs only.
    """
    started=time.perf_counter_ns()
    actual=eligible_design(sources)
    if canonical_hash(actual.to_dict())!=canonical_hash(design.to_dict()):
        raise ValueError('Preparation requires validated numeric design')
    root=Path(sources['root']).resolve(); output=Path(output).resolve()
    if not output.is_relative_to(root): raise ValueError('Preparation must be repository-local')
    raw=_read(_reference(root,sources['raw'])); config=resolve_experiment(raw['config'])
    confirmation_seeds(config.study_id,cohort_id,root_seed,0,design.repetitions)
    output.mkdir(parents=True,exist_ok=False)
    units=[]
    for i in range(design.K):
        seeds=confirmation_seeds(config.study_id,cohort_id,root_seed,i,design.repetitions)
        tc=repetition_config(config,seeds,0)
        teacher,data=make_from_config(tc)
        directory=output/f'unit-{i}'; directory.mkdir()
        _atomic_json(directory/'teacher.json',teacher.to_dict())
        np.save(directory/'F.npy',data.F.y,allow_pickle=False)
        np.savez_compressed(directory/'development.npz',**{
            f'{part}_{name}':getattr(getattr(data,part),name)
            for part in 'FGV' for name in ('X','y','row_id','group_id')})
        manifest=protected.test_manifest(data.manifest,
            namespace=f'confirmatory/{config.study_id}/{cohort_id}/T',seed=seeds['test'],n_test=design.n_T)
        unit=dict(unit_id=f'unit-{i}',seeds=seeds,teacher=_ref(root,directory/'teacher.json'),
            fit_targets=_ref(root,directory/'F.npy'),development=_ref(root,directory/'development.npz'),
            manifest=manifest.to_dict(),run_ids=[f'unit-{i}-rep-{j}' for j in range(design.repetitions)])
        _atomic_json(directory/'commitment.json',unit); units.append(unit)
    result=copy.deepcopy(sources)
    result['scientific']['confirmation_policy']=confirmation_policy(result['scientific'])
    result['cohort']=dict(study_id=config.study_id,cohort_id=cohort_id,root_seed=root_seed,units=units)
    _scientific_sources(result,design)
    receipt=dict(schema_version=1,units_sha256=canonical_hash({'units':units}),raw_sha256=sources['raw']['sha256'],
        seconds=(time.perf_counter_ns()-started)/1e9,
        retained_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        boundary='design validation, fresh teachers and F/G/V, T row-ID commitments and validation; excludes final preparation receipt and sources publication')
    _atomic_json(output/'preparation.json',receipt)
    result['cohort']['preparation']=_ref(root,output/'preparation.json')
    _atomic_json(output/'sources.json',result)
    return result


def load_development(root,unit):
    """Read F/G/V arrays and check all hashes and identities against frozen rows."""
    manifest=_manifest(unit['manifest'])
    with np.load(_reference(root,unit['development']),allow_pickle=False) as arrays:
        if set(arrays.files)!={f'{p}_{n}' for p in 'FGV' for n in ('X','y','row_id','group_id')}:
            raise ValueError('Development archive must contain only F/G/V')
        parts=[ObservationSet(**{name:arrays[f'{part}_{name}'] for name in ('X','y','row_id','group_id')},
            feature_names=manifest.feature_names) for part in 'FGV']
    data=DevelopmentData(*parts,teacher_id=manifest.teacher_id)
    development=data.manifest
    keep=[i for i,p in enumerate(manifest.assignments) if p!='T']
    if (development.dataset_id!=manifest.dataset_id or development.hashes!=manifest.hashes
            or development.row_id!=tuple(manifest.row_id[i] for i in keep)
            or development.group_id!=tuple(manifest.group_id[i] for i in keep)
            or development.assignments!=tuple(manifest.assignments[i] for i in keep)):
        raise ValueError('Retained development differs from frozen commitment')
    return data


def audit_models(root: Path, report: dict, directory: Path) -> None:
    """Bind every registered model to its actual final G-selected graph node."""
    import torch
    from ..checkpoint import _descriptor
    if set(report['models'])!=set(EVALUATION_METHODS): raise ValueError('Incomplete selected models')
    for role,reference in report['models'].items():
        path=Path(reference['path'])
        if path.is_absolute():
            if not path.resolve().is_relative_to(Path(root).resolve()) or _hash(path)!=reference['sha256']:
                raise ValueError('Selected model reference differs')
        else: path=_reference(Path(root),reference)
        stage=report['methods'][role]['stages'][-1]
        selected=Path(directory)/stage['path']/'checkpoints/best.json'
        _,model,model_payload=_descriptor(path)
        _,original,original_payload=_descriptor(selected)
        for key in ('identity_hashes','model','preprocessing','update','stage','growth_metadata'):
            if model['state'][key]!=original['state'][key]: raise ValueError('Selected model checkpoint metadata differs')
        actual=torch.load(model_payload,map_location='cpu',weights_only=True)['state']['model']
        expected=torch.load(original_payload,map_location='cpu',weights_only=True)['state']['model']
        if set(actual)!=set(expected) or any(not torch.equal(actual[k],expected[k]) for k in actual):
            raise ValueError('Selected model checkpoint weights differ')


def load_predictors(root: Path, models: dict) -> dict:
    """Load actual registered selected checkpoints, then expose X-only methods."""
    from ..baselines.causal import CausalPredictor
    from ..checkpoint import _descriptor
    if set(models)!=set(EVALUATION_METHODS): raise ValueError('Incomplete model inventory')
    result={}
    for role in EVALUATION_METHODS:
        reference=models[role]
        # fit_repetition's internal absolute locators are converted at registration.
        path=Path(reference['path'])
        if path.is_absolute():
            if not path.resolve().is_relative_to(Path(root).resolve()) or _hash(path)!=reference['sha256']:
                raise ValueError('Selected model hash drift or escaped root')
        else: path=_reference(Path(root),reference)
        _,descriptor,_=_descriptor(path)
        predictor=CausalPredictor.load(path,expected_hashes=descriptor['state']['identity_hashes'])
        result[role]=predictor.predict
    return result


def _fit_run(protocol,payload,unit,repetition,directory):
    root=locate(payload['sources']['root'])
    config=resolve_experiment(_read(_reference(root,payload['sources']['raw']))['config'])
    config=repetition_config(config,unit['seeds'],repetition)
    with producer_directory(root):
        report=fit_repetition(config,load_development(root,unit),directory,canonical_hash(config.to_dict()))
    _validate_pilot_graph(report,directory,config)
    audit_models(root,report,directory)
    report['models']={role:_ref(root,Path(ref['path'])) for role,ref in report['models'].items()}
    report['confirmation']=dict(protocol_hash=protocol.protocol_hash,unit_id=unit['unit_id'],
        run_id=unit['run_ids'][repetition],nominal_optimizer_seed=unit['seeds']['optimizers'][repetition],
        effective_seeds=config.to_dict()['seeds'],seed_policy=SEED_POLICY)
    _atomic_json(directory/'pair.json',report,replace=True)
    return report


def _payload(protocol):
    value=_read(protocol.path); payload=value['payload']
    checked=validate_frozen(protocol.path,{'root':str(protocol.path.resolve().parent.parent)})
    if checked.protocol_hash!=protocol.protocol_hash: raise ValueError('Frozen identity differs')
    return payload


@located
def run_confirmation(protocol: FrozenProtocol, study: str) -> Path:
    """Run every frozen unit serially, with finite same-unit attempts and resume.

    After registration, models are fixed forever. Incomplete T access is never
    retried; completed evaluation is loaded from sealed evidence. Failures before
    registration may retry the same seeds in a fresh full fitting attempt.
    """
    if study!='synthetic': raise ValueError('Only frozen synthetic H1 is supported')
    invocation_started=time.perf_counter_ns()
    payload=_payload(protocol); root=locate(payload['sources']['root'])
    config=resolve_experiment(_read(_reference(root,payload['sources']['raw']))['config'])
    with producer_directory(root):
        _runtime_preflight(config)
    directory=protocol.path.parent/'confirmation'/protocol.protocol_hash
    directory.mkdir(parents=True,exist_ok=True)
    with protected._locked(directory):
        path=directory/'confirmation.json'
        if path.exists():
            ledger=_read(path)
            if ledger['status']=='running': ledger['unmeasured_interruption']=True
            if ledger['protocol_hash']!=protocol.protocol_hash: raise ValueError('Confirmation ledger identity drift')
        else:
            ledger=dict(schema_version=1,protocol_hash=protocol.protocol_hash,study=study,
                seed_policy=SEED_POLICY,status='running',runs=[dict(unit_id=u['unit_id'],run_id=run,
                repetition=j,status='pending',attempts=[]) for u in payload['sources']['cohort']['units']
                for j,run in enumerate(u['run_ids'])],invocations=[],costs={})
        expected=[(u['unit_id'],run,j) for u in payload['sources']['cohort']['units'] for j,run in enumerate(u['run_ids'])]
        if [(r['unit_id'],r['run_id'],r['repetition']) for r in ledger['runs']]!=expected:
            raise ValueError('Declared cohort ledger differs')
        if ledger['status'] in ('completed','incomplete','resource_limited'): return directory
        started=invocation_started
        max_attempts=1+payload['design']['statistical_policy']['retries']
        units={u['unit_id']:u for u in payload['sources']['cohort']['units']}
        def save(): _atomic_json(path,ledger,replace=True)
        save()
        try:
            for run in ledger['runs']:
                if run['status'] in ('completed','failed'): continue
                unit=units[run['unit_id']]; run_id=run['run_id']
                link_path=protected._directory(protocol)/'runs'/run_id/'link.json'
                # A hard process stop leaves a running attempt; charge it once.
                if run['attempts'] and run['attempts'][-1]['status']=='running':
                    run['attempts'][-1].update(status='interrupted',error='Process ended before attempt completion',seconds=None,duration_incomplete=True)
                    save()
                while True:
                    events=protected.access_events(protocol) if link_path.exists() else []
                    sealed=any(e['event']=='evaluation_completed' and e['run_id']==run_id for e in events)
                    accessed=any(e['event']=='labels_access' and e['run_id']==run_id for e in events)
                    if accessed and not sealed:
                        run['reason']='Previous protected label access incomplete; automatic retry forbidden'
                        break
                    if sealed:
                        attempt=run['attempts'][-1]
                    else:
                        if len(run['attempts'])>=max_attempts: break
                        previous=run['attempts'][-1] if run['attempts'] else None
                        attempt=dict(index=len(run['attempts']),status='running',seconds=None,
                            phase='evaluation' if link_path.exists() else 'fitting_and_evaluation',
                            path=previous['path'] if link_path.exists() else
                                str((directory/run_id/f'attempt-{len(run["attempts"])}').relative_to(root)))
                        if link_path.exists() and previous and 'pair' in previous: attempt['pair']=previous['pair']
                        run['attempts'].append(attempt); save()
                    tick=time.perf_counter_ns()
                    try:
                        if link_path.exists():
                            models=_read(link_path)['models']
                            if 'pair' not in attempt:
                                attempt['pair']=_ref(root,root/attempt['path']/'pair.json')
                        else:
                            report=_fit_run(protocol,payload,unit,run['repetition'],root/attempt['path'])
                            models=report['models']
                            attempt['pair']=_ref(root,root/attempt['path']/'pair.json')
                            protected.register_run(protocol,_manifest(unit['manifest']),run_id=run_id,
                                models=models,f_target_variance=fit_variance(root,unit))
                        events=protected.access_events(protocol)
                        if any(e['event']=='evaluation_completed' and e['run_id']==run_id for e in events):
                            result=protected.load_evaluation(protocol,run_id=run_id)
                        else:
                            predictors=load_predictors(root,models)
                            result=protected.open_test(protocol,_manifest(unit['manifest']),run_id=run_id).evaluate(predictors)
                        attempt['status']='completed'; run['status']='completed'
                        run['predictions']=_ref(root,protected._directory(protocol)/'runs'/run_id/'predictions.json')
                        break
                    except BaseException as error:
                        attempt.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                            error=f'{type(error).__name__}: {error}',resource_limited=isinstance(error,(MemoryError,TimeoutError)))
                        if isinstance(error,KeyboardInterrupt):
                            run['status']='interrupted'; ledger['status']='interrupted'; raise
                        if not isinstance(error,Exception): raise
                        if sealed: break  # sealed evidence is never repaired by retrying
                        # The next iteration may retry inference only, with identical
                        # registered models; any label-access event forbids a retry.
                    finally:
                        elapsed=(time.perf_counter_ns()-tick)/1e9
                        attempt['seconds']=(attempt.get('seconds') or 0.)+elapsed
                        attempt['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
                        save()
                if run['status']!='completed': run['status']='failed'
                save()
            failed=[r for r in ledger['runs'] if r['status']!='completed']
            limited=any(a.get('resource_limited') for r in failed for a in r['attempts'])
            ledger['status']='resource_limited' if limited else 'incomplete' if failed else 'completed'
        finally:
            ledger['invocations'].append(dict(seconds=(time.perf_counter_ns()-started)/1e9))
            seconds=sum(i['seconds'] for i in ledger['invocations'])
            forecast=payload['design']['forecast']
            preparation=preparation_receipt(root,payload['sources']['cohort'])
            prep_seconds=preparation['seconds'] if preparation is not None else None
            interrupted=ledger.get('unmeasured_interruption',False) or any(a.get('duration_incomplete',False) or a['seconds'] is None
                for r in ledger['runs'] for a in r['attempts'])
            combined=prep_seconds+seconds if prep_seconds is not None and not interrupted else None
            storage_roots=[directory,protected._directory(protocol)]
            if preparation is not None:
                storage_roots.append(_reference(root,payload['sources']['cohort']['preparation']).parent)
            files={p.resolve() for base in storage_roots for p in base.rglob('*') if p.is_file()}
            retained=sum(p.stat().st_size for p in files)
            ledger['costs']=dict(seconds=seconds,preparation_seconds=prep_seconds,combined_seconds=combined,
                combined_reason=None if combined is not None else 'preparation unmeasured or hard interruption duration unknown',
                forecast_seconds=forecast['seconds'],
                seconds_deviation=combined-forecast['seconds'] if combined is not None else None,
                retained_bytes=retained,forecast_storage_bytes=forecast['storage_bytes'],
                storage_deviation=retained-forecast['storage_bytes'] if preparation is not None else None,
                aggregation='disjoint preparation plus active invocation clocks; child/control clocks are nested; reanalysis separate',
                boundary='active invocation preflight, all attempts, fitting, protected prediction and ledger IO; excludes final ledger write and no-op terminal validation calls',
                forecast_boundary='frozen preparation/setup plus repetitions/evaluation and retry allowance; reanalysis reported separately',
                hard_interrupt_cost_unknown=interrupted)
            save()
    return directory
