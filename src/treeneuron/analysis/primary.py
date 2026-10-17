"""Reconstruct fixed-cohort H1 from sealed predictions, one teacher per unit.

Repeated optimizer/allocation fits are averaged within teachers. Percentile
bootstrap and Student-t sensitivity are empirical approximations, not a proof of
coverage or prospective power. Missing/invalid pairs never become zero effects.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import math
import time
import uuid
import numpy as np
from scipy.stats import t

from ..locators import locate, relocated
from ..artifacts import _atomic_json
from ..config import canonical_hash, resolve_experiment
from ..evaluation.metrics import regression_metrics
from ..evaluation.protected import load_evaluation
from ..evaluation.confirm import _payload, repetition_config, audit_models, load_development, SEED_POLICY
from ..protocol import FrozenProtocol, EVALUATION_METHODS, _read, _reference, fit_variance, _validate_pilot_graph, confirmation_policy


def _average_metrics(records):
    """Equal repetition averages of descriptive scalar metrics."""
    return {key:float(np.mean([r[key] for r in records])) for key in records[0]
            if all(type(r.get(key)) in (int,float) for r in records)}


def aggregate_teachers(predictions: Any, runs: Any) -> list[dict]:
    """Inputs: run_id→saved prediction object and declared run identity records.

    Each run record includes teacher_id, unit_id, status and observed F variance.
    Repeated identical run records are deduplicated; conflicting identities fail.
    Every declared teacher remains, even when any repetition lacks valid evidence.
    """
    unique={}
    for run in runs:
        rid=run['run_id']
        if rid in unique and unique[rid]!=run: raise ValueError('Conflicting duplicate run identity')
        unique[rid]=run
    units=defaultdict(list); identities={}
    for run in unique.values():
        teacher=run['teacher_id']; unit=run['unit_id']
        if teacher in identities and identities[teacher]!=unit: raise ValueError('Teacher reused across data units')
        identities[teacher]=unit; units[(teacher,unit)].append(run)
    result=[]
    for (teacher,unit),records in units.items():
        methods=defaultdict(list); regimes=defaultdict(lambda:defaultdict(list)); errors=[]
        invalid=False; degenerate=False; missing=False
        for run in records:
            saved=predictions.get(run['run_id'])
            if saved is None or run['status']!='completed':
                invalid=invalid or run['status']=='invalid'
                missing=True; errors.append(dict(run_id=run['run_id'],reason=run.get('reason','missing or failed run'))); continue
            try:
                if saved['run_id']!=run['run_id']: raise ValueError('Saved run identity differs')
                obs=saved['observations']; y=np.asarray(obs['y'],dtype=np.float64)
                ids=saved['row_id']
                if len(ids)!=len(y) or len(set(ids))!=len(ids): raise ValueError('Invalid saved row identities')
                if not {'top','random'}<=set(saved['predictions']): raise ValueError('Missing paired method')
                local={name:regression_metrics(y,np.asarray(values),fit_target_variance=run['f_target_variance'],
                    groups=np.asarray(obs['group_id']),noisefree=np.asarray(obs['noise_free_target']))
                    for name,values in saved['predictions'].items()}
                if any(v['normalized_mse'] is None for v in local.values()): degenerate=True
                for name,metrics in local.items(): methods[name].append(metrics)
                for label,mask in obs.get('regime_masks',{}).items():
                    mask=np.asarray(mask,dtype=bool)
                    if mask.shape!=y.shape: raise ValueError('Invalid regime mask')
                    for name,values in saved['predictions'].items():
                        metric=(regression_metrics(y[mask],np.asarray(values)[mask],fit_target_variance=run['f_target_variance'],
                            noisefree=np.asarray(obs['noise_free_target'])[mask]) if mask.any() else {'n':0})
                        regimes[label][name].append(metric)
            except (ValueError,TypeError,KeyError,OverflowError) as error:
                invalid=True; errors.append(dict(run_id=run['run_id'],reason=str(error)))
        metrics={name:_average_metrics(values) for name,values in methods.items()}
        complete=not missing and not invalid
        original=metrics['random']['mse']-metrics['top']['mse'] if complete else None
        effect=metrics['random']['normalized_mse']-metrics['top']['normalized_mse'] if complete and not degenerate else None
        result.append(dict(teacher_id=teacher,unit_id=unit,repetitions=len(records),
            valid_repetitions=len(methods.get('top',[])),status='invalid' if invalid else 'incomplete' if missing else 'degenerate_risk' if degenerate else 'completed',
            D=effect,D_original=original,metrics=metrics,errors=errors,
            regimes={label:{name:dict(metrics=_average_metrics(values),
                status='available' if all(v['n']>0 for v in values) else 'unavailable_empty_slice',
                repetition_counts=[v['n'] for v in values]) for name,values in bymethod.items()}
                for label,bymethod in regimes.items()}))
    return result


def summarize_effects(effects,policy):
    """Whole-teacher percentile interval and explicitly separate t sensitivity."""
    values=np.asarray(effects,dtype=np.float64)
    if values.ndim!=1 or not np.isfinite(values).all(): raise ValueError('Finite teacher effects required')
    n=len(values)
    result=dict(independent_count=n,bootstrap_unit_count=n,effects=values.tolist(),
        mean=float(values.mean()) if n else None,median=float(np.median(values)) if n else None,
        interval=None,student_t_interval=None,student_t_p=None,endpoint_mc_error=None,
        superiority=False,status='degenerate',interpretation='unavailable',
        uncertainty_unit='whole teacher with all repetitions averaged first',
        limitation='Empirical bootstrap/t sensitivity does not establish nominal coverage or prospective power')
    if n<2 or float(np.var(values,ddof=1))<=0: return result
    draws=policy['bootstrap_iterations']; seed=policy['bootstrap_seed']; alpha=policy['alpha']
    if type(draws) is not int or draws<200 or type(seed) is not int or seed<0 or not 0<alpha<1:
        raise ValueError('Invalid frozen bootstrap policy')
    rng=np.random.default_rng(seed)
    # Bound temporary memory without changing RNG draw order or bootstrap units.
    samples=np.empty(draws)
    for start in range(0,draws,1000):
        stop=min(start+1000,draws)
        samples[start:stop]=values[rng.integers(n,size=(stop-start,n))].mean(axis=1)
    quantiles=[alpha/2,1-alpha/2]
    interval=np.quantile(samples,quantiles,method='linear')
    chunks=[np.quantile(part,quantiles,method='linear') for part in np.array_split(samples,10)]
    mc=float(np.std(chunks,axis=0,ddof=1).max()/math.sqrt(10))
    se=float(np.std(values,ddof=1)/math.sqrt(n)); critical=float(t.ppf(1-alpha/2,n-1))
    sensitivity=[result['mean']-critical*se,result['mean']+critical*se]
    precise=mc<=policy['endpoint_tolerance']; supported=policy['inference_supported']
    result.update(interval=interval.tolist(),student_t_interval=sensitivity,
        student_t_p=float(2*t.sf(abs(result['mean']/se),n-1)),student_t_df=n-1,
        endpoint_mc_error=mc,endpoint_mc_method='maximum endpoint SE over ten disjoint bootstrap chunks; linear quantiles',
        bootstrap_seed=seed,bootstrap_iterations=draws,
        status='supported' if supported and precise else 'unsupported' if not supported else 'mc_unresolved',
        superiority=bool(supported and precise and interval[0]>0),
        interpretation='positive' if interval[0]>0 else 'negative' if interval[1]<0 else 'inconclusive')
    return result


def holm_adjust(pvalues,families):
    """Adjust exactly declared named families, never infer a family from a count."""
    adjusted={}
    for family in families:
        ordered=sorted(family,key=lambda name:pvalues[name])
        previous=0.
        for i,name in enumerate(ordered):
            p=pvalues[name]
            if not np.isfinite(p) or not 0<=p<=1: raise ValueError('Invalid family p-value')
            previous=max(previous,min(1.,(len(ordered)-i)*p)); adjusted[name]=previous
    return adjusted


def analyze_primary(study_dir: Path, protocol: FrozenProtocol) -> dict[str, Any]:
    """Rebuild from original protected rows and declared identities, never scores."""
    from ..protocol import _read
    value = _read(protocol.path)
    with relocated(value["payload"]["sources"]["root"], protocol.path.resolve().parent.parent):
        return _analyze_primary(study_dir, protocol)


def _previous_analysis_seconds(work):
    """Reduce recorded durations independently of filesystem enumeration order."""
    return math.fsum(_read(path)['seconds'] for path in work.glob('*.json'))


def _analyze_primary(study_dir, protocol):
    started=time.perf_counter_ns()
    payload=_payload(protocol); root=locate(payload['sources']['root']); study_dir=Path(study_dir)
    expected=protocol.path.parent/'confirmation'/protocol.protocol_hash
    if study_dir.resolve()!=expected.resolve(): raise ValueError('Study directory differs from frozen protocol')
    ledger=_read(study_dir/'confirmation.json')
    if ledger['protocol_hash']!=protocol.protocol_hash: raise ValueError('Study identity differs')
    raw=_read(_reference(root,payload['sources']['raw'])); config=resolve_experiment(raw['config'])
    declared=[(u,rep,rid) for u in payload['sources']['cohort']['units'] for rep,rid in enumerate(u['run_ids'])]
    if [(r['unit_id'],r['repetition'],r['run_id']) for r in ledger['runs']]!=[(u['unit_id'],j,rid) for u,j,rid in declared]:
        raise ValueError('Run ledger omits or changes declared cohort')
    lookup={r['run_id']:r for r in ledger['runs']}; runs=[]; predictions={}
    for unit,rep,rid in declared:
        teacher=unit['manifest']['teacher_id']
        run=dict(teacher_id=teacher,unit_id=unit['unit_id'],run_id=rid,status='incomplete',f_target_variance=fit_variance(root,unit))
        entry=lookup[rid]
        try:
            if entry['status']!='completed': raise ValueError('Declared run incomplete')
            completed=[a for a in entry['attempts'] if a['status']=='completed']
            if len(completed)!=1: raise ValueError('Ambiguous completed fitting attempt')
            pair_path=_reference(root,completed[0]['pair']); pair=_read(pair_path)
            expected_config=repetition_config(config,unit['seeds'],rep)
            expected_provenance=dict(protocol_hash=protocol.protocol_hash,unit_id=unit['unit_id'],
                run_id=rid,nominal_optimizer_seed=unit['seeds']['optimizers'][rep],
                effective_seeds=expected_config.to_dict()['seeds'],seed_policy=SEED_POLICY)
            if pair.get('confirmation')!=expected_provenance:
                raise ValueError('Fitting provenance differs from frozen seed/run commitment')
            data=load_development(root,unit)
            if pair['teacher_id']!=data.teacher_id or pair['data_hash']!=data.manifest.assignment_hash:
                raise ValueError('Fitting data differs from frozen teacher/development commitment')
            _validate_pilot_graph(pair,pair_path.parent,expected_config)
            from ..checkpoint import _descriptor
            parent=[child for child in pair['child_runs'] if child['run_id']==pair['parent_run_id']][0]
            _,parent_descriptor,_=_descriptor(locate(parent['path'], producer=True)/'checkpoints/best.json')
            for key in ('seeds','model','training','growth','data'):
                if parent_descriptor['state']['config'][key]!=expected_config.to_dict()[key]:
                    raise ValueError('Parent recipe differs from frozen configuration')
            audit_models(root,pair,pair_path.parent)
            from ..evaluation.protected import _directory
            if _read(_directory(protocol)/'runs'/rid/'link.json')['models']!=pair['models']:
                raise ValueError('Registered models differ from selected fitting graph')
            saved=load_evaluation(protocol,run_id=rid)
            if set(saved['predictions'])!=set(EVALUATION_METHODS): raise ValueError('Missing control predictions')
            predictions[rid]=saved; run['status']='completed'
        except (ValueError,KeyError,OSError,TypeError) as error:
            run['reason']=str(error)
            if entry['status']=='completed': run['status']='invalid'
        runs.append(run)
    report=summarize_cohort(payload, ledger, runs, predictions, protocol.protocol_hash)
    # Content-addressed reports preserve earlier incomplete/reanalysis snapshots.
    report['report_sha256']=canonical_hash(report)
    path=study_dir/f'primary-{report["report_sha256"]}.json'
    if not path.exists(): _atomic_json(path,report)
    # Variable work measurements are separate from deterministic scientific output.
    work=study_dir/'analysis-work'; work.mkdir(exist_ok=True)
    previous=_previous_analysis_seconds(work)
    elapsed=(time.perf_counter_ns()-started)/1e9
    base=ledger['costs'].get('combined_seconds')
    _atomic_json(work/f'{uuid.uuid4().hex}.json',dict(schema_version=1,protocol_hash=protocol.protocol_hash,
        report_sha256=report['report_sha256'],seconds=elapsed,previous_analysis_seconds=previous,
        total_study_work_seconds=base+previous+elapsed if base is not None else None,
        total_reason=None if base is not None else 'preparation or interrupted execution work unmeasured',
        boundary='source/model/graph and saved-row audit, aggregation/inference and report IO; excludes final work receipt',
        aggregation='disjoint completed analysis invocations plus preparation/confirmation total; no nested child timers'))
    return report


def summarize_cohort(payload, ledger, runs, predictions, protocol_hash):
    """Pure saved-row analysis after the caller has verified evidence authority."""
    teachers=aggregate_teachers(predictions,runs)
    complete=(ledger['status']=='completed' and len(teachers)==payload['design']['K']
              and all(r['status']=='completed' for r in teachers))
    statistical=payload['design']['statistical_policy']
    summary=summarize_effects([r['D'] for r in teachers if r['D'] is not None],statistical)
    summary['superiority']=bool(summary['superiority'] and complete)
    controls={}
    contrasts=[(role,'top') for role in EVALUATION_METHODS]+[
        ('ordinary_upper','static_partial'),('ordinary_upper','widen_upper')]
    for left,right in contrasts:
        effects=[r['metrics'][left]['normalized_mse']-r['metrics'][right]['normalized_mse']
            for r in teachers if r['status']=='completed']
        controls[f'{left}:{right}']=dict(summarize_effects(effects,statistical),superiority=False,claim='descriptive')
    policy=confirmation_policy(payload['sources']['scientific'])
    family_p={}
    for family in policy['holm_families']:
        for contrast in family:
            left,right=contrast.split(':')
            values=[r['metrics'][left]['normalized_mse']-r['metrics'][right]['normalized_mse']
                for r in teachers if r['status']=='completed']
            inference=summarize_effects(values,statistical)
            if inference['student_t_p'] is not None: family_p[contrast]=inference['student_t_p']
    families=policy['holm_families']
    family_complete=all(c in family_p for f in families for c in f)
    adjusted=holm_adjust(family_p,families) if complete and family_complete else {}
    if any('random:top' in family for family in families):
        summary['superiority']=bool(summary['superiority'] and adjusted.get('random:top',1.)<=statistical['alpha'])
    report=dict(schema_version=1,protocol_hash=protocol_hash,
        status='completed' if complete else 'resource_limited' if ledger['status']=='resource_limited' else 'incomplete',
        declared_teachers=payload['design']['K'],observed_teachers=len(teachers),valid_teachers=summary['independent_count'],
        declared_runs=len(runs),valid_runs=sum(r['status']=='completed' for r in runs),
        teachers=teachers,runs=runs,primary=summary,controls=controls,holm_adjusted=adjusted,
        holm_basis='Student-t sensitivity p-values; only explicit frozen families; not bootstrap p-values',
        costs=ledger['costs'],claim='placement-only' if summary['superiority'] else 'no completed-cohort superiority',
        practical_advantage='not established by the H1 placement contrast',
        policy=policy,planning=payload['design']['branch'])
    return report
