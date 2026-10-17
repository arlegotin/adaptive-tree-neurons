"""Whole-teacher variance calibration and the pre-freeze design calculation.

Pure inputs can exercise scientific decision branches but are not a grant to
freeze. A consumer must verify evidence and the later protocol prerequisites.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any

import numpy as np
from scipy.stats import norm, t
from treeneuron.config import canonical_hash, canonical_json

CONTROLS = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6')
POLICIES = ('parent_updates', 'continuation_updates', 'patience', 'validation_interval',
            'logical_batch', 'physical_batch', 'bootstrap_iterations', 'probe_updates',
            'hpo_seconds', 'hpo_trials', 'timing_blocks', 'timing_relative_precision')
COSTS = ('max_seconds', 'max_storage_bytes', 'available_memory_bytes', 'peak_memory_bytes',
         'setup_seconds_per_teacher', 'seconds_per_repetition', 'storage_bytes_per_teacher',
         'storage_bytes_per_repetition', 'seconds_per_evaluation_row', 'bytes_per_evaluation_row')


@dataclass(frozen=True)
class PilotSummary:
    independent_teachers: int
    repetitions: int
    teacher_means: tuple[float, ...]
    repetition_variance: float | None
    paired_row_variance_upper: float | None
    between_teacher_variance_lower: float | None
    sd_interval: tuple[float, float] | None
    batch_runtimes: tuple[float, ...]
    h1_mean: float | None
    h1_interval: tuple[float, float] | None
    static_mean: float | None
    static_interval: tuple[float, float] | None
    endpoint_mc_error: float | None
    controls: tuple[str, ...]
    failed_pairs: int
    evidence_hashes: Mapping[str, str]
    purpose: str = 'engineering'
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DesignDecision:
    branch: str
    K_calc: int | None
    pilot_teachers: int
    K: int
    repetitions: int
    n_T: int | None
    delta_plan: float | None
    numeric_policies: Mapping[str, Any]
    budgets: Mapping[str, Any]
    forecast: Mapping[str, Any]
    power: Mapping[str, Any]
    precision_cost_curve: tuple[Mapping[str, Any], ...]
    limitations: tuple[str, ...]
    evidence_hashes: Mapping[str, str]
    statistical_policy: Mapping[str, Any]
    freeze_eligible: bool = False

    def __post_init__(self):
        if self.branch not in ('powered','estimation','blocked','calibrate'):
            raise ValueError('Invalid design branch')
        if self.branch in ('powered','estimation'):
            if set(POLICIES)-set(self.numeric_policies):
                raise ValueError('Selected design requires complete numeric policies')
            if set(COSTS)-set(self.budgets) or not self.forecast:
                raise ValueError('Selected design requires complete measured budgets and forecast')
            if self.n_T is None or self.n_T < 1 or self.K < 8 or self.repetitions < 2:
                raise ValueError('Selected design requires resolved numeric counts')
        if self.branch=='powered' and (self.K_calc is None or self.delta_plan is None
                or self.delta_plan<=0 or self.K < max(self.K_calc,self.pilot_teachers)):
            raise ValueError('Powered design requires effect/count calculation and cohort floor')

    def to_dict(self):
        return asdict(self)


def _interval(values, level=.95):
    return tuple(float(x) for x in np.quantile(values, [(1-level)/2, (1+level)/2]))


def _control_set(value, name):
    """Canonical control IDs, including explicitly declared additional anchors."""
    if not isinstance(value, (list, tuple)) or any(
            type(item) is not str or re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', item) is None
            for item in value):
        raise ValueError(f'{name}: controls must be a sequence of nonempty identifier strings')
    if len(set(value)) != len(value):
        raise ValueError(f'{name}: duplicate control identifiers')
    return set(value)


def summarize_pilot(records: Sequence[Mapping[str, Any]], *, batch_runtimes: Sequence[float],
                    bootstrap_iterations: int, seed: int, purpose='engineering') -> PilotSummary:
    """Average repeated optimizers within teacher and resample whole teachers.

    E row order/data is identical within each teacher. Row variance is computed
    AFTER averaging repetitions on each row, preserving their correlations.
    Conditional on a teacher, optimizer seeds and E rows are independent crossed
    draws. Their mean has nuisance variance A/R + B/N + C/(R*N), where C is
    optimizer-by-row interaction. Rep-mean variance/R + row-mean variance/N
    estimates A/R + B/N + 2*C/(R*N): deliberately over-subtract interaction.
    Subtract its upper bootstrap bound from the observed teacher-mean variance
    lower bound; unresolved intrinsic variance routes to finite-row estimation.
    """
    canonical_json({'records': records, 'batch_runtimes': batch_runtimes})
    if type(bootstrap_iterations) is not int or bootstrap_iterations < 200:
        raise ValueError('At least 200 bootstrap iterations required for endpoint diagnostics')
    if type(seed) is not int or seed < 0 or purpose not in ('engineering','scientific'):
        raise ValueError('Invalid bootstrap seed or purpose')
    groups = {}
    seen = set()
    recipe = None
    failed = 0
    for record in records:
        key = (record['teacher_id'], record['repetition'])
        if key in seen: raise ValueError('Duplicate teacher/repetition')
        seen.add(key)
        if record['status'] != 'completed':
            failed += 1
            continue
        if recipe is None: recipe = record['recipe_hash']
        if recipe != record['recipe_hash']: raise ValueError('Common cross-teacher recipe required')
        groups.setdefault(record['teacher_id'], []).append(record)
    if not groups: raise ValueError('No complete teacher measurements')
    lengths = {len(rows) for rows in groups.values()}
    if len(lengths) != 1: raise ValueError('Unbalanced repetitions cannot enter accepted pilot')
    repetitions = lengths.pop()
    means, static_means, within, rowvars, rowcounts = [], [], [], [], []
    controls = None
    for rows in groups.values():
        if {r['repetition'] for r in rows} != set(range(repetitions)):
            raise ValueError('Repetitions must be consecutive from zero')
        for key in ('data_hash','evaluation_hash'):
            if len({r[key] for r in rows}) != 1: raise ValueError('Fixed F/G/V/E required across repetitions')
        h = np.asarray([r['h1_rows'] for r in rows], dtype=float)
        s = np.asarray([r['static_rows'] for r in rows], dtype=float)
        if h.ndim != 2 or h.shape != s.shape or h.shape[1] < 2 or not np.isfinite([h,s]).all():
            raise ValueError('Require finite paired row differences and at least two E rows')
        means.append(float(h.mean()))
        static_means.append(float(s.mean()))
        within.append(float(h.mean(axis=1).var(ddof=1)) if repetitions > 1 else 0.)
        rowvars.append(float(h.mean(axis=0).var(ddof=1)))
        rowcounts.append(h.shape[1])
        for row in rows:
            observed = _control_set(row['controls'], 'record.controls')
            controls = observed if controls is None else controls & observed
    values, static = np.asarray(means), np.asarray(static_means)
    n = len(values)
    diagnostics = dict(effective_independent_count=n, optimizer_repetitions=repetitions,
        row_variance='variance across E rows after per-row repetition averaging; shared E correlations retained',
        bootstrap_unit='whole teacher; all inner repetitions remain together',
        bootstrap_iterations=bootstrap_iterations, bootstrap_seed=seed,
        student_t_df=n-1, coverage_limit='Empirical bootstrap and t sensitivity do not establish nominal coverage',
        future_namespaces='confirmatory teacher/data/observation namespaces reserved; pilot never reuses them',
        intrinsic_variance=dict(status='unresolved',
            assumptions='Conditional on teacher/development data: iid optimizer draws, iid E rows independent of optimizers, finite second moments; independent teachers',
            decomposition='D = teacher mean + optimizer main effect + E-row main effect + centered optimizer-by-E interaction',
            true_mean_noise='A/R + B/N + C/(R*N)',
            nuisance_allowance='rep-mean sample variance/R + row-mean sample variance/N; expectation A/R + B/N + 2*C/(R*N)',
            interaction_policy='conservative double counting; no residual subtraction',
            bound='max(0, observed teacher-mean variance 2.5% lower - mean nuisance allowance 97.5% upper)',
            tail_policy='Two 2.5% one-sided bootstrap tails; union bound needs no independence between endpoints',
            limitation='Conservative decomposition in expectation; empirical whole-teacher bootstrap bounds are conditional approximations, not finite-sample coverage guarantees',
            evaluation_rows_per_teacher=rowcounts))
    sd_ci = h_ci = s_ci = None
    mc = between_lower = row_upper = None
    if n >= 2:
        rng = np.random.default_rng(seed)
        indices = rng.integers(n, size=(bootstrap_iterations, n))
        draws = values[indices]
        sd = draws.std(axis=1,ddof=1)
        hmeans, smeans = draws.mean(axis=1), static[indices].mean(axis=1)
        sd_ci, h_ci, s_ci = _interval(sd), _interval(hmeans), _interval(smeans)
        component_endpoints = []
        if repetitions >= 2:
            # Shared E noise does not shrink as 1/R. Row-mean variance contains
            # its full main effect plus interaction/R. Rep-mean variance also
            # contains interaction/N; keeping both is conservative, unlike
            # subtracting an unstable estimated interaction component.
            nuisance = np.asarray(within)/repetitions + np.asarray(rowvars)/np.asarray(rowcounts)
            nuisance_draws = nuisance[indices].mean(axis=1)
            observed_lower = float(np.quantile(sd**2,.025))
            nuisance_upper = float(np.quantile(nuisance_draws,.975))
            between_lower = max(0., observed_lower-nuisance_upper)
            diagnostics['intrinsic_variance'].update(
                status='resolved' if between_lower>0 else 'unresolved',
                observed_teacher_variance_lower=observed_lower,
                mean_nuisance_allowance=float(nuisance.mean()),
                nuisance_allowance_per_teacher=nuisance.tolist(),
                nuisance_mean_upper=nuisance_upper)
            component_endpoints = [(sd**2,.025),(nuisance_draws,.975)]
        row_upper = float(np.quantile(np.asarray(rowvars)[indices].mean(axis=1),.975))
        # Ten independent RNG sections: MC standard error of endpoint estimators.
        endpoint_vectors = [np.concatenate([np.quantile(a,[.025,.975]) for a in
            (sd[part],hmeans[part],smeans[part])] +
            [np.atleast_1d(np.quantile(a[part],q)) for a,q in component_endpoints])
            for part in np.array_split(np.arange(bootstrap_iterations),10)]
        mc = float(np.std(endpoint_vectors,axis=0,ddof=1).max()/math.sqrt(10))
        margin = float(t.ppf(.975,n-1)*values.std(ddof=1)/math.sqrt(n))
        diagnostics.update(student_t_interval=(float(values.mean()-margin),float(values.mean()+margin)),
            median=float(np.median(values)), observed_teacher_variance=float(values.var(ddof=1)),
            within_teacher_variances=within, paired_row_variances=rowvars,
            variance_component_method='crossed optimizer/E nuisance upper bound subtracted from observed teacher variance lower bound',
            endpoint_mc_method='maximum endpoint SE over 10 disjoint bootstrap chunks; absolute mean/SD/variance endpoint values')
    return PilotSummary(n,repetitions,tuple(means),float(np.mean(within)) if repetitions>1 else None,
        row_upper,between_lower,sd_ci,tuple(batch_runtimes),float(values.mean()),h_ci,
        float(static.mean()),s_ci,mc,tuple(sorted(controls or ())),failed,
        {'raw_records':canonical_hash({'records':records})},purpose,diagnostics)


def calibration_status(summary: PilotSummary, endpoint_tolerance: float) -> tuple[bool, tuple[str,...]]:
    reasons=[]
    if summary.independent_teachers < 8: reasons.append('Initial pilot requires at least 8 independent teachers')
    if summary.repetitions < 2: reasons.append('Initial pilot requires at least 2 repetitions')
    if summary.failed_pairs: reasons.append('Unresolved failed pairs')
    if len(summary.batch_runtimes) < 2:
        reasons.append('Two completed runtime batches required')
    else:
        previous,current=summary.batch_runtimes[-2:]
        if previous <= 0 or current <= 0 or abs(current/previous-1) > .2+1e-12:
            reasons.append('Per-teacher runtime is not stable within 20%')
    if summary.sd_interval is not None and summary.sd_interval[1] > 0:
        low,high=summary.sd_interval
        if low <= 0 or high/low > 2+1e-12: reasons.append('Whole-teacher SD interval ratio exceeds 2')
        if summary.endpoint_mc_error is None or summary.endpoint_mc_error > endpoint_tolerance:
            reasons.append('Bootstrap endpoint Monte Carlo precision unresolved')
    return not reasons,tuple(reasons)


def _power(values, k, delta, sd, alpha, iterations, seed):
    centered=np.asarray(values,dtype=float)-np.mean(values)
    observed=float(centered.std(ddof=1))
    if k < 2 or observed <= 0:
        return dict(supported=False,reason='Degenerate empirical teacher distribution')
    centered *= sd/observed
    rng=np.random.default_rng(seed)
    success=0
    # Chunking avoids allocations proportional to iterations * potentially huge K.
    for _ in range(iterations):
        sample=centered[rng.integers(len(centered),size=k)]+delta
        se=float(sample.std(ddof=1))/math.sqrt(k)
        success += bool(se>0 and abs(float(sample.mean()))/se > t.ppf(1-alpha/2,k-1))
    p=success/iterations
    z=norm.ppf(.975)
    denominator=1+z*z/iterations
    center=(p+z*z/(2*iterations))/denominator
    half=z*math.sqrt(p*(1-p)/iterations+z*z/(4*iterations**2))/denominator
    return dict(supported=True,estimate=p,interval=(float(center-half),float(center+half)),
        mc_half_width=float(half),iterations=iterations,seed=seed,
        method='empirical centered whole-teacher resampling, scaled to SD upper bound, two-sided Student-t rejection',
        limitation='Conditional on observed empirical shape; cannot validate population coverage or unseen tails')


def choose_design(summary: PilotSummary, constraints: Mapping[str, Any]) -> DesignDecision:
    """Select numeric design, never silently truncate a floored cohort.

    Constraints describe measured feasibility and precision. No application
    utility threshold or equivalence margin is invented here.
    """
    c=dict(constraints)
    canonical_json(c); canonical_json(asdict(summary))
    for name in ('independent_teachers','repetitions','failed_pairs'):
        value=getattr(summary,name)
        if type(value) is not int or value < (0 if name=='failed_pairs' else 1):
            raise ValueError(f'Invalid summary count {name}')
    if summary.purpose not in ('scientific','engineering'):
        raise ValueError('Invalid summary purpose')
    for name in ('repetition_variance','paired_row_variance_upper','between_teacher_variance_lower','endpoint_mc_error'):
        value=getattr(summary,name)
        if value is not None and (type(value) not in (int,float) or value<0):
            raise ValueError(f'Invalid summary variance/precision {name}')
    for name in ('h1_interval','static_interval','sd_interval'):
        value=getattr(summary,name)
        if value is not None and (len(value)!=2 or value[0]>value[1]):
            raise ValueError(f'Invalid summary interval {name}')
    if 'inference_supported' in c and type(c['inference_supported']) is not bool:
        raise ValueError('inference_supported must be boolean')
    for name in ('endpoint_tolerance','power_mc_tolerance'):
        if name in c and (type(c[name]) not in (int,float) or c[name]<=0):
            raise ValueError(f'{name} must be positive numeric precision')
    allowed=set(COSTS)|{'retries','estimation_rows','alpha','families','adjusted_alpha',
        'inference_supported','power_iterations','bootstrap_seed','endpoint_tolerance',
        'power_mc_tolerance','required_controls','policies','evidence_hashes'}
    if set(c)-allowed: raise ValueError(f'Unknown constraints: {sorted(set(c)-allowed)}')
    if c.get('families',1)>1 and 'adjusted_alpha' not in c:
        raise ValueError('Additional families require explicit adjusted_alpha')
    alpha=c.get('adjusted_alpha',c.get('alpha',.05))
    if not 0 < alpha < 1: raise ValueError('Invalid alpha')
    if c.get('families',1)==1 and alpha != .05:
        raise ValueError('Sole H1 requires two-sided alpha .05')
    if c.get('families',1)>1 and alpha > c.get('alpha',.05):
        raise ValueError('adjusted_alpha cannot exceed nominal alpha')
    policies=c.get('policies',{})
    if set(policies)-set(POLICIES): raise ValueError('Unknown numeric policies')
    required=set(COSTS)|{'retries','estimation_rows','inference_supported','power_iterations',
        'bootstrap_seed','endpoint_tolerance','power_mc_tolerance','required_controls','evidence_hashes'}
    missing=sorted(required-set(c))
    missing += [f'policies.{name}' for name in POLICIES if name not in policies]
    reasons=[]
    for name in COSTS:
        if name in c and (type(c[name]) not in (int,float) or c[name] <= 0):
            raise ValueError(f'{name} must be positive measured numeric input')
    for name in ('retries','estimation_rows','power_iterations','bootstrap_seed','families'):
        if name in c and (type(c[name]) is not int or c[name] < (1 if name in ('estimation_rows','power_iterations','families') else 0)):
            raise ValueError(f'Invalid integer {name}')
    for name,value in policies.items():
        if name not in ('hpo_seconds','timing_relative_precision') and type(value) is not int:
            raise ValueError(f'{name} requires integer policy')
        if type(value) not in (int,float) or value < 0 or (value==0 and name not in ('hpo_seconds','hpo_trials')):
            raise ValueError(f'Invalid numeric policy {name}')
    if 'logical_batch' in policies and 'physical_batch' in policies and policies['logical_batch']!=policies['physical_batch']:
        raise ValueError('Unequal physical/logical batching is unsupported by current training')
    if policies.get('timing_relative_precision',.1)>=1:
        raise ValueError('timing_relative_precision must be below one')
    if policies.get('probe_updates',1)>policies.get('continuation_updates',1):
        raise ValueError('Probe allowance exceeds normal continuation')
    if policies.get('bootstrap_iterations',200)<200 or policies.get('timing_blocks',20)<20:
        raise ValueError('Bootstrap/timing precision diagnostics need at least 200 draws / 20 blocks')
    if len(summary.teacher_means)!=summary.independent_teachers:
        raise ValueError('Independent teacher count disagrees with means')
    if summary.independent_teachers < 1 or summary.repetitions < 1:
        raise ValueError('Positive observed counts required')
    if summary.sd_interval is not None and not 0 <= summary.sd_interval[0] <= summary.sd_interval[1]:
        raise ValueError('Invalid SD interval')
    n,k,r=summary.independent_teachers,summary.independent_teachers,summary.repetitions
    kcalc=delta=nt=None
    power={}; forecast={}; curve=()
    branch='estimation'
    if missing: reasons.append('Missing measured inputs: '+', '.join(missing))
    observed_controls = _control_set(summary.controls, 'summary.controls')
    declared_controls = _control_set(c.get('required_controls', []), 'required_controls')
    missing_controls = ((set(CONTROLS) - observed_controls) |
        (set(CONTROLS) - declared_controls) | (declared_controls - observed_controls))
    if missing_controls:
        reasons.append('Required baseline/control coverage or declaration missing: '+', '.join(sorted(missing_controls)))
    if summary.failed_pairs: reasons.append('Unresolved invalid pairs block design')
    if reasons: branch='blocked'
    else:
        accepted,calibration_reasons=calibration_status(summary,c['endpoint_tolerance'])
        if not accepted:
            branch='calibrate'; reasons.extend(calibration_reasons)
        else:
            unresolved=(summary.sd_interval is None or summary.sd_interval[1]<=0
                or summary.between_teacher_variance_lower is None or summary.between_teacher_variance_lower<=0
                or summary.paired_row_variance_upper is None)
            nt=c['estimation_rows'] if unresolved else max(1,math.ceil(
                100*summary.paired_row_variance_upper/summary.between_teacher_variance_lower))
            if unresolved:
                reasons.append('Zero/undefined variance: equivalent-model assessment or estimation; no equivalence margin supplied')
            else:
                for mean,interval,label in ((summary.h1_mean,summary.h1_interval,'H1'),
                        (summary.static_mean,summary.static_interval,'static-partial versus upper ordinary')):
                    if interval is not None and (interval[0]>0 or interval[1]<0) and mean is not None and mean!=0:
                        delta=.5*abs(mean); reasons.append('Planning effect from '+label); break
                if delta is not None:
                    sd=summary.sd_interval[1]
                    kcalc=max(1,math.ceil((norm.ppf(1-alpha/2)+norm.ppf(.8))**2*sd**2/delta**2))
                    k=max(kcalc,n)
                    branch='powered'
                else: reasons.append('Both intervals unresolved: fresh estimation cohort matches completed pilot counts')
            def cost(count):
                sec=count*(c['setup_seconds_per_teacher']+r*c['seconds_per_repetition']+nt*r*c['seconds_per_evaluation_row'])*(1+c['retries'])
                size=count*(c['storage_bytes_per_teacher']+r*c['storage_bytes_per_repetition']+nt*r*c['bytes_per_evaluation_row'])*(1+c['retries'])
                return dict(seconds=sec,storage_bytes=math.ceil(size))
            forecast=cost(k)
            approximate=cost(kcalc) if kcalc is not None else forecast
            forecast.update(K=k,K_calc=kcalc,pilot_teachers=n,repetitions=r,n_T=nt,
                floor_extra_seconds=forecast['seconds']-approximate['seconds'],
                floor_extra_storage_bytes=forecast['storage_bytes']-approximate['storage_bytes'],
                retries=c['retries'],peak_memory_bytes=c['peak_memory_bytes'],
                cost_boundary='Per-teacher setup plus complete repetitions including all controls/probes/failures; predictions charged per row; nested costs not added')
            feasible=(forecast['seconds']<=c['max_seconds'] and forecast['storage_bytes']<=c['max_storage_bytes']
                and c['peak_memory_bytes']<=c['available_memory_bytes'])
            if not feasible:
                branch='blocked'; reasons.append('Floored workload physically infeasible; no cohort truncation')
            elif branch=='powered':
                power=_power(summary.teacher_means,k,delta,summary.sd_interval[1],alpha,
                    c['power_iterations'],c['bootstrap_seed'])
                if (not c['inference_supported'] or not power.get('supported')
                    or power.get('mc_half_width',1)>c['power_mc_tolerance']
                    or power.get('interval',(0,0))[0]<.8):
                    branch='estimation'; reasons.append('Finite-sample/resampling inference or power precision unsupported')
            sd=summary.sd_interval[1] if summary.sd_interval else None
            if sd is not None:
                curve=tuple(dict(K=count,half_width=float(t.ppf(1-alpha/2,count-1)*sd/math.sqrt(count)),
                    detectable_effect=float((norm.ppf(1-alpha/2)+norm.ppf(.8))*sd/math.sqrt(count)),
                    **cost(count)) for count in (k,2*k,4*k))
            within=summary.repetition_variance
            between=summary.between_teacher_variance_lower
            repetition_cost=c['seconds_per_repetition']+nt*c['seconds_per_evaluation_row']
            optimum=(math.sqrt(within*c['setup_seconds_per_teacher']/
                (between*repetition_cost)) if within is not None and between and between>0 else None)
            candidates={2,r} if optimum is None else {2,r,max(2,math.floor(optimum)),max(2,math.ceil(optimum))}
            recommended=(min(candidates,key=lambda count:(between+within/count)*
                (c['setup_seconds_per_teacher']+count*repetition_cost))
                if optimum is not None else r)
            forecast['repetition_tradeoff']=dict(observed_within_variance=within,
                conservative_between_variance=between,
                continuous_cost_optimum=optimum,recommended_repetitions=recommended,
                requires_new_calibration=recommended!=r,
                selected=r,policy='Keep completed pilot repetitions; any expansion requires new variance/cost calibration, never count repeats as teachers')
    if summary.purpose!='scientific':
        branch='blocked'; reasons.append('Engineering fixtures are not genuine pilot/design evidence')
    statistical_policy={key:c[key] for key in ('bootstrap_seed','power_iterations','endpoint_tolerance',
        'power_mc_tolerance','retries','estimation_rows','inference_supported') if key in c}
    statistical_policy.update(alpha=alpha,families=c.get('families',1),
        interval='whole-teacher percentile bootstrap; Student-t sensitivity',
        power='two-sided t rejection under centered empirical teacher resampling',
        bootstrap_iterations=policies.get('bootstrap_iterations'))
    reasons.extend(['Fresh independent confirmation teachers and observations required',
        'Pilot-based count floor does not guarantee 95% coverage'])
    return DesignDecision(branch,kcalc,n,k,r,nt,delta,policies,{key:c[key] for key in COSTS if key in c},
        forecast,power,curve,tuple(reasons),dict(summary.evidence_hashes)|dict(c.get('evidence_hashes',{})),statistical_policy,False)
