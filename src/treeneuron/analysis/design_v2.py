"""Prospective teacher-level variance calibration; no permission to freeze.

The legacy effect/significance planning rule is deliberately left unchanged.
Simulation/finite-t/resource validation and immutable protocols are separate gates.
"""
import hashlib
import math

import numpy as np
from scipy.stats import chi2, norm

USEFUL_MARGIN=math.log(1/.98)
DESIGN_ALTERNATIVE=math.log(1/.95)
TEACHER_GRID=tuple(range(32,129,8))


def teacher_log_benefits(records):
    """Average exactly R2 risks within teacher/role, then take the log ratio.

    Preserve first teacher appearance in the declared attempt order; inferred
    lexical ID ordering cannot preserve an initial16 prefix when32 exist.
    Inputs are already original-unit/F-variance-normalized retained risks.
    This arithmetic helper does not authenticate their data or model provenance.
    """
    groups={}
    for row in records:
        teacher=row.get('teacher_id');repeat=row.get('repetition');role=row.get('role');risk=row.get('risk')
        if (type(teacher) is not str or not teacher or type(repeat) is not int or repeat not in (0,1)
                or role not in ('random','selected') or row.get('status')!='completed_eligible'
                or type(risk) not in (int,float) or not math.isfinite(risk) or risk<0):
            raise ValueError('Requires eligible finite teacher/R2 role risks without missing-case filtering')
        key=(repeat,role);group=groups.setdefault(teacher,{})
        if key in group:raise ValueError('Duplicate teacher/repeat/role risk')
        group[key]=risk
    if not groups:raise ValueError('No teacher risks')
    expected={(r,role) for r in range(2) for role in ('random','selected')};result=[]
    for teacher,values in groups.items():
        if set(values)!=expected:raise ValueError('Missing planned teacher/repeat/role risk')
        risks={role:math.fsum(values[r,role]/2 for r in range(2)) for role in ('random','selected')}
        if any(v<=0 for v in risks.values()):raise ValueError('Teacher log-risk benefit is undefined at zero risk')
        result.append(dict(teacher_id=teacher,random_risk=risks['random'],selected_risk=risks['selected'],
            log_benefit=math.log(risks['random'])-math.log(risks['selected'])))
    return result



def _centered_coordinates(values):
    raw=np.array(values,dtype=np.float64,copy=True)
    try:
        with np.errstate(over='raise',invalid='raise'):
            shifted=raw-raw[0]
            centered=shifted-np.mean(shifted)
    except FloatingPointError as error:
        raise ValueError('Teacher effect centering is not representable') from error
    if not np.isfinite(centered).all():raise ValueError('Nonfinite centered teacher effects')
    return raw,centered


def calibrate_variance(effects, *, runtime_prediction, seed):
    """Fixed10,000 whole-teacher resamples; extend16->32 only for precision.

    The supplied runtime P50/P90 is a declared forecast, not inferred quantiles
    or a guarantee of future dedicated runtime. Its provenance is validated by
    the eventual protocol consumer, not this pure function.
    """
    values=np.asarray(effects)
    if (values.ndim!=1 or len(values) not in (16,32) or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or type(seed) is not int or not 0<=seed<2**63
            or not isinstance(runtime_prediction,dict) or set(runtime_prediction)!={'p50_seconds','p90_seconds'}):
        raise ValueError('Calibration requires16 or32 independent teacher effects, seed and runtime forecast')
    p50,p90=(runtime_prediction[k] for k in ('p50_seconds','p90_seconds'))
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (p50,p90)) or p90<p50:
        raise ValueError('Invalid runtime forecast')
    raw_effects,values=_centered_coordinates(values);n=len(values);sd=float(np.std(values,ddof=1))
    if not math.isfinite(sd):raise ValueError('Teacher SD is not representable')
    indices=np.random.default_rng(seed).integers(0,n,size=(10000,n))
    bootstrap=np.std(values[indices],axis=1,ddof=1)
    low,high=(float(v) for v in np.quantile(bootstrap,[.05,.95],method='linear'))
    bootstrap_upper=float(np.quantile(bootstrap,.9,method='linear'))
    normal_upper=math.sqrt((n-1)*sd*sd/float(chi2.ppf(.1,n-1)))
    upper=max(normal_upper,bootstrap_upper)
    reasons=[]
    if low==0:reasons.append('bootstrap_SD_lower_zero')
    elif high/low>2:reasons.append('bootstrap_SD_interval_ratio_above_two')
    if p90/p50>2:reasons.append('runtime_P90_P50_above_two')
    identifiable=upper>0 and low>0
    if not identifiable and not reasons:reasons.append('unidentifiable_variance')
    extend=n==16 and bool(reasons)
    return dict(schema='research/v2-variance-calibration',teacher_count=n,repetitions_per_teacher=2,
        effects_float64_sha256=hashlib.sha256(raw_effects.astype('<f8').tobytes(order='C')).hexdigest(),
        effect_commitment_encoding='ordered little-endian float64 C bytes before centering',
        sample_SD=sd,bootstrap_SD_interval_90=[low,high],bootstrap_SD_percentile_90=bootstrap_upper,
        normal_theory_SD_upper_90=normal_upper,s_U=upper,bootstrap_resamples=10000,bootstrap_seed=seed,
        runtime_prediction=dict(runtime_prediction),precision_failure_reasons=reasons,
        extension_required=extend,extension_exhausted=n==32,variance_identifiable=identifiable,
        all_teacher_effects_exactly_zero=bool(np.all(raw_effects==0)),
        status='extend_once' if extend else 'estimation_only' if reasons else 'variance_ready',
        useful_margin=USEFUL_MARGIN,design_alternative=DESIGN_ALTERNATIVE,
        assumptions=['Normal-theory SD bound and whole-teacher bootstrap are assumption-dependent',
            'Neither is a distribution-free variance guarantee; effect mean/sign never sets the alternative'],
        freeze_eligible=False,T_accessed=False)


def starting_teacher_count(sd_upper, *, feasible_maximum=128):
    """Normal-approximation starting grid only, not validated power or a freeze."""
    if (type(sd_upper) not in (int,float) or not math.isfinite(sd_upper) or sd_upper<=0
            or type(feasible_maximum) is not int or not 32<=feasible_maximum<=128):
        raise ValueError('Positive identifiable SD and a32–128 feasibility ceiling required')
    required=math.ceil(((float(norm.ppf(.975))+float(norm.ppf(.9)))*sd_upper/
        (DESIGN_ALTERNATIVE-USEFUL_MARGIN))**2)
    feasible=[k for k in TEACHER_GRID if k<=feasible_maximum]
    count=next((k for k in feasible if k>=required),None)
    return dict(schema='research/v2-starting-size',K_calc=required,K=count,feasible_grid=feasible,
        useful_margin=USEFUL_MARGIN,design_alternative=DESIGN_ALTERNATIVE,target_power=.9,
        one_sided_alpha=.025,status='simulation_required' if count else 'estimation_only',
        largest_feasible_K=feasible[-1],finite_t_validated=False,simulation_validated=False,
        resource_validated=False,freeze_eligible=False)


def binomial_interval(successes, trials):
    """Two-sided95% exact Clopper–Pearson interval, including both endpoints."""
    from scipy.stats import beta
    if (type(successes) is not int or type(trials) is not int or trials<=0
            or not 0<=successes<=trials):raise ValueError('Invalid binomial counts')
    return [0. if successes==0 else float(beta.ppf(.025,successes,trials-successes+1)),
            1. if successes==trials else float(beta.ppf(.975,successes+1,trials-successes))]


def simulate_t_gate(effects, *, sd_upper, K, seed, chunk_size=2048):
    """Validate one fixed K using100,000 cohorts per declared centered law.

    Null and alternative reuse each centered simulated cohort. The pilot effect
    mean is removed; it is never the prospective alternative. Empirical atoms
    are scaled by their population SD so their resampling law has variance s_U².
    These pure simulations neither authenticate calibration nor authorize T.
    """
    from scipy.stats import nct, t
    from treeneuron.research.seeds import seed64
    values=np.asarray(effects)
    if (values.ndim!=1 or len(values) not in (16,32) or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or type(sd_upper) not in (int,float)
            or not math.isfinite(sd_upper) or sd_upper<=0 or type(K) is not int or K not in TEACHER_GRID
            or type(seed) is not int or not 0<=seed<2**63 or type(chunk_size) is not int or not 1<=chunk_size<=100000):
        raise ValueError('Fixed teacher grid/variance/calibration effects and seed required')
    _,centered=_centered_coordinates(values);scale=float(np.std(centered,ddof=0))
    if not math.isfinite(scale) or scale<=0:raise ValueError('Empirical teacher variance is unidentifiable')
    atoms=centered/scale*sd_upper
    critical=float(t.ppf(.975,K-1));shift=DESIGN_ALTERNATIVE-USEFUL_MARGIN
    finite_power=float(nct.sf(critical,K-1,shift*math.sqrt(K)/sd_upper));laws={}
    for law in ('empirical','normal','t5'):
        law_seed=seed64(seed,'design-v2-power',str(K),law);rng=np.random.default_rng(law_seed)
        counts=dict(power=0,null=0)
        for start in range(0,100000,chunk_size):
            shape=(min(chunk_size,100000-start),K)
            if law=='empirical':sample=atoms[rng.integers(0,len(atoms),size=shape)]
            elif law=='normal':sample=rng.standard_normal(shape)*sd_upper
            else:sample=rng.standard_t(5,size=shape)*(sd_upper*math.sqrt(3/5))
            means=np.mean(sample,axis=1);se=np.std(sample,axis=1,ddof=1)/math.sqrt(K)
            if not np.isfinite(means).all() or not np.isfinite(se).all():
                raise ValueError('Nonfinite simulated t statistic')
            # This also defines strict rejection correctly for zero empirical SE.
            counts['null']+=int(np.count_nonzero(means>critical*se))
            counts['power']+=int(np.count_nonzero(means+shift>critical*se))
        row=dict(seed=law_seed)
        for name,count in counts.items():
            low,high=binomial_interval(count,100000)
            row[name]=dict(rejections=count,trials=100000,estimate=count/100000,lower95=low,upper95=high)
        laws[law]=row
    passed=finite_power>=.9 and all(row['power']['lower95']>=.9 and row['null']['upper95']<=.03
        for row in laws.values())
    return dict(schema='research/v2-fixed-t-power-validation',K=K,teacher_calibration_count=len(values),
        sd_upper=sd_upper,useful_margin=USEFUL_MARGIN,design_alternative=DESIGN_ALTERNATIVE,
        one_sided_alpha=.025,critical_t=critical,finite_t_power=finite_power,laws=laws,
        simulated_cohorts_per_law=100000,seed=seed,chunk_size=chunk_size,
        empirical_scaling='centered atoms; population SD matches s_U; no pilot mean or effect-sign selection',
        validation_passed=passed,status='validated_at_fixed_K' if passed else 'failed_at_fixed_K',
        alternative_analysis_admitted=False,resource_validated=False,freeze_eligible=False,T_accessed=False)


def choose_teacher_count(calibration, effects, *, seed, feasible_maximum=128, initial_calibration=None):
    """Walk only the prespecified grid; retain failed gates; never freeze.

    The external method/variance-script lock, raw calibration authenticity and
    measured resource forecast remain required even if all power checks pass.
    """
    from treeneuron.config import canonical_hash
    if (not isinstance(calibration,dict) or type(seed) is not int or not 0<=seed<2**63
            or type(feasible_maximum) is not int or not 32<=feasible_maximum<=128):
        raise ValueError('Invalid fixed design/calibration')
    recomputed=calibrate_variance(effects,runtime_prediction=calibration['runtime_prediction'],
        seed=calibration['bootstrap_seed'])
    if canonical_hash(calibration)!=canonical_hash(recomputed):
        raise ValueError('Calibration differs from its retained teacher effects and fixed policy')
    extended=calibration['teacher_count']==32
    if extended:
        if not isinstance(initial_calibration,dict) or initial_calibration.get('teacher_count')!=16:
            raise ValueError('A32-teacher design requires its triggering initial16 calibration')
        first=calibrate_variance(np.asarray(effects)[:16],runtime_prediction=initial_calibration['runtime_prediction'],
            seed=initial_calibration['bootstrap_seed'])
        if (canonical_hash(first)!=canonical_hash(initial_calibration)
                or not first['extension_required']):
            raise ValueError('Calibration extension lacks the exact ordered prefix and triggering rule')
    elif initial_calibration is not None:raise ValueError('Initial calibration cannot extend itself')
    feasible=[k for k in TEACHER_GRID if k<=feasible_maximum]
    result=dict(schema='research/v2-teacher-size-decision' ,calibration_hash=canonical_hash(calibration),
        useful_margin=USEFUL_MARGIN,design_alternative=DESIGN_ALTERNATIVE,
        extension_lineage_required=extended,extension_lineage_validated=extended,
        initial_calibration_hash=canonical_hash(initial_calibration) if extended else None,
        feasible_grid=feasible,simulations=[],seed=seed,repetitions=2,K=feasible[-1],
        status='estimation_only',resource_validated=False,freeze_eligible=False,T_accessed=False)
    if calibration['extension_required']:
        return dict(result,K=None,status='calibration_extension_required')
    if calibration['status']!='variance_ready':return result
    starting=starting_teacher_count(calibration['s_U'],feasible_maximum=feasible_maximum)
    result['starting_size']=starting
    if starting['K'] is None:return result
    for K in feasible:
        if K<starting['K']:continue
        simulated=simulate_t_gate(effects,sd_upper=calibration['s_U'],K=K,seed=seed)
        result['simulations'].append(simulated)
        if simulated['validation_passed']:
            return dict(result,K=K,status='power_validated_pending_resources')
    return result
