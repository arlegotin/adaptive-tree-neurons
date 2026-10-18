"""P06 suggestions from the pinned official regression/default sampler.

Use a fresh private RealMLP worker. This samples configurations without opening
labels or fitting models. Whole-process startup remains part of the eventual
search budget in addition to this function's measured body.
"""
import math
import time

import numpy as np

from treeneuron.config import canonical_hash
from .seeds import seed64

_FIELDS=('num_emb_type','add_front_scale','lr','p_drop','wd','plr_sigma','act','hidden_sizes')


def _plain(value):
    if isinstance(value,np.generic):return _plain(value.item())
    if isinstance(value,dict):
        if any(type(k) is not str for k in value):raise ValueError('Native sampler returned non-string keys')
        return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v) for v in value]
    if value is None or type(value) in (str,int,bool):return value
    if type(value) is float and math.isfinite(value):return value
    raise ValueError('Native sampler returned unsupported/nonfinite recipe value')


def _materialize(feature_schema,hpo_seed,indices):
    started=time.perf_counter_ns()
    if type(hpo_seed) is not int or not 0<=hpo_seed<2**63:
        raise ValueError('P06 requires a fixed63-bit HPO seed')
    from .realmlp_native import RealMLPConfig,_source,_dependencies,_runtime
    from .ordinary_neural import _runtime_identity
    # Validate schema before loading the private native stack.
    config=RealMLPConfig.parse(dict(schema='research/v2-realmlp-config',family='P06',
        mode='native_anchor',feature_schema=feature_schema,parameters={}))
    source=_source();_dependencies()
    from pytabkit.models.alg_interfaces.nn_interfaces import RealMLPParamSampler
    from pytabkit.models.sklearn.default_params import DefaultParams
    dependencies=_dependencies();runtime=_runtime_identity(_runtime())
    defaults=_plain(DefaultParams.RealMLP_TD_REG);suggestions=[]
    for attempt_index in indices:
        native_seed=None if attempt_index==0 else seed64(hpo_seed,'P06','official-default-sampler',str(attempt_index))
        sampled=defaults if native_seed is None else _plain(RealMLPParamSampler(
            is_classification=False,hpo_space_name='default').sample_params(native_seed))
        if (set(sampled)!=set(defaults) or any(sampled[k]!=defaults[k] for k in defaults if k not in _FIELDS)):
            raise ValueError('Official sampler changed a setting outside the qualified native default support')
        parameters={} if native_seed is None else {name:sampled[name] for name in _FIELDS}
        trial=RealMLPConfig.parse(dict(schema='research/v2-realmlp-config',family='P06',
            mode='native_anchor' if attempt_index==0 else 'search',feature_schema=config.feature_schema,parameters=parameters))
        suggestions.append(dict(family_id='P06',attempt_index=attempt_index,hpo_seed=hpo_seed,
            native_sampler_seed=native_seed,config=trial.to_state(),official_sampled_recipe=sampled,
            native_distribution=dict(is_classification=False,hpo_space_name='default'),
            native_fits=0,labels_read=[],eligible_predictive_result=False,protected_access_authorized=False))
    after=_dependencies();runtime_after=_runtime_identity(_runtime())
    if (source!=_source() or canonical_hash(dependencies)!=canonical_hash(after)
            or canonical_hash(runtime)!=canonical_hash(runtime_after)):
        raise ValueError('Source/native runtime/dependencies changed during RealMLP suggestion')
    return dict(schema='research/v2-realmlp-search-schedule',hpo_seed=hpo_seed,suggestions=suggestions,
        source=source,dependencies=after,runtime=runtime_after,
        charged_sampler_body_ns=time.perf_counter_ns()-started,
        timer_scope='Entire schedule preparation paid once before trials; enclosing process startup remains separately paid',
        native_fits=0,labels_read=[],eligible_predictive_result=False,protected_access_authorized=False)


def suggest_realmlp(feature_schema,*,hpo_seed,attempt_index):
    """Produce one source-authenticated suggestion; batch preparation is cheaper."""
    if type(attempt_index) is not int or not 0<=attempt_index<32:
        raise ValueError('P06 requires an attempted configuration index0–31')
    schedule=_materialize(feature_schema,hpo_seed,[attempt_index]);suggestion=schedule.pop('suggestions')[0]
    return {**schedule,**suggestion,'schema':'research/v2-realmlp-suggestion'}


def realmlp_schedule(feature_schema,*,hpo_seed):
    """Materialize all32 independent native choices under one checked identity.

    The official random sampler has no outcome feedback, so this preserves its
    exact choices while avoiding repeated identical dependency authentication.
    These suggestions are not32completed or attempted model fits. Search must
    pay preparation before its first eligible trial and retain all failures/caps.
    """
    return _materialize(feature_schema,hpo_seed,list(range(32)))
