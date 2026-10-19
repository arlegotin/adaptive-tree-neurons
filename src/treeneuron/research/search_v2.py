"""Pinned serial TPE event replay, including an interrupted pending suggestion.

Native anchors are outer-controller attempts outside this surrogate. The caller
must deduct every anchor/failure/retry from its32attempt total and enforce the
whole-search time budget. This class does no fitting, label access or admission.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from treeneuron.config import canonical_hash,canonical_json
from treeneuron.evaluation.search import _optuna,ROOT
from .ledger import write_new_json


def _copy(value):
    return json.loads(canonical_json(dict(value=value)))['value']


def _finite(value):
    try:return type(value) in (int,float) and math.isfinite(value)
    except OverflowError:return False


def _support(value):
    if not isinstance(value,dict) or not value or any(type(k) is not str or not k for k in value):
        raise ValueError('Explicit nonempty named TPE support required')
    for name,spec in value.items():
        if not isinstance(spec,dict):raise ValueError('Invalid TPE parameter support')
        kind=spec.get('kind')
        if kind=='float' and set(spec)=={'kind','low','high','log'}:
            valid=(_finite(spec['low']) and _finite(spec['high']) and spec['low']<=spec['high']
                and type(spec['log']) is bool and (not spec['log'] or spec['low']>0))
        elif kind=='integer' and set(spec)=={'kind','low','high','step','log'}:
            low,high,step=spec['low'],spec['high'],spec['step']
            valid=(all(type(v) is int for v in (low,high,step)) and -2**62<=low<=high<2**62
                and step>0 and (high-low)%step==0 and type(spec['log']) is bool
                and (not spec['log'] or low>0 and step==1))
        elif kind=='categorical' and set(spec)=={'kind','choices'}:
            choices=spec['choices']
            valid=(isinstance(choices,list) and bool(choices) and all(v is None or type(v) in (str,bool)
                or _finite(v) for v in choices))
            if valid:valid=all(v not in choices[:i] for i,v in enumerate(choices))
        else:valid=False
        if not valid:raise ValueError(f'Invalid TPE support or silently adjusted range: {name}')
    return _copy(value)


def _dependencies(optuna):
    overlay=(ROOT/'environment/search-site').resolve();origin=Path(optuna.__file__).resolve()
    if not origin.is_relative_to(overlay):raise ValueError('TPE import escaped the qualified private search overlay')
    return dict(optuna_version=optuna.__version__,optuna_module_path=str(origin),
        optuna_module_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),
        requirements_sha256=hashlib.sha256((ROOT/'locks/search/requirements.txt').read_bytes()).hexdigest(),
        environment_sha256=hashlib.sha256((ROOT/'locks/search/environment.json').read_bytes()).hexdigest(),
        numpy_version=np.__version__,python=sys.version)


class ReplayTPEV2:
    def __init__(self,*,seed,support,identity,max_suggestions=32):
        if (type(seed) is not int or not 0<=seed<2**32 or type(max_suggestions) is not int
                or not 1<=max_suggestions<=32):raise ValueError('TPE requires a uint32 seed and1–32suggestion limit')
        self._seed=seed;self._max=max_suggestions
        self._support=_support(support);self._identity=_copy(identity)
        if not isinstance(self._identity,dict) or not self._identity:raise ValueError('Caller execution identity required')
        self._optuna=_optuna();self._dependency_identity=_dependencies(self._optuna)
        self._distributions={}
        for name,spec in sorted(self._support.items()):
            if spec['kind']=='float':
                distribution=self._optuna.distributions.FloatDistribution(spec['low'],spec['high'],log=spec['log'])
            elif spec['kind']=='integer':
                distribution=self._optuna.distributions.IntDistribution(spec['low'],spec['high'],step=spec['step'],log=spec['log'])
            else:distribution=self._optuna.distributions.CategoricalDistribution(spec['choices'])
            self._distributions[name]=distribution
        sampler=self._optuna.samplers.TPESampler(seed=seed,prior_weight=1.,consider_magic_clip=True,
            consider_endpoints=False,n_startup_trials=10,n_ei_candidates=24,multivariate=False,group=False,
            warn_independent_sampling=True,constant_liar=False)
        self._study=self._optuna.create_study(direction='minimize',sampler=sampler)
        self._history=[];self._pending=None

    @property
    def history(self):return _copy(self._history)

    @property
    def dependency_identity(self):return _copy(self._dependency_identity)

    @property
    def pending_suggestion(self):return None if self._pending is None else _copy(self._pending.params)

    def ask(self):
        if self._pending is not None:raise ValueError('Previous suggestion remains pending')
        if len(self._history)>=self._max:raise ValueError('Paid adaptive suggestion limit reached')
        self._pending=self._study.ask(fixed_distributions=self._distributions)
        return _copy(self._pending.params)

    def tell(self,value):
        if self._pending is None:raise ValueError('No pending TPE suggestion')
        if value is not None and (not _finite(value) or value<0):
            raise ValueError('V normalized MSE must be finite/nonnegative or None for a paid failure')
        trial=self._pending
        self._study.tell(trial,values=value,state=self._optuna.trial.TrialState.FAIL if value is None else self._optuna.trial.TrialState.COMPLETE)
        self._history.append(dict(number=trial.number,params=_copy(trial.params),value=value));self._pending=None

    def save(self,path):
        if _dependencies(_optuna())!=self._dependency_identity:
            raise ValueError('Pinned TPE dependency identity changed')
        state=dict(schema='research/v2-tpe-state',seed=self._seed,support=self._support,identity=self._identity,
            max_suggestions=self._max,dependency_identity=self._dependency_identity,history=self._history,
            pending=None if self._pending is None else dict(number=self._pending.number,params=_copy(self._pending.params)))
        path=Path(path)
        if path.is_symlink():raise ValueError('Sampler state cannot be published through a symbolic link')
        write_new_json(path,dict(state=state,sha256=canonical_hash(state)))
        return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    @classmethod
    def load(cls,path,*,expected_sha256,identity):
        path=Path(path)
        if (path.is_symlink() or not path.is_file() or type(expected_sha256) is not str
                or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256):
            raise ValueError('Sampler file commitment differs from its declared parent reference')
        payload=json.loads(path.read_text())
        if (not isinstance(payload,dict) or set(payload)!={'state','sha256'}
                or canonical_hash(payload['state'])!=payload['sha256']):raise ValueError('Sampler payload commitment mismatch')
        state=payload['state'];fields={'schema','seed','support','identity','max_suggestions','dependency_identity','history','pending'}
        if (not isinstance(state,dict) or set(state)!=fields or state['schema']!='research/v2-tpe-state'
                or canonical_json(state['identity'])!=canonical_json(identity) or not isinstance(state['history'],list)):
            raise ValueError('Sampler schema or caller execution identity changed')
        obj=cls(seed=state['seed'],support=state['support'],identity=identity,max_suggestions=state['max_suggestions'])
        if obj._dependency_identity!=state['dependency_identity']:raise ValueError('Sampler dependency identity changed')
        for index,row in enumerate(state['history']):
            if (not isinstance(row,dict) or set(row)!={'number','params','value'}
                    or type(row['number']) is not int or row['number']!=index or canonical_json(obj.ask())!=canonical_json(row['params'])):
                raise ValueError('Sampler replay differs from archived suggestion')
            obj.tell(row['value'])
        pending=state['pending']
        if pending is not None:
            if (not isinstance(pending,dict) or set(pending)!={'number','params'} or type(pending['number']) is not int
                    or pending['number']!=len(obj._history) or canonical_json(obj.ask())!=canonical_json(pending['params'])):
                raise ValueError('Pending sampler replay differs from archived suggestion')
        return obj
