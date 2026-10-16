"""Serial, costed exploratory search. Only this coordinator reads V labels.

The enclosing clock includes serialization, sampler work, imports, fitting,
scoring/probing, checkpoint publication and V evaluation. Per-trial clocks are
nested diagnostics, never added to the enclosure. Acquisition is external.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import numpy as np
from treeneuron.artifacts import RunPaths, _atomic_json, append_event, finish_run, update_run_observations
from treeneuron.config import canonical_hash, canonical_json, resolve_experiment
from treeneuron.data.types import DevelopmentData, array_hash

ROOT = Path(__file__).resolve().parents[3]
COST_BOUNDARY = ('Enclosing serial search: input publication, sampler/replay, isolated process startup/import, '
    'F preprocessing, fresh parent/base, score and all probes, continuation, compilation, G selection, '
    'checkpoint/save, V prediction/scoring, failed/rejected work and trial/sampler publication. '
    'Ends after final sampler publication, before terminal manifest/metrics publication. Nested durations must not be added. Data/weight acquisition precedes this boundary.')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _optuna():
    # Append: the overlay cannot shadow the already verified core distributions.
    overlay = str(ROOT/'environment/search-site')
    if overlay not in sys.path: sys.path.append(overlay)
    import optuna
    locked = json.loads((ROOT/'locks/search/environment.json').read_text())
    if file_hash(ROOT/'locks/search/requirements.txt') != locked['requirements_sha256']:
        raise RuntimeError('Search requirements lock hash mismatch')
    for name, expected in locked['packages'].items():
        if version(name) != expected: raise RuntimeError(f'Search dependency mismatch: {name}')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


@dataclass(frozen=True)
class SearchBudget:
    seconds: float
    max_trials: int = 1
    max_updates: int | None = None
    recipe: dict[str, Any] = field(default_factory=dict)
    support: dict[str, Any] | None = None
    resume_sampler: str | None = None
    trial_order: str = 'official_first_then_seeded_tpe'
    checkpoint_selection: str = 'G_only'
    backend_report: str = 'environment/search-backend.json'

    def __post_init__(self):
        if type(self.seconds) not in (int,float) or not math.isfinite(self.seconds) or self.seconds<=0:
            raise ValueError('seconds must be finite and positive')
        if type(self.max_trials) is not int or self.max_trials<=0: raise ValueError('max_trials must be positive int')
        if self.max_updates is not None and (type(self.max_updates) is not int or self.max_updates<=0):
            raise ValueError('max_updates must be positive int')
        if self.trial_order!='official_first_then_seeded_tpe' or self.checkpoint_selection!='G_only':
            raise ValueError('Require official anchors first and G-only checkpoint selection')
        canonical_json(asdict(self))


@dataclass(frozen=True)
class SearchResult:
    winner: int | None
    trials: tuple[dict[str, Any], ...]
    charged_ns: int
    sampler_state: str | None
    status: str
    identity: dict[str, Any]


class ReplayTPE:
    """Exact JSON event replay under a pinned Optuna implementation, no pickle.

    Replays every ask/tell, including failures and anchors, checking each sampled
    configuration. This restores both TPE and startup RNG trajectories without
    depending on private RNG serialization. A running ask cannot be published.
    """
    def __init__(self, seed, support, identity):
        if type(seed) is not int or not 0<=seed<2**32: raise ValueError('seed outside uint32')
        canonical_json(support); canonical_json(identity)
        self.optuna=_optuna(); self.seed=seed
        self.support=json.loads(canonical_json(support)); self.identity=json.loads(canonical_json(identity))
        self.history=[]; self.pending=None
        self.distributions={}
        for key,spec in sorted(self.support.items()):
            if spec.get('kind')=='float' and set(spec)=={'kind','low','high','log'}:
                if type(spec['log']) is not bool:raise ValueError('log must be bool')
                self.distributions[key]=self.optuna.distributions.FloatDistribution(spec['low'],spec['high'],log=spec['log'])
            elif spec.get('kind')=='categorical' and set(spec)=={'kind','choices'}:
                self.distributions[key]=self.optuna.distributions.CategoricalDistribution(spec['choices'])
            else: raise ValueError('Unknown search support schema')
        self.study=self.optuna.create_study(direction='minimize',sampler=self.optuna.samplers.TPESampler(
            seed=seed,prior_weight=1.,consider_magic_clip=True,consider_endpoints=False,n_startup_trials=10,
            n_ei_candidates=24,multivariate=False,group=False,warn_independent_sampling=True,constant_liar=False))

    def ask(self, anchor=None):
        if self.pending is not None:raise ValueError('Previous trial has no outcome')
        if anchor is not None:
            if set(anchor)!=set(self.support):raise ValueError('Anchor must specify every support parameter')
            self.study.enqueue_trial(dict(anchor))
        trial=self.study.ask(fixed_distributions=self.distributions)
        self.pending=(trial,None if anchor is None else dict(anchor))
        return dict(trial.params)

    def tell(self, value):
        if self.pending is None:raise ValueError('No pending trial')
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value)):
            raise ValueError('Objective must be finite or failed')
        trial,anchor=self.pending
        self.study.tell(trial,values=value,state=self.optuna.trial.TrialState.FAIL if value is None else self.optuna.trial.TrialState.COMPLETE)
        self.history.append(dict(params=dict(trial.params),anchor=anchor,value=value))
        self.pending=None

    def save(self,path):
        if self.pending is not None:raise ValueError('Cannot archive a pending ask')
        state=dict(schema_version=1,optuna_version=self.optuna.__version__,seed=self.seed,support=self.support,
                   identity=self.identity,history=self.history)
        _atomic_json(Path(path),dict(state=state,sha256=canonical_hash(state)))

    @classmethod
    def load(cls,path,*,identity):
        payload=json.loads(Path(path).read_text());state=payload['state']
        if set(payload)!={'state','sha256'} or canonical_hash(state)!=payload['sha256']:raise ValueError('Sampler hash mismatch')
        if set(state)!={'schema_version','optuna_version','seed','support','identity','history'} or state['schema_version']!=1:
            raise ValueError('Sampler schema mismatch')
        if canonical_hash(state['identity'])!=canonical_hash(identity):raise ValueError('Sampler identity mismatch')
        obj=cls(state['seed'],state['support'],identity)
        if state['optuna_version']!=obj.optuna.__version__:raise ValueError('Sampler version mismatch')
        for row in state['history']:
            if set(row)!={'params','anchor','value'} or obj.ask(row['anchor'])!=row['params']:
                raise ValueError('Sampler replay differs from archived parameters')
            obj.tell(row['value'])
        return obj


def _recipe(method, budget, experiment):
    recipe=dict(budget.recipe)
    if method=='Mean':
        if recipe:raise ValueError('Mean has no recipe keys')
        anchor={}; support={}
    elif method in ('Ridge','SplineRidge'):
        if set(recipe)-({'n_knots'} if method=='SplineRidge' else set()):raise ValueError('Unknown recipe keys')
        anchor={'alpha':1.};support={'alpha':dict(kind='float',low=1e-6,high=1e6,log=True)}
    elif method=='B7':
        if recipe:raise ValueError('B7 search retains the immutable year_node_shallow anchor')
        if budget.max_updates is None:raise ValueError('B7 requires explicit max_updates')
        anchor={};support={}
    elif method in ('B1','B2','B3','B4','B5','B6'):
        allowed={'B1':set(),'B2':{'variant','target_parameters','added_units'},'B3':{'variant','count','coordinates'},
                 'B4':set(),'B5':{'variant'},'B6':{'probe_updates'}}[method]
        if set(recipe)-allowed:raise ValueError('Unknown causal recipe keys')
        defaults={'B1':{},'B2':{'variant':'lower','target_parameters':1697},
                  'B3':{'variant':'partial','count':experiment['growth']['m_requested'],'coordinates':'centered'},
                  'B4':{},'B5':{'variant':'adam'},'B6':{'probe_updates':experiment['training']['continuation_updates']}}
        if method=='B2' and recipe.get('variant')=='widen':defaults[method]={'variant':'widen'}
        if method=='B3' and recipe.get('variant')=='full':defaults[method]={'variant':'full','coordinates':'centered'}
        recipe={**defaults[method],**recipe}
        anchor={k:experiment['training'][k] for k in ('learning_rate','weight_decay')}
        support={'learning_rate':dict(kind='float',low=1e-5,high=1e-1,log=True),
                 'weight_decay':dict(kind='categorical',choices=[0.,.0001,.001,.01,.1])}
        if budget.max_updates is not None:raise ValueError('Causal allowances come from the resolved experiment')
    else:raise ValueError(f'unsupported search method {method}; pending adapters cannot be searched')
    if budget.support is not None:
        if set(budget.support)!=set(support):raise ValueError('Support may expand implemented parameters only')
        support=budget.support
    return recipe,anchor,support


def _run_worker(request,directory,timeout):
    _atomic_json(directory/'request.json',dict(payload=request,sha256=canonical_hash(request)))
    env=dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    env['PYTORCH_ENABLE_MPS_FALLBACK']='0'
    env['PYTHONPATH']=str(ROOT/'src')
    with (directory/'stdout.txt').open('wb') as out,(directory/'stderr.txt').open('wb') as err:
        try:
            run=subprocess.run([sys.executable,'-m','treeneuron.evaluation.search_worker',str(directory/'request.json')],
                cwd=ROOT,env=env,stdout=out,stderr=err,timeout=max(timeout,1e-9))
        except subprocess.TimeoutExpired as exc:
            error=TimeoutError('Worker exceeded remaining search budget; process killed and waited')
            for manifest_path in (directory/'runs').rglob('manifest.json'):
                try:
                    manifest=json.loads(manifest_path.read_text())
                    if manifest['identity']['status'] not in ('completed','failed','interrupted'):
                        finish_run(RunPaths(manifest_path.parent.name,manifest_path.parent),
                            {'error':{'type':'TimeoutError','message':str(error)}},'interrupted')
                except Exception as publication_error:error.add_note(f'Child closure failed: {publication_error!r}')
            raise error from exc
    response=directory/'response.json'
    if not response.exists():raise RuntimeError(f'Worker exited {run.returncode} without response; retained stderr')
    payload=json.loads(response.read_text())
    if set(payload)!={'payload','sha256'} or canonical_hash(payload['payload'])!=payload['sha256']:
        raise ValueError('Worker response hash mismatch')
    result=payload['payload']
    if run.returncode or result.get('error'):raise WorkerError(result)
    return result


class WorkerError(RuntimeError):
    def __init__(self,result):
        self.result=result
        super().__init__(result.get('error',{}).get('message','worker failed'))


def search(method_id: str,data: DevelopmentData,budget: SearchBudget,seed: int,paths: RunPaths)->SearchResult:
    started=time.perf_counter_ns(); trials=[]; identity={'method':method_id,'seed':seed}; sampler=None; winner=None
    try:
        append_event(paths,{'type':'search_started','status':'running','method':method_id})
        update_run_observations(paths,{'method':{'scope':'coordinator'}})
        _atomic_json(paths.directory/'search.budget.json',asdict(budget))
        experiment=json.loads((paths.directory/'config.resolved.json').read_text())
        recipe,anchor,support=_recipe(method_id,budget,experiment)
        from treeneuron.train import source_hash,lock_hash
        from treeneuron.backend import runtime_hash
        identity=dict(source=source_hash(),runtime=runtime_hash(),search_environment=file_hash(ROOT/'locks/search/environment.json'),core_lock=lock_hash(),search_lock=file_hash(ROOT/'locks/search/requirements.txt'),
            method=method_id,recipe=recipe,experiment=experiment,split=data.manifest.to_dict(),seed=seed,
            max_updates=budget.max_updates,backend_report_sha256=file_hash(ROOT/budget.backend_report)
                if method_id in ('B1','B2','B3','B4','B5','B6') else None)
        _atomic_json(paths.directory/'search.identity.json',identity)
        update_run_observations(paths,{'identity':{'code_revision':identity['source'],'lock_hash':identity['core_lock']},
            'data':{'raw_hash':data.manifest.dataset_id,'split_hash':data.manifest.assignment_hash}})
        sampler=(ReplayTPE.load(budget.resume_sampler,identity=identity) if budget.resume_sampler else ReplayTPE(seed,support,identity))
        if sampler.seed!=seed or sampler.support!=support:raise ValueError('Resume seed/support mismatch')
        # Resume is a new attempt with a new wall budget; historical outcomes are
        # sampler evidence only. Historical model artifacts cannot silently win.
        arrays={f'{p}_{k}':getattr(getattr(data,p),attr) for p in 'FGV'
                for k,attr in [('X','X'),('ids','row_id')]+([] if p=='V' else [('y','y')])}
        np.savez(paths.directory/'inputs.npz',**arrays)
        array_refs={k:array_hash(v) for k,v in arrays.items()}
        prior_count=len(sampler.history)
        for number in range(budget.max_trials):
            remaining=budget.seconds-(time.perf_counter_ns()-started)/1e9
            if remaining<=0:break
            trial_start=time.perf_counter_ns(); directory=paths.directory/f'trial-{number:04d}';directory.mkdir()
            official=number==0 and prior_count==0
            params=sampler.ask(anchor if official else None)
            config={**recipe,**params}
            record=dict(number=number,sampler_number=prior_count+number,official_anchor=official,config=config,
                        status='running',validation_mse=None,error=None,resource_summary={},artifact_directory=str(directory))
            try:
                request=dict(schema_version=1,method=method_id,config=config,seed=seed,experiment=experiment,
                    max_updates=budget.max_updates,backend_report=str((ROOT/budget.backend_report).resolve()),
                    identity=identity,arrays=array_refs,input_path=str((paths.directory/'inputs.npz').resolve()),
                    input_sha256=file_hash(paths.directory/'inputs.npz'),manifest=data.manifest.to_dict())
                response=_run_worker(request,directory,budget.seconds-(time.perf_counter_ns()-started)/1e9)
                record['resource_summary']=response['resource_summary']
                if time.perf_counter_ns()-started>budget.seconds*1e9 or not response['quality_selection_eligible']:
                    record['status']='unavailable_at_budget'
                else:
                    prediction_path=directory/'prediction.npy'
                    if file_hash(prediction_path)!=response['prediction_sha256']:raise ValueError('Prediction hash mismatch')
                    prediction=np.load(prediction_path,allow_pickle=False)
                    if prediction.shape!=data.V.y.shape or not np.isfinite(prediction).all():raise ValueError('Invalid V prediction')
                    mse=float(np.mean((prediction-data.V.y)**2))
                    if not math.isfinite(mse):raise ValueError('Nonfinite V objective')
                    if time.perf_counter_ns()-started>budget.seconds*1e9:
                        record['status']='unavailable_at_budget'
                    else:record.update(status='completed',validation_mse=mse)
            except Exception as exc:
                if isinstance(exc,WorkerError):
                    record['resource_summary']=exc.result.get('resource_summary',{})
                    record['error']=exc.result['error']
                else:record['error']=dict(type=type(exc).__name__,message=str(exc))
                record['status']='unavailable_at_budget' if isinstance(exc,TimeoutError) else 'failed'
            record['charged_ns']=time.perf_counter_ns()-trial_start
            sampler.tell(record['validation_mse'])
            sampler.save(directory/'sampler.json')
            _atomic_json(directory/'trial.json',record);trials.append(record)
            append_event(paths,{'type':'search_trial','trial':record})
            if record['status']=='completed' and (winner is None or record['validation_mse']<trials[winner]['validation_mse']):winner=number
            if not support:break
        state_path=paths.directory/'sampler.final.json';sampler.save(state_path)
        elapsed=time.perf_counter_ns()-started
        result=SearchResult(winner,tuple(trials),elapsed,str(state_path),'completed',identity)
        update_run_observations(paths,{'costs':{'search':dict(nanoseconds=elapsed,status='measured',reason=None,boundary=COST_BOUNDARY)}})
        finish_run(paths,asdict(result),'completed')
        return result
    except BaseException as exc:
        # Preserve the original error even if a filesystem failure also prevents
        # terminal publication; known metadata lives in earlier immutable files.
        try:
            finish_run(paths,dict(error=dict(type=type(exc).__name__,message=str(exc)),identity=identity,
                trials=trials,charged_ns=time.perf_counter_ns()-started),'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed')
        except Exception as publication_error:
            exc.add_note(f'Terminal publication also failed: {publication_error!r}')
        raise


def produce_search_backend_receipt(path):
    """Run the existing numerical CPU producer in an actual fresh 1/1 process."""
    path=Path(path).resolve()
    if path.exists():raise FileExistsError(path)
    env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONPATH=str(ROOT/'src'))
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    run=subprocess.run([sys.executable,'-m','treeneuron.evaluation.search_worker','--backend-receipt',str(path)],
        cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
    if run.returncode:raise RuntimeError(f'Fresh search backend producer failed: {run.stderr}')
    from treeneuron.backend import validate_backend_report
    report=json.loads(path.read_text())
    validate_backend_report(report['backend_probes']['probes']['cpu'])
    return report


def expansion_diagnostics(trials,winner,support,curves=()):
    """Exploratory boundary/ceiling flags, never a quality-plateau certificate."""
    canonical_json({'trials':trials,'support':support,'curves':curves})
    boundaries=[]
    if winner is not None:
        row=trials[winner]
        if row['status']!='completed':raise ValueError('Winner must be completed')
        for key,spec in support.items():
            if spec['kind']=='float':
                for edge in ('low','high'):
                    if row['config'][key]==spec[edge]:boundaries.append(key+':'+edge)
    improving=False
    if len(curves)>=2:
        for previous,current in zip(curves,curves[1:]):
            if current['seconds']!=2*previous['seconds']:raise ValueError('Exploratory budgets must double')
        left,right=curves[-2:]
        improving=(left['best_validation_mse'] is not None and right['best_validation_mse'] is not None
                   and right['best_validation_mse']<left['best_validation_mse'])
    return dict(boundary_winners=boundaries,improving_ceiling=improving,freeze_ready=False,
                status='evidence_pending_WP05',reason='Expand winning boundaries/improving ceilings, retain official anchors; final quality/default plateaus and cross-teacher frozen design still required')
