"""Synchronized prediction blocks and raw-data reconstruction.

Warm host measurements include complete batching, preprocessing and output copy.
Resident mode is explicit capability-gated. RSS and MPS counters overlap and must
never be summed. Precision is descriptive bootstrap uncertainty, not study power.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
from contextlib import nullcontext
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any
import numpy as np
from treeneuron.artifacts import _atomic_json
from treeneuron.data.types import array_hash
from treeneuron.config import canonical_hash,canonical_json
from treeneuron.baselines.api import Predictor


@dataclass(frozen=True)
class TimingPolicy:
    batch_sizes: tuple[int,...]=(1,)
    device: str='cpu'
    mode: str='host_to_host'
    context_mode: str='none'
    cpu_threads: int=1
    warmup_window: int=5
    max_warmup: int=30
    min_repetitions: int=20
    max_repetitions: int=200
    relative_precision: float=.05
    warmup_relative_spread: float=.05
    tail_quantile: float=.95
    block_order: str='abba'
    seed: int=1602047

    def __post_init__(self):
        object.__setattr__(self,'batch_sizes',tuple(self.batch_sizes))
        if not self.batch_sizes or any(type(b) is not int or b<=0 for b in self.batch_sizes) or len(set(self.batch_sizes))!=len(self.batch_sizes):
            raise ValueError('Unique positive integer batch sizes required')
        if self.device not in ('cpu','mps') or self.mode not in ('host_to_host','resident_kernel') or self.context_mode not in ('none','cached','uncached'):
            raise ValueError('Unknown device/mode/context boundary')
        for key in ('cpu_threads','warmup_window','max_warmup','min_repetitions','max_repetitions'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError(f'{key} must be positive int')
        if self.warmup_window<2 or self.max_warmup<self.warmup_window or self.min_repetitions<4 or self.max_repetitions<self.min_repetitions:
            raise ValueError('Invalid warmup/repetition windows')
        for key in ('relative_precision','warmup_relative_spread','tail_quantile'):
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<1:raise ValueError(f'Invalid {key}')
        if self.block_order not in ('abba','balanced_random'):raise ValueError('Unknown block order')
        if type(self.seed) is not int or not 0<=self.seed<2**32:raise ValueError('Invalid seed')


def measured_call(call,*,sync=lambda:None,timer=time.perf_counter_ns):
    sync();start=timer()
    try:value=call()
    finally:
        sync();stop=timer()
    return value,stop-start


def _precision(values,quantile,seed):
    values=np.asarray(values,dtype=np.float64);point=float(np.quantile(values,quantile))
    rng=np.random.default_rng(seed)
    resampled=values[rng.integers(0,len(values),size=(1000,len(values)))]
    bounds=np.quantile(np.quantile(resampled,quantile,axis=1),[.025,.975])
    half=float((bounds[1]-bounds[0])/2)
    return dict(estimate_ns=point,interval_ns=bounds.tolist(),relative_half_width=half/point if point else None,
                method='1000 deterministic iid bootstrap replicates; diagnostic, correlation not corrected')


def rebuild_summary(raw_blocks,policy:TimingPolicy):
    batches={}
    for batch in policy.batch_sizes:
        rows=[r for r in raw_blocks if r['phase']=='measurement' and r['batch_size']==batch and r['status']=='ok']
        if not rows:continue
        durations=[r['elapsed_ns'] for r in rows]
        if any(type(n) is not int or n<=0 for n in durations):raise ValueError('Timings must be positive integer ns')
        if len({r['rows'] for r in rows})!=1:raise ValueError('Timing row counts differ')
        med=_precision(durations,.5,policy.seed);tail=_precision(durations,policy.tail_quantile,policy.seed+1)
        enough=len(rows)>=max(policy.min_repetitions,math.ceil(1/(1-policy.tail_quantile)))
        precise=enough and all(x['relative_half_width'] is not None and x['relative_half_width']<=policy.relative_precision for x in (med,tail))
        batches[str(batch)]=dict(repetitions=len(rows),median_ns=med['estimate_ns'],tail_ns=tail['estimate_ns'],
            median_precision=med,tail_precision=tail,precision_reached=precise,
            throughput_rows_per_second=rows[0]['rows']*1e9/med['estimate_ns'])
    peak=max((v['throughput_rows_per_second'] for v in batches.values()),default=None)
    selected=min((int(b) for b,r in batches.items() if r['throughput_rows_per_second']>=.95*peak),default=None) if peak else None
    return dict(batches=batches,selected_batch=selected,selection_rule='smallest successful measured batch >=95% peak median complete-call throughput',
                precision_reached=len(batches)==len(policy.batch_sizes) and all(r['precision_reached'] for r in batches.values()))


def _order(policy,rng,block):
    batches=list(policy.batch_sizes)
    if policy.block_order=='balanced_random':rng.shuffle(batches)
    elif block%2:batches.reverse()
    # Two mirrored passes retain the balanced block unit (ABBA for two batches).
    return batches+batches[::-1]


def _capability(predictor,X,policy):
    summary=predictor.resource_summary();caps=summary.get('capabilities',{})
    backend=caps.get('backend')
    if backend is None and summary.get('method')=='B7' and summary.get('backend')=='CPU float32 eager':backend='cpu'
    if backend is None and isinstance(summary.get('backend'),dict):backend=summary['backend'].get('device')
    if backend not in ('cpu','numpy_cpu','mps'):raise ValueError('Predictor must expose actual cpu/numpy_cpu/mps backend')
    if (policy.device=='mps')!=(backend=='mps'):raise ValueError('Requested device differs from actual predictor; no fallback')
    if policy.context_mode!='none' and not caps.get('cache'):raise ValueError('Context/cache boundary unsupported by predictor')
    if policy.context_mode=='none' and (caps.get('cache') or caps.get('context') is True):raise ValueError('Context predictor requires explicit cached/uncached policy')
    if policy.mode=='resident_kernel' and not all(hasattr(predictor,n) for n in ('prepare_timing_input','predict_resident')):
        raise ValueError('Predictor has no verified resident_kernel timing capability')
    if policy.context_mode=='uncached' and not hasattr(predictor,'clear_prediction_cache'):
        raise ValueError('Uncached mode requires explicit cache-clear capability')
    return summary


def measure_predictor(predictor:Predictor,X:Any,policy:TimingPolicy)->dict[str,Any]:
    if len(X)<1:raise ValueError('Empty prediction workload')
    summary=_capability(predictor,X,policy)
    import torch
    from threadpoolctl import threadpool_limits
    if policy.device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS unavailable; no fallback')
    sync=torch.mps.synchronize if policy.device=='mps' else lambda:None
    old_threads=torch.get_num_threads();torch.set_num_threads(policy.cpu_threads)
    raw=[];excluded=[];memory_checks={};rng=np.random.default_rng(policy.seed);windows={b:[] for b in policy.batch_sizes}
    stable={b:False for b in policy.batch_sizes};failed=set();block=0
    try:
        with threadpool_limits(limits=policy.cpu_threads),torch.inference_mode():
            prepared=X
            if policy.mode=='resident_kernel':
                prepared,cost=measured_call(lambda:predictor.prepare_timing_input(X),sync=sync)
                excluded.append(dict(reason='resident input preparation and host-to-device transfer',elapsed_ns=cost))
            def prediction(batch):
                if policy.context_mode=='uncached':predictor.clear_prediction_cache()
                pieces=[]
                for start in range(0,len(X),batch):
                    chunk=prepared[start:start+batch]
                    value=(predictor.predict_resident(chunk) if policy.mode=='resident_kernel' else predictor.predict(chunk,batch_size=batch))
                    if policy.mode=='host_to_host':value=np.asarray(value)
                    pieces.append(value)
                return torch.cat(pieces) if policy.mode=='resident_kernel' else np.concatenate(pieces)
            for phase,limit in (('warmup',policy.max_warmup),('measurement',policy.max_repetitions)):
                counts={b:0 for b in policy.batch_sizes}
                while any(counts[b]<limit for b in policy.batch_sizes
                          if b not in failed and not (phase=='warmup' and stable[b])):
                    for batch in _order(policy,rng,block):
                        if batch in failed or counts[batch]>=limit or (phase=='warmup' and stable[batch]):continue
                        tick=time.perf_counter_ns()
                        try:
                            if batch not in memory_checks:
                                from treeneuron.backend import _resources,_guard
                                memory_checks[batch]=_resources(torch)
                                try:_guard(memory_checks[batch])
                                except RuntimeError as guard_error:raise MemoryError(str(guard_error)) from guard_error
                                excluded.append(dict(reason='resource guard before batch '+str(batch),elapsed_ns=time.perf_counter_ns()-tick))
                            output,elapsed=measured_call(lambda:prediction(batch),sync=sync)
                            if tuple(output.shape)!=(len(X),):raise ValueError('Predictor output must be [N]')
                            finite=bool(torch.isfinite(output).all()) if isinstance(output,torch.Tensor) else bool(np.isfinite(output).all())
                            if not finite:raise ValueError('Nonfinite prediction')
                            row=dict(phase=phase,batch_size=batch,elapsed_ns=elapsed,rows=len(X),block=block,status='ok')
                            windows[batch].append(elapsed)
                            if phase=='warmup' and len(windows[batch])>=policy.warmup_window:
                                window=np.asarray(windows[batch][-policy.warmup_window:])
                                stable[batch]=float((window.max()-window.min())/np.median(window))<=policy.warmup_relative_spread
                        except (MemoryError,torch.OutOfMemoryError) as exc:
                            row=dict(phase=phase,batch_size=batch,elapsed_ns=time.perf_counter_ns()-tick,rows=len(X),block=block,status='excluded',reason=repr(exc))
                            failed.add(batch)
                        except BaseException as exc:
                            raw.append(dict(phase=phase,batch_size=batch,elapsed_ns=time.perf_counter_ns()-tick,rows=len(X),block=block,status='failed',reason=repr(exc)))
                            exc.timing_record=dict(raw_blocks=raw,excluded_work=excluded,policy=asdict(policy),error=dict(type=type(exc).__name__,message=str(exc)))
                            raise
                        raw.append(row);counts[batch]+=1
                    block+=1
                    if phase=='warmup' and all(stable[b] or b in failed for b in policy.batch_sizes):break
                    if phase=='measurement' and all(counts[b]>=policy.min_repetitions or b in failed for b in policy.batch_sizes):
                        if rebuild_summary(raw,policy)['precision_reached']:break
            measured_summary=rebuild_summary(raw,policy)
            unavailable=lambda reason:dict(value=None,reason=reason)
            from treeneuron.train import source_hash,lock_hash
            from treeneuron.backend import runtime_hash
            model_identity=(predictor.predictor.state.identity_hashes if isinstance(predictor,ResidentCausalPredictor)
                else predictor.state.identity_hashes if getattr(predictor,'state',None) is not None
                else getattr(predictor,'identity',None))
            if model_identity is None and hasattr(predictor,'_state'):model_identity={'snapshot_sha256':canonical_hash(predictor._state())}
            return dict(schema_version=1,policy=asdict(policy),rows=len(X),raw_blocks=raw,summary=measured_summary,
                identity=dict(source=source_hash(),core_lock=lock_hash(),runtime=runtime_hash(),input_sha256=array_hash(np.asarray(X)),
                    predictor=model_identity or unavailable('predictor exposes no immutable fitted identity')),
                resource_checks={str(k):v for k,v in memory_checks.items()},
                memory_scope='native pressure guard before each tested batch: 9GiB reserve +512MiB headroom; unobservable counters are not certified',
                memory_guard_observed=all(v['pressure_headroom_bytes']['value'] is not None for v in memory_checks.values()),
                warmup_stable={str(k):v for k,v in stable.items()},excluded_work=excluded,
                predictor_resources=summary,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                mps_allocated_bytes=torch.mps.current_allocated_memory() if policy.device=='mps' else unavailable('CPU predictor'),
                mps_driver_bytes=torch.mps.driver_allocated_memory() if policy.device=='mps' else unavailable('CPU predictor'),
                memory_accounting='RSS and MPS counters overlap; do not add; process peak RSS is not incremental model memory',
                boundary='inference_mode; complete row batching; host input/preprocessing through original-unit host output' if policy.mode=='host_to_host' else 'inference_mode; resident standardized float32 input through standardized resident output; excludes transforms, original-unit inversion and host transfers',
                precision_status='reached' if measured_summary['precision_reached'] and all(stable.values()) else 'unresolved_at_repetition_or_warmup_cap')
    except BaseException as exc:
        if not hasattr(exc,'timing_record'):
            exc.timing_record=dict(raw_blocks=raw,excluded_work=excluded,policy=asdict(policy),
                error=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:torch.set_num_threads(old_threads)


def write_timing(path,result):
    _atomic_json(Path(path),dict(payload=result,sha256=canonical_hash(result)))


def read_timing(path):
    envelope=json.loads(Path(path).read_text())
    if set(envelope)!={'payload','sha256'} or canonical_hash(envelope['payload'])!=envelope['sha256']:raise ValueError('Timing hash mismatch')
    result=envelope['payload'];policy=TimingPolicy(**result['policy'])
    if rebuild_summary(result['raw_blocks'],policy)!=result['summary']:raise ValueError('Timing summary differs from raw blocks')
    return result


def measure_cold_predictor(predictor,X,policy:TimingPolicy,directory:Path,*,timeout_seconds=60.):
    """Save outside measurement, then time fresh import/load/first host prediction.

    The serialization and launch/publication enclosure are retained separately.
    This does not claim OS disk-page-cache eviction or cold hardware caches.
    """
    import os
    import subprocess
    from treeneuron.evaluation.search import ROOT,file_hash
    _capability(predictor,X,policy)
    if policy.mode!='host_to_host' or policy.context_mode!='none':raise ValueError('Cold worker currently supports stateless host_to_host only')
    name=type(predictor).__name__
    if name not in ('MeanPredictor','RidgePredictor','SplineRidgePredictor','NodePredictor','CausalPredictor'):
        raise ValueError('No registered safe loader for predictor')
    directory=Path(directory).resolve();directory.mkdir(exist_ok=False)
    start=time.perf_counter_ns();predictor.save(directory/'predictor.json')
    np.save(directory/'X.npy',np.asarray(X),allow_pickle=False)
    metadata=dict(schema_version=1,method=name,policy=asdict(policy),model_sha256=file_hash(directory/'predictor.json'),
        input_sha256=file_hash(directory/'X.npy'),expected_hashes=predictor.state.identity_hashes if name=='CausalPredictor' else None)
    _atomic_json(directory/'request.json',dict(payload=metadata,sha256=canonical_hash(metadata)))
    serialization=time.perf_counter_ns()-start
    env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONPATH=str(ROOT/'src'))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]=str(policy.cpu_threads)
    start=time.perf_counter_ns()
    with (directory/'stdout.txt').open('wb') as out,(directory/'stderr.txt').open('wb') as err:
        run=subprocess.run([sys.executable,'-m','treeneuron.evaluation.timing_worker',str(directory)],cwd=ROOT,env=env,stdout=out,stderr=err,timeout=timeout_seconds)
    elapsed=time.perf_counter_ns()-start
    if run.returncode:raise RuntimeError(f'Cold worker failed ({run.returncode}); see retained stderr')
    envelope=json.loads((directory/'response.json').read_text());result=envelope['payload']
    if canonical_hash(result)!=envelope['sha256']:raise ValueError('Cold response hash mismatch')
    if file_hash(directory/'prediction.npy')!=result['prediction_sha256']:raise ValueError('Cold prediction hash mismatch')
    result.update(process_wall_ns=elapsed,excluded_serialization_ns=serialization,
        cold_scope='fresh interpreter/modules/model/first prediction; OS page cache not evicted')
    _atomic_json(directory/'cold.json',result)
    return result


class ResidentCausalPredictor:
    """Explicit timing view of an already fitted causal model; never fits/refits.

    Kernel output is standardized float32 resident output. Original-unit host
    inversion is deliberately outside this kernel boundary and retained as such.
    """
    def __init__(self,predictor):
        from treeneuron.baselines.causal import CausalPredictor
        if not isinstance(predictor,CausalPredictor) or predictor.state is None:
            raise ValueError('Resident causal timing requires a fitted CausalPredictor')
        self.predictor=predictor

    def resource_summary(self):return self.predictor.resource_summary()
    def predict(self,X,*,batch_size=None):return self.predictor.predict(X,batch_size=batch_size)

    def prepare_timing_input(self,X):
        import torch
        state=self.predictor.state
        return torch.tensor(state.preprocessing.transform_X(X),dtype=torch.float32,device=next(state.model.parameters()).device)

    def predict_resident(self,X):
        return self.predictor.state.model(X)
