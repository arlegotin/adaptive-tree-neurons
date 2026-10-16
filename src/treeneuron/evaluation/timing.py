"""Synchronized prediction blocks and raw-data reconstruction.

Warm host measurements include complete batching, preprocessing and output copy.
Resident mode is explicit capability-gated. RSS and MPS counters overlap and must
never be summed. Precision resamples complete blocks; between-block independence is an explicit
design assumption, and final study policy remains separate.
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


def _complete_blocks(rows):
    """Two declared repeat slots make a block; duplicate records add no evidence."""
    grouped={}
    for row in rows:
        block=row['block'];repeat=row.get('repeat')
        if type(block) is not int or block<0:raise ValueError('Invalid block identity')
        if type(repeat) is not int or repeat not in (0,1):
            raise ValueError('Timing records require explicit repeat slots 0/1')
        slots=grouped.setdefault(block,{})
        if repeat in slots and slots[repeat]!=row:raise ValueError('Conflicting duplicate timing slot')
        slots[repeat]=row
    complete={}
    for block,slots in grouped.items():
        if set(slots)!={0,1} or any(r['status']!='ok' for r in slots.values()):continue
        values=[slots[i]['elapsed_ns'] for i in (0,1)]
        if any(type(n) is not int or n<=0 for n in values):raise ValueError('Timings must be positive integer ns')
        complete[block]=values
    return complete


def _precision(values,quantile,seed):
    # Entire repeat pairs are drawn together. Identical seeded draws across
    # methods preserve common-block pairing, including shared machine drift.
    values=np.asarray(values,dtype=np.float64)
    point=float(np.quantile(values,quantile));rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(values),size=(1000,len(values)))
    samples=np.quantile(values[draws].reshape(1000,-1),quantile,axis=1)
    bounds=np.quantile(samples,[.025,.975]);half=float((bounds[1]-bounds[0])/2)
    return dict(estimate_ns=point,interval_ns=bounds.tolist(),relative_half_width=half/point if point else None,
                method='1000 deterministic complete-block bootstrap replicates; two repeat slots per block; between-block independence remains a design assumption')


def _batch_summary(blocks,rows,policy):
    values=[blocks[b] for b in sorted(blocks)]
    med=_precision(values,.5,policy.seed);tail=_precision(values,policy.tail_quantile,policy.seed+1)
    # At least two blocks even for median-only diagnostics; the tail needs
    # ceil(1/(1-q)) block units. This is a minimum evidence gate, not a coverage guarantee.
    required=max(2,math.ceil(policy.min_repetitions/2),math.ceil(1/(1-policy.tail_quantile)))
    precise=len(blocks)>=required and all(x['relative_half_width'] is not None and x['relative_half_width']<=policy.relative_precision for x in (med,tail))
    return dict(repetitions=2*len(blocks),independent_blocks=len(blocks),required_blocks=required,
        median_ns=med['estimate_ns'],tail_ns=tail['estimate_ns'],median_precision=med,tail_precision=tail,
        precision_reached=precise,throughput_rows_per_second=rows*1e9/med['estimate_ns'])


def rebuild_summary(raw_blocks,policy:TimingPolicy):
    batches={}
    for batch in policy.batch_sizes:
        rows=[r for r in raw_blocks if r['phase']=='measurement' and r['batch_size']==batch]
        if not rows:continue
        if len({r['rows'] for r in rows})!=1:raise ValueError('Timing row counts differ')
        blocks=_complete_blocks(rows)
        if blocks:batches[str(batch)]=_batch_summary(blocks,rows[0]['rows'],policy)
    peak=max((v['throughput_rows_per_second'] for v in batches.values()),default=None)
    selected=min((int(b) for b,r in batches.items() if r['throughput_rows_per_second']>=.95*peak),default=None) if peak else None
    return dict(batches=batches,selected_batch=selected,selection_rule='smallest successful measured batch >=95% peak median complete-call throughput',
                precision_reached=len(batches)==len(policy.batch_sizes) and all(r['precision_reached'] for r in batches.values()))


def _comparison_policies(policies):
    if len(policies)<2 or any(not isinstance(k,str) or not k for k in policies):raise ValueError('Comparison requires at least two explicit method IDs')
    policies={k:policies[k] if isinstance(policies[k],TimingPolicy) else TimingPolicy(**policies[k]) for k in sorted(policies)}
    controls=('cpu_threads','warmup_window','max_warmup','min_repetitions','max_repetitions','relative_precision','warmup_relative_spread','tail_quantile','block_order','seed')
    first=next(iter(policies.values()))
    if any(len(p.batch_sizes)!=1 for p in policies.values()):raise ValueError('Calibrate batches separately; comparison requires one batch per method')
    if any(any(getattr(p,k)!=getattr(first,k) for k in controls) for p in policies.values()):raise ValueError('Comparison block controls must match')
    return policies


def rebuild_comparison(raw_blocks,policies):
    policies=_comparison_policies(policies);by_method={}
    measured=[r for r in raw_blocks if r['phase']=='measurement']
    if any(r['method_id'] not in policies for r in measured):raise ValueError('Unknown comparative method')
    if len({r['rows'] for r in measured})>1:raise ValueError('Comparison workloads differ')
    for method,policy in policies.items():
        rows=[r for r in measured if r['method_id']==method]
        if any(r['batch_size']!=policy.batch_sizes[0] for r in rows):raise ValueError('Comparison batch mismatch')
        by_method[method]=_complete_blocks(rows)
    common=sorted(set.intersection(*(set(v) for v in by_method.values())))
    summaries={};paired={};reference=next(iter(policies));policy=policies[reference]
    for method,blocks in by_method.items():
        if common:summaries[method]=_batch_summary({b:blocks[b] for b in common},measured[0]['rows'],policies[method])
    if common:
        rng=np.random.default_rng(policy.seed);draws=rng.integers(0,len(common),size=(1000,len(common)))
        ref=np.array([by_method[reference][b] for b in common],dtype=float)
        ref_samples=np.median(ref[draws].reshape(1000,-1),axis=1)
        for method,blocks in by_method.items():
            if method==reference:continue
            values=np.array([blocks[b] for b in common],dtype=float)
            point=float(np.median(values)/np.median(ref))
            bounds=np.quantile(np.median(values[draws].reshape(1000,-1),axis=1)/ref_samples,[.025,.975])
            relative=float((bounds[1]-bounds[0])/(2*point))
            paired[method+'/'+reference]=dict(median_ratio=point,interval=bounds.tolist(),relative_half_width=relative,
                independent_blocks=len(common),precision_reached=len(common)>=summaries[method]['required_blocks'] and relative<=policy.relative_precision,
                method='paired complete-block bootstrap; identical block draws for numerator and denominator')
    return dict(complete_block_ids=common,methods=summaries,paired_ratios=paired,
        precision_reached=bool(common) and all(r['precision_reached'] for r in summaries.values()) and all(r['precision_reached'] for r in paired.values()))


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
                    repeats={b:0 for b in policy.batch_sizes}
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
                            row=dict(phase=phase,batch_size=batch,elapsed_ns=elapsed,rows=len(X),block=block,repeat=repeats[batch],status='ok')
                            windows[batch].append(elapsed)
                            if phase=='warmup' and len(windows[batch])>=policy.warmup_window:
                                window=np.asarray(windows[batch][-policy.warmup_window:])
                                stable[batch]=float((window.max()-window.min())/np.median(window))<=policy.warmup_relative_spread
                        except (MemoryError,torch.OutOfMemoryError) as exc:
                            row=dict(phase=phase,batch_size=batch,elapsed_ns=time.perf_counter_ns()-tick,rows=len(X),block=block,repeat=repeats[batch],status='excluded',reason=repr(exc))
                            failed.add(batch)
                        except BaseException as exc:
                            raw.append(dict(phase=phase,batch_size=batch,elapsed_ns=time.perf_counter_ns()-tick,rows=len(X),block=block,repeat=repeats[batch],status='failed',reason=repr(exc)))
                            exc.timing_record=dict(raw_blocks=raw,excluded_work=excluded,policy=asdict(policy),error=dict(type=type(exc).__name__,message=str(exc)))
                            raise
                        raw.append(row);counts[batch]+=1;repeats[batch]+=1
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
            return dict(schema_version=2,policy=asdict(policy),rows=len(X),raw_blocks=raw,summary=measured_summary,
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


def measure_comparison(cases:dict[str,tuple[Predictor,TimingPolicy]],X:Any)->dict[str,Any]:
    """Compare already fitted method/mode cases in shared complete blocks.

    Each case has one separately calibrated batch. IDs name the exact method and
    mode; boundaries can differ but are retained, never silently pooled. Every
    method appears twice in each mirrored block, including during warmup.
    """
    if len(X)<1:raise ValueError('Empty prediction workload')
    policies=_comparison_policies({k:v[1] for k,v in cases.items()})
    resources={k:_capability(v[0],X,policies[k]) for k,v in cases.items()}
    import torch
    from threadpoolctl import threadpool_limits
    from treeneuron.backend import _resources,_guard,runtime_hash
    from treeneuron.train import source_hash,lock_hash
    if any(p.device=='mps' for p in policies.values()) and not torch.backends.mps.is_available():raise ValueError('MPS unavailable; no fallback')
    policy=next(iter(policies.values()));raw=[];excluded=[];checks={};prepared={};identities={}
    stable={k:False for k in cases};windows={k:[] for k in cases};rng=np.random.default_rng(policy.seed)
    old_threads=torch.get_num_threads();torch.set_num_threads(policy.cpu_threads)
    synchronizers={k:torch.mps.synchronize if p.device=='mps' else lambda:None for k,p in policies.items()}
    try:
        with threadpool_limits(limits=policy.cpu_threads),torch.inference_mode():
            for method,(predictor,p) in cases.items():
                tick=time.perf_counter_ns();checks[method]=_resources(torch);_guard(checks[method])
                excluded.append(dict(method_id=method,reason='resource guard before comparative case',elapsed_ns=time.perf_counter_ns()-tick))
                prepared[method]=X
                if p.mode=='resident_kernel':
                    prepared[method],cost=measured_call(lambda:predictor.prepare_timing_input(X),sync=synchronizers[method])
                    excluded.append(dict(method_id=method,reason='resident input preparation and host-to-device transfer',elapsed_ns=cost))
                state=predictor.predictor.state if isinstance(predictor,ResidentCausalPredictor) else getattr(predictor,'state',None)
                identities[method]=(state.identity_hashes if state is not None else getattr(predictor,'identity',None))
                if identities[method] is None and hasattr(predictor,'_state'):identities[method]={'snapshot_sha256':canonical_hash(predictor._state())}
                if identities[method] is None:identities[method]={'value':None,'reason':'predictor exposes no immutable fitted identity'}
            def prediction(method):
                predictor,p=cases[method];batch=p.batch_sizes[0]
                if p.context_mode=='uncached':predictor.clear_prediction_cache()
                pieces=[]
                for start in range(0,len(X),batch):
                    chunk=prepared[method][start:start+batch]
                    value=predictor.predict_resident(chunk) if p.mode=='resident_kernel' else np.asarray(predictor.predict(chunk,batch_size=batch))
                    pieces.append(value)
                return torch.cat(pieces) if p.mode=='resident_kernel' else np.concatenate(pieces)
            block=0
            for phase,cap in (('warmup',policy.max_warmup),('measurement',policy.max_repetitions)):
                for _ in range(cap//2):
                    order=list(policies)
                    if policy.block_order=='balanced_random':rng.shuffle(order)
                    elif block%2:order.reverse()
                    repeats={k:0 for k in cases}
                    for method in order+order[::-1]:
                        tick=time.perf_counter_ns();row=dict(phase=phase,method_id=method,block=block,repeat=repeats[method],
                            batch_size=policies[method].batch_sizes[0],rows=len(X))
                        try:
                            output,elapsed=measured_call(lambda:prediction(method),sync=synchronizers[method])
                            if tuple(output.shape)!=(len(X),):raise ValueError('Predictor output must be [N]')
                            finite=bool(torch.isfinite(output).all()) if isinstance(output,torch.Tensor) else bool(np.isfinite(output).all())
                            if not finite:raise ValueError('Nonfinite prediction')
                        except BaseException as exc:
                            raw.append(dict(row,elapsed_ns=time.perf_counter_ns()-tick,status='failed',reason=repr(exc)));raise
                        raw.append(dict(row,elapsed_ns=elapsed,status='ok'));repeats[method]+=1;windows[method].append(elapsed)
                    block+=1
                    if phase=='warmup':
                        for method,values in windows.items():
                            if len(values)>=policy.warmup_window:
                                window=np.asarray(values[-policy.warmup_window:])
                                stable[method]=float((window.max()-window.min())/np.median(window))<=policy.warmup_relative_spread
                        if all(stable.values()):break
                    elif rebuild_comparison(raw,policies)['precision_reached']:break
            summary=rebuild_comparison(raw,policies)
            return dict(schema_version=2,kind='comparison',policies={k:asdict(v) for k,v in policies.items()},
                raw_blocks=raw,summary=summary,rows=len(X),warmup_stable=stable,excluded_work=excluded,
                identity=dict(source=source_hash(),core_lock=lock_hash(),runtime=runtime_hash(),input_sha256=array_hash(np.asarray(X)),predictors=identities),
                predictor_resources=resources,resource_checks=checks,memory_guard_observed=all(v['pressure_headroom_bytes']['value'] is not None for v in checks.values()),
                rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                mps_allocated_bytes=torch.mps.current_allocated_memory() if any(p.device=='mps' for p in policies.values()) else {'value':None,'reason':'CPU cases'},
                mps_driver_bytes=torch.mps.driver_allocated_memory() if any(p.device=='mps' for p in policies.values()) else {'value':None,'reason':'CPU cases'},
                memory_accounting='RSS and MPS counters overlap; do not add; process peak is not incremental model memory',
                boundary='per-case policy: host_to_host includes transforms, complete row batching and original-unit host output; resident_kernel excludes transforms, host transfers and target inversion',
                precision_status='reached' if summary['precision_reached'] and all(stable.values()) else 'unresolved_at_repetition_or_warmup_cap')
    except BaseException as exc:
        exc.timing_record=dict(raw_blocks=raw,excluded_work=excluded,policies={k:asdict(v) for k,v in policies.items()},
            error=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:torch.set_num_threads(old_threads)


def write_timing(path,result):
    _atomic_json(Path(path),dict(payload=result,sha256=canonical_hash(result)))


def read_timing(path):
    envelope=json.loads(Path(path).read_text())
    if set(envelope)!={'payload','sha256'} or canonical_hash(envelope['payload'])!=envelope['sha256']:raise ValueError('Timing hash mismatch')
    result=envelope['payload']
    if result.get('schema_version')!=2:raise ValueError('Timing schema requires its original source reader')
    summary=(rebuild_comparison(result['raw_blocks'],result['policies']) if result.get('kind')=='comparison'
        else rebuild_summary(result['raw_blocks'],TimingPolicy(**result['policy'])))
    if summary!=result['summary']:raise ValueError('Timing summary differs from raw blocks')
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
