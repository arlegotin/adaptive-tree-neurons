"""Finite H1 CPU logical-batch/density evidence, separate from MPS calibration.

The producer runs in a fresh CPU1/inter-op1/BLAS1 interpreter. Each optimizer
step consumes its whole logical batch; requested batches above F saturate at F.
Timing includes reconstruction and optimizer creation, excludes checkpoint IO,
G selection and the full paired fitting graph. It never selects a study recipe.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .backend import TOLERANCE, _guard, _resources, lock_hash, runtime_hash

LOGICAL_BATCHES = (128,256,512,1024,2048,4096,8192,16384)
DENSITIES = (577,2306,9224)
DIAGNOSTIC_BATCH = 32768
PHASES = ('base_epoch','grown_epoch','ordinary_epoch','widened_epoch','static_epoch','score')
BLAS_ENV = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def supported_batch(batch):
    return type(batch) is int and (0 < batch <=128 or batch in LOGICAL_BATCHES)


def requires_envelope(config):
    return config.data.n_fit>2306 or config.training.batch_size>128


def validate_workload(config, topology=None):
    """Only the required first-layer density family and its derived controls."""
    c=config.model
    if (config.backend.device!='cpu' or c.d!=16 or c.depth!=1
            or config.training.optimizer!='adamw' or config.data.n_fit not in DENSITIES
            or config.training.batch_size not in LOGICAL_BATCHES
            or config.growth.m_requested!=16):
        raise ValueError('Workload outside finite H1 CPU envelope')
    if c.coordinates=='selective':
        valid=64<=c.h<=95
        if topology is not None:
            count=len(topology['layer']['grown_ids'])
            valid=valid and (count<=16 if c.h==64 else count==0)
    elif c.coordinates=='widened':
        valid=c.parent_width==64 and 64<=c.h<=95
    else:
        valid=c.coordinates=='centered' and c.h==64 and 0<=c.static_count<=16
    if not valid:
        raise ValueError('Topology outside finite H1 CPU envelope')


def _positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0


def validate_cpu_envelope(report):
    """Validate every measured cell and numerical/resource identity, fail closed."""
    from .train import source_hash
    source=source_hash()
    try:
        def require(ok):
            if not ok: raise ValueError('CPU envelope contract failed')
        require(report['schema_version']==1 and report['status']=='passed' and report['seed']==1602042)
        require(report['check']=='h1_cpu_logical_batch_density_envelope')
        require(report['source_sha256']==source and report['lock_sha256']==lock_hash()
            and report['runtime_sha256']==runtime_hash())
        require(report['logical_batches']==list(LOGICAL_BATCHES) and report['densities']==list(DENSITIES))
        require(report['diagnostic_batch']==DIAGNOSTIC_BATCH and report['tolerance']==TOLERANCE)
        env=report['environment']
        require(env==dict(device='cpu',dtype='float32',execution='eager',threads=1,interop_threads=1,
            fallback_env='0',blas={k:'1' for k in BLAS_ENV}))
        require(report['selection_claim']=='finite candidates only; no universal throughput optimum')
        # Actual per-topology float32/CPU64 operations and independent causal
        # formula/AdamW/replay evidence for every newly admitted logical batch.
        operations=report['operations']
        expected={(phase,b) for phase in PHASES[:-1] for b in LOGICAL_BATCHES}
        require(len(operations)==len(expected))
        require({(op['phase'],op['logical_batch']) for op in operations}==expected)
        for op in operations:
            error=op['float64_max_normalized_error']
            require(type(error) in (int,float) and math.isfinite(error) and 0<=error<=1)
            from .baselines.causal_evidence import validate_reference_check
            validate_reference_check(op['float64_reference'])
            require(error==max(op['float64_reference'][k] for k in ('forward_max_normalized_error','conditioned_gradient_max_normalized_error','loss_max_normalized_error')))
            e=op['causal']
            require(e['schema_version']==2 and e['device']=='cpu' and e['dtype']=='float32'
                and e['execution']=='eager' and e['seed']==1602041)
            shapes=_parameter_shapes(op['phase'])
            require(e['parameter_shapes']==shapes and e['total_parameters']==sum(math.prod(v) for v in shapes.values()))
            validate_reference_check(e['reference_check'])
            require(e['status']=='passed' and e['source_hash']==source
                and e['batch_size']==op['logical_batch'] and e['actual_batch_size']==min(op['logical_batch'],9224)
                and op['actual_batch_size']==min(op['logical_batch'],9224) and e['reload_next_update_max_abs']==0)
            require(e['topology']==report['topologies'][op['phase']])
            for key in ('forward_max_abs','gradient_max_abs','optimizer_max_abs'):
                require(type(e[key]) in (int,float) and math.isfinite(e[key]) and 0<=e[key]<=1e-4)
        require(report['topologies']==_topologies())
        expected={(n,b,p) for n in DENSITIES for b in (*LOGICAL_BATCHES,DIAGNOSTIC_BATCH) for p in PHASES}
        rows=report['measurements']
        require(len(rows)==len(expected))
        require({(r['n_fit'],r['logical_batch'],r['phase']) for r in rows}==expected)
        for row in rows:
            n,b=row['n_fit'],row['logical_batch']
            require(type(n) is int and type(b) is int and row['physical_batch']==b)
            require(row['actual_max_rows']==min(n,b) and row['tail_rows']==(n%b or min(n,b)))
            require(row['updates_per_epoch']==(math.ceil(n/b) if row['phase']!='score' else 0))
            samples=row['samples_ns']
            require(len(samples)==5 and all(_positive(v) for v in samples))
            require(row['median_ns']==statistics.median(samples) and row['rows_per_second']==n*1e9/row['median_ns'])
            require(len(row['discarded_warmup_ns'])==1 and _positive(row['discarded_warmup_ns'][0]))
            require(_positive(row['peak_rss_bytes']))
        require(len(report['score_parity'])==len(DENSITIES)*len(LOGICAL_BATCHES))
        require({(r['n_fit'],r['batch']) for r in report['score_parity']}=={(n,b) for n in DENSITIES for b in LOGICAL_BATCHES})
        for row in report['score_parity']:
            require(_positive(row['tolerance']) and row['tolerance']==1e-10
                and 0<=row['max_normalized_error']<=1)
        for key in ('resources_before','resources_after'):
            require(_positive(report[key]['peak_rss_bytes']))
            _guard(report[key])
    except (KeyError,TypeError,ValueError,RuntimeError) as error:
        raise ValueError('Invalid source-bound CPU envelope evidence') from error


def validate_envelope_reference(reference, config, topology=None):
    """Resolve the same immutable producer receipt on fit and checkpoint reload."""
    from .locators import locate
    validate_workload(config,topology)
    import torch
    if (torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1
            or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0'
            or any(os.environ.get(k)!='1' for k in BLAS_ENV)):
        raise ValueError('Extended CPU runtime requires CPU1/inter-op1/BLAS1/fallback0')
    try:
        if set(reference)!={'path','sha256'}: raise ValueError('CPU envelope reference schema')
        path=locate(reference['path'],producer=True)
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=reference['sha256']:
            raise ValueError('CPU envelope receipt hash mismatch')
        validate_cpu_envelope(json.loads(payload)['cpu_envelope'])
    except (OSError,KeyError,TypeError) as error:
        raise ValueError('Missing source-bound CPU envelope receipt') from error


def _topologies():
    # JSON specifications avoid allocating models or consuming RNG on reload.
    def selective(h,ids):
        return dict(schema_version=1,kind='selective_relu_scalar',layer=dict(schema_version=1,
            d=16,h=h,tau=1.,growth_applied=bool(ids),grown_ids=ids))
    return dict(base_epoch=selective(64,[]),grown_epoch=selective(64,list(range(16))),
        ordinary_epoch=selective(95,[]),
        widened_epoch=dict(schema_version=1,kind='widened_relu_scalar',layer=dict(d=16,h=95,parent_width=64,tau=1.)),
        static_epoch=dict(schema_version=1,kind='static_gated_relu_scalar',layer=dict(d=16,h=64,count=16,coordinates='centered',tau=1.)))


def _parameter_shapes(phase):
    shapes={}
    def linear(name,rows,columns,bias=True):
        shapes[name+'.weight']=[rows,columns]
        if bias: shapes[name+'.bias']=[rows]
    if phase=='widened_epoch':
        linear('layer.base',64,16);linear('layer.added',31,16)
        linear('head.base',1,64);linear('head.added',1,31,False)
    elif phase=='static_epoch':
        linear('layer.ordinary',48,16)
        for part in ('gate','center','contrast'):linear('layer.'+part,16,16)
        linear('head',1,64)
    else:
        h=95 if phase=='ordinary_epoch' else 64
        linear('layer.base',h,16);linear('head',1,h)
        if phase=='grown_epoch':
            linear('layer.gate',16,16);linear('layer.contrast',16,16)
    return shapes


def probe_cpu_envelope():
    """Obtain actual finite CPU measurements in isolation, retaining failed output."""
    env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',**{k:'1' for k in BLAS_ENV})
    try:
        child=subprocess.run([sys.executable,'-m','treeneuron.cpu_envelope'],env=env,
            capture_output=True,text=True,timeout=240,check=False)
        if child.returncode: raise RuntimeError(child.stderr or child.stdout)
        report=json.loads(child.stdout)
        if report.get('status')=='passed': validate_cpu_envelope(report)
        return report
    except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as error:
        return dict(status='failed',reason=f'{type(error).__name__}: {error}')


def _run(report):
    import torch
    from .train import SelectiveRegressor, source_hash
    from .models.static import StaticRegressor
    from .models.widened import WidenedRegressor
    from .baselines.causal_evidence import probe_causal_cpu, compare_reference
    from .growth.score import score_units
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(1602042)
    report.update(schema_version=1,status='running',seed=1602042,check='h1_cpu_logical_batch_density_envelope',
        source_sha256=source_hash(),lock_sha256=lock_hash(),runtime_sha256=runtime_hash(),
        logical_batches=list(LOGICAL_BATCHES),diagnostic_batch=DIAGNOSTIC_BATCH,densities=list(DENSITIES),
        environment=dict(device='cpu',dtype='float32',execution='eager',threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),fallback_env=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
            blas={k:os.environ.get(k) for k in BLAS_ENV}),tolerance=TOLERANCE,
        selection_claim='finite candidates only; no universal throughput optimum',
        timing_boundary='reconstruction + AdamW creation + full F epoch; score includes CPU64 copy; no checkpoint IO/G selection/full paired graph; ambient interference possible',
        resources_before=_resources(torch),topologies=_topologies(),operations=[],measurements=[],score_parity=[])
    _guard(report['resources_before'])
    classes={'selective_relu_scalar':SelectiveRegressor,'static_gated_relu_scalar':StaticRegressor,
        'widened_relu_scalar':WidenedRegressor}
    models={phase:classes[spec['kind']].from_config(spec) for phase,spec in report['topologies'].items()}
    # from_config reconstructs zero contrast tensors; activate gates/contrasts so
    # the grown operation evidence exercises actual nontrivial branches.
    with torch.no_grad():
        models['grown_epoch'].layer.contrast.weight.normal_(0,.05)
        models['grown_epoch'].layer.contrast.bias.fill_(.01)
        models['grown_epoch'].layer.gate.weight.normal_(0,.04)
        models['grown_epoch'].layer.gate.bias.fill_(.1)
    for phase,model in models.items():
        for batch in LOGICAL_BATCHES:
            actual_batch=min(batch,9224)
            x=torch.randn(actual_batch,16);y=torch.randn(actual_batch)
            comparison=compare_reference(model,x,y,reference_dtype=torch.float64)
            error=max(comparison[k] for k in ('forward_max_normalized_error','conditioned_gradient_max_normalized_error','loss_max_normalized_error'))
            report['operations'].append(dict(phase=phase,logical_batch=batch,actual_batch_size=actual_batch,
                float64_max_normalized_error=error,float64_reference=comparison,
                causal=probe_causal_cpu(model,batch_size=batch,n_rows=9224)))
    for n in DENSITIES:
        x=torch.randn(n,16);y=torch.randn(n)
        x64,y64=x.double(),y.double()
        def score(batch):
            return score_units(copy.deepcopy(models['base_epoch']).double().eval(),x64,y64,
                epsilon=.05,tau=1.,adam_epsilon=1e-8,batch_size=batch,unit_block_size=16)
        reference=score(128)
        for batch in LOGICAL_BATCHES:
            actual=score(batch)
            errors=[]
            for key in ('gram','adam_scores','sgd_scores','eigenvalues','gradients'):
                a,b=getattr(actual,key),getattr(reference,key)
                errors.append(float(((a-b).abs()/(1e-10+1e-10*b.abs())).max()))
            error=max(errors)
            if not math.isfinite(error) or error>1: raise ValueError('CPU64 score batch parity failed')
            report['score_parity'].append(dict(n_fit=n,batch=batch,max_normalized_error=error,tolerance=1e-10))
        def epoch(original,batch):
            net=copy.deepcopy(original)
            opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01,foreach=False)
            for offset in range(0,n,batch):
                opt.zero_grad(set_to_none=True)
                (net(x[offset:offset+batch])-y[offset:offset+batch]).square().mean().backward()
                opt.step()
        for phase in PHASES:
            report['latest_resource_observation']=_resources(torch)
            _guard(report['latest_resource_observation'])
            for batch in (*LOGICAL_BATCHES,DIAGNOSTIC_BATCH):
                samples=[]
                for _ in range(6):
                    start=time.perf_counter_ns()
                    score(batch) if phase=='score' else epoch(models[phase],batch)
                    samples.append(time.perf_counter_ns()-start)
                report['measurements'].append(dict(phase=phase,n_fit=n,logical_batch=batch,physical_batch=batch,
                    actual_max_rows=min(n,batch),tail_rows=n%batch or min(n,batch),
                    updates_per_epoch=0 if phase=='score' else math.ceil(n/batch),
                    discarded_warmup_ns=samples[:1],samples_ns=samples[1:],median_ns=statistics.median(samples[1:]),
                    rows_per_second=n*1e9/statistics.median(samples[1:]),peak_rss_bytes=_resources(torch)['peak_rss_bytes']))
    report['resources_after']=_resources(torch);_guard(report['resources_after'])
    report['status']='passed'
    validate_cpu_envelope(report)
    return report


def main():
    if 'torch' in sys.modules or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or any(os.environ.get(k)!='1' for k in BLAS_ENV):
        raise RuntimeError('Fresh CPU envelope requires fallback0 and BLAS1 before numerical imports')
    report={}
    try: _run(report)
    except Exception as error: report.update(status='failed',reason=f'{type(error).__name__}: {error}')
    print(json.dumps(report,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
