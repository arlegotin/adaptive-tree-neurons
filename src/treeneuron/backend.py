"""Isolated eager float32 kernel and bounded whole-method calibration evidence.

The public probe launches a fresh interpreter with MPS fallback disabled before
Torch import. No process-wide RNG, dtype, thread or environment setting escapes.
Scoring is always a separately timed CPU64 copy; eigensolvers never run on MPS.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REQUIRED_OPERATIONS = ('gemm', 'tanh', 'sigmoid', 'relu', 'index_select', 'index_copy',
                       'layer_output', 'prediction', 'loss', 'input_gradient', 'parameter_gradients',
                       'adamw_step', 'adamw_resumed_step', 'serialization', 'resumed_update')
# Error budget fixed before native measurements. 64/256 binary32 eps account
# for short reductions/gradient chains, absolute protection at reference zeros.
TOLERANCE = {'atol': 64 * 2**-23, 'rtol': 256 * 2**-23}
DEFAULT_BATCHES = (16, 32, 64, 128)
RESERVE = 9 * 1024**3
ROOT = Path(__file__).resolve().parents[2]


def implementation_hash() -> str:
    digest = hashlib.sha256()
    for name in ('backend.py', 'models/units.py', 'growth/score.py', 'growth/allocate.py', 'growth/types.py'):
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def lock_hash() -> str:
    return hashlib.sha256((ROOT / 'uv.lock').read_bytes()).hexdigest()


def runtime_identity() -> dict[str, str]:
    return dict(python=platform.python_version(), torch=importlib.metadata.version('torch'),
                platform=platform.platform(), architecture=platform.machine())


def runtime_hash() -> str:
    return hashlib.sha256(json.dumps(runtime_identity(), sort_keys=True).encode()).hexdigest()


def _validate_physical_batches(batch_sizes: Sequence[int], *,
                               logical_batch_size: int = 128) -> list[int]:
    """One physical-batch contract for live inputs and imported timing evidence."""
    if (type(logical_batch_size) is not int or logical_batch_size != 128
            or not isinstance(batch_sizes, Sequence)
            or isinstance(batch_sizes, (str, bytes, bytearray))):
        raise ValueError('Require physical batches and Python-int logical batch128')
    batches = list(batch_sizes)
    if (not batches or any(type(b) is not int or b <= 0 or b > logical_batch_size
                          or logical_batch_size % b for b in batches)
            or any(b != 2*a for a,b in zip(batches,batches[1:]))):
        raise ValueError('Require doubled positive Python-int physical batch divisors of logical128')
    return batches


def _validate_batch_recipe(recipe: dict[str, int], *, batch_sizes: Sequence[int],
                           logical_batch_size: int) -> None:
    phases = {'base_epoch', 'grown_epoch', 'score'}
    if not isinstance(recipe, dict) or set(recipe) != phases:
        raise ValueError('Require exactly the measured batch recipe phases')
    for batch in recipe.values():
        _validate_physical_batches([batch], logical_batch_size=logical_batch_size)
        if batch not in batch_sizes:
            raise ValueError('Selected physical batch was not measured')


def probe_backend(device: str, *, seed: int, batch_sizes: Sequence[int]) -> dict[str, Any]:
    """Run measured kernels/calibration in isolation; unavailable MPS is no pass.

    Physical batches are a doubling sequence of divisors of logical batch128.
    This bounded probe is not a training recipe or a memory-limit search.
    """
    failed = dict(schema_version=1, device=device, status='failed', reason=None)
    if device not in ('cpu', 'mps'):
        return dict(failed, reason=f'Unsupported device: {device}')
    if type(seed) is not int or not 0 <= seed < 2**63 - 2:
        return dict(failed, reason='Require an integer seed in range')
    try:
        batches = _validate_physical_batches(batch_sizes)
    except ValueError as error:
        return dict(failed, reason=str(error))
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0')
    try:
        child = subprocess.run([sys.executable, '-m', 'treeneuron.backend', '--device', device,
                                '--seed', str(seed), '--batches', ','.join(map(str, batches))],
                               env=env, capture_output=True, text=True, timeout=240, check=False)
        if child.returncode:
            from treeneuron.doctor import _redact
            return dict(failed, reason=_redact(child.stderr.strip() or child.stdout.strip()
                                              or f'Probe exited {child.returncode}'))
        report = json.loads(child.stdout)
        json.dumps(report, allow_nan=False)
        if report.get('status') == 'passed':
            validate_backend_report(report)
        elif report.get('status') not in ('unavailable', 'failed') or not report.get('reason'):
            raise ValueError('Missing probe status/reason')
        return report
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as error:
        return dict(failed, reason=f'{type(error).__name__}: {error}')


def validate_backend_report(report: dict[str, Any]) -> None:
    """Reject availability/status-only fixtures and incomplete numeric evidence."""

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    def _require(condition):
        if not condition:
            raise ValueError('Backend numerical/schema contract failed')
    try:
        _require(report['schema_version'] == 1 and report['status'] == 'passed')
        _require(report['check'] == 'actual_backend_kernels_and_batches')
        _require(report['implementation_sha256'] == implementation_hash())
        _require(report['lock_sha256'] == lock_hash())
        _require(report['environment_sha256'] == runtime_hash())
        _require(all(report['environment'][k] == v for k,v in runtime_identity().items()))
        _require(report['environment']['fallback_env'] == '0')
        _require(report['device'] in ('cpu', 'mps'))
        _require(report['fallback_disabled_before_import'] is True)
        _require(report['environment']['dtype'] == 'torch.float32')
        _require(report['environment']['threads'] == 1)
        _require(report['environment']['python'] == platform.python_version())
        _require(report['reference'] == {'dtype': 'torch.float64', 'device': 'cpu'})
        _require(report['tolerance'] == TOLERANCE)
        _require(len(report['seeds']) == len(set(report['seeds'])) == 3
                 and all(type(v) is int for v in report['seeds'])
                 and report['magnitudes'] == [0.1, 1.0, 10.0])
        _require(set(report['operations']) == set(REQUIRED_OPERATIONS))
        for name, op in report['operations'].items():
            _require(op['status'] == 'passed' and len(op['trials']) == 9)
            _require(finite(op['max_abs']) and finite(op['max_relative']))
            _require(op['max_abs'] == max(t['max_abs'] for t in op['trials']))
            _require(op['max_relative'] == max(t['max_relative'] for t in op['trials']))
            _require({(t['seed'], t['magnitude']) for t in op['trials']} == {(s, m) for s in report['seeds'] for m in report['magnitudes']})
            for trial in op['trials']:
                _require(all((finite(trial[k]) and trial[k] >= 0 for k in ('max_abs', 'max_relative', 'max_normalized_error'))))
                _require(trial['max_normalized_error'] <= 1)
            if name in ('serialization', 'resumed_update'):
                _require(op['max_abs'] == 0)
        calibration = report['calibration']
        batches = _validate_physical_batches(report['batch_sizes'],
                                             logical_batch_size=calibration['logical_batch_size'])
        _require(calibration['n_rows'] == 2306)
        _validate_batch_recipe(calibration['selected_batches'], batch_sizes=batches,
                               logical_batch_size=calibration['logical_batch_size'])
        _require(calibration['score_boundary']['device'] == 'cpu')
        _require(calibration['score_boundary']['dtype'] == 'torch.float64')
        _require(finite(calibration['accumulation_max_abs']) and calibration['accumulation_max_abs'] < 1e-05)
        _require(calibration['updates_per_epoch'] == 19 and calibration['rows_per_epoch'] == 2306)
        for phase in ('base_epoch', 'grown_epoch', 'score'):
            rows = calibration[phase]
            row_batches = _validate_physical_batches([m['batch_size'] for m in rows],
                logical_batch_size=calibration['logical_batch_size'])
            _require(row_batches == batches)
            for row in rows:
                _require(len(row['samples_ns']) >= 5 and all((finite(t) and t > 0 for t in row['samples_ns'])))
                _require(row['median_ns'] == statistics.median(row['samples_ns']))
                _require(row['rows_per_second'] == 2306000000000.0 / row['median_ns'])
            maximum = max((m['rows_per_second'] for m in rows))
            _require(calibration['selected_batches'][phase] == min((m['batch_size'] for m in rows if m['rows_per_second'] >= 0.95 * maximum)))
        _require(finite(calibration['whole_method_ns']) and calibration['whole_method_ns'] > 0)
        whole = calibration['whole_method']
        _require(whole['batch_size'] is None)
        _validate_batch_recipe(whole['physical_batches'], batch_sizes=batches,
                               logical_batch_size=whole['logical_batch_size'])
        _require(whole['physical_batches'] == calibration['selected_batches'])
        _require(len(whole['samples_ns']) >= 5 and all((finite(t) and t > 0 for t in whole['samples_ns'])))
        _require(calibration['whole_method_ns'] == statistics.median(whole['samples_ns']))
        _require(report['resources_before']['initial_reservation_bytes'] == RESERVE)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        raise ValueError('Invalid actual backend probe schema or numerical evidence') from error

def select_backend(probes: dict[str, dict[str, Any]], *,
                   comparison_blocks: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Choose one validated training backend from balanced complete-method costs."""
    valid, excluded = {}, {}
    for device, report in probes.items():
        try:
            validate_backend_report(report)
            if report['device'] != device:
                raise ValueError('Backend identity mismatch')
            valid[device] = report
        except ValueError as error:
            excluded[device] = report.get('reason') or str(error)
    if 'cpu' not in valid:
        raise ValueError('CPU backend did not pass mandatory actual probes')
    costs = {d: r['calibration']['whole_method_ns'] for d,r in valid.items()}
    order = []
    block_costs = {}
    if comparison_blocks:
        order = [block['device'] for block in comparison_blocks]
        if order != ['cpu', 'mps', 'mps', 'cpu']:
            raise ValueError('Require balanced CPU/MPS/MPS/CPU comparison blocks')
        for block in comparison_blocks:
            report, device = block['report'], block['device']
            validate_backend_report(report)
            if report['device'] != device:
                raise ValueError('Comparison block identity mismatch')
            block_costs.setdefault(device, []).append(report['calibration']['whole_method_ns'])
        costs = {d: statistics.median(block_costs[d]) for d in valid}
    device = min(valid, key=lambda d: (costs[d], d != 'cpu'))
    chosen = valid[device]
    if comparison_blocks:
        candidates = [b['report'] for b in comparison_blocks if b['device'] == device]
        chosen = min(candidates, key=lambda r: abs(r['calibration']['whole_method_ns']-costs[device]))
    return dict(selected_backend=device, scoring_backend='cpu_float64',
                selection_basis='minimum median directly timed complete path; balanced ABBA when both backends pass',
                whole_method_ns=costs, comparison_order=order, block_medians_ns=block_costs,
                selected_batches=chosen['calibration']['selected_batches'], logical_batch_size=128,
                tolerance=chosen['tolerance'], excluded=excluded,
                scope='bounded d16/h64 F2306 calibration; remeasure for other workloads')


def collect_backend_report(*, seed: int = 1602030,
                           batch_sizes: Sequence[int] = DEFAULT_BATCHES) -> dict[str, Any]:
    probes = {d: probe_backend(d, seed=seed, batch_sizes=batch_sizes) for d in ('cpu', 'mps')}
    blocks = []
    if all(r['status'] == 'passed' for r in probes.values()):
        blocks = [dict(device=d, report=probes[d]) for d in ('cpu', 'mps')]
        for device in ('mps', 'cpu'):
            report = probe_backend(device, seed=seed, batch_sizes=batch_sizes)
            blocks.append(dict(device=device, report=report))
            probes[device] = report
    try:
        decision = select_backend(probes, comparison_blocks=blocks)
        return dict(schema_version=1, status='passed', probes=probes, decision=decision,
                    comparison_blocks=blocks)
    except ValueError as error:
        return dict(schema_version=1, status='failed', probes=probes, decision=None,
                    comparison_blocks=blocks, reason=str(error))


def _resources(torch) -> dict[str, Any]:
    from treeneuron.doctor import _probe
    observations = {k: _probe(cmd) for k,cmd in {
        'pressure': ['memory_pressure', '-Q'],
        'pressure_level': ['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
        'physical': ['sysctl', '-n', 'hw.memsize'],
        'swap': ['sysctl', '-n', 'vm.swapusage'], 'vm_stat': ['vm_stat'],
        'power': ['pmset', '-g', 'batt'],
        'rss': ['ps', '-o', 'rss=', '-p', str(os.getpid())]}.items()}
    pressure = observations['pressure']['value'] or ''
    physical = observations['physical']['value'] or ''
    match = re.search(r'free percentage:\s*(\d+)%', pressure)
    headroom = int(physical) * int(match[1]) // 100 if match and physical.isdigit() else None
    rss = observations['rss']['value']
    rss_value = int(rss) * 1024 if rss and rss.isdigit() else None
    swap_text = observations['swap']['value'] or ''
    swap_match = re.search(r'used\s*=\s*([\d.]+)([MG])', swap_text)
    swap_value = int(float(swap_match[1]) * 1024**(2 if swap_match[2] == 'M' else 3)) if swap_match else None
    counters = {}
    for name in ('recommended_max_memory', 'current_allocated_memory', 'driver_allocated_memory'):
        try:
            if not torch.backends.mps.is_available():
                raise RuntimeError('MPS unavailable or not observable')
            counters[name] = dict(value=int(getattr(torch.mps, name)()), reason=None)
        except Exception as error:
            counters[name] = dict(value=None, reason=f'{type(error).__name__}: {error}')
    return dict(utc=datetime.now(timezone.utc).isoformat(), observations=observations,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                (1 if sys.platform == 'darwin' else 1024),
                rss_bytes=dict(value=rss_value,reason=None if rss_value is not None else observations['rss']['reason']),
                swap_used_bytes=dict(value=swap_value,reason=None if swap_value is not None else observations['swap']['reason']),
                mps_memory_bytes=counters, initial_reservation_bytes=RESERVE,
                pressure_headroom_bytes=dict(value=headroom,
                    reason=None if headroom is not None else 'Native pressure/physical counters unavailable',
                    interpretation='coarse memory_pressure free percentage times physical; not free VM pages'),
                unified_memory_counters_overlap=True)


def _guard(resources):
    # 512MiB exceeds twice the measured ~250MiB first-use MPS RSS growth.
    # No RAM-budget-derived allocation or automatic doubling beyond128 occurs.
    level = resources['observations']['pressure_level']['value']
    headroom = resources['pressure_headroom_bytes']['value']
    if level is not None and level != '1':
        raise RuntimeError(f'Resource guard: memory pressure level {level}')
    if headroom is not None and headroom < RESERVE + 512 * 1024**2:
        raise RuntimeError('Resource guard: current pressure headroom below 9GiB reserve +512MiB')


def _run(device, seed, batches, *, evidence):
    # Called only by the fresh child after checking the import-time environment.
    batches = _validate_physical_batches(batches)
    import copy
    import torch
    from treeneuron.models.units import SelectiveAffineLayer
    from treeneuron.growth.score import score_units

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    report = evidence
    report.update(dict(schema_version=1, check='actual_backend_kernels_and_batches', device=device,
                  status='failed', reason=None, implementation_sha256=implementation_hash(),
                  lock_sha256=lock_hash(), environment_sha256=runtime_hash(),
                  fallback_disabled_before_import=True, utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                      platform=platform.platform(), architecture=platform.machine(),
                      dtype='torch.float32', threads=torch.get_num_threads(),
                      interop_threads=torch.get_num_interop_threads(),
                      mps_available=torch.backends.mps.is_available(),
                      fallback_env=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')),
                  reference=dict(dtype='torch.float64', device='cpu'),
                  tolerance=TOLERANCE,
                  tolerance_basis='64 eps absolute +256 eps relative, binary32 eps; short d<=16,h<=64 reductions. Fixed before native results; all errors recorded including CPU32 baseline, no MPS allowance.',
                  seeds=[seed,seed+1,seed+2], magnitudes=[.1,1.,10.], batch_sizes=batches,
                  exclusions=['float64 MPS forbidden; signed eigendecomposition CPU64 only',
                              'No implicit fallback; unsupported indispensable training op excludes whole backend'],
                  operations={name:dict(status='not_run', trials=[]) for name in REQUIRED_OPERATIONS}))
    if device == 'mps' and not torch.backends.mps.is_available():
        report.update(status='unavailable', reason='MPS built='+str(torch.backends.mps.is_built())+
                      ', available=False in this process; availability is not kernel evidence')
        return report

    def sync():
        if device == 'mps':
            torch.mps.synchronize()

    def measure(actual, reference):
        a, r = actual.detach().cpu().double(), reference.detach().cpu().double()
        delta = (a-r).abs()
        return dict(max_abs=float(delta.max()), max_relative=float((delta/r.abs().clamp_min(1e-12)).max()),
                    max_normalized_error=float((delta/(TOLERANCE['atol']+TOLERANCE['rtol']*r.abs())).max()))

    def record(name, a, r, trial_seed, magnitude):
        error = measure(a, r)
        trial = dict(seed=trial_seed, magnitude=magnitude, **error)
        report['operations'][name]['trials'].append(trial)
        if report['operations'][name]['status'] == 'not_run':
            report['operations'][name]['status'] = 'passed'
        if error['max_normalized_error'] > 1 or not all(math.isfinite(v) for v in error.values()):
            report['operations'][name]['status'] = 'failed'

    def flat(parameters):
        return torch.cat([p.detach().reshape(-1) for p in parameters])

    def model(grown=False):
        net = torch.nn.Sequential(SelectiveAffineLayer(16,64), torch.nn.ReLU(), torch.nn.Linear(64,1))
        if grown:
            net[0].grow(torch.arange(16), torch.randn(16,17)*.05)
            with torch.no_grad():
                net[0].gate.weight.normal_(0,.04)
                net[0].gate.bias.fill_(.1)
        return net

    before = _resources(torch)
    report['resources_before'] = before
    _guard(before)
    for trial_seed in report['seeds']:
        for magnitude in report['magnitudes']:
            generator = torch.Generator(device='cpu').manual_seed(trial_seed)
            x = torch.randn(23,16,generator=generator)*magnitude
            w = torch.randn(16,16,generator=generator)*.1
            ids = torch.tensor([1,4,8],dtype=torch.int64)
            for name in ('gemm','tanh','sigmoid','relu','index_select','index_copy'):
                report['active_stage'] = name
                outcomes=[]
                for backend,dtype in (('cpu',torch.float64),(device,torch.float32)):
                    v = x.to(device=backend,dtype=dtype).detach().requires_grad_()
                    if name == 'gemm':
                        z = v @ w.to(device=backend,dtype=dtype)
                    elif name == 'index_select':
                        z = v.index_select(1,ids.to(backend))
                    elif name == 'index_copy':
                        z = v.index_copy(1,ids.to(backend),v.index_select(1,ids.to(backend))*.3)
                    else:
                        z = getattr(torch,name)(v)
                    gradient = torch.autograd.grad(z.square().mean(),v)[0]
                    outcomes.append(torch.cat((z.detach().reshape(-1),gradient.reshape(-1))))
                record(name,outcomes[1],outcomes[0],trial_seed,magnitude)
            report['active_stage'] = 'actual_layer_adamw_resume'
            torch.manual_seed(trial_seed)
            original = model(grown=True)
            target = torch.randn(23,1,generator=generator)
            outcomes=[]
            for backend,dtype in (('cpu',torch.float64),(device,torch.float32)):
                net = copy.deepcopy(original).to(device=backend,dtype=dtype)
                v = x.to(device=backend,dtype=dtype).detach().requires_grad_()
                y = target.to(device=backend,dtype=dtype)
                opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01,foreach=False)
                pred=net(v)
                loss=(pred-y).square().mean()
                loss.backward()
                values=dict(layer_output=net[0](v).detach(),prediction=pred.detach().clone(),loss=loss.detach().reshape(1),
                            input_gradient=v.grad.detach().clone(),
                            parameter_gradients=flat([p.grad for p in net.parameters()]))
                opt.step()
                values['adamw_step']=flat(net.parameters())
                buffer=io.BytesIO()
                torch.save(dict(config=net[0].export_config(),state=net.state_dict(),optimizer=opt.state_dict()),buffer)
                buffer.seek(0)
                saved=torch.load(buffer,map_location=backend,weights_only=True)
                restored=torch.nn.Sequential(SelectiveAffineLayer.from_config(saved['config']),
                                             torch.nn.ReLU(),torch.nn.Linear(64,1)).to(device=backend,dtype=dtype)
                restored.load_state_dict(saved['state'])
                restored_opt=torch.optim.AdamW(restored.parameters(),lr=.001,foreach=False)
                restored_opt.load_state_dict(saved['optimizer'])
                if dtype == torch.float32:
                    record('serialization',flat(restored.parameters()),flat(net.parameters()),trial_seed,magnitude)
                for candidate,optimizer in ((net,opt),(restored,restored_opt)):
                    optimizer.zero_grad(set_to_none=True)
                    (candidate(v.detach())-y).square().mean().backward()
                    optimizer.step()
                if dtype == torch.float32:
                    record('resumed_update',flat(restored.parameters()),flat(net.parameters()),trial_seed,magnitude)
                values['adamw_resumed_step']=flat(net.parameters())
                outcomes.append(values)
            for name in outcomes[0]:
                record(name,outcomes[1][name],outcomes[0][name],trial_seed,magnitude)
    for name,op in report['operations'].items():
        op['max_abs']=max(t['max_abs'] for t in op['trials'])
        op['max_relative']=max(t['max_relative'] for t in op['trials'])
    if any(op['status'] != 'passed' for op in report['operations'].values()):
        report['reason']='Numerical operation failed fixed CPU64 tolerance'
        return report

    report['active_stage'] = 'calibration'
    torch.manual_seed(seed)
    base=model()
    grown=model(grown=True)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    cpu_x=torch.randn(2306,16,generator=generator)
    cpu_y=torch.randn(2306,1,generator=generator)
    x,y=cpu_x.to(device),cpu_y.to(device)

    def epoch(original,batch):
        net=copy.deepcopy(original).to(device)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01,foreach=False)
        for start in range(0,len(x),128):
            stop=min(start+128,len(x))
            optimizer.zero_grad(set_to_none=True)
            for offset in range(start,stop,batch):
                end=min(offset+batch,stop)
                ((net(x[offset:end])-y[offset:end]).square().sum()/(stop-start)).backward()
            optimizer.step()
        return net

    # Independent accumulated-vs-unsplit updates, including 2-row final batch.
    accumulation=[]
    for original in (base,grown):
        reference=flat(epoch(original,128).parameters()).cpu()
        for batch in batches:
            actual=flat(epoch(original,batch).parameters()).cpu()
            accumulation.append(float((actual-reference).abs().max()))
    eval_source=copy.deepcopy(base).to(device).eval()
    cpu64_x,cpu64_y=cpu_x.double(),cpu_y.double()
    boundary_samples=[]
    def scoring(batch, source=None):
        sync()
        start=time.perf_counter_ns()
        evaluation=copy.deepcopy(eval_source if source is None else source).cpu().double().eval()
        sync()
        boundary=time.perf_counter_ns()-start
        table=score_units(evaluation,cpu64_x,cpu64_y,epsilon=.05,tau=1.,adam_epsilon=1e-8,
                          batch_size=batch,unit_block_size=16)
        boundary_samples.append(boundary)
        return table

    def timed(fn):
        sync()
        start=time.perf_counter_ns()
        fn()
        sync()
        return time.perf_counter_ns()-start

    def benchmark(fn,batch):
        snapshots=_resources(torch)
        report['latest_resource_observation'] = snapshots
        _guard(snapshots)
        warm=[]
        for _ in range(9):
            warm.append(timed(fn))
            if len(warm)>=3 and (max(warm[-3:])-min(warm[-3:]))/statistics.median(warm[-3:]) <= .2:
                break
        samples=[]
        stable=False
        for _ in range(11):
            samples.append(timed(fn))
            if len(samples)>=5:
                recent=samples[-5:]
                stable=(max(recent)-min(recent))/statistics.median(recent) <= .2
                if stable:
                    break
        median=statistics.median(samples)
        return dict(batch_size=batch,samples_ns=samples,median_ns=median,
                    rows_per_second=2306e9/median,discarded_warmup_ns=warm,
                    warmup_stable=(max(warm[-3:])-min(warm[-3:]))/statistics.median(warm[-3:])<=.2,
                    timing_stable=stable,resources_before=snapshots)

    calibration=dict(n_rows=2306,d=16,h=64,logical_batch_size=128,rows_per_epoch=2306,
        updates_per_epoch=19,accumulation_max_abs=max(accumulation),
        weighting='sum squared losses / actual logical rows, then one AdamW step; tail logical2',
        timing='perf_counter_ns with MPS synchronize before/after whole region; includes reconstruction and AdamW creation',
        precision_rule='3..9 warmups until last3 range/median<=20%; 5..11 samples until last5 range/median<=20%; capped instability retained',
        order='base epoch, grown epoch, score; sizes ascending; isolated single process; no concurrent trials',
        memory_guard='bounded physical<=128;512MiB allowance above9GiB current pressure headroom; normal pressure if observable; missing counters explicitly uncertified',
        score_boundary=dict(device='cpu',dtype='torch.float64',unit_block_size=16,
            includes='synchronized trained-backend model copy to CPU, cast64, actual score_units; inputs already CPU64',
            transfer_samples_ns=boundary_samples), selected_batches={})
    report['calibration'] = calibration
    for phase,fn in (('base_epoch',lambda b:epoch(base,b)),('grown_epoch',lambda b:epoch(grown,b)),
                     ('score',scoring)):
        calibration[phase] = []
        for b in batches:
            calibration[phase].append(benchmark(lambda b=b:fn(b),b))
        maximum=max(row['rows_per_second'] for row in calibration[phase])
        calibration['selected_batches'][phase]=min(row['batch_size'] for row in calibration[phase]
                                                   if row['rows_per_second']>=.95*maximum)
    calibration['stage_medians_sum_ns']=sum(next(row['median_ns'] for row in calibration[phase]
        if row['batch_size']==calibration['selected_batches'][phase]) for phase in ('base_epoch','grown_epoch','score'))
    whole_recipe = dict(calibration['selected_batches'])
    _validate_batch_recipe(whole_recipe, batch_sizes=batches, logical_batch_size=128)
    def whole_method():
        from treeneuron.growth.allocate import allocate
        trained = epoch(base, whole_recipe['base_epoch'])
        table = scoring(whole_recipe['score'], trained.eval())
        allocation = allocate(table, 16, policy='adam', seed=seed)
        trained[0].grow(allocation.ids, allocation.contrasts.to(device=device,dtype=torch.float32))
        epoch(trained.train(), whole_recipe['grown_epoch'])
    calibration['whole_method'] = benchmark(whole_method, None)
    calibration['whole_method'].update(physical_batches=whole_recipe, logical_batch_size=128)
    calibration['whole_method_ns'] = calibration['whole_method']['median_ns']
    calibration['whole_method_boundary'] = 'base epoch -> CPU64 copy/score -> matched growth -> grown epoch; fresh optimizer per phase'
    report['calibration']=calibration
    report['resources_after']=_resources(torch)
    _guard(report['resources_after'])
    previous, current = before['swap_used_bytes']['value'], report['resources_after']['swap_used_bytes']['value']
    report['resource_deltas'] = dict(swap_used_bytes=current-previous if current is not None and previous is not None else None,
                                   reason=None if current is not None and previous is not None else 'Swap counter unavailable')
    if calibration['accumulation_max_abs']>=1e-5:
        report['reason']='Physical-batch accumulation failed logical-update parity'
    else:
        report.update(status='passed',reason=None)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',required=True,choices=['cpu','mps'])
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--batches',required=True)
    args=parser.parse_args()
    if 'torch' in sys.modules or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise RuntimeError('Fresh process requires PYTORCH_ENABLE_MPS_FALLBACK=0 before torch import')
    report = {}
    try:
        report=_run(args.device,args.seed,[int(b) for b in args.batches.split(',')],evidence=report)
    except Exception as error:
        from treeneuron.doctor import _redact
        report.update(schema_version=1,device=args.device,status='failed',
                      reason=_redact(f'{type(error).__name__}: {error}'))
        active = report.get('active_stage')
        if active in report.get('operations', {}):
            report['operations'][active]['status'] = 'failed'
            report['operations'][active]['reason'] = report['reason']
    report.pop('active_stage', None)
    print(json.dumps(report,allow_nan=False,sort_keys=True))


if __name__ == '__main__':
    main()
