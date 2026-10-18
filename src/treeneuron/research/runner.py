"""Native finite reference execution with retained raw data and status/cost ledger.

Fitting receives F/G-only views. V/E predictions and metrics are produced by
the coordinator after native training; protected data are never accepted here.
"""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from treeneuron import train
from treeneuron.baselines.causal import CausalPredictor
from treeneuron.checkpoint import _descriptor, load_checkpoint
from treeneuron.config import canonical_hash, resolve_experiment
from treeneuron.data.synthetic import make_from_config, sample_teacher
from treeneuron.evaluation.metrics import regression_metrics
from treeneuron.pilot.calibrate import fit_repetition
from .ledger import AttemptLedger, StudyLease, write_new_json
from .roles import fitting_view


def _threads():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _save_observations(path, data, evaluation):
    arrays = {}
    for role, rows in (('F', data.F), ('G', data.G), ('V', data.V), ('E', evaluation)):
        for name in ('X', 'y', 'row_id', 'group_id'):
            arrays[f'{role}_{name}'] = getattr(rows, name)
        if rows.noise_free_target is not None:
            arrays[f'{role}_evaluator_noisefree'] = rows.noise_free_target
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def _case_config(base, attempt, root):
    raw = json.loads(json.dumps(base))
    raw['seeds'].update({k: attempt[k+'_seed'] for k in
                        ('teacher', 'data', 'reference', 'optimizer', 'allocation', 'contrast')})
    raw['data']['population'] = attempt['population'] if attempt['population'] != 'positive' else 'primary'
    if attempt['population'] == 'positive':
        raw['model'].update(d=2, h=8)
        raw['growth']['m_requested'] = 2
        raw['data'].update(n_fit=512, n_growth=256, n_validation=256, noise_sigma=attempt['sigma'])
    raw['execution']['artifacts_root'] = str(root)
    return resolve_experiment(raw)


def _guard(directory):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
    if rss > 16*1024**3:
        raise MemoryError('Research worker exceeded16GiB peak RSS envelope')
    if shutil.disk_usage(directory).free < 256*1024**2:
        raise OSError('Insufficient free disk for checkpoint transaction; retain queue and resume')


def _metrics_and_predictions(root, report, data, evaluation):
    arrays = dict(V_y=data.V.y, V_row_id=data.V.row_id, E_y=evaluation.y, E_row_id=evaluation.row_id)
    variance = float(np.var(data.F.y, ddof=0))
    metrics = {}
    for name, reference in report['models'].items():
        pointer = Path(reference['path'])
        if (not pointer.resolve().is_relative_to(root.resolve())
                or hashlib.sha256(pointer.read_bytes()).hexdigest() != reference['sha256']):
            raise ValueError('Retained model pointer escaped root or changed bytes')
        _, descriptor, _ = _descriptor(pointer)
        model = (load_checkpoint(pointer, expected_hashes=descriptor['state']['identity_hashes'], device='cpu')
            if reference.get('loader') == 'training_state' else
            CausalPredictor.load(pointer, expected_hashes=descriptor['state']['identity_hashes']))
        metrics[name] = {}
        for role, rows in (('V', data.V), ('E', evaluation)):
            prediction = model.predict(rows.X)
            arrays[f'{name}_{role}'] = prediction
            metrics[name][role] = regression_metrics(rows.y, prediction, fit_target_variance=variance)
    with (root/'predictions.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    write_new_json(root/'coordinator-metrics.json', dict(metrics=metrics,
        V_labels_access='coordinator_only_after_fitting', T_accessed=False))
    return metrics


def physical_cost_summary(previous, invocation_ns):
    return dict(invocation_elapsed_ns=invocation_ns,
        cumulative_physical_elapsed_ns=invocation_ns+sum(r['invocation_elapsed_ns'] for r in previous))


def run(plan, output, *, resume=False):
    value = plan.to_dict()
    if value['source_sha256'] != train.source_hash():
        raise ValueError('Research plan source differs from current implementation')
    from .resources import validate_resource_admission
    admission = validate_resource_admission(value, output)
    from .identity import execution_identity
    if execution_identity(value['config']) != value['execution_identity']:
        raise ValueError('Research runtime/lock/backend receipt identity changed; create a new plan')
    _threads()
    output = Path(output)
    if not resume:
        ledger = AttemptLedger.create(output, value)
    with StudyLease(output):
        if resume:
            ledger = AttemptLedger.resume(output, value)
        return _run_locked(value, output, ledger, admission, resume)


def _run_locked(value, output, ledger, admission, resume):
    previous_results = [json.loads(p.read_text()) for p in sorted(output.glob('result-*.json'))]
    manifest = dict(schema='research/v2-run', source_sha256=train.source_hash(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        dirty_patch_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src', 'configs', 'uv.lock'])).hexdigest(),
        argv=sys.argv, cwd=str(Path.cwd()), threads=dict(intra=1, interop=1),
        plan_hash=value['plan_hash'], protected_labels_accessed=False,
        cost_boundary='whole coordinator; pair clocks nest; shared physical parent counted once',
        physical_elapsed_ns=0, status='running', completed_final_fits=0,
        resource_admission=admission)
    if not resume:
        write_new_json(output/'manifest.json', manifest)
    started = time.perf_counter_ns()
    groups = {}
    for attempt in value['attempts']:
        groups.setdefault((attempt['cell'], attempt['teacher_index'], attempt['repetition']), []).append(attempt)
    base = value['config']['base_experiment']
    try:
        for (cell, teacher, repeat), attempts in groups.items():
            pending = [a for a in attempts if a['attempt_id'] in ledger.pending_ids]
            if not pending:
                continue
            last = {e['attempt_id']: e for e in ledger.events}
            previous = [last[a['attempt_id']] for a in pending if a['attempt_id'] in last]
            recovery = 1 if previous else 0
            if previous:
                if not resume or any(e.get('recovery_attempt') == 1 and e['status'] == 'running'
                    for e in ledger.events if e['attempt_id'] in {a['attempt_id'] for a in pending}):
                    raise ValueError('One declared infrastructure rerun exhausted or resume not requested')
                for a in pending:
                    event = last.get(a['attempt_id'])
                    if event and event['status'] == 'running':
                        # OS lease proves that the previous worker is no longer
                        # active. Elapsed wall time includes idle time and is
                        # retained as a bound, never an exact efficiency input.
                        bound = max(0, time.time_ns()-event['started_utc_ns'])
                        ledger.record(a['attempt_id'], 'interrupted_recoverable',
                            elapsed_ns=event['elapsed_ns']+bound,
                            unrecorded_physical_bound_ns=bound,
                            cost_precision='wall_time_upper_bound_after_process_loss',
                            reason='Prior worker ended without a terminal event')
                    elif event and event['status'] != 'interrupted_recoverable':
                        raise ValueError('Only infrastructure interruption may rerun a native pair')
            pair_root = output/cell/f'teacher-{teacher}'/f'repeat-{repeat}-attempt-{recovery}'
            if pair_root.exists():
                raise ValueError('Native transaction directory already exists; preserve earlier artifacts')
            pair_root.parent.mkdir(parents=True, exist_ok=True)
            config = _case_config(base, attempts[0], pair_root)
            previous_costs = {a['attempt_id']: max((e['elapsed_ns'] for e in ledger.events
                if e['attempt_id'] == a['attempt_id']), default=0) for a in pending}
            for a in pending:
                ledger.record(a['attempt_id'], 'running', elapsed_ns=previous_costs[a['attempt_id']],
                    recovery_attempt=recovery, started_utc_ns=time.time_ns())
            tick = time.perf_counter_ns()
            try:
                _guard(output)
                control = None
                if attempts[0]['population'] == 'positive':
                    from .controls import make_positive_control
                    control = make_positive_control(instance_seed=attempts[0]['teacher_seed'],
                        data_seed=attempts[0]['data_seed'], gamma=attempts[0]['gamma'], sigma=attempts[0]['sigma'])
                    population, data, evaluation = control, control.development, control.evaluation
                else:
                    population, data = make_from_config(config)
                    evaluation = sample_teacher(population, value['config']['evaluation_rows'],
                                                attempts[0]['evaluation_seed'], config.data.noise_sigma)
                data_root = pair_root.parent/f'data-repeat-{repeat}-attempt-{recovery}'
                data_root.mkdir(exist_ok=False)
                _save_observations(data_root/'observations.npz', data, evaluation)
                write_new_json(data_root/'teacher.json', population.to_dict() if control is None else dict(
                    version='research/v2-planted', base=control.base_rows.tolist(), head=control.head.tolist(),
                    gamma=control.gamma, sigma=control.sigma, instance_seed=control.instance_seed,
                    data_seed=control.data_seed, generating_ids=list(control.generating_ids)))
                write_new_json(data_root/'split.json', data.manifest.to_dict())
                if value['config']['kind'] == 'm0_reference':
                    from .snapshots import SnapshotSink
                    sink = SnapshotSink(pair_root/'snapshots', (250, 1000))
                    report = fit_repetition(config, fitting_view(data), pair_root, value['plan_hash'],
                        resource_guard=lambda: _guard(output), parent_snapshot_updates=(250, 1000),
                        on_parent_snapshot=sink)
                    sink.finalize()
                else:
                    from .control_runner import fit_control_pair
                    report = fit_control_pair(config, fitting_view(data), pair_root, [a['role'] for a in attempts],
                        control=control.parameters if control is not None else None, guard=lambda: _guard(output))
                metrics = _metrics_and_predictions(pair_root, report, data, evaluation)
                for a in pending:
                    role = a['role']
                    record = report['methods'][role]
                    if record['status'] == 'completed_ineligible':
                        ledger.record(a['attempt_id'], 'completed_ineligible',
                            elapsed_ns=previous_costs[a['attempt_id']]+report['costs']['parent_ns']+report['costs']['score_ns'],
                            reason=record['reason'])
                        continue
                    # All component clocks are nested; expose attributed costs
                    # without summing them into the actual physical pair timer.
                    own = record['enclosing_ns']
                    shared = report['costs']['parent_ns'] if role in ('keep','top','random','greedy','widen_upper','oracle_current','oracle_true') else 0
                    selection = report['costs']['probe_ns'] if role == 'greedy' else 0
                    scoring = report['costs']['score_ns'] if role in ('top','random','greedy','oracle_current','oracle_true') else 0
                    ledger.record(a['attempt_id'], 'completed_eligible',
                        elapsed_ns=previous_costs[a['attempt_id']]+own+shared+selection+scoring, metrics=metrics[role],
                        retained_model=report['models'][role],
                        physical_pair_elapsed_ns=time.perf_counter_ns()-tick,
                        cost_precision='conservative_shared_transaction_bound' if recovery else 'measured_attributed_dependencies',
                        operational_reexecuted_roles=[a['role'] for a in attempts] if recovery else [],
                        standalone_attribution='parent/scorer/probes prerequisite charged to each applicable role; shared scorer contrast work for random disclosed')
            except BaseException as error:
                status = ('interrupted_recoverable' if isinstance(error, (KeyboardInterrupt, OSError)) and recovery == 0
                    else 'irrecoverable_missing' if isinstance(error, KeyboardInterrupt)
                    else 'resource_exhausted' if isinstance(error, (MemoryError, OSError))
                    else 'numerical_failure' if isinstance(error, FloatingPointError)
                    else 'input_or_dependency_failure')
                # Do not invent completed predictions for partial pair outputs.
                for a in pending:
                    if a['attempt_id'] in ledger.pending_ids:
                        ledger.record(a['attempt_id'], status,
                            elapsed_ns=previous_costs[a['attempt_id']]+time.perf_counter_ns()-tick,
                            reason=f'{type(error).__name__}: {error}', failure_scope='shared_pair_transaction',
                            cost_precision='conservative_shared_transaction_bound',
                            complete_primary_claim=False)
                raise
        statuses = {e['attempt_id']: e['status'] for e in ledger.events}
        manifest['status'] = ('incomplete' if ledger.pending_ids else 'completed'
            if all(s == 'completed_eligible' for s in statuses.values()) else 'completed_with_failures')
    except BaseException:
        manifest['status'] = 'interrupted_or_failed'
        raise
    finally:
        manifest.update(physical_elapsed_ns=time.perf_counter_ns()-started,
            completed_final_fits=sum(e['status']=='completed_eligible' for e in ledger.events),
            charged_standalone_ns=ledger.charged_ns, pending_attempt_ids=list(ledger.pending_ids))
        manifest.update(physical_cost_summary(previous_results, manifest['physical_elapsed_ns']))
        manifest['physical_elapsed_ns_scope'] = 'this invocation; see cumulative_physical_elapsed_ns'
        manifest['formal_cost_claim_eligible'] = not any('bound' in e.get('cost_precision', '') for e in ledger.events)
        missing_cost = {}
        for event in ledger.events:
            key = event['attempt_id'].rsplit('/', 1)[0]
            missing_cost[key] = max(missing_cost.get(key, 0), event.get('unrecorded_physical_bound_ns', 0))
        manifest['unrecorded_physical_upper_bound_ns'] = sum(missing_cost.values())
        manifest['physical_cost_status'] = ('incomplete_after_process_loss' if any(missing_cost.values())
                                            else 'measured_invocations')
        manifest['reexecuted_completed_roles_cost_scope'] = 'included in physical transaction; no exact standalone efficiency claim after recovery'
        # Attempt history remains append-only; immutable attempt summaries are new
        # files, and never overwrite an earlier failed/resumed run's receipt.
        summary = output/f'result-{len(tuple(output.glob("result-*.json")))}.json'
        write_new_json(summary, manifest)
    return manifest
