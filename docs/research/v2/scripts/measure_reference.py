"""Measure one predeclared M0 parent/probe/control graph; no efficacy inference.

Run with the qualified interpreter and numerical environment from reproduce.md.
This is an execution recipe composing tested native workers, not a new selector.
The first teacher/repetition is fixed by the full immutable M0 plan. Its outcomes
are development information and cannot become a fresh confirmation teacher.
"""
import time
STARTED = time.perf_counter_ns()
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    if os.environ.get(name) != '1':
        raise SystemExit(f'{name}=1 required before numerical imports')

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from treeneuron import train
from treeneuron.config import canonical_hash
from treeneuron.data.synthetic import make_from_config, sample_teacher
from treeneuron.pilot.calibrate import fit_repetition
from treeneuron.research.config import load_research_config, expand
from treeneuron.research.identity import execution_identity
from treeneuron.research.ledger import write_new_json
from treeneuron.research.roles import fitting_view
from treeneuron.research.runner import _case_config, _guard, _save_observations, _metrics_and_predictions
from treeneuron.research.snapshots import SnapshotSink


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = expand(load_research_config(args.config)).to_dict()
    if plan['config']['kind'] != 'm0_reference':
        raise ValueError('This finite qualification recipe measures M0 only')
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    attempt = plan['attempts'][0]
    config = _case_config(plan['config']['base_experiment'], attempt, root/'pair')
    manifest = dict(schema='research/v2-native-reference-measurement', status='running',
        purpose='resource qualification; one first declared teacher/repetition, not full M0',
        source_sha256=train.source_hash(), execution_identity=execution_identity(plan['config']),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        argv=sys.argv, cwd=str(Path.cwd()), parent_plan_hash=plan['plan_hash'],
        teacher_index=attempt['teacher_index'], repetition=attempt['repetition'],
        final_roles=8, nominal_probe_trials=config.model.h+1, protected_labels_accessed=False)
    write_new_json(root/'manifest.json', manifest)
    write_new_json(root/'parent-plan.json', plan)
    print('Starting fixed first M0 pair for native resource measurement', flush=True)
    try:
        _guard(root)
        population, data = make_from_config(config)
        evaluation = sample_teacher(population, plan['config']['evaluation_rows'],
                                    attempt['evaluation_seed'], config.data.noise_sigma)
        _save_observations(root/'observations.npz', data, evaluation)
        write_new_json(root/'teacher.json', population.to_dict())
        write_new_json(root/'split.json', data.manifest.to_dict())
        sink = SnapshotSink(root/'pair'/'snapshots', (250, 1000))
        report = fit_repetition(config, fitting_view(data), root/'pair', plan['plan_hash'],
            resource_guard=lambda: _guard(root), parent_snapshot_updates=(250, 1000), on_parent_snapshot=sink)
        sink.finalize()
        _metrics_and_predictions(root/'pair', report, data, evaluation)
        files = [p for p in root.rglob('*') if p.is_file()]
        payload = max((p.stat().st_size for p in files if p.suffix == '.pt'), default=0)
        metadata = max((p.stat().st_size for p in files if p.name.endswith('.meta.json')), default=0)
        history = max((p.stat().st_size for p in files if p.name.startswith('history-') and p.suffix == '.jsonl'), default=0)
        transaction = payload+metadata+history+4096
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
        retained = sum(p.stat().st_size for p in files)
        workload = dict(d=config.model.d, h=config.model.h, n_fit=config.data.n_fit,
            n_growth=config.data.n_growth, n_validation=config.data.n_validation,
            batch_size=config.training.batch_size)
        phase_times = dict(parent=report['costs']['parent_ns'], singleton_probe=report['costs']['probe_ns'],
            continuation=max(report['methods'][name]['enclosing_ns'] for name in
                             ('keep','top','random','greedy','widen_upper')))
        refs = []
        for phase, elapsed in phase_times.items():
            record = dict(schema='research/v2-resource-measurement',
                source_sha256=train.source_hash(), execution_identity_hash=canonical_hash(plan['execution_identity']),
                workload=workload, phase=phase, elapsed_ns=elapsed, peak_rss_bytes=rss,
                training_contract_hash=canonical_hash({'training': config.to_dict()['training']}),
                retained_bytes=retained, max_checkpoint_transaction_bytes=transaction,
                status='completed_eligible')
            path = root/f'measurement-{phase}.json'
            write_new_json(path, record)
            refs.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest.update(status='completed', measurements=refs, retained_bytes=retained,
            peak_rss_bytes=rss, max_checkpoint_transaction_bytes=transaction,
            transaction_estimator='conservative largest payload+metadata+complete history journal+4096-byte publication allowance; not measured write volume',
            elapsed_ns=time.perf_counter_ns()-STARTED,
            limitations=['one teacher/repetition; runtime variance not calibrated',
                         'phase clocks nested; do not add to containing elapsed',
                         'this is not a full resource admission or completed M0 cohort'])
    except BaseException as error:
        manifest.update(status='failed_or_interrupted', reason=f'{type(error).__name__}: {error}',
                        elapsed_ns=time.perf_counter_ns()-STARTED)
        raise
    finally:
        write_new_json(root/'result.json', manifest)
    print(json.dumps({k:manifest[k] for k in ('status','elapsed_ns','retained_bytes','peak_rss_bytes')}), flush=True)


if __name__ == '__main__':
    main()
