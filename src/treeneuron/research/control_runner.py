"""Finite planted/negative control fits, all forked from one common parent."""
import copy
from dataclasses import fields
from pathlib import Path
import time

import numpy as np
import torch

from treeneuron import train
from treeneuron.artifacts import create_run, finish_run
from treeneuron.checkpoint import save_checkpoint
from treeneuron.growth.allocate import allocate, Allocation
from treeneuron.growth.insert import insert_growth
from treeneuron.growth.score import score_units
from .controls import initialize_control_state
from .ledger import write_new_json
from .roles import FittingData
from .snapshots import SnapshotSink


def fit_control_pair(config, data, root, roles, *, control=None, guard=lambda: None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    original_backend = train.BACKEND_REPORT_PATH
    train.BACKEND_REPORT_PATH = Path(config.execution.backend_report)
    started = time.perf_counter_ns()
    report = dict(status='running', methods={}, models={}, costs={},
                  generating_eligibility=None, T_accessed=False, fit_V_accessed=False)
    try:
        if not isinstance(data, FittingData):
            raise ValueError('Control worker requires F/G-only fitting data')
        view = data
        paths = create_run(root/'runs', config.to_dict())
        tick = time.perf_counter_ns()
        sink = SnapshotSink(root/'snapshots', (250, 1000))
        if control is not None:
            parent = initialize_control_state(control, config, paths, data=view)
        else:
            parent = train.fit_parent(view, config, paths, snapshot_updates=(250, 1000), on_snapshot=sink)
            sink.finalize()
        finish_run(paths, {'selected_g_mse': parent.stage['best_mse']}, 'completed')
        report['costs']['parent_ns'] = time.perf_counter_ns()-tick
        save_checkpoint(parent, root/'parent.json')
        snapshot = train.pack_state(parent)
        tick = time.perf_counter_ns()
        table = score_units(copy.deepcopy(parent.model).double().eval(),
            torch.tensor(parent.preprocessing.transform_X(data.F.X), dtype=torch.float64),
            torch.tensor(parent.preprocessing.transform_y(data.F.y), dtype=torch.float64),
            epsilon=config.growth.epsilon, tau=config.model.tau,
            adam_epsilon=config.training.adam_epsilon, batch_size=config.training.batch_size,
            unit_block_size=config.growth.block_units)
        with (root/'score-table.npz').open('xb') as stream:
            np.savez(stream, **{f.name: getattr(table, f.name).numpy() if isinstance(getattr(table, f.name), torch.Tensor)
                              else np.asarray(getattr(table, f.name)) for f in fields(table)})
        report['costs'].update(score_ns=time.perf_counter_ns()-tick, probe_ns=0)
        eligible = allocate(table, len(table.ids), policy='adam', seed=config.seeds.allocation).ids.tolist()
        if control is not None:
            report['generating_eligibility'] = {str(i): i in eligible for i in control.generating_ids}
        for role in roles:
            guard()
            tick = time.perf_counter_ns()
            record = report['methods'][role] = dict(status='running')
            state = train.unpack_state(snapshot, expected_hashes=parent.identity_hashes, device='cpu')
            if role != 'keep':
                if role in ('top', 'random'):
                    allocation = allocate(table, config.growth.m_requested,
                        policy='adam' if role == 'top' else 'random', seed=config.seeds.allocation)
                else:
                    if any(i not in eligible for i in control.generating_ids):
                        record.update(status='completed_ineligible', reason='generating-ID eligibility failure retained')
                        continue
                    rows = [table.ids.tolist().index(i) for i in control.generating_ids]
                    contrasts = table.contrasts[rows].clone()
                    if role == 'oracle_true':
                        ts = state.preprocessing.to_state()
                        mu = np.asarray([c['mean'] for c in ts['columns']])
                        scale = np.asarray([c['scale'] for c in ts['columns']])
                        raw = control.contrasts
                        transported = raw.copy()
                        transported[:, :2] *= scale
                        transported[:, 2] += raw[:, :2] @ mu
                        contrasts = torch.from_numpy(transported)
                    allocation = Allocation(config.growth.m_requested, len(rows),
                        torch.tensor(control.generating_ids, dtype=torch.int64), contrasts, 'greedy',
                        config.seeds.allocation, table.table_hash, torch.zeros(len(rows), dtype=torch.int64),
                        {'shared_scored_units': len(table.ids), 'standalone_contrast_units_required': 0})
                insert_growth(state, allocation)
            paths = create_run(root/'runs', config.to_dict())
            state = train.continue_fit(state, view, train.TrainingPolicy.from_config(config, continuation=True), paths)
            finish_run(paths, {'selected_g_mse': state.stage['best_mse']}, 'completed')
            pointer = root/'models'/f'{role}.json'
            digest = save_checkpoint(state, pointer)
            import hashlib
            report['models'][role] = dict(path=str(pointer.resolve()),
                sha256=hashlib.sha256(pointer.read_bytes()).hexdigest(), loader='training_state')
            record.update(status='completed', enclosing_ns=time.perf_counter_ns()-tick,
                selected_absolute_update=state.update, selected_stage_update=state.update-state.stage['start_update'],
                executed_stage_updates=state.stage['updates_completed'], total_parameters=sum(p.numel() for p in state.model.parameters()))
        report['status'] = 'completed'
        return report
    except BaseException as error:
        report.update(status='failed', failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['physical_elapsed_ns'] = time.perf_counter_ns()-started
        write_new_json(root/'control-pair.json', report)
        train.BACKEND_REPORT_PATH = original_backend
