"""ANT-inspired restricted allocation: exhaustive independent single-unit probes.

G ranks short continuations; the fixed requested count comes from the parent's
resolved growth config. Continued keep is retained separately. This is not full
ANT reproduction. All attempted work is charged, including failures and losers.
The caller applies the returned allocation with build_causal('B6', {}, parent=...)
for a fresh original-parent joint continuation, then compares joint/keep on G.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, fields, replace
import json
import hashlib
import time
import sys

import numpy as np
import torch

from treeneuron.artifacts import (RunPaths, create_run, append_event, update_run_observations,
                                 finish_run, _atomic_json, TERMINAL_STATUSES)
from treeneuron.baselines.causal import _selected_parent, _validate_context
from treeneuron.checkpoint import save_checkpoint, prune_checkpoints
from treeneuron.experiments.paired import _json_tree
from treeneuron.config import canonical_hash
from treeneuron.data.types import DevelopmentData
from treeneuron.growth.allocate import Allocation, allocate
from treeneuron.growth.insert import insert_growth
from treeneuron.growth.score import ScoreTable
from treeneuron.train import (TrainingState, TrainingPolicy, capture_rng, restore_rng,
                              unpack_state, continue_fit, _fg_hashes)


def _eligible_rows(table):
    # Reuse the allocator's full common eligibility, retaining original tensor rows.
    eligible_ids = allocate(table, len(table.ids), policy='adam', seed=0).ids.tolist()
    return [row for row, unit in enumerate(table.ids.tolist()) if unit in eligible_ids]


def _allocation(table, rows, requested, seed):
    rows = sorted(rows, key=lambda row: int(table.ids[row]))
    selected = torch.tensor(rows, dtype=torch.int64)
    return Allocation(requested, len(rows), table.ids[selected].clone(), table.contrasts[selected].clone(),
        'greedy', seed, table.table_hash, torch.zeros(len(rows), dtype=torch.int64),
        {'shared_scored_units':len(table.ids), 'standalone_contrast_units_required':0})


def _rank_allocation(table, requested, seed, *, keep_mse, unit_mse):
    rows = _eligible_rows(table)
    if (set(unit_mse) != {int(table.ids[row]) for row in rows}
            or not np.isfinite([keep_mse, *unit_mse.values()]).all()):
        raise ValueError('Require complete finite G results for every eligible unit and keep')
    # Equivalent to descending keep_mse - unit_mse; no positive-improvement filter.
    ranked = sorted(rows, key=lambda row: (unit_mse[int(table.ids[row])], int(table.ids[row])))
    return _allocation(table, ranked[:min(requested,len(rows))], requested, seed)


def run_greedy_probes(parent: TrainingState, table: ScoreTable, data: DevelopmentData, *,
                      probe_updates: int, seed: int, paths: RunPaths,
                      stopping_mode: str = 'fixed_updates', resource_guard=None) -> Allocation:
    """Return a fixed-count allocation, retaining trials/costs in paths/greedy.json.

    Failures abort selection; unattempted candidates remain explicitly listed.
    The enclosing clock includes validation, parent snapshotting, all setup,
    continued keep, every probe, ranking and child publication. Its own final
    report/ledger publication is excluded from the measured boundary.
    Child durations nest inside it and must never be added again. Initial parent
    fitting and shared score construction precede this call and are excluded.
    """
    started = time.perf_counter_ns()
    before = capture_rng()
    report = dict(schema_version=1, method='B6', recipe='ANT-inspired restricted allocation',
        complete=False, allocation=None, trials=[], not_attempted=[],
        requested_count=parent.config.growth.m_requested, score_hash=table.table_hash,
        probe_updates=probe_updates, stopping_mode=stopping_mode, selection_split='G', V_accessed=False, T_accessed=False,
        cost_boundary='whole call including validation, cloning, keep, losers, failed attempts, checkpoint IO and ranking until final report publication; excludes prior parent fit/shared scoring and final report publication; nested durations are not additive')
    try:
        if stopping_mode not in ('fixed_updates', 'patience'):
            raise ValueError('Unknown probe stopping mode')
        snapshot = _selected_parent(parent)
        _validate_context(data)
        if parent.data_hashes != _fg_hashes(data):
            raise ValueError('Parent F/G identities differ from supplied data')
        normal = TrainingPolicy.from_config(parent.config, continuation=True)
        if type(probe_updates) is not int or not 0 < probe_updates <= normal.max_updates:
            raise ValueError('Probe allowance must be positive and no greater than continuation allowance')
        policy = replace(normal, max_updates=probe_updates)
        # Validate seed via the shared allocator, without using its placement ranking.
        allocate(table, 0, policy='keep', seed=seed)
        if table.ids.tolist() != list(range(parent.model.layer.h)) or table.contrasts.shape != (parent.model.layer.h,parent.model.layer.d+1):
            raise ValueError('Score table must cover the complete original parent unit rows')
        rows = _eligible_rows(table)
        candidates = [('keep',None)] + [(f'unit_{int(table.ids[r])}',r) for r in sorted(rows,key=lambda r:int(table.ids[r]))]
        report['not_attempted'] = [name for name,_ in candidates]
        report['eligible_ids'] = [int(table.ids[r]) for r in rows]
        report['effective_count'] = min(report['requested_count'],len(rows))
        score_path = paths.directory/'score-table.npz'
        with score_path.open('xb') as stream:
            np.savez(stream, **{field.name:(getattr(table,field.name).detach().cpu().numpy()
                if isinstance(getattr(table,field.name),torch.Tensor) else np.asarray(getattr(table,field.name)))
                for field in fields(table)})
        report['score_artifact'] = {'path':score_path.name,'sha256':hashlib.sha256(score_path.read_bytes()).hexdigest(),
                                    'format':'nonpickled numpy arrays; complete original ScoreTable fields'}
        counts = {split:{'rows':data.manifest.assignments.count(split),
            'groups':len({group for group,part in zip(data.manifest.group_id,data.manifest.assignments) if part==split})}
            for split in ('F','G','V')}

        report['policy'] = asdict(policy)
        report['parent_identity_hashes'] = copy.deepcopy(parent.identity_hashes)
        report['parent_state_sha256'] = canonical_hash(_json_tree(snapshot))
        report['parent_checkpoint_sha256'] = save_checkpoint(parent, paths.directory/'parent.json')
        update_run_observations(paths, {'method':{'scope':'coordinator'},
            'identity':{'score_hash':table.table_hash,'code_revision':parent.identity_hashes['code'],
                        'lock_hash':parent.identity_hashes['lock']}})
        append_event(paths, {'type':'probe_plan','status':'running', 'candidates':report['not_attempted'],
                            'parent_checkpoint_sha256':report['parent_checkpoint_sha256'], 'policy':asdict(policy)})
        unit_mse = {}
        for name,row in candidates:
            if resource_guard is not None: resource_guard()
            tick = time.perf_counter_ns()
            child = create_run(paths.directory/'probes',parent.config.to_dict())
            trial = dict(candidate=name, unit_id=None if row is None else int(table.ids[row]),
                run_id=child.run_id, path=str(child.directory.relative_to(paths.directory)),
                method_id='B6', stage='continued_keep_probe' if row is None else 'single_unit_probe',
                parent_identity_hashes=copy.deepcopy(parent.identity_hashes),
                requested_count=report['requested_count'],effective_count=report['effective_count'],
                selection_split='G',V_role='not_accessed',normalization='F target variance',counts=counts,
                outcome='running', stopping_mode=stopping_mode, allowance=probe_updates,
                selected_update=None, selected_stage_updates=None, executed_updates=None,
                stopping_reason=None, hit_ceiling=None,
                unavailable_reason='training has not returned a completed selection', nanoseconds=None)
            report['trials'].append(trial)
            report['not_attempted'].remove(name)
            try:
                # Durable linkage precedes reconstruction, insertion and every training call.
                append_event(paths, {'type':'probe_registered', **trial,
                    'parent_checkpoint_sha256':report['parent_checkpoint_sha256'], 'score_hash':table.table_hash})
                update_run_observations(child, {'identity':{'coordinator_run_id':paths.run_id,
                    'score_hash':table.table_hash,'code_revision':parent.identity_hashes['code'],
                    'lock_hash':parent.identity_hashes['lock']},
                    'data':{'raw_hash':parent.identity_hashes['raw'],'split_hash':parent.identity_hashes['split'],
                            'transform_hash':parent.identity_hashes['transform'],'counts':counts}})
                state = unpack_state(snapshot,expected_hashes=parent.identity_hashes,device=parent.device)
                state.history = copy.deepcopy(parent.history)
                if row is not None:
                    insert_growth(state,_allocation(table,[row],1,seed))
                state.growth_metadata['greedy_probe'] = dict(candidate=name,coordinator_run_id=paths.run_id,
                    parent_checkpoint_sha256=report['parent_checkpoint_sha256'],score_hash=table.table_hash,
                    requested_count=report['requested_count'],policy=asdict(policy),selection_split='G',
                    stopping_mode=stopping_mode,allowance=probe_updates)
                append_event(child, {'type':'probe_initialized','growth_hash':canonical_hash(state.growth_metadata)})
                append_event(child, {'type':'probe_started','candidate':name,
                    'selected_update':None,'executed_updates':None,'unavailable_reason':trial['unavailable_reason'],
                    'parent_update':parent.update,'parent_cursor':parent.cursor,'policy':asdict(policy)})
                selected = continue_fit(state,data,policy,child)
                trial.update(selected_update=selected.update,executed_updates=selected.stage['updates_completed'],
                    g_mse=selected.stage['best_mse'],unavailable_reason=None)
                if stopping_mode == 'fixed_updates' and trial['executed_updates'] != probe_updates:
                    raise ValueError('Incomplete probe allowance: patience stopped before the fixed F allowance')
                stage = selected.stage
                executed = trial['executed_updates']
                if (stage.get('complete') is not True or stage.get('policy') != asdict(policy)
                        or type(executed) is not int or not 0 <= executed <= probe_updates
                        or not np.isfinite(trial['g_mse'])):
                    raise ValueError('Probe must return a finite completed stage under its declared policy')
                ceiling = executed == probe_updates
                patience = stage['bad_evaluations'] >= policy.patience
                if not ceiling and not patience:
                    raise ValueError('Probe has unexplained shortened return')
                selected_updates = selected.update-stage['start_update']
                if not 0 <= selected_updates <= executed:
                    raise ValueError('Probe selected update is outside executed stage')
                trial.update(selected_stage_updates=selected_updates, hit_ceiling=ceiling,
                    stopping_reason='ceiling' if ceiling else 'patience')
                selected.growth_metadata['greedy_probe'].update({key:trial[key] for key in
                    ('executed_updates','selected_stage_updates','stopping_reason','hit_ceiling')})
                append_event(child, {'type':'probe_stopping_evidence', **{key:trial[key] for key in
                    ('stopping_mode','allowance','executed_updates','selected_stage_updates','stopping_reason','hit_ceiling')}})
                digest = save_checkpoint(selected, child.directory/'checkpoints'/'best.json')
                retention = prune_checkpoints(child.directory/'checkpoints')
                update_run_observations(child, {'identity':{'growth_hash':canonical_hash(selected.growth_metadata)},
                    'method':{'weight_hash':digest,
                    'serialized_bytes':retention['referenced_bytes'],'checkpoint_retention':retention}})
                trial['outcome'] = 'completed'
                if row is not None: unit_mse[int(table.ids[row])] = trial['g_mse']
            except BaseException as error:
                trial.update(outcome='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                             failure=f'{type(error).__name__}: {error}')
                raise
            finally:
                trial['nanoseconds'] = time.perf_counter_ns()-tick
                # Training may already have terminalized a failure. The coordinator
                # owns the enclosing attempt charge and never reopens that child.
                primary = sys.exception()
                publication_error = None
                try:
                    append_event(paths, {'type':'probe_result',**trial})
                except BaseException as error:
                    publication_error = error
                try:
                    manifest = json.loads((child.directory/'manifest.json').read_text())
                    if manifest['identity']['status'] not in TERMINAL_STATUSES:
                        update_run_observations(child, {'costs':{'probe':{
                            'nanoseconds':trial['nanoseconds'],'status':'measured','reason':None,
                            'boundary':'attempt creation through setup/training return; excludes final trial publication; nested in coordinator'}}})
                        finish_run(child,trial,trial['outcome'])
                except BaseException as error:
                    if publication_error is None: publication_error = error
                    else: publication_error.add_note(f'Child publication also failed: {error}')
                if publication_error is not None:
                    if primary is None: raise publication_error
                    primary.add_note(f'Probe publication also failed: {publication_error}')
        allocation = _rank_allocation(table,report['requested_count'],seed,
            keep_mse=report['trials'][0]['g_mse'],unit_mse=unit_mse)
        report['allocation'] = dict(policy='greedy',ids=allocation.ids.tolist(),
            contrasts=allocation.contrasts.tolist(),requested_count=allocation.requested_count,
            effective_count=allocation.effective_count,table_hash=allocation.table_hash,seed=seed)
        report['keep_g_mse'] = report['trials'][0]['g_mse']
        report['improvements'] = {str(unit):report['keep_g_mse']-mse for unit,mse in unit_mse.items()}
        report['keep_decision'] = 'separate candidate; compare with fresh joint continuation on G'
        report['complete'] = True
        return allocation
    except BaseException as error:
        report['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        primary = sys.exception()
        publication_error = None
        try:
            report['nanoseconds'] = time.perf_counter_ns()-started
            # Independent writes retain the measured charge even if report-file
            # publication fails. Preserve the original training/setup exception.
            publications = (
                lambda: _atomic_json(paths.directory/'greedy.json',report),
                lambda: update_run_observations(paths, {'costs':{'probe':{
                    'nanoseconds':report['nanoseconds'],'status':'measured','reason':None,
                    'boundary':report['cost_boundary']}}}),
                lambda: append_event(paths, {'type':'greedy_completed' if report['complete'] else 'greedy_failed',
                    'nanoseconds':report['nanoseconds'],'complete':report['complete']}))
            for publish in publications:
                try:
                    publish()
                except BaseException as error:
                    if publication_error is None: publication_error = error
                    else: publication_error.add_note(f'Further publication failure: {error}')
            if publication_error is not None:
                if primary is None: raise publication_error
                primary.add_note(f'Greedy report publication also failed: {publication_error}')
        finally:
            restore_rng(before)
