"""Minimal paired exploratory pilot driver and retained engineering attempts.

Scientific execution requires the separately verified pre-pilot readiness gate.
Tasks 9/10/13 implementation/review must precede any genuine numeric WP05 pilot.
No V labels are read and no protected observation namespace is generated.
"""
from __future__ import annotations
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
import yaml

from treeneuron import train
from treeneuron.locators import locate
from treeneuron.artifacts import create_run, finish_run, _atomic_json, TERMINAL_STATUSES
from treeneuron.backend import lock_hash, runtime_hash
from treeneuron.baselines.api import FitRequest
from treeneuron.baselines.causal import build_causal, ordinary_width_brackets
from treeneuron.baselines.greedy import run_greedy_probes
from treeneuron.checkpoint import load_checkpoint
from treeneuron.config import ExperimentConfig, canonical_hash, resolve_experiment, _UniqueLoader
from treeneuron.data.synthetic import derive_seed, make_from_config, sample_teacher
from treeneuron.data.types import array_hash
from treeneuron.experiments.paired import (pairing_snapshot, audit_pairing, _allocation_record,
    _score_record, _stage_record, file_hash, write_json)
from treeneuron.growth.allocate import allocate
from treeneuron.growth.score import score_units
from .design import PilotSummary, summarize_pilot, calibration_status, choose_design


def teacher_inputs(config: ExperimentConfig, index: int):
    """Independent teacher and F/G/V/E streams, fixed across inner repetitions."""
    if type(index) is not int or index < 0 or config.pilot is None:
        raise ValueError('Pilot config and nonnegative teacher index required')
    value=config.to_dict()
    for name in ('teacher','reference','data'):
        value['seeds'][name]=derive_seed(getattr(config.seeds,name),f'exploratory/pilot/teacher-{index}/{name}')
    teacher_config=resolve_experiment(value)
    teacher,data=make_from_config(teacher_config)
    evaluation_seed=derive_seed(teacher_config.seeds.data,'exploratory/pilot/E')
    e=sample_teacher(teacher,config.pilot.evaluation_rows,evaluation_seed,config.data.noise_sigma)
    if set(e.row_id)&set(data.manifest.row_id): raise ValueError('E overlaps development observations')
    return teacher,data,e,teacher_config


def repetition_config(config: ExperimentConfig, repetition: int) -> ExperimentConfig:
    if type(repetition) is not int or repetition < 0: raise ValueError('Invalid repetition')
    value=config.to_dict()
    for name in ('optimizer','allocation','contrast'):
        value['seeds'][name]=derive_seed(getattr(config.seeds,name),
            f'exploratory/pilot/teacher-{config.seeds.teacher}/repetition-{repetition}/{name}')
    value['seeds']['optimizer'] %= 2**32
    return resolve_experiment(value)


def _runtime_preflight(config):
    if config.backend.device!='cpu' or config.backend.cpu_threads!=1 or config.backend.interop_threads!=1:
        raise ValueError('Pilot engineering runtime requires explicit CPU 1/1')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1': raise ValueError(f'{name}=1 required before imports')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':
        raise ValueError('PYTORCH_ENABLE_MPS_FALLBACK=0 required before imports')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)  # fresh CLI process; never change pytest controller
    train.TrainingPolicy.from_config(config)
    train.TrainingPolicy.from_config(config,continuation=True)
    receipt=locate(config.execution.backend_report,producer=True)
    # The numerical training boundary validates full producer/lock/runtime receipt.
    return dict(cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        receipt_path=str(receipt.resolve()),receipt_sha256=file_hash(receipt))


def fit_repetition(config, data, root: Path, recipe_hash: str) -> dict:
    """Fresh complete parent, paired forks, B2/B3 attribution, B6 G ranking.

    No evaluation observations are accepted. All child runs are registered
    before training and failures retain terminal records and the enclosing cost.
    No cross-source or cross-attempt checkpoint reuse is supported.
    """
    started=time.perf_counter_ns()
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',teacher_id=data.teacher_id,optimizer_seed=config.seeds.optimizer,
        data_hash=data.manifest.assignment_hash,
        recipe_hash=recipe_hash,controls=[],methods={},costs={},V_accessed=False,T_accessed=False,
        source_sha256=train.source_hash(),failures=[])
    open_child=None
    original_backend=train.BACKEND_REPORT_PATH
    train.BACKEND_REPORT_PATH=locate(config.execution.backend_report,producer=True)
    def register(paths, metadata):
        nonlocal open_child
        open_child=paths
        report.setdefault('child_runs',[]).append(dict(run_id=paths.run_id,path=str(paths.directory),
            growth_metadata=metadata))
        _atomic_json(root/'pair.json',report,replace=True)
    def finish(paths,metrics):
        nonlocal open_child
        finish_run(paths,metrics,'completed'); open_child=None
    def load_parent():
        fork=load_checkpoint(checkpoint,expected_hashes=identities,device='cpu')
        if pairing_snapshot(fork)!=report['parent_pairing']:
            raise ValueError('Loaded parent pairing changed')
        return fork
    try:
        parent_paths=create_run(root/'runs',config.to_dict())
        register(parent_paths,{})
        tick=time.perf_counter_ns()
        parent=train.fit_parent(data,config,parent_paths)
        finish(parent_paths,{'selected_g_mse':parent.stage['best_mse']})
        report['costs']['parent_ns']=time.perf_counter_ns()-tick
        checkpoint=parent_paths.directory/'checkpoints'/'best.json'
        identities=dict(parent.identity_hashes)
        report.update(parent_pairing=pairing_snapshot(parent),parent_checkpoint_sha256=file_hash(checkpoint),
            parent_identity_hashes=identities,parent_run_id=parent_paths.run_id)
        tick=time.perf_counter_ns()
        x=torch.tensor(parent.preprocessing.transform_X(data.F.X),dtype=torch.float64)
        y=torch.tensor(parent.preprocessing.transform_y(data.F.y),dtype=torch.float64)
        table=score_units(copy.deepcopy(parent.model).double().eval(),x,y,
            epsilon=config.growth.epsilon,tau=config.model.tau,adam_epsilon=config.training.adam_epsilon,
            batch_size=config.training.batch_size,unit_block_size=config.growth.block_units)
        write_json(root/'score-table.json',_score_record(table))
        allocations={name:allocate(table,config.growth.m_requested,policy=policy,seed=config.seeds.allocation)
            for name,policy in (('top','adam'),('random','random'))}
        report['costs']['score_ns']=time.perf_counter_ns()-tick
        tick=time.perf_counter_ns()
        probes=create_run(root/'runs',config.to_dict()); register(probes,{})
        allocations['greedy']=run_greedy_probes(parent,table,data,
            probe_updates=config.pilot.probe_updates,seed=config.seeds.allocation,paths=probes)
        report['greedy_report']=str(probes.directory/'greedy.json')
        finish(probes,{'complete':True})
        report['costs']['probe_ns']=time.perf_counter_ns()-tick
        target=(config.model.d+2)*config.model.h+1+2*allocations['top'].effective_count*(config.model.d+1)
        lower,upper=ordinary_width_brackets(config.model.d,target)
        report['capacity']=dict(effective_count=allocations['top'].effective_count,
            target_parameters=target,ordinary_widths=[lower,upper],
            ordinary_parameters=[(config.model.d+2)*w+1 for w in (lower,upper)])
        # Discard in-memory parent before reconstruction; every warm fork loads
        # this attempt's full G-selected checkpoint with source/lock validation.
        del parent,x,y
        recipes=[('keep','B1',{},True),('top','B5',{'variant':'adam'},True),
            ('random','B4',{},True),('greedy','B6',{},True),
            ('widen_upper','B2',dict(variant='widen',added_units=upper-config.model.h,seed=config.seeds.contrast),True),
            ('ordinary_lower','B2',dict(variant='lower',target_parameters=target,experiment=config.to_dict()),False),
            ('ordinary_upper','B2',dict(variant='upper',target_parameters=target,experiment=config.to_dict()),False),
            ('static_partial','B3',dict(variant='partial',coordinates='centered',
                count=allocations['top'].effective_count,experiment=config.to_dict()),False)]
        fitted={}
        for name,method,recipe,warm in recipes:
            tick=time.perf_counter_ns()
            allocation=allocations.get(name)
            record=dict(method_id=method,recipe=recipe,stages=[],status='running')
            report['methods'][name]=record
            if warm:
                parent=load_parent()
                adapter=build_causal(method,recipe,parent=parent,allocation=allocation)
                del parent
                record.update(parent_run_id=parent_paths.run_id,
                    parent_checkpoint_sha256=report['parent_checkpoint_sha256'],
                    initial_pairing=pairing_snapshot(adapter.state))
                if record['initial_pairing']!=report['parent_pairing']: raise ValueError('Insertion changes pairing')
                if allocation is not None: record['allocation']=_allocation_record(allocation)
                allowances=[config.training.continuation_updates]
            else:
                adapter=build_causal(method,recipe)
                allowances=[config.training.base_updates,config.training.continuation_updates]
            for allowance in allowances:
                request=FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,config.seeds.optimizer,
                    {'development_data':data},{'max_updates':allowance},root/'runs')
                adapter.fit(request,on_run_created=register)
                record['stages'].append(_stage_record(adapter.state,adapter.run_paths,root))
                finish(adapter.run_paths,{'selected_g_mse':adapter.state.stage['best_mse']})
            record.update(status='completed',executed_row_ids=record['stages'][-1]['row_hashes'],
                selected_g_mse=adapter.state.stage['best_mse'],
                total_parameters=sum(p.numel() for p in adapter.state.model.parameters()),
                enclosing_ns=time.perf_counter_ns()-tick)
            fitted[name]=adapter
        report['pairing_audit']=audit_pairing(report)
        report['greedy_keep_decision']=('greedy' if report['methods']['greedy']['selected_g_mse']<
            report['methods']['keep']['selected_g_mse'] else 'keep')
        report['models']={}
        for name,adapter in fitted.items():
            path=root/'models'/f'{name}.json'
            path.parent.mkdir(exist_ok=True)
            adapter.save(path)
            # Caller remaps these locators to its repository-relative references.
            report['models'][name]=dict(path=str(path.resolve()),sha256=file_hash(path))
        report.update(controls=['B1','B2','B3','B4','B5','B6'],status='completed')
        return report
    except BaseException as error:
        report['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        report['failures'].append(f'{type(error).__name__}: {error}')
        if open_child is not None:
            manifest=json.loads((open_child.directory/'manifest.json').read_text())
            if manifest['identity']['status'] not in TERMINAL_STATUSES:
                finish_run(open_child,{'failure':report['failures'][-1]},report['status'])
        raise
    finally:
        report['enclosing_ns']=time.perf_counter_ns()-started
        report['costs']['aggregation']='enclosing only; all component clocks are nested'
        report['costs']['boundary']='fresh parent, shared scoring, B6 keep/loser probes, all controls, selected model saves and IO; excludes final pair ledger write'
        report['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
        report['retained_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        try:
            _atomic_json(root/'pair.json',report,replace=True)
        finally:
            train.BACKEND_REPORT_PATH=original_backend


def run_repetition(config, data, evaluation, root: Path, recipe_hash: str) -> dict:
    """Exploratory E wrapper around the common label-free fitting graph."""
    started=time.perf_counter_ns()
    report=fit_repetition(config,data,root,recipe_hash)
    root=Path(root)
    from treeneuron.evaluation.confirm import load_predictors
    try:
        predictors=load_predictors(root,report['models'])
        predictions={name:predict(evaluation.X) for name,predict in predictors.items()}
        np.savez_compressed(root/'E-predictions.npz',**predictions,y=evaluation.y,row_id=evaluation.row_id)
        scale=float(np.var(data.F.y))
        if scale<=0: raise ValueError('Undefined F target variance for paired normalized loss')
        losses={name:(prediction-evaluation.y)**2/scale for name,prediction in predictions.items()}
        report.update(evaluation_hash=canonical_hash({'X':array_hash(evaluation.X),
            'y':array_hash(evaluation.y),'rows':array_hash(evaluation.row_id)}),
            h1_rows=(losses['random']-losses['top']).tolist(),
            static_rows=(losses['ordinary_upper']-losses['static_partial']).tolist(),
            metric='paired squared error / F target variance (ddof=0); positive favors top/static',
            E_predictions_sha256=file_hash(root/'E-predictions.npz'))
        return report
    except BaseException as error:
        report['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        report['failures'].append(f'{type(error).__name__}: {error}')
        raise
    finally:
        report['enclosing_ns']=time.perf_counter_ns()-started
        report['costs']['boundary']='fresh parent, shared scoring, B6 keep/loser probes, all controls, predictions and IO; excludes final pair ledger write'
        report['retained_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        _atomic_json(root/'pair.json',report,replace=True)


def regenerate_design(directory: Path):
    """Rebuild summaries and design outputs from this attempt's immutable raw ledger."""
    directory=Path(directory)
    raw=json.loads((directory/'raw.json').read_text())
    if raw['provenance']['source_sha256']!=train.source_hash():
        raise ValueError('Pilot source drift; explicit versioned reanalysis required')
    seal=json.loads((directory/'raw-seal.json').read_text())
    if seal['raw_sha256']!=file_hash(directory/'raw.json'):
        raise ValueError('Pilot raw ledger hash mismatch')
    # Completed batch prefix only; a failed or interrupted later batch remains
    # in attempts/costs but cannot become a balanced accepted pilot by omission.
    records=raw['accepted_records']
    if not records: raise ValueError('No complete batch available for summary')
    config=resolve_experiment(raw['config'])
    summary=summarize_pilot(records,batch_runtimes=raw['batch_runtimes'],
        bootstrap_iterations=config.pilot.bootstrap_iterations,seed=config.pilot.bootstrap_seed,
        purpose=raw['purpose'])
    hashes=dict(summary.evidence_hashes,raw=file_hash(directory/'raw.json'),
        source=raw['provenance']['source_sha256'],config=canonical_hash(raw['config']))
    from dataclasses import replace
    summary=replace(summary,evidence_hashes=hashes,failed_pairs=raw['failed_pairs'],
        diagnostics=dict(summary.diagnostics,attempt_directory=str(directory),
            batch_completed_counts=raw['batch_completed_counts'],
            engineering_limitations='Bounded machinery exercise, not final policies or genuine pilot evidence'))
    decision=choose_design(summary,raw['constraints'])
    _atomic_json(directory/'summary.json',asdict(summary),replace=True)
    _atomic_json(directory/'design_decision.json',decision.to_dict(),replace=True)
    _atomic_json(directory/'resource_forecast.json',dict(decision.forecast,
        actual_attempt_ns=raw['enclosing_ns'],actual_retained_bytes=raw['retained_bytes'],
        status=decision.branch,limitations=decision.limitations),replace=True)
    markdown=('# Pilot design decision\n\n'
        f'Purpose: {summary.purpose}. Branch: **{decision.branch}**. Freeze eligible: false.\n\n'
        f'Completed independent teachers: {summary.independent_teachers}; repetitions: {summary.repetitions}. '
        f'K_calc: {decision.K_calc}; K: {decision.K}; n_T: {decision.n_T}; planning effect: {decision.delta_plan}.\n\n'
        +'\n'.join(f'- {reason}' for reason in decision.limitations)+'\n\n'
        +'Raw ledger SHA-256: '+hashes['raw']+'\n\n'
        +'Genuine WP05 execution waits for Tasks 9/10/13 implementation, review and committed source.\n')
    (directory/'design_decision.md').write_text(markdown)
    return summary


def calibrate(config: ExperimentConfig) -> PilotSummary:
    """Run bounded engineering batches (8,16,32... for genuine future policy).

    Engineering may explicitly use smaller fixtures; their summaries cannot
    authorize a design. New invocations always create retained fresh attempts.
    The time cap is checked between atomic paired repetitions, not a hard kill.
    """
    config=resolve_experiment(config.to_dict())
    if config.pilot is None: raise ValueError('Explicit pilot config required')
    if config.model.coordinates != 'selective' or config.model.depth != 1:
        raise ValueError('Pilot requires the ordinary depth-one parent')
    pilot=config.pilot
    readiness=None
    if pilot.purpose=='scientific':
        from treeneuron.protocol import validate_pilot_readiness, _ref
        repository=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
        ready_path=Path(pilot.readiness_path)
        if not ready_path.is_absolute(): ready_path=repository/ready_path
        validate_pilot_readiness(config,ready_path,root=repository)
        readiness=_ref(repository,ready_path)
    root=Path(config.execution.artifacts_root)/('pilot-'+uuid.uuid4().hex)
    root.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter_ns()
    raw=dict(schema_version=1,purpose=pilot.purpose,status='running',config=config.to_dict(),
        utc=datetime.now(timezone.utc).isoformat(),attempts=[],accepted_records=[],failed_pairs=0,
        batch_runtimes=[],batch_completed_counts=[],constraints={},
        provenance=dict(source_sha256=train.source_hash(),lock_sha256=lock_hash(),runtime_sha256=runtime_hash(),
            checkout=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            checkout_status=subprocess.check_output(['git','status','--porcelain'],text=True).splitlines()),
        axes_changed=[],boundary_expansions=[],
        claim_scope='engineering only; common fixed H1 recipe, no V HPO, no protected T; B7/H3 readiness separate')
    if readiness is not None:
        raw['readiness']=readiness
        raw['claim_scope']='scientific exploratory variance calibration; T remains protected'
    _atomic_json(root/'raw.json',raw,replace=True)
    try:
        raw['runtime']=_runtime_preflight(config)
        if pilot.constraints_path:
            path=Path(pilot.constraints_path)
            raw['constraints']=yaml.load(path.read_text(),Loader=_UniqueLoader)
            raw['constraints_sha256']=file_hash(path)
        target=pilot.initial_teachers
        completed=0
        recipe_hash=canonical_hash(config.to_dict())
        while target <= pilot.max_teachers:
            batch_started=time.perf_counter_ns()
            batch=[]
            for index in range(completed,target):
                teacher,data,e,tc=teacher_inputs(config,index)
                directory=root/f'teacher-{index}'; directory.mkdir()
                write_json(directory/'teacher.json',teacher.to_dict())
                write_json(directory/'splits.json',data.manifest.to_dict())
                np.savez_compressed(directory/'observations.npz',E_X=e.X,E_y=e.y,E_rows=e.row_id,
                    **{f'{part}_{key}':getattr(getattr(data,part),key) for part in 'FGV' for key in ('X','y','row_id')})
                for repetition in range(pilot.repetitions):
                    if (time.perf_counter_ns()-started)/1e9 > pilot.max_seconds:
                        raise TimeoutError('Engineering pilot time cap reached between paired repetitions')
                    entry=dict(teacher_index=index,repetition=repetition,status='running')
                    raw['attempts'].append(entry); _atomic_json(root/'raw.json',raw,replace=True)
                    tick=time.perf_counter_ns()
                    try:
                        record=run_repetition(repetition_config(tc,repetition),data,e,
                            directory/f'repetition-{repetition}',recipe_hash)
                        record['repetition']=repetition
                        # Publish the complete accepted schema before the raw ledger
                        # can reference it; the consumer compares all three records.
                        _atomic_json(directory/f'repetition-{repetition}'/'pair.json',record,replace=True)
                        entry.update(status='completed',record=record)
                        batch.append(record)
                    except BaseException as error:
                        entry.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                            failure=f'{type(error).__name__}: {error}')
                        raw['failed_pairs']+=1
                        raise
                    finally:
                        entry['enclosing_ns']=time.perf_counter_ns()-tick
                        _atomic_json(root/'raw.json',raw,replace=True)
            raw['batch_runtimes'].append((time.perf_counter_ns()-batch_started)/1e9/(target-completed))
            raw['accepted_records'].extend(batch)
            raw['batch_completed_counts'].append(target)
            summary=summarize_pilot(raw['accepted_records'],batch_runtimes=raw['batch_runtimes'],
                bootstrap_iterations=pilot.bootstrap_iterations,seed=pilot.bootstrap_seed,purpose=pilot.purpose)
            accepted,reasons=calibration_status(summary,pilot.endpoint_tolerance)
            raw['calibration_status']=dict(accepted=accepted,reasons=reasons)
            completed=target
            if accepted: break
            target*=2
            if target<=pilot.max_teachers: raw['boundary_expansions'].append(dict(axis='teachers',target=target))
        raw['status']='completed'
    except BaseException as error:
        raw['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        raw['failure']=f'{type(error).__name__}: {error}'
        raise
    finally:
        raw['enclosing_ns']=time.perf_counter_ns()-started
        raw['retained_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        raw['cost_boundary']='whole attempt including failed/incomplete batches; child/probe clocks nested, never additive; excludes final publication'
        _atomic_json(root/'raw.json',raw,replace=True)
        if pilot.purpose=='scientific':
            from treeneuron.protocol import pilot_seal
            seal=pilot_seal(root)
        else:
            seal=dict(raw_sha256=file_hash(root/'raw.json'),source_sha256=raw['provenance']['source_sha256'])
        write_json(root/'raw-seal.json',seal)
    return regenerate_design(root)
