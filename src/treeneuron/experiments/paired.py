"""Serial, one-event exploratory studies with explicit local receipt provenance.

Only F/G/V are generated. G selects complete states; V is descriptive. A failed
attempt is terminal and retained. Retry by invoking the runner again (new UUID).
"""
from __future__ import annotations

import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from treeneuron import train
from treeneuron.artifacts import create_run, finish_run, append_event, update_run_observations, TERMINAL_STATUSES
from treeneuron.backend import runtime_hash, lock_hash
from treeneuron.baselines.api import FitRequest
from treeneuron.baselines.causal import build_causal, ordinary_width_brackets
from treeneuron.baselines.simple import MeanPredictor, RidgePredictor
from treeneuron.checkpoint import load_checkpoint
from treeneuron.config import ExperimentConfig, canonical_hash, canonical_json, resolve_experiment
from treeneuron.data.synthetic import make_from_config
from treeneuron.data.types import array_hash
from treeneuron.doctor import collect_system_report
from treeneuron.evaluation.metrics import regression_metrics
from treeneuron.growth.allocate import allocate
from treeneuron.growth.score import score_units


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, record: dict) -> None:
    payload = canonical_json(record) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(payload)


def _json_tree(value):
    if isinstance(value, torch.Tensor):
        return {'dtype': str(value.dtype), 'shape': list(value.shape),
                'hash': array_hash(value.detach().cpu().numpy())}
    if isinstance(value, np.ndarray):
        return {'dtype': str(value.dtype), 'shape': list(value.shape), 'hash': array_hash(value)}
    if isinstance(value, dict): return {str(k): _json_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_json_tree(v) for v in value]
    return value


def pairing_snapshot(state) -> dict:
    """Training-stream identity, independent of topology and placement RNG."""
    schedule = state.scheduler.state_dict()
    # New groups inherit the same LR; compare the schedule function, not group count.
    if len(set(schedule['base_lrs'])) != 1:
        raise ValueError('pairing requires a common base LR for all groups')
    schedule['base_lrs'] = schedule['base_lrs'][:1]
    return dict(update=state.update, epoch=state.epoch, cursor=state.cursor,
        order=array_hash(state.order.numpy()), sampler=array_hash(state.sampler_generator.get_state().numpy()),
        rng=canonical_hash(_json_tree(state.rng_states)), schedule=schedule,
        logical_batch=state.config.training.batch_size,
        allowance=state.config.training.continuation_updates,
        validation_interval=state.config.training.validation_interval, patience=state.config.training.patience,
        checkpoint_interval=state.config.training.checkpoint_interval)


def audit_pairing(study: dict) -> dict:
    """Fail on initial stream/policy mutation or unequal executed common prefixes."""
    warm = [m for m in study['methods'].values() if m.get('parent_run_id')]
    expected = study['parent_pairing']
    for method in warm:
        if method['initial_pairing'] != expected:
            raise ValueError('pairing initial sampler/RNG/schedule/policy mismatch')
        if method['parent_checkpoint_sha256'] != study['parent_checkpoint_sha256']:
            raise ValueError('pairing parent checkpoint mismatch')
    for method in warm:
        final_stage = method['stages'][-1]
        if (method['executed_row_ids'] != final_stage['row_hashes']
                or len(final_stage['row_hashes']) != final_stage['executed_updates']):
            raise ValueError('pairing duplicate stream/count disagrees with final stage')
    reference = study['methods']['keep']['executed_row_ids']
    for method in warm:
        actual = method['executed_row_ids']
        n = min(len(reference), len(actual))
        if reference[:n] != actual[:n]:
            raise ValueError('pairing executed row stream mismatch')
    top, random = [study['methods'][name]['allocation'] for name in ('top', 'random')]
    for key in ('requested_count', 'effective_count', 'table_hash'):
        if top[key] != random[key]: raise ValueError('pairing allocation count/table mismatch')
    by_id = {j: q for j, q in zip(top['ids'], top['contrasts_float64'])}
    for j, q in zip(random['ids'], random['contrasts_float64']):
        if j in by_id and q != by_id[j]: raise ValueError('pairing per-ID contrast mismatch')
    return dict(passed=True, warm_forks=len(warm), common_prefix_updates=min(len(m['executed_row_ids']) for m in warm),
        row_stream_representation='SHA256 per ordered int64 row-ID minibatch; full IDs in stage events',
        early_stopping='G strict improvement, earliest tie; common executed prefix, no unused-budget reassignment')


def _score_record(table):
    """Finite diagnostic view; the identity still binds the original tensor bytes."""
    record = {}
    unavailable = []
    for field in fields(table):
        value = getattr(table, field.name)
        if not isinstance(value, torch.Tensor):
            record[field.name] = value
            continue
        exported = value.tolist()
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            # Only row-local, explicitly ineligible statistics may be unavailable.
            # Global Gram statistics and eligible rows remain hard failures.
            if field.name == 'gram':
                raise ValueError('Nonfinite global score statistics')
            for index in (~torch.isfinite(value)).nonzero().tolist():
                unit = index[0]
                if bool(table.eligible[unit]):
                    raise ValueError('Nonfinite eligible score statistics')
                target = exported
                for coordinate in index[:-1]:
                    target = target[coordinate]
                target[index[-1]] = None
                unavailable.append(dict(field=field.name, index=index,
                    unit_id=int(table.ids[unit]), reason=table.status[unit]))
        record[field.name] = exported
    record.update(table_hash=table.table_hash, unavailable_diagnostics=unavailable,
        export_schema_version=1,
        identity_semantics='table_hash binds original ScoreTable bytes; nulls are an export-only diagnostic view')
    return record


def _allocation_record(value):
    return dict(requested_count=value.requested_count, effective_count=value.effective_count,
        ids=value.ids.tolist(), contrasts_float64=value.contrasts.tolist(),
        contrasts_float32=value.contrasts.float().tolist(),
        inserted_contrasts_hash=array_hash(value.contrasts.float().numpy()),
        table_hash=value.table_hash, policy=value.policy, seed=value.seed,
        direction_attempts=value.direction_attempts.tolist(), work_requirements=value.cost)


def _gate_probe(model, x, y, allocation, epsilon, table):
    """First fresh Adam step, full F MSE, gate only, frozen parent/contrasts."""
    before = train.capture_rng()
    try:
        candidate = copy.deepcopy(model).double().eval()
        candidate.layer.grow(allocation.ids, allocation.contrasts)
        if allocation.effective_count == 0:
            return dict(learning_rate=1e-7, predicted_decrease=0., observed_decrease=0., gate_gradient_norm=0., table_predicted_decrease=0., table_gradient_max_abs=0.)
        for parameter in candidate.parameters(): parameter.requires_grad_(False)
        for parameter in candidate.layer.gate.parameters(): parameter.requires_grad_(True)
        loss = (candidate(x)-y).square().mean()
        loss.backward()
        gradient = torch.cat([candidate.layer.gate.weight.grad, candidate.layer.gate.bias.grad[:, None]], 1)
        lr = 1e-7
        predicted = float((gradient.square() / (gradient.abs()+epsilon)).sum()) * lr
        optimizer = torch.optim.AdamW(candidate.layer.gate.parameters(), lr=lr, eps=epsilon, weight_decay=0)
        optimizer.step()
        observed = float(loss.detach() - (candidate(x)-y).square().mean().detach())
        return dict(learning_rate=lr, predicted_decrease=predicted, observed_decrease=observed,
                    table_predicted_decrease=lr*float(table.adam_scores[allocation.ids].sum())*2*(model.layer.d+1),
                    table_gradient_max_abs=float((gradient-table.gradients[allocation.ids]).abs().max()),
                    gate_gradient_norm=float(gradient.norm()), objective='full F standardized MSE, gate-only fresh Adam')
    finally:
        train.restore_rng(before)


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    probabilities = [0., .25, .5, .75, 1.]
    return dict(mean=float(values.mean()), variance=float(values.var()),
        min=float(values.min()), max=float(values.max()),
        rms=float(np.sqrt(np.mean(values**2))), quantile_probabilities=probabilities,
        quantiles=np.quantile(values, probabilities).tolist(),
        quantile_method='linear', variance_ddof=0)


def _regime_occupancy(teacher, data):
    from treeneuron.data.synthetic import TeacherSpec, evaluate_teacher
    common = dict(population='F', rows=len(data.F.X), role='oracle_diagnostic_only')
    if isinstance(teacher, TeacherSpec):
        _, gates = evaluate_teacher(teacher, data.F.X, return_gates=True)
        return dict(**common, status='observed', definition='per teacher component on raw F inputs',
            mean_gate_probability=gates.mean(axis=0).tolist(),
            fraction_at_or_above_half=(gates >= .5).mean(axis=0).tolist())
    if getattr(teacher, 'kind', None) == 'conditional_slope_2d':
        left = int((data.F.X[:, 0] < 0).sum())
        counts = dict(left=left, right=len(data.F.X)-left)
        return dict(**common, status='observed', definition='left: raw x0 < 0; right: raw x0 >= 0',
            counts=counts, fractions={key: value/len(data.F.X) for key, value in counts.items()})
    return dict(**common, status='not_applicable', value=None,
        reason='No supplied population with an applicable synthetic regime definition')


def _diagnostics(state, data, teacher=None):
    """Selected-state F diagnostics; teacher observations never enter fit/score inputs."""
    unavailable = dict(status='not_applicable', value=None, reason='Model has no gated leaf pairs')
    result = dict(schema_version=1, population='F', role='diagnostic_only',
        gate=copy.deepcopy(unavailable), leaf_disagreement=copy.deepcopy(unavailable),
        synthetic_regime_occupancy=_regime_occupancy(teacher, data),
        input_norms=dict(definition='Euclidean norm per F row, excluding the explicit bias coordinate',
            raw=_distribution(np.linalg.norm(data.F.X, axis=1)),
            standardized=dict(status='not_applicable', value=None, reason='No neural F transform supplied')))
    if state is None:
        canonical_json(result)
        return result
    x = torch.tensor(state.preprocessing.transform_X(data.F.X), dtype=torch.float32)
    y = torch.tensor(state.preprocessing.transform_y(data.F.y), dtype=torch.float32)
    result['input_norms']['standardized'] = _distribution(np.linalg.norm(x.numpy().astype(np.float64), axis=1))
    model = copy.deepcopy(state.model).eval()
    loss = (model(x)-y).square().mean()
    loss.backward()
    result.update(fit_standardized_mse=float(loss.detach()),
        gradient_norms={name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None},
        parameter_norms={name: float(p.norm().detach()) for name, p in model.named_parameters()})
    gate = getattr(model.layer, 'gate', None)
    if gate is not None:
        with torch.no_grad():
            probability = torch.sigmoid(gate(x)/state.config.model.tau).double().numpy()
            result['gate'] = dict(**_distribution(probability), status='observed',
                aggregation='pooled F rows and gated units', rows=len(x), units=probability.shape[1],
                mean_g_one_minus_g=float(np.mean(probability*(1-probability))),
                saturation_thresholds=[.01,.99],
                saturated_fraction=float(((probability < .01)|(probability > .99)).mean()))
            contrast = getattr(model.layer, 'contrast', None)
            if contrast is not None:
                difference, coordinates = 2*contrast(x), 'centered'
            else:
                difference, coordinates = model.layer.left(x)-model.layer.right(x), 'direct_leaf'
            rms = float(difference.double().square().mean().sqrt())
            result['leaf_disagreement'] = dict(status='observed', rms=rms, coordinates=coordinates,
                definition='RMS left-minus-right affine preactivation over F rows and gated units')
            result['leaf_separation_rms'] = rms
    canonical_json(result)
    return result


def _predictions(predictor, data, directory, name):
    metrics, links = {}, {}
    durations = {}
    for split in ('G', 'V'):
        rows = getattr(data, split)
        tick = time.perf_counter_ns()
        prediction = predictor.predict(rows.X)
        durations[split] = time.perf_counter_ns()-tick
        metrics[split] = regression_metrics(rows.y, prediction, fit_target_variance=float(np.var(data.F.y)),
            groups=rows.group_id, noisefree=rows.noise_free_target)
        path = directory/'predictions'/f'{name}-{split}.json'
        write_json(path, dict(prediction=prediction.tolist(), y=rows.y.tolist(),
            noise_free_target=None if rows.noise_free_target is None else rows.noise_free_target.tolist(),
            row_id=rows.row_id.tolist(), group_id=rows.group_id.tolist(), split=split,
            selection_role='checkpoint' if split=='G' else 'descriptive_only'))
        links[split] = dict(path=str(path.relative_to(directory)), sha256=file_hash(path))
    return metrics, links, durations


def _stage_record(state, paths, root):
    events = [json.loads(line) for line in (paths.directory/'events.jsonl').read_text().splitlines()]
    updates = [e for e in events if e['type']=='update']
    evaluations = [e for e in events if e['type']=='evaluation']
    executed = len(updates)
    policy = state.stage['policy']
    return dict(run_id=paths.run_id, path=str(paths.directory.relative_to(root)),
        selected_update=state.update, executed_updates=executed, max_updates=policy['max_updates'],
        checkpoint_interval=policy['checkpoint_interval'],
        hit_ceiling=executed==policy['max_updates'],
        improving_at_ceiling=executed==policy['max_updates'] and bool(evaluations[-1]['improved']),
        evaluations=evaluations,
        training_loss=[dict(update=e['update'], value=e['loss_standardized_mse']) for e in updates],
        row_hashes=[array_hash(np.array(e['row_ids'], dtype=np.int64)) for e in updates])


def run_paired(config: ExperimentConfig) -> Path:
    """Execute all declared causal controls serially and return immutable artifacts."""
    config = resolve_experiment(config.to_dict())
    train.TrainingPolicy.from_config(config)  # explicit allowances required
    if config.backend.device != 'cpu' or config.model.coordinates != 'selective':
        raise ValueError('Paired exploration requires the validated ordinary CPU parent')
    if config.training.scheduler != 'constant':
        raise ValueError('This first paired study declares constant LR with no restarts')
    if (config.growth.policy != 'adam' or config.growth.direction != 'eigenvector'
            or config.growth.projection_rms_floor != 1e-12):
        raise ValueError('Paired fixed causal inventory requires growth policy=adam, direction=eigenvector, projection_rms_floor=1e-12')
    tick = time.perf_counter_ns()
    paths = create_run(Path(config.execution.artifacts_root), config.to_dict())
    root = paths.directory
    previous_receipt = train.BACKEND_REPORT_PATH
    previous_rng = train.capture_rng()
    previous_threads = torch.get_num_threads()
    open_child = None
    study: dict[str, Any] = dict(schema_version=1, study_id=config.study_id, phase='exploratory',
        status='running', test_labels_accessed=False, methods={}, costs={}, failures=[],
        execution_semantics=dict(inventory='fixed_causal_comparison',
            proposed=dict(policy='adam', direction='eigenvector'), projection_rms_floor=1e-12,
            selectors='growth policy/direction identify the proposed arm; controls remain fixed'),
        interpretation='One teacher/optimizer seed, descriptive G/V only; no population confidence or superiority claim',
        config=config.to_dict(), config_hash=canonical_hash(config.to_dict()))
    primary_error = None
    def publication_failure(error, phase, primary=None):
        record = dict(type=type(error).__name__, message=str(error), phase=phase,
            primary_error=None if primary is None else dict(type=type(primary).__name__, message=str(primary)))
        failures = study.setdefault('publication_failures', [])
        failures.append(record)
        if primary is not None:
            primary.add_note(f'Secondary {phase} publication failed: {type(error).__name__}: {error}')
        # Independent best-effort evidence, even when study/index/manifest cannot publish.
        try:
            write_json(root/f'publication-failure-{len(failures)}.json', record)
        except BaseException as recording_error:
            (primary if primary is not None else error).add_note(
                f'Publication failure sidecar also failed: {recording_error}')
    try:
        torch.set_num_threads(1)
        train.BACKEND_REPORT_PATH = Path(config.execution.backend_report)
        receipt = json.loads(train.BACKEND_REPORT_PATH.read_text())
        write_json(root/'backend-receipt.json', receipt)
        hardware = json.loads(json.dumps(collect_system_report(), allow_nan=False))
        write_json(root/'hardware.json', hardware)
        hardware_hash = file_hash(root/'hardware.json')
        study['provenance'] = dict(source_sha256=train.source_hash(), lock_sha256=lock_hash(), runtime_sha256=runtime_hash(),
            checkout=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            checkout_status=subprocess.check_output(['git','status','--porcelain'], text=True).splitlines(),
            backend_report_path=str(train.BACKEND_REPORT_PATH.resolve()), backend_report_sha256=file_hash(train.BACKEND_REPORT_PATH),
            hardware_report_sha256=hardware_hash)
        update_run_observations(paths, {'identity': {'hardware_hash':hardware_hash,
            'code_revision':study['provenance']['source_sha256'], 'lock_hash':lock_hash()}})
        teacher, data = make_from_config(config)
        write_json(root/'teacher.json', teacher.to_dict())
        write_json(root/'splits.json', data.manifest.to_dict())
        np.savez_compressed(root/'development.npz', **{f'{split}_{key}':getattr(getattr(data, split),key)
            for split in ('F','G','V') for key in ('X','y','row_id','group_id')})
        study['data'] = dict(teacher_id=data.teacher_id, split_hash=data.manifest.assignment_hash,
            development_sha256=file_hash(root/'development.npz'))
        def request(allowance):
            return FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,config.seeds.optimizer,
                {'development_data':data},{'max_updates':allowance},root/'runs')
        normalization = {key: teacher.to_dict()[key] for key in
            ('reference_mean', 'reference_std', 'reference_rows', 'reference_seed')}
        normalization['target_normalization'] = teacher.to_dict().get('target_normalization', 'reference_center_scale')
        development_observations = dict(
            counts={part: dict(rows=len(getattr(data, part).y), groups=len(np.unique(getattr(data, part).group_id)))
                    for part in ('F', 'G', 'V')},
            teacher_normalization=normalization, boundaries={'V': 'descriptive_only'})
        unavailable_model = {'value': None, 'reason': 'Study coordinator; see individual child methods'}
        update_run_observations(paths, {
            'data': {**development_observations, 'raw_hash': data.manifest.dataset_id,
                     'split_hash': data.manifest.assignment_hash, 'transform_hash': unavailable_model},
            'method': {'scope': 'coordinator', **{key: unavailable_model for key in
                ('trainable_parameters', 'total_parameters', 'serialized_bytes', 'weight_checkpoint',
                 'weight_hash', 'weight_terms', 'context_bytes', 'cache_bytes', 'prediction_artifacts')}}})
        def observe_child(run, state=None, parent_id=None, allocation=None, predictions=None, growth_metadata=None):
            identity = {'hardware_hash': hardware_hash, 'coordinator_run_id': paths.run_id,
                        'growth_hash': canonical_hash(state.growth_metadata if state is not None else growth_metadata)}
            if parent_id is not None: identity['parent_run_id'] = parent_id
            if allocation is not None: identity['score_hash'] = allocation.table_hash
            update_run_observations(run, {'identity': identity, 'data': development_observations,
                'method': {'scope': 'single_model', 'prediction_artifacts': predictions if predictions is not None else
                           {'value': None, 'reason': 'No final G/V prediction artifacts published for this stage'}},
                'resources': {'memory_accounting': 'Process lifetime RSS high-water; MPS unused; not incremental or additive'}})
        def register_child(run, record, name, stage_index, metadata, parent_id=None, allocation=None):
            nonlocal open_child
            open_child = run
            record['growth_metadata'] = copy.deepcopy(metadata)
            stage = dict(run_id=run.run_id, path=str(run.directory.relative_to(root)),
                observation_status='registered_before_training', selected_update=None,
                executed_updates=None, unavailable_reason='Training has not returned a selected state')
            record['stages'].append(stage)
            append_event(paths, dict(type='child_registered', method=name, stage_index=stage_index,
                run_id=run.run_id, path=stage['path'], parent_run_id=parent_id))
            observe_child(run, parent_id=parent_id, allocation=allocation, growth_metadata=metadata)
        # Parent setup, preprocessing, backend validation, fit, diagnostics and prediction are enclosed.
        start = time.perf_counter_ns()
        parent_paths = create_run(root/'runs', config.to_dict())
        study['methods']['parent'] = dict(method_id='B1-parent',status='running',stages=[],
            declared_total_updates=config.training.base_updates)
        register_child(parent_paths, study['methods']['parent'], 'parent', 0, {})
        parent = train.fit_parent(data,config,parent_paths)
        observe_child(parent_paths, parent)
        update_run_observations(paths, {'method': {'backend': {'actual': parent.backend},
            'threads': {'cpu': torch.get_num_threads(), 'interop': torch.get_num_interop_threads()}}})
        metrics, predictions, prediction_ns = _predictions(parent,data,root,'parent')
        parent_stage = _stage_record(parent,parent_paths,root)
        parent_checkpoint = parent_paths.directory/'checkpoints'/'best.json'
        parent_identities = dict(parent.identity_hashes)
        study['parent_pairing'] = pairing_snapshot(parent)
        study['parent_checkpoint_sha256'] = file_hash(parent_checkpoint)
        study['methods']['parent'] = dict(method_id='B1-parent',status='completed',metrics=metrics,predictions=predictions,
            stages=[parent_stage],prediction_ns=prediction_ns,diagnostics=_diagnostics(parent,data,teacher),
            total_parameters=sum(p.numel() for p in parent.model.parameters()),
            selected_update=parent.update,declared_total_updates=config.training.base_updates)
        observe_child(parent_paths, parent, predictions=predictions)
        finish_run(parent_paths,metrics,'completed')
        open_child = None
        study['methods']['parent']['enclosing_ns'] = time.perf_counter_ns()-start
        event_start = time.perf_counter_ns()
        # Real scoring inputs use the learned F-only coordinates, never raw X/y.
        start = time.perf_counter_ns()
        x = torch.tensor(parent.preprocessing.transform_X(data.F.X),dtype=torch.float64)
        y = torch.tensor(parent.preprocessing.transform_y(data.F.y),dtype=torch.float64)
        table = score_units(copy.deepcopy(parent.model).double().eval(), x, y,
            epsilon=config.growth.epsilon,tau=config.model.tau,adam_epsilon=config.training.adam_epsilon,
            batch_size=config.training.batch_size,unit_block_size=config.growth.block_units)
        study['costs']['shared_scoring_ns'] = time.perf_counter_ns()-start
        table_record = _score_record(table)
        table_record.update(table_hash=table.table_hash,transform_hash=parent.identity_hashes['transform'],
                            coordinates='F-only standardized X and y; explicit bias last')
        write_json(root/'score-table.json',table_record)
        update_run_observations(paths, {'identity': {'score_hash': table.table_hash}})
        allocations = {name:allocate(table,config.growth.m_requested,policy=policy,
            seed=config.seeds.contrast if policy=='gaussian' else config.seeds.allocation)
            for name,policy in [('top','adam'),('random','random'),('gaussian','gaussian'),('sgd','sgd')]}
        start = time.perf_counter_ns()
        study['tiny_gate_probe'] = _gate_probe(parent.model,x,y,allocations['top'],config.training.adam_epsilon,table)
        study['costs']['tiny_gate_probe_ns'] = time.perf_counter_ns()-start
        study['costs']['standalone_random_scoring_ns'] = None
        study['costs']['standalone_random_scoring_reason'] = 'Sampled-ID scorer not implemented; shared table measured once, standalone work requirement is not elapsed time'
        parent_prediction = {s:parent.predict(getattr(data,s).X) for s in ('G','V')}
        del parent, x, y
        study['costs']['shared_event_enclosing_ns']=time.perf_counter_ns()-event_start
        study['costs']['shared_parent_enclosing_ns']=study['methods']['parent']['enclosing_ns']
        target = (config.model.d+2)*config.model.h+1+2*allocations['top'].effective_count*(config.model.d+1)
        widths = ordinary_width_brackets(config.model.d,target)
        recipes = [('keep','B1',{},True,None),('sham','B1',{},True,None),
            ('top','B5',{'variant':'adam'},True,allocations['top']),
            ('random','B4',{},True,allocations['random']),
            ('gaussian','B5',{'variant':'gaussian'},True,allocations['gaussian']),
            ('sgd','B5',{'variant':'sgd'},True,allocations['sgd'])]
        for side,width in zip(('lower','upper'),widths):
            recipes.append((f'widen_{side}','B2',{'variant':'widen','added_units':width-config.model.h,
                            'seed':config.seeds.contrast},True,None))
            recipes.append((f'ordinary_{side}','B2',{'variant':side,'target_parameters':target,
                            'experiment':config.to_dict()},False,None))
        for coordinates in ('centered','direct_leaf'):
            for variant in ('partial','full'):
                recipe=dict(experiment=config.to_dict(),variant=variant,coordinates=coordinates)
                if variant=='partial': recipe['count']=allocations['top'].effective_count
                recipes.append((f'static_{variant}_{coordinates}','B3',recipe,False,None))
        for name,method_id,recipe,warm,allocation in recipes:
            start = time.perf_counter_ns()
            record = dict(method_id=method_id,role='proposed' if name=='top' else 'control',recipe=recipe,
                status='running',stages=[],declared_total_updates=config.training.base_updates+config.training.continuation_updates)
            study['methods'][name] = record
            if warm:
                # No prior loaded parent survives into the next load. Checkpoint loader validates full identities.
                fork = load_checkpoint(parent_checkpoint,expected_hashes=parent_identities,device='cpu')
                if pairing_snapshot(fork)!=study['parent_pairing']: raise ValueError('pairing loaded parent mismatch')
                adapter = build_causal(method_id,recipe,parent=fork,
                    allocation=allocation)
                adapted = dict(adapter.state.model.named_parameters())
                for old_name,p in fork.model.named_parameters():
                    new_name = old_name
                    if method_id=='B2' and old_name.startswith('head.'):
                        new_name=old_name.replace('head.','head.base.',1)
                    q=adapted[new_name]
                    if p.data_ptr()==q.data_ptr(): raise ValueError('Independent fork storage required')
                    if not torch.equal(p,q): raise ValueError('Inherited parameter changed')
                    if _json_tree(fork.optimizer.state.get(p,{})) != _json_tree(adapter.state.optimizer.state.get(q,{})):
                        raise ValueError('Inherited optimizer history changed')
                record['inherited_parameter_optimizer_audit']='exact values, moments, steps; independent parameter storage'
                if allocation is not None and allocation.effective_count:
                    actual=torch.cat((adapter.state.model.layer.contrast.weight,
                                      adapter.state.model.layer.contrast.bias[:,None]),1).detach()
                    if not torch.equal(actual,allocation.contrasts.float()):
                        raise ValueError('Inserted float32 contrast differs from declared cast')
                del fork
                record.update(parent_run_id=parent_paths.run_id,parent_checkpoint_sha256=study['parent_checkpoint_sha256'],
                    initial_pairing=pairing_snapshot(adapter.state))
                if record['initial_pairing'] != study['parent_pairing']: raise ValueError('pairing insertion altered stream')
                record['initial_prediction_max_abs'] = max(float(np.abs(adapter.predict(getattr(data,s).X)-parent_prediction[s]).max()) for s in ('G','V'))
                if record['initial_prediction_max_abs'] != 0: raise ValueError('Insertion changed parent predictions')
                if allocation is not None: record['allocation']=_allocation_record(allocation)
                allowances = [config.training.continuation_updates]
            else:
                adapter = build_causal(method_id,recipe)
                # Fresh controls have the same two G-selected stages and total opportunity.
                allowances = [config.training.base_updates,config.training.continuation_updates]
            stage_parent_id = parent_paths.run_id if warm else None
            for stage_index, allowance in enumerate(allowances):
                adapter.fit(request(allowance), on_run_created=lambda run, metadata:
                    register_child(run, record, name, stage_index, metadata, stage_parent_id, allocation))
                stage = _stage_record(adapter.state,adapter.run_paths,root)
                record['stages'][-1] = stage
                observe_child(adapter.run_paths, adapter.state, stage_parent_id, allocation)
                if stage_index < len(allowances)-1:
                    finish_run(adapter.run_paths, {'selected_g_mse': adapter.state.stage['best_mse']}, 'completed')
                    open_child = None
                    stage_parent_id = adapter.run_paths.run_id
            record['executed_row_ids'] = record['stages'][-1]['row_hashes']
            record['growth_metadata'] = copy.deepcopy(adapter.state.growth_metadata)
            record['metrics'],record['predictions'],record['prediction_ns'] = _predictions(adapter,data,root,name)
            record['diagnostics'] = _diagnostics(adapter.state,data,teacher)
            record['selected_update'] = adapter.state.update
            record['total_parameters'] = sum(p.numel() for p in adapter.state.model.parameters())
            record['identities'] = adapter.state.identity_hashes
            observe_child(adapter.run_paths, adapter.state, stage_parent_id, allocation, record['predictions'])
            finish_run(adapter.run_paths, record['metrics'], 'completed')
            open_child = None
            record['status']='completed'
            record['enclosing_ns']=time.perf_counter_ns()-start
            append_event(paths,{'type':'method_completed','name':name,'enclosing_ns':record['enclosing_ns']})
            del adapter
        for name,cls in [('mean',MeanPredictor),('ridge',lambda:RidgePredictor(1.))]:
            start = time.perf_counter_ns()
            predictor = cls().fit(request(config.training.base_updates))
            metrics,predictions,prediction_ns = _predictions(predictor,data,root,name)
            predictor.save(root/f'{name}.json')
            study['methods'][name]=dict(method_id='B0',status='completed',metrics=metrics,predictions=predictions,
                recipe={'alpha':1.} if name=='ridge' else {},prediction_ns=prediction_ns,
                diagnostics=_diagnostics(None,data,teacher),
                declared_total_updates=0,stages=[],enclosing_ns=time.perf_counter_ns()-start)
        study['pairing_audit']=audit_pairing(study)
        study['status']='completed'
    except BaseException as error:
        primary_error = error
        study['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        study['failures'].append(dict(type=type(error).__name__,message=str(error)))
        if open_child is not None:
            try:
                child_manifest = json.loads((open_child.directory/'manifest.json').read_text())
                if child_manifest['identity']['status'] not in TERMINAL_STATUSES:
                    finish_run(open_child, {'failure': study['failures'][-1]}, study['status'])
            except BaseException as publication_error:
                publication_failure(publication_error, 'child terminal', error)
        for record in study['methods'].values():
            if record['status']=='running':
                record['status']=study['status']
                record['enclosing_ns']=time.perf_counter_ns()-start
                record['failure']=study['failures'][-1]
        raise
    finally:
        try:
            study['costs']['total_study_ns']=time.perf_counter_ns()-tick
            study['costs']['boundary']='Enclosing generation, hardware observations, setup, factory/probes, base, shared scoring, continuation, G/V predictions, artifact IO; excludes imports and final index publication. Per-method regions exclude shared parent/scoring. Nested validation is never added again.'
            study['resources']=dict(process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
                memory_accounting='process lifetime RSS high-water; MPS not used; not incremental or summed')
            measured = lambda ns, boundary: dict(nanoseconds=ns, status='measured', reason=None, boundary=boundary)
            costs = {'enclosing': measured(study['costs']['total_study_ns'], study['costs']['boundary'])}
            for key, slot in [('shared_scoring_ns', 'score'), ('tiny_gate_probe_ns', 'probe')]:
                if key in study['costs']:
                    costs[slot] = measured(study['costs'][key], 'Shared study region nested inside enclosing cost; do not add again')
            update_run_observations(paths, {'costs': costs, 'resources': {
                'rss_bytes': study['resources']['process_peak_rss_bytes'],
                'memory_accounting': study['resources']['memory_accounting'],
                'timing_exclusions': ['imports', 'final artifact-index and terminal manifest publication']}})
            # Hash every retained artifact, including failed checkpoints and full raw row streams.
            study['artifacts']={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()
                and p.name not in ('.ledger.lock','manifest.json','events.jsonl')}
            # Include child run manifests/events: only the enclosing study is still open.
            for p in sorted((root/'runs').rglob('*')) if (root/'runs').exists() else []:
                if p.is_file() and p.name in ('manifest.json','events.jsonl'):
                    study['artifacts'][str(p.relative_to(root))]=file_hash(p)
            write_json(root/'study.json',study)
            finish_run(paths,{'study':'study.json','study_sha256':file_hash(root/'study.json'),
                             'failures':study['failures']},study['status'])
        except BaseException as publication_error:
            publication_failure(publication_error, 'final study', primary_error)
            if primary_error is None:
                raise
        finally:
            train.BACKEND_REPORT_PATH=previous_receipt
            try:
                train.restore_rng(previous_rng)
            finally:
                torch.set_num_threads(previous_threads)
    return root
