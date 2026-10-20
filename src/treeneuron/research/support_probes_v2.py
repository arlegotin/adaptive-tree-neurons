"""Full-shape short engineering exercises; no native quality qualification.

These cases exercise declared large architectures on the original complete F/G
archives. They complement full native anchors, never replace their training or
prove that every hyperparameter combination will fit a resource envelope.
"""
import copy
import math
import os
from pathlib import Path
import signal
import sys
import time

from treeneuron.config import canonical_hash
from .immutable_inputs import stage_immutable_file
from .ledger import write_new_json
from .method_worker import (_ROOT, _THREADS, _digest, _read_json, _sha,
    _validate_request, scientific_source_sha256)
from .process_supervision_v2 import supervise_process
from .qualify import qualification_locks_sha256
from .search_supports_v2 import _config, _parse, family_search_spec
from .transfer import TransferStore
from .trial_bridge_v2 import EngineeringTrialBridge
from .workload import _checked_manifest
from .workqueue import _inventory


def support_probe_cases(family, feature_schema):
    """Finite geometry catalog, independent of labels, seeds and validation rank."""
    spec = family_search_spec(family, feature_schema)
    schema = spec['feature_schema']; configs = []
    def add(parameters, variant=None):
        configs.append(_config(family, schema, parameters, mode='search', variant=variant))
    if family in ('P00', 'P01', 'P02', 'P05'):
        variants = ('lower', 'upper') if family == 'P02' else (
            'centered_partial', 'centered_full', 'direct_partial', 'direct_full') if family == 'P05' else (None,)
        for variant in variants:
            for batch in (128, 256):
                params = dict(width=128, batch_size=batch, scheduler='cosine')
                params.update(dropout=.2) if family == 'P05' else params.update(growth_fraction=.5)
                add(params, variant)
    elif family == 'P03':
        for activation in ('ReLU', 'GELU'):
            for batch in (128, 256):
                add(dict(hidden_dims=[512]*3, activation=activation, dropout=.2, batch_size=batch))
    elif family == 'P04':
        for batch in (128, 256):
            add(dict(width=512, blocks=4, hidden_dropout=.2, residual_dropout=.2, batch_size=batch))
    elif family == 'P06':
        for embedding in ('none', 'pbld', 'pl', 'plr'):
            for hidden in ([256]*3, [64]*5, [512]):
                add(dict(num_emb_type=embedding, hidden_sizes=hidden))
    elif family == 'P07':
        add(dict(n_blocks=4, d_block=1024), 'plain')
        add(dict(n_blocks=4, d_block=1024, n_bins=128, d_embedding=32), 'piecewise')
    elif family == 'P08': add(dict(depth=10))
    elif family == 'P09': add(dict(max_depth=9, min_child_weight=1.))
    elif family == 'P10': add(dict(num_leaves=127, min_data_in_leaf=5))
    elif family == 'P11': add(dict(num_leaves=31, min_data_in_leaf=20))
    elif family == 'P12': add(dict(trees=2048, depth=6, lr_multiplier=1))
    else:
        finite = spec['finite_configurations']
        configs = [next(c for c in finite if c['estimator'] == 'spline'
            and c['parameters'] == dict(alpha=1e-6, n_knots=16)),
            next(c for c in finite if c['estimator'] == 'ridge' and c['parameters']['alpha'] == 1e-6)]
    return [dict(case_id=f'{family}-geometry-{index:02}', native_config=cfg,
        support_sha256=spec['support_hash'], quality_eligible=False, native_workload_qualified=False)
        for index, cfg in enumerate(configs)]


def engineering_probe_configuration(native_config, *, fit_rows):
    if type(native_config) is not dict or type(fit_rows) is not int or not 2 <= fit_rows <= 20000:
        raise ValueError('Exact native configuration and complete F count required')
    cfg = copy.deepcopy(native_config); family = cfg.get('family')
    if family != 'P13' and cfg.get('mode') not in ('search', 'native_anchor'):
        raise ValueError('Only declared native configurations can supply probe geometry')
    if _parse(family, cfg) != cfg: raise ValueError('Native configuration is not canonical')
    budget = dict(native_stopping=True)
    if family == 'P12':
        # Retain the actual2048-tree geometry and effective512-row batch. One
        # complete sampler epoch plus one update, not a native stopping result.
        budget = dict(max_updates=math.ceil(fit_rows/512)+1)
    elif family != 'P13':
        cfg['mode'] = 'engineering_fixture'
        if family in ('P00', 'P01', 'P02', 'P05'):
            cfg.update(base_updates=2, continuation_updates=2, validation_interval=1, patience=2)
        elif family in ('P03', 'P04'): cfg.update(max_epochs=1, patience=1)
        elif family in ('P06', 'P07'): cfg.update(max_epochs=1)
        else: cfg.update(max_rounds=2, patience=1)
        cfg = _parse(family, cfg)
    return dict(configuration=cfg, budget=budget, full_native_training=False,
        quality_eligible=False, native_workload_qualified=False,
        scope='Full declared model geometry and original F/G inputs; short engineering optimization only. Complete native training, all-combination resource admission and recovery qualification are separate.')


def plan_support_probes(*, prepared_root, expected_manifest_sha256, family):
    root = Path(prepared_root)
    if (not root.is_absolute() or root.is_symlink() or root.resolve() != root
            or not _digest(expected_manifest_sha256) or (root/'manifest.json').is_symlink()
            or _sha(root/'manifest.json') != expected_manifest_sha256):
        raise ValueError('Exact externally committed primary prepared manifest required')
    manifest = _read_json(root/'manifest.json'); _checked_manifest(manifest['dataset'], manifest)
    fold = 4 if manifest['dataset'] == 'bike' else 0
    split = TransferStore(root)._split(fold)
    schema = dict(columns=manifest['feature_names'], categorical=manifest['categorical_features'])
    result = dict(schema='research/v2-support-probe-plan', family_id=family,
        prepared_root=str(root), manifest_sha256=expected_manifest_sha256,
        dataset=manifest['dataset'], outer_fold=fold, split_sha256=split['split_hash'],
        source_sha256=scientific_source_sha256(), locks_sha256=qualification_locks_sha256(),
        fit_seed=17, cases=support_probe_cases(family, schema),
        quality_eligible=False, native_workload_qualified=False, V_labels_read=False, T_accessed=False)
    result['plan_sha256'] = canonical_hash(result)
    return result


def run_support_probe(*, plan, case_id, output, interpreter, extra_pythonpath=(), rss_limit_bytes=4*1024**3):
    """One paid immutable short probe; no retries, selection or V label access."""
    started = time.perf_counter_ns(); output = Path(output); interpreter = Path(interpreter)
    if type(plan) is not dict: raise ValueError('Immutable support probe plan required')
    expected = plan_support_probes(prepared_root=plan.get('prepared_root'),
        expected_manifest_sha256=plan.get('manifest_sha256'), family=plan.get('family_id'))
    if plan != expected: raise ValueError('Support probe source/locks/manifest/cases changed')
    matches = [c for c in plan['cases'] if c['case_id'] == case_id]
    root = Path(plan['prepared_root']); fold = plan['outer_fold']
    if (len(matches) != 1 or output.exists() or output.is_symlink()
            or output.resolve().is_relative_to(root) or root.is_relative_to(output.resolve())
            or any(output.resolve().is_relative_to(_ROOT/p) for p in ('src', 'tests', 'locks', 'configs', 'protocol'))
            or not interpreter.is_absolute() or not interpreter.is_file()
            or type(rss_limit_bytes) is not int or rss_limit_bytes <= 0):
        raise ValueError('One declared case, fresh external output, qualified interpreter and owned RSS guard required')
    overlays = []
    for entry in extra_pythonpath:
        path = Path(entry).resolve()
        if not path.is_dir() or not path.is_relative_to((_ROOT/'environment').resolve()):
            raise ValueError('Probe overlays must use the pinned project environment')
        overlays.append(str(path))
    store = TransferStore(root); manifest = store.manifest
    output.mkdir(parents=True, exist_ok=False); output = output.resolve()
    write_new_json(output/'plan.json', plan)
    result = dict(schema='research/v2-support-probe-result', status='input_or_dependency_failure',
        source_sha256=plan['source_sha256'], plan_sha256=plan['plan_sha256'], case_id=case_id,
        family_id=plan['family_id'], dataset=plan['dataset'], outer_fold=fold,
        snapshot_reload_exact=False, query_predictions_finite=False,
        full_native_training=False, native_workload_qualified=False, quality_eligible=False,
        V_labels_read=False, T_accessed=False,
        scope='Short engineering geometry exercise; no native quality, complete support qualification or production authority')
    try:
        store._access(fold, ['F_labels', 'G_labels', 'V_features'])
        inputs = output/'input'; inputs.mkdir(); files = {}; material = []
        for role in ('F', 'G', 'V_features'):
            relative = f'outer-{fold}/'+('V-features.npz' if role == 'V_features' else role+'.npz')
            original = root/relative; target = inputs/(role+'.npz')
            digest = manifest['files'][relative]['sha256']
            material.append(stage_immutable_file(original, target, expected_sha256=digest))
            files[role] = dict(path=target.name, sha256=digest)
        write_new_json(inputs/'materialization.json', dict(files=material))
        # Read the ordinary F archive through the validated worker input path;
        # metadata's role count is rechecked by the actual loaded F matrix.
        split = store._split(fold)
        fit_rows = split['actual_rows']['F']
        probe = engineering_probe_configuration(matches[0]['native_config'], fit_rows=fit_rows)
        request = dict(schema='research/v2-method-worker-request', purpose='engineering_qualification',
            job_id='support-probe/'+case_id, family_id=plan['family_id'], fit_seed=17,
            scientific_source_sha256=plan['source_sha256'], config=probe['configuration'], budget=probe['budget'],
            observation_identity=dict(dataset=plan['dataset'], split_kind='temporal' if plan['dataset']=='bike' else 'grouped',
                split_sha256=plan['split_sha256']), files=files)
        request_path = inputs/'request.json'; write_new_json(request_path, request); digest = _sha(request_path)
        is_node = plan['family_id'] == 'P12'
        _, arrays = _validate_request(request_path, digest, recoverable_training=is_node)
        if len(arrays['F']['X']) != fit_rows: raise ValueError('Actual complete F count differs from declared split')
        result.update(request_sha256=digest, probe=probe, fit_rows=fit_rows,
            select_rows=len(arrays['G']['X']), query_rows=len(arrays['V_features']['X']))
        del arrays
        argv = [str(interpreter), '-B', '-m', 'treeneuron.research.method_worker', '--request', str(request_path),
            '--expected-request-sha256', digest, '--output', str(output/'worker')]
        if is_node: argv += ['--training-checkpoints', str(output/'training-checkpoints')]
        environment = {k:'1' for k in (*_THREADS, 'PYTHONDONTWRITEBYTECODE')}
        environment.update(PYTHONPATH=os.pathsep.join([str(_ROOT/'src'), *overlays]), PYTORCH_ENABLE_MPS_FALLBACK='0',
            KMP_DUPLICATE_LIB_OK=None, MPLCONFIGDIR=str(output/'cache/mpl'), XDG_CACHE_HOME=str(output/'cache/xdg'))
        process = supervise_process(argv=argv, cwd=_ROOT, env_overrides=environment, output=output/'process',
            timeout_ns=None, rss_limit_bytes=rss_limit_bytes,
            identity=dict(plan_sha256=plan['plan_sha256'], case_id=case_id, request_sha256=digest, source_sha256=plan['source_sha256']))
        status = process['execution_status']
        if status == 'exited':
            if process['returncode'] in (-signal.SIGINT, -signal.SIGTERM, -signal.SIGKILL):
                status = 'interrupted_recoverable'
            else:
                terminal = EngineeringTrialBridge._worker_terminal(output/'worker', request, digest)
                status = terminal['status']
                if (process['returncode'] == 0) != (status == 'completed_ineligible'):
                    raise ValueError('Probe process exit contradicts worker terminal status')
                if status == 'completed_ineligible':
                    if terminal.get('snapshot_reload_exact') is not True: raise ValueError('Closed probe snapshot reload failed')
                    result.update(snapshot_reload_exact=True, query_predictions_finite=True,
                        worker_terminal_sha256=_sha(output/'worker/terminal.json'))
        result.update(status=status, child_reaped=process['child_reaped'], process_receipt_sha256=_sha(output/'process/process.json'),
            process_stop_reason=process['stop_reason'])
        _validate_request(request_path, digest, recoverable_training=is_node)
        if plan_support_probes(prepared_root=root, expected_manifest_sha256=plan['manifest_sha256'], family=plan['family_id']) != plan:
            raise ValueError('Probe original data/source/locks changed during execution')
        for role, ref in files.items():
            relative=f'outer-{fold}/'+('V-features.npz' if role=='V_features' else role+'.npz')
            if _sha(root/relative) != ref['sha256']: raise ValueError('Original probe role bytes changed')
        return result
    except BaseException as error:
        status = ('interrupted_recoverable' if isinstance(error, (KeyboardInterrupt, SystemExit))
            else 'resource_exhausted' if isinstance(error, MemoryError)
            else 'numerical_failure' if isinstance(error, ArithmeticError) else 'input_or_dependency_failure')
        result.update(status=status, snapshot_reload_exact=False, query_predictions_finite=False,
            error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        result['elapsed_ns'] = time.perf_counter_ns()-started
        result['clock_scope'] = 'Function entry through result preparation; caller pays final publication/return. Process time is nested.'
        write_new_json(output/'result.json', result)


def load_support_probe(reference, *, prepared_root, expected_manifest_sha256,
        expected_family, expected_case_id):
    """Check a closed short probe; no new fit, inference or quality promotion.

    Original managed training/reload receipts remain trusted. Complete input
    inclusion does not imply every row was visited during short optimization.
    """
    import numpy as np
    from .native_anchor_v2 import _closed_reference, _bound_json
    from .deployment_worker_v2 import _json

    reference = copy.deepcopy(reference)
    root, inventory = _closed_reference(reference)
    plan = plan_support_probes(prepared_root=prepared_root,
        expected_manifest_sha256=expected_manifest_sha256, family=expected_family)
    choices = [c for c in plan['cases'] if c['case_id'] == expected_case_id]
    if len(choices) != 1 or _bound_json(root, inventory, 'plan.json') != plan:
        raise ValueError('Closed geometry plan/source/case differs')
    result = _bound_json(root, inventory, 'result.json')
    expected_identity = dict(source_sha256=plan['source_sha256'], plan_sha256=plan['plan_sha256'],
        family_id=expected_family, case_id=expected_case_id, dataset=plan['dataset'], outer_fold=plan['outer_fold'])
    if (result.get('schema') != 'research/v2-support-probe-result'
            or result.get('status') != 'completed_ineligible'
            or any(result.get(k) != v for k, v in expected_identity.items())
            or any(result.get(k) is not True for k in ('snapshot_reload_exact', 'query_predictions_finite', 'child_reaped'))
            or any(result.get(k) is not False for k in ('full_native_training', 'native_workload_qualified',
                'quality_eligible', 'V_labels_read', 'T_accessed'))):
        raise ValueError('Only a passed, unqualified short-geometry record is admitted')
    request_path = root/'input/request.json'
    request_sha = result.get('request_sha256')
    if not _digest(request_sha) or _sha(request_path) != request_sha:
        raise ValueError('Original geometry request commitment differs')
    request = _bound_json(root, inventory, 'input/request.json')
    store = TransferStore(Path(prepared_root)); manifest = store.manifest; fold = plan['outer_fold']
    files = {}; originals = {}
    for role in ('F', 'G', 'V_features'):
        relative = f'outer-{fold}/'+('V-features.npz' if role == 'V_features' else role+'.npz')
        path = Path(prepared_root)/relative; digest = manifest['files'][relative]['sha256']
        if path.resolve() != path or path.is_symlink() or _sha(path) != digest:
            raise ValueError('Original primary geometry role bytes differ')
        originals[path] = digest
        files[role] = dict(path=role+'.npz', sha256=digest)
    # The reader is a new, permitted F/G/features access, never a V scorer.
    store._access(fold, ['F_labels', 'G_labels', 'V_features'])
    is_node = expected_family == 'P12'
    _, arrays = _validate_request(request_path, request_sha, recoverable_training=is_node)
    counts = {role:len(a['X']) for role, a in arrays.items()}
    probe = engineering_probe_configuration(choices[0]['native_config'], fit_rows=counts['F'])
    expected_request = dict(schema='research/v2-method-worker-request', purpose='engineering_qualification',
        job_id='support-probe/'+expected_case_id, family_id=expected_family, fit_seed=17,
        scientific_source_sha256=plan['source_sha256'], config=probe['configuration'], budget=probe['budget'],
        observation_identity=dict(dataset=plan['dataset'], split_kind='temporal' if plan['dataset']=='bike' else 'grouped',
            split_sha256=plan['split_sha256']), files=files)
    if (request != expected_request or result.get('probe') != probe
            or counts['F'] != result.get('fit_rows') or counts['G'] != result.get('select_rows')
            or counts['V_features'] != result.get('query_rows')
            or counts['F'] != store._split(fold)['actual_rows']['F']):
        raise ValueError('Geometry configuration, short budget or complete primary input counts differ')
    process = _bound_json(root, inventory, 'process/process.json')
    argv = process.get('argv', []); env = process.get('environment_overrides', {})
    paths = env.get('PYTHONPATH', '').split(os.pathsep)
    expected_argv = ['-B', '-m', 'treeneuron.research.method_worker', '--request', str(request_path),
        '--expected-request-sha256', request_sha, '--output', str(root/'worker')]
    if is_node: expected_argv += ['--training-checkpoints', str(root/'training-checkpoints')]
    if (process.get('schema') != 'research/v2-process-receipt' or process.get('execution_status') != 'exited'
            or process.get('returncode') != 0 or process.get('child_started') is not True
            or process.get('child_reaped') is not True or process.get('timeout_ns') is not None
            or any(process.get(k) is not False for k in ('eligible_model', 'protected_access_authorized',
                'native_recipe_runtime_qualified', 'whole_process_cost_authenticated', 'strict_peak_memory_limit_qualified'))
            or process.get('cwd') != str(_ROOT) or not argv
            or Path(argv[0]).resolve() != Path(sys.executable).resolve() or argv[1:] != expected_argv
            or process.get('identity') != dict(plan_sha256=plan['plan_sha256'], case_id=expected_case_id,
                request_sha256=request_sha, source_sha256=plan['source_sha256'])
            or any(env.get(k) != '1' for k in (*_THREADS, 'PYTHONDONTWRITEBYTECODE'))
            or env.get('KMP_DUPLICATE_LIB_OK') is not None or env.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0'
            or not paths or paths[0] != str(_ROOT/'src')
            or any(not Path(p).resolve().is_relative_to((_ROOT/'environment').resolve()) for p in paths[1:])
            or result.get('process_receipt_sha256') != _sha(root/'process/process.json')
            or result.get('process_stop_reason') != process.get('stop_reason')):
        raise ValueError('Owned geometry process/source/environment proof differs')
    started = _bound_json(root, inventory, 'process/started.json')
    if (started.get('schema') != 'research/v2-process-start'
            or any(process.get(k) != v for k, v in started.items() if k != 'schema')):
        raise ValueError('Process start and terminal records differ')
    refs = process.get('files', [])
    if len(refs) != 5 or {r.get('path') for r in refs} != {
            'started.json', 'spawned.json', 'stdout.log', 'stderr.log', 'resource-samples.jsonl'}:
        raise ValueError('Complete owned-process evidence inventory required')
    for ref in refs:
        path = root/'process'/ref['path']
        if _sha(path) != ref.get('sha256') or path.stat().st_size != ref.get('bytes'):
            raise ValueError('Process evidence commitment differs')
    spawned = _bound_json(root, inventory, 'process/spawned.json')
    if (spawned.get('schema') != 'research/v2-owned-process' or spawned.get('new_session') is not True
            or type(process.get('child_pid')) is not int or process['child_pid'] <= 0
            or spawned.get('child_pid') != process['child_pid'] or spawned.get('child_pgid') != process['child_pid']):
        raise ValueError('Owned geometry process group differs')
    samples = [_json(line) for line in (root/'process/resource-samples.jsonl').read_bytes().splitlines()]
    elapsed = process.get('elapsed_ns'); limit = process.get('rss_limit_bytes')
    if (not samples or type(elapsed) is not int or elapsed <= 0 or type(limit) is not int or limit <= 0
            or process.get('rss_sample_count') != len(samples)
            or any(type(s) is not dict or set(s) != {'elapsed_ns', 'root_rss_bytes'}
                or type(s['elapsed_ns']) is not int or not 0 <= s['elapsed_ns'] <= elapsed
                or type(s['root_rss_bytes']) is not int or not 0 <= s['root_rss_bytes'] <= limit for s in samples)
            or any(a['elapsed_ns'] >= b['elapsed_ns'] for a, b in zip(samples, samples[1:]))
            or max(s['root_rss_bytes'] for s in samples) <= 0
            or process.get('observed_peak_root_rss_bytes') != max(s['root_rss_bytes'] for s in samples)):
        raise ValueError('Sampled geometry RSS/process summary differs from raw observations')
    worker = EngineeringTrialBridge._worker_terminal(root/'worker', request, request_sha)
    native = request['budget'] == {'native_stopping':True}
    expected_budget = dict(requested=request['budget'], adapter={} if native and not is_node else request['budget'],
        policy='native_stopping' if native else 'explicit_training_recovery')
    snapshot_name = 'state.json' if expected_family in ('P08', 'P09', 'P10', 'P11') else 'manifest.json'
    snapshot = root/'worker/snapshot'/snapshot_name
    if (worker.get('status') != 'completed_ineligible' or worker.get('snapshot_reload_exact') is not True
            or worker.get('request_path') != str(request_path)
            or worker.get('budget_resolution') != expected_budget
            or result.get('worker_terminal_sha256') != _sha(root/'worker/terminal.json')
            or worker.get('snapshot_commitment') != dict(path='snapshot/'+snapshot_name, sha256=_sha(snapshot))
            or worker.get('snapshot_state_sha256' if snapshot_name == 'state.json' else 'snapshot_manifest_sha256') != _sha(snapshot)
            or worker.get('prediction_rows') != counts['V_features']):
        raise ValueError('Closed worker budget/reload/snapshot proof differs')
    query = arrays['V_features']; ids = ['row_id', 'group_id']+(['timestamps'] if 'timestamps' in query else [])
    with np.load(root/'worker/predictions.npz', allow_pickle=False) as saved:
        if (set(saved.files) != {'prediction', *ids} or len(saved.files) != len(ids)+1
                or any(saved[k].dtype != query[k].dtype or not np.array_equal(saved[k], query[k]) for k in ids)
                or saved['prediction'].dtype not in (np.dtype('float32'), np.dtype('float64'))
                or saved['prediction'].shape != query['row_id'].shape or not np.isfinite(saved['prediction']).all()):
            raise ValueError('Geometry query prediction rows/dtype/finiteness differ')
    if (plan_support_probes(prepared_root=prepared_root, expected_manifest_sha256=expected_manifest_sha256,
            family=expected_family) != plan or _inventory(root) != inventory
            or any(path.is_symlink() or path.resolve() != path or _sha(path) != digest for path, digest in originals.items())):
        raise ValueError('Geometry proof or original source/primary roles changed during reading')
    return dict(schema='research/v2-authenticated-support-geometry', status='authenticated_short_geometry_evidence',
        reference=copy.deepcopy(reference), **expected_identity, complete_input_rows=counts,
        native_configuration=copy.deepcopy(choices[0]['native_config']), engineering_configuration=probe['configuration'],
        budget=probe['budget'], query_predictions_finite=True, managed_snapshot_reload_record_checked=True,
        full_native_training=False, native_workload_qualified=False, quality_eligible=False,
        all_training_rows_visited_authenticated=False, independent_new_inference=False, HPO_journal_authenticated=False,
        protected_access_authorized=False, V_labels_read=False, T_accessed=False,
        scope='Current-source primary input, geometry request, managed process, closed worker/reload records and finite saved queries. Original fitting/reload is trusted; no full native training, row-visit, independent inference, complete workload/support registry or production/protected authority.')
