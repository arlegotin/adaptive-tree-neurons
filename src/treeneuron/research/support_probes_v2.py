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
