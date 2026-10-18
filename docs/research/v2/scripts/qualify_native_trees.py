"""Bounded official-native F/G-only tree fixtures; never a quality benchmark."""
import sys
sys.dont_write_bytecode = True

import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
METHODS = ('catboost', 'xgboost', 'lightgbm_constant', 'lightgbm_linear')
THREADS = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTORCH_ENABLE_MPS_FALLBACK='0',
    PYTHONDONTWRITEBYTECODE='1')


def plain(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'Unexpected JSON value: {type(value).__name__}')


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=plain, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_record(array):
    return dict(shape=list(array.shape), dtype=array.dtype.str,
        sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def import_auditor():
    path = ROOT / 'locks/research-v2/tree-imports.py'
    spec = importlib.util.spec_from_file_location('native_tree_import_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resource_guard(directory):
    disk = shutil.disk_usage(directory)
    vm = subprocess.run(['/usr/bin/vm_stat'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    import re
    page_size = int(re.search(r'page size of (\d+) bytes', vm.stdout)[1])
    free = int(re.search(r'Pages free:\s+(\d+)', vm.stdout)[1])
    inactive = int(re.search(r'Pages inactive:\s+(\d+)', vm.stdout)[1])
    reclaimable_proxy = (free + inactive) * page_size
    return dict(free_disk_bytes=disk.free, free_plus_inactive_ram_proxy_bytes=reclaimable_proxy,
        ram_proxy_limit='vm_stat free+inactive is an admission proxy, not guaranteed available RAM',
        minimum_disk_bytes=256*1024**2, minimum_ram_proxy_bytes=512*1024**2,
        admitted=disk.free >= 256*1024**2 and reclaimable_proxy >= 512*1024**2,
        vm_stat_output=vm.stdout, utc_ns=time.time_ns())


def seed64(root, *labels):
    raw = json.dumps([root, *labels], separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') & ((1 << 63) - 1)


def make_fixture():
    import numpy as np
    rng = np.random.default_rng(seed64(2026101802, 'WP07-tree-fixture', 'synthetic-v1', 'observations'))
    result = {}
    for role, count in [('F', 192), ('G', 64)]:
        x = rng.uniform(-1, 1, size=(count, 3))
        category = np.resize(np.asarray(['a', 'b', 'c'], dtype='<U6'), count)
        rng.shuffle(category)
        if role == 'G':
            category[::7] = 'unseen'
        effects = {'a': -1., 'b': .5, 'c': 1.75, 'unseen': .2}
        clean = (2 + 1.5*x[:, 0] - .75*x[:, 1] + .4*x[:, 2]
            + 1.25*np.maximum(x[:, 0] - .25, 0) + .65*x[:, 0]*x[:, 1]
            + np.asarray([effects[c] for c in category]))
        noise = rng.normal(0, .1, count)
        result.update({f'X_{role}': x, f'category_{role}': category,
            f'clean_{role}': clean, f'noise_{role}': noise, f'y_{role}': clean + noise})
    result['y_G_reverse'] = 4 - result['clean_G'] + result['noise_G']
    return result


def encode_fixture(method, fixture, categorical):
    import numpy as np
    import pandas as pd
    means = fixture['X_F'].mean(axis=0) if method == 'lightgbm_linear' else np.zeros(3)
    scales = fixture['X_F'].std(axis=0, ddof=0) if method == 'lightgbm_linear' else np.ones(3)
    if np.any(scales <= 0) or not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise FloatingPointError('Invalid F-only scaling statistics')
    vocabulary = sorted(set(fixture['category_F'].tolist())) if categorical else []
    result = dict(vocabulary=vocabulary, transform=dict(mean=means.tolist(), scale=scales.tolist(),
        fit_role='F_only', scaled_continuous=method == 'lightgbm_linear'), unknown_G_count=0)
    for role in ('F', 'G'):
        frame = pd.DataFrame((fixture[f'X_{role}'] - means)/scales, columns=['x1', 'x2', 'x3'])
        if categorical:
            categories = fixture[f'category_{role}']
            unknown = ~np.isin(categories, vocabulary)
            if method == 'catboost':
                frame['category'] = np.where(unknown, '__ATN_UNKNOWN__', categories).astype(str)
            else:
                frame['category'] = pd.Categorical(np.where(unknown, None, categories),
                    categories=vocabulary, ordered=False)
            if role == 'G':
                result['unknown_G_count'] = int(unknown.sum())
        result[role] = frame
    return result


def self_check():
    import numpy as np
    data = make_fixture()
    assert data['X_F'].shape == (192, 3) and data['X_G'].shape == (64, 3)
    assert set(data['category_F']) == {'a', 'b', 'c'}
    assert np.count_nonzero(data['category_G'] == 'unseen') == 10
    assert np.array_equal(data['y_G_reverse'], 4 - data['clean_G'] + data['noise_G'])
    for method in METHODS:
        encoded = encode_fixture(method, data, True)
        assert encoded['vocabulary'] == ['a', 'b', 'c']
        assert encoded['unknown_G_count'] == 10
        assert list(encoded['F'].columns) == ['x1', 'x2', 'x3', 'category']
        assert len(encoded['F']) == 192 and len(encoded['G']) == 64
        if method == 'catboost':
            assert set(encoded['F']['category']) == {'a', 'b', 'c'}
            assert np.count_nonzero(encoded['G']['category'] == '__ATN_UNKNOWN__') == 10
        else:
            assert np.count_nonzero(encoded['G']['category'].cat.codes == -1) == 10
        if method == 'lightgbm_linear':
            assert np.max(np.abs(encoded['F'].iloc[:, :3].mean().to_numpy())) < 1e-14
            assert np.max(np.abs(encoded['F'].iloc[:, :3].std(ddof=0).to_numpy() - 1)) < 1e-14
            changed = dict(data, X_G=data['X_G'] + 400, category_G=np.full(64, 'newer'))
            again = encode_fixture(method, changed, True)
            assert encoded['F'].equals(again['F'])
            assert encoded['transform'] == again['transform']
            assert again['unknown_G_count'] == 64
    print('F/G fixture and F-only native encoding self-checks passed; no fit executed.')


def parameters(method, seed):
    if method == 'catboost':
        return dict(iterations=128, depth=4, learning_rate=.1, l2_leaf_reg=3.,
            random_strength=1., bootstrap_type='MVS', subsample=.8,
            loss_function='RMSE', eval_metric='RMSE', task_type='CPU', thread_count=1,
            random_seed=seed, counter_calc_method='SkipTest', use_best_model=True)
    if method == 'xgboost':
        return dict(objective='reg:squarederror', eval_metric='rmse', tree_method='hist',
            device='cpu', nthread=1, seed=seed, max_depth=3, eta=.1,
            min_child_weight=1., subsample=1., colsample_bytree=1.,
            reg_lambda=1., reg_alpha=0., max_bin=64, validate_parameters=True)
    return dict(objective='regression', metric='rmse', device_type='cpu',
        tree_learner='serial', num_threads=1, seed=seed, learning_rate=.1,
        num_leaves=7, min_data_in_leaf=8, lambda_l2=.01, linear_lambda=.01,
        linear_tree=method == 'lightgbm_linear', max_bin=31, deterministic=True,
        force_col_wise=True, bagging_fraction=1., bagging_freq=1, verbosity=-1)


def leaf_records(tree):
    if 'left_child' in tree:
        return leaf_records(tree['left_child']) + leaf_records(tree['right_child'])
    return [tree]


def fit_one(method, data, encoded, label_mode, config, destination):
    import numpy as np
    destination.mkdir()
    F, G = encoded['F'], encoded['G']
    yF = data['y_F']
    yG = data['y_G'] if label_mode == 'ordinary' else data['y_G_reverse']
    fit_started = time.perf_counter_ns()
    saved_config = dict(config)
    extra = {}
    if method == 'catboost':
        from catboost import CatBoostRegressor, Pool
        indices = [3] if encoded['vocabulary'] else []
        dF, dG = Pool(F, yF, cat_features=indices), Pool(G, yG, cat_features=indices)
        assert dF.num_row() == 192 and dG.num_row() == 64
        assert dF.num_col() == F.shape[1] and dG.num_col() == G.shape[1]
        np.testing.assert_allclose(dF.get_label(), yF, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(dG.get_label(), yG, rtol=1e-6, atol=1e-7)
        assert dF.get_cat_feature_indices() == indices
        saved_config.update(train_dir=str(destination/'native-training'), allow_writing_files=True)
        model = CatBoostRegressor(**saved_config)
        model.fit(dF, eval_set=dG, early_stopping_rounds=8, verbose=False)
        history = model.get_evals_result()['validation']['RMSE']
        best_zero = int(model.get_best_iteration())
        retained = int(model.tree_count_)
        assert retained == best_zero + 1
        prediction = np.asarray(model.predict(dG), dtype=np.float64)
        fit_F_prediction = np.asarray(model.predict(dF), dtype=np.float64)
        path = destination/'model.cbm'
        model.save_model(path)
        loaded = CatBoostRegressor()
        loaded.load_model(path)
        restored = np.asarray(loaded.predict(dG), dtype=np.float64)
        extra['resolved_parameters'] = model.get_all_params()
        extra['native_row_counts'] = dict(F=dF.num_row(), G=dG.num_row())
        extra['category_indices'] = indices
        extra['loaded_category_indices'] = loaded.get_cat_feature_indices()
        assert extra['loaded_category_indices'] == indices
    elif method == 'xgboost':
        import xgboost as xgb
        dF = xgb.QuantileDMatrix(F, yF, max_bin=64, nthread=1, enable_categorical=True)
        dG = xgb.QuantileDMatrix(G, yG, max_bin=64, nthread=1, enable_categorical=True, ref=dF)
        assert dF.num_row() == 192 and dG.num_row() == 64
        assert dF.num_col() == F.shape[1] and dG.num_col() == G.shape[1]
        np.testing.assert_array_equal(dF.get_label(), yF.astype(np.float32))
        np.testing.assert_array_equal(dG.get_label(), yG.astype(np.float32))
        evaluation = {}
        callback = xgb.callback.EarlyStopping(rounds=8, data_name='G', metric_name='rmse',
            maximize=False, save_best=True, min_delta=0.)
        model = xgb.train(saved_config, dF, num_boost_round=128, evals=[(dG, 'G')],
            callbacks=[callback], evals_result=evaluation, verbose_eval=False)
        history = evaluation['G']['rmse']
        best_zero = int(model.best_iteration)
        retained = model.num_boosted_rounds()
        assert retained == best_zero + 1
        prediction = np.asarray(model.predict(dG), dtype=np.float64)
        fit_F_prediction = np.asarray(model.predict(dF), dtype=np.float64)
        path = destination/'model.ubj'
        model.save_model(path)
        loaded = xgb.Booster(model_file=path)
        restored = np.asarray(loaded.predict(dG), dtype=np.float64)
        extra.update(native_row_counts=dict(F=dF.num_row(), G=dG.num_row()),
            F_quantizer_referenced_by_G=True, feature_types=model.feature_types,
            loaded_feature_types=loaded.feature_types, resolved_configuration=json.loads(model.save_config()))
        assert loaded.feature_types == model.feature_types
        if encoded['vocabulary']:
            assert model.feature_types[-1] == 'c'
    else:
        import lightgbm as lgb
        categories = ['category'] if encoded['vocabulary'] else []
        # Dataset construction receives linear_tree before a handle exists.
        dataset_params = {key: saved_config[key] for key in
            ('linear_tree', 'max_bin', 'min_data_in_leaf', 'num_threads', 'verbosity')}
        dF = lgb.Dataset(F, label=yF, categorical_feature=categories,
            params=dataset_params, free_raw_data=False)
        dF.construct()
        dG = lgb.Dataset(G, label=yG, reference=dF, categorical_feature=categories,
            params=dataset_params, free_raw_data=False)
        dG.construct()
        assert bool(dF.get_params()['linear_tree']) == (method == 'lightgbm_linear')
        assert dG.reference is dF
        assert dF.num_data() == 192 and dG.num_data() == 64
        assert dF.num_feature() == F.shape[1] and dG.num_feature() == G.shape[1]
        np.testing.assert_array_equal(dF.get_label(), yF.astype(np.float32))
        np.testing.assert_array_equal(dG.get_label(), yG.astype(np.float32))
        evaluation = {}
        model = lgb.train(saved_config, dF, num_boost_round=128,
            valid_sets=[dG], valid_names=['G'], callbacks=[
                lgb.record_evaluation(evaluation),
                lgb.early_stopping(8, first_metric_only=True, verbose=False, min_delta=0.)])
        history = evaluation['G']['rmse']
        best_zero = int(model.best_iteration) - 1
        retained = int(model.best_iteration)
        assert 1 <= retained <= len(history)
        prediction = np.asarray(model.predict(G, num_iteration=retained, num_threads=1), dtype=np.float64)
        fit_F_prediction = np.asarray(model.predict(F, num_iteration=retained, num_threads=1), dtype=np.float64)
        path = destination/'model.txt'
        model.save_model(path, num_iteration=retained)
        loaded = lgb.Booster(model_file=path)
        restored = np.asarray(loaded.predict(G, num_iteration=retained, num_threads=1), dtype=np.float64)
        dump = model.dump_model(num_iteration=retained)
        write(destination/'model-dump.json', dump)
        leaves = [leaf_records(info['tree_structure']) for info in dump['tree_info']]
        linear = [leaf for group in leaves for leaf in group if len(leaf.get('leaf_coeff', []))]
        nonzero = [leaf for leaf in linear if np.linalg.norm(leaf['leaf_coeff']) > 1e-12]
        if method == 'lightgbm_linear' and label_mode == 'ordinary':
            assert len(leaves) > 1 and nonzero, 'Native linear coefficients were not exercised'
            assert all(not leaf.get('leaf_coeff') or not np.any(leaf['leaf_coeff']) for leaf in leaves[0])
            assert all(3 not in leaf['leaf_features'] for leaf in linear), 'Category entered leaf regression'
        if method == 'lightgbm_constant':
            assert not linear
        extra.update(native_row_counts=dict(F=dF.num_data(), G=dG.num_data()),
            F_dataset_parameters=dF.get_params(), G_references_F=True,
            linear_tree_at_construction=method == 'lightgbm_linear',
            linear_leaves=len(linear), nonzero_linear_leaves=len(nonzero),
            first_tree_linear_coefficients=[leaf.get('leaf_coeff', []) for leaf in leaves[0]],
            pandas_categorical=dump.get('pandas_categorical'),
            loaded_pandas_categorical=loaded.pandas_categorical)
        assert loaded.pandas_categorical == dump.get('pandas_categorical')
    fit_and_save_ns = time.perf_counter_ns() - fit_started
    assert prediction.shape == (64,) and fit_F_prediction.shape == (192,)
    if not np.isfinite(prediction).all() or not np.isfinite(fit_F_prediction).all():
        raise FloatingPointError('Nonfinite native predictions')
    assert np.array_equal(prediction, restored), 'Save/load predictions differ'
    assert 0 <= best_zero < len(history) <= 128
    assert abs(history[best_zero] - min(history)) <= 1e-12 * max(1., abs(min(history)))
    applied_G_rmse = float(np.sqrt(np.mean((prediction-yG)**2)))
    difference = abs(applied_G_rmse-history[best_zero])
    tolerance = 1e-6 * max(1., abs(history[best_zero]))
    assert difference <= tolerance, 'Applied G loss differs from recorded selected native loss'
    if label_mode == 'reverse_G_stopping':
        assert len(history) < 128 and retained < len(history), 'Patience restoration not exercised'
    np.savez_compressed(destination/'predictions.npz', F=fit_F_prediction, G=prediction,
        G_after_load=restored, F_row_id=np.arange(192), G_row_id=np.arange(192, 256))
    result = dict(label_mode=label_mode, config=saved_config, G_history=history,
        best_iteration_zero_based=best_zero, executed_rounds=len(history), retained_rounds=retained,
        native_patience=8, maximum_rounds=128, selected_original_unit_G_rmse=applied_G_rmse,
        applied_vs_recorded_G_absolute_error=difference, applied_vs_recorded_G_tolerance=tolerance,
        predictions_finite=True, save_load_predictions_exact=True,
        original_unit_predictions=True, rows_F=192, rows_G=64,
        fit_preprocess_predict_save_load_ns=fit_and_save_ns,
        model_bytes=path.stat().st_size, model_sha256=file_hash(path),
        predictions_sha256=file_hash(destination/'predictions.npz'),
        quality_eligible=False, eligibility_scope='engineering_fixture_only',
        gradients='not_applicable_to_native_tree_boosting', **extra)
    write(destination/'result.json', result)
    return result


def audit_phase(auditor, package, destination, overlay, *, completed_fits):
    """Keep the import-only auditor output and publish actual process context."""
    raw = destination/'raw-auditor-call'
    raw.mkdir()
    auditor.worker(package, raw, overlay)
    value = json.loads((raw/'import.json').read_text())
    value.update(auditor_call_phase='after-fitting' if completed_fits else 'before-fitting',
        completed_native_fits=completed_fits, models_constructed=completed_fits > 0,
        fits_executed=completed_fits > 0, raw_auditor_record_sha256=file_hash(raw/'import.json'),
        raw_auditor_flags_scope='The reusable auditor itself constructs/fits no model; process flags here include preceding fixture work')
    write(destination/'import.json', value)


def checked_worker_manifest(study):
    manifest = json.loads((study/'manifest.json').read_text())
    identities = dict(script_sha256=Path(__file__), executable_sha256=Path(sys.executable),
        import_auditor_sha256=ROOT/'locks/research-v2/tree-imports.py',
        tree_lock_sha256=ROOT/'locks/research-v2/tree-requirements.txt',
        core_lock_sha256=ROOT/'uv.lock', fixture_sha256=study/'fixture.npz')
    for field, path in identities.items():
        if file_hash(path) != manifest[field]:
            raise ValueError(f'Fixture input identity changed after manifest creation: {field}')
    if sys.executable != manifest['executable_path']:
        raise ValueError('Fixture interpreter path differs from manifest')
    return manifest


def worker(study, case_id, destination, overlay):
    # Check the dynamically imported auditor and every declared executable/lock
    # binding before importing it, reading the NumPy archive or fitting anything.
    manifest = checked_worker_manifest(study)
    case = manifest['cases'][case_id]
    method = case['method']
    package = method.split('_')[0]
    auditor = import_auditor()
    pre = destination/'before-fitting'
    pre.mkdir()
    audit_phase(auditor, package, pre, overlay, completed_fits=0)
    import numpy as np
    with np.load(study/'fixture.npz', allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    for array in data.values():
        array.setflags(write=False)
    begin = time.perf_counter_ns()
    encoded = encode_fixture(method, data, case['categorical'])
    preprocessing_ns = time.perf_counter_ns() - begin
    write(destination/'encoding.json', {key: value for key, value in encoded.items() if key not in ('F', 'G')})
    results = [fit_one(method, data, encoded, label_mode, case['parameters'], destination/label_mode)
        for label_mode in ('ordinary', 'reverse_G_stopping')]
    post = destination/'after-fitting'
    post.mkdir()
    audit_phase(auditor, package, post, overlay, completed_fits=2)
    runtime = json.loads((post/'import.json').read_text())
    assert not runtime['imported_torch_after']
    result = dict(schema='research/v2-native-tree-fixture', case=case,
        status='completed_fixture_only', preprocessing_ns=preprocessing_ns,
        fits=results, runtime_record_sha256=file_hash(post/'import.json'),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        loaded_openmp_paths=runtime['distinct_loaded_openmp_paths'],
        dual_runtime_disposition='Actual paths retained; bounded fitting observed here does not prove general compatibility',
        models_fitted=2, quality_eligible=False, native_full_anchor_qualified=False,
        accessed_roles=['synthetic_F', 'synthetic_G'], prepared_data_accessed=False,
        completed_utc_ns=time.time_ns())
    write(destination/'result.json', result)


def run_child(argv, env, destination, case, attempt, manifest):
    """Always reap the child and retain cost/error evidence before propagating."""
    start = time.perf_counter_ns()
    started_utc_ns = time.time_ns()
    process = None
    peak_rss = 0
    stopped_for = None
    original_error = original_tb = None
    supervisor_error = None
    cleanup = []
    log_path = destination/'process.log'
    try:
        with log_path.open('x') as log:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                sampled = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                if sampled.returncode == 0 and sampled.stdout.strip():
                    peak_rss = max(peak_rss, int(sampled.stdout.strip()) * 1024)
                if peak_rss > manifest['maximum_child_rss_bytes']:
                    stopped_for = 'resource_exhausted'
                if (time.perf_counter_ns()-start)/1e9 > manifest['maximum_case_wall_seconds']:
                    stopped_for = 'timeout'
                if shutil.disk_usage(destination).free < 128*1024**2:
                    stopped_for = 'resource_exhausted'
                if stopped_for:
                    break
                time.sleep(.25)
    except BaseException as error:
        original_error, original_tb = error, error.__traceback__
        supervisor_error = dict(exception=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), utc_ns=time.time_ns())
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    try:
                        process.terminate()
                        cleanup.append('terminate_sent')
                    except ProcessLookupError:
                        cleanup.append('child_exited_before_terminate')
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                        cleanup.append('kill_sent_after_terminate_timeout')
                    except ProcessLookupError:
                        cleanup.append('child_exited_before_kill')
                    process.wait(timeout=5)
                cleanup.append('wait_completed')
            except BaseException as error:
                cleanup.append(dict(exception=type(error).__name__, message=str(error)))
                if original_error is None:
                    original_error, original_tb = error, error.__traceback__
                    supervisor_error = dict(exception=type(error).__name__, message=str(error),
                        traceback=traceback.format_exc(), utc_ns=time.time_ns())
                else:
                    original_error.add_note(f'Child cleanup also raised {type(error).__name__}: {error}')
                # A failed/interruptible terminate/wait must still attempt kill
                # and wait. Keep the first supervisor error as the raised error.
                try:
                    if process.poll() is None:
                        process.kill()
                        cleanup.append('kill_sent_during_cleanup_recovery')
                    process.wait(timeout=5)
                    cleanup.append('recovery_wait_completed')
                except BaseException as cleanup_error:
                    cleanup.append(dict(exception=type(cleanup_error).__name__, message=str(cleanup_error)))
                    original_error.add_note(f'Final child reaping failed: {cleanup_error!r}')
    error_file = destination/'error.json'
    if original_error is not None:
        status = ('interrupted_recoverable' if isinstance(original_error, KeyboardInterrupt)
            else 'input_or_dependency_failure' if isinstance(original_error, OSError)
            else 'fixture_supervisor_failure')
    elif stopped_for:
        status = stopped_for
    elif process.returncode == 0:
        status = 'completed_fixture_only'
    elif error_file.exists():
        status = json.loads(error_file.read_text())['status']
    else:
        status = 'interrupted_recoverable'
    terminal = dict(status=status, case=case, attempt=attempt,
        exit_code=None if process is None else process.returncode,
        pid=None if process is None else process.pid, child_created=process is not None,
        child_reaped=process is None or process.returncode is not None, cleanup_actions=cleanup,
        argv=argv, cwd=str(ROOT), environment_overrides={**THREADS,
            **{key: env[key] for key in ('PYTHONPATH', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'MPLBACKEND')},
            'KMP_DUPLICATE_LIB_OK': None}, started_utc_ns=started_utc_ns, ended_utc_ns=time.time_ns(),
        complete_process_wall_ns=time.perf_counter_ns()-start,
        sampled_peak_rss_bytes=peak_rss, stopped_for=stopped_for,
        log_sha256=file_hash(log_path) if log_path.exists() else None,
        supervisor_error=supervisor_error)
    if (destination/'result.json').exists():
        terminal['result_sha256'] = file_hash(destination/'result.json')
    try:
        if supervisor_error is not None:
            write(destination/'supervisor-error.json', supervisor_error)
        write(destination/'command.json', terminal)
    except BaseException as receipt_error:
        if original_error is None:
            raise
        original_error.add_note(f'Terminal receipt publication also failed: {receipt_error!r}')
        raise original_error.with_traceback(original_tb)
    if original_error is not None:
        raise original_error.with_traceback(original_tb)
    return terminal


def run(output, overlay):
    import numpy as np
    output.mkdir(parents=True, exist_ok=False)
    auditor = import_auditor()
    environment_before = auditor.validate_environment(overlay)
    write(output/'environment-before.json', environment_before)
    # The worker archive contains only declared F/G observations/labels. Keep
    # generation-only noiseless targets and sampled noise out of fit processes.
    data = {key: value for key, value in make_fixture().items()
        if not key.startswith(('clean_', 'noise_'))}
    np.savez_compressed(output/'fixture.npz', **data)
    seed = seed64(2026101802, 'WP07-tree-fixture', 'synthetic-v1', 'native-training') & ((1 << 31) - 1)
    cases = [dict(case_id=i, method=method, categorical=categorical, seed=seed,
        parameters=parameters(method, seed)) for i, (method, categorical) in enumerate(
            (m, c) for m in METHODS for c in (False, True))]
    manifest = dict(schema='research/v2-native-tree-fixture-plan',
        source_root=str(ROOT), script_sha256=file_hash(__file__),
        executable_path=sys.executable, executable_sha256=file_hash(sys.executable),
        import_auditor_sha256=file_hash(ROOT/'locks/research-v2/tree-imports.py'),
        tree_lock_sha256=file_hash(ROOT/'locks/research-v2/tree-requirements.txt'),
        core_lock_sha256=file_hash(ROOT/'uv.lock'), fixture_sha256=file_hash(output/'fixture.npz'),
        arrays={key: array_record(value) for key, value in data.items()},
        generation=dict(root=2026101802, namespace=['WP07-tree-fixture', 'synthetic-v1'],
            observation_seed=seed64(2026101802, 'WP07-tree-fixture', 'synthetic-v1', 'observations'),
            F=192, G=64, numeric_features=3, input_distribution='independent Uniform[-1,1]',
            category_F='balanced a/b/c, shuffled before labels', category_G='balanced/shuffled a/b/c, every seventh row unseen',
            target='2+1.5*x1-.75*x2+.4*x3+1.25*relu(x1-.25)+.65*x1*x2+category_effect+N(0,.1^2)',
            category_effect=dict(a=-1., b=.5, c=1.75, unseen=.2),
            numeric_variant='same labels, categorical predictor omitted',
            stopping_stress='G target=4-clean_G+same_noise_G; intentional opposing signal, engineering only'),
        cases=cases, planned_fits=16, maximum_rounds_per_fit=128, patience=8,
        maximum_case_wall_seconds=90, maximum_child_rss_bytes=2*1024**3,
        concurrency=1, retry='one identical seed/config infrastructure retry only; no numerical/assertion retries',
        quality_eligible=False, native_anchor_eligible=False,
        confirmation_or_transfer_data=False, created_utc_ns=time.time_ns())
    write(output/'manifest.json', manifest)
    shutil.copyfile(__file__, output/'source.py')
    statuses = []
    for case in cases:
        case_root = output/f"case-{case['case_id']}-{case['method']}-{'categorical' if case['categorical'] else 'numeric'}"
        case_root.mkdir()
        for attempt in range(2):
            guard = resource_guard(output)
            destination = case_root/f'attempt-{attempt}'
            destination.mkdir()
            write(destination/'admission.json', guard)
            if not guard['admitted']:
                terminal = dict(status='resource_exhausted', exit_code=None,
                    reason='Pre-fit disk/RAM admission proxy failed', case=case, attempt=attempt)
                write(destination/'command.json', terminal)
            else:
                env = dict(os.environ)
                env.pop('KMP_DUPLICATE_LIB_OK', None)
                env.update(THREADS)
                env.update(PYTHONPATH=str(overlay), MPLCONFIGDIR=str(destination/'mpl-config'),
                    XDG_CACHE_HOME=str(destination/'cache'), MPLBACKEND='Agg')
                argv = [sys.executable, str(Path(__file__).resolve()), '--worker',
                    '--study', str(output), '--case-id', str(case['case_id']),
                    '--output', str(destination), '--overlay', str(overlay)]
                terminal = run_child(argv, env, destination, case, attempt, manifest)
            statuses.append(terminal)
            if terminal['status'] not in ('resource_exhausted', 'input_or_dependency_failure', 'interrupted_recoverable'):
                break
            if attempt == 0:
                time.sleep(1)
    environment_after = auditor.validate_environment(overlay)
    write(output/'environment-after.json', environment_after)
    assert environment_before == environment_after, 'Installed/core identity changed during fixtures'
    terminal_by_case = {x['case']['case_id']: x for x in statuses}
    failures = [x['case']['case_id'] for x in terminal_by_case.values() if x['status'] != 'completed_fixture_only']
    suite = dict(schema='research/v2-native-tree-fixture-suite',
        status='bounded_native_fixtures_passed' if not failures else 'fixture_qualification_incomplete',
        planned_cases=8, planned_fits=16, attempts=statuses, failed_cases=failures,
        same_seeds_and_configs_on_retry=True, installed_core_and_overlay_unchanged=True,
        quality_eligible=False, native_full_anchor_qualified=False, Gate_D='pending',
        accessed_roles=['synthetic_F', 'synthetic_G'], prepared_data_accessed=False,
        completed_utc_ns=time.time_ns())
    write(output/'suite.json', suite)
    print(json.dumps(dict(status=suite['status'], failed_cases=failures,
        cases=8, planned_fits=16, attempts=len(statuses), output=str(output))))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--study', type=Path)
    parser.add_argument('--case-id', type=int)
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.worker:
        try:
            worker(args.study.resolve(), args.case_id, args.output.resolve(), args.overlay.resolve())
        except Exception as error:
            status = ('numerical_failure' if isinstance(error, FloatingPointError) else
                'input_or_dependency_failure' if isinstance(error, (ImportError, OSError)) else
                'fixture_assertion_or_api_failure')
            write(args.output/'error.json', dict(status=status, exception=type(error).__name__,
                message=str(error), traceback=traceback.format_exc(), utc_ns=time.time_ns()))
            traceback.print_exc()
            raise SystemExit(1)
    elif args.output and args.overlay:
        run(args.output.resolve(), args.overlay.resolve())
    else:
        parser.error('Use --self-check or --output DIR --overlay DIR')
