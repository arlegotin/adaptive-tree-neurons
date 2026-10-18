"""Opt-in tiny official parity in fresh method-specific, one-thread processes."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest


def fixture(categorical):
    rng = np.random.default_rng(1729)
    frames, targets = [], []
    for count in (192, 64):
        values = rng.uniform(-1, 1, (count, 3))
        categories = np.resize(np.array(['a', 'b', 'c'], dtype='<U8'), count)
        rng.shuffle(categories)
        if count == 64:
            categories[::7] = 'unknown'
        effects = {'a': -1., 'b': .5, 'c': 1.75, 'unknown': .2}
        y = (2+1.5*values[:, 0]-.75*values[:, 1]+.4*values[:, 2]
            + 1.25*np.maximum(values[:, 0]-.25, 0)+.65*values[:, 0]*values[:, 1]
            + np.array([effects[x] for x in categories])+rng.normal(0, .1, count))
        frame = pd.DataFrame(values, columns=['pressure [raw]', 'x2', 'x3'])
        if categorical:
            frame['kind'] = categories
        frames.append(frame)
        targets.append(y)
    return frames, targets


def direct(family, frames, targets, categorical):
    """Independent literal recipe/encoding, not the adapter's helper functions."""
    F, G = [x.copy(deep=True) for x in frames]
    if family == 'P11':
        mean, scale = F.iloc[:, :3].mean(), F.iloc[:, :3].std(ddof=0)
        F.iloc[:, :3] = (F.iloc[:, :3]-mean)/scale
        G.iloc[:, :3] = (G.iloc[:, :3]-mean)/scale
    F.columns = [f'f{i}' for i in range(F.shape[1])]
    G.columns = F.columns
    if categorical:
        vocabulary = ['str:a', 'str:b', 'str:c']
        for frame in (F, G):
            tokens = ['str:'+x for x in frame['f3']]
            if family == 'P08':
                frame['f3'] = [x if x in vocabulary else '__ATN_UNKNOWN__' for x in tokens]
            else:
                frame['f3'] = pd.Categorical([x if x in vocabulary else None for x in tokens], categories=vocabulary)
    yF, yG = targets
    if family == 'P08':
        from catboost import CatBoostRegressor, Pool
        model = CatBoostRegressor(iterations=64, depth=4, learning_rate=.1, l2_leaf_reg=3.,
            loss_function='RMSE', eval_metric='RMSE', task_type='CPU', thread_count=1,
            random_seed=7, use_best_model=True, counter_calc_method='SkipTest',
            random_strength=1., bootstrap_type='MVS', subsample=.8, allow_writing_files=False)
        model.fit(Pool(F, yF, cat_features=[3] if categorical else []),
            eval_set=Pool(G, yG, cat_features=[3] if categorical else []), early_stopping_rounds=4, verbose=False)
        prediction = model.predict(G)
    elif family == 'P09':
        import xgboost as xgb
        dF = xgb.QuantileDMatrix(F, yF, max_bin=64, nthread=1, enable_categorical=True)
        dG = xgb.QuantileDMatrix(G, yG, max_bin=64, nthread=1, enable_categorical=True, ref=dF)
        model = xgb.train(dict(objective='reg:squarederror', eval_metric='rmse', tree_method='hist',
            device='cpu', nthread=1, seed=7, max_depth=3, eta=.1, min_child_weight=1.,
            subsample=1., colsample_bytree=1., reg_lambda=1., reg_alpha=0., max_bin=64,
            validate_parameters=True), dF, num_boost_round=64, evals=[(dG, 'G')],
            callbacks=[xgb.callback.EarlyStopping(4, data_name='G', metric_name='rmse',
                maximize=False, save_best=True, min_delta=0.)], verbose_eval=False)
        prediction = model.predict(dG)
    else:
        import lightgbm as lgb
        parameters = dict(objective='regression', metric='rmse', device_type='cpu', tree_learner='serial',
            num_threads=1, seed=7, learning_rate=.1, num_leaves=7, min_data_in_leaf=8,
            lambda_l2=.01, linear_tree=family == 'P11', max_bin=31, deterministic=True,
            force_col_wise=True, bagging_fraction=1., bagging_freq=1, verbosity=-1)
        if family == 'P11':
            parameters['linear_lambda'] = .01
        dF = lgb.Dataset(F, yF, params=parameters, categorical_feature=['f3'] if categorical else [], free_raw_data=False)
        dG = lgb.Dataset(G, yG, reference=dF, params=parameters, categorical_feature=['f3'] if categorical else [], free_raw_data=False)
        model = lgb.train(parameters, dF, num_boost_round=64, valid_sets=[dG], valid_names=['G'],
            callbacks=[lgb.early_stopping(4, verbose=False)])
        prediction = model.predict(G, num_iteration=model.best_iteration)
    return np.asarray(prediction, np.float64)


def worker(family, categorical, output):
    from treeneuron.baselines.api import FitRequest
    from treeneuron.research.native_trees import NativeTreePredictor
    frames, targets = fixture(categorical)
    names = list(frames[0].columns)
    parameters = ({'depth': 4, 'learning_rate': .1, 'l2_leaf_reg': 3.} if family == 'P08' else
        {'max_depth': 3, 'eta': .1, 'max_bin': 64} if family == 'P09' else
        {'num_leaves': 7, 'min_data_in_leaf': 8, 'learning_rate': .1, 'lambda_l2': .01,
         'max_bin': 31, **({'linear_lambda': .01} if family == 'P11' else {})})
    config = dict(schema='research/v2-native-tree-config', family=family, mode='engineering_fixture',
        feature_schema=dict(columns=names, categorical=['kind'] if categorical else []),
        parameters=parameters, max_rounds=64, patience=4)
    arrays = {f'X_{role}': frame.iloc[:, :3].to_numpy(np.float64)
              for role, frame in zip(('F', 'G'), frames)}
    arrays.update(y_F=targets[0], y_G=targets[1])
    if categorical:
        arrays.update({f'category_{role}': frame['kind'].to_numpy(dtype=str)
                       for role, frame in zip(('F', 'G'), frames)})
    np.savez_compressed(output/'fixture.npz', **arrays)
    class RoleSafeRequest(FitRequest):
        @property
        def V(self):
            raise AssertionError('Fitting accessed V labels')
        @property
        def T(self):
            raise AssertionError('Fitting accessed T labels')
    model = NativeTreePredictor(family).fit(RoleSafeRequest(frames[0], targets[0], frames[1], targets[1],
        7, config, {}, output/'adapter-fit'))
    prediction = model.predict(frames[1], batch_size=13)
    direct_prediction = direct(family, frames, targets, categorical)
    np.testing.assert_array_equal(prediction, direct_prediction)
    model.save(output/'snapshot')
    trusted_state = hashlib.sha256((output/'snapshot/state.json').read_bytes()).hexdigest()
    restored = NativeTreePredictor.load(output/'snapshot', expected_state_sha256=trusted_state)
    np.testing.assert_array_equal(prediction, restored.predict(frames[1]))
    np.savez_compressed(output/'predictions.npz', adapter=prediction,
        direct=direct_prediction, restored=restored.predict(frames[1]))
    assert model.predict(frames[1].iloc[:0]).shape == (0,)
    summary = model.resource_summary()
    assert summary['F_rows'] == 192 and summary['G_rows'] == 64
    assert summary['checkpoint_eligible'] is False and summary['native_full_anchor_qualified'] is False
    assert 1 <= summary['selected_rounds'] <= summary['executed_rounds'] <= 64
    assert len(summary['G_rmse_history']) == summary['executed_rounds']
    assert summary['total_fit_wall_ns'] > 0 and summary['process_peak_rss_bytes'] > 0
    assert summary['runtime']['openmp_environment_context']['caller_value'] is None
    context = summary['runtime']['openmp_environment_context']
    assert context['observed_after_native_import'] == (None if family == 'P08' else 'True')
    assert context['observed_after_runtime_inspection'] == 'True'
    assert bool(context['sklearn_init_sha256']) == (family != 'P08')
    assert context['threadpoolctl_sha256']
    for corrupt in ('model', 'source', 'dependency', 'transform'):
        changed = output/f'corrupt-{corrupt}'
        shutil.copytree(output/'snapshot', changed)
        state_path = changed/'state.json'
        envelope = json.loads(state_path.read_text())
        if corrupt == 'model':
            with (changed/envelope['payload']['model_file']).open('ab') as stream:
                stream.write(b'corrupt')
        else:
            if corrupt == 'source':
                envelope['payload']['source']['package_sha256'] = '0'*64
            if corrupt == 'dependency':
                envelope['payload']['dependencies']['environment_sha256'] = '0'*64
            if corrupt == 'transform':
                envelope['payload']['transform']['scales'][0] = 0.
            envelope['sha256'] = hashlib.sha256(json.dumps(envelope['payload'], sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            state_path.write_text(json.dumps(envelope))
        with pytest.raises(ValueError):
            # Supply the new digest intentionally to exercise semantic guards;
            # a real coordinator retains the original trusted external digest.
            NativeTreePredictor.load(changed,
                expected_state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest())
    if family == 'P11':
        import lightgbm as lgb
        native = lgb.Booster(model_file=output/'snapshot/model.txt')
        def leaves(tree):
            return leaves(tree['left_child'])+leaves(tree['right_child']) if 'left_child' in tree else [tree]
        trees = native.dump_model()['tree_info']
        assert all(not leaf.get('leaf_coeff') for leaf in leaves(trees[0]['tree_structure']))
        affine = [leaf for tree in trees[1:] for leaf in leaves(tree['tree_structure']) if leaf.get('leaf_coeff')]
        assert affine and any(any(x != 0 for x in leaf['leaf_coeff']) for leaf in affine)
        assert not categorical or all(3 not in leaf['leaf_features'] for leaf in affine)
    if categorical:
        stress = NativeTreePredictor(family).fit(RoleSafeRequest(frames[0], targets[0], frames[1],
            4-targets[1], 7, config, {}, output/'stopping-stress'))
        stop = stress.resource_summary()
        assert stop['selected_rounds'] < stop['executed_rounds'] < 64
    assert 'torch' not in sys.modules
    package = {'P08': 'catboost', 'P09': 'xgboost', 'P10': 'lightgbm', 'P11': 'lightgbm'}[family]
    assert not (set(('catboost', 'xgboost', 'lightgbm'))-{package}) & set(sys.modules)
    (output/'verified.json').write_text(json.dumps(dict(family=family, categorical=categorical,
        parity='exact', save_load='exact', snapshot_corruption_checks=4, accessed_roles=['synthetic_F', 'synthetic_G'],
        native_fits=3 if categorical else 2, summary=summary), indent=2, sort_keys=True))


@pytest.mark.parametrize('family,categorical', [(f, c) for f in ('P08', 'P09', 'P10', 'P11') for c in (False, True)])
def test_tiny_official_parity_and_bound_snapshots(tmp_path, family, categorical):
    overlay = os.environ.get('ATN_TREE_OVERLAY')
    if not overlay:
        pytest.skip('Optional pinned tree overlay not admitted for this test run')
    import treeneuron.research.native_trees as module
    assert hasattr(module, 'NativeTreePredictor'), 'Native fit adapter is missing'
    output = Path(os.environ.get('ATN_TREE_ADAPTER_TEST_OUTPUT', tmp_path))/f'{family}-{categorical}'
    output.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(output).free > 256*1024**2
    vm = subprocess.run(['/usr/bin/vm_stat'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    page = int(re.search(r'page size of (\d+) bytes', vm.stdout)[1])
    free = int(re.search(r'Pages free:\s+(\d+)', vm.stdout)[1])
    inactive = int(re.search(r'Pages inactive:\s+(\d+)', vm.stdout)[1])
    guard = dict(free_disk_bytes=shutil.disk_usage(output).free,
        free_plus_inactive_ram_proxy_bytes=(free+inactive)*page,
        ram_proxy_is_not_a_guarantee=True, vm_stat_output=vm.stdout)
    (output/'admission.json').write_text(json.dumps(guard, indent=2))
    if guard['free_plus_inactive_ram_proxy_bytes'] < 512*1024**2:
        pytest.skip('Recorded resource admission proxy is below512MiB; retain and retry the same case later')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+overlay,
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
        MPLCONFIGDIR=str(output/'mpl'), XDG_CACHE_HOME=str(output/'cache'))
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    argv = [sys.executable, str(Path(__file__).resolve()), '--worker', family, str(int(categorical)), str(output)]
    started = time.perf_counter_ns()
    with (output/'process.log').open('x') as log:
        try:
            process = subprocess.run(argv, env=env, cwd=Path(__file__).resolve().parents[1],
                stdout=log, stderr=subprocess.STDOUT, timeout=120)
            status = dict(exit_code=process.returncode, timeout=False)
        except subprocess.TimeoutExpired:
            status = dict(exit_code=None, timeout=True)
    (output/'command.json').write_text(json.dumps(dict(argv=argv, environment_overrides={key: env[key]
        for key in ('PYTHONPATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'MPLCONFIGDIR', 'XDG_CACHE_HOME')},
        cwd=str(Path(__file__).resolve().parents[1]),
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        complete_process_wall_ns=time.perf_counter_ns()-started, **status), indent=2))
    assert status['exit_code'] == 0, (output/'process.log').read_text()
    assert (output/'verified.json').exists()


if __name__ == '__main__':
    assert sys.argv[1] == '--worker'
    worker(sys.argv[2], bool(int(sys.argv[3])), Path(sys.argv[4]))
