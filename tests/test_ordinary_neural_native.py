"""Fresh one-thread tiny F/G fits versus direct pinned author invocation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from test_ordinary_neural_contract import api, config


@pytest.mark.parametrize('family', ['P03', 'P04'])
@pytest.mark.parametrize('categorical', [False, True])
def test_complete_tiny_fit_save_load_and_tamper_rejection_match_direct_author(family, categorical, tmp_path):
    module = api()
    assert hasattr(module, '_train_model'), 'Complete native F/G training is missing'
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/'src'))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['MPLCONFIGDIR'], env['XDG_CACHE_HOME'] = str(tmp_path/'mpl'), str(tmp_path/'cache')
    argv = [sys.executable, str(Path(__file__).resolve()), '--worker', family,
        str(int(categorical)), str(tmp_path/'worker')]
    started = time.perf_counter_ns()
    outcome, error, result = 'not_started', None, None
    try:
        result = subprocess.run(argv, env=env, cwd=root, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=60)
        outcome = 'completed' if result.returncode == 0 else 'failed'
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        outcome = 'supervisor_error'
        raise
    finally:
        (tmp_path/'command.json').write_text(json.dumps(dict(argv=argv, cwd=str(root),
            environment={k: env.get(k) for k in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'KMP_DUPLICATE_LIB_OK')},
            status=outcome, error=error, total_wall_ns=time.perf_counter_ns()-started,
            returncode=result.returncode if result is not None else None), indent=2)+'\n')
        (tmp_path/'worker.log').write_text(result.stdout if result is not None else str(error))
    assert result.returncode == 0, result.stdout
    receipt = json.loads((tmp_path/'worker'/'parity.json').read_text())
    assert receipt['prediction_max_abs_error'] == 0.
    assert receipt['checkpoint_reload_max_abs_error'] == 0.
    assert receipt['snapshot_rejection_cases'] == ['wrong_trusted_hash', 'learning_rate_edit',
        'fixture_promotion', 'tensor_dtype_edit', 'source_lock_edit', 'target_transform_edit',
        'feature_transform_edit', 'same_dtype_weight_edit', 'optimizer_defaults_edit',
        'actual_runtime_authentication', 'retained_runtime_image_edit', 'negative_G_risk',
        'negative_mean_risk', 'finite_fallback_edit', 'nonfinite_fallback_edit']
    assert receipt['full_quality_anchor_executed'] is False


def test_loading_requires_external_trusted_digest_before_reading_payload(tmp_path):
    module = api()
    assert hasattr(module.OrdinaryNeuralPredictor, 'load'), 'Bound snapshot loading is missing'
    with pytest.raises(ValueError, match='trusted'):
        module.OrdinaryNeuralPredictor.load(tmp_path/'missing', expected_manifest_sha256=None)


def worker(family, categorical, output):
    import copy
    import shutil
    from treeneuron.baselines.api import FitRequest
    module = api()
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < 256*1024**2:
        raise OSError(28, 'Need256MiB free for bounded fixture transaction')
    rng = np.random.default_rng(7719)
    rawF, rawG, rawQ = (rng.normal(size=(n, 2)) for n in (160, 33, 19))
    yF = 7.+2.*rawF[:, 0]+.3*rawF[:, 1]+rng.normal(0, .01, 160)
    yG = 7.+2.*rawG[:, 0]+.3*rawG[:, 1]+rng.normal(0, .01, 33)
    schema = dict(columns=['x', 'kind' if categorical else 'z'],
        categorical=['kind'] if categorical else [])
    if categorical:
        F, G, Q = (a.astype(object) for a in (rawF, rawG, rawQ))
        F[:, 1] = np.array(['a', 'b', 'c', 'a']*40)
        G[:, 1] = np.array(['b', 'new', 'a']*11)
        Q[:, 1] = np.array(['new']*19)
    else:
        F, G, Q = rawF, rawG, rawQ
    cfg = config(family, 'engineering_fixture', feature_schema=schema, max_epochs=3, patience=2)
    request = FitRequest(F, yF, G, yG, 853, cfg, {}, output/'fit')
    fitted = module.OrdinaryNeuralPredictor(family).fit(request)
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
    from ordinary_neural_reference import official_definitions, official_model, official_threshold
    prediction = fitted.predict(Q, batch_size=7)
    summary = fitted.resource_summary()
    assert summary['status'] == 'completed_ineligible'
    assert summary['checkpoint_eligible'] is False and summary['quality_eligible'] is False
    assert summary['fit_rows'] == 160 and summary['select_rows'] == 33
    assert summary['full_F_coverage_each_epoch'] is True
    assert summary['import_environment']['caller_KMP_DUPLICATE_LIB_OK'] is None
    assert summary['import_environment']['post_import_KMP_DUPLICATE_LIB_OK'] == 'True'
    # Independent author preprocessing, with the declared F-only target boundary.
    defs = official_definitions()
    numeric_indices = [0] if categorical else [0, 1]
    quantile = defs['TabrQuantileTransformer'](random_state=0).fit(
        np.ascontiguousarray(rawF[:, numeric_indices], dtype=np.float32))
    encoded = []
    for raw in (F, G, Q):
        values = np.empty(raw.shape, dtype=np.float32)
        values[:, numeric_indices] = quantile.transform(np.ascontiguousarray(raw[:, numeric_indices], dtype=np.float32))
        if categorical:
            values[:, 1] = [dict(a=0, b=1, c=2).get(x, 3) for x in raw[:, 1]]
        encoded.append(values)
    target = StandardScaler().fit(yF[:, None])
    tF, tG = (target.transform(y[:, None]).astype(np.float32) for y in (yF, yG))
    with torch.random.fork_rng():
        torch.manual_seed(853)
        author = official_model(family, schema, [4] if categorical else [], {})
        optimizer = torch.optim.AdamW(author.parameters(), lr=.001, weight_decay=.01)
        train = DataLoader(TensorDataset(torch.from_numpy(encoded[0]), torch.from_numpy(tF)),
            batch_size=128, shuffle=True)
        select = DataLoader(TensorDataset(torch.from_numpy(encoded[1]), torch.from_numpy(tG)),
            batch_size=128, shuffle=False)
        best, threshold, misses, history, best_weights, best_epoch = float('inf'), float('inf'), 0, [], None, 0
        for epoch in range(1, 4):
            author.train()
            for x, y in train:
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(author(x), y)
                loss.backward(); optimizer.step()
            author.eval()
            total = 0.
            with torch.no_grad():
                for x, y in select:
                    total += torch.nn.functional.mse_loss(author(x), y).item()*len(x)
            risk = total/len(tG)
            history.append(risk)
            if risk < best:
                best, best_epoch, best_weights = risk, epoch, copy.deepcopy(author.state_dict())
            if risk < threshold:
                threshold, misses = official_threshold(risk), 0
            else:
                misses += 1
            if misses >= 2:
                break
        author.load_state_dict(best_weights)
        author.eval()
        mean = float(np.mean(tF))
        mean_risk = float(np.mean((tG-mean)**2))
        with torch.no_grad():
            # Same declared batch-size query; BN stays in evaluation mode.
            p = np.concatenate([author(torch.from_numpy(encoded[2][start:start+7])).numpy().reshape(-1)
                for start in range(0, len(Q), 7)])
        if mean_risk < min(history):
            p[:] = mean
        expected = target.inverse_transform(p[:, None].astype(np.float64)).reshape(-1)
    np.testing.assert_array_equal(prediction, expected)
    assert summary['selected_epoch'] == best_epoch
    assert summary['executed_epochs'] == len(history)
    np.testing.assert_array_equal([row['G_mse'] for row in summary['history']], history)
    snapshot = output/'snapshot'
    fitted.save(snapshot)
    digest = hashlib.sha256((snapshot/'manifest.json').read_bytes()).hexdigest()
    loaded = module.OrdinaryNeuralPredictor.load(snapshot, expected_manifest_sha256=digest)
    np.testing.assert_array_equal(loaded.predict(Q, batch_size=7), prediction)
    assert loaded.predict(Q[:0]).shape == (0,)
    import pytest as check
    with check.raises(ValueError):
        module.OrdinaryNeuralPredictor.load(snapshot, expected_manifest_sha256='0'*64)
    cases = ['wrong_trusted_hash']
    for name in ('learning_rate_edit', 'fixture_promotion', 'tensor_dtype_edit', 'source_lock_edit',
            'target_transform_edit', 'feature_transform_edit', 'same_dtype_weight_edit', 'optimizer_defaults_edit',
            'actual_runtime_authentication', 'retained_runtime_image_edit', 'negative_G_risk',
            'negative_mean_risk', 'finite_fallback_edit', 'nonfinite_fallback_edit'):
        if name == 'actual_runtime_authentication':
            from unittest.mock import patch
            with patch.object(module, '_runtime', side_effect=ValueError('Actual native runtime authentication failed')):
                with check.raises(ValueError, match='Actual native runtime authentication failed'):
                    module.OrdinaryNeuralPredictor.load(snapshot, expected_manifest_sha256=digest)
            cases.append(name)
            continue
        altered = output/name
        shutil.copytree(snapshot, altered)
        manifest = json.loads((altered/'manifest.json').read_text())
        if name == 'learning_rate_edit':
            manifest['config']['parameters']['lr'] = .002
        elif name == 'fixture_promotion':
            manifest['config']['mode'] = 'search'
            del manifest['config']['max_epochs']; del manifest['config']['patience']
            manifest['resources']['mode'] = 'search'
            manifest['resources']['checkpoint_eligible'] = True
            manifest['resources']['status'] = 'completed_eligible'
        elif name == 'source_lock_edit':
            manifest['source']['package_sha256'] = '0'*64
        elif name == 'target_transform_edit':
            manifest['target_transform']['mean'] += .1
        elif name == 'feature_transform_edit':
            manifest['feature_transform']['quantiles'][0][0] -= .1
        elif name == 'optimizer_defaults_edit':
            manifest['resources']['optimizer_defaults']['lr'] = .002
        elif name == 'retained_runtime_image_edit':
            manifest['resources']['runtime']['native_images'][0]['sha256'] = '0'*64
        elif name == 'negative_G_risk':
            manifest['resources']['history'][manifest['resources']['selected_epoch']-1]['G_mse'] = -1.
        elif name in ('negative_mean_risk', 'finite_fallback_edit', 'nonfinite_fallback_edit'):
            manifest['resources']['mean_G_mse'] = -1. if name == 'negative_mean_risk' else 0.
            manifest['resources']['selected_F_mean'] = True
            manifest['resources']['mean_prediction_standardized'] = float('nan') if name == 'nonfinite_fallback_edit' else 123.
        else:
            with np.load(altered/'weights.npz', allow_pickle=False) as archive:
                state = {key: archive[key].copy() for key in archive.files}
            key = next(k for k, v in state.items() if v.dtype == np.float32)
            if name == 'same_dtype_weight_edit':
                state[key].flat[0] += np.float32(.01)
            else:
                state[key] = state[key].astype(np.float64)
            np.savez(altered/'weights.npz', **state)
            manifest['weights_sha256'] = hashlib.sha256((altered/'weights.npz').read_bytes()).hexdigest()
        (altered/'manifest.json').write_text(json.dumps(manifest, sort_keys=True,
            allow_nan=name == 'nonfinite_fallback_edit')+'\n')
        revised = hashlib.sha256((altered/'manifest.json').read_bytes()).hexdigest()
        with check.raises(ValueError):
            module.OrdinaryNeuralPredictor.load(altered, expected_manifest_sha256=revised)
        cases.append(name)
    np.savez(output/'fixture.npz', numeric_F=rawF, numeric_G=rawG, numeric_query=rawQ, yF=yF, yG=yG,
        prediction=prediction, direct_author_prediction=expected)
    (output/'parity.json').write_text(json.dumps(dict(family=family, categorical=categorical,
        seed=853, teacher_seed=7719, config=cfg, prediction_max_abs_error=float(np.max(np.abs(prediction-expected))),
        checkpoint_reload_max_abs_error=float(np.max(np.abs(loaded.predict(Q, batch_size=7)-prediction))),
        snapshot_rejection_cases=cases, full_quality_anchor_executed=False,
        official_reference='Pinned author architecture/preprocessor with explicit F/G boundary; no full author wrapper',
        resources=summary), sort_keys=True, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(family=family, categorical=categorical, parity='exact', epochs=len(history))))


if __name__ == '__main__':
    assert sys.argv[1] == '--worker'
    worker(sys.argv[2], bool(int(sys.argv[3])), Path(sys.argv[4]))
