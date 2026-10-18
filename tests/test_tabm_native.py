"""Sequential tiny official TabM fits, without prepared transfer/V/T data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from test_tabm_contract import api


@pytest.mark.parametrize('variant', ['plain', 'piecewise'])
@pytest.mark.parametrize('categorical', [False, True])
def test_tiny_official_fit_prediction_snapshot_and_audit(variant, categorical, tmp_path):
    module = api()
    assert hasattr(module, 'TabMPredictor'), 'Complete official TabM fit is missing'
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root/'src'),
        str(root/'environment/research-v2-tabm-site')]), PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env['MPLCONFIGDIR'], env['XDG_CACHE_HOME'] = str(tmp_path/'mpl'), str(tmp_path/'cache')
    argv = [sys.executable, str(Path(__file__).resolve()), '--worker', variant,
        str(int(categorical)), str(tmp_path/'worker')]
    started, result, error = time.perf_counter_ns(), None, None
    try:
        result = subprocess.run(argv, cwd=root, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=90)
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        raise
    finally:
        (tmp_path/'command.json').write_text(json.dumps(dict(argv=argv, cwd=str(root),
            environment={k: env.get(k) for k in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'KMP_DUPLICATE_LIB_OK')},
            status='supervisor_error' if error else 'completed' if result.returncode == 0 else 'failed',
            error=error, total_wall_ns=time.perf_counter_ns()-started,
            returncode=result.returncode if result is not None else None), indent=2)+'\n')
        (tmp_path/'worker.log').write_text(result.stdout if result is not None else str(error))
    assert result.returncode == 0, result.stdout
    receipt = json.loads((tmp_path/'worker/parity.json').read_text())
    assert receipt['prediction_max_abs_error'] == 0.
    assert receipt['reload_max_abs_error'] == 0.
    assert receipt['full_quality_anchor_executed'] is False
    assert len(receipt['rejected_mutations']) >= 12


def worker(variant, categorical, output):
    import copy
    import hashlib
    import shutil
    import types
    import numpy as np
    import pandas as pd
    from treeneuron.baselines.api import FitRequest
    from test_tabm_contract import config
    from tabm_reference import direct
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(2026101817)
    def draw(n, query=False):
        x = rng.normal(size=n)
        frame = pd.DataFrame(dict(x=x, low_distinct=np.where(x >= 0, 1., -1.), constant=np.full(n, 7.)))
        y = np.asarray(3.+2*x+.2*x*x, np.float32)
        if categorical:
            frame['kind'] = ['a' if i % 2 else ('new' if query else 'b') for i in range(n)]
        return frame, y
    F, yF = draw(96); G, yG = draw(23); Q, _ = draw(11, True)
    schema = dict(columns=list(F), categorical=['kind'] if categorical else [])
    value = config(variant, 'engineering_fixture', feature_schema=schema,
        parameters=dict(n_blocks=2, d_block=64), max_epochs=3)
    request = FitRequest(F, yF, G, yG, 2026101818, value, {}, output/'fit')
    predictor = api().TabMPredictor().fit(request)
    prediction, summary = predictor.predict(Q), predictor.resource_summary()
    cfg = api().TabMConfig.parse(value)
    reference, history, row_orders, expected_transforms = direct(F, yF, G, yG, Q, cfg, request.seed)
    np.testing.assert_array_equal(prediction, reference.astype(np.float64))
    assert summary['history'] == history
    with np.load(output/'fit/F-row-orders.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['row_indices'], row_orders)
    assert predictor._transform.to_state() == expected_transforms['features']
    assert predictor._target.to_state() == expected_transforms['target']
    assert summary['checkpoint_eligible'] is False and summary['quality_eligible'] is False
    assert summary['accessed_roles'] == ['F', 'G'] and summary['full_F_coverage_each_epoch'] is True
    assert summary['dependencies']['core_distribution_count'] == 27
    assert summary['dependencies']['overlay_distribution_count'] == 2
    assert summary['dependencies']['import_environment']['caller_KMP_DUPLICATE_LIB_OK'] is None
    snapshot = output/'snapshot'
    predictor.save(snapshot)
    digest = hashlib.sha256((snapshot/'manifest.json').read_bytes()).hexdigest()
    reloaded = api().TabMPredictor.load(snapshot, expected_manifest_sha256=digest)
    np.testing.assert_array_equal(reloaded.predict(Q), prediction)
    np.savez(output/'fixture.npz', yF=yF, yG=yG, prediction=prediction, reference=reference)
    (output/'fixture-features.json').write_text(json.dumps(dict(F=F.to_dict('list'),
        G=G.to_dict('list'), Q=Q.to_dict('list'), schema=schema), indent=2)+'\n')
    cases = []
    for name in ('trusted_hash', 'source', 'config_lr', 'fixture_promotion', 'weights_dtype',
            'weight_edit', 'quantile', 'target', 'negative_G', 'optimizer', 'runtime', 'bins',
            'native_image_authentication'):
        if name == 'native_image_authentication':
            from unittest.mock import patch
            with patch.object(api(), '_runtime', side_effect=ValueError('simulated native image authentication failure')):
                with pytest.raises(ValueError, match='authentication'):
                    api().TabMPredictor.load(snapshot, expected_manifest_sha256=digest)
            cases.append(name); continue
        altered = output/name
        shutil.copytree(snapshot, altered)
        manifest = json.loads((altered/'manifest.json').read_text())
        if name == 'source':
            manifest['source']['package_sha256'] = '0'*64
        elif name == 'config_lr':
            manifest['config']['parameters']['lr'] = .003
        elif name == 'fixture_promotion':
            manifest['config']['mode'] = 'search'; del manifest['config']['max_epochs']
            manifest['resources']['mode'] = 'search'
            manifest['resources']['status'] = 'completed_eligible'
            manifest['resources']['checkpoint_eligible'] = True
        elif name == 'quantile':
            manifest['feature_transform']['quantiles'][0][0] -= .01
        elif name == 'target':
            manifest['target_transform']['mean'] = float(np.float32(manifest['target_transform']['mean']+.1))
        elif name == 'negative_G':
            manifest['resources']['history'][0]['G_rmse'] = -1.
        elif name == 'optimizer':
            manifest['resources']['optimizer_defaults']['lr'] = .003
        elif name == 'runtime':
            manifest['resources']['runtime']['native_images'][0]['sha256'] = '0'*64
        elif name == 'bins':
            if variant == 'piecewise':
                manifest['bins'][0][0] -= .01
            else:
                manifest['bins'] = [[0., 1.]]
        elif name in ('weights_dtype', 'weight_edit'):
            with np.load(altered/'weights.npz', allow_pickle=False) as archive:
                weights = {k: archive[k].copy() for k in archive.files}
            key = next(k for k, v in weights.items() if v.dtype == np.float32)
            if name == 'weights_dtype':
                weights[key] = weights[key].astype(np.float64)
            else:
                weights[key].flat[0] += np.float32(.01)
            np.savez(altered/'weights.npz', **weights)
            manifest['weights_sha256'] = hashlib.sha256((altered/'weights.npz').read_bytes()).hexdigest()
        (altered/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        updated = hashlib.sha256((altered/'manifest.json').read_bytes()).hexdigest()
        with pytest.raises(ValueError):
            api().TabMPredictor.load(altered, expected_manifest_sha256='0'*64 if name == 'trusted_hash' else updated)
        cases.append(name)
    from unittest.mock import patch
    import tabm
    external = output/'external_tabm.py'
    external.write_text('# Version matching does not authenticate the imported module.\n')
    stub = types.SimpleNamespace(__version__=tabm.__version__, __file__=str(external), __spec__=tabm.__spec__)
    with patch.dict(sys.modules, {'tabm': stub}):
        with pytest.raises(ValueError, match='origin'):
            api()._dependencies()
    cases.append('version_matching_external_import')
    if variant == 'plain':
        # Actual official categorical-only forward, after the zero-numeric boundary check.
        with __import__('torch').random.fork_rng():
            catmodel = api().build_model(cfg, 0, [3], [])
        assert api()._members(catmodel, np.empty((3, 0), np.float32), np.array([[0], [1], [2]], np.int64), 8096).shape == (3, 32)
    (output/'parity.json').write_text(json.dumps(dict(variant=variant, categorical=categorical,
        prediction_max_abs_error=float(np.max(np.abs(prediction-reference))), reload_max_abs_error=0.,
        rejected_mutations=cases, full_quality_anchor_executed=False, source=predictor._source,
        summary=summary), indent=2)+'\n')


if __name__ == '__main__':
    assert sys.argv[1] == '--worker'
    worker(sys.argv[2], bool(int(sys.argv[3])), Path(sys.argv[4]))
