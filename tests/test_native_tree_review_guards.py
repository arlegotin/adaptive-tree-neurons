"""Read-only-native review probes: mutated snapshot copies, no fitting."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import pytest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(kind, output, reference):
    from treeneuron.research import native_trees as module
    if kind in ('eta', 'promotion', 'promotion-stopping', 'native-configuration'):
        shutil.copytree(reference, output/'copy')
        path = output/'copy/state.json'
        envelope = json.loads(path.read_text())
        value = envelope['payload']
        # This is an explicitly forged test input, not a migration or a new
        # scientific claim for old weights. It isolates recipe validation from
        # the independent package-source guard after this fix changes source.
        value['source'] = module._source()
        if kind == 'eta':
            value['config']['parameters']['eta'] = .2
        elif kind == 'native-configuration':
            value['resources']['resolved_parameters']['learner']['gradient_booster']['tree_train_param']['eta'] = '.2'
        else:
            value['config']['mode'] = value['resources']['mode'] = 'search'
            for key in ('max_rounds', 'patience'):
                value['config'].pop(key)
            value['config']['parameters'].pop('max_bin')
            value['resources'].update(status='completed_eligible', checkpoint_eligible=True)
            if kind == 'promotion-stopping':
                # Bypass the earlier requested-parameter mismatch deliberately:
                # 30 selected /34 executed still cannot satisfy patience100.
                value['resources']['requested_parameters'].pop('max_bin')
        envelope['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        path.write_text(json.dumps(envelope))
        with pytest.raises(ValueError, match='recipe|parameter|stopping|eligib'):
            module.NativeTreePredictor.load(output/'copy', expected_state_sha256=digest(path))
        with pytest.raises(ValueError, match='trusted'):
            module.NativeTreePredictor.load(output/'copy', expected_state_sha256=digest(reference/'state.json'))
    else:
        fake = output/'external.py'
        fake.write_text('external origin must never be treated as a pinned native implementation\n')
        if kind == 'module-origin':
            import importlib.machinery
            sys.modules['xgboost'] = types.SimpleNamespace(__version__='3.4.1', __file__=str(fake),
                __spec__=importlib.machinery.ModuleSpec('xgboost', loader=None, origin=str(fake)),
                core=types.SimpleNamespace(_LIB=types.SimpleNamespace(_name=str(fake))))
        else:
            imported, _, _ = module._dependencies('P09')
            imported.core._LIB._name = str(fake)
        with pytest.raises(ValueError, match='origin|library|RECORD'):
            module._dependencies('P09')
    (output/'verified.json').write_text(json.dumps(dict(kind=kind, native_fits=0,
        forged_input_only=True, original_snapshot_unchanged=True)))


@pytest.mark.parametrize('kind', ['eta', 'promotion', 'promotion-stopping', 'native-configuration',
                                'module-origin', 'library-origin'])
def test_reviewed_snapshot_and_import_guards_without_fitting(tmp_path, kind):
    overlay = os.environ.get('ATN_TREE_OVERLAY')
    reference = os.environ.get('ATN_TREE_SNAPSHOT_FIXTURE')
    if not overlay or not reference:
        pytest.skip('Optional locked environment and retained snapshot required for no-fit review probes')
    output = Path(os.environ.get('ATN_TREE_REVIEW_OUTPUT', tmp_path))/kind
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+overlay,
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(output/'mpl'), MPLBACKEND='Agg')
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    argv = [sys.executable, str(Path(__file__).resolve()), '--worker', kind, str(output), reference]
    started = time.perf_counter_ns()
    process = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    (output/'process.log').write_text(process.stdout)
    (output/'command.json').write_text(json.dumps(dict(argv=argv, exit_code=process.returncode,
        environment_overrides={key: env[key] for key in ('PYTHONPATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'MPLCONFIGDIR')},
        test_source_sha256=digest(__file__), complete_process_wall_ns=time.perf_counter_ns()-started)))
    assert process.returncode == 0, process.stdout


if __name__ == '__main__':
    worker(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
