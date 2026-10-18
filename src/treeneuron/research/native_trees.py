"""Pinned, lazy native P08–P11 regressors; F/G fitting and feature-only queries.

The synchronous adapter does not qualify full workloads or resume interrupted
native training. Engineering fixtures never supply quality-eligible models.
"""
from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

from treeneuron.baselines.api import FitRequest


FAMILIES = {'P08': ('catboost', 10000, 200), 'P09': ('xgboost', 10000, 100),
            'P10': ('lightgbm', 10000, 100), 'P11': ('lightgbm', 5000, 100)}


def _schema(value):
    if (type(value) is not dict or set(value) != {'columns', 'categorical'}
            or type(value['columns']) is not list or not value['columns']
            or type(value['categorical']) is not list
            or any(type(x) is not str or not x for x in value['columns'] + value['categorical'])
            or len(set(value['columns'])) != len(value['columns'])
            or len(set(value['categorical'])) != len(value['categorical'])
            or not set(value['categorical']) <= set(value['columns'])):
        raise ValueError('Require an explicit ordered feature schema with unique category names')
    return json.loads(json.dumps(value))


def _number(value, lower, upper):
    return (type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper)


@dataclass(frozen=True)
class NativeTreeConfig:
    family: str
    mode: str
    feature_schema: dict[str, Any]
    parameters: dict[str, Any]
    max_rounds: int
    patience: int

    @classmethod
    def parse(cls, family, value):
        if family not in FAMILIES:
            raise ValueError('Require a native tree family P08–P11')
        fields = {'schema', 'family', 'mode', 'feature_schema', 'parameters'}
        if (type(value) is not dict or value.get('schema') != 'research/v2-native-tree-config'
                or value.get('family') != family
                or value.get('mode') not in ('native_anchor', 'search', 'engineering_fixture')):
            raise ValueError('Invalid versioned native tree configuration')
        fixture = value['mode'] == 'engineering_fixture'
        if set(value) != fields | ({'max_rounds', 'patience'} if fixture else set()):
            raise ValueError('Unknown native setting or implicit training-budget override')
        _, rounds, patience = FAMILIES[family]
        if fixture:
            rounds, patience = value['max_rounds'], value['patience']
            if type(rounds) is not int or not 1 <= rounds <= 128 or type(patience) is not int or not 1 <= patience <= 8:
                raise ValueError('Engineering fixtures require at most128 rounds/patience8')
        parameters = value['parameters']
        if type(parameters) is not dict or value['mode'] == 'native_anchor' and parameters:
            raise ValueError('Native anchors have no supplied parameter overrides')
        supports = {
            'P08': {'depth': (4, 6, 8, 10), 'learning_rate': (.01, .3), 'l2_leaf_reg': (1., 100.)},
            'P09': {'max_depth': (3, 5, 7, 9), 'eta': (.01, .3), 'min_child_weight': (1., 100.),
                    'subsample': (.6, 1.), 'colsample_bytree': (.6, 1.),
                    'reg_lambda': (1e-3, 100.), 'reg_alpha': (0., 10.)},
            'P10': {'num_leaves': (15, 31, 63, 127), 'learning_rate': (.01, .2),
                    'min_data_in_leaf': (5, 10, 20, 50, 100), 'feature_fraction': (.6, 1.),
                    'bagging_fraction': (.6, 1.), 'lambda_l2': (0., 100.)},
            'P11': {'num_leaves': (7, 15, 31), 'learning_rate': (.01, .2),
                    'min_data_in_leaf': (20, 50, 100), 'lambda_l2': (0., 100.), 'linear_lambda': (0., 100.)}}
        categorical = {'depth', 'max_depth', 'num_leaves', 'min_data_in_leaf'}
        allowed = supports[family]
        if fixture:
            allowed = dict(allowed)
            if family == 'P09':
                allowed['max_bin'] = (64,)
            if family in ('P10', 'P11'):
                allowed.update(num_leaves=(7, 15, 31), min_data_in_leaf=(8, 20, 50, 100), max_bin=(31,))
        for key, item in parameters.items():
            if key not in allowed:
                raise ValueError(f'Unknown or reserved native parameter: {key}')
            bounds = allowed[key]
            if key in categorical or key == 'max_bin':
                valid = type(item) is int and item in bounds
            else:
                valid = _number(item, *bounds)
                minimum_positive = {'reg_alpha': 1e-4, 'lambda_l2': 1e-3, 'linear_lambda': 1e-3}.get(key)
                if minimum_positive is not None:
                    valid = valid and (item == 0 or item >= minimum_positive)
            if not valid:
                raise ValueError(f'Native parameter outside frozen support: {key}')
        return cls(family, value['mode'], _schema(value['feature_schema']),
                   dict(parameters), rounds, patience)

    def to_state(self):
        value = dict(schema='research/v2-native-tree-config', family=self.family,
            mode=self.mode, feature_schema=self.feature_schema, parameters=self.parameters)
        if self.mode == 'engineering_fixture':
            value.update(max_rounds=self.max_rounds, patience=self.patience)
        return json.loads(json.dumps(value))


def _frame(X, schema):
    if isinstance(X, pd.DataFrame):
        if list(X.columns) != schema['columns']:
            raise ValueError('Input feature names/order differ from the fitted schema')
        frame = X.copy(deep=True)
    else:
        values = np.asarray(X, dtype=object)
        if values.ndim != 2 or values.shape[1] != len(schema['columns']):
            raise ValueError('Input feature shape differs from the fitted schema')
        frame = pd.DataFrame(values, columns=schema['columns'])
    return frame


def _category(value):
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) is str:
        return 'str:' + value
    if type(value) is bool:
        return 'bool:' + str(value)
    if type(value) is int:
        return 'int:' + str(value)
    if type(value) is float and math.isfinite(value):
        return 'float:' + value.hex()
    raise ValueError('Categories require finite numbers, strings or booleans; missing values are unsupported')


@dataclass(frozen=True)
class FeatureTransform:
    schema: dict[str, Any]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    vocabularies: tuple[tuple[str, ...], ...]
    linear: bool

    @classmethod
    def fit(cls, X, schema, *, linear):
        schema = _schema(schema)
        if type(linear) is not bool:
            raise ValueError('Linear scaling flag must be boolean')
        frame = _frame(X, schema)
        if not len(frame):
            raise ValueError('F must contain observations')
        means, scales, vocabularies = [], [], []
        for name in schema['columns']:
            if name in schema['categorical']:
                means.append(0.)
                scales.append(1.)
                vocabularies.append(tuple(sorted({_category(x) for x in frame[name]})))
            else:
                try:
                    values = frame[name].to_numpy(dtype=np.float64)
                except (ValueError, TypeError) as error:
                    raise ValueError('Numerical features must be finite numbers') from error
                if not np.isfinite(values).all():
                    raise ValueError('Numerical features must be finite; missing values are unsupported')
                mean = float(values.mean()) if linear else 0.
                scale = float(values.std(ddof=0)) if linear else 1.
                if not math.isfinite(mean) or not math.isfinite(scale):
                    raise ValueError('Numerical F statistics overflowed')
                means.append(mean)
                scales.append(scale if scale > 0 else 1.)
                vocabularies.append(())
        return cls(schema, tuple(means), tuple(scales), tuple(vocabularies), linear)

    def transform(self, X, family):
        if family not in FAMILIES:
            raise ValueError('Unknown native tree family')
        frame = _frame(X, self.schema)
        result = pd.DataFrame(index=np.arange(len(frame)))
        for index, name in enumerate(self.schema['columns']):
            if name in self.schema['categorical']:
                tokens = [_category(value) for value in frame[name]]
                vocabulary = self.vocabularies[index]
                known = set(vocabulary)
                if family == 'P08':
                    result[f'f{index}'] = [token if token in known else '__ATN_UNKNOWN__' for token in tokens]
                else:
                    result[f'f{index}'] = pd.Categorical(
                        [token if token in known else None for token in tokens], categories=vocabulary)
            else:
                try:
                    values = frame[name].to_numpy(dtype=np.float64)
                    values = (values-self.means[index])/self.scales[index]
                except (ValueError, TypeError) as error:
                    raise ValueError('Numerical features must be finite numbers') from error
                if not np.isfinite(values).all():
                    raise ValueError('Numerical query features must be finite')
                result[f'f{index}'] = values
        return result

    def to_state(self):
        return dict(schema='research/v2-native-tree-transform', feature_schema=self.schema,
            means=list(self.means), scales=list(self.scales),
            vocabularies=[list(x) for x in self.vocabularies], linear=self.linear)

    @classmethod
    def from_state(cls, value):
        if (type(value) is not dict or set(value) != {'schema', 'feature_schema', 'means', 'scales', 'vocabularies', 'linear'}
                or value['schema'] != 'research/v2-native-tree-transform' or type(value['linear']) is not bool):
            raise ValueError('Invalid native transform snapshot')
        schema = _schema(value['feature_schema'])
        width = len(schema['columns'])
        if (any(type(value[key]) is not list or len(value[key]) != width for key in ('means', 'scales', 'vocabularies'))
                or any(not _number(x, -float('inf'), float('inf')) for x in value['means'])
                or any(not _number(x, 0, float('inf')) or x <= 0 for x in value['scales'])):
            raise ValueError('Invalid native transform numerical state')
        for i, vocabulary in enumerate(value['vocabularies']):
            category = schema['columns'][i] in schema['categorical']
            if (type(vocabulary) is not list or any(type(x) is not str for x in vocabulary)
                    or vocabulary != sorted(set(vocabulary)) or bool(vocabulary) != category
                    or category and (value['means'][i] != 0 or value['scales'][i] != 1)
                    or not value['linear'] and (value['means'][i] != 0 or value['scales'][i] != 1)):
                raise ValueError('Invalid native category/scaling snapshot')
        return cls(schema, tuple(value['means']), tuple(value['scales']),
                   tuple(tuple(x) for x in value['vocabularies']), value['linear'])


def native_parameters(config, seed):
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('Native seed must be a declared nonnegative31-bit integer')
    if config.family == 'P08':
        value = dict(iterations=config.max_rounds, loss_function='RMSE', eval_metric='RMSE',
            task_type='CPU', thread_count=1, random_seed=seed, counter_calc_method='SkipTest',
            use_best_model=True, allow_writing_files=False, bootstrap_type='MVS', subsample=.8,
            random_strength=1.)
    elif config.family == 'P09':
        value = dict(objective='reg:squarederror', eval_metric='rmse', tree_method='hist',
            device='cpu', nthread=1, seed=seed, max_depth=6, eta=.3, min_child_weight=1.,
            subsample=1., colsample_bytree=1., reg_lambda=1., reg_alpha=0., validate_parameters=True)
    else:
        value = dict(objective='regression', metric='rmse', device_type='cpu', tree_learner='serial',
            num_threads=1, seed=seed, learning_rate=.1, num_leaves=31, min_data_in_leaf=20,
            lambda_l2=0., linear_tree=config.family == 'P11', deterministic=True,
            force_col_wise=True, bagging_fraction=1., bagging_freq=1, verbosity=-1)
        if config.family == 'P11':
            value['linear_lambda'] = 0.
    value.update(config.parameters)
    return value


_ROOT = Path(__file__).resolve().parents[3]
_THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
_OPENMP_CONTEXT = None


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _source():
    files = {str(path.relative_to(_ROOT)): _sha(path)
        for path in sorted((_ROOT/'src/treeneuron').rglob('*.py'))}
    for name in ('uv.lock', 'locks/research-v2/core.json', 'locks/research-v2/tree-requirements.txt',
                 'locks/research-v2/tree-resolution.json', 'locks/research-v2/tree-imports.py'):
        files[name] = _sha(_ROOT/name)
    return dict(package_sha256=_canonical(files), files=files)


def _verified_installed_source(distribution, relative, path):
    declared = next(x for x in importlib.metadata.distribution(distribution).files if str(x) == relative)
    digest = _sha(path)
    encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')
    if declared.hash.mode != 'sha256' or declared.hash.value != encoded:
        raise ValueError('Upstream import source differs from its locked RECORD')
    return digest


def _dependencies(family):
    """Verify the exact locked overlay before importing the selected native method."""
    global _OPENMP_CONTEXT
    if any(os.environ.get(key) != '1' for key in _THREADS):
        raise ValueError('Native workers require all declared numerical thread variables set to1 before import')
    caller_flag = os.environ.get('KMP_DUPLICATE_LIB_OK')
    if _OPENMP_CONTEXT is None and caller_flag is not None:
        raise ValueError('Caller-supplied OpenMP conflict suppression is not an admitted native treatment')
    if _OPENMP_CONTEXT is not None and caller_flag != _OPENMP_CONTEXT.get(
            'observed_after_runtime_inspection', _OPENMP_CONTEXT['observed_after_native_import']):
        raise ValueError('OpenMP environment changed after the verified native import')
    if _OPENMP_CONTEXT is None and 'sklearn' in sys.modules:
        raise ValueError('Use a fresh method-specific native worker to record its import environment transition')
    package = FAMILIES[family][0]
    distribution = importlib.metadata.distribution(package)
    overlay = Path(distribution._path).resolve().parent
    path = _ROOT/'locks/research-v2/tree-imports.py'
    spec = importlib.util.spec_from_file_location('treeneuron_native_tree_environment', path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    environment = auditor.validate_environment(overlay)
    wheel = next(x for x in json.loads((_ROOT/'locks/research-v2/tree-resolution.json').read_text())['wheels']
                 if x['name'] == package)
    expected_init = (overlay/package/'__init__.py').resolve()
    module_spec = importlib.util.find_spec(package)
    if module_spec is None or module_spec.origin is None or Path(module_spec.origin).resolve() != expected_init:
        raise ValueError('Native module import origin differs from its locked overlay')
    _verified_installed_source(package, f'{package}/__init__.py', expected_init)
    module = importlib.import_module(package)
    if Path(module.__file__).resolve() != expected_init:
        raise ValueError('Imported native module origin differs from its locked overlay')
    if module.__version__ != wheel['version']:
        raise ValueError('Native module version differs from the qualified lock')
    if package == 'catboost':
        native = Path(module._catboost.__file__).resolve()
    elif package == 'xgboost':
        native = Path(module.core._LIB._name).resolve()
    else:
        native = Path(module.basic._LIB._name).resolve()
    if not native.is_relative_to((overlay/package).resolve()):
        raise ValueError('Native library origin escaped its locked package')
    _verified_installed_source(package, str(native.relative_to(overlay)), native)
    if _OPENMP_CONTEXT is None:
        sklearn = sys.modules.get('sklearn')
        init_path = Path(sklearn.__file__).resolve() if sklearn is not None else None
        init_hash = None
        if init_path is not None:
            init_hash = _verified_installed_source('scikit-learn', 'sklearn/__init__.py', init_path)
        _OPENMP_CONTEXT = dict(caller_value=caller_flag,
            observed_after_native_import=os.environ.get('KMP_DUPLICATE_LIB_OK'),
            sklearn_init_path=str(init_path) if init_path is not None else None,
            sklearn_init_sha256=init_hash,
            provenance='Pinned sklearn and threadpoolctl use setdefault for KMP_DUPLICATE_LIB_OK; phase values distinguish native import from runtime inspection. Adapter neither sets nor removes it',
            compatibility_scope='No observed crash/warning under upstream import policy is not compatibility assurance without suppression')
    identity = dict(environment_sha256=_canonical(environment), package=package,
        version=module.__version__, module_path=str(Path(module.__file__).resolve()),
        module_sha256=_sha(module.__file__), native_library_path=str(native),
        native_library_sha256=_sha(native), wheel_sha256=wheel['sha256'], official_source=wheel['source'],
        openmp_environment_context=json.loads(json.dumps(_OPENMP_CONTEXT)),
        interpreter_path=sys.executable, interpreter_sha256=_sha(sys.executable),
        python=sys.version, platform=platform.platform(), machine=platform.machine())
    return module, auditor, identity


def _runtime(auditor):
    global _OPENMP_CONTEXT
    before_inspection = os.environ.get('KMP_DUPLICATE_LIB_OK')
    already_imported = 'threadpoolctl' in sys.modules
    import threadpoolctl
    if 'observed_after_runtime_inspection' not in _OPENMP_CONTEXT:
        inspector_path = Path(threadpoolctl.__file__).resolve()
        _OPENMP_CONTEXT.update(observed_before_runtime_inspection=before_inspection,
            observed_after_runtime_inspection=os.environ.get('KMP_DUPLICATE_LIB_OK'),
            threadpoolctl_previously_imported=already_imported,
            threadpoolctl_path=str(inspector_path), threadpoolctl_sha256=_verified_installed_source(
                'threadpoolctl', 'threadpoolctl.py', inspector_path))
    images = auditor.dyld_images() if sys.platform == 'darwin' else []
    paths = sorted({str(Path(path).resolve()) for path in images if 'libomp' in path.lower() or 'libgomp' in path.lower()})
    return dict(loaded_openmp=[dict(path=path, sha256=_sha(path)) for path in paths],
        threadpools=threadpoolctl.threadpool_info(), imported_torch='torch' in sys.modules,
        thread_environment={key: os.environ.get(key) for key in _THREADS},
        openmp_environment_context=json.loads(json.dumps(_OPENMP_CONTEXT)),
        openmp_disposition='Caller starts without suppression; sklearn or the threadpoolctl inspector introduces its own flag. Bounded observations under that policy do not establish unsuppressed compatibility')


def _peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value*1024


def _target(value, rows):
    target = np.asarray(value, dtype=np.float64)
    if target.shape != (rows,) or not rows or not np.isfinite(target).all():
        raise ValueError('Native target must be a nonempty finite vector matching its role rows')
    return target.copy()


def _native_value_equal(actual, expected):
    if type(expected) in (int, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-12)
        except (ValueError, TypeError):
            return False
    return actual == expected


def _validate_native_configuration(config, requested, resolved):
    """Compare recipe settings with the retained official training config."""
    try:
        if config.family == 'P08':
            names = {'iterations', 'random_seed', 'bootstrap_type', 'subsample', 'random_strength',
                     'loss_function', 'eval_metric', 'task_type', *config.parameters}
            pairs = [(resolved[name], requested[name]) for name in names]
        elif config.family == 'P09':
            learner = resolved['learner']
            generic = learner['generic_param']
            booster = learner['gradient_booster']
            tree = booster['tree_train_param']
            names = {'max_depth', 'eta', 'min_child_weight', 'subsample', 'colsample_bytree',
                     'reg_lambda', 'reg_alpha', *config.parameters}
            pairs = [(tree[name], requested[name]) for name in names]
            pairs += [(generic[name], requested[name]) for name in ('device', 'nthread', 'seed')]
            pairs += [(learner['objective']['name'], requested['objective']),
                      (booster['gbtree_train_param']['tree_method'], 'hist')]
        else:
            pairs = [(resolved[name], item) for name, item in requested.items()]
            pairs.append((resolved['num_iterations'], config.max_rounds))
        if not all(_native_value_equal(actual, expected) for actual, expected in pairs):
            raise ValueError('Saved native configuration differs from its declared recipe parameters')
    except (KeyError, TypeError) as error:
        raise ValueError('Saved native configuration lacks required recipe parameters') from error


def _validate_checkpoint_contract(config, seed, summary):
    requested = native_parameters(config, seed)
    if (type(summary) is not dict or summary.get('family') != config.family
            or summary.get('mode') != config.mode or summary.get('fitted') is not True
            or _canonical(summary.get('requested_parameters')) != _canonical(requested)):
        raise ValueError('Native checkpoint recipe parameters or execution mode differ')
    _validate_native_configuration(config, requested, summary.get('resolved_parameters'))
    selected, executed, history = (summary.get(name) for name in
        ('selected_rounds', 'executed_rounds', 'G_rmse_history'))
    if (type(selected) is not int or type(executed) is not int
            or not 1 <= selected <= executed <= config.max_rounds
            or type(history) is not list or len(history) != executed
            or any(not _number(x, 0, float('inf')) for x in history)
            or selected != int(np.argmin(history))+1):
        raise ValueError('Invalid native checkpoint stopping history or selected round')
    capped = executed == config.max_rounds
    if (summary.get('capped') is not capped
            or summary.get('native_stop') != ('max_rounds' if capped else 'G_patience')
            or not capped and executed-selected != config.patience):
        raise ValueError('Native stopping history does not satisfy its declared recipe')
    budget = summary.get('budget')
    elapsed = summary.get('total_fit_wall_ns')
    if (type(budget) is not dict or not set(budget) <= {'max_seconds'}
            or 'max_seconds' in budget and not _number(budget['max_seconds'], 1e-9, float('inf'))
            or type(elapsed) is not int or elapsed <= 0):
        raise ValueError('Native eligibility lacks its measured execution budget')
    timeout = elapsed > budget.get('max_seconds', float('inf'))*1e9
    eligible = config.mode != 'engineering_fixture' and not timeout
    status = 'timeout' if timeout else 'completed_eligible' if eligible else 'completed_ineligible'
    if (summary.get('checkpoint_eligible') is not eligible or summary.get('time_limit_exceeded') is not timeout
            or summary.get('status') != status or summary.get('native_full_anchor_qualified') is not False):
        raise ValueError('Native checkpoint eligibility contradicts its recipe/budget')


def _failure_domain(error):
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return 'interrupted_recoverable', 'interruption', False
    if isinstance(error, MemoryError) or isinstance(error, OSError) and error.errno in (12, 28):
        return 'resource_exhausted', 'memory_or_storage', False
    if isinstance(error, (FloatingPointError, ArithmeticError)):
        return 'numerical_failure', 'numerical', False
    if isinstance(error, (ImportError, OSError)):
        return 'input_or_dependency_failure', 'infrastructure_or_dependency', True
    if isinstance(error, (TypeError, ValueError)):
        return 'input_or_dependency_failure', 'input_or_native_API', False
    return 'input_or_dependency_failure', 'unclassified_native_exception_requires_diagnosis', False


def _fit_native(module, config, parameters, F, yF, G, yG):
    categories = [i for i, name in enumerate(config.feature_schema['columns'])
                  if name in config.feature_schema['categorical']]
    if config.family == 'P08':
        dF, dG = module.Pool(F, yF, cat_features=categories), module.Pool(G, yG, cat_features=categories)
        assert dF.num_row() == len(F) and dG.num_row() == len(G)
        np.testing.assert_allclose(dF.get_label(), yF, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(dG.get_label(), yG, rtol=1e-6, atol=1e-7)
        model = module.CatBoostRegressor(**parameters)
        model.fit(dF, eval_set=dG, early_stopping_rounds=config.patience, verbose=False)
        history = model.get_evals_result()['validation']['RMSE']
        selected = int(model.get_best_iteration())+1
        assert model.tree_count_ == selected
        resolved = model.get_all_params()
    elif config.family == 'P09':
        bins = parameters.get('max_bin', 256)
        dF = module.QuantileDMatrix(F, yF, max_bin=bins, nthread=1, enable_categorical=True)
        dG = module.QuantileDMatrix(G, yG, max_bin=bins, nthread=1, enable_categorical=True, ref=dF)
        assert dF.num_row() == len(F) and dG.num_row() == len(G)
        np.testing.assert_array_equal(dF.get_label(), yF.astype(np.float32))
        np.testing.assert_array_equal(dG.get_label(), yG.astype(np.float32))
        evaluation = {}
        model = module.train(parameters, dF, num_boost_round=config.max_rounds, evals=[(dG, 'G')],
            callbacks=[module.callback.EarlyStopping(config.patience, data_name='G', metric_name='rmse',
                maximize=False, save_best=True, min_delta=0.)], evals_result=evaluation, verbose_eval=False)
        history = evaluation['G']['rmse']
        selected = int(model.best_iteration)+1
        assert model.num_boosted_rounds() == selected
        resolved = json.loads(model.save_config())
    else:
        categorical_names = [f'f{i}' for i in categories]
        dF = module.Dataset(F, yF, params=parameters, categorical_feature=categorical_names, free_raw_data=False)
        dG = module.Dataset(G, yG, reference=dF, params=parameters,
                           categorical_feature=categorical_names, free_raw_data=False)
        dF.construct()
        dG.construct()
        assert dF.num_data() == len(F) and dG.num_data() == len(G)
        np.testing.assert_array_equal(dF.get_label(), yF.astype(np.float32))
        np.testing.assert_array_equal(dG.get_label(), yG.astype(np.float32))
        evaluation = {}
        model = module.train(parameters, dF, num_boost_round=config.max_rounds,
            valid_sets=[dG], valid_names=['G'], callbacks=[module.early_stopping(config.patience, verbose=False),
                                                       module.record_evaluation(evaluation)])
        history = evaluation['G']['rmse']
        selected = int(model.best_iteration)
        resolved = dict(model.params)
    if not 1 <= selected <= len(history) <= config.max_rounds or not np.isfinite(history).all():
        raise FloatingPointError('Native selected/executed rounds or G history are invalid')
    return model, list(map(float, history)), selected, resolved


def _predict_native(module, family, model, frame, selected):
    if family == 'P08':
        value = model.predict(frame)
    elif family == 'P09':
        value = model.predict(module.DMatrix(frame, nthread=1, enable_categorical=True))
    else:
        value = model.predict(frame, num_iteration=selected, num_threads=1)
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (len(frame),) or not np.isfinite(value).all():
        raise FloatingPointError('Native predictions are nonfinite or have an unexpected shape')
    return value


class NativeTreePredictor:
    def __init__(self, family):
        if family not in FAMILIES:
            raise ValueError('Require a native tree family P08–P11')
        self.family = family
        self._model = None
        self._summary = dict(fitted=False, checkpoint_eligible=False, native_full_anchor_qualified=False)

    def fit(self, request: FitRequest):
        if self._model is not None:
            raise RuntimeError('Use a fresh native predictor for every immutable fit attempt')
        started, utc = time.perf_counter_ns(), time.time_ns()
        directory = Path(request.artifact_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if (directory/'attempt.json').exists() or (directory/'manifest.json').exists():
            raise ValueError('Native attempt artifacts already exist; no silent overwrite')
        phase = 'input_validation'
        try:
            config = NativeTreeConfig.parse(self.family, request.config)
            parameters = native_parameters(config, request.seed)
            if (type(request.budget) is not dict or not set(request.budget) <= {'max_seconds'}
                    or 'max_seconds' in request.budget and not _number(request.budget['max_seconds'], 1e-9, float('inf'))):
                raise ValueError('Only a positive max_seconds wall-time observation budget is supported')
            transform = FeatureTransform.fit(request.X_fit, config.feature_schema, linear=self.family == 'P11')
            F, G = transform.transform(request.X_fit, self.family), transform.transform(request.X_select, self.family)
            yF, yG = _target(request.y_fit, len(F)), _target(request.y_select, len(G))
            source = _source()
            phase = 'dependency_validation'
            module, auditor, dependencies = _dependencies(self.family)
            _write(directory/'manifest.json', dict(schema='research/v2-native-tree-attempt',
                family=self.family, config=config.to_state(), seed=request.seed, budget=request.budget,
                source=source, dependencies=dependencies, F_rows=len(F), G_rows=len(G),
                F_target_sha256=hashlib.sha256(yF.tobytes()).hexdigest(),
                G_target_sha256=hashlib.sha256(yG.tobytes()).hexdigest(),
                F_features_sha256=_canonical(F.astype(object).where(F.notna(), None).values.tolist()),
                G_features_sha256=_canonical(G.astype(object).where(G.notna(), None).values.tolist()),
                accessed_roles=['F', 'G'], transform=transform.to_state(), started_utc_ns=utc))
            phase = 'native_fit'
            native_begin = time.perf_counter_ns()
            model, history, selected, resolved = _fit_native(module, config, parameters, F, yF, G, yG)
            native_ns = time.perf_counter_ns()-native_begin
            phase = 'prediction_validation'
            prediction = _predict_native(module, self.family, model, G, selected)
            actual_rmse = float(np.sqrt(np.mean((prediction-yG)**2)))
            if not math.isclose(actual_rmse, history[selected-1], rel_tol=1e-6, abs_tol=1e-6):
                raise FloatingPointError('Selected native G risk differs from applied-model predictions')
            phase = 'runtime_inspection'
            runtime = _runtime(auditor)
            dependencies['openmp_environment_context'] = json.loads(json.dumps(_OPENMP_CONTEXT))
            elapsed = time.perf_counter_ns()-started
            timed_out = elapsed > request.budget.get('max_seconds', float('inf'))*1e9
            eligible = config.mode != 'engineering_fixture' and not timed_out
            summary = dict(schema='research/v2-native-tree-resources', fitted=True,
                status='timeout' if timed_out else 'completed_eligible' if eligible else 'completed_ineligible',
                family=self.family, mode=config.mode, checkpoint_eligible=eligible,
                native_full_anchor_qualified=False, Gate_D='pending_external_workload_and_native_anchor_qualification',
                F_rows=len(F), G_rows=len(G), transformed_dimensions=F.shape[1],
                selected_rounds=selected, executed_rounds=len(history), G_rmse_history=history,
                G_applied_rmse=actual_rmse, capped=len(history) == config.max_rounds,
                native_stop='max_rounds' if len(history) == config.max_rounds else 'G_patience',
                native_fit_wall_ns=native_ns, total_fit_wall_ns=elapsed,
                process_peak_rss_bytes=_peak_rss(), rss_scope='whole process lifetime peak; not role-incremental',
                hard_resource_enforcement='outer execution harness required', time_limit_exceeded=timed_out,
                budget=dict(request.budget),
                requested_parameters=parameters, resolved_parameters=resolved, runtime=runtime,
                output_units='original_target', categories='F vocabulary; CatBoost reserved unknown / XGBoost-LightGBM missing unknown',
                interrupted_training_resume='not_implemented; deterministic same-seed rerun requires retained paid attempt',
                accessed_roles=['F', 'G'], finished_utc_ns=time.time_ns())
            _validate_checkpoint_contract(config, request.seed, summary)
            phase = 'receipt_publication'
            _write(directory/'attempt.json', summary)
            self._model, self._module, self._transform = model, module, transform
            self._config, self._source, self._dependencies, self._seed = config, source, dependencies, request.seed
            self._summary = summary
            return self
        except BaseException as error:
            status, domain, retry = _failure_domain(error)
            summary = dict(schema='research/v2-native-tree-failure', fitted=False,
                status=status, failure_domain=domain, failure_phase=phase,
                infrastructure_retry_requires_diagnosis=retry,
                family=self.family, checkpoint_eligible=False, native_full_anchor_qualified=False,
                exception=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
                total_fit_wall_ns=time.perf_counter_ns()-started, process_peak_rss_bytes=_peak_rss(),
                started_utc_ns=utc, finished_utc_ns=time.time_ns())
            self._summary = summary
            try:
                _write(directory/'attempt.json', summary)
            except BaseException as receipt_error:
                error.add_note(f'Native failure receipt could not be published: {receipt_error!r}')
            raise

    def predict(self, X, *, batch_size=None):
        if self._model is None:
            raise RuntimeError('Native predictor is not fitted')
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise ValueError('batch_size must be a positive integer or None')
        frame = self._transform.transform(X, self.family)
        if not len(frame):
            return np.empty(0, dtype=np.float64)
        size = batch_size or len(frame)
        return np.concatenate([_predict_native(self._module, self.family, self._model,
            frame.iloc[start:start+size], self._summary['selected_rounds']) for start in range(0, len(frame), size)])

    def resource_summary(self):
        return json.loads(json.dumps(self._summary, allow_nan=False))

    def save(self, path):
        if self._model is None:
            raise RuntimeError('Native predictor is not fitted')
        _validate_checkpoint_contract(self._config, self._seed, self._summary)
        if _source() != self._source:
            raise ValueError('Native source changed since fitting')
        _, _, dependencies = _dependencies(self.family)
        if dependencies != self._dependencies:
            raise ValueError('Native dependencies changed since fitting')
        path = Path(path)
        if path.exists():
            raise ValueError('Native snapshot exists; no silent overwrite')
        path.mkdir(parents=True, exist_ok=False)
        name = 'model.cbm' if self.family == 'P08' else 'model.ubj' if self.family == 'P09' else 'model.txt'
        if self.family in ('P08', 'P09'):
            self._model.save_model(path/name)
        else:
            self._model.save_model(path/name, num_iteration=self._summary['selected_rounds'])
        with (path/name).open('rb') as stream:
            os.fsync(stream.fileno())
        payload = dict(schema='research/v2-native-tree-snapshot', family=self.family,
            config=self._config.to_state(), seed=self._seed, transform=self._transform.to_state(),
            source=self._source, dependencies=self._dependencies, resources=self._summary,
            model_file=name, model_sha256=_sha(path/name))
        _write(path/'state.json', dict(payload=payload, sha256=_canonical(payload)))

    @classmethod
    def load(cls, path, *, expected_state_sha256):
        path = Path(path)
        if (type(expected_state_sha256) is not str or len(expected_state_sha256) != 64
                or any(x not in '0123456789abcdef' for x in expected_state_sha256)
                or _sha(path/'state.json') != expected_state_sha256):
            raise ValueError('Native snapshot differs from the caller trusted state hash')
        envelope = json.loads((path/'state.json').read_text())
        if (type(envelope) is not dict or set(envelope) != {'payload', 'sha256'}
                or _canonical(envelope['payload']) != envelope['sha256']):
            raise ValueError('Native snapshot state identity changed')
        value = envelope['payload']
        fields = {'schema', 'family', 'config', 'seed', 'transform', 'source', 'dependencies',
                  'resources', 'model_file', 'model_sha256'}
        if type(value) is not dict or set(value) != fields or value['schema'] != 'research/v2-native-tree-snapshot':
            raise ValueError('Invalid native snapshot schema')
        config = NativeTreeConfig.parse(value['family'], value['config'])
        native_parameters(config, value['seed'])
        transform = FeatureTransform.from_state(value['transform'])
        if transform.schema != config.feature_schema or transform.linear != (config.family == 'P11'):
            raise ValueError('Native transform differs from the fitted recipe')
        if value['source'] != _source():
            raise ValueError('Native snapshot source identity changed')
        name = 'model.cbm' if config.family == 'P08' else 'model.ubj' if config.family == 'P09' else 'model.txt'
        if value['model_file'] != name or _sha(path/name) != value['model_sha256']:
            raise ValueError('Native model bytes changed or filename escaped snapshot')
        summary = value['resources']
        _validate_checkpoint_contract(config, value['seed'], summary)
        module, auditor, dependencies = _dependencies(config.family)
        _runtime(auditor)
        dependencies['openmp_environment_context'] = json.loads(json.dumps(_OPENMP_CONTEXT))
        if dependencies != value['dependencies']:
            raise ValueError('Native snapshot dependency identity changed')
        if config.family == 'P08':
            model = module.CatBoostRegressor()
            model.load_model(path/name)
            # The pinned loader leaves sklearn's n_features_in_ wrapper field0;
            # feature_names_ comes from the persisted native model instead.
            names, rounds = model.feature_names_, model.tree_count_
        elif config.family == 'P09':
            model = module.Booster(model_file=path/name)
            names, rounds = model.feature_names, model.num_boosted_rounds()
        else:
            model = module.Booster(model_file=path/name)
            names, rounds = model.feature_name(), model.current_iteration()
        if names != [f'f{i}' for i in range(len(config.feature_schema['columns']))] or rounds != summary['selected_rounds']:
            raise ValueError('Native model topology differs from its bound snapshot')
        result = cls(config.family)
        result._model, result._module, result._transform = model, module, transform
        result._config, result._seed, result._source = config, value['seed'], value['source']
        result._dependencies, result._summary = dependencies, summary
        return result
