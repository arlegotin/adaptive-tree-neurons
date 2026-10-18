"""Pinned official P07 TabM with explicit F-only protocol boundaries.

Author modules/licenses are pinned under locks/research-v2/tabm. This adapter
does not qualify full workloads, hard deadlines or interrupted training recovery.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np

from treeneuron.research.native_trees import (_schema, _frame, _category, _number,
    _target, _write, _failure_domain, _peak_rss, _sha, _canonical)

_ROOT = Path(__file__).resolve().parents[3]
_THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
_IMPORT_CONTEXT = None


@dataclass(frozen=True)
class TabMConfig:
    variant: str
    mode: str
    feature_schema: dict[str, Any]
    parameters: dict[str, Any]
    max_epochs: int
    patience: int = 16

    @classmethod
    def parse(cls, value):
        fields = {'schema', 'family', 'variant', 'mode', 'feature_schema', 'parameters'}
        if (type(value) is not dict or value.get('schema') != 'research/v2-tabm-config'
                or value.get('family') != 'P07' or value.get('variant') not in ('plain', 'piecewise')
                or value.get('mode') not in ('native_anchor', 'search', 'engineering_fixture')):
            raise ValueError('Invalid versioned P07 TabM configuration')
        fixture = value['mode'] == 'engineering_fixture'
        if set(value) != fields | ({'max_epochs'} if fixture else set()):
            raise ValueError('Unknown TabM setting or implicit native budget override')
        epochs = value['max_epochs'] if fixture else 1_000_000_000
        if fixture and (type(epochs) is not int or not 1 <= epochs <= 3):
            raise ValueError('Engineering fixtures require at most3 epochs')
        params = value['parameters']
        if type(params) is not dict or value['mode'] == 'native_anchor' and params:
            raise ValueError('Native anchors cannot override official settings')
        allowed = {'n_blocks', 'd_block', 'lr', 'weight_decay', 'dropout'}
        if value['variant'] == 'piecewise':
            allowed |= {'n_bins', 'd_embedding'}
        for name, item in params.items():
            if name not in allowed:
                raise ValueError(f'Unknown or reserved TabM setting: {name}')
            if name == 'n_blocks':
                valid = type(item) is int and item in (2, 3, 4)
            elif name in ('d_block', 'd_embedding', 'n_bins'):
                lower, upper, step = {'d_block': (64, 1024, 16), 'd_embedding': (8, 32, 4),
                    'n_bins': (2, 128, 1)}[name]
                valid = type(item) is int and lower <= item <= upper and (item-lower) % step == 0
            elif name == 'lr':
                valid = _number(item, 1e-4, 5e-3)
            elif name == 'weight_decay':
                valid = _number(item, 0., .1) and (item == 0 or item >= 1e-4)
            else:
                valid = _number(item, 0., .5)
            if not valid:
                raise ValueError(f'TabM setting outside frozen support: {name}')
        return cls(value['variant'], value['mode'], _schema(value['feature_schema']),
            json.loads(json.dumps(params)), epochs)

    def to_state(self):
        value = dict(schema='research/v2-tabm-config', family='P07', variant=self.variant,
            mode=self.mode, feature_schema=self.feature_schema, parameters=self.parameters)
        if self.mode == 'engineering_fixture':
            value['max_epochs'] = self.max_epochs
        return json.loads(json.dumps(value))

    def resolved(self):
        embedded = self.variant == 'piecewise'
        params = dict(n_blocks=2 if embedded else 3, d_block=512, dropout=.1,
            activation='ReLU', k=32, arch_type='tabm',
            start_scaling_init='normal' if embedded else 'random-signs',
            n_bins=48 if embedded else None, d_embedding=16 if embedded else None,
            embedding_activation=False if embedded else None, embedding_version='B' if embedded else None,
            lr=.002, weight_decay=.0003, batch_size=256, eval_batch_size=8096,
            optimizer='AdamW', betas=[.9, .999], eps=1e-8, clip_grad_norm=1.,
            max_epochs=self.max_epochs, patience=self.patience, patience_stop='remaining_less_than_zero',
            scheduler='none', shared_training_batches=True, training_loss='mean_member_MSE',
            selection_score='negative_original_float32_G_RMSE', checkpoint_rule='strict_first_G_best_epoch',
            prediction_order='inverse_each_member_then_mean_float32', mean_fallback='none',
            target_fit_role='F', dtype='float32', device='cpu', AMP=False, compile=False)
        params.update(self.parameters)
        return params


@dataclass(frozen=True)
class TabMTarget:
    mean: float
    scale: float
    constant: bool

    @classmethod
    def fit(cls, value):
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            y = np.asarray(value, dtype=np.float32)
            if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
                raise ValueError('F target requires a finite float32 vector')
            constant = bool(np.all(y == y[0]))
            mean, scale = y.mean().item(), y.std().item()
        if not math.isfinite(mean) or not math.isfinite(scale):
            raise FloatingPointError('Nonfinite/overflow float32 F target statistics')
        if not constant and scale == 0:
            raise FloatingPointError('Nonconstant float32 F target std underflow')
        return cls.from_state(dict(mean=mean, scale=1. if constant else scale,
            constant_float32_F=constant, input_dtype='float32', reduction_dtype='float32'))

    @classmethod
    def from_state(cls, value):
        if (type(value) is not dict or set(value) != {'mean', 'scale', 'constant_float32_F',
                'input_dtype', 'reduction_dtype'} or value['input_dtype'] != 'float32'
                or value['reduction_dtype'] != 'float32' or type(value['constant_float32_F']) is not bool
                or not _number(value['mean'], -np.finfo(np.float32).max, np.finfo(np.float32).max)
                or not _number(value['scale'], float(np.nextafter(np.float32(0), np.float32(1))),
                    float(np.finfo(np.float32).max))
                or value['mean'] != float(np.float32(value['mean']))
                or value['scale'] != float(np.float32(value['scale']))
                or value['constant_float32_F'] and value['scale'] != 1.):
            raise ValueError('Invalid native float32 F target transform')
        return cls(float(value['mean']), float(value['scale']), value['constant_float32_F'])

    def transform(self, value):
        with np.errstate(over='ignore', invalid='ignore'):
            y = (np.asarray(value, dtype=np.float32)-self.mean)/self.scale
        if not np.isfinite(y).all():
            raise FloatingPointError('Nonfinite float32 standardized target')
        return y

    def to_state(self):
        return dict(mean=self.mean, scale=self.scale, constant_float32_F=self.constant,
            input_dtype='float32', reduction_dtype='float32')


@dataclass(frozen=True)
class TabMTransform:
    schema: dict[str, Any]
    vocabularies: tuple[tuple[str, ...], ...]
    numeric_keep: tuple[bool, ...]
    dropped_values: dict[str, float]
    quantiles: np.ndarray
    references: np.ndarray
    fit_rows: int

    @property
    def numeric_names(self):
        names = [x for x in self.schema['columns'] if x not in self.schema['categorical']]
        return [name for name, keep in zip(names, self.numeric_keep, strict=True) if keep]

    @property
    def cardinalities(self):
        return [len(v)+1 for v in self.vocabularies]

    @classmethod
    def fit(cls, X, schema):
        from sklearn.preprocessing import QuantileTransformer
        schema = _schema(schema)
        frame = _frame(X, schema)
        if not 10 <= len(frame) <= 20000:
            raise ValueError('Finite native quantile safety envelope requires10..20000 F rows')
        names = [x for x in schema['columns'] if x not in schema['categorical']]
        vocab = tuple(tuple(sorted({_category(x) for x in frame[name]}))
            for name in schema['columns'] if name in schema['categorical'])
        raw = np.ascontiguousarray(frame[names].to_numpy(dtype=np.float32))
        if not np.isfinite(raw).all():
            raise ValueError('Numerical F features must be finite in float32')
        keep = tuple(bool(x) for x in ~np.all(raw == raw[0], axis=0))
        dropped = {name: float(raw[0, index]) for index, name in enumerate(names) if not keep[index]}
        numeric = np.ascontiguousarray(raw[:, keep])
        if numeric.shape[1]:
            noise = np.random.default_rng(0).normal(0., 1e-5, numeric.shape).astype(np.float32)
            qt = QuantileTransformer(n_quantiles=max(min(len(frame)//30, 1000), 10),
                output_distribution='normal', subsample=1_000_000_000).fit(numeric+noise)
            quantiles, refs = qt.quantiles_.copy(), qt.references_.copy()
        else:
            quantiles, refs = np.empty((0, 0), np.float64), np.empty(0, np.float64)
        return cls(schema, vocab, keep, dropped, quantiles, refs, len(frame))

    def transform(self, X):
        from sklearn.preprocessing import QuantileTransformer
        frame = _frame(X, self.schema)
        names = [x for x in self.schema['columns'] if x not in self.schema['categorical']]
        numeric = np.ascontiguousarray(frame[names].to_numpy(dtype=np.float32))
        if not np.isfinite(numeric).all():
            raise ValueError('Numerical query features must be finite in float32')
        numeric = np.ascontiguousarray(numeric[:, self.numeric_keep])
        if len(frame) and numeric.shape[1]:
            qt = QuantileTransformer(n_quantiles=len(self.references), output_distribution='normal',
                subsample=1_000_000_000)
            qt.n_quantiles_, qt.n_features_in_ = len(self.references), len(self.numeric_names)
            qt.quantiles_, qt.references_ = self.quantiles.copy(), self.references.copy()
            numeric = qt.transform(numeric)
        cats = []
        for name, vocab in zip((x for x in self.schema['columns'] if x in self.schema['categorical']),
                self.vocabularies, strict=True):
            lookup = {x: i for i, x in enumerate(vocab)}
            cats.append([lookup.get(_category(x), len(vocab)) for x in frame[name]])
        categorical = np.asarray(cats, np.int64).T.copy() if cats else np.empty((len(frame), 0), np.int64)
        if not np.isfinite(numeric).all():
            raise FloatingPointError('Nonfinite native quantile output')
        return numeric, categorical

    def to_state(self):
        return dict(schema=self.schema, vocabularies=[list(v) for v in self.vocabularies],
            numeric_keep_mask=list(self.numeric_keep), dropped_numeric_values=self.dropped_values,
            constant_detection_dtype='float32', quantiles=self.quantiles.tolist(),
            references=self.references.tolist(), fit_rows=self.fit_rows,
            quantile_random_state=None, noise_seed=0, noise_std=1e-5,
            input_dtype='float32', noisy_fit_dtype='float32', output_distribution='normal')

    @classmethod
    def from_state(cls, value):
        fields = {'schema', 'vocabularies', 'numeric_keep_mask', 'dropped_numeric_values',
            'constant_detection_dtype', 'quantiles', 'references', 'fit_rows', 'quantile_random_state',
            'noise_seed', 'noise_std', 'input_dtype', 'noisy_fit_dtype', 'output_distribution'}
        if type(value) is not dict or set(value) != fields:
            raise ValueError('Invalid native quantile state fields')
        schema = _schema(value['schema'])
        rows, keep, dropped = value['fit_rows'], value['numeric_keep_mask'], value['dropped_numeric_values']
        names = [x for x in schema['columns'] if x not in schema['categorical']]
        if (type(rows) is not int or not 10 <= rows <= 20000 or type(keep) is not list
                or len(keep) != len(names) or any(type(x) is not bool for x in keep)
                or type(dropped) is not dict or set(dropped) != {x for x, k in zip(names, keep) if not k}
                or any(not _number(x, -float(np.finfo(np.float32).max), float(np.finfo(np.float32).max))
                    or x != float(np.float32(x)) for x in dropped.values())
                or value['quantile_random_state'] is not None or value['noise_seed'] != 0
                or value['noise_std'] != 1e-5 or value['output_distribution'] != 'normal'
                or any(value[x] != 'float32' for x in ('constant_detection_dtype', 'input_dtype', 'noisy_fit_dtype'))):
            raise ValueError('Unsupported F-only native quantile or constant policy')
        vocab = value['vocabularies']
        if (type(vocab) is not list or len(vocab) != len(schema['categorical'])
                or any(type(v) is not list or not v or any(type(x) is not str for x in v)
                    or v != sorted(set(v)) for v in vocab)):
            raise ValueError('Invalid F-only typed category vocabularies')
        q, refs = np.asarray(value['quantiles'], np.float64), np.asarray(value['references'], np.float64)
        count = sum(keep)
        if count:
            n_quantiles = max(min(rows//30, 1000), 10)
            if (q.shape != (n_quantiles, count) or refs.shape != (n_quantiles,)
                    or not np.isfinite(q).all() or np.any(np.diff(q, axis=0) < 0)
                    or not np.array_equal(refs, np.linspace(0, 1, n_quantiles))):
                raise ValueError('Invalid native quantile arrays')
        elif q.size or refs.size:
            raise ValueError('Categorical-only transform cannot contain numeric quantiles')
        else:
            q = q.reshape(0, 0)
        return cls(schema, tuple(tuple(v) for v in vocab), tuple(keep), dict(dropped), q, refs, rows)


def validate_workload(config, fit_rows, select_rows, n_numeric, cardinalities):
    if (type(fit_rows) is not int or not 10 <= fit_rows <= 20000
            or type(select_rows) is not int or not 1 <= select_rows <= 20000
            or type(n_numeric) is not int or not 0 <= n_numeric <= 128
            or len(config.feature_schema['columns']) > 128
            or len(cardinalities) != len(config.feature_schema['categorical'])
            or any(type(c) is not int or c < 2 or c > 4096 for c in cardinalities)
            or sum(cardinalities) > 4096 or not n_numeric+len(cardinalities)):
        raise ValueError('Input exceeds the finite TabM safety envelope or has no surviving inputs')
    if config.variant == 'piecewise':
        if not n_numeric:
            raise ValueError('Piecewise variant requires a surviving numeric column')
        if config.resolved()['n_bins'] >= fit_rows:
            raise ValueError('Native n_bins must be smaller than the complete F count')


@dataclass
class TabMSelection:
    patience: int = 16
    best_loss: float = math.inf
    best_epoch: int = -1
    misses: int = 0

    def observe(self, rmse, epoch):
        if not _number(rmse, 0., float(np.finfo(np.float32).max)):
            raise FloatingPointError('Nonfinite/negative native original-unit G RMSE')
        better = rmse < self.best_loss
        if better:
            self.best_loss, self.best_epoch, self.misses = rmse, epoch, 0
        else:
            self.misses += 1
        return better, self.misses > self.patience


def member_loss(prediction, target):
    import torch
    if prediction.ndim != 2 or target.shape != (len(prediction),):
        raise ValueError('Require row/member prediction matrix and one target per row')
    return torch.nn.functional.mse_loss(prediction.flatten(0, 1),
        target.repeat_interleave(prediction.shape[1]))


def original_mean(prediction, target):
    values = np.asarray(prediction)
    if values.dtype != np.float32 or values.ndim != 2:
        raise ValueError('Native member predictions must be a float32 matrix')
    result = (values*target.scale+target.mean).mean(1)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite original-unit native prediction')
    return result


def _source():
    files = {str(p.relative_to(_ROOT)): _sha(p) for p in sorted((_ROOT/'src/treeneuron').rglob('*.py'))}
    pins = _ROOT/'locks/research-v2/tabm'
    lock = json.loads((pins/'source-lock.json').read_text())
    for row in lock['sources']:
        path = pins/row['local']
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if _sha(path) != row['sha256'] or blob != row['git_blob_sha1']:
            raise ValueError('Pinned official TabM/embedding source bytes changed')
    for path in sorted(pins.rglob('*')):
        if path.is_file():
            files[str(path.relative_to(_ROOT))] = _sha(path)
    for name in ('locks/research-v2/core.json', 'uv.lock'):
        files[name] = _sha(_ROOT/name)
    return dict(package_sha256=_canonical(files), files=files)


def _dependencies():
    """Exact frozen core identities, all overlay bytes, actual import origins."""
    if not sys.dont_write_bytecode:
        raise ValueError('TabM worker requires bytecode writes disabled before native imports: PYTHONDONTWRITEBYTECODE=1 or -B')
    from treeneuron.research.ordinary_neural import _record_file
    global _IMPORT_CONTEXT
    if any(os.environ.get(x) != '1' for x in _THREADS):
        raise ValueError('All numerical thread variables must be1 before native imports')
    before = os.environ.get('KMP_DUPLICATE_LIB_OK')
    if _IMPORT_CONTEXT is None and ('sklearn' in sys.modules or before is not None):
        raise ValueError('Use a fresh worker without caller OpenMP suppression or preimported sklearn')
    pins = _ROOT/'locks/research-v2/tabm'
    core = json.loads((_ROOT/'locks/research-v2/core.json').read_text())
    lock = json.loads((pins/'source-lock.json').read_text())
    install = json.loads((pins/'installed-files.json').read_text())
    overlay = (_ROOT/'environment/research-v2-tabm-site').resolve()
    actual = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata['Name'].lower().replace('_', '-')
        if name in actual:
            raise ValueError(f'Duplicate distribution identity: {name}')
        actual[name] = dist
    expected = {x['name'].lower().replace('_', '-'): x for x in core['distributions']}
    extras = {x['package']: x for x in lock['wheels']}
    if set(actual) != set(expected) | set(extras):
        raise ValueError('TabM worker requires exactly the27 frozen core and two pinned overlay packages')
    for name, row in expected.items():
        dist = actual[name]
        if (dist.version != row['version'] or str(Path(dist._path).resolve()) != row['path']
                or _sha(Path(dist._path)/'RECORD') != row['record_sha256']):
            raise ValueError(f'Frozen core distribution changed: {name}')
    owned = set(install['ignored_installer_files'])
    for name, row in extras.items():
        dist = actual[name]
        path = Path(dist._path).resolve()
        if dist.version != row['version'] or path.parent != overlay:
            raise ValueError('TabM dependency escaped its pinned overlay/version')
        entries = install['distributions'].get(path.name)
        if entries is None or _sha(path/'RECORD') != entries['record_sha256']:
            raise ValueError('TabM installed overlay RECORD changed')
        if _sha(path/'METADATA') != row['metadata_sha256']:
            raise ValueError('TabM installed metadata differs from the official wheel')
        for entry in entries['files']:
            owned.add(entry['path'])
            file = (overlay/entry['path']).resolve()
            if (not file.is_relative_to(overlay) or not file.is_file()
                    or file.stat().st_size != entry['bytes'] or _sha(file) != entry['sha256']):
                raise ValueError('TabM installed overlay bytes differ from the verified wheel/installation')
    installed = {str(p.relative_to(overlay)) for p in overlay.rglob('*') if p.is_file()}
    if installed != owned:
        raise ValueError('Unowned or missing files in the isolated TabM overlay')
    imported, files = {}, []
    for package, distribution in (('torch', 'torch'), ('numpy', 'numpy'), ('sklearn', 'scikit-learn'),
            ('rtdl_num_embeddings', 'rtdl-num-embeddings'), ('tabm', 'tabm')):
        dist = actual[distribution]
        relative = package+'.py' if distribution in extras else package+'/__init__.py'
        origin = Path(dist.locate_file(relative)).resolve()
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != origin:
            raise ValueError(f'Actual native module origin differs from qualified installation: {package}')
        files.append(_record_file(distribution, origin))
        mod = importlib.import_module(package)
        if Path(mod.__file__).resolve() != origin:
            raise ValueError('Native module selected by import escaped its qualified origin')
        imported[package] = mod
    for name, row in extras.items():
        mod = imported['rtdl_num_embeddings' if name == 'rtdl-num-embeddings' else name]
        if _sha(mod.__file__) != row['module_sha256']:
            raise ValueError('Imported TabM/embedding module differs from pinned author source')
    torch = imported['torch']
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    files += [_record_file('torch', torch._C.__file__),
        _record_file('numpy', importlib.import_module('numpy._core._multiarray_umath').__file__),
        _record_file('scikit-learn', importlib.import_module('sklearn.preprocessing._data').__file__)]
    if _IMPORT_CONTEXT is None:
        _IMPORT_CONTEXT = dict(caller_KMP_DUPLICATE_LIB_OK=before,
            post_import_KMP_DUPLICATE_LIB_OK=os.environ.get('KMP_DUPLICATE_LIB_OK'),
            upstream_policy='Pinned sklearn/threadpoolctl use setdefault; adapter never sets/removes/preloads it',
            compatibility='Observed execution under upstream suppression does not assure unsuppressed compatibility')
    elif before != _IMPORT_CONTEXT['post_import_KMP_DUPLICATE_LIB_OK']:
        raise ValueError('Native import environment changed after qualification')
    return dict(core_manifest_sha256=_sha(_ROOT/'locks/research-v2/core.json'),
        overlay_manifest_sha256=_sha(pins/'installed-files.json'), overlay=str(overlay),
        executable=sys.executable, executable_sha256=_sha(sys.executable), python=sys.version,
        platform=platform.platform(), core_distribution_count=len(expected), overlay_distribution_count=len(extras),
        bytecode_writes_disabled=True,
        package_versions={k: d.version for k, d in sorted(actual.items())}, imported_files=files,
        import_environment=copy.deepcopy(_IMPORT_CONTEXT),
        validation_scope='All27 core versions/locations/RECORD digests, every overlay byte, selected actual imported/native core bytes')


def _runtime():
    from treeneuron.research.ordinary_neural import _runtime as inspect
    return inspect()


def _bytes(value):
    return hashlib.sha256(value).hexdigest()


def _state_identity(model):
    return _canonical({name: dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
        sha256=_bytes(tensor.detach().cpu().numpy().tobytes()))
        for name, tensor in sorted(model.state_dict().items())})


def _validated_bins(config, n_numeric, bins):
    if config.variant == 'plain':
        if bins != []:
            raise ValueError('Plain TabM cannot contain numerical embedding bins')
        return []
    if type(bins) is not list or len(bins) != n_numeric:
        raise ValueError('Native bins differ from the retained numeric column count')
    result = []
    for edges in bins:
        values = np.asarray(edges, np.float32)
        if (values.ndim != 1 or not 2 <= len(values) <= config.resolved()['n_bins']+1
                or not np.isfinite(values).all() or np.any(np.diff(values) <= 0)
                or values.tolist() != edges):
            raise ValueError('Native bins require strictly increasing exact float32 edges')
        result.append(values)
    return result


def build_model(config, n_numeric, cardinalities, bins):
    import torch
    import tabm
    import rtdl_num_embeddings as numemb
    values = _validated_bins(config, n_numeric, bins)
    p = config.resolved()
    embeddings = numemb.PiecewiseLinearEmbeddings([torch.from_numpy(x.copy()) for x in values],
        p['d_embedding'], activation=False, version='B') if config.variant == 'piecewise' else None
    return tabm.TabM.make(n_num_features=n_numeric, cat_cardinalities=cardinalities, d_out=1,
        num_embeddings=embeddings, n_blocks=p['n_blocks'], d_block=p['d_block'], dropout=p['dropout'],
        activation='ReLU', k=32, arch_type='tabm', start_scaling_init=p['start_scaling_init'])


def _members(model, numeric, categorical, size):
    import torch
    model.eval()
    with torch.no_grad():
        parts = [model(torch.from_numpy(numeric[i:i+size]) if numeric.shape[1] else None,
            torch.from_numpy(categorical[i:i+size]) if categorical.shape[1] else None).squeeze(-1).float()
            for i in range(0, len(numeric), size)]
        # The native head emits column-major tensors. torch.cat before NumPy
        # preserves the notebook's row-major float32 mean reduction order.
        return torch.cat(parts).cpu().numpy() if parts else np.empty((0, 32), np.float32)


def _train_model(model, config, F, yF, G, yG, target):
    import torch
    from sklearn.metrics import mean_squared_error
    p = config.resolved()
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    numeric, categorical = F
    train_num, train_cat, labels = torch.from_numpy(numeric), torch.from_numpy(categorical), torch.from_numpy(yF)
    tracker, history, orders, best_weights, updates = TabMSelection(config.patience), [], [], None, 0
    initial_rng = _bytes(torch.get_rng_state().numpy().tobytes())
    for epoch in range(config.max_epochs):
        order = torch.randperm(len(yF))
        loss_sum, norms = 0., []
        for rows in order.split(256):
            model.train()
            optimizer.zero_grad()
            prediction = model(train_num[rows] if train_num.shape[1] else None,
                train_cat[rows] if train_cat.shape[1] else None).squeeze(-1).float()
            loss = member_loss(prediction, labels[rows])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite native TabM member training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite native TabM training gradient')
            norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite native TabM gradient norm')
            norms.append(float(norm))
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite native TabM updated parameter')
            loss_sum += loss.item()*len(rows)
            updates += 1
        row_ids = order.numpy().copy()
        if not np.array_equal(np.sort(row_ids), np.arange(len(yF))):
            raise RuntimeError('Native TabM permutation must cover every F row exactly once')
        orders.append(row_ids)
        prediction = original_mean(_members(model, *G, 8096), target)
        rmse = float(mean_squared_error(yG, prediction)**.5)
        improved, stop = tracker.observe(rmse, epoch)
        history.append(dict(epoch=epoch, F_online_member_mse=loss_sum/len(yF), G_rmse=rmse,
            applied_updates=updates, row_order_sha256=_bytes(row_ids.tobytes()),
            gradient_norm_min=min(norms), gradient_norm_max=max(norms)))
        if improved:
            best_weights = copy.deepcopy(model.state_dict())
        if stop:
            break
    final_rng = _bytes(torch.get_rng_state().numpy().tobytes())
    model.load_state_dict(best_weights, strict=True)
    model.eval()
    return dict(history=history, selected_epoch=tracker.best_epoch, executed_epochs=len(history),
        applied_updates=updates, native_stop='G_patience_17_misses' if tracker.misses > config.patience else 'max_epochs',
        full_F_coverage_each_epoch=True, optimizer_defaults=json.loads(json.dumps(optimizer.defaults)),
        initial_training_rng_sha256=initial_rng, final_training_rng_sha256=final_rng), np.stack(orders)


def _prediction_identity(summary):
    return _canonical({key: summary[key] for key in ('selected_native_state_sha256',
        'feature_transform_sha256', 'target_transform_sha256', 'bins_sha256', 'selected_epoch')})


def _validate_summary(config, summary):
    if (summary['family'] != 'P07' or summary['mode'] != config.mode or summary['variant'] != config.variant
            or summary['config_sha256'] != _canonical(config.to_state())
            or summary['resolved_parameters'] != config.resolved()):
        raise ValueError('TabM snapshot configuration differs from its fitted native contract')
    p = config.resolved()
    optimizer = dict(lr=p['lr'], weight_decay=p['weight_decay'], betas=[.9, .999], eps=1e-8,
        amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False,
        fused=None, decoupled_weight_decay=True)
    if summary['optimizer_defaults'] != optimizer:
        raise ValueError('TabM native optimizer defaults differ from the pinned recipe')
    validate_workload(config, summary['fit_rows'], summary['select_rows'],
        summary['n_numeric'], summary['cardinalities'])
    history = summary['history']
    if (type(history) is not list or not history or len(history) > config.max_epochs
            or len(history) != summary['executed_epochs']):
        raise ValueError('Invalid native TabM history')
    tracker, stopped = TabMSelection(config.patience), False
    updates = math.ceil(summary['fit_rows']/256)
    for epoch, row in enumerate(history):
        if (stopped or row['epoch'] != epoch or row['applied_updates'] != updates*(epoch+1)
                or not _number(row['F_online_member_mse'], 0., 1e300)
                or not _number(row['G_rmse'], 0., 1e300)
                or not _number(row['gradient_norm_min'], 0., 1e300)
                or not _number(row['gradient_norm_max'], row['gradient_norm_min'], 1e300)):
            raise ValueError('Invalid native TabM loss/update history')
        _, stopped = tracker.observe(row['G_rmse'], epoch)
    if (summary['selected_epoch'] != tracker.best_epoch or summary['applied_updates'] != len(history)*updates
            or summary['native_stop'] != ('G_patience_17_misses' if stopped else 'max_epochs')
            or not stopped and len(history) != config.max_epochs):
        raise ValueError('TabM stopping/selected epoch does not follow the native treatment')
    eligible = config.mode != 'engineering_fixture' and not summary['time_limit_exceeded']
    status = 'timeout' if summary['time_limit_exceeded'] else 'completed_eligible' if eligible else 'completed_ineligible'
    if (summary['checkpoint_eligible'] != eligible or summary['status'] != status
            or summary['quality_eligible'] is not False or summary['native_full_anchor_qualified'] is not False
            or summary['selected_prediction_state_sha256'] != _prediction_identity(summary)):
        raise ValueError('TabM fitted prediction identity/mode eligibility mismatch')


class TabMPredictor:
    def __init__(self):
        self._model, self._summary = None, {}

    def fit(self, request):
        if self._model is not None:
            raise RuntimeError('Use a fresh TabM predictor for each immutable fit attempt')
        started = time.perf_counter_ns()
        root = Path(request.artifact_dir)
        root.mkdir(parents=True, exist_ok=False)
        phase = 'input_validation'
        try:
            config = TabMConfig.parse(request.config)
            if type(request.seed) is not int or not 0 <= request.seed < 2**63:
                raise ValueError('TabM model/sampler seed requires a nonnegative63-bit integer')
            if (type(request.budget) is not dict or not set(request.budget) <= {'max_seconds'}
                    or 'max_seconds' in request.budget and not _number(request.budget['max_seconds'], 1e-9, 1e12)):
                raise ValueError('Only a positive observed max_seconds budget is supported')
            if (not 10 <= len(request.X_fit) <= 20000 or not 1 <= len(request.X_select) <= 20000
                    or len(config.feature_schema['columns']) > 128):
                raise ValueError('Inputs exceed finite TabM safety envelope before preprocessing')
            yF = _target(request.y_fit, len(request.X_fit)).astype(np.float32)
            yG = _target(request.y_select, len(request.X_select)).astype(np.float32)
            if not np.isfinite(yF).all() or not np.isfinite(yG).all():
                raise ValueError('Targets must remain finite after native float32 conversion')
            source = _source()
            phase = 'dependency_validation'
            dependency_started = time.perf_counter_ns()
            dependencies = _dependencies()
            dependency_ns = time.perf_counter_ns()-dependency_started
            phase = 'F_preprocessing'
            transform = TabMTransform.fit(request.X_fit, config.feature_schema)
            target = TabMTarget.fit(yF)
            F, G = transform.transform(request.X_fit), transform.transform(request.X_select)
            validate_workload(config, len(yF), len(yG), F[0].shape[1], transform.cardinalities)
            import torch
            import rtdl_num_embeddings as numemb
            bins = [x.tolist() for x in numemb.compute_bins(torch.from_numpy(F[0]),
                n_bins=config.resolved()['n_bins'])] if config.variant == 'piecewise' else []
            _write(root/'input-manifest.json', dict(schema='research/v2-tabm-attempt', source=source,
                dependencies=dependencies, seed=request.seed, config=config.to_state(),
                resolved_parameters=config.resolved(), feature_transform=transform.to_state(),
                target_transform=target.to_state(), bins=bins, budget=request.budget,
                F_numeric_sha256=_bytes(F[0].tobytes()), F_categorical_sha256=_bytes(F[1].tobytes()),
                G_numeric_sha256=_bytes(G[0].tobytes()), G_categorical_sha256=_bytes(G[1].tobytes()),
                F_target_sha256=_bytes(yF.tobytes()), G_target_sha256=_bytes(yG.tobytes()),
                fit_rows=len(yF), select_rows=len(yG), accessed_roles=['F', 'G']))
            phase = 'native_training'
            native_started = time.perf_counter_ns()
            with torch.random.fork_rng():
                torch.manual_seed(request.seed)
                model = build_model(config, F[0].shape[1], transform.cardinalities, bins)
                if sum(p.numel() for p in model.parameters()) > 10_000_000:
                    raise ValueError('TabM exceeds the finite10M-parameter safety envelope')
                training, orders = _train_model(model, config, F, target.transform(yF), G, yG, target)
            native_ns = time.perf_counter_ns()-native_started
            if source != _source():
                raise ValueError('Scientific source changed during native TabM fitting')
            phase = 'runtime_inspection'
            runtime = _runtime()
            np.savez(root/'F-row-orders.npz', row_indices=orders)
            elapsed = time.perf_counter_ns()-started
            timed_out = elapsed > request.budget.get('max_seconds', math.inf)*1e9
            eligible = config.mode != 'engineering_fixture' and not timed_out
            summary = dict(schema='research/v2-tabm-resources', family='P07', mode=config.mode,
                variant=config.variant, fitted=True, config_sha256=_canonical(config.to_state()),
                resolved_parameters=config.resolved(), seed=request.seed, fit_rows=len(yF), select_rows=len(yG),
                n_numeric=F[0].shape[1], cardinalities=transform.cardinalities, dependencies=dependencies,
                status='timeout' if timed_out else 'completed_eligible' if eligible else 'completed_ineligible',
                checkpoint_eligible=eligible, quality_eligible=False, native_full_anchor_qualified=False,
                Gate_D='pending_full_native_anchor_and_external_workload_qualification',
                native_training_wall_ns=native_ns, dependency_import_wall_ns=dependency_ns,
                total_fit_wall_ns=elapsed, process_peak_rss_bytes=_peak_rss(),
                rss_scope='whole process lifetime peak, not role-incremental', runtime=runtime,
                time_limit_exceeded=timed_out, budget=dict(request.budget),
                hard_resource_enforcement='outer worker harness required; max_seconds here is post-fit observation',
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                row_orders_sha256=_sha(root/'F-row-orders.npz'), accessed_roles=['F', 'G'],
                feature_transform_sha256=_canonical(transform.to_state()), target_transform_sha256=_canonical(target.to_state()),
                bins_sha256=_canonical(bins), selected_native_state_sha256=_state_identity(model),
                interrupted_training_resume='not_implemented; deterministic same-seed rerun retains all paid attempts',
                **training)
            summary['selected_prediction_state_sha256'] = _prediction_identity(summary)
            _validate_summary(config, summary)
            phase = 'receipt_publication'
            _write(root/'attempt.json', summary)
            self._model, self._summary, self._config, self._seed = model, summary, config, request.seed
            self._source, self._dependencies, self._transform, self._target, self._bins = source, dependencies, transform, target, bins
            return self
        except BaseException as error:
            status, domain, retry = _failure_domain(error)
            self._summary = dict(status=status, failure_domain=domain, infrastructure_retry_permitted=retry,
                checkpoint_eligible=False, exception=type(error).__name__, message=str(error),
                traceback=traceback.format_exc(), failure_phase=phase,
                total_fit_wall_ns=time.perf_counter_ns()-started, process_peak_rss_bytes=_peak_rss())
            try:
                _write(root/'attempt.json', self._summary)
            except BaseException as receipt_error:
                error.add_note(f'Failure receipt could not be published: {receipt_error!r}')
            raise

    def predict(self, X, *, batch_size=None):
        if self._model is None:
            raise RuntimeError('TabM predictor must be fitted before prediction')
        size = 8096 if batch_size is None else batch_size
        if type(size) is not int or not 1 <= size <= 65536:
            raise ValueError('Prediction batch size must be a positive bounded integer')
        num, cat = self._transform.transform(X)
        return original_mean(_members(self._model, num, cat, size), self._target).astype(np.float64)

    def resource_summary(self):
        return copy.deepcopy(self._summary)

    def save(self, path):
        if self._model is None:
            raise RuntimeError('TabM predictor must be fitted before saving')
        if self._source != _source():
            raise ValueError('TabM scientific source changed after fitting')
        _validate_summary(self._config, self._summary)
        if (self._summary['selected_native_state_sha256'] != _state_identity(self._model)
                or self._summary['feature_transform_sha256'] != _canonical(self._transform.to_state())
                or self._summary['target_transform_sha256'] != _canonical(self._target.to_state())
                or self._summary['bins_sha256'] != _canonical(self._bins)):
            raise ValueError('TabM prediction state changed after the fitted selection')
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        state = {k: v.detach().cpu().numpy().copy() for k, v in self._model.state_dict().items()}
        np.savez(path/'weights.npz', **state)
        _write(path/'manifest.json', dict(schema='research/v2-tabm-snapshot', config=self._config.to_state(),
            seed=self._seed, source=self._source, dependencies=self._dependencies, feature_transform=self._transform.to_state(),
            target_transform=self._target.to_state(), bins=self._bins, resources=self._summary,
            weights_sha256=_sha(path/'weights.npz'), payload={k: dict(dtype=str(v.dtype), shape=list(v.shape)) for k, v in state.items()},
            snapshot_scope='portable native prediction state; optimizer/sampler training recovery not implemented'))

    @classmethod
    def load(cls, path, *, expected_manifest_sha256):
        if (type(expected_manifest_sha256) is not str or len(expected_manifest_sha256) != 64
                or any(x not in '0123456789abcdef' for x in expected_manifest_sha256)):
            raise ValueError('TabM loading requires an external trusted manifest SHA256')
        path = Path(path)
        if _sha(path/'manifest.json') != expected_manifest_sha256:
            raise ValueError('TabM snapshot differs from its external trusted manifest hash')
        value = json.loads((path/'manifest.json').read_text())
        if (set(value) != {'schema', 'config', 'seed', 'source', 'dependencies', 'feature_transform',
                'target_transform', 'bins', 'resources', 'weights_sha256', 'payload', 'snapshot_scope'}
                or value['schema'] != 'research/v2-tabm-snapshot'):
            raise ValueError('Unknown TabM snapshot schema/fields')
        config = TabMConfig.parse(value['config'])
        source, dependencies = _source(), _dependencies()
        if value['source'] != source or value['dependencies'] != dependencies:
            raise ValueError('TabM snapshot source/dependency identity differs from the qualified worker')
        summary = value['resources']
        from treeneuron.research.ordinary_neural import _runtime_identity
        runtime = _runtime()
        if _runtime_identity(runtime) != _runtime_identity(summary['runtime']):
            raise ValueError('Actual TabM native images/bytes/thread treatment differ from the retained runtime')
        _validate_summary(config, summary)
        transform, target = TabMTransform.from_state(value['feature_transform']), TabMTarget.from_state(value['target_transform'])
        if (transform.schema != config.feature_schema or transform.fit_rows != summary['fit_rows']
                or len(transform.numeric_names) != summary['n_numeric'] or transform.cardinalities != summary['cardinalities']
                or type(value['seed']) is not int or not 0 <= value['seed'] < 2**63 or value['seed'] != summary['seed']):
            raise ValueError('TabM snapshot fitted schema/seed/shapes disagree')
        bins = value['bins']
        _validated_bins(config, summary['n_numeric'], bins)
        if (summary['feature_transform_sha256'] != _canonical(transform.to_state())
                or summary['target_transform_sha256'] != _canonical(target.to_state())
                or summary['bins_sha256'] != _canonical(bins)):
            raise ValueError('TabM preprocessing/bins differ from the fitted prediction state')
        if _sha(path/'weights.npz') != value['weights_sha256']:
            raise ValueError('TabM native tensor payload hash mismatch')
        import torch
        with torch.random.fork_rng():
            model = build_model(config, summary['n_numeric'], transform.cardinalities, bins)
        expected, buffers = model.state_dict(), dict(model.named_buffers())
        with np.load(path/'weights.npz', allow_pickle=False) as archive:
            if set(archive.files) != set(expected) or set(value['payload']) != set(expected):
                raise ValueError('TabM tensor names differ from the native architecture')
            state = {}
            for name, tensor in expected.items():
                actual = archive[name]
                if (actual.dtype != tensor.numpy().dtype or actual.shape != tuple(tensor.shape)
                        or not np.isfinite(actual).all()
                        or value['payload'][name] != dict(dtype=str(actual.dtype), shape=list(actual.shape))):
                    raise ValueError('TabM tensor dtype/shape/finiteness mismatch; implicit casts forbidden')
                if name in buffers and not np.array_equal(actual, tensor.numpy()):
                    raise ValueError('Native embedding/category buffer differs from its retained constructor state')
                state[name] = torch.from_numpy(actual.copy())
        model.load_state_dict(state, strict=True)
        if _state_identity(model) != summary['selected_native_state_sha256']:
            raise ValueError('TabM weights differ from the fitted selected checkpoint')
        model.eval()
        result = cls()
        result._model, result._summary, result._config, result._seed = model, summary, config, value['seed']
        result._source, result._dependencies, result._transform, result._target, result._bins = source, dependencies, transform, target, bins
        return result
