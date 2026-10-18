"""Versioned P03/P04 recipe with F-only preprocessing and lazy native imports.

Tiny fixtures do not qualify production workloads, hard process resource limits,
or interrupted-training recovery. Author-source notices live in the source lock.
Architecture/preprocessing adapted from Apache2.0-licensed PyTabKit/RTDL; exact
source bytes, commit identities and licenses are in locks/research-v2/ordinary-neural.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import base64
import ctypes
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
class OrdinaryNeuralConfig:
    family: str
    mode: str
    feature_schema: dict[str, Any]
    parameters: dict[str, Any]
    max_epochs: int
    patience: int

    @classmethod
    def parse(cls, family, value):
        fields = {'schema', 'family', 'mode', 'feature_schema', 'parameters'}
        if (family not in ('P03', 'P04') or type(value) is not dict
                or value.get('schema') != 'research/v2-ordinary-neural-config'
                or value.get('family') != family
                or value.get('mode') not in ('native_anchor', 'search', 'engineering_fixture')):
            raise ValueError('Invalid versioned P03/P04 configuration')
        fixture = value['mode'] == 'engineering_fixture'
        if set(value) != fields | ({'max_epochs', 'patience'} if fixture else set()):
            raise ValueError('Unknown setting or implicit native budget override')
        epochs, patience = (value['max_epochs'], value['patience']) if fixture else (1000, 20)
        if fixture and (type(epochs) is not int or not 1 <= epochs <= 8
                or type(patience) is not int or not 1 <= patience <= 4):
            raise ValueError('Engineering fixtures require at most8 epochs/patience4')
        params = value['parameters']
        if type(params) is not dict or value['mode'] == 'native_anchor' and params:
            raise ValueError('Native anchors cannot override published settings')
        allowed = {'lr', 'weight_decay', 'batch_size'} | ({'hidden_dims', 'activation', 'dropout'}
            if family == 'P03' else {'blocks', 'width', 'hidden_dropout', 'residual_dropout'})
        for name, item in params.items():
            if name not in allowed:
                raise ValueError(f'Unknown native setting: {name}')
            if name == 'hidden_dims':
                valid = type(item) is list and (item == [128, 256, 128] or
                    1 <= len(item) <= 3 and all(type(x) is int and x in (64, 128, 256, 512) for x in item)
                    and len(set(item)) == 1)
            elif name == 'activation':
                valid = item in ('ReLU', 'GELU')
            elif name in ('dropout', 'hidden_dropout', 'residual_dropout'):
                valid = type(item) in (int, float) and item in ((0, .1, .2, .25)
                    if name == 'hidden_dropout' else (0, .1, .2))
            elif name in ('batch_size', 'blocks', 'width'):
                valid = type(item) is int and item in {'batch_size': (128, 256),
                    'blocks': (2, 3, 4), 'width': (128, 256, 512)}[name]
            elif name == 'weight_decay':
                valid = _number(item, 0., 1e-2) and (item == 0 or item >= 1e-7)
            else:
                valid = _number(item, 1e-4, 1e-2)
            if not valid:
                raise ValueError(f'Native setting outside frozen support: {name}')
        candidate = cls(family, value['mode'], _schema(value['feature_schema']),
            json.loads(json.dumps(params)), epochs, patience)
        resolved = candidate.resolved()
        exceptional = (resolved['hidden_dims'] == [128, 256, 128] if family == 'P03'
            else resolved['hidden_dropout'] == .25)
        if exceptional:
            anchor = cls(family, 'native_anchor', candidate.feature_schema, {}, epochs, patience).resolved()
            if resolved != anchor:
                raise ValueError('Exceptional coordinates belong only to the complete published anchor')
        return candidate

    def to_state(self):
        value = dict(schema='research/v2-ordinary-neural-config', family=self.family,
            mode=self.mode, feature_schema=self.feature_schema, parameters=self.parameters)
        if self.mode == 'engineering_fixture':
            value.update(max_epochs=self.max_epochs, patience=self.patience)
        return json.loads(json.dumps(value))

    def resolved(self):
        params = dict(lr=.001, weight_decay=.01, batch_size=128, embedding_dim=8,
            max_epochs=self.max_epochs, patience=self.patience,
            progress_relative_threshold=1e-4, checkpoint_rule='strict_G_best_epoch',
            scheduler='none', optimizer='AdamW', betas=[.9, .999], eps=1e-8,
            quantile_random_state=0, mean_fallback='G_selected_F_mean',
            dtype='float32', device='cpu', target_fit_role='F')
        if self.family == 'P03':
            params.update(hidden_dims=[256, 256, 256] if self.mode == 'search' else [128, 256, 128],
                activation='ReLU', dropout=.1)
        else:
            params.update(blocks=2, width=128, expansion=2, normalization='BatchNorm',
                activation='ReLU', hidden_dropout=.1 if self.mode == 'search' else .25, residual_dropout=.1)
        params.update(self.parameters)
        return params


@dataclass(frozen=True)
class TargetTransform:
    mean: float
    scale: float

    @classmethod
    def fit(cls, value):
        from sklearn.preprocessing import StandardScaler
        y = np.asarray(value, dtype=np.float64)
        if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
            raise ValueError('F target requires a nonempty finite vector')
        scaler = StandardScaler().fit(y[:, None])
        return cls.from_state(dict(mean=float(scaler.mean_[0]), scale=float(scaler.scale_[0])))

    @classmethod
    def from_state(cls, value):
        if (type(value) is not dict or set(value) != {'mean', 'scale'}
                or not _number(value['mean'], -1e300, 1e300)
                or not _number(value['scale'], 1e-300, 1e300)):
            raise ValueError('Invalid F target transform')
        return cls(float(value['mean']), float(value['scale']))

    def transform(self, value):
        result = ((np.asarray(value, dtype=np.float64)-self.mean)/self.scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite transformed target')
        return result

    def inverse(self, value):
        return np.asarray(value, dtype=np.float64)*self.scale+self.mean

    def to_state(self):
        return dict(mean=self.mean, scale=self.scale)


@dataclass(frozen=True)
class NeuralTransform:
    schema: dict[str, Any]
    vocabularies: tuple[tuple[str, ...], ...]
    quantiles: np.ndarray
    references: np.ndarray
    fit_rows: int

    @property
    def cardinalities(self):
        return [len(v)+1 for v in self.vocabularies]

    @classmethod
    def fit(cls, X, schema):
        from sklearn.preprocessing import QuantileTransformer
        schema = _schema(schema)
        frame = _frame(X, schema)
        if len(frame) < 10:
            raise ValueError('Native quantile workload requires at least10 F rows')
        names = [x for x in schema['columns'] if x not in schema['categorical']]
        vocab = tuple(tuple(sorted({_category(x) for x in frame[name]}))
            for name in schema['columns'] if name in schema['categorical'])
        if names:
            numeric = np.ascontiguousarray(frame[names].to_numpy(dtype=np.float32))
            if not np.isfinite(numeric).all():
                raise ValueError('Numerical F features must be finite')
            # Exact selected author quantile_tabr noise, including float64 promotion.
            noise_std = 1e-3 / np.maximum(np.std(numeric, axis=0, keepdims=True), 1e-3)
            noisy = numeric + noise_std * np.random.default_rng(0).standard_normal(numeric.shape)
            qt = QuantileTransformer(output_distribution='normal',
                n_quantiles=max(min(len(frame)//30, 1000), 10),
                subsample=1_000_000_000, random_state=0).fit(noisy)
            quantiles, refs = qt.quantiles_.copy(), qt.references_.copy()
        else:
            quantiles, refs = np.empty((0, 0), np.float64), np.empty(0, np.float64)
        return cls(schema, vocab, quantiles, refs, len(frame))

    def transform(self, X):
        from sklearn.preprocessing import QuantileTransformer
        frame = _frame(X, self.schema)
        numeric_names = [x for x in self.schema['columns'] if x not in self.schema['categorical']]
        result = np.empty(frame.shape, dtype=np.float32)
        if numeric_names:
            numeric = np.ascontiguousarray(frame[numeric_names].to_numpy(dtype=np.float32))
            if not np.isfinite(numeric).all():
                raise ValueError('Numerical query features must be finite')
            if len(frame):
                qt = QuantileTransformer(output_distribution='normal',
                    n_quantiles=len(self.references), subsample=1_000_000_000, random_state=0)
                qt.n_quantiles_, qt.n_features_in_ = len(self.references), len(numeric_names)
                qt.quantiles_, qt.references_ = self.quantiles.copy(), self.references.copy()
                result[:, [self.schema['columns'].index(x) for x in numeric_names]] = qt.transform(numeric)
        vocab_index = 0
        for index, name in enumerate(self.schema['columns']):
            if name in self.schema['categorical']:
                vocab = self.vocabularies[vocab_index]
                lookup = {x: i for i, x in enumerate(vocab)}
                result[:, index] = [lookup.get(_category(x), len(vocab)) for x in frame[name]]
                vocab_index += 1
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite transformed features')
        return result

    def to_state(self):
        return dict(schema=self.schema, vocabularies=[list(v) for v in self.vocabularies],
            quantiles=self.quantiles.tolist(), references=self.references.tolist(),
            fit_rows=self.fit_rows, quantile_random_state=0, noise=.001,
            input_dtype='float32', noisy_fit_dtype='float64', output_distribution='normal')

    @classmethod
    def from_state(cls, value):
        fields = {'schema', 'vocabularies', 'quantiles', 'references', 'fit_rows',
            'quantile_random_state', 'noise', 'input_dtype', 'noisy_fit_dtype', 'output_distribution'}
        if type(value) is not dict or set(value) != fields:
            raise ValueError('Invalid quantile transform fields')
        schema = _schema(value['schema'])
        rows = value['fit_rows']
        if (type(rows) is not int or not 10 <= rows <= 20000
                or value['quantile_random_state'] != 0 or value['noise'] != .001
                or value['input_dtype'] != 'float32' or value['noisy_fit_dtype'] != 'float64'
                or value['output_distribution'] != 'normal'):
            raise ValueError('Unsupported quantile recipe or workload')
        vocab = value['vocabularies']
        if (type(vocab) is not list or len(vocab) != len(schema['categorical'])
                or any(type(v) is not list or not v or any(type(x) is not str for x in v)
                    or v != sorted(set(v)) for v in vocab)):
            raise ValueError('Invalid F category vocabulary')
        dimension = len(schema['columns'])-len(vocab)
        nquantiles = max(min(rows//30, 1000), 10) if dimension else 0
        q, r = np.asarray(value['quantiles'], dtype=np.float64), np.asarray(value['references'], dtype=np.float64)
        if dimension == 0 and q.size == 0:
            q = q.reshape(0, 0)
        if (q.shape != (nquantiles, dimension) or r.shape != (nquantiles,)
                or not np.isfinite(q).all() or not np.isfinite(r).all()
                or (dimension and (not np.array_equal(r, np.linspace(0, 1, nquantiles))
                    or np.any(np.diff(q, axis=0) < 0)))):
            raise ValueError('Invalid retained numerical quantiles')
        return cls(schema, tuple(tuple(v) for v in vocab), q.copy(), r.copy(), rows)


def validate_workload(config, fit_rows, select_rows, cardinalities):
    """Finite safety envelope, not a receipt of full-quality qualification."""
    if (type(fit_rows) is not int or not 10 <= fit_rows <= 20000
            or type(select_rows) is not int or not 1 <= select_rows <= 20000
            or len(config.feature_schema['columns']) > 128
            or len(cardinalities) != len(config.feature_schema['categorical'])
            or any(type(x) is not int or not 2 <= x <= 20001 for x in cardinalities)
            or sum(cardinalities) > 100000):
        raise ValueError('Ordinary neural input exceeds the versioned finite safety envelope')
    if config.family == 'P04' and fit_rows % config.resolved()['batch_size'] == 1:
        raise ValueError('Native BatchNorm final singleton batch unsupported; rows are never dropped')


def build_model(config, cardinalities):
    """Port of the locked author definitions; ReLU anchors fixture-checked exactly."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    params = config.resolved()
    columns = config.feature_schema['columns']
    mask = torch.tensor([x in config.feature_schema['categorical'] for x in columns], dtype=torch.bool)
    d_in = len(columns)-len(cardinalities)
    if len(cardinalities) != len(config.feature_schema['categorical']):
        raise ValueError('Category cardinalities differ from the feature schema')

    class NativeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.categorical_indicator = mask if cardinalities else None
            self.categories = list(cardinalities)
            width_in = d_in
            if cardinalities:
                offsets = torch.tensor(np.concatenate([[0], np.array(cardinalities[:-1], dtype=np.int64)])).cumsum(0)
                self.register_buffer('category_offsets', offsets)
                self.category_embeddings = nn.Embedding(sum(cardinalities), 8)
                nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
                for i, count in enumerate(cardinalities):
                    self.category_embeddings.weight.data[offsets[i]+count-1].zero_()
                width_in += len(cardinalities)*8
            if config.family == 'P03':
                self.num_emb_layer = nn.Identity()
                dims = params['hidden_dims']
                self.layers = nn.ModuleList([nn.Linear(dims[i-1] if i else width_in, width)
                    for i, width in enumerate(dims)])
                self.head = nn.Linear(dims[-1], 1)
            else:
                width, hidden = params['width'], params['width']*2
                self.first_layer = nn.Linear(width_in, width)
                self.layers = nn.ModuleList([nn.ModuleDict(dict(norm=nn.BatchNorm1d(width),
                    linear0=nn.Linear(width, hidden), linear1=nn.Linear(hidden, width)))
                    for _ in range(params['blocks'])])
                self.last_normalization = nn.BatchNorm1d(width)
                self.head = nn.Linear(width, 1)

        def forward(self, x):
            if self.categorical_indicator is None:
                combined = x
            else:
                num = x[:, ~self.categorical_indicator].float()
                cat = x[:, self.categorical_indicator].long()
                embedded = self.category_embeddings(cat+self.category_offsets[None]).view(len(x), -1)
                combined = torch.cat([num, embedded], dim=-1)
            if config.family == 'P03':
                for layer in self.layers:
                    combined = layer(combined)
                    combined = F.relu(combined) if params['activation'] == 'ReLU' else F.gelu(combined)
                    if params['dropout']:
                        combined = F.dropout(combined, params['dropout'], self.training)
            else:
                combined = self.first_layer(combined)
                for layer in self.layers:
                    z = layer['linear0'](layer['norm'](combined))
                    z = F.relu(z)
                    if params['hidden_dropout']:
                        z = F.dropout(z, params['hidden_dropout'], self.training)
                    z = layer['linear1'](z)
                    if params['residual_dropout']:
                        z = F.dropout(z, params['residual_dropout'], self.training)
                    combined = combined+z
                combined = F.relu(self.last_normalization(combined))
            return self.head(combined)
    return NativeModel()


@dataclass
class SelectionTracker:
    patience: int
    best_loss: float = math.inf
    best_epoch: int = 0
    progress_threshold: float = math.inf
    misses: int = 0

    def observe(self, loss, epoch):
        if not math.isfinite(loss) or loss < 0:
            raise FloatingPointError('Nonfinite or negative native G squared loss')
        better = loss < self.best_loss
        if better:
            self.best_loss, self.best_epoch = loss, epoch
        if loss < self.progress_threshold:
            self.progress_threshold, self.misses = loss-1e-4*loss, 0
        else:
            self.misses += 1
        return better, self.misses >= self.patience


def _source():
    files = {str(p.relative_to(_ROOT)): _sha(p) for p in sorted((_ROOT/'src/treeneuron').rglob('*.py'))}
    sources = _ROOT/'locks/research-v2/ordinary-neural'
    lock = json.loads((sources/'sources.json').read_text())
    for row in lock['sources']:
        data = (sources/row['local']).read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if _sha(sources/row['local']) != row['sha256'] or blob != row['git_blob_sha1']:
            raise ValueError('Pinned official source bytes changed')
    for path in sorted(sources.iterdir()):
        if path.is_file():
            files[str(path.relative_to(_ROOT))] = _sha(path)
    for name in ('locks/research-v2/core.json', 'uv.lock'):
        files[name] = _sha(_ROOT/name)
    return dict(package_sha256=_canonical(files), files=files)


def _record_file(distribution, path):
    dist = importlib.metadata.distribution(distribution)
    path = Path(path).resolve()
    base = Path(dist.locate_file('')).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f'Imported native file escaped its qualified installation: {path}')
    relative = str(path.relative_to(base))
    found = [x for x in dist.files or [] if str(x) == relative]
    if len(found) != 1 or found[0].hash is None or found[0].hash.mode != 'sha256':
        raise ValueError(f'Imported native file lacks a locked RECORD identity: {path}')
    digest = _sha(path)
    encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')
    if encoded != found[0].hash.value:
        raise ValueError(f'Imported native file differs from its RECORD: {path}')
    return dict(path=str(path), sha256=digest, distribution=distribution)


def _dependencies():
    """Validate the exact existing core and selected import bytes before fitting."""
    global _IMPORT_CONTEXT
    if any(os.environ.get(x) != '1' for x in _THREADS):
        raise ValueError('All numerical thread variables must be1 before native imports')
    before = os.environ.get('KMP_DUPLICATE_LIB_OK')
    if _IMPORT_CONTEXT is None and ('sklearn' in sys.modules or before is not None):
        raise ValueError('Use a fresh worker without caller OpenMP suppression or preimported sklearn')
    lock = json.loads((_ROOT/'locks/research-v2/core.json').read_text())
    actual = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata['Name'].lower().replace('_', '-')
        if name in actual:
            raise ValueError(f'Duplicate distribution identity: {name}')
        actual[name] = dist
    expected = {x['name'].lower().replace('_', '-'): x for x in lock['distributions']}
    if set(actual) != set(expected):
        raise ValueError('Native neural worker must use exactly the qualified frozen core')
    for name, row in expected.items():
        dist = actual[name]
        if (dist.version != row['version'] or str(Path(dist._path).resolve()) != row['path']
                or _sha(Path(dist._path)/'RECORD') != row['record_sha256']):
            raise ValueError(f'Frozen core distribution changed: {name}')
    imported, files = {}, []
    for package, distribution in (('torch', 'torch'), ('numpy', 'numpy'), ('sklearn', 'scikit-learn')):
        dist = actual[distribution]
        expected_init = Path(dist.locate_file(package+'/__init__.py')).resolve()
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected_init:
            raise ValueError(f'Native module import origin differs from the frozen core: {package}')
        files.append(_record_file(distribution, expected_init))
        mod = importlib.import_module(package)
        if Path(mod.__file__).resolve() != expected_init:
            raise ValueError('Imported native module origin changed')
        imported[package] = mod
    torch = imported['torch']
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    files.append(_record_file('torch', torch._C.__file__))
    np_native = importlib.import_module('numpy._core._multiarray_umath')
    files.append(_record_file('numpy', np_native.__file__))
    preprocessing = importlib.import_module('sklearn.preprocessing._data')
    files.append(_record_file('scikit-learn', preprocessing.__file__))
    if _IMPORT_CONTEXT is None:
        _IMPORT_CONTEXT = dict(caller_KMP_DUPLICATE_LIB_OK=before,
            post_import_KMP_DUPLICATE_LIB_OK=os.environ.get('KMP_DUPLICATE_LIB_OK'),
            upstream_policy='Pinned sklearn/threadpoolctl use setdefault; adapter never sets/removes/preloads it',
            compatibility='Bounded observations under upstream suppression do not assure unsuppressed compatibility')
    elif before != _IMPORT_CONTEXT['post_import_KMP_DUPLICATE_LIB_OK']:
        raise ValueError('Native import environment changed after qualification')
    return dict(core_manifest_sha256=_sha(_ROOT/'locks/research-v2/core.json'),
        executable=sys.executable, executable_sha256=_sha(sys.executable), python=sys.version,
        platform=platform.platform(), package_versions={k: d.version for k, d in sorted(actual.items())},
        imported_files=files, import_environment=copy.deepcopy(_IMPORT_CONTEXT),
        validation_scope='All27 distribution versions/locations/RECORD digests plus selected imported Python/native bytes; not a rehash of every core installed file')


def _runtime():
    import torch
    import threadpoolctl
    files = [_record_file('threadpoolctl', threadpoolctl.__file__)]
    native = []
    if sys.platform == 'darwin':
        process = ctypes.CDLL(None)
        count, name = process._dyld_image_count, process._dyld_get_image_name
        count.restype, name.restype = ctypes.c_uint32, ctypes.c_char_p
        name.argtypes = [ctypes.c_uint32]
        for index in range(count()):
            value = name(index)
            if value:
                path = Path(os.fsdecode(value)).resolve()
                if any(x in path.name.lower() for x in ('libomp', 'libgomp', 'libtorch_cpu', 'libtorch_python')):
                    native.append(dict(path=str(path), sha256=_sha(path)))
                    owned = False
                    for distribution in ('torch', 'scikit-learn', 'numpy', 'scipy'):
                        dist = importlib.metadata.distribution(distribution)
                        if path.is_relative_to(Path(dist.locate_file('')).resolve()):
                            try:
                                files.append(_record_file(distribution, path))
                                owned = True
                                break
                            except ValueError:
                                continue
                    if not owned:
                        raise ValueError('Loaded native image lacks a qualified core RECORD identity')
    pools = threadpoolctl.threadpool_info()
    if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or any(pool['num_threads'] != 1 for pool in pools)):
        raise ValueError('Actual native runtime threads differ from the one-thread treatment')
    return dict(threadpools=pools, native_images=native,
        verified_native_files=files, threads={x: os.environ.get(x) for x in _THREADS},
        torch_intra_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        post_inspection_KMP_DUPLICATE_LIB_OK=os.environ.get('KMP_DUPLICATE_LIB_OK'))


def _runtime_identity(value):
    """Stable image/RECORD/thread identity; discovery list order is immaterial."""
    return dict(native_images=sorted(value['native_images'], key=lambda x: x['path']),
        verified_native_files=sorted(value['verified_native_files'], key=lambda x: (x['path'], x['distribution'])),
        threadpools=sorted(value['threadpools'], key=lambda x: x['filepath']),
        threads=value['threads'], torch_intra_threads=value['torch_intra_threads'],
        torch_interop_threads=value['torch_interop_threads'],
        post_inspection_KMP_DUPLICATE_LIB_OK=value['post_inspection_KMP_DUPLICATE_LIB_OK'])


def _train_model(model, config, F, yF, G, yG):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    params = config.resolved()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    train = DataLoader(TensorDataset(torch.from_numpy(F), torch.from_numpy(yF[:, None]),
        torch.arange(len(F))), batch_size=params['batch_size'], shuffle=True, num_workers=0)
    select = DataLoader(TensorDataset(torch.from_numpy(G), torch.from_numpy(yG[:, None])),
        batch_size=params['batch_size'], shuffle=False, num_workers=0)
    tracker = SelectionTracker(config.patience)
    history, orders, best_weights, updates = [], [], None, 0
    initial_rng = _sha_bytes(torch.get_rng_state().numpy().tobytes())
    for epoch in range(1, config.max_epochs+1):
        model.train()
        row_ids, loss_sum, gradient_norms = [], 0., []
        for x, y, rows in train:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite native training loss')
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('Nonfinite native training gradient')
            gradient_norms.append(float(sum(g.square().sum().item() for g in grads)**.5))
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite native updated parameter')
            loss_sum += loss.item()*len(x)
            row_ids.extend(rows.tolist())
            updates += 1
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if not np.array_equal(np.sort(row_ids), np.arange(len(F))):
            raise RuntimeError('Native sampler did not cover every F row exactly once')
        orders.append(row_ids)
        model.eval()
        select_sum = 0.
        with torch.no_grad():
            for x, y in select:
                loss = torch.nn.functional.mse_loss(model(x), y)
                select_sum += loss.item()*len(x)
        risk = select_sum/len(G)
        improved, stop = tracker.observe(risk, epoch)
        history.append(dict(epoch=epoch, F_online_mse=loss_sum/len(F), G_mse=risk,
            applied_updates=updates, row_order_sha256=_sha_bytes(row_ids.tobytes()),
            gradient_norm_min=min(gradient_norms), gradient_norm_max=max(gradient_norms)))
        if improved:
            best_weights = copy.deepcopy(model.state_dict())
        if stop:
            break
    final_rng = _sha_bytes(torch.get_rng_state().numpy().tobytes())
    model.load_state_dict(best_weights, strict=True)
    model.eval()
    # Retain the pinned author's G-selected F-mean safeguard; no nonfinite rescue.
    mean = float(np.mean(yF))
    mean_risk = float(np.mean((yG-mean)**2))
    if not math.isfinite(mean_risk):
        raise FloatingPointError('Nonfinite constant-reference G loss')
    return dict(history=history, selected_epoch=tracker.best_epoch,
        executed_epochs=len(history), mean_prediction_standardized=mean,
        mean_G_mse=mean_risk, selected_F_mean=mean_risk < tracker.best_loss,
        native_stop='G_relative_patience' if tracker.misses >= config.patience else 'max_epochs',
        full_F_coverage_each_epoch=True, applied_updates=updates,
        optimizer_defaults=json.loads(json.dumps(optimizer.defaults)),
        initial_training_rng_sha256=initial_rng, final_training_rng_sha256=final_rng), np.stack(orders)


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _state_identity(model):
    return _canonical({name: dict(shape=list(value.shape), dtype=str(value.dtype),
        sha256=_sha_bytes(value.detach().cpu().numpy().tobytes()))
        for name, value in sorted(model.state_dict().items())})


def _prediction_identity(summary):
    return _canonical({key: summary[key] for key in ('selected_native_state_sha256',
        'feature_transform_sha256', 'target_transform_sha256', 'selected_F_mean',
        'mean_prediction_standardized', 'mean_G_mse')})


def _validate_summary(config, summary):
    if (summary['family'] != config.family or summary['mode'] != config.mode
            or summary['resolved_parameters'] != config.resolved()
            or summary['config_sha256'] != _canonical(config.to_state())):
        raise ValueError('Snapshot configuration differs from its retained training contract')
    expected_optimizer = dict(lr=config.resolved()['lr'], weight_decay=config.resolved()['weight_decay'],
        betas=[.9, .999], eps=1e-8, amsgrad=False, foreach=None, maximize=False,
        capturable=False, differentiable=False, fused=None, decoupled_weight_decay=True)
    if summary['optimizer_defaults'] != expected_optimizer:
        raise ValueError('Recorded optimizer defaults differ from the pinned native recipe')
    validate_workload(config, summary['fit_rows'], summary['select_rows'], summary['cardinalities'])
    history = summary['history']
    if (not history or len(history) > config.max_epochs
            or len(history) != summary['executed_epochs']):
        raise ValueError('Invalid native epoch history')
    if (not _number(summary['mean_G_mse'], 0., 1e300)
            or not _number(summary['mean_prediction_standardized'], -1e300, 1e300)
            or type(summary['selected_F_mean']) is not bool):
        raise ValueError('Invalid native fallback squared risk or prediction')
    tracker = SelectionTracker(config.patience)
    stop = False
    updates_per_epoch = math.ceil(summary['fit_rows']/config.resolved()['batch_size'])
    for index, row in enumerate(history, 1):
        if (stop or row['epoch'] != index or row['applied_updates'] != updates_per_epoch*index
                or not _number(row['F_online_mse'], 0., 1e300)):
            raise ValueError('Invalid native update/selection history')
        if not _number(row['G_mse'], 0., 1e300):
            raise ValueError('Native G squared risk must be finite and nonnegative')
        _, stop = tracker.observe(row['G_mse'], index)
    if (summary['selected_epoch'] != tracker.best_epoch
            or summary['applied_updates'] != updates_per_epoch*len(history)
            or summary['native_stop'] != ('G_relative_patience' if stop else 'max_epochs')
            or not stop and len(history) != config.max_epochs
            or summary['selected_F_mean'] != (summary['mean_G_mse'] < tracker.best_loss)):
        raise ValueError('Snapshot stopping/selection contract is inconsistent')
    eligible = config.mode != 'engineering_fixture' and not summary['time_limit_exceeded']
    status = 'timeout' if summary['time_limit_exceeded'] else ('completed_eligible' if eligible else 'completed_ineligible')
    if (summary['checkpoint_eligible'] != eligible or summary['status'] != status
            or summary['quality_eligible'] is not False or summary['native_full_anchor_qualified'] is not False):
        raise ValueError('Snapshot mode/eligibility does not follow its measured native contract')
    if summary['selected_prediction_state_sha256'] != _prediction_identity(summary):
        raise ValueError('Native fallback/prediction state differs from its fitted selection')


class OrdinaryNeuralPredictor:
    def __init__(self, family):
        if family not in ('P03', 'P04'):
            raise ValueError('Require P03 or P04')
        self.family, self._model, self._summary = family, None, {}

    def fit(self, request):
        if self._model is not None:
            raise RuntimeError('Use a fresh predictor for every immutable fit attempt')
        started = time.perf_counter_ns()
        self._model = None
        root = Path(request.artifact_dir)
        root.mkdir(parents=True, exist_ok=False)
        phase = 'input_validation'
        try:
            self._config = OrdinaryNeuralConfig.parse(self.family, request.config)
            if type(request.seed) is not int or not 0 <= request.seed < 2**63:
                raise ValueError('Native model/sampler seed must be a nonnegative63-bit integer')
            if (type(request.budget) is not dict or not set(request.budget) <= {'max_seconds'}
                    or 'max_seconds' in request.budget and not _number(request.budget['max_seconds'], 1e-9, 1e12)):
                raise ValueError('Only a positive max_seconds observation budget is supported')
            if (not 10 <= len(request.X_fit) <= 20000 or not 1 <= len(request.X_select) <= 20000
                    or len(self._config.feature_schema['columns']) > 128):
                raise ValueError('Input exceeds the finite native safety envelope before preprocessing')
            yF, yG = _target(request.y_fit, len(request.X_fit)), _target(request.y_select, len(request.X_select))
            self._source = _source()
            phase = 'dependency_validation'
            dependency_start = time.perf_counter_ns()
            self._dependencies = _dependencies()
            dependency_ns = time.perf_counter_ns()-dependency_start
            phase = 'F_preprocessing'
            self._transform = NeuralTransform.fit(request.X_fit, self._config.feature_schema)
            self._target = TargetTransform.fit(yF)
            F, G = self._transform.transform(request.X_fit), self._transform.transform(request.X_select)
            tF, tG = self._target.transform(yF), self._target.transform(yG)
            validate_workload(self._config, len(F), len(G), self._transform.cardinalities)
            self._seed = request.seed
            _write(root/'input-manifest.json', dict(schema='research/v2-ordinary-neural-attempt',
                source=self._source, dependencies=self._dependencies, seed=request.seed,
                config=self._config.to_state(), resolved_parameters=self._config.resolved(),
                feature_transform=self._transform.to_state(), target_transform=self._target.to_state(),
                F_features_sha256=_sha_bytes(F.tobytes()), G_features_sha256=_sha_bytes(G.tobytes()),
                F_target_sha256=_sha_bytes(yF.tobytes()), G_target_sha256=_sha_bytes(yG.tobytes()),
                accessed_roles=['F', 'G'], fit_rows=len(F), select_rows=len(G), budget=request.budget))
            import torch
            phase = 'native_training'
            native_start = time.perf_counter_ns()
            with torch.random.fork_rng():
                torch.manual_seed(request.seed)
                model = build_model(self._config, self._transform.cardinalities)
                if sum(p.numel() for p in model.parameters()) > 10_000_000:
                    raise ValueError('Model exceeds the finite10M-parameter safety envelope')
                training, orders = _train_model(model, self._config, F, tF, G, tG)
            if self._source != _source():
                raise ValueError('Source changed during native fitting; no successful checkpoint is admitted')
            native_ns = time.perf_counter_ns()-native_start
            phase = 'runtime_inspection'
            runtime = _runtime()
            np.savez(root/'F-row-orders.npz', row_indices=orders)
            elapsed = time.perf_counter_ns()-started
            timed_out = elapsed > request.budget.get('max_seconds', math.inf)*1e9
            eligible = self._config.mode != 'engineering_fixture' and not timed_out
            summary = dict(schema='research/v2-ordinary-neural-resources', family=self.family,
                mode=self._config.mode, fitted=True, fit_rows=len(F), select_rows=len(G),
                cardinalities=self._transform.cardinalities, seed=request.seed,
                resolved_parameters=self._config.resolved(), config_sha256=_canonical(self._config.to_state()),
                status='timeout' if timed_out else 'completed_eligible' if eligible else 'completed_ineligible',
                checkpoint_eligible=eligible, quality_eligible=False, native_full_anchor_qualified=False,
                Gate_D='pending_full_native_anchor_and_external_workload_qualification',
                native_training_wall_ns=native_ns, dependency_import_wall_ns=dependency_ns,
                total_fit_wall_ns=elapsed, process_peak_rss_bytes=_peak_rss(),
                rss_scope='whole process lifetime peak, not role-incremental',
                import_environment=self._dependencies['import_environment'], runtime=runtime,
                time_limit_exceeded=timed_out, budget=dict(request.budget),
                hard_resource_enforcement='outer worker harness required; max_seconds here is post-fit observation',
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                row_orders_sha256=_sha(root/'F-row-orders.npz'), accessed_roles=['F', 'G'],
                feature_transform_sha256=_canonical(self._transform.to_state()),
                target_transform_sha256=_canonical(self._target.to_state()),
                selected_native_state_sha256=_state_identity(model),
                interrupted_training_resume='not_implemented; any admitted deterministic rerun retains all paid attempts',
                **training)
            summary['selected_prediction_state_sha256'] = _prediction_identity(summary)
            _validate_summary(self._config, summary)
            phase = 'receipt_publication'
            _write(root/'attempt.json', summary)
            self._model, self._summary = model, summary
            return self
        except BaseException as error:
            status, domain, retry = _failure_domain(error)
            self._summary = dict(status=status, failure_domain=domain, infrastructure_retry_permitted=retry,
                checkpoint_eligible=False, exception=type(error).__name__, message=str(error),
                traceback=traceback.format_exc(), total_fit_wall_ns=time.perf_counter_ns()-started,
                failure_phase=phase, peak_rss_bytes=_peak_rss())
            try:
                _write(root/'attempt.json', self._summary)
            except BaseException as receipt_error:
                error.add_note(f'Failure receipt could not be published: {receipt_error!r}')
            raise

    def predict(self, X, *, batch_size=None):
        if self._model is None:
            raise RuntimeError('Predictor must be fitted before prediction')
        size = self._config.resolved()['batch_size'] if batch_size is None else batch_size
        if type(size) is not int or not 1 <= size <= 65536:
            raise ValueError('Prediction batch size must be a positive bounded integer')
        values = self._transform.transform(X)
        import torch
        self._model.eval()
        with torch.no_grad():
            if self._summary['selected_F_mean']:
                prediction = np.full(len(values), self._summary['mean_prediction_standardized'], dtype=np.float32)
            else:
                prediction = np.concatenate([self._model(torch.from_numpy(values[i:i+size])).numpy().reshape(-1)
                    for i in range(0, len(values), size)]) if len(values) else np.empty(0, np.float32)
        result = self._target.inverse(prediction)
        if result.shape != (len(values),) or not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite or invalid native prediction')
        return result

    def resource_summary(self):
        return copy.deepcopy(self._summary)

    def save(self, path):
        if self._model is None:
            raise RuntimeError('Predictor must be fitted before saving')
        if self._source != _source():
            raise ValueError('Source changed after native fitting')
        _validate_summary(self._config, self._summary)
        if (self._summary['feature_transform_sha256'] != _canonical(self._transform.to_state())
                or self._summary['target_transform_sha256'] != _canonical(self._target.to_state())
                or self._summary['selected_native_state_sha256'] != _state_identity(self._model)):
            raise ValueError('Native prediction state changed after the fitted checkpoint')
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        state = {name: tensor.detach().cpu().numpy().copy() for name, tensor in self._model.state_dict().items()}
        np.savez(path/'weights.npz', **state)
        _write(path/'manifest.json', dict(schema='research/v2-ordinary-neural-snapshot',
            family=self.family, config=self._config.to_state(), seed=self._seed,
            source=self._source, dependencies=self._dependencies,
            feature_transform=self._transform.to_state(), target_transform=self._target.to_state(),
            resources=self._summary, weights_sha256=_sha(path/'weights.npz'),
            payload={k: dict(dtype=str(v.dtype), shape=list(v.shape)) for k, v in state.items()},
            snapshot_scope='portable native prediction state; optimizer/sampler training recovery not implemented'))

    @classmethod
    def load(cls, path, *, expected_manifest_sha256):
        if (type(expected_manifest_sha256) is not str or len(expected_manifest_sha256) != 64
                or any(c not in '0123456789abcdef' for c in expected_manifest_sha256)):
            raise ValueError('Loading requires an external trusted manifest SHA256')
        path = Path(path)
        if _sha(path/'manifest.json') != expected_manifest_sha256:
            raise ValueError('Snapshot differs from its external trusted manifest hash')
        value = json.loads((path/'manifest.json').read_text())
        fields = {'schema', 'family', 'config', 'seed', 'source', 'dependencies', 'feature_transform',
            'target_transform', 'resources', 'weights_sha256', 'payload', 'snapshot_scope'}
        if set(value) != fields or value['schema'] != 'research/v2-ordinary-neural-snapshot':
            raise ValueError('Unknown native snapshot schema/fields')
        config = OrdinaryNeuralConfig.parse(value['family'], value['config'])
        source = _source()
        if value['source'] != source:
            raise ValueError('Native snapshot belongs to a different source identity')
        dependencies = _dependencies()
        if value['dependencies'] != dependencies:
            raise ValueError('Native snapshot belongs to a different qualified runtime')
        summary = value['resources']
        current_runtime = _runtime()
        if _runtime_identity(current_runtime) != _runtime_identity(summary['runtime']):
            raise ValueError('Actual native images/bytes/thread treatment differ from the retained runtime')
        _validate_summary(config, summary)
        transform = NeuralTransform.from_state(value['feature_transform'])
        target = TargetTransform.from_state(value['target_transform'])
        if (transform.schema != config.feature_schema or transform.fit_rows != summary['fit_rows']
                or transform.cardinalities != summary['cardinalities'] or value['seed'] != summary['seed']
                or type(value['seed']) is not int or not 0 <= value['seed'] < 2**63):
            raise ValueError('Snapshot transform, seed and training roles disagree')
        if (summary['feature_transform_sha256'] != _canonical(transform.to_state())
                or summary['target_transform_sha256'] != _canonical(target.to_state())):
            raise ValueError('Snapshot preprocessing differs from its fitted native checkpoint')
        if _sha(path/'weights.npz') != value['weights_sha256']:
            raise ValueError('Native tensor payload hash mismatch')
        import torch
        with torch.random.fork_rng():
            model = build_model(config, transform.cardinalities)
        expected = model.state_dict()
        with np.load(path/'weights.npz', allow_pickle=False) as archive:
            if set(archive.files) != set(expected) or set(value['payload']) != set(expected):
                raise ValueError('Native tensor names differ from the declared architecture')
            state = {}
            for name, tensor in expected.items():
                actual = archive[name]
                if (actual.dtype != tensor.numpy().dtype or actual.shape != tuple(tensor.shape)
                        or not np.isfinite(actual).all()
                        or value['payload'][name] != dict(dtype=str(actual.dtype), shape=list(actual.shape))):
                    raise ValueError('Native tensor shape/dtype/finiteness mismatch; implicit casts forbidden')
                if name.endswith('running_var') and np.any(actual < 0):
                    raise ValueError('Negative BatchNorm variance')
                if name.endswith('num_batches_tracked') and int(actual) != summary['selected_epoch']*math.ceil(
                        summary['fit_rows']/config.resolved()['batch_size']):
                    raise ValueError('BatchNorm age differs from selected checkpoint')
                state[name] = torch.from_numpy(actual.copy())
        model.load_state_dict(state, strict=True)
        if _state_identity(model) != summary['selected_native_state_sha256']:
            raise ValueError('Native weights differ from the selected fitted checkpoint')
        model.eval()
        result = cls(config.family)
        result._model, result._config, result._seed = model, config, value['seed']
        result._source, result._dependencies = source, dependencies
        result._transform, result._target, result._summary = transform, target, summary
        return result
