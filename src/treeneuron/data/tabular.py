"""Pinned transfer-table schemas and F-only numeric encoding.

Raw targets are parsed only by preparation/coordinator code. Native adapters
receive role-restricted views rather than this complete table.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from treeneuron.config import canonical_hash
from .types import _readonly, array_hash


@dataclass(frozen=True)
class DatasetSchema:
    uci_id: int
    raw_member: str
    raw_sha256: str
    rows: int
    groups: int
    raw_columns: tuple[str, ...]
    features: tuple[str, ...]
    target: str
    group_features: tuple[str, ...]
    categorical: tuple[str, ...] = ()
    simulation_only: bool = False


_VOICE = ('Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
          'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
          'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE')
_AIR = ('frequency', 'angle', 'chord', 'velocity', 'thickness')
_CONCRETE = ('cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age')
_BIKE = ('season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed')
_ENERGY = tuple(f'X{i}' for i in range(1, 9))
DATASETS = {
    'airfoil': DatasetSchema(291, 'airfoil_self_noise.dat',
        '74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3', 1503, 106,
        (*_AIR, 'sound_pressure'), _AIR, 'sound_pressure', _AIR[1:]),
    'concrete': DatasetSchema(165, 'Concrete_Data.xls',
        '710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4', 1030, 428,
        (*_CONCRETE, 'strength'), _CONCRETE, 'strength', _CONCRETE[:-1]),
    'ccpp': DatasetSchema(294, 'CCPP/Folds5x2_pp.xlsx',
        'ccd490981db2a2f079963b3d9f0aea30d9d338900a0285428dfc6385396f4651', 9568, 9527,
        ('AT', 'V', 'AP', 'RH', 'PE'), ('AT', 'V', 'AP', 'RH'), 'PE', ('AT', 'V', 'AP', 'RH')),
    'parkinsons': DatasetSchema(189, 'parkinsons_updrs.data',
        'f2c7d5025dec4e92e7feae367a5f7ccf58789a10ac6b54bdf15976c599f9dd39', 5875, 42,
        ('subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS', *_VOICE),
        ('age', 'sex', 'test_time', *_VOICE), 'motor_UPDRS', ('subject#',), ('sex',)),
    'bike': DatasetSchema(275, 'hour.csv',
        'e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f', 17379, 731,
        ('instant', 'dteday', *_BIKE, 'casual', 'registered', 'cnt'), _BIKE, 'cnt',
        ('dteday',), ('season', 'mnth', 'hr', 'weekday', 'weathersit', 'holiday', 'workingday')),
    'energy': DatasetSchema(242, 'ENB2012_data.xlsx',
        '0089fcffc1415e41e2ff63730cca5280efe54fc43d722dfde1b0aaa808e35dc4', 768, 12,
        (*_ENERGY, 'Y1', 'Y2'), _ENERGY, 'Y1', _ENERGY[:5], ('X6', 'X8'), True),
}


@dataclass(frozen=True)
class TabularTable:
    name: str
    X: np.ndarray
    y: np.ndarray
    row_id: np.ndarray
    group_id: np.ndarray
    feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...]
    raw_sha256: str | None = None
    timestamps: np.ndarray | None = None

    def __post_init__(self):
        schema = DATASETS[self.name]
        if self.feature_names != schema.features or self.categorical_features != schema.categorical:
            raise ValueError('Transfer feature allowlist differs from its semantic schema')
        if (self.X.shape != (len(self.y), len(schema.features)) or not np.isfinite(self.X).all()
                or not np.isfinite(self.y).all() or not len(self.y)):
            raise ValueError('Transfer arrays must have finite matching nonempty dimensions')
        for name in ('X', 'y', 'row_id', 'group_id', 'timestamps'):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, _readonly(values, values.dtype))
        if (self.row_id.dtype != np.int64 or self.group_id.dtype != np.int64
                or self.row_id.shape != self.y.shape or self.group_id.shape != self.y.shape
                or len(np.unique(self.row_id)) != len(self.y)):
            raise ValueError('Transfer row/group identities must be aligned int64; rows unique')

    @property
    def table_id(self):
        return canonical_hash(dict(name=self.name, schema=asdict(DATASETS[self.name]),
            raw_sha256=self.raw_sha256, arrays={name: array_hash(getattr(self, name))
                for name in ('X', 'y', 'row_id', 'group_id')},
            timestamps=None if self.timestamps is None else array_hash(self.timestamps)))


def table_from_frame(name, frame, *, raw_sha256=None):
    """Schema-only fixture/preparation boundary; never infer predictors from y."""
    schema = DATASETS[name]
    if len(frame.columns) != len(schema.raw_columns) or set(frame.columns) != set(schema.raw_columns):
        raise ValueError(f'{name}: raw columns differ from pinned semantic schema')
    x = frame.loc[:, schema.features].to_numpy(dtype=np.float64, copy=True)
    y = frame[schema.target].to_numpy(dtype=np.float64, copy=True)
    if name == 'bike':
        dates = pd.to_datetime(frame['dteday'], format='%Y-%m-%d', errors='raise')
        times = dates.to_numpy(dtype='datetime64[h]')+frame['hr'].to_numpy(dtype='timedelta64[h]')
        if len(np.unique(times)) != len(times) or np.any(np.diff(times) <= np.timedelta64(0, 'h')):
            raise ValueError('Bike timestamps must be unique and increasing; do not fabricate missing hours')
        group = np.unique(dates.to_numpy(dtype='datetime64[D]'), return_inverse=True)[1]
    else:
        times = None
        group = np.unique(frame.loc[:, schema.group_features].to_numpy(dtype=np.float64), axis=0, return_inverse=True)[1]
    return TabularTable(name, x, y, np.arange(len(frame), dtype=np.int64),
        group.astype(np.int64), schema.features, schema.categorical, raw_sha256, times)


def load_table(name, path):
    """Accept only the exact acquired raw byte identity, then parse its schema."""
    schema, path = DATASETS[name], Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != schema.raw_sha256:
        raise ValueError(f'{name}: raw SHA256 differs; retain changed bytes as a separate version')
    if name == 'airfoil':
        frame = pd.read_csv(path, sep=r'\s+', header=None, names=schema.raw_columns)
    elif name == 'concrete':
        frame = pd.read_excel(path, engine='xlrd')
        if len(frame.columns) != len(schema.raw_columns):
            raise ValueError('Concrete pinned workbook column count changed')
        # Names below are semantic aliases for the pinned workbook column order.
        frame.columns = schema.raw_columns
    elif name in ('ccpp', 'energy'):
        frame = pd.read_excel(path, sheet_name='Sheet1' if name == 'ccpp' else 0, engine='openpyxl')
        if name == 'energy':
            frame = frame.dropna(axis=0, how='all').dropna(axis=1, how='all')
    else:
        frame = pd.read_csv(path)
    if len(frame) != schema.rows:
        raise ValueError(f'{name}: raw row count differs from pinned table')
    table = table_from_frame(name, frame, raw_sha256=digest)
    if len(np.unique(table.group_id)) != schema.groups:
        raise ValueError(f'{name}: group count differs from the pinned grouping contract')
    return table


@dataclass(frozen=True)
class NumericEncoder:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    categorical_indices: tuple[int, ...]
    categories: tuple[tuple[float, ...], ...]
    fit_rows: int

    @classmethod
    def fit(cls, X, *, categorical_indices=()):
        values = np.asarray(X, dtype=np.float64)
        cats = tuple(categorical_indices)
        if (values.ndim != 2 or not len(values) or not np.isfinite(values).all()
                or len(set(cats)) != len(cats) or any(type(i) is not int or not 0 <= i < values.shape[1] for i in cats)):
            raise ValueError('Encoder requires finite F matrix and unique valid categorical indices')
        means, scales = values.mean(axis=0), values.std(axis=0, ddof=0)
        scales[scales == 0] = 1.
        return cls(tuple(means), tuple(scales), cats,
                   tuple(tuple(np.unique(values[:, i]).tolist()) for i in cats), len(values))

    def transform(self, X):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.means) or not np.isfinite(values).all():
            raise ValueError('Encoder query matrix must be finite with matching raw dimensions')
        categorical = dict(zip(self.categorical_indices, self.categories))
        blocks = []
        for i in range(values.shape[1]):
            if i not in categorical:
                blocks.append(((values[:, i]-self.means[i])/self.scales[i])[:, None])
            else:
                matches = values[:, i:i+1] == np.asarray(categorical[i])[None, :]
                blocks.append(np.column_stack((matches, ~matches.any(axis=1))).astype(np.float64))
        return np.concatenate(blocks, axis=1)

    def to_dict(self):
        return dict(schema='research/v2-numeric-encoder', **asdict(self),
                    unknown_policy='reserved_unknown_column_per_categorical_feature')
