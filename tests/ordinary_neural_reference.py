"""Test-only direct invocation of verified pinned author definitions."""
import ast
import hashlib
import json
import math
import numbers
from pathlib import Path
import typing as ty

import numpy as np


def official_definitions():
    import torch
    from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
    from sklearn.preprocessing import QuantileTransformer
    root = Path(__file__).resolve().parents[1]/'locks/research-v2/ordinary-neural'
    lock = json.loads((root/'sources.json').read_text())
    names = {'RTDL_MLP', 'ResNet', 'get_activation_fn', 'get_nonglu_activation_fn',
        'reglu', 'geglu', 'TabrQuantileTransformer'}
    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, Tensor=torch.Tensor,
        math=math, numbers=numbers, ty=ty, np=np, BaseEstimator=BaseEstimator,
        TransformerMixin=TransformerMixin, check_is_fitted=check_is_fitted,
        QuantileTransformer=QuantileTransformer)
    for name in ('pytabkit-rtdl_resnet.py.txt', 'pytabkit-utils.py.txt'):
        path = root/name
        row = next(x for x in lock['sources'] if x['local'] == name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        tree = ast.parse(path.read_text())
        selected = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in names]
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def official_model(family, schema, cardinalities, params):
    import torch
    definitions = official_definitions()
    categories = cardinalities or None
    catmask = torch.tensor([x in schema['categorical'] for x in schema['columns']], dtype=torch.bool) if categories else None
    common = dict(d_in=len(schema['columns'])-len(schema['categorical']), categories=categories,
        d_embedding=8, d_out=1, regression=True, categorical_indicator=catmask)
    if family == 'P03':
        return definitions['RTDL_MLP'](**common, n_layers=3, d_layers=[128, 256, 128],
            d_first_layer=128, d_last_layer=128, dropout=.1)
    return definitions['ResNet'](**common, d=128, d_hidden_factor=2, n_layers=2,
        activation='relu', normalization='batchnorm', hidden_dropout=.25, residual_dropout=.1)


def official_threshold(loss):
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[1]/'locks/research-v2/ordinary-neural'
    lock = json.loads((root/'sources.json').read_text())
    row = next(x for x in lock['sources'] if x['local'] == 'skorch-training.py.txt')
    path = root/row['local']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    tree = ast.parse(path.read_text())
    early = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'EarlyStopping')
    method = next(x for x in early.body if isinstance(x, ast.FunctionDef) and x.name == '_calc_new_threshold')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_calc_new_threshold'](SimpleNamespace(threshold_mode='rel', threshold=1e-4,
        lower_is_better=True), loss)
