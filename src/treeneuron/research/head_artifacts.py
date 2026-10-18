"""Portable retained frozen-head predictors; no refit during evaluation."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from treeneuron import train
from treeneuron.config import canonical_json,canonical_hash
from treeneuron.data.transforms import FittedTransform
from treeneuron.data.types import _readonly
from .diagnostics import _parent
from .headroom import HEAD_ALPHAS,HeadRefit,FrozenHeadPredictor,refit_frozen_head
from .ledger import write_new_json


def _array(value):
    return dict(dtype=str(value.dtype),shape=list(value.shape),values=value.tolist())


def _decode(value,dtype,shape=None):
    if (not isinstance(value,dict) or set(value)!={'dtype','shape','values'}
            or value['dtype']!=dtype):raise ValueError('Frozen-head array dtype/schema differs')
    array=np.asarray(value['values'],dtype=np.dtype(dtype))
    if (list(array.shape)!=value['shape'] or (shape is not None and array.shape!=shape)
            or not np.isfinite(array).all()):raise ValueError('Frozen-head array shape/values differ')
    return _readonly(array,np.dtype(dtype))


def _verify_selection(record,coefficient):
    evidence=json.loads(record['records_json']);canonical_json(evidence)
    if (not isinstance(evidence,dict) or set(evidence)!={'trials','score_units','selection'}
            or evidence['score_units']!='provided_target_coordinates_squared'
            or evidence['selection']!='G_mse_smallest_alpha_with_1e-12_relative_floor_ties'
            or not isinstance(evidence['trials'],list) or len(evidence['trials'])!=len(HEAD_ALPHAS)):
        raise ValueError('Frozen-head selection evidence is incomplete')
    trials=evidence['trials']
    for alpha,row in zip(HEAD_ALPHAS,trials):
        required={'alpha','G_mse','rank','singular_values','coefficients','intercept','elapsed_ns','status'}
        if (not isinstance(row,dict) or set(row)!=required or row['status']!='completed_eligible'
                or any(type(row[k]) not in (int,float) or not math.isfinite(row[k])
                       for k in ('alpha','G_mse','intercept'))
                or row['alpha']!=alpha or row['G_mse']<0
                or type(row['rank']) is not int or not 0<=row['rank']<=len(coefficient)
                or type(row['elapsed_ns']) is not int or row['elapsed_ns']<0):
            raise ValueError('Frozen-head selection solve record is invalid')
        vector=np.asarray(row['coefficients'],dtype=np.float64)
        singular=np.asarray(row['singular_values'],dtype=np.float64)
        if (vector.shape!=coefficient.shape or not np.isfinite(vector).all()
                or singular.ndim!=1 or not np.isfinite(singular).all() or (singular<0).any()):
            raise ValueError('Frozen-head selection solve arrays are invalid')
    best=min(row['G_mse'] for row in trials)
    winner=next(row for row in trials if row['G_mse']<=best+1e-12*max(1.,abs(best)))
    if (record['alpha']!=winner['alpha'] or record['intercept']!=winner['intercept']
            or not np.array_equal(coefficient,np.asarray(winner['coefficients'],dtype=np.float64))):
        raise ValueError('Frozen-head retained parameters disagree with G selection')


def fit_head_artifact(parent,data,output):
    started=time.perf_counter_ns();_,fingerprint=_parent(parent)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    report=dict(schema='research/v2-head-artifact',status='running',parent_sha256=fingerprint,
        parent_identity_hashes=dict(parent.identity_hashes),V_accessed=False,E_accessed=False,T_accessed=False)
    write_new_json(root/'request.json',report);before=train.capture_rng()
    try:
        fitted=refit_frozen_head(parent,data);head=fitted.head
        record=dict(schema='research/v2-frozen-head',source_sha256=train.source_hash(),
            parent_sha256=fingerprint,parent_identity_hashes=fitted.parent_identity_hashes,
            preprocessing=fitted.preprocessing.to_state(),alpha=head.alpha,intercept=head.intercept,
            records_json=head.records_json,
            arrays={name:_array(getattr(head,name)) for name in
                ('coefficients','feature_mean','feature_scale','active_columns')},
            weights=_array(fitted.weights),bias=_array(fitted.bias))
        path=root/'model.json';write_new_json(path,record)
        selected=next(r for r in head.records if r['alpha']==head.alpha)
        report.update(status='completed_eligible',selected_alpha=head.alpha,ridge_solve_count=len(head.records),
            trials=head.records,selected_g_mse=selected['G_mse']*fitted.preprocessing.to_state()['target_scale']**2,
            models={'selected':dict(loader='research_frozen_head',path=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parent_sha256=fingerprint)})
        return report
    except BaseException as error:
        report.update(status='numerical_failure' if isinstance(error,(np.linalg.LinAlgError,FloatingPointError))
            else 'interrupted_recoverable' if isinstance(error,KeyboardInterrupt)
            else 'resource_exhausted' if isinstance(error,(MemoryError,OSError)) else 'input_or_dependency_failure',
            error_type=type(error).__name__,reason=str(error),trials=getattr(error,'head_records',[]))
        raise
    finally:
        active=sys.exc_info()[1]
        report['costs']=dict(enclosing_head_ns=time.perf_counter_ns()-started,standalone_total_ns=None,
            reason='outer controller must add validated parent prerequisite costs')
        try:
            try:write_new_json(root/'result.json',report)
            except BaseException as error:
                if active is None:raise
                active.add_note(f'Frozen-head result publication also failed: {error}')
        finally:train.restore_rng(before)


def load_head_artifact(reference):
    if (not isinstance(reference,dict) or set(reference)!={'loader','path','sha256','parent_sha256'}
            or reference['loader']!='research_frozen_head'):
        raise ValueError('Invalid frozen-head predictor reference')
    path=Path(reference['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']:
        raise ValueError('Frozen-head model checksum mismatch')
    record=json.loads(path.read_text());canonical_json(record)
    required={'schema','source_sha256','parent_sha256','parent_identity_hashes','preprocessing',
        'alpha','intercept','records_json','arrays','weights','bias'}
    if (set(record)!=required or record['schema']!='research/v2-frozen-head'
            or record['source_sha256']!=train.source_hash()
            or record['parent_identity_hashes'].get('code')!=train.source_hash()
            or record['parent_sha256']!=reference['parent_sha256']):
        raise ValueError('Frozen-head source/parent identity mismatch')
    identities=record['parent_identity_hashes']
    if (set(identities)!={'code','lock','runtime','config','split','raw','transform'}
            or any(type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
                   for v in identities.values())
            or identities['lock']!=train.lock_hash() or identities['runtime']!=train.runtime_hash()
            or identities['transform']!=canonical_hash(record['preprocessing'])):
        raise ValueError('Frozen-head lock/runtime/transform identity mismatch')
    weights=_decode(record['weights'],'float32')
    if weights.ndim!=2:raise ValueError('Frozen-head weights require h by d shape')
    h,d=weights.shape;bias=_decode(record['bias'],'float32',(h,))
    arrays=record['arrays']
    if set(arrays)!={'coefficients','feature_mean','feature_scale','active_columns'}:
        raise ValueError('Frozen-head geometry inventory differs')
    active=_decode(arrays['active_columns'],'bool',(h,))
    mean=_decode(arrays['feature_mean'],'float64',(h,))
    scale=_decode(arrays['feature_scale'],'float64',(h,))
    coefficient=_decode(arrays['coefficients'],'float64',(int(active.sum()),))
    transform=FittedTransform.from_state(record['preprocessing'])
    if (len(transform.to_state()['columns'])!=d or not (scale>0).all()
            or any(type(record[k]) not in (int,float) or not math.isfinite(record[k]) for k in ('alpha','intercept'))
            or record['alpha']<0):raise ValueError('Invalid frozen-head geometry/scalars')
    _verify_selection(record,coefficient)
    head=HeadRefit(record['alpha'],record['intercept'],coefficient,mean,scale,active,record['records_json'])
    return FrozenHeadPredictor(head,weights,bias,transform,canonical_json(record['parent_identity_hashes']))
