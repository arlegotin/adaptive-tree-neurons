"""Independent original-unit prediction arithmetic on already authorized arrays.

No file, predictor, producer metric helper or protected-data reader is invoked.
Protocol/source/model/role authentication belongs to the caller's sealed reader.
"""
import math

import numpy as np

from treeneuron.data.types import array_hash
from treeneuron.research.ledger import TERMINAL


def _vector(value,name,dtype):
    array=np.asarray(value)
    if array.ndim!=1 or not len(array) or array.dtype!=np.dtype(dtype):
        raise ValueError(f'{name} must be a nonempty one-dimensional {dtype} array')
    if dtype=='float64' and not np.isfinite(array).all():raise ValueError(f'{name} must be finite')
    return array


def _mean(values):
    if not all(math.isfinite(v) for v in values):raise ValueError('Unrepresentable loss or target variance')
    # Preserve representable subnormal means. Only rescale first when the
    # unscaled sum overflows despite the mean itself being representable.
    try:result=math.fsum(values)/len(values)
    except OverflowError:result=math.fsum(float(v)/len(values) for v in values)
    if not math.isfinite(result):raise ValueError('Unrepresentable loss or target variance')
    return result


def _variance(values):
    anchor=float(values[0]);offset=[float(v)-anchor for v in values]
    if not all(math.isfinite(v) for v in offset):raise ValueError('Unrepresentable F-target variance')
    center=_mean(offset);squares=[(v-center)*(v-center) for v in offset]
    return _mean(squares)


def _metrics(y,prediction,groups,grouping,variance):
    differences=[float(p)-float(v) for p,v in zip(prediction,y)]
    squared=[v*v for v in differences];absolute=[abs(v) for v in differences]
    row_mse=_mean(squared);row_mae=_mean(absolute)
    if grouping=='row':primary_mse=row_mse;primary_mae=row_mae
    else:
        by_group={}
        for group,square,error in zip(groups,squared,absolute):
            pair=by_group.setdefault(int(group),([],[]));pair[0].append(square);pair[1].append(error)
        primary_mse=_mean([_mean(v[0]) for v in by_group.values()])
        primary_mae=_mean([_mean(v[1]) for v in by_group.values()])
    normalized=primary_mse/variance if variance>0 else None
    if normalized is not None and not math.isfinite(normalized):raise ValueError('Unrepresentable normalized risk')
    return dict(primary_MSE=primary_mse,primary_RMSE=math.sqrt(primary_mse),primary_MAE=primary_mae,
        normalized_risk=normalized,row_MSE=row_mse,row_RMSE=math.sqrt(row_mse),row_MAE=row_mae)


def audit_prediction_arrays(*,F_y,y,row_id,group_id,grouping,timestamps,
        planned_methods,statuses,predictions):
    """Recompute every available planned method; never impute missing risks.

    F variance uses ddof0. Groups/days receive equal weight, and rows inside a
    group receive equal weight. Synthetic row grouping keeps group_id==row_id;
    transfer grouping deliberately permits repeated group IDs.
    """
    F=_vector(F_y,'F targets','float64');target=_vector(y,'evaluation targets','float64')
    rows=_vector(row_id,'row IDs','int64');groups=_vector(group_id,'group IDs','int64')
    if (len(F)<2 or len(rows)!=len(target) or len(groups)!=len(target)
            or len(np.unique(rows))!=len(rows) or grouping not in ('row','group','day')):
        raise ValueError('Invalid target/row/group geometry or grouping policy')
    if grouping=='row' and not np.array_equal(rows,groups):
        raise ValueError('Synthetic row grouping requires group IDs equal to row IDs')
    if (type(planned_methods) is not list or not planned_methods
            or any(type(v) is not str or not v for v in planned_methods)
            or len(set(planned_methods))!=len(planned_methods)
            or type(statuses) is not dict or set(statuses)!=set(planned_methods)
            or any(v not in TERMINAL|{'absent','running'} for v in statuses.values())
            or type(predictions) is not dict):
        raise ValueError('Complete explicit method inventory and statuses required')
    eligible={name for name in planned_methods if statuses[name]=='completed_eligible'}
    if set(predictions)!=eligible:raise ValueError('Prediction vectors must match exactly the eligible planned methods')
    times_hash=None
    if grouping=='day':
        times=np.asarray(timestamps)
        if (times.shape!=target.shape or times.dtype.kind!='M' or np.isnat(times).any()
                or np.datetime_data(times.dtype)[0] not in ('h','m','s','ms','us','ns')
                or np.any(times[1:]<=times[:-1])
                or not np.array_equal(groups,times.astype('datetime64[D]').astype(np.int64))):
            raise ValueError('Day grouping requires ordered actual timestamps and matching day IDs')
        times_hash=array_hash(times)
    elif timestamps is not None:raise ValueError('Timestamp metadata is declared only for temporal day grouping')
    variance=_variance(F);methods={};prediction_hashes={}
    try:
        for name in planned_methods:
            if name not in eligible:continue
            record=predictions[name]
            if type(record) is not dict or set(record)!={'row_id','values'}:
                raise ValueError('Prediction record requires values and exact row identities')
            ids=_vector(record['row_id'],'prediction row IDs','int64')
            values=_vector(record['values'],'predictions','float64')
            if not np.array_equal(ids,rows) or values.shape!=target.shape:
                raise ValueError('Method predictions do not align with the evaluation rows')
            methods[name]=_metrics(target,values,groups,grouping,variance)
            prediction_hashes[name]=dict(row_id=array_hash(ids),values=array_hash(values))
    except (OverflowError,FloatingPointError) as error:
        raise ValueError('Unrepresentable prediction arithmetic') from error
    missing=[dict(method_id=name,status=statuses[name]) for name in planned_methods if name not in eligible]
    return dict(schema='confirmation/v2-prediction-arithmetic',
        status='incomplete_predictions' if missing else 'undefined_F_variance' if variance==0 else 'complete_arithmetic',
        planned_methods=list(planned_methods),statuses=dict(statuses),missing_methods=missing,
        rows=len(target),groups=len(np.unique(groups)),grouping=grouping,F_target_variance=variance,
        methods=methods,F_targets_sha256=array_hash(F),evaluation_targets_sha256=array_hash(target),
        row_ids_sha256=array_hash(rows),group_ids_sha256=array_hash(groups),timestamps_sha256=times_hash,
        prediction_hashes=prediction_hashes,risk_inventory_complete=not missing and variance>0,
        protected_access_authenticated=False,raw_model_provenance_authenticated=False,claim_authorized=False,
        scope='Arithmetic only; caller must separately authenticate frozen data/model/source/access and role independence')
