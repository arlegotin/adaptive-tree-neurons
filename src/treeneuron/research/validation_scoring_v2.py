"""File-bound primary V arithmetic in the declared transfer coordinator.

This component grants no model, qualification, search eligibility or protected
evaluation authority. Ordinary fitting workers never receive these labels.
"""
import hashlib
import io
from pathlib import Path
import re
import zipfile

import numpy as np

from treeneuron.analysis import prediction_arithmetic_v2
from treeneuron.data.types import array_hash
from .transfer import TransferStore


def _digest(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('External SHA256 commitment required')


def _content(path, digest):
    _digest(digest)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Committed regular evidence file required')
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError('Evidence byte commitment changed')
    return content


def _arrays(content, fields):
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as saved:
            if set(saved.files) != set(fields) or len(saved.files) != len(fields):
                raise ValueError('Exact declared prediction/role archive fields required')
            return {key: saved[key].copy() for key in fields}
    except (OSError, EOFError, TypeError, zipfile.BadZipFile) as error:
        raise ValueError('Invalid closed array archive') from error


def _identities(arrays, *, temporal):
    rows, groups = arrays['row_id'], arrays['group_id']
    if (rows.ndim != 1 or not len(rows) or rows.dtype != np.dtype('int64')
            or groups.shape != rows.shape or groups.dtype != np.dtype('int64')
            or len(np.unique(rows)) != len(rows)):
        raise ValueError('Unique ordered int64 row IDs and aligned group IDs required')
    if temporal:
        times = arrays['timestamps']
        if (times.shape != rows.shape or times.dtype.kind != 'M'
                or np.datetime_data(times.dtype)[0] not in ('h', 'm', 's', 'ms', 'us', 'ns')
                or np.isnat(times).any() or np.any(times[1:] <= times[:-1])):
            raise ValueError('Actual ordered clock hours and matching day IDs required')
        days = times.astype('datetime64[D]').astype('int64')
        # Sealed Bike group IDs index observed dates; they are not epoch days.
        # Preserve those identities and verify the partition before canonicalizing
        # only the arithmetic input to the independent day-ordinal convention.
        if not len(np.unique(np.column_stack([groups, days]), axis=0)) == len(np.unique(groups)) == len(np.unique(days)):
            raise ValueError('Each temporal group must identify exactly one actual day')


def _role_geometry(arrays, columns, *, temporal, targets):
    _identities(arrays, temporal=temporal)
    X = arrays['X']
    if (X.shape != (len(arrays['row_id']), columns) or X.dtype != np.dtype('float64')
            or not np.isfinite(X).all()):
        raise ValueError('Finite original-coordinate float64 role features required')
    if targets:
        y = arrays['y']
        if (y.shape != arrays['row_id'].shape or y.dtype != np.dtype('float64')
                or len(y) < 2 or not np.isfinite(y).all()):
            raise ValueError('Finite original-unit float64 F targets required')


def score_primary_validation_prediction(*, root, outer_fold, expected_manifest_sha256,
        prediction_path, expected_prediction_sha256):
    """Authenticate one primary V prediction and recompute the fixed metric.

    Hash/decode use the same prediction/F/feature bytes. The existing logged V
    reader is called only after feature-only prediction alignment succeeds.
    Every opened reference is rechecked before returning a score.
    """
    root = Path(root).resolve()
    manifest_path = root/'manifest.json'
    _content(manifest_path, expected_manifest_sha256)
    store = TransferStore(root)
    split = store._split(outer_fold)  # Secondary release remains forbidden.
    prediction_path = Path(prediction_path).absolute()
    # Data-store paths cannot masquerade as a worker prediction, including T.
    if prediction_path.resolve().is_relative_to(root):
        raise ValueError('Worker prediction must be outside the sealed data store')
    prediction_bytes = _content(prediction_path, expected_prediction_sha256)
    manifest = store.manifest
    temporal = manifest['dataset'] == 'bike'
    identities = ['row_id', 'group_id']+(['timestamps'] if temporal else [])
    prediction = _arrays(prediction_bytes, ['prediction', *identities])
    _identities(prediction, temporal=temporal)
    values = prediction['prediction']
    if (values.shape != prediction['row_id'].shape
            or values.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(values).all()):
        raise ValueError('Finite one-dimensional float32/float64 predictions required')
    references = {manifest_path: expected_manifest_sha256, prediction_path: expected_prediction_sha256}

    def role_path(relative):
        path = root/relative
        if (not path.resolve().is_relative_to(root)
                or any(parent.is_symlink() for parent in (path, *path.parents) if parent != root and parent.is_relative_to(root))
                or relative not in manifest['files']):
            raise ValueError('Role evidence path escapes sealed store')
        digest = manifest['files'][relative]['sha256']
        references[path] = digest
        return path, digest

    def read_role(relative, fields):
        path, digest = role_path(relative)
        return _arrays(_content(path, digest), fields)

    features = read_role(f'outer-{outer_fold}/V-features.npz', ['X', *identities])
    _role_geometry(features, len(manifest['feature_names']), temporal=temporal, targets=False)
    if any(prediction[key].dtype != features[key].dtype
           or not np.array_equal(prediction[key], features[key]) for key in identities):
        raise ValueError('Prediction rows/groups/timestamps differ from primary V features')
    F = read_role(f'outer-{outer_fold}/F.npz', ['X', 'y', *identities])
    _role_geometry(F, len(manifest['feature_names']), temporal=temporal, targets=True)
    store._access(outer_fold, ['F_labels', 'V_features', 'V_group_or_day_ids'])
    labels_path, labels_digest = role_path(f'coordinator/outer-{outer_fold}/V-labels.npz')
    labels_bytes = _content(labels_path, labels_digest)
    for path, digest in references.items():
        _content(path, digest)
    V = store.coordinator_validation(outer_fold)
    # Validate the original closed label payload after the logged reader has
    # opened it: ObservationSet intentionally converts its y to float64, and
    # that legacy conversion must not hide malformed v2 source payloads here.
    labels = _arrays(labels_bytes, ['row_id', 'y'])
    if (labels['row_id'].dtype != np.dtype('int64')
            or not np.array_equal(labels['row_id'], features['row_id'])
            or labels['y'].dtype != np.dtype('float64')
            or labels['y'].shape != features['row_id'].shape
            or not np.isfinite(labels['y']).all()
            or not np.array_equal(labels['y'], V.y)):
        raise ValueError('Exact original-unit float64 V labels and row identities required')
    if (not np.array_equal(V.X, features['X'])
            or not np.array_equal(V.row_id, features['row_id'])
            or not np.array_equal(V.group_id, features['group_id'])
            or (temporal and not np.array_equal(V.dates, features['timestamps']))):
        raise ValueError('Coordinator role identities changed while scoring')
    arithmetic = prediction_arithmetic_v2.audit_prediction_arrays(
        F_y=F['y'], y=V.y, row_id=V.row_id,
        group_id=V.dates.astype('datetime64[D]').astype('int64') if temporal else V.group_id,
        grouping='day' if temporal else 'group', timestamps=V.dates,
        planned_methods=['candidate'], statuses={'candidate': 'completed_eligible'},
        predictions={'candidate': dict(row_id=prediction['row_id'], values=values.astype('float64'))})
    for path, digest in references.items():
        _content(path, digest)
    metrics = arithmetic['methods']['candidate']
    return dict(schema='research/v2-primary-validation-score', status=arithmetic['status'],
        dataset=manifest['dataset'], outer_fold=outer_fold, manifest_hash=manifest['manifest_hash'],
        split_hash=split['split_hash'], prediction_dtype=str(values.dtype), metric_dtype='float64',
        sealed_group_ids_sha256=array_hash(V.group_id),
        day_id_mapping=[dict(sealed_group_id=int(group), day_ordinal=int(day))
            for group, day in np.unique(np.column_stack([
                V.group_id, V.dates.astype('datetime64[D]').astype('int64')]), axis=0)] if temporal else None,
        grouping=arithmetic['grouping'], F_target_variance=arithmetic['F_target_variance'],
        score=metrics['normalized_risk'], metrics=metrics,
        sources=[dict(path=str(path), sha256=digest) for path, digest in sorted(references.items())],
        arithmetic=arithmetic, quality_eligible=False, selection_authorized=False,
        protected_access_authenticated=False, raw_acquisition_authenticated=False,
        model_generation_authenticated=False,
        scope='Primary coordinator byte binding and V arithmetic only; native model/qualification and search eligibility require separate validation')
