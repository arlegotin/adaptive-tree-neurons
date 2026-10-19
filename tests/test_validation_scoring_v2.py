"""Primary V coordinator arithmetic; artificial sealed tables only."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from treeneuron.data.tabular import DATASETS, table_from_frame
from treeneuron.config import canonical_hash
from treeneuron.research.transfer import TransferStore, seal_table


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive(path, values):
    with path.open('wb') as stream:
        np.savez_compressed(stream, **values)


def _case(tmp_path, name='energy', constant=False):
    schema = DATASETS[name]
    if name == 'energy':
        groups = np.repeat(np.arange(12), np.arange(2, 14))
        frame = pd.DataFrame({key: np.arange(len(groups), dtype=float)+1 for key in schema.raw_columns})
        for key in ('X1', 'X2', 'X3', 'X4', 'X5'):
            frame[key] = groups
        fold = 0
    else:
        hours = np.arange(np.datetime64('2011-01-01T00', 'h'),
                          np.datetime64('2013-01-01T00', 'h'), np.timedelta64(1, 'h'))
        # Unequal hour counts inside primary V, without manufacturing rows.
        hours = hours[~np.isin(hours, np.array(['2012-09-01T02', '2012-09-01T07'], dtype='datetime64[h]'))]
        frame = pd.DataFrame({key: np.arange(len(hours), dtype=float)+1 for key in schema.raw_columns})
        frame['dteday'] = hours.astype('datetime64[D]').astype(str)
        frame['hr'] = (hours-hours.astype('datetime64[D]')).astype('timedelta64[h]').astype(int)
        fold = 4
    if constant:
        frame[schema.target] = 7.
    root = tmp_path/'sealed'
    seal_table(table_from_frame(name, frame), root)
    with np.load(root/f'outer-{fold}/V-features.npz', allow_pickle=False) as saved:
        metadata = {key: saved[key].copy() for key in saved.files if key != 'X'}
    with np.load(root/f'coordinator/outer-{fold}/V-labels.npz', allow_pickle=False) as saved:
        target = saved['y'].copy()
    units = np.unique(metadata['group_id'])
    errors = np.asarray([1. if group == units[0] else 3. for group in metadata['group_id']])
    prediction = tmp_path/'prediction.npz'
    _archive(prediction, dict(metadata, prediction=target+errors))
    kwargs = dict(root=root, outer_fold=fold, expected_manifest_sha256=_sha(root/'manifest.json'),
                  prediction_path=prediction, expected_prediction_sha256=_sha(prediction))
    return kwargs, metadata, target, errors


def _score(**kwargs):
    from treeneuron.research.validation_scoring_v2 import score_primary_validation_prediction
    return score_primary_validation_prediction(**kwargs)


@pytest.mark.parametrize('name', ['energy', 'bike'])
def test_primary_group_or_day_metric_and_logged_coordinator_scope(tmp_path, name):
    kwargs, metadata, target, errors = _case(tmp_path, name)
    result = _score(**kwargs)
    groups = metadata['group_id']
    mse = np.mean([np.mean(errors[groups == group]**2) for group in np.unique(groups)])
    mae = np.mean([np.mean(abs(errors[groups == group])) for group in np.unique(groups)])
    with np.load(kwargs['root']/f"outer-{kwargs['outer_fold']}/F.npz", allow_pickle=False) as saved:
        variance = np.var(saved['y'], ddof=0)
    assert result['metrics']['primary_MSE'] == pytest.approx(mse)
    assert result['metrics']['primary_RMSE'] == pytest.approx(np.sqrt(mse))
    assert result['metrics']['primary_MAE'] == pytest.approx(mae)
    assert result['metrics']['row_RMSE'] == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert result['metrics']['row_MAE'] == pytest.approx(np.mean(abs(errors)))
    assert result['score'] == pytest.approx(mse/variance)
    assert result['F_target_variance'] == pytest.approx(variance)
    assert result['grouping'] == ('day' if name == 'bike' else 'group')
    assert result['prediction_dtype'] == 'float64' and result['metric_dtype'] == 'float64'
    assert all(result[key] is False for key in ('quality_eligible', 'selection_authorized',
        'protected_access_authenticated', 'raw_acquisition_authenticated', 'model_generation_authenticated'))
    events = [json.loads(line) for line in (kwargs['root']/'access.jsonl').read_text().splitlines()]
    assert [event['roles'] for event in events] == [
        ['F_labels', 'V_features', 'V_group_or_day_ids'], ['V_labels', 'V_features', 'V_group_or_day_ids']]
    assert not any('T' in role for event in events for role in event['roles'])
    assert len(result['sources']) == 5


def test_float32_prediction_is_explicit_metric_upcast_only(tmp_path):
    kwargs, metadata, target, errors = _case(tmp_path)
    prediction = (target+errors).astype('float32')
    _archive(kwargs['prediction_path'], dict(metadata, prediction=prediction))
    kwargs['expected_prediction_sha256'] = _sha(kwargs['prediction_path'])
    result = _score(**kwargs)
    assert result['prediction_dtype'] == 'float32' and result['metric_dtype'] == 'float64'
    assert result['metrics']['row_MSE'] == pytest.approx(np.mean((prediction.astype('float64')-target)**2))


@pytest.mark.parametrize('change', ['rows', 'duplicates', 'groups', 'extra_labels', 'object',
    'integer', 'nan', 'column', 'prediction_hash', 'manifest_hash', 'temporal_metadata'])
def test_invalid_prediction_rejects_before_V_labels(tmp_path, monkeypatch, change):
    kwargs, metadata, target, errors = _case(tmp_path)
    values = dict(metadata, prediction=target+errors)
    if change == 'rows': values['row_id'] = values['row_id'][::-1]
    elif change == 'duplicates': values['row_id'][:] = values['row_id'][0]
    elif change == 'groups': values['group_id'] = values['group_id']+1
    elif change == 'extra_labels': values['y'] = target
    elif change == 'object': values['prediction'] = values['prediction'].astype(object)
    elif change == 'integer': values['prediction'] = values['prediction'].astype('int64')
    elif change == 'nan': values['prediction'][0] = np.nan
    elif change == 'column': values['prediction'] = values['prediction'][:, None]
    elif change == 'temporal_metadata': values['timestamps'] = np.arange(len(target)).astype('datetime64[h]')
    _archive(kwargs['prediction_path'], values)
    kwargs['expected_prediction_sha256'] = _sha(kwargs['prediction_path'])
    if change == 'prediction_hash': kwargs['expected_prediction_sha256'] = '0'*64
    if change == 'manifest_hash': kwargs['expected_manifest_sha256'] = '0'*64
    def forbidden(*args, **kw):
        pytest.fail('Invalid prediction opened V labels')
    monkeypatch.setattr(TransferStore, 'coordinator_validation', forbidden)
    with pytest.raises(ValueError): _score(**kwargs)


def test_secondary_and_protected_prediction_path_reject_before_any_label_reader(tmp_path, monkeypatch):
    kwargs, _, _, _ = _case(tmp_path)
    def forbidden(*args, **kw):
        pytest.fail('Forbidden role opened')
    monkeypatch.setattr(TransferStore, '_read', forbidden)
    monkeypatch.setattr(TransferStore, 'coordinator_validation', forbidden)
    with pytest.raises(PermissionError): _score(**dict(kwargs, outer_fold=1))
    protected = kwargs['root']/'protected/outer-0/T-labels.npz'
    with pytest.raises(ValueError):
        _score(**dict(kwargs, prediction_path=protected, expected_prediction_sha256=_sha(protected)))
    assert not (kwargs['root']/'access.jsonl').exists()


def test_zero_F_variance_preserves_original_metrics_and_unavailable_score(tmp_path):
    kwargs, _, _, _ = _case(tmp_path, constant=True)
    result = _score(**kwargs)
    assert result['status'] == 'undefined_F_variance' and result['score'] is None
    assert result['F_target_variance'] == 0 and result['metrics']['primary_MSE'] > 0


@pytest.mark.parametrize('name', ['prediction', 'manifest', 'F', 'V_features', 'V_labels'])
def test_late_reference_drift_rejects_result(tmp_path, monkeypatch, name):
    from treeneuron.analysis import prediction_arithmetic_v2
    kwargs, _, _, _ = _case(tmp_path)
    real = prediction_arithmetic_v2.audit_prediction_arrays
    paths = dict(prediction=kwargs['prediction_path'], manifest=kwargs['root']/'manifest.json',
                 F=kwargs['root']/'outer-0/F.npz', V_features=kwargs['root']/'outer-0/V-features.npz',
                 V_labels=kwargs['root']/'coordinator/outer-0/V-labels.npz')
    def drift(**kw):
        result = real(**kw)
        with paths[name].open('ab') as stream: stream.write(b'changed')
        return result
    monkeypatch.setattr(prediction_arithmetic_v2, 'audit_prediction_arrays', drift)
    with pytest.raises(ValueError, match='changed|commitment|drift'): _score(**kwargs)


def test_Bike_timestamp_mismatch_rejects_before_labels(tmp_path, monkeypatch):
    kwargs, metadata, target, errors = _case(tmp_path, 'bike')
    metadata['timestamps'] = metadata['timestamps']+np.timedelta64(1, 'h')
    _archive(kwargs['prediction_path'], dict(metadata, prediction=target+errors))
    kwargs['expected_prediction_sha256'] = _sha(kwargs['prediction_path'])
    def forbidden(*args, **kw): pytest.fail('Mismatched clock hours opened V labels')
    monkeypatch.setattr(TransferStore, 'coordinator_validation', forbidden)
    with pytest.raises(ValueError): _score(**kwargs)


@pytest.mark.parametrize('change', ['float32_labels', 'extra_field'])
def test_logged_V_labels_reject_silent_cast_or_unexpected_fields(tmp_path, change):
    kwargs, _, _, _ = _case(tmp_path)
    relative = 'coordinator/outer-0/V-labels.npz'
    path = kwargs['root']/relative
    with np.load(path, allow_pickle=False) as saved:
        fields = {key: saved[key].copy() for key in saved.files}
    if change == 'float32_labels': fields['y'] = fields['y'].astype('float32')
    else: fields['unknown'] = np.ones_like(fields['y'])
    _archive(path, fields)
    # This is an artificial malformed sealed source, with a new explicit byte
    # identity; it must fail semantics rather than merely stale file hashes.
    manifest_path = kwargs['root']/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][relative] = dict(sha256=_sha(path), bytes=path.stat().st_size)
    del manifest['manifest_hash']
    manifest['manifest_hash'] = canonical_hash(manifest)
    manifest_path.write_text(json.dumps(manifest))
    kwargs['expected_manifest_sha256'] = _sha(manifest_path)
    with pytest.raises(ValueError): _score(**kwargs)
    events = [json.loads(line) for line in (kwargs['root']/'access.jsonl').read_text().splitlines()]
    assert events[-1]['roles'][0] == 'V_labels'
