"""Group/temporal roles and semantic schemas precede predictive experimentation."""
import numpy as np
import pandas as pd
import pytest


def test_prepare_rejects_duplicate_configuration_keys_before_any_output(tmp_path):
    from treeneuron.research.transfer import prepare_data
    config = tmp_path/'config.json'
    config.write_text('{"schema":"research/v2-data-preparation","schema":"research/v2-data-preparation",'
        '"acquisition_manifest":"absent","acquisition_manifest_sha256":"absent",'
        '"datasets":["airfoil","concrete","ccpp","parkinsons","bike","energy"]}')
    with pytest.raises(ValueError, match='Duplicate'):
        prepare_data(config, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_semantic_predictors_exclude_proxy_targets_and_date_identity():
    from treeneuron.data.tabular import table_from_frame, DATASETS
    schema = DATASETS['parkinsons']
    frame = pd.DataFrame({name: np.arange(8, dtype=float)+1 for name in schema.raw_columns})
    frame['subject#'] = [1, 1, 2, 2, 3, 3, 4, 4]
    frame['sex'] = [0, 1]*4
    first = table_from_frame('parkinsons', frame)
    frame['motor_UPDRS'] = frame['motor_UPDRS'].iloc[::-1].to_numpy()
    frame['total_UPDRS'] += 1000
    again = table_from_frame('parkinsons', frame)
    assert first.feature_names == schema.features and first.X.shape == (8, 19)
    assert not {'motor_UPDRS', 'total_UPDRS', 'subject#'} & set(first.feature_names)
    np.testing.assert_array_equal(first.X, again.X)
    np.testing.assert_array_equal(first.group_id, again.group_id)
    assert not np.array_equal(first.y, again.y)
    assert first.categorical_features == ('sex',)


def test_group_folds_keep_duplicate_groups_and_energy_minima():
    from treeneuron.research.splits import grouped_splits
    groups = np.repeat(np.arange(12, dtype=np.int64), 64)
    first = grouped_splits('energy', groups)
    again = grouped_splits('energy', groups)
    assert first == again
    assert len(first) == 5 and sum(s['primary'] for s in first) == 1
    test_groups = []
    for split in first:
        roles = np.asarray(split['roles'])
        assert len(roles) == len(groups)
        for group in np.unique(groups):
            assert len(set(roles[groups == group])) == 1
        for role in 'FGV':
            assert len(np.unique(groups[roles == role])) >= 2
        test_groups.extend(np.unique(groups[roles == 'T']).tolist())
        assert sum(split['actual_rows'].values()) == len(groups)
        assert split['actual_rows']['F'] < .7*(len(groups)-split['actual_rows']['T'])
    assert sorted(test_groups) == list(range(12))
    unequal = np.repeat(np.arange(14, dtype=np.int64), [80, 40, 20, 10, 8, 7, 6, 5, 4, 4, 3, 3, 2, 1])
    assert len(grouped_splits('fixture', unequal)) == 5


def test_bike_embargo_uses_clock_hours_and_preserves_gaps_and_future_exclusion():
    from treeneuron.research.splits import bike_splits
    hours = np.arange(np.datetime64('2011-01-01T00', 'h'),
                      np.datetime64('2013-01-01T00', 'h'), np.timedelta64(1, 'h'))
    # Missing hours near test boundary make24rows an invalid embargo.
    hours = hours[~np.isin(hours, np.array(['2012-01-01T02', '2012-01-01T07'], dtype='datetime64[h]'))]
    splits = bike_splits(hours)
    assert len(splits) == 5 and [s['outer_fold'] for s in splits if s['primary']] == [4]
    first = splits[0]
    roles = np.asarray(first['roles'])
    assert int(np.sum((hours >= np.datetime64('2012-01-01')) & (hours < np.datetime64('2012-01-02')))) == 22
    assert set(roles[(hours >= np.datetime64('2012-01-01')) & (hours < np.datetime64('2012-01-02'))]) == {'embargo'}
    assert roles[np.flatnonzero(hours == np.datetime64('2012-01-02T00'))[0]] == 'T'
    assert set(roles[hours >= np.datetime64('2012-03-14')]) == {'unused_future'}
    for role in ('G', 'V', 'T'):
        indices = np.flatnonzero(roles == role)
        boundary = np.datetime64(first['role_boundaries'][role], 'h')
        assert hours[indices].min() >= boundary+np.timedelta64(24, 'h')
    assert hours[roles == 'F'].max() < hours[roles == 'G'].min() < hours[roles == 'V'].min() < hours[roles == 'T'].min()


def test_F_only_encoder_has_explicit_unknown_category_and_no_target_argument():
    from treeneuron.data.tabular import NumericEncoder
    x = np.asarray([[1., 0.], [3., 1.], [5., 0.]])
    encoder = NumericEncoder.fit(x, categorical_indices=(1,))
    result = encoder.transform(np.asarray([[100., 2.], [1., 0.]]))
    assert result.shape == (2, 4)  # numeric + two F categories +unknown
    np.testing.assert_array_equal(result[0, 1:], [0., 0., 1.])
    np.testing.assert_array_equal(result[1, 1:], [1., 0., 0.])
    assert result[0, 0] == pytest.approx((100-3)/np.std(x[:, 0], ddof=0))
    assert encoder.to_dict()['fit_rows'] == 3
    with pytest.raises(TypeError):
        NumericEncoder.fit(x, y=np.ones(3), categorical_indices=(1,))


def test_transfer_store_blocks_T_and_secondary_development_before_freeze(tmp_path):
    from treeneuron.data.tabular import table_from_frame, DATASETS
    from treeneuron.research.transfer import seal_table, TransferStore
    schema = DATASETS['energy']
    frame = pd.DataFrame({name: np.arange(24, dtype=float) for name in schema.raw_columns})
    # Twelve geometry groups, two rows each; orientation/glazing varies inside.
    for name in ('X1', 'X2', 'X3', 'X4', 'X5'):
        frame[name] = np.repeat(np.arange(12), 2)
    table = table_from_frame('energy', frame)
    manifest = seal_table(table, tmp_path/'sealed')
    assert manifest['protected_labels_read_for_modeling'] is False
    store = TransferStore(tmp_path/'sealed')
    view = store.development(0)
    assert len(view.F.y) and len(view.G.y) and view.validation_features.ndim == 2
    with pytest.raises(PermissionError):
        _ = view.V
    with pytest.raises(PermissionError):
        store.labels(0, 'T')
    with pytest.raises(PermissionError):
        store.development(1)
    assert np.isfinite(store.labels(0, 'V')).all()
    metadata = store.manifest
    metadata['splits'][1]['primary'] = True
    with pytest.raises(PermissionError):
        store.development(1)
    metadata['files']['outer-0/F.npz']['sha256'] = '0'*64
    assert len(store.development(0).F.y) == len(view.F.y)
    validation = store.coordinator_validation(0)
    np.testing.assert_array_equal(validation.X, view.validation_features)
    np.testing.assert_array_equal(validation.row_id, view.validation_row_ids)
    assert len(np.unique(validation.group_id)) >= 2
    assert validation.y.shape == validation.row_id.shape
    with pytest.raises(ValueError):
        seal_table(table, tmp_path/'sealed')
