"""Fail closed at the development boundary, including malformed namespaces."""
from dataclasses import replace
import numpy as np
import pytest

from treeneuron.data.synthetic import make_teacher, sample_teacher, make_development
from treeneuron.data.types import ObservationSet, SplitManifest, predictor_matrix


def test_features_exclude_oracle_information():
    rows = sample_teacher(make_teacher(71), n=5, seed=72, sigma=.1)
    assert rows.X.shape == (5, 16)
    assert rows.y.shape == (5,)
    assert rows.feature_names == tuple(f'x{i}' for i in range(16))
    np.testing.assert_array_equal(predictor_matrix(rows), rows.X)
    for name in ['y', 'noise_free_target', 'gates', 'teacher', 'reference_mean', 'row_id', 'group_id']:
        with pytest.raises(ValueError, match='allowlist'):
            predictor_matrix(rows, [name])
    assert predictor_matrix(rows, ['x2', 'x0']).shape == (5, 2)


@pytest.mark.parametrize('counts,seeds', [
    ({'F': 2, 'G': 2, 'V': 2, 'T': 2}, {'F': 1, 'G': 2, 'V': 3}),
    ({'F': 2, 'G': 2, 'V': 2}, {'F': 1, 'G': 1, 'V': 3}),
    ({'F': 2, 'G': 2, 'V': 2}, {'exploratory/F': 1, 'G': 2, 'V': 3}),
    ({'F': 2, 'G': 2, 'V': 2}, {'F': 1, 'G': -1, 'V': 3}),
    ({'F': 0, 'G': 2, 'V': 2}, {'F': 1, 'G': 2, 'V': 3}),
])
def test_development_rejects_protected_malformed_and_overlapping_namespaces(counts, seeds):
    with pytest.raises(ValueError):
        make_development(make_teacher(1), counts, seeds, .1)


def test_development_has_no_test_path_interface():
    teacher = make_teacher(1)
    with pytest.raises(TypeError):
        make_development(teacher, test_path='/does/not/matter')


def test_manifest_rejects_duplicate_rows_and_cross_partition_groups():
    kwargs = dict(dataset_id='d', teacher_id='t', phase='exploratory', fold=None)
    with pytest.raises(ValueError):
        SplitManifest(row_id=(1, 1), group_id=(1, 2), assignments=('F', 'G'), **kwargs)
    with pytest.raises(ValueError):
        SplitManifest(row_id=(1, 2), group_id=(9, 9), assignments=('F', 'G'), **kwargs)
    rows = sample_teacher(make_teacher(1), 2, 3, .1)
    with pytest.raises(ValueError):
        replace(rows, y=np.array([1., np.nan]))
    with pytest.raises(ValueError):
        replace(rows, row_id=np.array([1, 1], dtype=np.int64))


def test_development_detects_reused_observation_rows():
    data = make_development(make_teacher(1), {'F': 2, 'G': 2, 'V': 2}, {'F': 1, 'G': 2, 'V': 3})
    with pytest.raises(ValueError):
        replace(data, G=data.F)


def test_development_rejects_mismatched_feature_schemas():
    data = make_development(make_teacher(1), {'F': 2, 'G': 2, 'V': 2}, {'F': 1, 'G': 2, 'V': 3})
    wrong = replace(data.G, feature_names=tuple(f'other{i}' for i in range(16)))
    with pytest.raises(ValueError):
        replace(data, G=wrong, manifest=None)
