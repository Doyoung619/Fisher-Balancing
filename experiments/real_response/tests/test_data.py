import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from real_response.data import _chronological, _encode_features, _pca, _subsample, _zip_read


def test_chronological_preserves_boundary_timestamp_group():
    times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 8, 9])
    p, t, meta = _chronological(times)
    assert times[p].max() < times[t].min()
    assert set(p).isdisjoint(t)
    assert set(np.r_[p, t]) == set(range(10))


def test_pool_feature_fit_is_independent_of_test_covariates():
    frame = pd.DataFrame({'value': [1., 2., np.nan] * 10 + [100., np.nan],
                          'category': ['a'] * 15 + ['b'] * 15 + ['unseen', 'a']})
    pool = np.arange(30); test = np.array([30, 31])
    A, B, artifact = _encode_features(frame, pool, test, ['value'], ['category'], [], 'other')
    changed = frame.copy(); changed.loc[test, 'value'] = [-1e12, 1e12]
    changed.loc[test, 'category'] = ['changed', 'changed']
    AA, BB, altered = _encode_features(changed, pool, test, ['value'], ['category'], [], 'other')
    np.testing.assert_array_equal(A, AA)
    assert artifact['numeric'] == altered['numeric']
    assert artifact['categorical']['category']['kept_levels'] == ['a', 'b']
    assert not np.array_equal(B, BB)


def test_nested_unwhitened_pca_pool_scaling_and_sign():
    rng = np.random.default_rng(8)
    A = rng.normal(size=(90, 25)) * np.linspace(.2, 5, 25)
    B = rng.normal(size=(12, 25))
    X, Xt, model, rank = _pca(A, B)
    XX, XXt, other, _ = _pca(A, B * 10000)
    np.testing.assert_array_equal(X, XX)
    assert np.linalg.norm(X, axis=1).max() <= 1 + 1e-12
    assert np.linalg.norm(XXt, axis=1).max() > 1
    assert np.ptp(X[:, 0]) == 0
    assert np.std(X[:, 1]) != pytest.approx(np.std(X[:, 19]))
    for row in model['components']:
        assert row[np.argmax(np.abs(row))] > 0
    for d in (5, 10, 20):
        assert np.linalg.matrix_rank(X[:, :d]) == d


def test_hash_subsample_deterministic_and_label_independent():
    ids = np.array([f'row:{i}' for i in range(100)])
    x = _subsample(np.arange(100), ids, 20, 20261004, 'dataset')
    y = _subsample(np.arange(100), ids, 20, 20261004, 'dataset')
    np.testing.assert_array_equal(x, y)
    assert len(set(x)) == 20


def test_archive_path_traversal_is_rejected(tmp_path):
    path = tmp_path / 'bad.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('../outside.tsv', 'secret')
    with pytest.raises(RuntimeError, match='Unsafe'):
        _zip_read(path, '../outside.tsv')
