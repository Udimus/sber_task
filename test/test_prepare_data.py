"""
Test for module/prepare_data.py
"""
from unittest.mock import patch

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from module.prepare_data import (
    BinFeaturesTransformer,
    CatFeaturesTransformer,
)


TEST_CAT_FEATURES = ['cat_feat']
TEST_NUM_FEATURES = ['num_feat']
TEST_BIN_FEATURES = ['bin_feat_1', 'bin_feat_2']
TEST_BIN_DF = pd.DataFrame({
    'cat_feat': ['A', 'B', 'C', 'D', 'E'],
    'num_feat': [0, 1, 2, 3, 4],
    'bin_feat_1': [0, 1, 0, 1, 0],
    'bin_feat_2': ['A', 'A', 'A', 'B', 'B'],
})
TEST_TRANSFORMED_DF = pd.DataFrame({
    'cat_feat': ['A', 'B', 'C', 'D', 'E'],
    'num_feat': [0, 1, 2, 3, 4],
    'bin_feat_1': [0, 1, 0, 1, 0],
    'bin_feat_2': [0, 0, 0, 1, 1],
})


class TestBinFeaturesTransformer:
    def test_bin_as_numeric(self):
        transformer = BinFeaturesTransformer(
            num_features=TEST_NUM_FEATURES,
            cat_features=TEST_CAT_FEATURES,
            bin_features=TEST_BIN_FEATURES,
            bin_as_numeric=True,
        )
        assert_frame_equal(TEST_TRANSFORMED_DF,
                           transformer.fit_transform(TEST_BIN_DF))

    def test_bin_as_cat(self):
        transformer = BinFeaturesTransformer(
            num_features=TEST_NUM_FEATURES,
            cat_features=TEST_CAT_FEATURES,
            bin_features=TEST_BIN_FEATURES,
            bin_as_numeric=False,
        )
        assert_frame_equal(TEST_BIN_DF, transformer.fit_transform(TEST_BIN_DF))

    @pytest.mark.parametrize('bin_as_num', [True, False])
    def test_get_features(self, bin_as_num):
        transformer = BinFeaturesTransformer(
            num_features=TEST_NUM_FEATURES,
            cat_features=TEST_CAT_FEATURES,
            bin_features=TEST_BIN_FEATURES,
            bin_as_numeric=bin_as_num,
        )
        transformer.fit(TEST_BIN_DF)
        features = transformer.get_features()
        if bin_as_num:
            assert features['cat_features'] == TEST_CAT_FEATURES
            assert features['num_features'] == (TEST_NUM_FEATURES
                                                + TEST_BIN_FEATURES)
        else:
            assert features['cat_features'] == (TEST_CAT_FEATURES
                                                + TEST_BIN_FEATURES)
            assert features['num_features'] == TEST_NUM_FEATURES


TEST_CAT_DF = pd.DataFrame({
    'cat_1': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
    'cat_2': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    'cat_3': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
    'cat_4': ['A', 'B', 'C', 'A', 'A', 'B', 'A', 'B', 'B', 'D'],
    'num_1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})
CAT_FEATURES = [col for col in TEST_CAT_DF.columns if col.startswith('cat')]
TEST_CAT_TRANSFORMED_DF = pd.DataFrame({
    'cat_1': [0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6],
    'cat_2': [0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6],
    'cat_3': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'cat_4': [0.25, 0.75, 0, 0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 1],
    'num_1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})
TEST_CAT_ALPHA_TRANSFORMED_DF = pd.DataFrame({
    'cat_1': [7/15, 7/15, 7/15, 7/15, 7/15,
              8/15, 8/15, 8/15, 8/15, 8/15],
    'cat_2': [7/15, 7/15, 7/15, 7/15, 7/15,
              8/15, 8/15, 8/15, 8/15, 8/15],
    'cat_3': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'cat_4': [6/14, 8/14, 5/11, 6/14, 6/14, 8/14, 6/14, 8/14, 8/14, 6/11],
    'num_1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})
TEST_CAT_FIT_TRANSFORMED_DF_EXP = pd.DataFrame({
    'cat_1': [0, 0, 0.5, 1/3, 0.5, 0, 1, 0.5, 2/3, 0.5],
    'cat_2': [0, 0, 0.5, 1/3, 0.5, 0, 1, 0.5, 2/3, 0.5],
    'cat_3': [0, 0, 0.5, 1/3, 0.5, 2/5, 0.5, 3/7, 0.5, 4/9],
    'cat_4': [0, 0, 0, 0, 0.5, 1, 1/3, 1, 1, 0],
    'num_1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})
TEST_CAT_FIT_TRANSFORMED_DF_EXP = pd.DataFrame({
    'cat_1': [0, 0, 0.5, 1/3, 0.5, 0, 1, 0.5, 2/3, 0.5],
    'cat_2': [0, 0, 0.5, 1/3, 0.5, 0, 1, 0.5, 2/3, 0.5],
    'cat_3': [0, 0, 0.5, 1/3, 0.5, 2/5, 0.5, 3/7, 0.5, 4/9],
    'cat_4': [0, 0, 0, 0, 0.5, 1, 1/3, 1, 1, 0],
    'num_1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})
TEST_CAT_FIT_TRANSFORMED_DF_KFOLD = pd.DataFrame({
    'cat_1': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'cat_2': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'cat_3': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'cat_4': [1, 0, 0.5, 0, 1, 0, 1, 0, 1, 0.5],
    'num_1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})


class KFoldMocker:
    def __init__(self, n_splits, *args, **kwargs):
        self._n_splits = n_splits

    def split(self, df, *args, **kwargs):
        all_indices = list(range(len(df)))
        for i in range(self._n_splits):
            batch = list(range(i, len(df), self._n_splits))
            remaining_part = [i for i in all_indices if i not in batch]
            yield remaining_part, batch


class TestCatFeaturesTransformer:
    def test_fit(self):
        transformer = CatFeaturesTransformer(cat_features=CAT_FEATURES)
        print(TEST_CAT_DF)
        y = TEST_CAT_DF['target']
        X = TEST_CAT_DF.drop(columns=['target'])
        transformer.fit(X, y)
        assert transformer._global_mean == 0.5
        for col in CAT_FEATURES:
            assert col in transformer._encodings

    def test_transform(self):
        transformer = CatFeaturesTransformer(
            alpha=0,
            cat_features=CAT_FEATURES,
            expanding=True)
        y = TEST_CAT_DF['target']
        X = TEST_CAT_DF.drop(columns=['target'])
        transformer.fit(X, y)
        transformed_df = transformer.transform(X)
        assert_frame_equal(TEST_CAT_TRANSFORMED_DF,
                           transformed_df,
                           check_dtype=False)

    def test_alpha_transform(self):
        transformer = CatFeaturesTransformer(
            alpha=10,
            cat_features=CAT_FEATURES,
            expanding=True)
        y = TEST_CAT_DF['target']
        X = TEST_CAT_DF.drop(columns=['target'])
        transformer.fit(X, y)
        transformed_df = transformer.transform(X)
        assert_frame_equal(TEST_CAT_ALPHA_TRANSFORMED_DF,
                           transformed_df,
                           check_dtype=False)

    def test_fit_transform_exp(self):
        transformer = CatFeaturesTransformer(
            alpha=0,
            cat_features=CAT_FEATURES,
            expanding=True)
        y = TEST_CAT_DF['target']
        X = TEST_CAT_DF.drop(columns=['target'])
        transformed_df = transformer.fit_transform(X, y)
        assert_frame_equal(TEST_CAT_FIT_TRANSFORMED_DF_EXP,
                           transformed_df,
                           check_dtype=False)

    @patch('module.prepare_data.StratifiedKFold')
    def test_fit_transform_kfold(self, mock_kfold):
        mock_kfold.side_effect = KFoldMocker
        transformer = CatFeaturesTransformer(
            folds_number=2,
            alpha=0,
            cat_features=CAT_FEATURES,
            expanding=False)
        y = TEST_CAT_DF['target']
        X = TEST_CAT_DF.drop(columns=['target'])
        transformed_df = transformer.fit_transform(X, y)
        assert_frame_equal(TEST_CAT_FIT_TRANSFORMED_DF_KFOLD,
                           transformed_df,
                           check_dtype=False)
