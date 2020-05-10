"""
Test for module/prepare_data.py
"""
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from module.prepare_data import (
    BinFeaturesTransformer
)

TEST_CAT_FEATURES = ['cat_feat']
TEST_NUM_FEATURES = ['num_feat']
TEST_BIN_FEATURES = ['bin_feat_1', 'bin_feat_2']
TEST_DF = pd.DataFrame({
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
        assert_frame_equal(TEST_TRANSFORMED_DF, transformer.fit_transform(TEST_DF))

    def test_bin_as_cat(self):
        transformer = BinFeaturesTransformer(
            num_features=TEST_NUM_FEATURES,
            cat_features=TEST_CAT_FEATURES,
            bin_features=TEST_BIN_FEATURES,
            bin_as_numeric=False,
        )
        assert_frame_equal(TEST_DF, transformer.fit_transform(TEST_DF))
