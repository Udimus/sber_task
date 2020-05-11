"""
Test for module/model.py
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pandas as pd
from pandas.testing import assert_frame_equal
from catboost import Pool
from lightgbm import Dataset

from module.model import (
    our_loss_function,
    our_loss_lgb_objective,
    OurLossCBObjective,
    CatboostWrapper,
    LightgbmWrapper,
)

TEST_OUR_LOSS_FUNCTION_PARAMS = [
    (
        np.array([0]),
        np.array([6]),
        36,
        np.array([12]),
        np.array([2]),
    ),
    (
        np.array([0]),
        np.array([-6]),
        6,
        np.array([-1]),
        np.array([0]),
    ),
    (
        np.array([0, 0]),
        np.array([-6, 6]),
        21,
        np.array([-1, 12]),
        np.array([0, 2]),
    )
]
TEST_OUR_LOSS_FUNCTION_PARAM_NAMES = ','.join([
    'y_true',
    'y_pred',
    'loss',
    'grad',
    'hess',
])


@pytest.mark.parametrize(TEST_OUR_LOSS_FUNCTION_PARAM_NAMES,
                         TEST_OUR_LOSS_FUNCTION_PARAMS)
def test_our_loss_function(y_true, y_pred, loss, grad, hess):
    assert loss == our_loss_function(y_true, y_pred)


@pytest.mark.parametrize(TEST_OUR_LOSS_FUNCTION_PARAM_NAMES,
                         TEST_OUR_LOSS_FUNCTION_PARAMS)
def test_our_loss_lgb_objective(y_true, y_pred, loss, grad, hess):
    test_grad, test_hess = our_loss_lgb_objective(y_true, y_pred)
    assert_array_equal(test_grad, grad)
    assert_array_equal(test_hess, hess)


@pytest.mark.parametrize(TEST_OUR_LOSS_FUNCTION_PARAM_NAMES,
                         TEST_OUR_LOSS_FUNCTION_PARAMS)
def test_our_loss_lgb_objective(y_true, y_pred, loss, grad, hess):
    objective = OurLossCBObjective()
    result = objective.calc_ders_range(y_pred, y_true)
    test_grad, test_hess = zip(*result)
    test_grad = -np.array(test_grad)
    test_hess = -np.array(test_hess)
    assert_array_equal(test_grad, grad)
    assert_array_equal(test_hess, hess)


TEST_REGRESSORS_DF = pd.DataFrame({
    'target': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'cat_feat': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'],
    'num_feat': [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
})
TEST_REGRESSORS_POSSIBLE_PREDICTION = [1.5, 1.5, 4, 4, 4, 6.5, 6.5, 9, 9, 9]
CATBOOST_PARAMS = [
    {},
    # {'loss_function': OurLossCBObjective}
]


class TestCatboostWrapper:
    def test_prepare_pool(self):
        clf = CatboostWrapper(cat_features=['cat_feat'])
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        pool = clf._prepare_pool(X, y)
        assert isinstance(pool, Pool)
        assert_array_equal(pool.get_label(), y.values)
        assert pool.num_col() == X.shape[1]
        assert pool.num_row() == X.shape[0]

    @pytest.mark.parametrize('params', CATBOOST_PARAMS)
    def test_fit(self, params):
        clf = CatboostWrapper(cat_features=['cat_feat'], **params)
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        clf.fit(X, y, verbose=False)

    @pytest.mark.parametrize('params', CATBOOST_PARAMS)
    def test_predict(self, params):
        clf = CatboostWrapper(cat_features=['cat_feat'], **params)
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        clf.fit(X, y, verbose=False)
        possible_prediction = np.array(TEST_REGRESSORS_POSSIBLE_PREDICTION)
        assert_allclose(possible_prediction, clf.predict(X),
                        rtol=0.01, atol=0.1)

    @pytest.mark.parametrize('params', CATBOOST_PARAMS)
    def test_fit_predict(self, params):
        clf = CatboostWrapper(cat_features=['cat_feat'], **params)
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        possible_prediction = np.array(TEST_REGRESSORS_POSSIBLE_PREDICTION)
        assert_allclose(possible_prediction,
                        clf.fit(X, y, verbose=False).predict(X),
                        rtol=0.01, atol=0.1)


LIGHTGBM_PARAMS = [
    {'min_child_samples': 1},
]


class TestLightgbmWrapper:
    def test_prepare_pool(self):
        clf = LightgbmWrapper(cat_features=['cat_feat'])
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        new_X = clf._prepare_pool(X, y)
        assert_array_equal(X.values, new_X.values)
        assert new_X['cat_feat'].dtype.name == 'category'

    @pytest.mark.parametrize('params', LIGHTGBM_PARAMS)
    def test_fit(self, params):
        clf = LightgbmWrapper(cat_features=['cat_feat'], **params)
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        clf.fit(X, y, verbose=False)

    @pytest.mark.parametrize('params', LIGHTGBM_PARAMS)
    def test_predict(self, params):
        clf = LightgbmWrapper(cat_features=['cat_feat'], **params)
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        clf.fit(X, y, verbose=False)
        possible_prediction = np.array(TEST_REGRESSORS_POSSIBLE_PREDICTION)
        assert_allclose(possible_prediction, clf.predict(X),
                        rtol=0.01, atol=0.1)

    @pytest.mark.parametrize('params', LIGHTGBM_PARAMS)
    def test_fit_predict(self, params):
        clf = LightgbmWrapper(cat_features=['cat_feat'], **params)
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        possible_prediction = np.array(TEST_REGRESSORS_POSSIBLE_PREDICTION)
        assert_allclose(possible_prediction,
                        clf.fit(X, y, verbose=False).predict(X),
                        rtol=0.01, atol=0.1)
