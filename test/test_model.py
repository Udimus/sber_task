"""
Test for module/model.py
"""
from functools import reduce

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pandas as pd
from pandas.testing import assert_frame_equal
from catboost import Pool
from lightgbm import Dataset

from module.model import (
    our_loss_function,
    our_loss_lgbm_objective,
    OurLossCBObjective,
    OurLossCBMetric,
    CatboostWrapper,
    LightgbmWrapper,
    OUR_LOSS_HESS_CONST,
    get_our_loss_best_const,
)

TEST_OUR_LOSS_INPUTS = [
    (
        np.array([0]),
        np.array([6]),
    ),
    (
        np.array([0]),
        np.array([-6]),
    ),
    (
        np.array([0, 0]),
        np.array([-6, 6]),
    )
]
TEST_OUR_LOSS_VALUES = [
    (
        36,
    ),
    (
        6,
    ),
    (
        21,
    )
]
TEST_OUR_LOSS_LGBM_GRAD_HESS = [
    (
        np.array([12]),
        np.array([2]),
    ),
    (
        np.array([-1]),
        np.array([OUR_LOSS_HESS_CONST]),
    ),
    (
        np.array([-1, 12]),
        np.array([OUR_LOSS_HESS_CONST, 2]),
    )
]
TEST_OUR_LOSS_CATBOOST_GRAD_HESS = [
    (
        np.array([12]),
        np.array([2]),
    ),
    (
        np.array([-1]),
        np.array([0]),
    ),
    (
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


def join_params(list_of_params):
    result = []
    for tuples in zip(*list_of_params):
        result.append(reduce(lambda x, y: x + y, tuples))
    return result


@pytest.mark.parametrize('y_true,y_pred,loss',
                         join_params([TEST_OUR_LOSS_INPUTS,
                                      TEST_OUR_LOSS_VALUES]))
def test_our_loss_function(y_true, y_pred, loss):
    assert loss == our_loss_function(y_true, y_pred)


@pytest.mark.parametrize('y_true,y_pred,grad,hess',
                         join_params([TEST_OUR_LOSS_INPUTS,
                                      TEST_OUR_LOSS_LGBM_GRAD_HESS]))
def test_our_loss_lgb_objective(y_true, y_pred, grad, hess):
    test_grad, test_hess = our_loss_lgbm_objective(y_true, y_pred)
    assert_array_equal(test_grad, grad)
    assert_array_equal(test_hess, hess)


@pytest.mark.parametrize('y_true,y_pred,grad,hess',
                         join_params([TEST_OUR_LOSS_INPUTS,
                                      TEST_OUR_LOSS_CATBOOST_GRAD_HESS]))
def test_our_loss_cb_objective(y_true, y_pred, grad, hess):
    objective = OurLossCBObjective()
    result = objective.calc_ders_range(y_pred, y_true)
    test_grad, test_hess = zip(*result)
    test_grad = -np.array(test_grad)
    test_hess = -np.array(test_hess)
    assert_array_equal(test_grad, grad)
    assert_array_equal(test_hess, hess)


@pytest.mark.parametrize('y_true,y_pred,loss',
                         join_params([TEST_OUR_LOSS_INPUTS,
                                      TEST_OUR_LOSS_VALUES]))
def test_loss_cb_metric(y_true, y_pred, loss):
    metric = OurLossCBMetric()
    assert metric.is_max_optimal() is False
    error_sum, weight_sum = metric.evaluate([y_pred], y_true)
    assert metric.get_final_error(error_sum, weight_sum) == loss


TEST_REGRESSORS_DF = pd.DataFrame({
    'target': [1, 2, 3, 4, 5, 6, 7, 8, 9, 13],
    'cat_feat': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'],
    'num_feat': [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
})
CATBOOST_TEST_PARAMS = [
    (
        {},
        [1.5, 1.5, 4, 4, 4, 6.5, 6.5, 10, 10, 10]
    ),
    (
        {
            'loss_function': OurLossCBObjective(),
            'eval_metric': OurLossCBMetric(),
        },
        [1.5, 1.5, 4, 4, 4, 6.5, 6.5, 9, 9, 9]
    )

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

    @pytest.mark.parametrize('params,prediction', CATBOOST_TEST_PARAMS)
    def test_fit(self, params, prediction):
        clf = CatboostWrapper(cat_features=['cat_feat'], **params)
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        clf.fit(X, y, verbose=False)

    @pytest.mark.parametrize('params,prediction', CATBOOST_TEST_PARAMS)
    def test_predict(self, params, prediction):
        clf = CatboostWrapper(cat_features=['cat_feat'], **params)
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        clf.fit(X, y, verbose=False)
        possible_prediction = np.array(prediction)
        assert_allclose(possible_prediction, clf.predict(X),
                        rtol=0.01, atol=0.1)

    @pytest.mark.parametrize('params,prediction', CATBOOST_TEST_PARAMS)
    def test_fit_predict(self, params, prediction):
        clf = CatboostWrapper(cat_features=['cat_feat'], **params)
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        possible_prediction = np.array(prediction)
        assert_allclose(possible_prediction,
                        clf.fit(X, y, verbose=False).predict(X),
                        rtol=0.01, atol=0.1)


BEST_CONST_PARAMS = [
    (
        [0, 1, 2],
        1
    ),
    (
        [0, 1, 100],
        1
    ),
    (
        [0, 1],
        0.5
    ),
    (
        [0, 10],
        0.5
    ),
]


@pytest.mark.parametrize('y_true,best_const', BEST_CONST_PARAMS)
def test_get_our_loss_best_const(y_true, best_const):
    assert best_const == get_our_loss_best_const(y_true)


LIGHTGBM_PARAMS = [
    (
        {'min_child_samples': 1},
        [1.5, 1.5, 4, 4, 4, 6.5, 6.5, 10, 10, 10]
    ),
    (
        {'objective': our_loss_lgbm_objective, 'min_child_samples': 1},
        [1.5, 1.5, 4, 4, 4, 6.5, 6.5, 9, 9, 9]
    )
]


class TestLightgbmWrapper:
    def test_prepare_pool(self):
        clf = LightgbmWrapper(cat_features=['cat_feat'])
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        new_X = clf._prepare_pool(X, y)
        assert_array_equal(X.values, new_X.values)
        assert new_X['cat_feat'].dtype.name == 'category'

    @pytest.mark.parametrize('params,prediction', LIGHTGBM_PARAMS)
    def test_fit(self, params, prediction):
        clf = LightgbmWrapper(cat_features=['cat_feat'], **params)
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        clf.fit(X, y, verbose=False)

    @pytest.mark.parametrize('params,prediction', LIGHTGBM_PARAMS)
    def test_predict(self, params, prediction):
        clf = LightgbmWrapper(cat_features=['cat_feat'], **params)
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        clf.fit(X, y, verbose=False)
        possible_prediction = np.array(prediction)
        assert_allclose(possible_prediction, clf.predict(X),
                        rtol=0.01, atol=0.1)

    @pytest.mark.parametrize('params,prediction', LIGHTGBM_PARAMS)
    def test_fit_predict(self, params, prediction):
        clf = LightgbmWrapper(cat_features=['cat_feat'], **params)
        X = TEST_REGRESSORS_DF.drop(columns=['target'], inplace=False)
        y = TEST_REGRESSORS_DF['target']
        possible_prediction = np.array(prediction)
        assert_allclose(possible_prediction,
                        clf.fit(X, y, verbose=False).predict(X),
                        rtol=0.01, atol=0.1)
