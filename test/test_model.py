"""
Test for module/model.py
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from module.model import (
    our_loss_function,
    our_loss_lgb_objective,
    OurLossCBObjective,
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
