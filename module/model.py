"""
Tools to deal with model.
"""
import numpy as np
from catboost import CatBoostRegressor, Pool
from lightgbm import Dataset, LGBMRegressor
from sklearn.base import RegressorMixin
from scipy.optimize import minimize

OUR_LOSS_HESS_CONST = 0.1

class OurLossCBObjective:
    def calc_ders_range(self, approxes, targets, weights=None):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            if approxes[index] < targets[index]:
                der1 = 1
                der2 = 0
            else:
                der1 = 2 * (targets[index] - approxes[index])
                der2 = -2

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result


class OurLossCBMetric(object):
    def get_final_error(self, error, weight=None):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight=None):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            diff = approx[i] - target[i]
            if diff < 0:
                error_sum += - w * diff
            else:
                error_sum += w * diff ** 2

        return error_sum, weight_sum


def our_loss_lgbm_objective(y_true, y_pred, weights=None):
    assert len(y_true) == len(y_pred)
    if weights is not None:
        assert len(weights) == len(y_pred)
    else:
        weights = np.ones_like(y_pred)
    residual = (y_true - y_pred).astype("float") * weights
    grad = np.where(residual < 0, -2 * residual, -1)
    hess = np.where(residual < 0, 2, OUR_LOSS_HESS_CONST)
    return grad, hess


def our_loss_function(y_true, y_pred, weights=None):
    assert len(y_true) == len(y_pred)
    if weights is not None:
        assert len(weights) == len(y_pred)
    else:
        weights = np.ones_like(y_pred)
    residual = (y_true - y_pred).astype("float") * weights
    loss = np.where(residual < 0, residual ** 2, residual)
    return np.mean(loss)


def our_loss_lgbm_metric(y_true, y_pred, weights=None):
    loss = our_loss_function(y_true, y_pred, weights)
    return "our_custom_metric", np.mean(loss), False


def get_our_loss_best_const(y_true):
    ones = np.ones_like(y_true)

    def const_loss(koef):
        return our_loss_function(y_true, ones * koef[0])

    def const_jac(koef):
        grad, _ = our_loss_lgbm_objective(y_true, ones * koef[0])
        return np.array([np.mean(grad)])

    def const_hess(koef):
        _, hess = our_loss_lgbm_objective(y_true, ones * koef[0])
        return np.array([np.mean(hess)])

    res = minimize(
        fun=const_loss,
        method='BFGS',
        x0=np.array([0]),
        jac=const_jac,
        hess=const_hess,
    )
    return res.x[0]


class BaseRegressorWrapper(RegressorMixin):
    def __init__(self):
        self.model = None

    def _prepare_pool(self, X, y=None):
        return X

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return our_loss_function(y, y_pred, sample_weight)

    def fit(self, X, y, **params):
        self.model.fit(self._prepare_pool(X, y), **params)
        return self

    def predict(self, X, **params):
        return self.model.predict(self._prepare_pool(X), **params)

    def fit_predict(self, X, y, **params):
        return self.model.fit(X, y, **params).predict(X)


class CatboostWrapper(BaseRegressorWrapper):
    def __init__(self, cat_features=None, **params):
        super().__init__()
        self.params = params
        if cat_features is None:
            cat_features = []
        self.cat_features = cat_features
        self.model = CatBoostRegressor(cat_features=cat_features, **params)

    def _prepare_pool(self, X, y=None):
        X = X.copy()
        for col in self.cat_features:
            X[col] = X[col].astype('category')
        return Pool(
            X,
            label=y,
            cat_features=self.cat_features,
            feature_names=X.columns.tolist()
        )

    def predict(self, X, **params):
        return self.model.predict(self._prepare_pool(X),
                                  prediction_type='RawFormulaVal',
                                  **params)


class LightgbmWrapper(BaseRegressorWrapper):
    def __init__(self, cat_features=None, **params):
        super().__init__()
        self.params = params
        if cat_features is None:
            cat_features = []
        self.cat_features = cat_features
        self._our_loss = our_loss_lgbm_objective == params.get('objective')
        if self._our_loss:
            self.const = 0
        self.model = LGBMRegressor(**params)

    def _prepare_pool(self, X, y=None):
        X = X.copy()
        for col in self.cat_features:
            X[col] = X[col].astype('category')
        return X

    def fit(self, X, y, **params):
        if self._our_loss:
            self.const = get_our_loss_best_const(y)
            self.model.fit(self._prepare_pool(X),
                           y,
                           init_score=np.ones_like(y) * self.const,
                           categorical_feature=self.cat_features,
                           **params)
        else:
            self.model.fit(self._prepare_pool(X),
                           y,
                           categorical_feature=self.cat_features,
                           **params)
        return self

    def predict(self, X, **params):
        prediction = self.model.predict(self._prepare_pool(X), **params)
        if self._our_loss:
            prediction += np.ones(len(X)) * self.const
        return prediction
