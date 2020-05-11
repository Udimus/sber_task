"""
Tools to deal with model.
"""
import numpy as np
from catboost import CatBoostRegressor, Pool
from lightgbm import Dataset, LGBMRegressor
from sklearn.base import RegressorMixin

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


def our_loss_lgb_objective(y_true, y_pred, weights=None):
    assert len(y_true) == len(y_pred)
    if weights is not None:
        assert len(weights) == len(y_pred)
    else:
        weights = np.ones_like(y_pred)
    residual = (y_true - y_pred).astype("float") * weights
    grad = np.where(residual < 0, -2 * residual, -1)
    hess = np.where(residual < 0, 2, 0)
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

    def predict(self, X):
        return self.model.predict(self._prepare_pool(X))

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


class LightgbmWrapper(BaseRegressorWrapper):
    def __init__(self, cat_features=None, **params):
        super().__init__()
        self.params = params
        if cat_features is None:
            cat_features = []
        self.cat_features = cat_features
        self.model = LGBMRegressor(**params)

    def _prepare_pool(self, X, y=None):
        X = X.copy()
        for col in self.cat_features:
            X[col] = X[col].astype('category')
        return X

    def fit(self, X, y, **params):
        self.model.fit(self._prepare_pool(X),
                       y,
                       categorical_feature=self.cat_features,
                       **params)
        return self
