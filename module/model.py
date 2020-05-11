"""
Tools to deal with model.
"""
import numpy as np


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


def our_loss_lgb_objective(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * residual, -1)
    hess = np.where(residual < 0, 2, 0)
    return grad, hess


def our_loss_function(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, residual ** 2, residual)
    return np.mean(loss)
