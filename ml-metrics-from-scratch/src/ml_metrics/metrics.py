import numpy as np

def _pair(y_true, y_pred, binary=False):
    true = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if true.ndim != 1 or predicted.ndim != 1 or true.size == 0 or true.shape != predicted.shape:
        raise ValueError('Expected nonempty one-dimensional arrays with the same shape.')
    if not np.isfinite(true).all() or not np.isfinite(predicted).all():
        raise ValueError('NaN and infinity are not valid metric inputs.')
    if binary and (not np.isin(true, [0, 1]).all() or not np.isin(predicted, [0, 1]).all()):
        raise ValueError('Binary labels must be 0 or 1; pass labels, not probabilities.')
    return true, predicted

def mae(y_true, y_pred):
    true, predicted = _pair(y_true, y_pred)
    return float(np.mean(np.abs(true - predicted)))

def mse(y_true, y_pred):
    true, predicted = _pair(y_true, y_pred)
    return float(np.mean((true - predicted) ** 2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2(y_true, y_pred):
    true, predicted = _pair(y_true, y_pred)
    if len(true) < 2:
        raise ValueError('R-squared requires at least two observations.')
    residual = np.sum((true - predicted) ** 2)
    total = np.sum((true - np.mean(true)) ** 2)

    if total == 0:
        return 1.0 if residual == 0 else 0.0
    return float(1 - residual / total)

def accuracy(y_true, y_pred):
    true, predicted = _pair(y_true, y_pred, binary=True)

    correct = sum(actual == prediction for actual, prediction in zip(true, predicted))
    return float(correct / len(true))

def confusion_matrix(y_true, y_pred):
    true, predicted = _pair(y_true, y_pred, binary=True)
    tn = np.sum((true == 0) & (predicted == 0))
    fp = np.sum((true == 0) & (predicted == 1))
    fn = np.sum((true == 1) & (predicted == 0))
    tp = np.sum((true == 1) & (predicted == 1))
    return np.array([[tn, fp], [fn, tp]], dtype=int)

def _divide(numerator, denominator, zero_division):
    if zero_division not in (0, 1):
        raise ValueError('zero_division must be 0 or 1.')
    return float(numerator / denominator) if denominator else float(zero_division)

def precision(y_true, y_pred, zero_division=0):
    (_, fp), (_, tp) = confusion_matrix(y_true, y_pred)
    return _divide(tp, tp + fp, zero_division)

def recall(y_true, y_pred, zero_division=0):
    (_, _), (fn, tp) = confusion_matrix(y_true, y_pred)
    return _divide(tp, tp + fn, zero_division)

def f1(y_true, y_pred, zero_division=0):
    (_, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    return _divide(2 * tp, 2 * tp + fp + fn, zero_division)
