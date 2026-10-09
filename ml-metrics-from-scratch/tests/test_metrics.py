import numpy as np
import pytest
from sklearn import metrics as reference
import ml_metrics as own

@pytest.mark.parametrize('seed', range(10))
def test_regression_matches_sklearn_for_random_inputs(seed):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=60)
    predicted = true + rng.normal(scale=.7, size=60)
    for actual, expected in [
        (own.mae(true, predicted), reference.mean_absolute_error(true, predicted)),
        (own.mse(true, predicted), reference.mean_squared_error(true, predicted)),
        (own.rmse(true, predicted), np.sqrt(reference.mean_squared_error(true, predicted))),
        (own.r2(true, predicted), reference.r2_score(true, predicted)),
    ]:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize('seed', range(10))
def test_binary_metrics_match_sklearn_for_random_inputs(seed):
    rng = np.random.default_rng(seed)
    true, predicted = rng.integers(0, 2, 60), rng.integers(0, 2, 60)
    for function, expected in [(own.accuracy, reference.accuracy_score),
                               (own.precision, reference.precision_score),
                               (own.recall, reference.recall_score), (own.f1, reference.f1_score)]:
        np.testing.assert_allclose(function(true, predicted), expected(true, predicted), rtol=1e-12)
    np.testing.assert_array_equal(own.confusion_matrix(true, predicted),
                                  reference.confusion_matrix(true, predicted, labels=[0, 1]))

@pytest.mark.parametrize('true,predicted', [([], []), ([1], [1, 2]), ([[1]], [[1]]),
                                           ([np.nan], [1]), ([1], [np.inf])])
@pytest.mark.parametrize('function', [own.mae, own.mse, own.rmse, own.r2, own.accuracy,
                                     own.precision, own.recall, own.f1, own.confusion_matrix])
def test_invalid_input_is_rejected(true, predicted, function):
    with pytest.raises(ValueError):
        function(true, predicted)

@pytest.mark.parametrize('function', [own.accuracy, own.precision, own.recall, own.f1, own.confusion_matrix])
def test_probabilities_and_multiclass_labels_are_rejected(function):
    with pytest.raises(ValueError):
        function([0, 1], [.2, .8])
    with pytest.raises(ValueError):
        function([0, 2], [0, 1])

@pytest.mark.parametrize('zero_division', [0, 1])
@pytest.mark.parametrize('function,reference_function', [
    (own.precision, reference.precision_score), (own.recall, reference.recall_score),
    (own.f1, reference.f1_score)])
def test_zero_denominators_match_explicit_sklearn_policy(zero_division, function, reference_function):
    assert function([0, 0], [0, 0], zero_division=zero_division) == reference_function(
        [0, 0], [0, 0], zero_division=zero_division)

@pytest.mark.parametrize('function', [own.precision, own.recall, own.f1])
def test_invalid_zero_division_is_rejected(function):
    with pytest.raises(ValueError):
        function([0, 1], [0, 1], zero_division=3)

def test_r2_constant_target_and_negative_score():
    assert own.r2([2, 2], [2, 2]) == reference.r2_score([2, 2], [2, 2]) == 1
    assert own.r2([2, 2], [1, 1]) == reference.r2_score([2, 2], [1, 1]) == 0
    assert own.r2([0, 1], [10, 10]) < 0
    with pytest.raises(ValueError):
        own.r2([1], [1])

def test_original_accuracy_example():
    assert own.accuracy([1, 0, 1, 1, 0, 1], [1, 0, 0, 1, 0, 1]) == 5 / 6
