from pathlib import Path
import sys
import numpy as np
from sklearn import metrics as reference
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import ml_metrics as own

def main():
    regression_true = [3., -.5, 2., 7.]
    regression_pred = [2.5, 0., 2., 8.]
    binary_true = [1, 0, 1, 1, 0, 1]
    binary_pred = [1, 0, 0, 1, 0, 1]
    for name, actual, expected in [
        ('MAE', own.mae(regression_true, regression_pred), reference.mean_absolute_error(regression_true, regression_pred)),
        ('MSE', own.mse(regression_true, regression_pred), reference.mean_squared_error(regression_true, regression_pred)),
        ('RMSE', own.rmse(regression_true, regression_pred), np.sqrt(reference.mean_squared_error(regression_true, regression_pred))),
        ('R2', own.r2(regression_true, regression_pred), reference.r2_score(regression_true, regression_pred)),
        ('Accuracy', own.accuracy(binary_true, binary_pred), reference.accuracy_score(binary_true, binary_pred)),
        ('Precision', own.precision(binary_true, binary_pred), reference.precision_score(binary_true, binary_pred)),
        ('Recall', own.recall(binary_true, binary_pred), reference.recall_score(binary_true, binary_pred)),
        ('F1', own.f1(binary_true, binary_pred), reference.f1_score(binary_true, binary_pred)),
    ]:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        print(f'{name:10} own={actual:.9f} sklearn={expected:.9f}')
    np.testing.assert_array_equal(own.confusion_matrix(binary_true, binary_pred),
                                  reference.confusion_matrix(binary_true, binary_pred, labels=[0, 1]))

if __name__ == '__main__':
    main()
