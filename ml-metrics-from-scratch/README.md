# ML Metrics From Scratch

**Восемь метрик на Python/NumPy с проверкой по sklearn.**
Цель — понимать вычисления, а не только вызывать готовую функцию.
`sklearn.metrics` используется в примерах и тестах **только как эталон**;
код вычислений в [metrics.py](src/ml_metrics/metrics.py) импортирует только NumPy.

Проект развивает существующий пример Accuracy из
[ProjectsPython](https://github.com/mirovif/ProjectsPython/blob/master/Metrics_python/Accuracy.py):
сохранён понятный подсчёт совпадений циклом, добавлены единая проверка входов и другие метрики.

## Регрессия

Пусть `y_i` — фактическое значение, `ŷ_i` — предсказание, `n` — число наблюдений.

| Метрика | Формула | Что показывает |
|---|---|---|
| MAE | `sum(abs(y − ŷ)) / n` | средняя абсолютная ошибка в единицах target |
| MSE | `sum((y − ŷ)²) / n` | сильнее штрафует большие ошибки, единицы target² |
| RMSE | `sqrt(MSE)` | ошибка с большим штрафом за выбросы в единицах target |
| R² | `1 − sum((y − ŷ)²) / sum((y − mean(y))²)` | качество относительно среднего target; может быть отрицательным |

```python
from ml_metrics import mae, mse, rmse, r2
y_true = [3., -.5, 2., 7.]
y_pred = [2.5, 0., 2., 8.]
print(mae(y_true, y_pred))
print(mse(y_true, y_pred))
print(rmse(y_true, y_pred))
print(r2(y_true, y_pred))
```

Например, абсолютные ошибки — `[0.5, 0.5, 0, 1]`; сумма 2, значит MAE = 2/4 = 0.5.
R² с постоянным target совпадает с `sklearn(force_finite=True)`:
1 для точного предсказания, иначе 0. Для одного наблюдения API явно отклоняет R²,
в отличие от sklearn, возвращающего NaN с предупреждением.

## Бинарная классификация

Положительный класс — **1**, отрицательный — **0**. На вход подаются labels, не вероятности.
TP — true positives, FP — false positives, TN — true negatives, FN — false negatives.

| Метрика | Формула | Что показывает |
|---|---|---|
| Accuracy | `(TP + TN) / n` | доля правильных решений; может скрывать дисбаланс |
| Precision | `TP / (TP + FP)` | доля истинных positives среди предсказанных positives |
| Recall | `TP / (TP + FN)` | доля обнаруженных фактических positives |
| F1 | `2TP / (2TP + FP + FN)` | баланс precision и recall |

```python
from ml_metrics import accuracy, precision, recall, f1, confusion_matrix
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 0, 1, 0, 1]
print(accuracy(y_true, y_pred))
print(precision(y_true, y_pred))
print(recall(y_true, y_pred))
print(f1(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

Confusion matrix: строки — истинные классы, столбцы — предсказанные,
порядок `[0, 1]`, то есть `[[TN, FP], [FN, TP]]`.
В примере TP=3, FP=0, FN=1, TN=2: precision=1, recall=3/4, F1=6/7.
Если знаменатель равен нулю, по умолчанию возвращается 0;
можно явно выбрать `zero_division=1`, как в sklearn.

## Фактическая проверка

| Метрика | Собственная реализация | sklearn | Абсолютная разница |
| --- | --- | --- | --- |
| MAE | 0.500000000 | 0.500000000 | 0.0e+00 |
| MSE | 0.375000000 | 0.375000000 | 0.0e+00 |
| RMSE | 0.612372436 | 0.612372436 | 0.0e+00 |
| R2 | 0.948608137 | 0.948608137 | 0.0e+00 |
| Accuracy | 0.833333333 | 0.833333333 | 0.0e+00 |
| Precision | 1.000000000 | 1.000000000 | 0.0e+00 |
| Recall | 0.750000000 | 0.750000000 | 0.0e+00 |
| F1 | 0.857142857 | 0.857142857 | 0.0e+00 |

`examples.py` проверяет числа через `np.testing.assert_allclose(rtol=1e-12, atol=1e-12)`
и выводит результаты в терминал.
81 тест покрывает случайные данные, исходный пример Accuracy, постоянный target,
отрицательный R², нулевые знаменатели, NaN/inf, пустые/несовпадающие массивы
и отклонение probabilities/multiclass. Weighted и multiclass метрики намеренно не реализованы.

## Запуск

Python 3.12; команды из папки `ml-metrics-from-scratch`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python examples.py
```

В Windows вместо `source .venv/bin/activate` используйте `.venv\Scripts\Activate.ps1`.

В примерах выше предполагается, что `src` находится в PYTHONPATH; `examples.py` и pytest
настраивают это сами. Для отдельного скрипта добавить `src` или установить свой пакет.
Для самих вычислений требуется только NumPy; sklearn нужен для сверки результатов.

```text
├── src/ml_metrics/
├── tests/
└── examples.py
```
