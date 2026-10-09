# ProjectsPython

Мои проекты на Python: оценка стоимости жилья, классификация пассажиров Titanic,
Transformer на PyTorch и расчёт метрик на NumPy.

| Проект | Задача | Результат |
|---|---|---|
| [Apartment Price Prediction](apartment-price-prediction/) | Цена дома по его характеристикам | RMSE $25,054 · R² 0.922 |
| [Titanic Survival Classification](titanic-survival-classification/) | Вероятность выживания пассажира | ROC-AUC 0.808 · F1 0.656 |
| [Transformer From Scratch](transformer-from-scratch/) | Копирование последовательности токенов | Полное совпадение при генерации: 32.8% |
| [ML Metrics From Scratch](ml-metrics-from-scratch/) | MAE, MSE, RMSE, R², accuracy, precision, recall, F1 | Сравнение со scikit-learn, 81 тест |

Описание данных, графики и команды запуска находятся в папке каждого проекта.
Зависимости устанавливаются из его `requirements.txt`; для каждого проекта нужно отдельное окружение Python 3.12.
Тесты запускаются в [GitHub Actions](https://github.com/mirovif/ProjectsPython/actions).

## Другие проекты

- [AI Agent](https://github.com/mirovif/ai-agent)
- [ML Competition](https://github.com/mirovif/ml-competition)
- [Metrics Python](https://github.com/mirovif/metrics-python)
