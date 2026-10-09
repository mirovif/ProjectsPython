# ProjectsPython

Проекты по Data Science и Machine Learning: обработка данных, классические ML pipelines,
метрики на NumPy и encoder–decoder Transformer на PyTorch.

## Проекты

| Проект | Задача и стек | Результат |
|---|---|---|
| [Apartment Price Prediction](apartment-price-prediction/) | Цена жилья на Ames Housing · Pandas, Scikit-learn | RMSE $25,054 · R² 0.922 |
| [Titanic Survival Classification](titanic-survival-classification/) | Классификация · Feature engineering, Scikit-learn | ROC-AUC 0.808 · F1 0.656 |
| [Transformer From Scratch](transformer-from-scratch/) | Ручной attention и encoder–decoder · PyTorch | Copy task: token accuracy 80.9% · sequence exact match 32.8% |
| [ML Metrics From Scratch](ml-metrics-from-scratch/) | Восемь метрик · Python, NumPy | Сверка со sklearn · 81 тест |

В каждом проекте есть README с данными, подходом, результатами и командами запуска.
Для каждого проекта создаётся отдельное виртуальное окружение; команды выполняются
из его папки. Зависимости зафиксированы в `requirements-lock.txt`.

## Transformer

`Input → Embedding + Positional Encoding → Encoder → Decoder → Linear → Vocabulary`

Модель состоит из `Transformer.py`, `MultiHeadAttention.py`, `InputEmbedding.py`,
`Encoder.py`, `Encoder_Block.py`, `Decoder.py`, `Decoder_Block.py`, `FeedForward.py`
и `AddNorm.py`. Обучение и autoregressive generation запускаются отдельно.

## Проверка

108 тестов: regression — 2, Titanic — 3, Transformer — 22, metrics — 81.
GitHub Actions проверяет каждый проект отдельно на Python 3.12.
Результаты обучения получены реальными запусками; источники данных и ограничения
приведены в README проектов.
