# Transformer From Scratch

Encoder–decoder Transformer на PyTorch. Attention рассчитывается через проекции Q, K, V
и матричные операции; готовый `nn.Transformer` не используется.
Модель обучается копировать последовательность токенов и генерирует ответ по одному токену.

## Архитектура

![Архитектура Transformer](images/architecture.svg)

`Input → Embedding + Positional Encoding → Encoder → Decoder → Linear → Vocabulary`

Encoder обрабатывает исходную последовательность. Decoder получает предыдущие токены
ответа и выход encoder через cross-attention. Линейный слой выдаёт logits по словарю.

| Файл | Компонент |
|---|---|
| [Transformer.py](Transformer.py) | Сборка модели и выходной линейный слой |
| [InputEmbedding.py](InputEmbedding.py) | Embedding, синусоидальное positional encoding и dropout |
| [MultiHeadAttention.py](MultiHeadAttention.py) | Проекции Q/K/V, разделение на головы и scaled dot-product attention |
| [Encoder.py](Encoder.py), [Encoder_Block.py](Encoder_Block.py) | Стек encoder и его блок |
| [Decoder.py](Decoder.py), [Decoder_Block.py](Decoder_Block.py) | Стек decoder и его блок |
| [FeedForward.py](FeedForward.py) | Два линейных слоя с ReLU |
| [AddNorm.py](AddNorm.py) | Residual connection и LayerNorm после сложения |

Encoder block: `Self-attention → AddNorm → FFN → AddNorm`.
Decoder block: `Causal self-attention → AddNorm → Cross-attention → AddNorm → FFN → AddNorm`.
Dropout применяется в embedding.

$$Attention(Q,K,V) = softmax(QK^T / sqrt{d_k} + M)V$$

### Формы тензоров

`B` — размер батча, `T` — длина последовательности, `D` — размер embedding,
`V` — размер словаря, `h` — число голов, `d_k = D / h`, `N` — число блоков.

| Тензор | Форма |
|---|---|
| Токены source / target | `(B, T_src)` / `(B, T_tgt)` |
| Embedding | `(B, T, D)` |
| Q/K/V по головам | `(B, h, T, d_k)` |
| Self-attention scores | `(B, h, T, T)` |
| Cross-attention scores | `(B, h, T_tgt, T_src)` |
| Feed Forward | `(B, T, D)` → `(B, T, D_ff)` → `(B, T, D)` |
| Выход модели | `(B, T_tgt, V)` |

### Маски

В boolean mask `True` разрешает attention, `False` закрывает его.
`Transformer.forward` принимает `src_mask` и `tgt_mask`; при обучении и генерации
их создаёт [training_helpers.py](training_helpers.py).
Source mask имеет форму `(B, 1, 1, T_src)`, target mask — `(B, 1, T_tgt, T_tgt)`.
Первая закрывает padding, вторая — padding и будущие токены.
Causal mask действует при переданном `tgt_mask`. PAD исключён из расчёта loss.

## Обучение

В задаче копирования ответ должен повторить входную последовательность и завершиться EOS.
Длина входа — 3–7 токенов, значения — от 3 до 23. Служебные токены: PAD=0, BOS=1, EOS=2.
Обучающая выборка содержит 1,024 последовательности, validation и test — по 128.
Последовательности уникальны и не пересекаются между частями; seeds — 42, 43 и 45.
Разбиения записаны в [copy_splits.json](data/copy_splits.json).

Конфигурация: `D=32`, `h=4`, `d_ff=64`, `N=2`, `max_len=16`, `dropout=0.1`.
Размер словаря — 24, число параметров — 45,080.
Обучение на CPU: Adam, `lr=0.001`, batch size 64, gradient clipping 1.0, 2,400 шагов.
На вход decoder подаются BOS и предыдущие правильные токены ответа — teacher forcing.
Loss — cross-entropy. По минимальному validation loss выбран checkpoint шага 2,350.

![Кривая обучения](images/training_curve.png)

| Метрика | Значение |
|---|---:|
| Validation loss до обучения | 3.3370 |
| Validation loss выбранной модели | 0.6312 |
| Test loss | 0.6317 |
| Test token accuracy с teacher forcing | 80.876% |
| Test полное совпадение при greedy generation | 32.812% |

Token accuracy измеряет следующий токен при правильном предыдущем контексте.
Полное совпадение проверяет самостоятельную генерацию всей последовательности, включая EOS.
Модель правильно копирует около трети тестовых примеров; высокий результат по отдельным
токенам пока не даёт такого же качества при генерации целого ответа.

| Вход | Сгенерированный ответ |
|---|---|
| `[16, 18, 11, 5, 12]` | `[16, 18, 11, 5, 12]` |
| `[3, 5, 18, 3, 6]` | `[3, 5, 18, 6]` |

В обоих примерах сгенерирован EOS.
[Конфигурация и метрики](reports/summary.json) ·
[История обучения](reports/training_history.csv) ·
[Все тестовые ответы](reports/test_generations.json)

Пока обучение проверено на синтетических последовательностях.
Следующий шаг — токенизация текстового корпуса и обучение на задаче перевода.
Архитектура основана на [Attention Is All You Need](https://arxiv.org/abs/1706.03762);
параметры обучения и dropout отличаются от описанных в статье.

## Тесты

22 теста проверяют расчёт attention, causal и padding masks, градиенты,
сохранение весов, формы тензоров, неверные входы и переобучение на маленькой выборке.
## Запуск

Python 3.12. Команды для PowerShell из папки `transformer-from-scratch`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python train.py
python generate.py 3 7 12 4
```

В Linux и macOS окружение активируется командой `source .venv/bin/activate`.

Зависимости включают CPU-версию PyTorch.
`train.py` сохраняет веса в `checkpoints/copy_model.pt`, отчёты — в `reports/`,
разбиения — в `data/`, кривую обучения — в `images/`.
Для работы `generate.py` сначала запустите обучение: веса не хранятся в Git.
