def calculate_accuracy(y_true, y_pred):
    # Проверяем, что длины списков совпадают
    if len(y_true) != len(y_pred):
        return "Ошибка: Списки должны быть одинаковой длины"

    # Считаем количество совпадений
    correct_predictions = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct_predictions += 1

    # Вычисляем долю правильных ответов
    accuracy = correct_predictions / len(y_true)
    return accuracy


# Пример использования:
true_labels = [1, 0, 1, 1, 0, 1]
pred_labels = [1, 0, 0, 1, 0, 1]

result = calculate_accuracy(true_labels, pred_labels)
print(f"Accuracy: {result:.2%}")  # Выведет 83.33%