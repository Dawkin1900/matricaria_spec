"""
Программа: Получение предсказания модели на test_generator
Версия: 1.0
"""

from typing import Any
import pandas as pd
import numpy as np


def model_pred(model: Any, test_gen: Any) -> None:
    """
    Предсказание значений.

    Parameters
    ----------
    model: Any
        Обученная модель.
    test_gen: Any
        Генератор тестовых данных.

    Returns
    -------
    None
        Вывод графика confusion matrix и classification report.
    """
    # сделаем сброс генератора, чтобы он точно начал с начала
    test_gen.reset()

    # предсказанные метки классов
    y_pred = model.predict(test_gen, verbose=1)
    y_pred_labels = np.argmax(y_pred, axis=1).tolist()

    # реальные метки классов
    y_true = test_gen.classes

    return y_true, y_pred_labels
