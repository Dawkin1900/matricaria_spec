"""
Программа: Получение метрик
Версия: 1.0
"""

from typing import Any
import pandas as pd


def get_metrics(model: Any, test_gen: Any) -> dict:
    """
    Получение словаря с метриками и запись в словарь.

    Parameters
    ----------
    model: Any
        Обученная модель.
    test_gen: Any
        Генератор тестовых данных.

    Returns
    -------
    dict
        Словарь с метриками.
    """
    score = model.evaluate(test_gen, verbose=0)
    dict_metrics = {"loss": score[0], "accuracy": score[1], "f1": score[2]}

    return dict_metrics
