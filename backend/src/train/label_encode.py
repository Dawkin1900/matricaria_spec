"""
Программа: Получение словаря с соответствием меток классов
Версия: 1.0
"""

from typing import Any
import pandas as pd


def get_label_map(gen: Any) -> dict:
    """
    Получение словаря с метками классов.

    Parameters
    ----------
    gen: Any
        Генератор данных.

    Returns
    -------
    dict
        Словарь с метками классов.
    """
    return gen.class_indices
