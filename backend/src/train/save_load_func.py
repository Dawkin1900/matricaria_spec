"""
Программа: Сохранение и загрузка словаря с данными.
Версия: 1.0
"""

import json
import yaml
import pandas as pd


def save_func(result: dict, result_path: str) -> None:
    """
    Получение и сохранение словаря с метками классов.

    Parameters
    ----------
    result: dict
        Словарь с данными.
    label_map_path:
        Путь для сохранения словаря с данными.

    Returns
    -------
    None
        Сохраняет словарь с данными.
    """
    with open(result_path, "w") as file:
        json.dump(result, file)


def load_func(config_path: str, name: str) -> dict:
    """
    Получение словаря с метками классов из файла.

    Parameters
    ----------
    config_path:
        Путь до конфигурационного файла.
    name: str
        Название необходимого файла.

    Returns
    -------
    dict
        Словарь с данными.
    """
    # открытие конфигурационного файла
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # открытие словаря с метками классов
    with open(config["train"][name]) as json_file:
        result = json.load(json_file)

    return result
