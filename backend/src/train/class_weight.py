"""
Программа: Учет дисбаланса классов
Версия: 1.0
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.utils import class_weight


def get_class_weights(train_gen: Any) -> dict:
    """
    Расчет весов при дисбалансе классов.

    Parameters
    ----------
    img_path: str
        Путь к изображению.
    image_size: Tuple[int, int]
        Размер изображения.

    Returns
    -------
    np.ndarray
        Подготовленное изображение в виде массива.
    """
    cls_wt = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes,
    )
    return dict(zip(np.unique(train_gen.classes), cls_wt))
