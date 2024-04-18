"""
Программа: Предобработка изображений
Версия: 1.0
"""

from typing import Tuple
import pandas as pd
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array


def prepare_image(img: Image, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Подготавливает изображение для подачи в модель.

    Parameters
    ----------
    img_path: Image
        Объект PIL.Image с изображением.
    image_size: Tuple[int, int]
        Размер изображения.

    Returns
    -------
    np.ndarray
        Подготовленное изображение в виде массива.
    """
    img_array = img_to_array(img.resize(image_size))
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    return img_array
