"""
Программа: Создание генераторов
Версия: 1.0
"""

import pandas as pd
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import keras


def datagen_make(train_flg: bool) -> ImageDataGenerator:
    """
    Создает объект ImageDataGenerator.

    Parameters
    ----------
    train_flg: bool
        Указатель на создание генератора для тренировочных данных.

    Returns
    -------
    ImageDataGenerator
        Объект ImageDataGenerator.
    """
    # для тренировочных данных
    if train_flg:
        return ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.9, 1.1],
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
        )
    # для валидационных и тестовых данных
    return ImageDataGenerator(rescale=1.0 / 255)


def generator_make(
    data: pd.DataFrame, train_flg: bool, **kwargs
) -> keras.src.legacy.preprocessing.image.DataFrameIterator:
    """
    Создает генератор изображений на основе DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame с данными.
    train_flg: bool
        Указатель на создание генератора для тренировочных данных.
    **kwargs
        Дополнительные аргументы.

    Returns
    -------
    keras.src.legacy.preprocessing.image.DataFrameIterator
        Генератор изображений.
    """
    image_size = (kwargs["img_size"], kwargs["img_size"])
    datagen = datagen_make(train_flg=train_flg)

    # генератор для тренировочных данных
    if train_flg:
        return datagen.flow_from_dataframe(
            data,
            x_col="img",
            y_col=kwargs["target_column"],
            target_size=image_size,
            batch_size=kwargs["batch_size"],
            class_mode="categorical",
            shuffle=True,
            color_mode="rgb",
            seed=kwargs["random_state"],
        )
    # генератор для валидационных и тестовых данных
    return datagen.flow_from_dataframe(
        data,
        x_col="img",
        y_col=kwargs["target_column"],
        target_size=image_size,
        batch_size=kwargs["batch_size"],
        class_mode="categorical",
        shuffle=False,
        color_mode="rgb",
        seed=kwargs["random_state"],
    )
