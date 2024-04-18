"""
Программа: Тренировка модели
Версия: 1.0
"""

import warnings
from typing import Tuple, Any, Type
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from keras.applications import MobileNetV2

import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from keras.optimizers import Adam

from keras.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

from ..train.metrics import get_metrics
from ..train.save_load_func import save_func

warnings.filterwarnings("ignore")


def create_model(
    img_shape: Tuple[int, int, int], n_class: int, optimizer: Type[Adam], rand_seed: int
) -> Sequential:
    """
    Создает модель нейронной сети на основе заданной базовой модели.

    Parameters
    ----------
    img_shape (Tuple[int, int, int]):
        Размер входного изображения.
    n_class: int
        Количество классов.
    optimizer: Type[Adam]
        Оптимизатор для модели.
    rand_seed: int
        random_state для воспроизводимости результата.

    Returns
    -------
    Sequential
        Созданная модель нейронной сети.
    """
    # создадим базовую модель
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=img_shape
    )
    # заморозим слои
    no_base_layers = len(base_model.layers)
    no_finetune_layers = int(no_base_layers / 4)
    base_model.trainable = True
    for layer in base_model.layers[:-no_finetune_layers]:
        layer.trainable = False

    # создадим модель CNN на основе базовой модели
    model = Sequential()
    model.add(Input(shape=img_shape))
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(rate=0.2, seed=rand_seed))
    model.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(rate=0.1, seed=rand_seed))
    model.add(Dense(n_class, activation="softmax"))

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.F1Score(average="macro")],
    )

    return model


def callbacks_make(
    checkpoint_path: str,
    csv_logger_path: str,
    checkpoint_flg: bool = False,
    reduce_lr_flg: bool = False,
    csv_logger_flg: bool = False,
) -> list:
    """
    Создает список callbacks для обучения модели.

    Parameters
    ----------
    checkpoint_path: str
        Путь для сохранения checkpoint.
    csv_logger_path: str
        Путь для сохранения csv_logger.
    checkpoint_flg: bool
        Флаг использования ModelCheckpoint.
    reduce_lr_flg: bool
        Флаг использования ReduceLROnPlateau.
    csv_logger_flg: bool
        Флаг использования CSVLogger.

    Returns
    -------
    List
        Список callbacks.
    """
    early_stopping = EarlyStopping(
        monitor="val_f1_score",
        patience=5,
        mode="max",
        verbose=0,
        restore_best_weights=False,
    )
    callbacks = [early_stopping]

    if checkpoint_flg:
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor="val_f1_score",
            mode="max",
            verbose=1,
            save_best_only=True,
        )
        callbacks.append(checkpoint)

    if reduce_lr_flg:
        reduce_lr = ReduceLROnPlateau(
            monitor="val_f1_score", min_lr=0.0000001, mode="max", patience=2, verbose=1
        )
        callbacks.append(reduce_lr)

    if csv_logger_flg:
        csv_logger = CSVLogger(csv_logger_path, separator=",", append=True)
        callbacks.append(csv_logger)

    return callbacks


def train_model(
    train_gen: Any, val_gen: Any, test_gen: Any, class_weights: dict, **kwargs
) -> Sequential:
    """
    Обучение модели на лучших параметрах

    Parameters
    ----------
    train_gen: Any
        Генератор тренировочных данных.
    val_gen: Any
        Генератор валидационных данных.
    test_gen: Any
        Генератор тестовых данных.
    class_weights: dict,
        Словарь с весами.
    **kwargs
        Параметры файла config.


    Returns
    -------
    Sequential
        Модель CNN.
    """
    # params
    img_shape = (kwargs["img_size"], kwargs["img_size"], kwargs["channels"])

    # создадим модель для классификации
    if os.path.exists(kwargs["model_path"]):
        n_epochs = kwargs["n_epoch_fine"]
        model = create_model(
            img_shape=img_shape,
            n_class=kwargs["n_class"],
            optimizer=Adam(1e-5),
            rand_seed=kwargs["random_state"],
        )
        model.load_weights(kwargs["model_path"])
    else:
        n_epochs = kwargs["n_epoch_train"]
        model = create_model(
            img_shape=img_shape,
            n_class=kwargs["n_class"],
            optimizer="adam",
            rand_seed=kwargs["random_state"],
        )

    # training
    callbacks = callbacks_make(
        checkpoint_path=kwargs["model_path"],
        csv_logger_path=kwargs["history_path"],
        checkpoint_flg=True,
        reduce_lr_flg=True,
        csv_logger_flg=True,
    )
    history = model.fit(
        x=train_gen,
        validation_data=val_gen,
        epochs=n_epochs,
        verbose=1,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    # save metrics
    metrics_dict = get_metrics(model=model, test_gen=test_gen)
    save_func(result=metrics_dict, result_path=kwargs["metrics_path"])

    return model
