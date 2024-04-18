"""
Программа: Получение предсказания на основе обученной модели
Версия: 1.0
"""

import yaml
import pandas as pd
from PIL import Image
from keras.optimizers import Adam
from ..transform.transform import prepare_image
from ..train.train import create_model


def pipeline_evaluate(config_path: str, image: Image) -> list:
    """
    Предобработка входных данных и получение предсказаний

    Parameters
    ----------
    config_path: str
        Путь до конфигурационного файла.
    img: PIL.Image.Image
        Объект PIL.Image с изображением.

    Returns
    -------
    list
        Предсказания.
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    train_config = config["train"]

    # params
    image_size = (train_config["img_size"], train_config["img_size"])
    img_shape = (image_size[0], image_size[1], train_config["channels"])

    # preprocessing
    preprocessed_image = prepare_image(img=image, image_size=image_size)

    # prediction
    model = create_model(
        img_shape=img_shape,
        optimizer=Adam(1e-5),
        n_class=train_config["n_class"],
        rand_seed=train_config["random_state"],
    )
    model.load_weights(train_config["model_path"])

    prediction = model.predict(preprocessed_image)

    return prediction
