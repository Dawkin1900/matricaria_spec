"""
Программа: Отрисовка графиков prediction
Версия: 1.0
"""

from io import BytesIO
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def plot_image(image: BytesIO, prediction: np.ndarray, lst_cls: list) -> None:
    """
    Отображает изображение с предсказанием класса.

    Parameters
    ----------
    image: BytesIO
        Изображение.
    prediction: np.ndarray
        Предсказание модели.
    lst_cls: list
        Список классов.

    Returns
    -------
    None
     Вывод изображения с предсказанием класса.
    """
    img = Image.open(image)
    # img = plt.imread(image)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {lst_cls[np.argmax(prediction)]}")


def plot_value_array(prediction: np.ndarray, lst_cls: list) -> None:
    """
    Отображает график предсказаний модели для входного изображения.

    Parameters
    ----------
    prediction: np.ndarray
        Предсказание модели.
    lst_cls: list
        Список классов.

    Returns
    -------
    None
        Выводит график с предсказанием классов в %.
    """
    res = [x * 100 for x in prediction]
    thisplot = plt.bar(lst_cls, res, color="gray")
    predicted_label = np.argmax(prediction)
    thisplot[predicted_label].set_color("green")
    plt.ylabel("Persantage")
    plt.xticks(rotation=45)
    plt.title("Model predictions classes")


def plot_final_result(
    img: BytesIO, prediction: np.ndarray, lst_cls: list
) -> matplotlib.figure.Figure:
    """
    Отображает результаты предсказаний для нескольких изображений.

    Parameters
    ----------
    img: BytesIO
        Изображение.
    predictions: np.ndarray
        Предсказания модели для каждого изображения.
    lst_cls: list
        Список классов.

    Returns
    -------
    matplotlib.figure.Figure
        Выводит изображения и соответствующие им предсказания классов.
    """
    nrows = 2
    ncols = 1
    fig = plt.figure(figsize=(10, 5 * nrows))

    plt.subplot(nrows, ncols, 1)
    plot_image(image=img, prediction=prediction, lst_cls=lst_cls)
    plt.subplot(nrows, ncols, 2)
    plot_value_array(prediction=prediction, lst_cls=lst_cls)

    plt.tight_layout()
    return fig
