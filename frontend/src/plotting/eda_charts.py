"""
Программа: Отрисовка графиков EDA
Версия: 1.0
"""

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def barplot_make(data: pd.DataFrame, target: str) -> matplotlib.figure.Figure:
    """
    Построение barplot с указанием значений над столбцами.

    Parameters
    ----------
    data: pd.DataFrame
        Данные.
    target: str
        Название столбца для нормирования данных.

    Returns
    -------
    matplotlib.figure.Figure
        Выводит график barplot с указанием значений над столбцами.
    """
    data_label = (
        data[target]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percent")
        .reset_index()
    )

    fig = plt.figure(figsize=(10, 6))

    ax = sns.barplot(x=target, y="percent", data=data_label, palette="viridis")

    for p in ax.patches:
        percentage = f"{p.get_height():.1f}"
        if p.get_height() != 0:
            ax.annotate(
                percentage,  # текст
                # координата xy
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                # центрирование
                ha="center",
                va="center",
                xytext=(0, 7),
                # точка смещения относительно координаты
                textcoords="offset points",
                fontsize=12,
            )

    plt.title("Соотношение классов", fontsize=16)
    plt.xlabel(target, fontsize=12)
    plt.ylabel("Percent", fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    return fig


def cls_example(
    data: pd.DataFrame, col_img: str, col_cls: str
) -> matplotlib.figure.Figure:
    """
    Показывает примеры изображений каждого класса.

    Parameters
    ----------
    data: pd.DataFrame:
        Данные.
    col_img: str
        Колонка с путем к изображениям.
    col_cls: str
        Колонка с метками классов.

    Returns
    -------
    matplotlib.figure.Figure
        Выводит примеры изображений каждого класса.
    """

    names_cls = data[col_cls].unique().tolist()

    # определим размеры графика и количество axes
    ncols = 3
    nrows = len(names_cls) // 3 + 1
    fig = plt.figure(figsize=(10, 10))

    # пройдем по каждому классу
    for i, label in enumerate(names_cls):
        df_target = data[data[col_cls] == label]
        np.random.seed(26)
        random_num = np.random.randint(0, df_target.shape[0])
        image_path = df_target[col_img].iloc[random_num]
        img = plt.imread(image_path)
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")

    plt.tight_layout()

    return fig
