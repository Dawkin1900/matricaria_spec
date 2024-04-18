"""
Программа: Отрисовка графиков train
Версия: 1.0
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
import matplotlib.pyplot as plt


def show_history_plot(history: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Отображает график метрик обучения модели.

    Parameters
    ----------
    history:  pd.DataFrame
        История обучения модели.

    Returns
    -------
    matplotlib.figure.Figure
        Вывод графика метрик обучения модели.
    """
    # определим количество эпох обучения
    training_accuracy = history["accuracy"]
    epochs = range(1, len(training_accuracy) + 1)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

    # вывод accuracy на тренировочных и валидационных данных
    axes[0].plot(
        epochs, history["accuracy"], "c", label="Training accuracy", marker="o"
    )
    axes[0].plot(
        epochs, history["val_accuracy"], "g", label="Validation accuracy", marker="o"
    )
    axes[0].set_title("Training and Validation accuracy", fontsize=14)
    axes[0].set_xlabel("Epochs", fontsize=12)
    axes[0].set_ylabel("accuracy", fontsize=12)
    axes[0].legend()
    axes[0].grid(True)

    # вывод f1_score на тренировочных и валидационных данных
    axes[1].plot(
        epochs, history["f1_score"], "c", label="Training f1 score", marker="o"
    )
    axes[1].plot(
        epochs, history["val_f1_score"], "g", label="Validation f1 score", marker="o"
    )
    axes[1].set_title("Training and Validation f1 score", fontsize=14)
    axes[1].set_xlabel("Epochs", fontsize=12)
    axes[1].set_ylabel("f1_score", fontsize=12)
    axes[1].legend()
    axes[1].grid(True)

    # вывод loss на тренировочных и валидационных данных
    axes[2].plot(epochs, history["loss"], "c", label="Training loss", marker="o")
    axes[2].plot(epochs, history["val_loss"], "g", label="Validation loss", marker="o")
    axes[2].set_title("Training and Validation Loss", fontsize=14)
    axes[2].set_xlabel("Epochs", fontsize=12)
    axes[2].set_ylabel("Loss", fontsize=12)
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    return fig


def show_conf_matrix(
    y_true: list, y_pred_labels: list, label_dict: dict
) -> matplotlib.figure.Figure:
    """
    Отображает графика confusion matrix .

    Parameters
    ----------
    y_true: list
        Реальные метки классов.
    y_pred_labels: list
        Предсказанные метки классов.
    label_dict: dict
        Словарь с соответствием закодированных меток классов

    Returns
    -------
    matplotlib.figure.Figure
        Вывод графика confusion matrix.
    """
    labels = list(label_dict.values())
    classes = list(label_dict.keys())

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred_labels, labels=labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    # график confusion matrix
    cmap = plt.cm.get_cmap("viridis")
    cm_display.plot(cmap=cmap, colorbar=False, xticks_rotation="vertical")

    plt.title("Confusion Matrix", fontsize=16)
    fig = plt.gcf()

    return fig
