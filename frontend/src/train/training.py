"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import json
import requests
import streamlit as st
from ..data.get_data import get_dataset
from ..plotting.train_charts import show_history_plot, show_conf_matrix


def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели с выводом результатов

    Parameters
    ----------
    config: dict
        Конфигурационный файл.
    endpoint: object
        endpoint.

    Returns
    -------
    None
        Вывод метрик и графиков.
    """
    # Last metrics
    if os.path.exists(config["train"]["metrics_path"]):
        with open(config["train"]["metrics_path"]) as json_file:
            old_metrics = json.load(json_file)
    else:
        # если до этого не обучали модель и нет прошлых значений метрик
        old_metrics = {"loss": 0, "accuracy": 0, "f1": 0}

    # Train
    with st.spinner("Модель подбирает параметры..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Success!")

    new_metrics = output.json()["metrics"]

    # diff metrics
    loss, accuracy, f1_metric = st.columns(3)
    loss.metric(
        "loss",
        f"{new_metrics['loss']:.3f}",
        f"{new_metrics['loss']-old_metrics['loss']:.3f}",
    )
    accuracy.metric(
        "accuracy",
        f"{new_metrics['accuracy']:.3f}",
        f"{new_metrics['accuracy']-old_metrics['accuracy']:.3f}",
    )
    f1_metric.metric(
        "F1 score",
        f"{new_metrics['f1']:.3f}",
        f"{new_metrics['f1']-old_metrics['f1']:.3f}",
    )

    # plot history
    history = get_dataset(config["train"]["history_path"], index_flg=False)
    fig_history = show_history_plot(history)

    # plot confusion matrix
    with open(config["train"]["label_map_path"]) as json_file:
        label_dict = json.load(json_file)
    y_true = output.json()["y_true"]
    y_pred_labels = output.json()["y_pred_labels"]
    fig_matrix = show_conf_matrix(
        y_true=y_true, y_pred_labels=y_pred_labels, label_dict=label_dict
    )

    st.pyplot(fig_history, use_container_width=True)
    st.pyplot(fig_matrix, use_container_width=True)
