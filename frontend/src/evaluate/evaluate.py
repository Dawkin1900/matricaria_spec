"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st
from ..plotting.pred_charts import plot_final_result


def evaluate_from_file(endpoint: object, image_data: BytesIO) -> None:
    """
    Получение входных данных в качестве файла -> вывод результата в виде графика

    Parameters
    ----------
    endpoint: object
        endpoint для отправки запроса.
    image_data: BytesIO
        данные изображения в формате BytesIO.

    Returns
    -------
    None
        Вывод графика.
    """
    button_ok = st.button("Predict")
    if button_ok:
        files = {"file": ("image.png", image_data, "multipart/form-data")}
        with st.spinner("Модель предсказывает класс..."):
            output = requests.post(endpoint, files=files, timeout=8000)
        prediction = output.json()["prediction"]
        lst_cls = output.json()["lst_cls"]
        st.pyplot(plot_final_result(image_data, prediction, lst_cls))
