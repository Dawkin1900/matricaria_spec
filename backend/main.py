"""
Программа: Модель для определения ромашки лекарственной и ее примесей по фото
Версия: 1.0
"""

import io
import numpy as np
import pandas as pd
from PIL import Image
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile

from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.train.save_load_func import load_func

import warnings

warnings.filterwarnings("ignore")

app = FastAPI()
CONFIG_PATH = "../config/params.yml"


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    y_true, y_pred_labels = pipeline_training(config_path=CONFIG_PATH)
    metrics = load_func(config_path=CONFIG_PATH, name="metrics_path")
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred_labels": y_pred_labels,
    }


@app.post("/predict")
def prediction(file: bytes = File(...)):
    """
    Предсказание модели по данным из файла
    """
    image = Image.open(io.BytesIO(file))
    result = pipeline_evaluate(config_path=CONFIG_PATH, image=image)
    result_list = result[0].tolist()
    label_map = load_func(config_path=CONFIG_PATH, name="label_map_path")
    lst_cls = list(label_map.keys())
    return {"prediction": result_list, "lst_cls": lst_cls}


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)
