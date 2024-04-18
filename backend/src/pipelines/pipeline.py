"""
Программа: Сборный конвейер для тренировки модели
Версия: 1.0
"""

import yaml

from ..data.get_data import get_dataset
from ..data.split_dataset import get_train_val_test
from ..transform.data_generator import generator_make
from ..train.train import train_model
from ..train.class_weight import get_class_weights
from ..train.label_encode import get_label_map
from ..train.save_load_func import save_func
from ..train.pred import model_pred


def pipeline_training(config_path: str) -> None:
    """
    Полный цикл получения данных, предобработки и тренировки модели.

    Parameters
    ----------
    config_path:
        путь до файла с конфигурациями

    Returns
    -------
    None
        Получает данные, предобрабатывает их и тренирует модель.
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    prep_config = config["preprocessing"]
    train_config = config["train"]

    # get data
    train_data = get_dataset(dataset_path=prep_config["train_path_proc"])

    # split data
    df_train, df_val, df_test = get_train_val_test(dataset=train_data, **train_config)

    # preprocessing
    train_generator = generator_make(data=df_train, train_flg=True, **train_config)
    val_generator = generator_make(data=df_val, train_flg=False, **train_config)
    test_generator = generator_make(data=df_test, train_flg=False, **train_config)

    # label encode
    label_map = get_label_map(gen=train_generator)
    save_func(result=label_map, result_path=train_config["label_map_path"])

    # class_weight
    class_weights = get_class_weights(train_gen=train_generator)

    # train
    model = train_model(
        train_gen=train_generator,
        val_gen=val_generator,
        test_gen=test_generator,
        class_weights=class_weights,
        **train_config
    )

    # prediction
    y_true, y_pred_labels = model_pred(model=model, test_gen=test_generator)

    return y_true, y_pred_labels
