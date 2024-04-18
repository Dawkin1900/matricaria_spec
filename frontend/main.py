"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os

import yaml
import streamlit as st
from src.data.get_data import load_data, get_dataset
from src.plotting.eda_charts import barplot_make, cls_example
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_from_file

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    st.title("MLOps project: Chamomile and its impurities 🌸")
    st.image(
        "https://d2seqvvyy3b8p2.cloudfront.net/47bfefe76db5de7db93be7f983f00223.jpg",
        width=600,
    )

    st.markdown("## Описание проекта")
    st.write(
        """
        Задача - создать модель машинного обучения для определения 
        ромашки лекарственной и ее примесей по фото."""
    )
    st.write(
        """*Ромашка аптечная (Matricaria chamomilla)* — это растение, 
             цветки которого используются в качестве лекарственного сырья
            как противовоспалительное и спазмолитическое средство при
            расстройствах деятельности желудочно-кишечного тракта, 
            для полоскания рта, для клизм и ванн."""
    )
    st.write(
        """
        При сборе цветков ромашки ее следует отличать от других растений, 
        похожих на нее по внешнему виду. 
        Данная задача является крайне необходимой для обеспечения качества и 
        безопасности лекарственных средств и здоровья потребителей"""
    )
    st.markdown(
        """
        **Для анализа взяты следующий растения:**
        - Ромашка аптечная (Matricaria chamomilla);
        - Ромашка непахучая, или Трёхрёберник (Tripleurospermum inodorum);
        - Ромашка пахучая (Matricaria discoidea);
        - виды Пупавки (Anthemis cotula, Anthemis arvensis, Anthemis ruthenica);
        - виды Пиретрума (Tanacetum parthenium, Tanacetum corymbosum);
        - Поповник, или Нивяник (Leucanthemum vulgare).
    """
    )
    st.markdown(
        """
        ## Источник данных
        Данные получены с сайта GBIF:
        - GBIF.org (22 March 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.5y8e9v
        - GBIF.org (22 March 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.mwhkp7
        - GBIF.org (22 March 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.qeykw8
        - GBIF.org (22 March 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.t7kjqb
        - GBIF.org (22 March 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.ag46f4
        - GBIF.org (22 March 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.mhzjgg
        - GBIF.org (22 March 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.c6phkv
        - GBIF.org (22 March 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.fwf425
        - GBIF.org (22 March 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.6rws37
        - GBIF.org (28 March 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.vzqpxy
        - GBIF.org (28 March 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.9qt846
        - GBIF.org (28 March 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.xty8vx
    """
    )
    st.markdown(
        """
        ## Описание полей
        - img - путь к изображениям;
        - label - метки классов (*целевая переменная*)
        """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load and write dataset
    data = get_dataset(dataset_path=config["preprocessing"]["train_path_proc"])
    st.write(data.head())

    # plotting with checkbox
    st.write("**Выберите графики:**")
    class_imbalance = st.checkbox("Соотношение классов")
    img_exsample = st.checkbox("Примеры изображений")

    if class_imbalance:
        st.pyplot(barplot_make(data=data, target=config["train"]["target_column"]))
        st.markdown(
            """
            **Вывод:**
            - наблюдается дисбаланс классов.
            """
        )
    if img_exsample:
        st.pyplot(
            cls_example(
                data=data, col_img="img", col_cls=config["train"]["target_column"]
            )
        )
        st.markdown(
            """
            **Вывод:**
            - больше всего по внешнему выделяются ромашка пахучая (matricaria_discoidea) 
            и виды пиретрума (pyrethrum_spec);
            - больше всего по внешнему схожи виды пупавки (anthemis), 
            ромашка аптечная (matricaria_chamomilla) и трехреберник (tripleurospermum_inodorum).
            """
        )


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["jpg", "png", "jpeg"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        image_data = load_data(data=upload_file)
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(endpoint=endpoint, image_data=image_data)
        else:
            st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Project description": main_page,
        "Exploratory data analysis": exploratory,
        "Training model": training,
        "Prediction from file": prediction_from_file,
    }
    selected_page = st.sidebar.radio("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
