import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
import pickle
from pathlib import Path

st.set_page_config(page_title="Модель предсказания стоимости автомобиля на csv-данных", page_icon="🎯", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models_artefacts"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "ohe_encoder.pkl"
TRAIN_DATA_PATH = DATA_DIR / "df_train.parquet"

# список правильных колонок 
NAME_COLS_IN_TRAIN = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "max_torque_rpm",
    "seats",
    "name"
]

# функция проверки входных данных
def check_data(df, name_cols):
    """
    Проверка входных данных на наличие ошибок
    """

    # проверяем наличие пустых строк
    if df.empty:
        st.error("😕 В вашем файле нет данных!")
        st.stop()
    
    # проверяем наличие пустых столбцов
    if df.columns.empty:
        st.error("😕 В вашем файле нет столбцов!")
        st.stop()

    if df.isnull().values.any():
        st.error("😕 Заполнены не все поля!")        
        st.stop()   

    if not set(df.columns).issubset(set(name_cols)):
        st.error(f"😕 В вашем файле есть недопустимые столбцы! {df.columns}")
        st.stop()

    if len(df.columns) != len(name_cols):
        st.error(f"😕 В вашем файле есть недопустимые столбцы! {df.columns}")
        st.stop()


@st.cache_resource
def load_model():
    """Загружаем модель и необходимые обработчики данных через pickle"""

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    return model


def prepare_features(df, df_train):
    """Приводим данные к формату обучения модели."""
    top_n = 20
    hashing_n_features = 10
    target_col = 'selling_price'
    df_proc = df.copy()

    # Загрузка scaler
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Ошибка загрузки стандартизатора: {e}")
        st.stop()

    # Загрузка ohe encoder
    try:
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Ошибка загрузки обработчика признаков: {e}")
        st.stop()   

    # приведем колонки к строгому порядку как в обучении
    df_proc = df_proc[NAME_COLS_IN_TRAIN]

    # Преобразуем к float
    df_proc['mileage'] = df_proc['mileage'].astype(float)
    df_proc['engine'] = df_proc['engine'].astype(float)
    df_proc['max_power'] = df_proc['max_power'].astype(float)
    df_proc['torque'] = df_proc['torque'].astype(float)
    df_proc['max_torque_rpm'] = df_proc['max_torque_rpm'].astype(float)

    # приведем колонки engine и seats к типу int
    df_proc['seats'] = df_proc['seats'].astype(int)
    df_proc['engine'] = df_proc['engine'].astype(int)

    # Frequency Encoding
    freq = df_train['name'].value_counts()
    df_proc['name_freq'] = df_proc['name'].map(freq)
    # если в тестовой выборке нет такой категории заполняем пропуск 0
    df_proc['name_freq'] = df_proc['name_freq'].fillna(0)
    # логарифмированная частота
    df_proc['name_freq_log'] = np.log1p(df_proc['name_freq'])
    
    # Top-N категорий
    top_cats = freq.head(top_n).index
    
    # One-hot для топ-N
    for i, cat in enumerate(top_cats, 1):
        df_proc[f'name_cat_{i:02d}'] = (df_proc['name'] == cat).astype(int)
    
    # Target Encoding 
    if target_col and target_col in df_train.columns:
        # Сглаженное среднее
        global_mean = df_train[target_col].mean()
        # коэф. сглаживания
        smoothing = 100
        
        def smoothed_target(group):
            '''Функция возвращает сглаженное кодирование target encoding'''
            n = len(group)
            if n == 0:
                return global_mean
            group_mean = group.mean()
            return (n * group_mean + smoothing * global_mean) / (n + smoothing)

        # посчитаем target encoder
        target_map = df_train.groupby('name')[target_col].apply(smoothed_target).to_dict()
        # применим на трейн и тест выборке
        df_proc['name_target'] = df_proc['name'].map(target_map)
        # если в тестовой выборке нет такой категории заполняем пропуск просто глобальным средним
        df_proc['name_target'] = df_proc['name_target'].fillna(global_mean)

    # Преобразуем строки в формат для хеширования
    name_strings_test = [[str(x)] for x in df_proc['name'].values]
    
    # Создаем FeatureHasher
    hasher = FeatureHasher(n_features=hashing_n_features, input_type='string')
    
    # Применяем хеширование
    hashed_features_test = hasher.transform(name_strings_test).toarray()
    
    hashed_test_df = pd.DataFrame(
        hashed_features_test,
        columns=[f'name_hash_{i}' for i in range(hashing_n_features)],
        index=df_proc.index
    )
    
    # Добавляем к основному DataFrame
    df_proc = pd.concat([df_proc, hashed_test_df], axis=1)

    # Также создаем агрегированные признаки из хешированных
    df_proc['name_hash_sum'] = hashed_features_test.sum(axis=1)
    df_proc['name_hash_mean'] = hashed_features_test.mean(axis=1)
    df_proc['name_hash_std'] = hashed_features_test.std(axis=1)
    
    # Удалим исходную колонку
    df_proc = df_proc.drop(columns=['name'])

    # Применим ohe энкодер на тест выборке
    encoded_test_array = encoder.transform(df_proc['seats'].values.reshape(-1, 1))

    # Получим имена новых колонок
    feature_names = encoder.get_feature_names_out(['seats'])

    # Создаем DataFrame с закодированными признаками test
    encoded_test_df = pd.DataFrame(
        encoded_test_array,
        columns=feature_names,
        index=df_proc.index
    )

    # Добавляем к основному DataFrame
    X_test_cat = pd.concat([df_proc, encoded_test_df], axis=1)
    
    # Удалим колонку `seats` 
    X_test_cat = X_test_cat.drop(columns=['seats'])

    # стандартизируем входные данные
    X_test_scaled = scaler.transform(X_test_cat)

    return X_test_scaled


# Загружаем модель
try:
    MODEL = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()


# --- Основной интерфейс ---
st.title("🎯 Предсказание стоимости автомобиля по введенным данным")

# --- Форма для заполнения и предсказания ---
st.subheader("🔮 Сделать предсказание для одного автомобиля")

with st.form("prediction_form"):
    col_left, col_right = st.columns(2)
    input_data = {}
    
    with col_left:
        st.write("**Числовые:**")
        for col in ["year", "seats", "mileage", "km_driven", "engine", "max_power", "torque", "max_torque_rpm"]:
            input_data[col] = st.number_input(col, min_value=0, key=col)

    with col_right:
        st.write("**Категориальные:**")
        for col in ["name"]:
            input_data[col] = st.text_input(
                "Введите название автомобиля",
                placeholder="Например: Honda Civic 1.8 S AT",
                key=col
            )

    submitted = st.form_submit_button("Предсказать", use_container_width=True)

if submitted:
    try:
        input_df = pd.DataFrame([input_data])
        # Проверяем входные данные
        check_data(input_df, NAME_COLS_IN_TRAIN)
        # загружаем тренировочные данные для предобработки входных данных
        train_df = pd.read_parquet(TRAIN_DATA_PATH)
        features = prepare_features(input_df, train_df)
        prediction = round(np.expm1(MODEL.predict(features))[0], 2)

        st.success(f"**Результат:** {prediction} y.e.")
    except Exception as e:
        st.error(f"❌ Ошибка при предсказании: {e}")