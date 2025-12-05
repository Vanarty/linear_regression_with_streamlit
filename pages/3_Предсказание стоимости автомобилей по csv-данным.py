import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction import FeatureHasher
from io import BytesIO
import pickle
from pathlib import Path
import re

st.set_page_config(page_title="Модель предсказания стоимости автомобиля на csv-данных", page_icon="🎯", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models_artefacts"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "ohe_encoder.pkl"
IMPUTER_PATH = MODEL_DIR / "miss_imputer.pkl"
TRAIN_DATA_PATH = DATA_DIR / "df_train.parquet"

# список правильных колонок 
NAME_COLS_IN_TRAIN = [
    "name",
    "year",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "km_driven",
    "fuel",
    "transmission",
    "seller_type",
    "owner",
    "seats"
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

    if not set(df.columns).issubset(set(name_cols)):
        st.error(f"😕 В вашем файле есть недопустимые столбцы! {df.columns}")
        st.stop()

    if len(df.columns) != len(name_cols):
        st.error(f"😕 В вашем файле есть недопустимые столбцы! {df.columns}")
        st.stop()


# вспомогательные функции для предобработки данных
def convert_mileage(value):
    """
    Обработка признака mileage: удаление единиц измерения и преобразование к float
    """
    # если значение пустое
    if pd.isna(value) or value == '':
        # возвращаем nan
        return np.nan

    # приводим к нижнему регистру и удаляем лишние пробелы
    value = str(value).lower().strip()
    
    # удаляем единицы измерения
    value = value.replace('km/kg', '').replace('kmpl', '')
    value = value.strip()
    
    try:
        return float(value)
    except:
        return np.nan


def convert_engine(value):
    """
    Обработка признака engine: удаление 'CC' и преобразование к float
    """
    # если значение пустое
    if pd.isna(value) or value == '':
        # возвращаем nan
        return np.nan

    # приводим к нижнему регистру, удаляем лишние пробелы и удаляем единицы измерения
    value = str(value).lower().replace('cc', '').strip()
    
    try:
        return float(value)
    except:
        return np.nan


def convert_max_power(value):
    """
    Обработка признака max_power: удаление 'bhp' и преобразование к float
    """
    # если значение пустое
    if pd.isna(value) or value == '':
        # возвращаем nan
        return np.nan

    # приводим к нижнему регистру, удаляем лишние пробелы и удаляем единицы измерения
    value = str(value).lower().replace('bhp', '').strip()
    
    try:
        return float(value)
    except:
        return np.nan


def extract_torque_components(value):
    """
    Обработка признака torque: разделение на torque и max_torque_rpm
    Правила обработки:
    - '190Nm@ 2000rpm' -> (190.0, 2000.0)
    - '11.5@ 4,500(kgm@ rpm)' -> (112.7, 4500.0)
    - '25.5 kgm at 2400-2800 rpm' -> (249.9, 2600.0)
    - '35.7@ 1750-3000(kgm@ rpm)' -> (350.0, 2375.0)
    - '48@ 3000+/-500(nm@ rpm)' -> (48.0, 3000.0)  # игнорируем +/- отклонение
    """
    # если значение пустое
    if pd.isna(value) or value == '':
        # возвращаем nan
        return np.nan, np.nan

    # приводим к нижнему регистру и удаляем лишние пробелы
    value = str(value).lower().strip()
    
    # инициализируем переменные
    torque_value = np.nan
    rpm_value = np.nan
    
    try:
        # Удаляем запятые в числах
        value = value.replace(',', '')
        
        # Случай 1: Формат со скобками
        if '(' in value and ')' in value:
            # Извлекаем основную часть до скобок
            main_part = value.split('(')[0].strip()
            # Извлекаем единицы измерения из скобок
            units_part = value.split('(')[1].split(')')[0].strip()
            
            # Разделяем основную часть
            if '@' in main_part:
                torque_str, rpm_str = main_part.split('@', 1)
                torque_str = torque_str.strip()
                rpm_str = rpm_str.strip()
                
                # Извлекаем числовое значение torque
                torque_value = float(torque_str)
                
                # Обрабатываем RPM - может быть с +/- отклонением
                rpm_str_clean = re.sub(r'\+/-.*', '', rpm_str)  # удаляем +/- часть
                rpm_str_clean = re.sub(r'±.*', '', rpm_str_clean)  # удаляем ± часть
                
                if '-' in rpm_str_clean and not rpm_str_clean.startswith('-'):
                    # Диапазон RPM: '1750-3000'
                    rpm_parts = rpm_str_clean.split('-')
                    if len(rpm_parts) == 2:
                        rpm_min, rpm_max = map(float, rpm_parts)
                        rpm_value = (rpm_min + rpm_max) / 2
                else:
                    # Одиночное значение RPM
                    rpm_numbers = re.findall(r'\d+\.?\d*', rpm_str_clean)
                    if rpm_numbers:
                        rpm_value = float(rpm_numbers[0])
                
                # Определяем единицы измерения torque
                if 'kgm' in units_part and 'nm' not in units_part:
                    # Преобразуем kgm в Nm (1 kgm = 9.80665 Nm)
                    torque_value *= 9.80665
            
            continue_processing = False
        else:
            continue_processing = True
        
        # Случай 2: Стандартная обработка для других форматов
        if continue_processing:
            # Разделяем на части по @ или at
            if '@' in value:
                parts = value.split('@')
            elif 'at' in value:
                parts = value.split('at')
            else:
                parts = [value]
            
            # Извлекаем значение крутящего момента
            torque_part = parts[0].strip()
            
            # Паттерны для разных единиц измерения
            patterns = [
                r'(\d+\.?\d*)\s*nm',
                r'(\d+\.?\d*)\s*kgm',  
                r'(\d+\.?\d*)\s*kg',
            ]
            
            torque_found = False
            for pattern in patterns:
                match = re.search(pattern, torque_part)
                if match:
                    torque_value = float(match.group(1))
                    if 'kg' in pattern and 'nm' not in torque_part.lower():
                        torque_value *= 9.80665
                    torque_found = True
                    break
            
            # Если не нашли с единицами, пробуем извлечь просто число
            if not torque_found:
                numbers = re.findall(r'\d+\.?\d*', torque_part)
                if numbers:
                    torque_value = float(numbers[0])
            
            # Извлекаем RPM
            if len(parts) > 1:
                rpm_part = parts[1].strip()
                
                # Очищаем RPM часть от +/- отклонений
                rpm_part_clean = re.sub(r'\+/-.*', '', rpm_part)  # удаляем +/- часть
                rpm_part_clean = re.sub(r'±.*', '', rpm_part_clean)  # удаляем ± часть
                
                # Обрабатываем диапазон RPM
                rpm_range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*rpm', rpm_part_clean)
                if rpm_range_match:
                    rpm_min = float(rpm_range_match.group(1))
                    rpm_max = float(rpm_range_match.group(2))
                    rpm_value = (rpm_min + rpm_max) / 2
                else:
                    # Пробуем найти диапазон без указания 'rpm'
                    rpm_range_simple = re.search(r'(\d+)\s*-\s*(\d+)', rpm_part_clean)
                    if rpm_range_simple:
                        rpm_min = float(rpm_range_simple.group(1))
                        rpm_max = float(rpm_range_simple.group(2))
                        rpm_value = (rpm_min + rpm_max) / 2
                    else:
                        # Ищем одиночное значение RPM
                        rpm_match = re.search(r'(\d+\.?\d*)\s*rpm', rpm_part_clean)
                        if rpm_match:
                            rpm_value = float(rpm_match.group(1))
                        else:
                            # Пробуем найти число без явного указания rpm
                            rpm_numbers = re.findall(r'\d+\.?\d*', rpm_part_clean)
                            if rpm_numbers:
                                rpm_value = float(rpm_numbers[0])
        
    except Exception as e:
        print(f"Ошибка при обработке torque: '{value}', ошибка: {e}")
    
    return round(torque_value, 2), rpm_value


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

    # Загрузка imputer для заполнения пропусков
    try:
        with open(IMPUTER_PATH, 'rb') as f:
            median_imputer = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Ошибка загрузки imputer: {e}")
        st.stop()

    # Загрузка ohe encoder
    try:
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Ошибка загрузки обработчика признаков: {e}")
        st.stop()   

    # предобработка признаков mileage, engine, max_power, torque
    df_proc['mileage'] = df_proc['mileage'].apply(convert_mileage)
    df_proc['engine'] = df_proc['engine'].apply(convert_engine) 
    df_proc['max_power'] = df_proc['max_power'].apply(convert_max_power)
    # Обрабатываем torque
    torque_results = df_proc['torque'].apply(extract_torque_components)
    df_proc['torque'] = [x[0] for x in torque_results]
    df_proc['max_torque_rpm'] = [x[1] for x in torque_results]
    # Преобразуем к float
    df_proc['mileage'] = df_proc['mileage'].astype(float)
    df_proc['engine'] = df_proc['engine'].astype(float)
    df_proc['max_power'] = df_proc['max_power'].astype(float)
    df_proc['torque'] = df_proc['torque'].astype(float)
    df_proc['max_torque_rpm'] = df_proc['max_torque_rpm'].astype(float)

    # Заполняем пропуски с помощью imputer
    try:
        for col in df_proc.columns:
            if (df_proc[col].dtype in ('object', 'bool')) | (df_proc[col].isnull().sum() > 0):
                df_proc[col] = df_proc[col].fillna(df_train[col].mode().iloc[0])  # заполняем пропуски с помощью самого частого значения
            elif (df_proc[col].dtype in ('int', 'float')) |  (df_proc[col].isnull().sum() == 0):
                if col in median_imputer:
                    df_proc[col] = df_proc[col].fillna(median_imputer[col])
                else:
                    df_proc[col] = df_proc[col].fillna(df_train[col].median())
    except Exception as e:
        st.error(f"❌ Ошибка при заполнении пропусков: {e}")
        st.stop()   
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

    # Удалим исходную колонку и остальные категориальные признаки и целевую переменную
    X_test_cat = X_test_cat.drop(columns=['seats', 'fuel', 'seller_type', 'transmission', 'owner'])

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
st.title("🎯 Предсказание стоимости автомобиля")

# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)
# если есть целевая колонка исключаем е
if 'selling_price' in df.columns:
    df = df.drop(columns=['selling_price'])
# Проверяем входные данные
check_data(df, NAME_COLS_IN_TRAIN)

# загружаем тренировочные данные для предобработки входных данных
train_df = pd.read_parquet(TRAIN_DATA_PATH)

try:
    features = prepare_features(df, train_df)
    predictions = np.expm1(MODEL.predict(features))

    df['prediction'] = predictions

except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()


# --- Метрики ---
st.subheader("📊 Результаты")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Всего записей об автомобилях", len(df))
with col2:
    mean_selling_price_predict = df['prediction'].mean() 
    st.metric("Средняя предсказанная цена автомобилей: ", f"{mean_selling_price_predict:.2f} y.e.")
with col3:
    min_predict_price = df['prediction'].min() 
    st.metric("Минимальная предсказанная цена автомобиля: ", f"{min_predict_price:.2f} y.e")
    max_predict_price = df['prediction'].max() 
    st.metric("Максимальная предсказанная цена автомобиля: ", f"{max_predict_price:.2f} y.e")

# Создаем буфер для записи Excel файла
output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Sheet1')

# Получаем данные из буфера
excel_data = output.getvalue()

# Кнопка для скачивания
st.download_button(
    label="📥 Скачать Excel с результатами предсказания",
    data=excel_data,
    file_name="данные.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# --- Визуализации ---
st.subheader("📈 Визуализации")

pred_counts = df['prediction'].value_counts().sort_index()
fig1 = px.histogram(
    data_frame=df['prediction'],
    x='prediction',
    nbins=60,  
    title="Распределение предсказаний"
)
st.plotly_chart(fig1, use_container_width=True)