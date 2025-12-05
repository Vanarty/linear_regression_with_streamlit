import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="Визуализация весов обученной модели", page_icon="🎯", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models_artefacts"
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"

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

@st.cache_resource
def load_model():
    """Загружаем модель и необходимые обработчики данных через pickle"""

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    return model


# Загружаем модель
try:
    MODEL = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

# Загрузка ohe encoder
try:
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
except Exception as e:
    st.error(f"❌ Ошибка загрузки названий признаков: {e}")
    st.stop()   

# --- Основной интерфейс ---
st.title(" Визуализация весов обученной модели")

# получаем названия фичей
if feature_names is not None:
    # Используем переданные названия
    st.success(f"✅ Используются переданные названия фичей: {len(feature_names)} признаков")

 # создаем DataFrame с весами
coefficients = MODEL.coef_

weights_df = pd.DataFrame({
    'Признак': feature_names,
    'Вес': coefficients,
    'Абсолютный_вес': np.abs(coefficients),
    'Знак': np.where(coefficients >= 0, '📈 Положительный', '📉 Отрицательный'),
    'Ранг': np.arange(1, len(coefficients) + 1)
})

# отсортируем по абсолютному значению
weights_df = weights_df.sort_values('Абсолютный_вес', ascending=False).reset_index(drop=True)
weights_df['Ранг_важности'] = weights_df.index + 1

# ТОП-15 наиболее важных признаков
st.header("🏆 Наиболее важные признаки")

# Позволяем пользователю выбрать количество признаков для отображения
n_top = st.slider(
    "Количество признаков для отображения:",
    min_value=5,
    max_value=min(30, len(weights_df)),
    value=15,
    key="n_top_slider"
)

top_df = weights_df.head(n_top).copy()

fig1 = px.bar(
    top_df,
    x='Абсолютный_вес',
    y='Признак',
    orientation='h',
    color='Знак',
    color_discrete_map={
        '📈 Положительный': '#2E86AB',
        '📉 Отрицательный': '#A23B72'
    },
    title=f'<b>Топ-{n_top} важнейших признаков</b>',
    text='Вес',
    hover_data=['Ранг_важности', 'Абсолютный_вес'],
    template='plotly_white+gridon',
    height=max(400, n_top * 25)
)

fig1.update_layout(
    xaxis_title="<b>Абсолютное значение веса</b>",
    yaxis_title="<b>Признак</b>",
    yaxis={'categoryorder': 'total ascending'},
    title_font_size=18,
    font_size=12,
    showlegend=True
)

fig1.update_traces(
    texttemplate='%{text:.4f}',
    textposition='outside',
    marker_line_color='black',
    marker_line_width=0.5
)

st.plotly_chart(fig1, use_container_width=True)

fig2 = px.histogram(
    weights_df,
    x='Вес',
    nbins=30,
    color='Знак',
    color_discrete_map={
        '📈 Положительный': '#2E86AB',
        '📉 Отрицательный': '#A23B72'
    },
    title='<b>Распределение весов</b>',
    template='plotly_white',
    marginal='box',
    opacity=0.8
)

fig2.update_layout(
    height=400,
    xaxis_title="<b>Значение веса</b>",
    yaxis_title="<b>Количество признаков</b>",
    showlegend=True
)

fig2.add_vline(
    x=0, 
    line_dash="dash", 
    line_color="gray", 
    opacity=0.7,
    annotation_text="Ноль", 
    annotation_position="top right"
)

st.plotly_chart(fig2, use_container_width=True)