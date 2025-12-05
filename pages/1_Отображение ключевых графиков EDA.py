import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from itertools import combinations


st.set_page_config(page_title="Отображение ключевых графиков EDA", page_icon="🎯", layout="wide")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DATA_PATH = DATA_DIR / "df_train.parquet"

# --- Основной интерфейс ---
st.title("Отображение ключевых графиков EDA")

# загружаем тренировочные данные для предобработки входных данных
train_df = pd.read_parquet(TRAIN_DATA_PATH)

# Определяем числовые колонки
numeric_cols = train_df.select_dtypes(include=['number']).columns.to_list()

st.markdown("<h5 style='text-align: left;'>График корреляции Пирсона и Спирмена по признакам</h5>", unsafe_allow_html=True)

if st.checkbox("Показать тепловые карты корреляций", value=True):
    # Числовые колонки
    numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    
    # Две карты в колонках
    col1, col2 = st.columns(2)
    
    with col1:
        pearson_corr = train_df[numeric_cols].corr(method='pearson')
        fig1 = px.imshow(pearson_corr, color_continuous_scale='RdBu')
        fig1.update_layout(title="Корреляция Пирсона", height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        spearman_corr = train_df[numeric_cols].corr(method='spearman')
        fig2 = px.imshow(spearman_corr, color_continuous_scale='RdBu')
        fig2.update_layout(title="Корреляция Спирмена", height=400)
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("<h5 style='text-align: left;'>Гистограммы по выбранным признакам</h5>", unsafe_allow_html=True)

# Выбор признаков для отображения
selected_cols = st.multiselect(
    "Выберите признаки для отображения:",
    options=numeric_cols,
    default=numeric_cols[:5],
    key="select_hist_cols"
)

if selected_cols:
    # Настройки визуализации
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nbins = st.slider("Количество бинов:", 10, 100, 30, key="nbins_slider")
    
    with col2:
        marginal_type = st.selectbox(
            "Дополнительный график:",
            ["none", "rug", "box", "violin"],
            key="marginal_type"
        )
    
    # Создаем отдельные графики
    for col in selected_cols:
        with st.expander(f"📈 {col}", expanded=True):
            fig = px.histogram(
                train_df,
                x=col,
                nbins=nbins,
                title=f"<b>Распределение: {col}</b>",
                template="plotly_white",
                marginal=marginal_type if marginal_type != "none" else None,
                opacity=0.8
            )
            
            fig.update_layout(
                height=400,
                xaxis_title=f"<b>{col}</b>",
                yaxis_title="<b>Частота</b>",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)


st.markdown("<h5 style='text-align: left;'>Попарный график распределения по выбранным признакам</h5>", unsafe_allow_html=True)

# Выбор признаков для отображения
selected_cols = st.multiselect(
    "Выберите признаки для анализа:",
    options=numeric_cols,
    default=numeric_cols[:4],  # Первые 4 по умолчанию
    key="select_scatter_cols"
)

if len(selected_cols) >= 2:
    # Настройки визуализации
    col1, col2 = st.columns(2)
    
    with col1:
        point_size = st.slider(
            "Размер точек:", 
            min_value=1, 
            max_value=20, 
            value=5, 
            key="point_size"
        )
    
    with col2:
        # Выбор цвета по категориальному признаку
        categorical_cols = train_df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        if categorical_cols:
            color_by = st.selectbox(
                "Цвет по категории:",
                options=['Нет'] + categorical_cols,
                key="color_by"
            )
        else:
            color_by = 'Нет'
    
    # Создаем все возможные пары
    pairs = list(combinations(selected_cols, 2))
    
    # Создаем отдельные графики для каждой пары
    for x_col, y_col in pairs:
        with st.expander(f"📊 {x_col} vs {y_col}", expanded=True):
            # Настройка цвета
            color_param = None if color_by == 'Нет' else color_by
            
            # Создаем scatter plot
            fig = px.scatter(
                train_df,
                x=x_col,
                y=y_col,
                color=color_param,
                title=f"<b>{x_col} vs {y_col}</b>",
                opacity=0.7,
                template="plotly_white"
            )
            
            # Настраиваем размер точек
            fig.update_traces(marker=dict(size=point_size))
            
            # Добавляем коэффициент корреляции
            correlation = train_df[x_col].corr(train_df[y_col])
            fig.add_annotation(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=f"Corr: {correlation:.3f}",
                showarrow=False,
                font=dict(size=12, color="red"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
            
            # Обновляем layout
            fig.update_layout(
                height=500,
                xaxis_title=f"<b>{x_col}</b>",
                yaxis_title=f"<b>{y_col}</b>",
                showlegend=(color_by != 'Нет')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Показываем статистику корреляции под графиком
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Корреляция Пирсона", f"{correlation:.3f}")
            with col_stat2:
                st.metric("Количество точек", len(train_df))
            with col_stat3:
                # Вычисляем R^2
                r_squared = correlation ** 2
                st.metric("R²", f"{r_squared:.3f}")

else:
    st.warning("⚠️ Выберите как минимум 2 признака для анализа")