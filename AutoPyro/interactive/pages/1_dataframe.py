import streamlit as st
import pandas as pd
from core.data import DataTable

# Config
st.set_page_config(
    page_title="Таблица",
    page_icon="img/favicon.ico",
    layout="wide",
)

# Title
st.title("Исследование данных")

# Upload datatable 
data_upload = st.file_uploader(
    "Загрузить данные", type=["csv", "xlsx", "json"]
)

df = DataTable()
# Save dataframe into session storage to access on other pages
st.session_state.dataframe = df

if data_upload:
    
    st.dataframe(data=df, use_container_width=True)
    
    data_format = st.radio(
        "Выберите формат данных для скачивания", ('.xlsx', '.csv')
    )
    if data_format == ".xlsx":
        st.download_button(
            "Скачать", df,
            f"{data_upload.name}_processed.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    elif data_format == '.csv':
        st.download_button(
            "Скачать", df,
            f"{data_upload.name}_processed.csv", "text/csv"
        )