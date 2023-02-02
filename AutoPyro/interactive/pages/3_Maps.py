import streamlit as st

# Config
st.set_page_config(
    page_title="Карты",
    page_icon="img/favicon.ico",
    layout="wide",
    )

# Title
st.title("Исследование данных")

data_upload = st.file_uploader("Загрузить данные", type=["csv", "xlsx", "json"])
if data_upload:
	st.download_button("Download your file here", data_upload, "my_image.png", "image/png")

st.dataframe(data=data_upload, use_container_width=True)