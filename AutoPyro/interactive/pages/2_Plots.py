import streamlit as st
from streamlit_plotly_events import plotly_events
import plotly.express as px

# Config
st.set_page_config(
    page_title="Графики",
    page_icon="img/favicon.ico",
    layout="wide",
)

# Title
st.title("Построение графиков")

# Plot options
option = st.selectbox('Выберите тип графика который хотите построить?', TYPES.keys())
st.json()
st.button("Выбрать")


# Plotly Figure with event handling
fig = px.line(df, x="year", y="lifeExp", color='country')
selected_points = plotly_events(fig, click_event=True, hover_event=True)

# Slider to control the figure
sliders = {}
for name in TYPES[option]:
    sliders[name] = st.slider(name, 0, 130, 25)

st.write('Значения параметров:', A, K, B, Q, C, M)