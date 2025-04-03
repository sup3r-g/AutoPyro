import os
import sys

import panel as pn
from common import PRIMARY_COLOR, SECONDARY_COLOR
from maps import MAP_UI
from tables import TABLE_UI
from plots import PLOT_UI


# module_path = os.path.abspath("../../AutoPyro")
# if module_path not in sys.path:
#     sys.path.append(module_path)

css = [
    f".bk-root .bk-btn-primary {{background-color: {PRIMARY_COLOR}; border-color: {PRIMARY_COLOR};}}",
    f".bk-root .bk-btn-primary.bk-active {{background-color: {SECONDARY_COLOR}; border-color: {SECONDARY_COLOR};}}",
    f".bk-root .bk-btn-primary:hover {{background-color: {SECONDARY_COLOR}; border-color: {SECONDARY_COLOR};}}",
]
pn.config.raw_css.extend(css)

pn.extension(sizing_mode="stretch_width")

material = pn.template.MaterialTemplate(
    title="Главная",
    logo="interactive/panel/img/favicon.ico",
    favicon="interactive/panel/img/favicon.ico",
    sidebar_width=200,
    accent_base_color=PRIMARY_COLOR,
    header_background=PRIMARY_COLOR,
    align="center",
)

header = pn.pane.Markdown(
    """
    ## Добро пожаловать в веб-интерфейс инструмента AutoPyro!
    
    ### Возможности инструмента:  
    + Работа с данными: восстановление пропусков в пиролитических данных (восстановление TOC).  
    + Создание графических материалов:  
        + Построение основных графиков, использующих для анализа ОВ и продуктивности НГМТ.  
        + Карт распространения и зрелости ОВ (используя параметры $$T_{max}$$).  
    + Автоматизация анализа информация:  
        + Вывод о типа ОВ, используя вероятностный подход для определения положения точек на цифровых вариантах. 
    """,
    align="center",
    sizing_mode="stretch_width",
)
powered_by = pn.Row(
    pn.pane.PNG(
        "https://github.com/holoviz/holoviz/raw/main/examples/assets/panel.png",
        embed=True,
        width=125,
    ),
    pn.pane.PNG(
        "https://github.com/holoviz/holoviz/raw/main/examples/assets/hvplot.png",
        embed=True,
        width=120,
    ),
    pn.pane.PNG(
        "https://github.com/holoviz/holoviz/raw/main/examples/assets/holoviews.png",
        embed=True,
        width=200,
    ),
    pn.pane.PNG(
        "https://github.com/holoviz/holoviz/raw/main/examples/assets/geoviews.png",
        embed=True,
        width=195,
    ),
    pn.pane.PNG(
        "https://github.com/holoviz/holoviz/raw/main/examples/assets/param.png",
        embed=True,
        width=130,
    ),
    pn.pane.PNG(
        "https://github.com/holoviz/holoviz/raw/main/examples/assets/colorcet.png",
        embed=True,
        width=205,
    ),
)

home_page = pn.Column(
    "# Главная",
    pn.Row(
        pn.layout.HSpacer(),
        header,
        pn.layout.HSpacer(),
        align="center",
        sizing_mode="stretch_width",
    ),
    # powered_by,
)

pages = {
    "Главная": home_page,
    "Таблицы": TABLE_UI,
    "Графики": PLOT_UI,
    "Карты": MAP_UI,
}


def show(page):
    material.title = page
    return pages[page]


starting_page = pn.state.session_args.get("page", ["Главная"])[0]
menu = pn.widgets.RadioButtonGroup(
    name="Меню",
    value=starting_page,
    options=list(pages.keys()),
    button_type="primary",
    orientation="vertical",
)
page_show = pn.bind(show, page=menu)
pn.state.location.sync(menu, {"value": "page"})
pn.state.location.sync(material, {"title": "page"})

material.main.append(page_show)
material.sidebar.extend([pn.pane.Markdown("## Меню", align="center"), menu])

material.servable(target="main")
