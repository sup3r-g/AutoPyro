# import panel as pn
# import sys
# import os
# # from maps import 
# from common import *
# from home import

# module_path = os.path.abspath("../../AutoPyro")
# if module_path not in sys.path:
#     sys.path.append(module_path)

# pn.extension(sizing_mode="stretch_width")

# pages = {
#     "Главная": header,
#     "Таблицы": pn.Column("# Таблицы", "...more bla"),
#     "Графики": pn.Column("# Графики", "...more bla"),
#     "Карты": pn.Column("# Карты", "...more bla"),
# }

# def show(page):
#     return pages[page]


# starting_page = pn.state.session_args.get("page", ["Главная"])[0]
# menu = radio_group = pn.widgets.RadioButtonGroup(
#     name="Меню",
#     value=starting_page,
#     options=list(pages.keys()),
#     button_type="primary",
#     orientation="vertical",
# )
# ishow = pn.bind(show, page=menu)
# pn.state.location.sync(menu, {"value": "page"})

# pn.template.MaterialTemplate(
#     title="Главная",
#     logo="interactive/panel/img/favicon.ico",
#     favicon="interactive/panel/img/favicon.ico",
#     accent_base_color=PRIMARY_COLOR,
#     header_background=PRIMARY_COLOR,
#     sidebar=[pn.pane.Markdown("### Меню"), menu],
#     main=ishow,
# ).servable(target="main")