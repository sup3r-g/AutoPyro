"""Edit stratigraphy column appearance and correlation layout."""

from __future__ import annotations

from ..i18n import tr, message
from PySide6.QtWidgets import QDialog, QFormLayout, QLabel, QPushButton, QVBoxLayout
from autopyro.geology.stratigraphy import COLUMN_STYLE, column_style
from ..widgets.documents import parameter_page
from ..widgets.stratigraphy_style import ColumnStyleForm
from ..widgets.stratigraphy_correlation import CorrelationDialog
from ..widgets.stratigraphy_templates import ColumnStyleTemplates, template_style
from ..widgets.stratigraphy_labels import geological_label, geological_options
from ..theme import icon


def create_column_style(editor):
    page = parameter_page(editor.parameter_tabs, tr('2. Колонка'), 'sliders-horizontal')
    editor.style_templates = ColumnStyleTemplates(editor)
    page.addWidget(editor.style_templates)
    editor.column_style_form = form = ColumnStyleForm()
    page.addWidget(form)
    form.section(tr('Режим и геологические элементы'))
    form.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    for key, title, values in [
        ("display_mode", tr('Вид колонки'), geological_options(["log", "default", "grainsize"])),
        ("feature_mode", tr('Геологические элементы'), [(geological_label("features_default" if value == "default" else value), value) for value in ["off", "default", "merge", "semi-merge"]]),
        ("grain_preset", tr('Шкала зернистости'), geological_options(["clastic", "volcanic", "sedimentary", "dunham", "carbonate"])),
    ]:
        form.combo(key, title, values)
    form.form.addRow(form.hierarchy_group)
    form.form.addRow(form.comparison_group)
    form.correlation_button.clicked.connect(editor.configure_correlation)
    for key, label in [("unit_borders", tr('Границы слоёв')), ("spines", tr('Рамка осей')),
                       ("consec_units", tr('Объединять соседние слои одинаковой литологии'))]:
        form.check(key, label, COLUMN_STYLE[key])
    for key, label in [("mineral_size", tr('Размер минералов')), ("feature_size", tr('Размер признаков'))]:
        form.number(key, label, COLUMN_STYLE[key], .1, 10)
    form.section(tr('Размер и качество'))
    for key, label, low, high, integer in [
        ("width_mm", tr('Ширина, мм'), 30, 1000, False),
        ("height_mm", tr('Высота, мм'), 30, 1000, False),
        ("dpi", "DPI", 50, 600, True), ("ppi", tr('PPI узоров'), 50, 1200, True),
        ("alpha", tr('Непрозрачность заливки'), 0, 1, False),
    ]:
        form.number(key, label, COLUMN_STYLE[key], low, high, integer=integer)
    form.check("transparent", tr('Прозрачный фон'), False)
    form.section(tr('Подписи и оси'))
    form.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    form.font_selector("font_family", tr('Шрифт'), COLUMN_STYLE["font_family"])
    for key, label in [("title", tr('Заголовок (пусто — автоматически)')),
                       ("x_label", tr('Подпись X')),
                       ("y_label", tr('Подпись глубины'))]:
        form.text(key, label, COLUMN_STYLE[key])
    for key, label in [("labels", tr('Подписи слоёв')), ("x_axis", tr('Ось зернистости')),
                       ("y_axis", tr('Деления глубины')), ("grid", tr('Сетка глубины')),
                       ("info", tr('Система глубин и редакция под рисунком'))]:
        form.check(key, label, COLUMN_STYLE[key])
    for key, label in [("label_size", tr('Размер подписей слоёв')),
                       ("axis_size", tr('Размер подписей осей')), ("tick_size", tr('Размер делений')),
                       ("title_size", tr('Размер заголовка')), ("legend_size", tr('Размер легенды'))]:
        form.number(key, label, COLUMN_STYLE[key], 5, 48)
    for key, label in [("depth_min", tr('Глубина от (пусто — автоматически)')),
                       ("depth_max", tr('Глубина до (пусто — автоматически)')),
                       ("xmax", tr('Предел X (пусто — автоматически)'))]:
        form.text(key, label, optional=True)
    form.section(tr('Легенда'))
    form.check("legend", tr('Показывать легенду'), False)
    form.combo("legend_loc", tr('Расположение'), ["right", "left", "top", "bottom"])
    form.number("legend_columns", tr('Колонок легенды'), 1, 1, 10, integer=True)
    form.check("legend_border", tr('Рамка легенды'), False)
    editor.style_details_dialog = QDialog(editor)
    editor.style_details_dialog.setWindowTitle(tr('Дополнительное оформление колонки'))
    editor.style_details_dialog.resize(740, 760)
    details_layout = QVBoxLayout(editor.style_details_dialog)
    details_layout.addWidget(form.details)
    done = QPushButton(icon('check'), tr('Готово'))
    done.clicked.connect(editor.style_details_dialog.accept)
    details_layout.addWidget(done)
    details_button = QPushButton(icon('sliders-horizontal'), tr('Настроить легенду, шкалу и свои породы…'))
    details_button.clicked.connect(editor.style_details_dialog.exec)
    page.addWidget(details_button)
    hint = QLabel(tr('Задайте деления и группы в таблицах. Для своей породы выберите узор из каталога или изображение, цвет и название.'))
    hint.setWordWrap(True)
    page.addWidget(hint)
    form.defaults.update(COLUMN_STYLE)
    form.fields['grain_preset'][0].currentIndexChanged.connect(
        lambda: setattr(form.scale_editor, 'preset', form.fields['grain_preset'][0].currentData()))
    form.add_reset_button()
    editor.load_style(editor.owner.model.project.view.get("stratigraphy_style", {}))
    form.edited.connect(editor.update_style_controls)
    form.edited.connect(editor.schedule_column_preview)
    form.fields['display_mode'][0].currentIndexChanged.connect(
        lambda: form.fields['x_axis'][0].setChecked(form.fields['display_mode'][0].currentData() != 'log'))
    editor.update_style_controls()
    page.addStretch()


def load_style(editor, options):
    editor.column_style_form.load(column_style(options))
    editor.column_style_form.scale_editor.preset = editor.column_style_form.fields['grain_preset'][0].currentData()


def update_style_controls(editor):
    form = editor.column_style_form
    grain_mode = form.fields['display_mode'][0].currentData() != 'log'
    for key in ('grain_preset', 'x_axis', 'x_label', 'xmax'):
        form.fields[key][0].setEnabled(grain_mode)
    form.details.setTabEnabled(1, grain_mode)
    features = form.fields['feature_mode'][0].currentData() != 'off'
    for key in ('mineral_size', 'feature_size'):
        form.fields[key][0].setEnabled(features)
    legend = form.fields['legend'][0].isChecked()
    for key in ('legend_loc', 'legend_columns', 'legend_border', 'legend_size'):
        form.fields[key][0].setEnabled(legend)
    form.details.setTabEnabled(0, legend)


def style_values(editor):
    return column_style(editor.column_style_form.values())


def render_options(editor, template=None):
    options = editor.style_values()
    if template is not None:
        options.update(template_style(template))
    if options['legend_titles'] == COLUMN_STYLE['legend_titles']:
        options['legend_titles'] = list(editor.column_style_form.title_labels)
    if not options['correlation'].get('enabled', True):
        options['correlation'] = {}
    if options['correlation'] and options['y_label']:
        options['y_label'] = tr('Глубина относительно горизонта')
    elif options['y_label'] == COLUMN_STYLE['y_label']:
        options['y_label'] = tr('Глубина')
    return options


def configure_correlation(editor):
    try:
        scheme = editor.read_scheme()
    except ValueError as exc:
        editor.status.setText(tr('Ошибка: {0}', message(exc)))
        return
    form = editor.column_style_form
    levels = {editor.level.currentData(), *(rank for rank, check in form.hierarchy_checks.items()
                                        if check.isChecked())}
    plotted = {item['wellbore'] for item in scheme['intervals'] if item['level'] in levels}
    wells = [well for well in dict.fromkeys([editor.column_well.currentData(), *form.comparison_wells])
             if well in plotted]
    dialog = CorrelationDialog(wells, scheme['intervals'], form.correlation, scheme['depth_unit'], editor)
    if dialog.exec() == dialog.DialogCode.Accepted:
        settings = dialog.values()
        if settings != form.correlation:
            form.set_correlation(settings)
            form.edited.emit()
    dialog.deleteLater()
