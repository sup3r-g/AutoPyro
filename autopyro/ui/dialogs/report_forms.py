"""Build the report composer, navigation and command menus."""

from __future__ import annotations

from collections import Counter

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from autopyro.curves.joint_sources import joint_report_snapshots

from ..i18n import tr
from ..theme import icon
from ..widgets.groups import group_layout
from ..widgets.parameters import add_parameter_tools
from .report_preview import ReportPreview
from .report_templates import PLOT_KEY_ROLE


def build(sheet):
    layout = QVBoxLayout(sheet)
    splitter = QSplitter()
    layout.addWidget(splitter)
    panel, template, labels = _settings_panel(sheet, splitter)
    _preview_panel(sheet, splitter)
    build_button, export_button = _commands(sheet, layout)
    _connect_inputs(sheet)
    columns_label, plots_label, conclusions_label, logo_label = labels
    group_layout(panel, [
        (columns_label, tr('Показатели')), (plots_label, tr('Графики')),
        (conclusions_label, tr('Результат')), (logo_label, tr('Оформление и форматы')),
    ])
    _bind_parameters(sheet)
    _install_shortcuts(sheet, template, build_button, export_button)


def _settings_panel(sheet, splitter):
    controls = QWidget()
    panel = QVBoxLayout(controls)
    template = _template_button(sheet, panel)
    form, logo_controls = _metadata_controls(sheet)
    _binding_controls(sheet)
    _report_settings(sheet, panel, form)
    columns_label = _column_controls(sheet, panel)
    plots_label = _plot_controls(sheet, panel)
    conclusions_label = _conclusions(sheet, panel)
    logo_label = _format_controls(panel, logo_controls)
    settings = QScrollArea()
    settings.setWidgetResizable(True)
    settings.setWidget(controls)
    splitter.addWidget(settings)
    return panel, template, (columns_label, plots_label, conclusions_label, logo_label)


def _template_button(sheet, panel):
    template = QPushButton(icon('layout-template'), tr('Применить шаблон…'))
    template.setToolTip(tr('Встроенный шаблон задаёт разделы и группировку. Показатели, интерполяции и лимиты остаются текущими.'))
    template.clicked.connect(sheet.load_template)
    panel.addWidget(template)
    return template


def _metadata_controls(sheet):
    form = QFormLayout()
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    sheet.title = QLineEdit(tr('Отчёт по данным пиролиза'))
    sheet.author = QLineEdit()
    sheet.logo_button = QPushButton(icon('folder-open'), tr('Выбрать логотип…'))
    sheet.logo_button.clicked.connect(sheet.choose_logo)
    clear_logo = QPushButton(icon('x'), tr('Убрать'))
    clear_logo.clicked.connect(lambda: sheet.set_logo(None))
    logo_controls = QHBoxLayout()
    logo_controls.addWidget(sheet.logo_button)
    logo_controls.addWidget(clear_logo)
    return form, logo_controls


def _binding_controls(sheet):
    owner = sheet.owner
    sheet.scope = QComboBox()
    for label, key in [
        (tr('Текущая таблица'), "filtered"),
        (tr('Все пробы'), "all"),
        (tr('Выделенные строки'), "selected"),
    ]:
        sheet.scope.addItem(tr(label), key)
    sheet.group = QComboBox()
    sheet.group.addItem(tr('Без группировки'), None)
    sheet.subgroup = QComboBox()
    sheet.subgroup.addItem(tr('Без второго уровня'), None)
    sheet.profile_depth = QComboBox()
    sheet.profile_depth.addItem(tr('По привязке или имени колонки'), None)
    sheet.columns = QListWidget()
    project_columns = owner.model.project.columns
    duplicates = Counter(column.name.casefold() for column in project_columns)
    visible_ids = set(owner.visible_ids())
    for number, column in enumerate(project_columns, 1):
        label = f"{column.name} [{column.unit}]" if column.unit else column.name
        if duplicates[column.name.casefold()] > 1:
            label += tr(' · колонка {0}', number)
        item = QListWidgetItem(label, sheet.columns)
        item.setData(Qt.ItemDataRole.UserRole, column.id)
        item.setCheckState(
            Qt.CheckState.Checked
            if column.id in visible_ids
            else Qt.CheckState.Unchecked
        )
        sheet.group.addItem(label, column.id)
        sheet.subgroup.addItem(label, column.id)
        if not column.metadata.get("method"):
            sheet.profile_depth.addItem(label, column.id)


def _report_settings(sheet, panel, form):
    owner = sheet.owner
    sheet.text_numbers = QCheckBox(tr('Использовать числа в текстовых колонках'))
    sheet.interpolated = QCheckBox(tr('Включать интерполяции в статистику'))
    sheet.detailed_correlations = QCheckBox(tr('Показывать все пары корреляций'))
    sheet.interpretation = QCheckBox(tr('Пересчитать сохранённую схему\nинтерпретации'))
    sheet.interpretation.setChecked(True)
    sheet.stratigraphy = QCheckBox(tr('Включить стратиграфическую привязку'))
    sheet.stratigraphy.setChecked(True)
    sheet.nonpyro = QCheckBox(tr('Включить геохимию вне пиролиза'))
    sheet.nonpyro.setChecked(True)
    sheet.limit, sheet.qc_limit, sheet.protocol_limit = (QSpinBox() for _ in range(3))
    for limit in (sheet.limit, sheet.qc_limit, sheet.protocol_limit):
        limit.setRange(0, 10000)
        limit.setValue(100)
    sheet.qc_open_only = QCheckBox(tr('Только замечания QC к рассмотрению'))
    sheet.joint_snapshot = QComboBox()
    snapshots = joint_report_snapshots(owner.model.project)
    for snapshot_id, snapshot in sorted(snapshots.items(), key=lambda item: item[1].get("saved_at", ""), reverse=True):
        axes = " → ".join(snapshot.get("axes", ()))
        saved = snapshot.get("saved_at", "").replace("T", " ", 1)
        label = f"{saved} · {axes} · N={len(snapshot.get('row_ids', ()))}"
        if snapshot.get("bootstrap") and (confidence := snapshot.get("plots", {}).get("confidence")) is not None:
            label += f" · {confidence*100:g}%"
        sheet.joint_snapshot.addItem(icon("chart-no-axes-combined"), label, snapshot_id)
    if not snapshots:
        sheet.joint_snapshot.addItem(tr("Нет сохранённых снимков"), None)
        sheet.joint_snapshot.setEnabled(False)
    sheet.scope_summary = QLabel()
    sheet.scope_summary.setWordWrap(True)
    form.addRow(tr('Выборка'), sheet.scope)
    form.addRow(sheet.scope_summary)
    for label, widget in [
        (tr('Заголовок'), sheet.title),
        (tr('Автор'), sheet.author),
        (tr('Группировка'), sheet.group),
        (tr('Второй уровень группировки'), sheet.subgroup),
        (tr('Глубина профиля'), sheet.profile_depth),
        (tr('Строк в приложении'), sheet.limit),
        (tr('Строк в таблицах QC'), sheet.qc_limit),
        (tr('Строк в протоколах геохимии'), sheet.protocol_limit),
        (tr('Снимок совместной подгонки'), sheet.joint_snapshot),
    ]:
        form.addRow(label, widget)
    form.addRow(sheet.qc_open_only)
    panel.addLayout(form)


def _column_controls(sheet, panel):
    columns_label = QLabel(tr('Колонки таблицы и статистики:'))
    columns_label.setBuddy(sheet.columns)
    sheet.columns.setAccessibleName(tr('Колонки таблицы и статистики'))
    panel.addWidget(columns_label)
    sheet.column_search = sheet.add_list_controls(panel, sheet.columns, tr('Поиск колонок'))
    panel.addWidget(sheet.columns)
    panel.addWidget(sheet.text_numbers)
    panel.addWidget(sheet.interpolated)
    panel.addWidget(sheet.detailed_correlations)
    panel.addWidget(sheet.interpretation)
    panel.addWidget(sheet.stratigraphy)
    panel.addWidget(sheet.nonpyro)
    return columns_label


def _plot_controls(sheet, panel):
    owner = sheet.owner
    plots_label = QLabel(tr('Графики (все строятся на выборке отчёта):'))
    panel.addWidget(plots_label)
    sheet.plots = QListWidget()
    plots_label.setBuddy(sheet.plots)
    sheet.plots.setAccessibleName(tr('Графики для отчёта'))
    configs = [
        ("saved:" + name, tr('Сохранённый: {0}', name), config)
        for name, config in owner.model.project.view.get("plots", {}).items()
    ]
    if owner.model.project.view.get("plot_current"):
        configs.insert(0, ("current", tr('Текущий график'), owner.model.project.view["plot_current"]))
    for key, name, config in configs:
        item = QListWidgetItem(name, sheet.plots)
        item.setData(Qt.ItemDataRole.UserRole, config)
        item.setData(PLOT_KEY_ROLE, key)
        item.setCheckState(Qt.CheckState.Unchecked)
    sheet.plot_search = sheet.add_list_controls(panel, sheet.plots, tr('Поиск графиков'))
    panel.addWidget(sheet.plots)
    sheet.map_view = QComboBox()
    sheet.map_view.addItem(tr('Без карты'), None)
    for name in owner.model.project.view.get("map_views", {}):
        sheet.map_view.addItem(name, name)
    map_label = QLabel(tr('Сохранённый вид карты (раздел «Графики»)'))
    map_label.setBuddy(sheet.map_view)
    panel.addWidget(map_label)
    panel.addWidget(sheet.map_view)
    return plots_label


def _conclusions(sheet, panel):
    conclusions_label = QLabel(tr('Выводы автора (без автоматических геологических заключений):'))
    conclusions_label.setWordWrap(True)
    panel.addWidget(conclusions_label)
    sheet.conclusions = QTextEdit()
    conclusions_label.setBuddy(sheet.conclusions)
    sheet.conclusions.setAccessibleName(tr('Выводы автора'))
    sheet.conclusions.setMaximumHeight(90)
    panel.addWidget(sheet.conclusions)
    return conclusions_label


def _format_controls(panel, logo_controls):
    logo_label = QLabel(tr('Логотип для DOCX'))
    panel.addWidget(logo_label)
    panel.addLayout(logo_controls)
    note = QLabel(
        tr('JSON содержит все колонки и исходные ячейки выбранных проб.\nЛимит строк относится только к приложению HTML/PDF/DOCX.\nРазбиение PDF и DOCX проверяется через меню «Страницы…».')
    )
    note.setWordWrap(True)
    panel.addWidget(note)
    return logo_label


def _preview_panel(sheet, splitter):
    preview_panel = QWidget()
    preview_layout = QVBoxLayout(preview_panel)
    preview_layout.setContentsMargins(0, 0, 0, 0)
    sheet.preview_sections = QComboBox()
    sheet.preview_sections.setAccessibleName(tr('Переход к разделу отчёта'))
    sheet.preview_sections.setPlaceholderText(tr('К разделу…'))
    sheet.preview_sections.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    sheet.preview_sections.setEnabled(False)
    navigation = QHBoxLayout()
    navigation.addWidget(sheet.preview_sections, 1)
    sheet.check_actions = QPushButton(icon('shield-check'), tr('К настройкам проверки…'))
    sheet.check_actions.hide()
    sheet.check_menu = QMenu(sheet.check_actions)
    sheet.check_actions.setMenu(sheet.check_menu)
    navigation.addWidget(sheet.check_actions)
    sheet.page_preview = QPushButton(icon('file-text'), tr('Страницы…'))
    sheet.page_preview.setEnabled(False)
    pages_menu = QMenu(sheet.page_preview)
    pages_menu.addAction(icon('file-text'), tr('Предпросмотр PDF…'), lambda: sheet.preview_pages("pdf"))
    pages_menu.addAction(icon('file-text'), tr('Открыть DOCX в редакторе…'), lambda: sheet.preview_pages("docx"))
    sheet.page_preview.setMenu(pages_menu)
    navigation.addWidget(sheet.page_preview)
    preview_layout.addLayout(navigation)
    sheet.preview_stale = QLabel(tr('Устарел. Пересоберите отчёт перед экспортом.'))
    sheet.preview_stale.setWordWrap(True)
    sheet.preview_stale.hide()
    preview_layout.addWidget(sheet.preview_stale)
    sheet.preview = ReportPreview()
    sheet.preview.setOpenLinks(False)
    sheet.preview.setTextInteractionFlags(
        sheet.preview.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByKeyboard
    )
    sheet.preview.setAccessibleName(tr('Предпросмотр отчёта'))
    preview_layout.addWidget(sheet.preview, 1)
    sheet.preview_sections.activated.connect(
        lambda index: sheet.preview.go_to_section(sheet.preview_sections.itemData(index))
    )
    splitter.addWidget(preview_panel)
    splitter.setSizes([440, 760])


def _commands(sheet, layout):
    buttons = QHBoxLayout()
    layout.addLayout(buttons)
    setup = QPushButton(icon('settings-2'), tr('Настройки'))
    setup_menu = QMenu(setup)
    setup.setMenu(setup_menu)
    buttons.addWidget(setup)
    for icon_name, label, callback in (
        ('layers', 'Разделы…', sheet.edit_sections),
        ('file-text', 'Комментарии…', sheet.edit_comments),
        ('save', 'Сохранить шаблон…', sheet.save_template),
        ('logs', 'История отчётов…', sheet.show_history),
    ):
        setup_menu.addAction(icon(icon_name), tr(label), callback)
    build = QPushButton(icon('refresh-cw'), tr('Собрать отчёт'))
    build.clicked.connect(sheet.build)
    buttons.addWidget(build)
    export_button = QPushButton(icon('log-out'), tr('Экспорт'))
    export_menu = QMenu(export_button)
    export_button.setMenu(export_menu)
    buttons.addWidget(export_button)
    batch = export_menu.addAction(icon('files'), tr('Пакет отчётов…'), sheet.export_batch)
    batch.setEnabled(False)
    export_menu.addSeparator()
    sheet.exports = [batch]
    for kind in ("html", "pdf", "docx", "json"):
        action = export_menu.addAction(icon('file-text'), tr('Экспорт {0}', kind.upper()),
                                       lambda checked=False, kind=kind: sheet.export(kind))
        action.setEnabled(False)
        sheet.exports.append(action)
    archive = export_menu.addAction(icon('save'), tr('Сохранить в историю…'), sheet.archive_report)
    archive.setEnabled(False)
    sheet.exports.append(archive)
    close = QPushButton(icon('x'), tr('Закрыть'))
    close.clicked.connect(sheet.reject)
    buttons.addWidget(close)
    sheet.status = QLabel(tr('Выберите разделы и соберите отчёт.'))
    sheet.status.setTextFormat(Qt.TextFormat.PlainText)
    sheet.status.setWordWrap(True)
    layout.addWidget(sheet.status)
    return build, export_button


def _connect_inputs(sheet):
    for widget in (sheet.title, sheet.author):
        widget.textChanged.connect(sheet.invalidate)
    for widget in (sheet.scope, sheet.group, sheet.subgroup, sheet.profile_depth):
        widget.currentIndexChanged.connect(sheet.invalidate)
    for widget in (
        sheet.text_numbers,
        sheet.interpolated,
        sheet.detailed_correlations,
        sheet.interpretation,
        sheet.stratigraphy,
        sheet.nonpyro,
    ):
        widget.toggled.connect(sheet.invalidate)
    for limit in (sheet.limit, sheet.qc_limit, sheet.protocol_limit):
        limit.valueChanged.connect(sheet.invalidate)
    sheet.qc_open_only.toggled.connect(sheet.invalidate)
    sheet.columns.itemChanged.connect(sheet.invalidate)
    sheet.plots.itemChanged.connect(sheet.invalidate)
    sheet.map_view.currentIndexChanged.connect(sheet.invalidate)
    sheet.joint_snapshot.currentIndexChanged.connect(sheet.invalidate)
    sheet.conclusions.textChanged.connect(sheet.invalidate)


def _bind_parameters(sheet):
    owner = sheet.owner
    add_parameter_tools(sheet, changed=sheet.invalidate, exclude=(sheet.column_search, sheet.plot_search))
    owner.analysis_scope.bind(sheet.scope, sheet.scope_changed)
    sheet.update_scope_summary()


def _install_shortcuts(sheet, template, build, export_button):
    QWidget.setTabOrder(template, sheet.scope)
    QWidget.setTabOrder(sheet.scope, sheet.title)
    template.setFocus(Qt.FocusReason.OtherFocusReason)
    # Typing in a filter should never activate a report action.
    for button in sheet.findChildren(QPushButton):
        button.setAutoDefault(False)
    sheet.report_shortcut(sheet, ("Ctrl+Return", "Ctrl+Enter"), sheet.build)
    sheet.report_shortcut(sheet, ("Ctrl+L",), sheet.focus_preview_sections)
    sheet.report_shortcut(sheet, ("Ctrl+Shift+E",), export_button.showMenu)
    build.setToolTip(tr('Собрать отчёт (Ctrl+Enter)'))
    export_button.setToolTip(tr('Открыть меню экспорта (Ctrl+Shift+E)'))
    sheet.preview_sections.setToolTip(tr('Переход к разделу (Ctrl+L)'))


def report_shortcut(parent, keys, callback):
    shortcut = QShortcut(parent)
    shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
    shortcut.setKeys([QKeySequence(key) for key in keys])
    shortcut.activated.connect(callback)


def add_list_controls(sheet, layout, items, search_label):
    controls = QHBoxLayout()
    search = QLineEdit()
    search.setPlaceholderText(search_label)
    search.setAccessibleName(search_label)
    search.setClearButtonEnabled(True)
    controls.addWidget(search, 1)
    select = QPushButton(icon('check'), tr('Отметить'))
    clear = QPushButton(icon('x'), tr('Снять'))
    select.setToolTip(tr('Отметить все видимые позиции в списке'))
    clear.setToolTip(tr('Снять отметки со всех видимых позиций в списке'))
    select.setAccessibleName(f"{search_label} · {select.toolTip()}")
    clear.setAccessibleName(f"{search_label} · {clear.toolTip()}")
    controls.addWidget(select)
    controls.addWidget(clear)
    layout.addLayout(controls)

    def filter_items(query):
        words = query.casefold().split()
        matches = False
        for i in range(items.count()):
            item = items.item(i)
            visible = all(word in item.text().casefold() for word in words)
            item.setHidden(not visible)
            matches |= visible
        select.setEnabled(matches)
        clear.setEnabled(matches)

    def check_visible(state):
        changed = False
        with QSignalBlocker(items):
            for i in range(items.count()):
                item = items.item(i)
                if not item.isHidden() and item.checkState() != state:
                    item.setCheckState(state)
                    changed = True
        if changed:
            sheet.invalidate()

    search.textChanged.connect(filter_items)
    select.clicked.connect(lambda: check_visible(Qt.CheckState.Checked))
    clear.clicked.connect(lambda: check_visible(Qt.CheckState.Unchecked))
    filter_items("")
    return search
