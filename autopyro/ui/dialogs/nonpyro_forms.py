"""Build non-pyrolysis geochemistry parameter pages, tables and commands."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableView,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from autopyro.analysis.nonpyro import METHODS

from ..i18n import tr
from ..widgets.documents import analysis_document, parameter_page, wrapped_check
from ..widgets.forms import Form
from ..widgets.parameters import add_parameter_tools
from .nonpyro_results import ProfileResultsModel
from ..theme import icon


def build(editor):
    panel, layout = analysis_document(editor, parameter_icon='flask-conical')
    context = _method_fields(editor)
    _register_fields(editor, panel)
    _parameter_pages(editor, panel)
    _result_views(editor, layout)
    _calculate_controls(editor, layout)
    action_layout = _result_actions(editor, layout)
    _chart_actions(editor, action_layout)
    _finish(editor, panel, layout, context)


def _method_fields(editor):
    editor.form = Form()
    editor.binding_options = Form()
    editor.conditions = Form()
    editor.form.section(tr('Методика'))
    editor.method = editor.form.combo(
        "method",
        tr('Метод расчёта'),
        [(tr(METHODS[key]["name"]), key) for key in editor.method_ids],
    )
    editor.binding_options.section(tr('Автоматический подбор колонок'))
    editor.group = editor.binding_options.combo("group", tr('Источник измерений'), [])
    editor.variant = editor.binding_options.combo(
        "variant",
        tr('Способ измерения'),
        [
            (tr('Авто · по привязкам'), ""),
            (tr('Площадь [a]'), "a"),
            (tr('Высота [h]'), "h"),
            (tr('Конц. [ch]'), "ch"),
            (tr('Калибр. площадь [ca]'), "ca"),
            (tr('Неопределённый [un]'), "un"),
            (tr('Объём [vol]'), "vol"),
            (tr('Моли [mol]'), "mol"),
        ],
    )
    editor.form.section(tr('Выборка'))
    editor.form.combo(
        "scope",
        tr('Выборка'),
        [
            (tr('Текущая таблица'), "filtered"),
            (tr('Все пробы'), "all"),
            (tr('Выделенные строки'), "selected"),
        ],
    )
    editor.conditions.section(tr('Контекст измерений'))
    context = editor.conditions.text("context", tr('Материал и условия анализа'))
    context.setPlaceholderText(tr('Например: нефть, сопоставимые C15+ фракции'))
    editor.conditions.check("text_numbers", tr('Читать числа в текстовых колонках'))
    editor.conditions.section(tr('Условия применимости'))
    editor.conditions.combo(
        "isotope_reference",
        tr('Шкала углеродных изотопов'),
        [(tr('Не подтверждена'), ""), (tr('VPDB (подтверждено)'), "VPDB")],
    )
    editor.conditions.combo(
        "material",
        tr('Объект Sofer'),
        [
            (tr('Не подтверждён'), ""),
            (tr('Нефть, сопоставимые C15+ Sat/Arom'), "oil_C15plus"),
        ],
    )
    editor.conditions.check(
        "population_confirmed", tr('Исходная популяция Ro подтверждена\n(modal/user)')
    )
    editor.conditions.section(tr('Допуски'))
    editor.conditions.number("closure_tolerance", tr('Допуск баланса, п.п.'), 2, 0, 100)
    editor.conditions.number(
        "comparison_tolerance", tr('Допуск сравнения CV'), 0.05, 0, 100, decimals=4
    )
    editor.conditions.section(tr('Подтверждение аналитика'))
    editor.conditions.check("confirmed", tr('Данные сопоставимы, условия методики соблюдены'))
    # One settings registry keeps project protocols and reset/load behavior
    # unchanged while the editors are arranged across three workflow pages.
    return context


def _register_fields(editor, panel):
    for page in (editor.binding_options, editor.conditions):
        editor.form.fields.update(page.fields)
        editor.form.defaults.update(page.defaults)
    for widget, kind in editor.form.fields.values():
        if kind == "combo":
            widget.setMinimumContentsLength(12)
            widget.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
        elif kind == "check":
            field_form = widget.parentWidget().layout()
            row, role = field_form.getWidgetPosition(widget)
            taken = field_form.takeRow(row)
            holder = QWidget()
            holder_layout = QVBoxLayout(holder)
            holder_layout.setContentsMargins(0, 0, 0, 0)
            wrapped_check(holder_layout, widget)
            field_form.insertRow(row, holder)
    panel.addWidget(editor.form)
    editor.parameter_tabs.setTabText(0, tr('1. Метод'))
    input_hint = editor.input_hint = QLabel()
    input_hint.setWordWrap(True)
    panel.insertWidget(0, input_hint)


def _parameter_pages(editor, panel):
    editor.notes = QTextBrowser()
    editor.notes.setOpenExternalLinks(True)
    editor.notes.setMinimumHeight(150)
    bindings_page = parameter_page(editor.parameter_tabs, tr('2. Привязки'), 'tags')
    binding_hint = QLabel(tr('Для каждого показателя выберите исходную колонку и единицу её значений. Обязательные показатели должны быть привязаны; необязательные можно оставить пустыми.'))
    binding_hint.setWordWrap(True)
    bindings_page.addWidget(binding_hint)
    bindings_page.addWidget(editor.binding_options)
    editor.autobind_button = QPushButton(icon('tags'), tr('Подобрать незаполненные привязки'))
    editor.autobind_button.setToolTip(tr('Подбирает колонки по источнику и способу измерения. Уже выбранные колонки сохраняются.'))
    editor.autobind_button.clicked.connect(editor.hints_changed)
    editor.variant.currentIndexChanged.connect(editor.hints_changed)
    bindings_page.addWidget(editor.autobind_button)
    editor.bindings_summary = QLabel()
    editor.bindings_summary.setWordWrap(True)
    bindings_page.addWidget(editor.bindings_summary)
    cards = QWidget()
    editor.bindings_layout = QVBoxLayout(cards)
    editor.bindings_layout.setContentsMargins(0, 0, 0, 0)
    bindings_page.addWidget(cards)
    conditions_page = parameter_page(editor.parameter_tabs, tr('3. Условия'), 'settings-2')
    conditions_hint = QLabel(tr('Проверьте условия методики и подтвердите сопоставимость данных. Затем нажмите «Рассчитать». Подробности находятся на вкладке «Методика».'))
    conditions_hint.setWordWrap(True)
    conditions_page.addWidget(conditions_hint)
    conditions_page.addWidget(editor.conditions)
    for page, text, index, image in ((panel, tr('Далее: привязать колонки →'), 1, 'redo-2'),
                              (bindings_page, tr('Далее: проверить условия →'), 2, 'redo-2')):
        button = QPushButton(icon(image), text)
        button.setAutoDefault(False)
        button.clicked.connect(lambda checked=False, target=index: editor.parameter_tabs.setCurrentIndex(target))
        page.addWidget(button)
    conditions_page.addStretch()


def _result_views(editor, layout):
    editor.results = QTableView()
    editor.results_model = ProfileResultsModel([], [], editor.results)
    editor.results.setModel(editor.results_model)
    editor.results.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    editor.results.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
    editor.results.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
    editor.result_tabs = QTabWidget()
    editor.result_tabs.addTab(editor.results, icon('table-2'), tr('Результаты'))
    editor.result_tabs.addTab(editor.notes, icon('file-text'), tr('Методика'))
    layout.addWidget(editor.result_tabs)
    editor.history_button = QPushButton(icon('logs'), tr('История расчётов…'))
    editor.history_button.clicked.connect(editor.open_history)
    layout.addWidget(editor.history_button)


def _calculate_controls(editor, layout):
    editor.workflow_hint = QLabel()
    editor.workflow_hint.setWordWrap(True)
    calculate_row = QHBoxLayout()
    calculate_row.addWidget(editor.workflow_hint, 1)
    editor.preview_button = QPushButton(icon('shield-check'), tr('Проверить выборку…'))
    editor.preview_button.setAutoDefault(False)
    editor.preview_button.clicked.connect(editor.show_profile_coverage)
    calculate_row.addWidget(editor.preview_button)
    editor.calculate_button = QPushButton(icon('calculator'), tr('Рассчитать'))
    editor.calculate_button.setIcon(icon('calculator'))
    editor.calculate_button.setAutoDefault(False)
    editor.calculate_button.clicked.connect(editor.calculate)
    calculate_row.addWidget(editor.calculate_button)
    layout.insertLayout(0, calculate_row)


def _result_actions(editor, layout):
    actions = QGroupBox(tr('Действия с готовым результатом'))
    action_layout = QVBoxLayout(actions)
    editor.result_actions = {}
    for key, text, slot, help_text, image in [
        ('promote', tr('Добавить в таблицу'), editor.promote,
         tr('Создаёт защищённые расчётные колонки с пересчётом при изменении входов. Доступна отмена.'), 'columns-3'),
        ('plot', tr('Построить график…'), editor.plot,
         tr('Добавляет расчётные показатели в таблицу и открывает их в редакторе графиков.'), 'chart-no-axes-combined'),
        ('select', tr('Показать пробы'), editor.select_samples,
         tr('Сначала выделите строки результатов. Их пробы будут выбраны в основной таблице.'), 'table-2'),
        ('export', tr('Сохранить протокол…'), editor.export,
         tr('Сохраняет текущий расчёт в JSON или CSV. Добавлять колонки для этого не требуется.'), 'log-out'),
    ]:
        button = QPushButton(icon(image), text)
        button.setAutoDefault(False)
        button.setToolTip(help_text)
        button.clicked.connect(slot)
        editor.result_actions[key] = button
        if key in ('promote', 'select'):
            action_row = QHBoxLayout()
            action_layout.addLayout(action_row)
        action_row.addWidget(button)
    layout.addWidget(actions)
    return action_layout


def _chart_actions(editor, action_layout):
    editor.alkane_profile_button = QPushButton(icon('chart-no-axes-combined'), tr('Профиль н-алканов'))
    editor.alkane_profile_button.setAutoDefault(False)
    editor.alkane_profile_button.setToolTip(tr('Выберите одну строку результатов с ненулевой суммой сигналов.'))
    editor.alkane_profile_button.clicked.connect(editor.show_alkane_distribution)
    action_layout.addWidget(editor.alkane_profile_button)
    editor.sara_plot_button = QPushButton(icon('chart-no-axes-combined'), tr('Треугольная диаграмма SARA'))
    editor.sara_plot_button.setAutoDefault(False)
    editor.sara_plot_button.setToolTip(tr('Показывает пробы с полным составом SARA и балансом в пределах допуска.'))
    editor.sara_plot_button.clicked.connect(editor.show_ternary)
    action_layout.addWidget(editor.sara_plot_button)
    editor.sara_profiles_button = QPushButton(icon('files'), tr('Сравнить профили SARA'))
    editor.sara_profiles_button.setAutoDefault(False)
    editor.sara_profiles_button.setToolTip(tr('Выберите от одной до восьми строк результатов SARA.'))
    editor.sara_profiles_button.clicked.connect(editor.show_component_profiles)
    action_layout.addWidget(editor.sara_profiles_button)
    editor.homohopane_button = QPushButton(icon('files'), tr('Сравнить профили C31–C35'))
    editor.homohopane_button.setAutoDefault(False)
    editor.homohopane_button.setToolTip(tr('Выберите от одной до восьми строк результатов C31–C35.'))
    editor.homohopane_button.clicked.connect(editor.show_component_profiles)
    action_layout.addWidget(editor.homohopane_button)
    editor.chung_button = QPushButton(icon('chart-no-axes-combined'), tr('Диаграмма Chung'))
    editor.chung_button.setAutoDefault(False)
    editor.chung_button.setToolTip(tr('Выберите от одной до восьми строк с изотопами газа.'))
    editor.chung_button.clicked.connect(editor.show_component_profiles)
    action_layout.addWidget(editor.chung_button)


def _finish(editor, panel, layout, context):
    owner = editor.owner
    module_id = editor.module_id
    editor.status = QLabel(
        tr('Выберите профиль и проверьте привязки. Неприменимые или отсутствующие данные не подставляются.')
    )
    editor.status.setWordWrap(True)
    editor.status.setMaximumHeight(70)
    layout.addWidget(editor.status)
    editor.results.selectionModel().selectionChanged.connect(editor.update_result_actions)
    editor.method.currentIndexChanged.connect(editor.method_changed)
    editor.form.edited.connect(editor.invalidate)
    editor.conditions.edited.connect(editor.invalidate)
    context.textChanged.connect(editor.invalidate)
    current = owner.model.project.view.get("geochemistry_current", {}).get(module_id,
                owner.model.project.view.get("nonpyro_current", {}))
    if not current:
        current = {"method": owner.model.project.source.get("method_profile")}
    editor.method.setCurrentIndex(
        max(0, editor.method.findData(current.get("method", "sofer")))
    )
    editor.loading = False
    editor.method_changed()
    editor.connect_model()
    add_parameter_tools(editor, reset=editor.restore_parameters, layout=panel,
                        reset_text=tr('Сбросить настройки модуля'))
    panel.addStretch()
    owner.analysis_scope.bind(editor.form.fields['scope'][0], editor.input_changed)
    editor.update_workflow()
