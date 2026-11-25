"""Build curve fitting forms, actions and result views."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from matplotlib.figure import Figure

from ..i18n import tr
from ..theme.controls import apply_control_icons
from ..widgets.documents import analysis_document
from ..widgets.figure_canvas import CanvasView, FittedFigureCanvas
from ..widgets.navigation import NavigationToolbar
from .curve_fitting_inputs import _model_options
from ..theme import icon


def build(editor):
    controls, content = analysis_document(editor)
    form = _axes_controls(editor, controls)
    _model_controls(editor, controls, form)
    _regression_controls(editor, controls, form)
    _band_controls(editor, controls, form)
    _run_controls(editor, controls)
    editor.build_comparison_controls()
    editor.build_joint_controls(_model_options)
    _results_views(editor, content)
    editor.build_analysis_tools()
    editor.build_joint_view()
    _connect_inputs(editor)
    editor.canvas_view.bind_project(editor.owner, 'curve_fitting')
    apply_control_icons(editor)


def _axes_controls(editor, controls):
    owner = editor.owner
    form = QFormLayout()
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    controls.addLayout(form)
    editor.scope = QComboBox()
    owner.analysis_scope.bind(editor.scope, editor.invalidate)
    form.addRow(tr("Область"), editor.scope)
    editor.x_column, editor.y_column = QComboBox(), QComboBox()
    editor.sx_column, editor.sy_column = QComboBox(), QComboBox()
    for title, widget in (("X", editor.x_column), ("Y", editor.y_column),
                          ("σX (ODR)", editor.sx_column), ("σY", editor.sy_column)):
        form.addRow(title, widget)
    return form


def _model_controls(editor, controls, form):
    editor.model_choice = QComboBox()
    for label, key in (("Линейная", "linear"), ("Полиномиальная", "polynomial"),
                       ("Экспоненциальная", "exponential"), ("Степенная", "power"),
                       ("Логарифмическая", "logarithmic"), ("Константа", "constant"),
                       ("Пользовательская", "custom")):
        editor.model_choice.addItem(tr(label), key)
    form.addRow(tr("Модель"), editor.model_choice)
    editor.degree = QSpinBox()
    editor.degree.setRange(1, 10)
    editor.degree.setValue(2)
    form.addRow(tr("Степень"), editor.degree)
    editor.expression = QLineEdit("a*x + b")
    editor.parameter_names = QLineEdit("a, b")
    editor.initial = QLineEdit()
    editor.initial.setPlaceholderText(tr("Автоматически; для своей модели: 1, 1"))
    editor.model_formula = QLabel()
    editor.model_formula.setWordWrap(True)
    form.addRow(tr("Формула модели"), editor.model_formula)
    form.addRow(tr("Пользовательская формула Y ="), editor.expression)
    form.addRow(tr("Имена параметров"), editor.parameter_names)
    editor.initial.setToolTip(tr("Значения через запятую в порядке параметров; десятичный разделитель — точка"))
    form.addRow(tr("Начальные значения"), editor.initial)
    editor.lower_bounds, editor.upper_bounds = QLineEdit(), QLineEdit()
    for label, widget in ((tr("Нижние границы"), editor.lower_bounds), (tr("Верхние границы"), editor.upper_bounds)):
        widget.setPlaceholderText(tr("Пусто: без ограничений; пример: 0, -inf"))
        form.addRow(label, widget)
    editor.saved_models = QComboBox()
    editor.parameter_rules = QLineEdit()
    editor.parameter_rules.setPlaceholderText(tr("Фиксация: b=0; связь: c=2*a"))
    form.addRow(tr("Фиксация и связи параметров"), editor.parameter_rules)
    editor.free_parameter_label = QLabel()
    editor.free_parameter_label.setWordWrap(True)
    form.addRow(tr("Порядок свободных параметров"), editor.free_parameter_label)
    form.addRow(tr("Мои модели"), editor.saved_models)
    editor.refresh_saved_models()
    editor.saved_models.currentIndexChanged.connect(editor.load_custom_model)
    save_model = QPushButton(icon('save'), tr("Сохранить пользовательскую модель…"))
    save_model.clicked.connect(editor.save_custom_model)
    controls.addWidget(save_model)


def _regression_controls(editor, controls, form):
    editor.method = QComboBox()
    editor.method.addItem("OLS / WLS", "ols")
    editor.method.addItem("ODR / TLS", "odr")
    form.addRow(tr("Регрессия"), editor.method)
    editor.loss = QComboBox()
    editor.loss.addItem(tr("Квадратичная (обычная)"), "linear")
    editor.loss.addItem("Huber", "huber")
    editor.loss.addItem("Soft L1", "soft_l1")
    form.addRow(tr("Функция потерь"), editor.loss)
    editor.f_scale = QDoubleSpinBox()
    editor.f_scale.setDecimals(6)
    editor.f_scale.setRange(.000001, 1e9)
    editor.f_scale.setValue(1.)
    editor.f_scale.setToolTip(tr("Порог для остатков, делённых на σ; без σ — в единицах соответствующей оси"))
    form.addRow(tr("Робастный масштаб"), editor.f_scale)
    editor.starts = QSpinBox()
    editor.starts.setRange(1, 100)
    editor.starts.setValue(1)
    editor.start_seed = QSpinBox()
    editor.start_seed.setRange(0, 2**31-1)
    form.addRow(tr("Начальных приближений"), editor.starts)
    form.addRow(tr("Seed начальных приближений"), editor.start_seed)
    editor.absolute_sigma = QCheckBox(tr("Погрешности известны в абсолютных единицах"))
    controls.addWidget(editor.absolute_sigma)


def _band_controls(editor, controls, form):
    editor.confidence = QDoubleSpinBox()
    editor.confidence.setRange(50, 99.99)
    editor.confidence.setValue(95)
    editor.confidence.setSuffix(" %")
    form.addRow(tr("Уровень доверия"), editor.confidence)
    editor.band_method = QComboBox()
    editor.band_method.addItem(tr("Авто: аналитические / bootstrap"), "auto")
    editor.band_method.addItem(tr("Bootstrap (percentile)"), "bootstrap")
    editor.band_method.addItem(icon('chart-no-axes-combined'), tr("Bootstrap с одновременной полосой"), "bootstrap_simultaneous")
    editor.band_method.addItem(tr("Только аналитические"), "analytic")
    form.addRow(tr("Метод полос"), editor.band_method)
    editor.bootstrap_resamples = QSpinBox()
    editor.bootstrap_resamples.setRange(100, 10000)
    editor.bootstrap_resamples.setValue(500)
    form.addRow(tr("Bootstrap-повторы"), editor.bootstrap_resamples)
    editor.bootstrap_seed = QSpinBox()
    editor.bootstrap_seed.setRange(0, 2**31-1)
    form.addRow(tr("Bootstrap Seed"), editor.bootstrap_seed)
    editor.bootstrap_group = QComboBox()
    editor.bootstrap_group.setToolTip(tr("Выберите колонку с идентификатором группы. При расчёте bootstrap все точки группы пересэмплируются вместе."))
    form.addRow(tr("Группы bootstrap"), editor.bootstrap_group)
    editor.bootstrap_block_length = QSpinBox()
    editor.bootstrap_block_length.setRange(1, 2**31-1)
    editor.bootstrap_block_length.setSpecialValueText(tr("Без блоков"))
    editor.bootstrap_block_length.setToolTip(tr("Блоки формируются по возрастанию X. Используйте для одной серии с различными X и примерно равным шагом."))
    form.addRow(tr("Длина блока bootstrap"), editor.bootstrap_block_length)
    editor.prediction_sigma = QDoubleSpinBox()
    editor.prediction_sigma.setDecimals(6)
    editor.prediction_sigma.setRange(0, 1e12)
    editor.prediction_sigma.setSpecialValueText(tr("Эмпирические остатки"))
    editor.prediction_sigma.setToolTip(tr("0: смесь сырых остатков. Положительное значение: будущая σY для масштабирования стандартизованных остатков bootstrap."))
    form.addRow(tr("Будущая σY (bootstrap)"), editor.prediction_sigma)
    editor.show_confidence = QCheckBox(tr("Доверительная полоса"))
    editor.show_prediction = QCheckBox(tr("Предсказательная полоса"))
    for widget in (editor.show_confidence, editor.show_prediction):
        widget.setChecked(True)
        controls.addWidget(widget)
        widget.toggled.connect(editor.redraw)
    editor.confidence.valueChanged.connect(editor.redraw)


def _run_controls(editor, controls):
    editor.calculate_button = QPushButton(icon('calculator'), tr("Рассчитать"))
    editor.calculate_button.clicked.connect(editor.calculate)
    controls.addWidget(editor.calculate_button)
    editor.cancel_bootstrap = QPushButton(icon('x'), tr("Отменить bootstrap"))
    editor.cancel_bootstrap.setEnabled(False)
    editor.cancel_bootstrap.clicked.connect(lambda: editor._bootstrap_cancel.set())
    controls.addWidget(editor.cancel_bootstrap)
    editor.export_button = QPushButton(icon('file-text'), tr("Экспорт результата JSON…"))
    editor.export_button.clicked.connect(editor.export_result)
    editor.export_button.setEnabled(False)
    controls.addWidget(editor.export_button)
    editor.export_plot_button = QPushButton(icon('log-out'), tr('Экспорт графика…'))
    editor.export_plot_button.setEnabled(False)
    controls.addWidget(editor.export_plot_button)
    explanation = QLabel(tr(
        "TLS: ODR с единичными σX/σY; результат зависит от единиц осей. "
        "При выборе σY OLS становится WLS. Пропуски, ND, < / > и исключения QC "
        "для статистики исключаются попарно. Интерполированные колонки не используются. "
        "Формулы: x, параметры, pi, e, + - * / **, exp, log, log10, sqrt, sin, cos, tan, abs, tanh. "
        "Числа и начальные значения вводятся с десятичной точкой."
    ))
    explanation.setWordWrap(True)
    controls.addWidget(explanation)
    controls.addStretch()


def _results_views(editor, content):
    editor.tabs = QTabWidget()
    editor.figure = Figure(layout="constrained")
    editor.canvas = FittedFigureCanvas(editor.figure)
    plot = QWidget()
    editor.plot_page = plot
    layout = QVBoxLayout(plot)
    editor.toolbar = NavigationToolbar(editor.canvas, plot)
    layout.addWidget(editor.toolbar)
    editor.canvas_view = CanvasView(editor.canvas)
    editor.canvas_view.add_to_toolbar(editor.toolbar)
    layout.addWidget(editor.canvas_view.scroll, 1)
    editor.export_plot_button.clicked.connect(editor.toolbar.save_figure)
    editor.tabs.addTab(plot, icon('chart-no-axes-combined'), tr("График"))
    editor.summary = QTextEdit()
    editor.summary.setReadOnly(True)
    editor.tabs.addTab(editor.summary, icon('chart-column'), tr("Коэффициенты и метрики"))
    editor.comparison_table = QTableView()
    editor.comparison_table.clicked.connect(editor.show_oof_diagnostics)
    editor.tabs.addTab(editor.comparison_table, icon('files'), tr("Кросс-валидация"))
    content.addWidget(editor.tabs)
    editor.status = QLabel(tr("Выберите X/Y и нажмите «Рассчитать»"))
    editor.status.setWordWrap(True)
    content.addWidget(editor.status)
    band_note = QLabel(tr("Полосы приближённые. Bootstrap пересэмплирует строки, группы или блоки. Предсказательная полоса точечная и предполагает точный будущий X. На границах покрытие может отличаться от заданного."))
    band_note.setWordWrap(True)
    content.addWidget(band_note)


def _connect_inputs(editor):
    owner = editor.owner
    editor.sync_columns()
    if editor.y_column.count() > 1:
        editor.y_column.setCurrentIndex(1)
    editor.restore_settings(owner.model.project.view.get("curve_fitting", {}))
    for combo in (editor.x_column, editor.y_column, editor.sx_column, editor.sy_column,
                  editor.model_choice, editor.method, editor.loss, editor.band_method, editor.bootstrap_group):
        combo.currentIndexChanged.connect(editor.invalidate)
    for edit in (editor.expression, editor.parameter_names, editor.initial, editor.lower_bounds, editor.upper_bounds, editor.parameter_rules):
        edit.textChanged.connect(editor.invalidate)
    editor.degree.valueChanged.connect(editor.invalidate)
    editor.absolute_sigma.toggled.connect(editor.invalidate)
    editor.f_scale.valueChanged.connect(editor.invalidate)
    editor.starts.valueChanged.connect(editor.invalidate)
    editor.start_seed.valueChanged.connect(editor.invalidate)
    editor.bootstrap_resamples.valueChanged.connect(editor.invalidate)
    editor.bootstrap_seed.valueChanged.connect(editor.invalidate)
    editor.bootstrap_block_length.valueChanged.connect(editor.invalidate)
    editor.prediction_sigma.valueChanged.connect(editor.invalidate)
    editor.band_method.currentIndexChanged.connect(editor.update_controls)
    editor.bootstrap_group.currentIndexChanged.connect(editor.update_controls)
    editor.loss.currentIndexChanged.connect(editor.update_controls)
    editor.model_choice.currentIndexChanged.connect(editor.update_controls)
    editor.model_choice.currentIndexChanged.connect(editor.reset_parameter_inputs)
    editor.method.currentIndexChanged.connect(editor.update_controls)
    editor.expression.textChanged.connect(editor.update_controls)
    editor.parameter_names.textChanged.connect(editor.update_controls)
    editor.parameter_rules.textChanged.connect(editor.update_controls)
    editor.degree.valueChanged.connect(editor.update_controls)
    editor.degree.valueChanged.connect(editor.reset_parameter_inputs)
    owner.model.columnsInserted.connect(editor.sync_columns)
    owner.model.columnsRemoved.connect(editor.sync_columns)
    owner.model.headerDataChanged.connect(editor.sync_columns)
    owner.model.changed.connect(editor.invalidate)
    editor.update_controls()
