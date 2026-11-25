"""Build the sample, analysis and archive views of the geochemistry workspace."""

from PySide6.QtCore import Qt, QStringListModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from matplotlib.figure import Figure
from ..i18n import tr
from ..widgets.documents import analysis_document, parameter_page, wrapped_check
from ..widgets.figure_canvas import FittedFigureCanvas
from ..widgets.tables import readonly_table
from .geochemistry_workspace_display import _biodegradation_labels, _biodegradation_indicators, _observation_labels


class ComparisonChoicesModel(QStringListModel):
    def __init__(self, parent):
        super().__init__(parent)
        self.identifiers = []

    def set_choices(self, labels, identifiers):
        self.identifiers = identifiers
        self.setStringList(labels)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.UserRole:
            return self.identifiers[index.row()]
        return super().data(index, role)


def build(workspace):
    controls, content = analysis_document(workspace)
    _sample_controls(workspace, controls)
    _comparison_controls(workspace)
    _passport(workspace)
    _profiles(workspace)
    _model_results(workspace)
    _archives(workspace)
    _document_actions(workspace, content)


def _sample_controls(workspace, controls):
    workspace.scope = QComboBox()
    help_text = QLabel(tr('Список проб определяется общей выборкой. Паспорт включает все исходные записи связанной пробы.'))
    help_text.setWordWrap(True)
    controls.addWidget(help_text)
    controls.addWidget(workspace.scope)
    workspace.refresh_button = QPushButton(tr('Обновить список проб'))
    workspace.refresh_button.setAutoDefault(False)
    workspace.refresh_button.clicked.connect(lambda: workspace.refresh())
    controls.addWidget(workspace.refresh_button)
    workspace.repair_links_button = QPushButton(tr('Исправить объединение проб…'))
    workspace.repair_links_button.clicked.connect(workspace.repair_invalid_link)
    workspace.repair_links_button.hide()
    controls.addWidget(workspace.repair_links_button)
    workspace.search = QLineEdit()
    workspace.search.setPlaceholderText(tr('Поиск по пробе, скважине или ID'))
    workspace.search.textChanged.connect(workspace.filter_samples)
    controls.addWidget(workspace.search)
    workspace.samples = QListWidget()
    workspace.samples.setUniformItemSizes(True)
    workspace.samples.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
    workspace.samples.itemSelectionChanged.connect(workspace.show_passport)
    controls.addWidget(workspace.samples, 1)
    workspace.count = QLabel()
    controls.addWidget(workspace.count)
    for text, callback in (
        (tr('Объединить выбранные записи…'), workspace.link_selected),
        (tr('Отменить ручное объединение'), workspace.unlink_selected),
        (tr('Связать исходный и производный материал…'), workspace.link_materials),
        (tr('Удалить связи материала'), workspace.remove_material_links),
        (tr('Сохранить выбранные пробы как группу…'), workspace.save_selected_group),
        (tr('Сбросить ручные связи и выбор измерений'), workspace.reset_links),
    ):
        button = QPushButton(text)
        button.setAutoDefault(False)
        button.clicked.connect(callback)
        controls.addWidget(button)


def _comparison_controls(workspace):
    comparison = parameter_page(workspace.parameter_tabs, tr('Сравнение'), 'files')
    workflow = QLabel(tr('Выберите A и B, отметьте показатели и проверьте пару перед корреляцией.'))
    workflow.setWordWrap(True)
    comparison.addWidget(workflow)
    workspace.a, workspace.b = QComboBox(), QComboBox()
    workspace.comparison_choices = ComparisonChoicesModel(workspace)
    for combo in (workspace.a, workspace.b):
        combo.setModel(workspace.comparison_choices)
        combo.view().setUniformItemSizes(True)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(12)
        combo.currentTextChanged.connect(combo.setToolTip)
    form = QFormLayout()
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    form.addRow(tr('Эталонная проба / группа A'), workspace.a)
    form.addRow(tr('Сравниваемая проба / группа B'), workspace.b)
    comparison.addLayout(form)
    workspace.mixed = QCheckBox(tr('Подтверждаю сравнение разных материалов'))
    wrapped_check(comparison, workspace.mixed)
    workspace.fields = QListWidget()
    workspace.fields.setUniformItemSizes(True)
    workspace.fields.setMaximumHeight(150)
    comparison.addWidget(QLabel(tr('Исходные числовые показатели')))
    field_tools = QHBoxLayout()
    workspace.field_search = QLineEdit()
    workspace.field_search.setPlaceholderText(tr('Поиск показателя'))
    workspace.field_search.textChanged.connect(workspace.filter_comparison_fields)
    comparison.addWidget(workspace.field_search)
    workspace.field_preset = QComboBox()
    for label, method in (
        (tr('Н-алканы nC24–nC34'), 'alkane_indices'),
        (tr('Стераны C27–C29'), 'sterane_distribution'),
        (tr('Гомогопаны C31–C35'), 'homohopane_distribution'),
        (tr('Ster/Hop · 17/16 пиков'), 'ster_hop_from_peaks'),
    ):
        workspace.field_preset.addItem(label, method)
    comparison.addWidget(workspace.field_preset)
    workspace.field_basis = QComboBox()
    for label, basis in ((tr('Высота [h]'), 'h'), (tr('Площадь [a]'), 'a'),
                         (tr('Концентрация [ch]'), 'ch'), (tr('Калибр. площадь [ca]'), 'ca')):
        workspace.field_basis.addItem(label, basis)
    workspace.field_preset.currentIndexChanged.connect(lambda *_: workspace.field_basis.setCurrentIndex(
        workspace.field_basis.findData('a' if workspace.field_preset.currentData() == 'alkane_indices' else 'h')))
    workspace.field_basis.setCurrentIndex(workspace.field_basis.findData('a'))
    preset_button = QPushButton(tr('Добавить набор'))
    preset_button.clicked.connect(workspace.select_field_preset)
    for widget in (workspace.field_basis, preset_button):
        field_tools.addWidget(widget)
    comparison.addLayout(field_tools)
    comparison.addWidget(workspace.fields, 1)
    comparison_actions = QHBoxLayout()
    workspace.compare_button = QPushButton(tr('Сравнить'))
    workspace.compare_button.clicked.connect(workspace.calculate_comparison)
    comparison_actions.addWidget(workspace.compare_button)
    workspace.correlation_preview_button = QPushButton(tr('Проверить пару…'))
    workspace.correlation_preview_button.setToolTip(tr('Проверьте пригодность измерений A и B до расчёта корреляции.'))
    workspace.correlation_preview_button.clicked.connect(workspace.preview_oil_correlation)
    comparison_actions.addWidget(workspace.correlation_preview_button)
    comparison.addLayout(comparison_actions)
    workspace.profile_match_button = QPushButton(tr('Сопоставить профили н-алканов'))
    workspace.profile_match_button.setEnabled(False)
    workspace.profile_match_button.setToolTip(tr('Сравните две пробы по полному набору пиков nC24–nC34.'))
    workspace.profile_match_button.clicked.connect(workspace.show_profile_match)
    comparison.addWidget(workspace.profile_match_button)
    workspace.correlation_isotope_basis = QCheckBox(tr('Подтверждаю для обеих проб δ¹³C фракций C15+ по шкале VPDB'))
    workspace.correlation_isotope_basis.setEnabled(False)
    workspace.correlation_isotope_basis.toggled.connect(workspace.invalidate_oil_correlation)
    workspace.correlation_isotope_basis.toggled.connect(workspace.update_correlation_readiness)
    wrapped_check(comparison, workspace.correlation_isotope_basis)
    workspace.oil_correlation_button = QPushButton(tr('Корреляция нефть–нефть / нефть–порода'))
    workspace.oil_correlation_button.setEnabled(False)
    workspace.oil_correlation_button.setToolTip(tr(
        'Выберите две пробы и сопоставимые показатели. '
        'Для перекрытия добавьте полный набор Sat-GC nC24–nC34. '
        'Для Ster/Hop выберите все 17 стерановых и 16 гопановых пиков одной основы.'
    ))
    workspace.oil_correlation_button.clicked.connect(lambda: workspace.show_profile_match(oil_correlation=True))
    comparison.addWidget(workspace.oil_correlation_button)
    workspace.correlation_readiness = QLabel()
    workspace.correlation_readiness.setWordWrap(True)
    comparison.addWidget(workspace.correlation_readiness)
    models = parameter_page(workspace.parameter_tabs, tr('Модели и архивы'), 'logs')
    workspace.clr = QCheckBox(tr('Части одного состава (CLR)'))
    workspace.clr.setEnabled(False)
    workspace.clr.setToolTip(tr('Подтвердите, что показатели являются частями одного состава и имеют общую основу измерения. CLR исключает пробы с неположительными долями.'))
    workspace.clr.toggled.connect(workspace.change_preprocessing)
    models.addWidget(workspace.clr)
    workspace.pca_button = QPushButton(tr('Анализ главных компонент (PCA)'))
    workspace.pca_button.setEnabled(False)
    workspace.pca_button.clicked.connect(workspace.calculate_pca)
    models.addWidget(workspace.pca_button)
    workspace.clustering_button = QPushButton(tr('Иерархическая кластеризация…'))
    workspace.clustering_button.setEnabled(False)
    workspace.clustering_button.clicked.connect(workspace.calculate_clustering)
    models.addWidget(workspace.clustering_button)
    workspace.kmeans_button = QPushButton(tr('Кластеризация K-средних…'))
    workspace.kmeans_button.setEnabled(False)
    workspace.kmeans_button.clicked.connect(lambda: workspace.calculate_clustering(method='kmeans'))
    models.addWidget(workspace.kmeans_button)
    workspace.dbscan_button = QPushButton(tr('Кластеризация DBSCAN…'))
    workspace.dbscan_button.setEnabled(False)
    workspace.dbscan_button.clicked.connect(lambda: workspace.calculate_clustering(method='dbscan'))
    models.addWidget(workspace.dbscan_button)
    workspace.prediction_button = QPushButton(tr('Линейный прогноз A → B…'))
    workspace.prediction_button.setEnabled(False)
    workspace.prediction_button.setToolTip(tr('Группа A служит для обучения, группа B — для проверки на других скважинах.'))
    workspace.prediction_button.clicked.connect(workspace.calculate_prediction)
    models.addWidget(workspace.prediction_button)
    workspace.oil_mixing_button = QPushButton(tr('Бинарное смешение нефтей…'))
    workspace.oil_mixing_button.setEnabled(False)
    workspace.oil_mixing_button.setToolTip(tr('Выберите два конечных компонента в группе A и одну смесь в группе B.'))
    workspace.oil_mixing_button.clicked.connect(workspace.calculate_oil_mixing)
    models.addWidget(workspace.oil_mixing_button)
    workspace.prediction_cv = QCheckBox(tr('Перекрёстная проверка по скважинам внутри A'))
    workspace.prediction_cv.setEnabled(False)
    wrapped_check(models, workspace.prediction_cv)
    workspace.prediction_cv.setToolTip(tr('По очереди исключает из обучения каждую скважину группы A. Группа B остаётся отдельной проверкой.'))
    archive_button = QPushButton(tr('Архив запусков моделей'))
    archive_button.clicked.connect(workspace.show_model_archive)
    models.addWidget(archive_button)
    review_archive_button = QPushButton(tr('Архив экспертных заключений'))
    review_archive_button.clicked.connect(workspace.show_review_archive)
    models.addWidget(review_archive_button)
    models.addStretch()
    delete_group = QPushButton(tr('Удалить сохранённую группу…'))
    delete_group.clicked.connect(workspace.delete_group)
    comparison.addWidget(delete_group)
    for combo in (workspace.a, workspace.b):
        combo.currentIndexChanged.connect(workspace.invalidate_comparison)
    workspace.mixed.toggled.connect(workspace.invalidate_comparison)
    workspace.fields.itemChanged.connect(workspace.invalidate_comparison)


def _passport(workspace):
    workspace.details = QTextBrowser()
    workspace.details.setOpenExternalLinks(False)
    workspace.measurements = readonly_table([
        tr('Показатель'), tr('Исходное значение'), tr('Единица'), tr('Анализ'), tr('Повтор'),
        tr('Рабочее в строке'), tr('Выбор паспорта'), tr('QC / допуск к сравнению'), tr('Исходная запись'),
    ])
    workspace.measurements.setSelectionBehavior(workspace.measurements.SelectionBehavior.SelectRows)
    workspace.measurements.setSelectionMode(workspace.measurements.SelectionMode.SingleSelection)
    workspace.measurements.itemSelectionChanged.connect(workspace.update_measurement_actions)
    workspace.methods = readonly_table([tr('Методика'), tr('Исходные данные'), tr('Что требуется')])
    workspace.methods.setSelectionBehavior(workspace.methods.SelectionBehavior.SelectRows)
    workspace.methods.setSelectionMode(workspace.methods.SelectionMode.SingleSelection)
    workspace.passport_tabs = QTabWidget()
    workspace.passport_tabs.addTab(workspace.details, tr('Паспорт пробы'))
    workspace.passport_tabs.addTab(workspace.measurements, tr('Анализы и повторы'))
    workspace.passport_tabs.addTab(workspace.methods, tr('Доступность методик'))
    workspace.comparison_table = readonly_table([
        tr('Показатель'), tr('Единица'), tr('N A / всего'), tr('Среднее A'), tr('Медиана A'), tr('SD A'),
        tr('N B / всего'), tr('Среднее B'), tr('Медиана B'), tr('SD B'), tr('Разность B − A'), tr('Исключения'),
    ])
    workspace.comparison_table.setSelectionBehavior(workspace.comparison_table.SelectionBehavior.SelectRows)
    workspace.comparison_table.itemSelectionChanged.connect(workspace.draw_comparison)
    workspace.comparison_table.cellDoubleClicked.connect(workspace.open_comparison_measurement)
    workspace.figure = Figure(figsize=(8, 3), layout='constrained')
    workspace.canvas = FittedFigureCanvas(workspace.figure)
    workspace.passport_tabs.addTab(workspace.comparison_table, tr('Сравнение проб и групп'))
    workspace.passport_tabs.addTab(workspace.canvas, tr('График сравнения'))


def _profiles(workspace):
    workspace.profile_page = QWidget()
    profile_layout = QVBoxLayout(workspace.profile_page)
    workspace.profile_canvas = FittedFigureCanvas(Figure(layout='constrained'))
    workspace.profile_views = QTabWidget()
    workspace.profile_views.addTab(workspace.profile_canvas, tr('Профили н-алканов'))
    workspace.profile_evidence = readonly_table([
        tr('Показатель'), tr('Единица'), 'A', 'B', tr('Разность B − A'),
        tr('Наблюдение биодеградации'), tr('Влияние на интерпретацию'),
    ])
    workspace.profile_evidence_notes = {}
    workspace.profile_views.addTab(workspace.profile_evidence, tr('Дополнительные показатели'))
    workspace.biodegradation_table = readonly_table([
        tr('Проба'), tr('Признак'), tr('Наблюдение'), tr('Источник анализа'), tr('Измеренный показатель'),
    ])
    workspace.profile_views.addTab(workspace.biodegradation_table, tr('Наблюдения биодеградации'))
    workspace.biodegradation_table.setToolTip(tr('Предел обнаружения и отсутствие хроматограммы означают недостаток данных.'))
    workspace.biodegradation_observations = {}
    indicators = _biodegradation_indicators()
    workspace.biodegradation_table.setRowCount(2 * len(indicators))
    for row, (side, indicator) in enumerate((side, key) for side in ('a', 'b') for key in indicators):
        workspace.biodegradation_table.setItem(row, 0, QTableWidgetItem(side.upper()))
        workspace.biodegradation_table.setItem(row, 1, QTableWidgetItem(indicators[indicator]))
        status = QComboBox()
        for value, label in _observation_labels().items():
            status.addItem(label, value)
        source = QLineEdit()
        source.setPlaceholderText(tr('Номер анализа, хроматограмма или отчёт'))
        workspace.biodegradation_table.setCellWidget(row, 2, status)
        workspace.biodegradation_table.setCellWidget(row, 3, source)
        workspace.biodegradation_observations[side, indicator] = status, source
    workspace.biodegradation_table.resizeColumnsToContents()
    workspace.biodegradation_table.horizontalHeader().setStretchLastSection(True)
    profile_layout.addWidget(workspace.profile_views, 3)
    workspace.sensitivity_button = QPushButton(tr('Чувствительность к составу показателей…'))
    workspace.sensitivity_button.hide()
    workspace.sensitivity_button.clicked.connect(workspace.show_correlation_sensitivity)
    profile_layout.addWidget(workspace.sensitivity_button)
    profile_layout.addWidget(QLabel(tr('Экспертное заключение по этой паре')))
    assessment = QFormLayout()
    workspace.biodegradation = {}
    for side, title in (('a', tr('Признаки биодеградации в A')), ('b', tr('Признаки биодеградации в B'))):
        combo = QComboBox()
        for status, label in _biodegradation_labels().items():
            combo.addItem(label, status)
        combo.setToolTip(tr('Укажите экспертную оценку и обоснуйте её в заключении.'))
        workspace.biodegradation[side] = combo
        assessment.addRow(title, combo)
    profile_layout.addLayout(assessment)
    workspace.profile_note = QPlainTextEdit()
    workspace.profile_note.setPlaceholderText(tr('Укажите, какие данные подтверждают или ограничивают интерпретацию.'))
    profile_layout.addWidget(workspace.profile_note, 1)
    save_review = QPushButton(tr('Сохранить заключение'))
    save_review.clicked.connect(workspace.save_profile_review)
    profile_layout.addWidget(save_review)
    workspace.profile_tab = workspace.passport_tabs.addTab(workspace.profile_page, tr('Сопоставление профилей'))
    workspace.passport_tabs.setTabVisible(workspace.profile_tab, False)


def _model_results(workspace):
    workspace.pca_page = QWidget()
    pca_layout = QVBoxLayout(workspace.pca_page)
    workspace.pca_canvas = FittedFigureCanvas(Figure(layout='constrained'))
    pca_layout.addWidget(workspace.pca_canvas, 3)
    workspace.pca_coefficients = readonly_table([tr('Показатель'), 'PC1', 'PC2'])
    pca_layout.addWidget(workspace.pca_coefficients, 1)
    workspace.pca_tab = workspace.passport_tabs.addTab(workspace.pca_page, 'PCA')
    workspace.passport_tabs.setTabVisible(workspace.pca_tab, False)
    workspace.clustering_page = QWidget()
    clustering_layout = QVBoxLayout(workspace.clustering_page)
    workspace.clustering_canvas = FittedFigureCanvas(Figure(layout='constrained'))
    clustering_layout.addWidget(workspace.clustering_canvas, 3)
    workspace.cluster_members = readonly_table([tr('№'), tr('Проба'), tr('Группа'), tr('Кластер'), tr('Роль DBSCAN')])
    clustering_layout.addWidget(workspace.cluster_members, 2)
    workspace.clustering_tab = workspace.passport_tabs.addTab(workspace.clustering_page, tr('Кластеры проб'))
    workspace.passport_tabs.setTabVisible(workspace.clustering_tab, False)
    workspace.prediction_page = QWidget()
    prediction_layout = QVBoxLayout(workspace.prediction_page)
    workspace.prediction_canvas = FittedFigureCanvas(Figure(layout='constrained'))
    prediction_layout.addWidget(workspace.prediction_canvas, 3)
    workspace.prediction_table = readonly_table([tr('Проба'), tr('Измерено'), tr('Прогноз'), tr('Вне диапазона обучения')])
    workspace.prediction_details = QTabWidget()
    workspace.prediction_details.addTab(workspace.prediction_table, tr('Проверочная группа B'))
    workspace.prediction_cv_table = readonly_table([tr('Скважина'), tr('Проб'), 'RMSE', 'MAE', 'R²',
                                               tr('RMSE базового прогноза')])
    workspace.prediction_details.addTab(workspace.prediction_cv_table, tr('Ошибки по скважинам A'))
    workspace.prediction_well_table = readonly_table([tr('Скважина'), tr('Проб'), 'RMSE', 'MAE',
                                                 tr('RMSE базового прогноза'), tr('MAE базового прогноза')])
    workspace.prediction_details.addTab(workspace.prediction_well_table, tr('Ошибки по скважинам B'))
    prediction_layout.addWidget(workspace.prediction_details, 2)
    workspace.prediction_tab = workspace.passport_tabs.addTab(workspace.prediction_page, tr('Проверка прогноза'))
    workspace.passport_tabs.setTabVisible(workspace.prediction_tab, False)
    workspace.oil_mixing_table = readonly_table([
        tr('Маркер'), tr('Единица'), tr('Компонент A'), tr('Компонент B'), tr('Смесь'),
        tr('Допуск'), tr('Расчёт'), tr('Остаток'), tr('Участвует в подгонке'), tr('В пределах допуска'),
    ])
    workspace.oil_mixing_tab = workspace.passport_tabs.addTab(workspace.oil_mixing_table, tr('Бинарное смешение нефтей'))
    workspace.passport_tabs.setTabVisible(workspace.oil_mixing_tab, False)
    workspace.application_page = QWidget()
    application_layout = QVBoxLayout(workspace.application_page)
    workspace.application_table = readonly_table([
        tr('Проба'), tr('Прогноз'), tr('Единица'), tr('Вне диапазона обучения'), tr('Причина исключения'),
    ])
    application_layout.addWidget(workspace.application_table)
    application_export = QPushButton(tr('Экспорт прогнозов JSON…'))
    application_export.clicked.connect(workspace.export_prediction_application)
    application_layout.addWidget(application_export)
    workspace.application_tab = workspace.passport_tabs.addTab(workspace.application_page, tr('Прогноз для выбранных проб'))
    workspace.passport_tabs.setTabVisible(workspace.application_tab, False)


def _archives(workspace):
    workspace.model_archive_page = QWidget()
    archive_layout = QVBoxLayout(workspace.model_archive_page)
    workspace.model_runs = readonly_table([tr('Время (UTC)'), tr('Метод'), tr('Подготовка данных'), tr('Проб')])
    workspace.model_runs.setSelectionBehavior(workspace.model_runs.SelectionBehavior.SelectRows)
    workspace.model_runs.setSelectionMode(workspace.model_runs.SelectionMode.SingleSelection)
    workspace.model_runs.doubleClicked.connect(workspace.open_model_run)
    workspace.model_runs.itemSelectionChanged.connect(workspace.display_model_passport)
    archive_layout.addWidget(workspace.model_runs, 1)
    archive_layout.addWidget(QLabel(tr('Паспорт метода и запуска')))
    workspace.model_passport = QPlainTextEdit()
    workspace.model_passport.setReadOnly(True)
    archive_layout.addWidget(workspace.model_passport, 2)
    archive_actions = QGridLayout()
    for index, (label, action) in enumerate((
        (tr('Открыть запуск'), workspace.open_model_run),
        (tr('Проверить воспроизводимость'), workspace.check_model_run),
        (tr('Экспорт запуска JSON…'), workspace.export_model_run),
        (tr('Импорт запуска JSON…'), workspace.import_model_run),
    )):
        button = QPushButton(label)
        button.clicked.connect(action)
        archive_actions.addWidget(button, index // 2, index % 2)
    archive_layout.addLayout(archive_actions)
    apply_button = QPushButton(tr('Применить модель к выбранным пробам'))
    apply_button.setToolTip(tr('Выделите пробы в списке проб и выберите модель линейного прогноза в архиве.'))
    apply_button.clicked.connect(workspace.apply_archived_prediction)
    archive_layout.addWidget(apply_button)
    workspace.model_archive_tab = workspace.passport_tabs.addTab(workspace.model_archive_page, tr('Архив запусков'))
    workspace.passport_tabs.setTabVisible(workspace.model_archive_tab, False)
    workspace.review_archive_page = QWidget()
    review_layout = QVBoxLayout(workspace.review_archive_page)
    workspace.review_archive = readonly_table([tr('Время (UTC)'), 'A', 'B'])
    workspace.review_archive.setSelectionBehavior(workspace.review_archive.SelectionBehavior.SelectRows)
    workspace.review_archive.setSelectionMode(workspace.review_archive.SelectionMode.SingleSelection)
    workspace.review_archive.itemSelectionChanged.connect(workspace.display_archived_review)
    review_layout.addWidget(workspace.review_archive, 1)
    workspace.archived_review_text = QPlainTextEdit()
    workspace.archived_review_text.setReadOnly(True)
    review_layout.addWidget(workspace.archived_review_text, 2)
    workspace.export_review_button = QPushButton(tr('Экспорт заключения JSON…'))
    workspace.export_review_button.setEnabled(False)
    workspace.export_review_button.clicked.connect(workspace.export_archived_review)
    review_layout.addWidget(workspace.export_review_button)
    workspace.review_archive_tab = workspace.passport_tabs.addTab(workspace.review_archive_page, tr('Архив заключений'))
    workspace.passport_tabs.setTabVisible(workspace.review_archive_tab, False)


def _document_actions(workspace, content):
    content.addWidget(workspace.passport_tabs, 1)
    passport_actions = QGridLayout()
    workspace.use_measurement = QPushButton(tr('Выбрать измерение для паспорта'))
    workspace.use_measurement.clicked.connect(workspace.choose_measurement)
    passport_actions.addWidget(workspace.use_measurement, 0, 0)
    workspace.reset_button = QPushButton(tr('Сбросить выбор показателя'))
    workspace.reset_button.clicked.connect(workspace.reset_measurement)
    passport_actions.addWidget(workspace.reset_button, 0, 1)
    workspace.method_button = QPushButton(tr('Открыть выбранную методику…'))
    workspace.method_button.clicked.connect(workspace.open_method)
    workspace.method_button.setEnabled(False)
    workspace.methods.itemSelectionChanged.connect(lambda: workspace.method_button.setEnabled(bool(workspace.methods.selectedIndexes())))
    passport_actions.addWidget(workspace.method_button, 1, 0, 1, 2)
    content.addLayout(passport_actions)
    exports = QHBoxLayout()
    workspace.export_passport_button = QPushButton(tr('Экспорт паспорта JSON…'))
    workspace.export_passport_button.clicked.connect(workspace.export_passport)
    exports.addWidget(workspace.export_passport_button)
    workspace.export_comparison_button = QPushButton(tr('Экспорт таблицы групп…'))
    workspace.export_comparison_button.clicked.connect(workspace.export_comparison)
    exports.addWidget(workspace.export_comparison_button)
    workspace.export_correlation_button = QPushButton(tr('Экспорт корреляции…'))
    workspace.export_correlation_button.setEnabled(False)
    workspace.export_correlation_button.clicked.connect(workspace.export_correlation)
    exports.addWidget(workspace.export_correlation_button)
    content.addLayout(exports)
    workspace.status = QLabel(tr('Выберите пробу для просмотра паспорта. Для объединения выберите несколько проб.'))
    workspace.status.setWordWrap(True)
    content.addWidget(workspace.status)
