from __future__ import annotations

from ...ui.i18n import message, tr
from PySide6.QtCore import Qt

import json

from PySide6.QtWidgets import (
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QListWidget,
    QListWidgetItem,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...ui.tasks import TaskDialog
from ..geochem import inventory, local_parameters, read_profile, read_profiles, snapshot_profile_coverage
from autopyro.analysis.nonpyro import METHODS, REPORTED_BIOMARKER_METHODS
from ...ui.widgets.forms import Form
from ...ui.widgets.tables import readonly_table
from ...ui.settings import database_parameters
from ...ui.widgets.parameters import add_parameter_tools
from ...ui.geochemistry import group_label, status_label
from ...ui.theme import icon


class NonPyroDatabaseDialog(TaskDialog):
    def __init__(self, method_id="sofer", parent=None, *, append=False):
        super().__init__(parent)
        self.append = append
        self.project = None
        self.catalog = None
        self.setWindowTitle(tr('Импорт геохимии из БД'))
        available = (parent.screen() if parent is not None else self.screen()).availableGeometry()
        self.resize(min(1100, available.width() - 40), min(900, available.height() - 60))
        layout = QVBoxLayout(self)
        hint = QLabel(tr('Загрузите исходные измерения в таблицу. Анализ выполняется после импорта в разделе «Обработка → Геохимия».'))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setMinimumHeight(120)
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_scroll.setWidget(settings)
        layout.addWidget(self.settings_scroll, 2)
        self.connection_form = Form()
        self.connection_form.section(tr('Подключение к источнику'))
        parameters = database_parameters(local_parameters())
        for key, label, value in [
            ("host", tr("Сервер"), "127.0.0.1"),
            ("port", tr('Порт'), "15432"),
            ("dbname", tr('База'), "geochemistry"),
            ("user", tr('Пользователь'), "readonly"),
            ("password", tr('Пароль'), ""),
        ]:
            self.connection_form.text(key, label, parameters.get(key, value))
        self.connection_form.fields["password"][0].setEchoMode(
            QLineEdit.EchoMode.Password
        )
        for key in ("host", "port", "dbname", "user"):
            self.connection_form.fields[key][0].textChanged.connect(self.connection_form.fields["password"][0].clear)
        settings_layout.addWidget(self.connection_form)
        self.form = Form()
        self.form.section(tr('Состав и область загрузки'))
        method_ids = tuple(METHODS)
        self.method = self.form.combo(
            "method_id",
            tr('Набор показателей'),
            [(tr(METHODS[key]["name"]), key) for key in method_ids],
        )
        self.group = self.form.combo("group", tr('Аналитическая группа'), [])
        self.form.combo(
            "variant",
            tr('Вариант измерения'),
            [
                (tr('Без метки'), ""),
                (tr('Площадь [a]'), "a"),
                (tr('Высота [h]'), "h"),
                (tr('Калибр. площадь [ca]'), "ca"),
                (tr('Концентрация [ch]'), "ch"),
                (tr('Неопределённый [un]'), "un"),
                (tr('Объём [vol]'), "vol"),
                (tr('Моли [mol]'), "mol"),
            ],
        )
        self.batch = self.form.combo("batch_id", tr('Загрузка'), [(tr('Все загрузки'), None)])
        self.form.text("well_name", tr('Точное имя скважины (пусто — все)'))
        self.form.number(
            "repeat_number", tr('Номер повтора анализа'), 1, 1, 1000, integer=True
        )
        self.form.number(
            "limit", tr('Лимит строк (без усечения)'), 10000, 1, 100000, integer=True
        )
        settings_layout.addWidget(self.form)
        self.include_repeats = self.form.check('include_repeats', tr('Сохранить все повторы анализов'), False)
        self.combine = self.form.check('combine', tr('Загрузить несколько наборов показателей'), False)
        self.profile_list = QListWidget()
        self.profile_list.setMaximumHeight(150)
        for key, definition in METHODS.items():
            item = QListWidgetItem(tr(definition['name']), self.profile_list)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
        self.profile_list.setVisible(False)
        self.combine.toggled.connect(self.profile_list.setVisible)
        self.combine.toggled.connect(self.combined_options_changed)
        self.profile_list.itemChanged.connect(self.invalidate)
        settings_layout.addWidget(self.profile_list)
        settings_layout.addStretch()
        self.method.currentIndexChanged.connect(self.profile_changed)
        self.method.setCurrentIndex(self.method.findData(method_id))
        self.profile_changed()
        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        audit = QPushButton(icon('database'), tr('Каталог показателей (необязательно)'))
        audit.clicked.connect(self.audit)
        buttons.addWidget(audit)
        read = QPushButton(icon('download'), tr('1. Загрузить предпросмотр'))
        read.clicked.connect(self.read)
        buttons.addWidget(read)
        self.use = QPushButton(icon('list-plus' if append else 'table-2'),
                               tr('2. Добавить записи в текущий проект') if append else tr('2. Открыть в основной таблице'))
        self.use.setEnabled(False)
        self.use.clicked.connect(self.accept)
        buttons.addWidget(self.use)
        tabs = self.tabs = QTabWidget()
        self.fields_table = readonly_table(
            [tr('Группа'), tr('Показатель'), tr('Единица'), tr('Тип'), tr('Непусто (все повторы)')]
        )
        self.preview_table = readonly_table(
            [tr('Скважина'), tr('Проба'), tr('Исходная запись'), tr('Анализы / повторы')]
        )
        tabs.addTab(self.fields_table, icon('flask-conical'), tr('Доступные непиролитические показатели'))
        tabs.addTab(self.preview_table, icon('table-2'), tr('Предпросмотр снимка'))
        self.coverage_table = readonly_table([
            tr('Метод'), tr('Основа'), tr('Обязательные колонки'), tr('Заполнено'),
            tr('С результатом после проверок'), tr('Причины отсутствия результата'),
        ])
        for table, flexible in ((self.fields_table, 1), (self.preview_table, 3), (self.coverage_table, 5)):
            header = table.horizontalHeader()
            for section in range(header.count()):
                header.resizeSection(section, max(header.defaultSectionSize(), header.sectionSizeHint(section)))
            header.setSectionResizeMode(flexible, QHeaderView.ResizeMode.Stretch)
        coverage_tab = QWidget()
        coverage_layout = QVBoxLayout(coverage_tab)
        coverage_note = QLabel(tr(
            'Проверяются рабочие значения выбранных повторов и действующие исключения QC снимка. '
            'Изотопные оценки условны до подтверждения VPDB и, для Sofer, нефтяных фракций C15+.'
        ))
        coverage_note.setWordWrap(True)
        coverage_layout.addWidget(coverage_note)
        coverage_layout.addWidget(self.coverage_table)
        tabs.addTab(coverage_tab, icon('flask-conical'), tr('Покрытие методов'))
        layout.addWidget(tabs, 1)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr('Поиск по группе / показателю каталога'))
        self.search.textChanged.connect(self.filter_fields)
        layout.addWidget(self.search)
        self.status = QLabel(
            tr('Исходные данные в БД сохраняются. В таблицу загружается выбранный повтор анализа; пробы с одинаковыми именами остаются отдельными записями. Пароль не сохраняется в проект.')
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.connection_form.edited.connect(self.invalidate)
        self.form.edited.connect(self.invalidate)
        add_parameter_tools(self, changed=self.profile_changed)

    def invalidate(self, *_):
        if self.project is not None:
            self.status.setText(tr('Настройки загрузки изменены. Загрузите предпросмотр заново.'))
        self.project = None
        self.use.setEnabled(False)
        self.preview_table.setRowCount(0)
        self.coverage_table.setRowCount(0)

    def combined_options_changed(self, enabled):
        self.group.setEnabled(not enabled)
        variant = self.form.fields['variant'][0]
        variant.setEnabled(not enabled)
        tip = tr('Комбинированная загрузка включает все доступные основы выбранных наборов.') if enabled else ''
        variant.setToolTip(tip)
        self.combine.setToolTip(tip)
        self.invalidate()

    def profile_changed(self):
        previous = self.group.currentData()
        definition = METHODS[self.method.currentData()]
        self.group.blockSignals(True)
        self.group.clear()
        if definition["same_group"]:
            for group in definition["groups"]:
                self.group.addItem(group_label(group), group)
        else:
            self.group.addItem(
                tr('Совместный профиль: ') + ", ".join(definition["groups"]), ""
            )
        self.group.setCurrentIndex(max(0, self.group.findData(previous)))
        self.group.blockSignals(False)
        key = self.method.currentData()
        variant = (
            "vol"
            if key == "gas_ratios"
            else "h"
            if key in ("stereo_profile", "aromatic_profile", "sterane_distribution",
                       "terpane_hopane_profile", "homohopane_distribution", "homohopane_epimers",
                       "c27_diasterane_ratio", "aromatic_indices_from_peaks", "ster_hop_from_peaks"
                       ) or key in REPORTED_BIOMARKER_METHODS
            else "a"
            if key in ("gc_profile", "alkane_indices")
            else ""
        )
        self.form.load({"variant": variant})

    def parameters(self):
        return self.connection_form.values()

    def audit(self):
        params = self.parameters()
        self.run_task(lambda: inventory(params), self.audited)

    def audited(self, catalog):
        self.invalidate()
        self.catalog = catalog
        self.fields_table.setRowCount(len(catalog["fields"]))
        for row, field in enumerate(catalog["fields"]):
            for col, key in enumerate(
                (
                    "group_name",
                    "source_label",
                    "unit_label",
                    "source_dtype",
                    "populated",
                )
            ):
                self.fields_table.setItem(
                    row, col, QTableWidgetItem(str(field[key] or ""))
                )
        self.batch.blockSignals(True)
        self.batch.clear()
        self.batch.addItem(tr('Все загрузки'), None)
        for batch in catalog["batches"]:
            self.batch.addItem(
                f"{batch['import_batch_id']}: {batch['source_name']}",
                batch["import_batch_id"],
            )
        self.batch.blockSignals(False)
        self.filter_fields(self.search.text())
        self.tabs.setCurrentWidget(self.fields_table)
        self.status.setText(
            tr('Каталог: {0} полей, заполнено {1}; аналитических групп: {2}; transaction_read_only={3}. Наличие данных не означает наличие подтверждённой методики для каждого поля.', len(catalog['fields']), sum(f['populated'] > 0 for f in catalog['fields']), len(catalog['groups']), catalog['read_only'])
        )

    def filter_fields(self, text):
        for row in range(self.fields_table.rowCount()):
            value = " ".join(self.fields_table.item(row, col).text() for col in (0, 1))
            self.fields_table.setRowHidden(row, text.casefold() not in value.casefold())

    def read(self):
        self.invalidate()
        params = self.parameters()
        options = self.form.values()
        method_id = options.pop("method_id")
        combine = options.pop('combine')
        methods = self.selected_methods()
        if combine:
            options.update(group='', variant='')

        def prepare():
            project = (read_profiles(params, methods, **options) if combine else
                       read_profile(params, method_id, **options))
            coverage = snapshot_profile_coverage(project, methods, group=options['group'],
                                                 variant=options['variant'], combined=combine)
            return project, coverage

        self.run_task(prepare, lambda result: self.loaded(*result))

    def selected_methods(self):
        methods = [self.method.currentData()]
        if self.combine.isChecked():
            methods.extend(self.profile_list.item(i).data(Qt.ItemDataRole.UserRole)
                           for i in range(self.profile_list.count())
                           if self.profile_list.item(i).checkState() == Qt.CheckState.Checked)
        return list(dict.fromkeys(methods))

    def loaded(self, project, coverage=None):
        if not project.rows:
            self.invalidate()
            self.status.setText(tr('По выбранным условиям записи не найдены. Измените фильтры и загрузите предпросмотр заново.'))
            return
        if coverage is None:
            combined = self.combine.isChecked()
            coverage = snapshot_profile_coverage(project, self.selected_methods(), combined=combined,
                group='' if combined else self.group.currentData(),
                variant='' if combined else self.form.fields['variant'][0].currentData())
        self.project = project
        preview = project.rows[:100]
        self.preview_table.setRowCount(len(preview))
        well = next(c.id for c in project.columns if c.name == "Well / Well Name")
        name = next(c.id for c in project.columns if c.name == "Sample / Name")
        runs = {}
        if project.source.get('include_repeats'):
            for analysis in project.science.analyses.values():
                if analysis.metadata.get('analysis_run_id') is not None:
                    runs.setdefault(analysis.sample_id, []).append(analysis.metadata)
        for row, item in enumerate(preview):
            values = [
                item.values[well],
                item.values[name],
                item.source["source_record_id"],
                json.dumps(runs.get(item.id, item.source["analysis_runs"]), ensure_ascii=False),
            ]
            for col, value in enumerate(values):
                self.preview_table.setItem(
                    row, col, QTableWidgetItem(str(value) if value is not None else "")
                )
        self.use.setEnabled(True)
        self.coverage_table.setRowCount(len(coverage))
        for row, (method, basis, bindings, complete, effective, error) in enumerate(coverage):
            if bindings is None:
                columns = tr('Привязки неоднозначны или отсутствуют')
            elif bindings[1]:
                columns = f'{bindings[0]} / {bindings[1]}'
            else:
                columns = tr('Альтернативные входы')
            counts = {}
            if effective is not None:
                for failure in effective['failures']:
                    status = failure['status']
                    counts[status] = counts.get(status, 0) + failure['count']
            reasons = (', '.join(f'{status_label(status)}: {count}' for status, count in counts.items())
                       if effective is not None else message(error) if error else tr('Недостаточно входных данных'))
            ready = f"{effective['with_values']} / {effective['total']}" if effective is not None else '—'
            values = (tr(METHODS[method]['name']), basis or '—', columns, str(complete), ready, reasons or '—')
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.coverage_table.setItem(row, col, item)
        self.tabs.setCurrentWidget(self.preview_table)
        summary = tr('Подготовлено {0} отдельных записей, {1} колонок; предпросмотр первых {2}. Исходные записи, единицы, номера анализов и повторы сохранены.',
                     len(project.rows), len(project.columns), len(preview))
        action = (tr('Записи будут добавлены отдельными строками в текущий проект.') if self.append
                  else tr('Открытие заменит текущий локальный проект с проверкой несохранённых изменений.'))
        working_repeat = project.source.get('repeat_number', self.form.values().get('repeat_number', 1))
        ready_count = sum(bool(effective and effective['with_values']) for _, _, _, _, effective, _ in coverage)
        coverage_summary = tr('Рабочий повтор: {0}. Методик и основ с пригодными строками: {1} из {2}.',
                              working_repeat, ready_count, len(coverage))
        if ready_count:
            best = max((row for row in coverage if row[4] and row[4]['with_values']),
                       key=lambda row: row[4]['with_values'])
            method, basis, _, _, effective, _ = best
            coverage_summary += ' ' + tr('Больше всего пригодных строк: {0} из {1} ({2}, {3}).',
                                         effective['with_values'], effective['total'], tr(METHODS[method]['name']), basis or '—')
        elif coverage:
            coverage_summary += ' ' + tr('Проверьте вкладку «Покрытие методов» перед импортом.')
        self.status.setText(f'{summary} {action}\n{coverage_summary}')
