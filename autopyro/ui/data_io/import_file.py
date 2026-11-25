from __future__ import annotations

from ..i18n import tr, message

from copy import deepcopy
import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
)

from autopyro.data_io.io import (
    build_project,
    file_source,
    infer_kind,
    read_file,
    excel_sheet_names,
    ImportValidationError,
)
from autopyro.model.field_types import FIELD_KINDS
from ..field_types import field_kind_label
from ..tasks import TaskDialog, run_with_progress
from ..table.model import prepare_project
from ..settings import load_preferences
from ..widgets.parameters import add_parameter_tools
from ..theme import icon


class ImportDialog(TaskDialog):
    def __init__(self, path=None, snapshot=None, parent=None, *, template_view=None):
        super().__init__(parent)
        self.setWindowTitle(tr('Импорт · предпросмотр и типы колонок'))
        self.resize(970, 640)
        self.path, self.snapshot = path, snapshot
        self.raw = None
        self.preview_source = None
        self.project = None
        self.project_prepared = False
        self.pending_template = self.active_template = None
        self.import_errors = []
        layout = QVBoxLayout(self)
        self.source_label = QLabel(str(path or snapshot[3]))
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        from .templates import TemplateBar
        self.templates = TemplateBar("reader", template_view or {}, self)
        self.templates.applyRequested.connect(self.apply_template)
        self.templates.saveRequested.connect(self.save_template)
        self.templates.setEnabled(not bool(snapshot))
        layout.addWidget(self.templates)
        form = QGridLayout()
        self.separator = QComboBox()
        self.separator.addItems([";", ",", "TAB", "|"])
        self.encoding = QComboBox()
        self.encoding.addItems(["utf-8-sig", "utf-8", "cp1251", "utf-16"])
        preferences = load_preferences()
        self.separator.setCurrentText("TAB" if preferences.csv_delimiter == "\t" else preferences.csv_delimiter)
        self.encoding.setCurrentText(preferences.csv_encoding)
        self.sheet = QComboBox()
        self.sheet.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sheet.setMinimumContentsLength(10)
        if path and Path(path).suffix.lower() == ".xlsx":
            self.sheet.addItems(excel_sheet_names(path))
        self.headers = QSpinBox()
        self.headers.setRange(1, 2)
        for column, (label, widget) in enumerate((
            (tr('Разделитель'), self.separator),
            (tr('Кодировка'), self.encoding),
            (tr('Лист'), self.sheet),
            (tr('Строк заголовка'), self.headers),
        )):
            caption = QLabel(label)
            caption.setBuddy(widget)
            form.addWidget(caption, 0, column)
            form.addWidget(widget, 1, column)
        layout.addLayout(form)
        ranges = QGridLayout()
        self.header_row, self.data_end, self.first_column, self.last_column = QSpinBox(), QSpinBox(), QSpinBox(), QSpinBox()
        for widget in (self.header_row, self.first_column):
            widget.setRange(1, 2_000_000_000)
        for widget in (self.data_end, self.last_column):
            widget.setRange(0, 2_000_000_000)
            widget.setSpecialValueText(tr('До конца'))
        self.decimal = QComboBox()
        for label, value in ((tr('Точка или запятая'), "auto"), (tr('Точка'), "."), (tr('Запятая'), ",")):
            self.decimal.addItem(label, value)
        self.decimal.setCurrentIndex(self.decimal.findData(preferences.decimal_separator))
        xlsx = bool(path and Path(path).suffix.lower() == ".xlsx")
        for position, (label, widget) in enumerate((
            (tr('Строка заголовка'), self.header_row), (tr('Последняя строка'), self.data_end),
            (tr('Первая колонка №'), self.first_column), (tr('Последняя колонка №'), self.last_column),
            (tr('Десятичный знак'), self.decimal),
        )):
            caption = QLabel(label)
            caption.setBuddy(widget)
            row, column = (0, position) if position < 2 else (1, position - 2)
            ranges.addWidget(caption, row * 2, column)
            ranges.addWidget(widget, row * 2 + 1, column)
        layout.addLayout(ranges)
        load = QPushButton(icon('refresh-cw'), tr('Прочитать / обновить предпросмотр'))
        load.clicked.connect(self.load)
        layout.addWidget(load, alignment=Qt.AlignmentFlag.AlignLeft)
        self.decimal.setEnabled(not xlsx and not bool(snapshot))
        for widget in (self.separator, self.encoding):
            widget.setEnabled(not xlsx and not bool(snapshot))
        self.tabs = QTabWidget()
        self.mapping = QTableWidget(0, 6)
        self.mapping.setHorizontalHeaderLabels(
            [tr('Импорт'), tr('Исходная колонка'), tr('Название'), tr('Тип'), tr('Единица'), tr('Синонимы (через ;)')]
        )
        self.mapping.horizontalHeader().setStretchLastSection(True)
        self.preview = QTableWidget()
        self.preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabs.addTab(self.mapping, icon('columns-3'), tr('Колонки и типы'))
        self.tabs.addTab(self.preview, icon('table-2'), tr('Первые 100 строк'))
        self.errors = QTextEdit()
        self.errors.setReadOnly(True)
        self.tabs.addTab(self.errors, icon('circle-question-mark'), tr('Ошибки импорта'))
        layout.addWidget(self.tabs)
        self.summary = QLabel(
            tr('Тип можно изменить. В числовых колонках ND и <LOD сохраняются с квалификатором; единицы не пересчитываются.')
        )
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.summary)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('Импортировать'))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('Отмена'))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.buttons.accepted.connect(self.import_data)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        # Changing parse settings invalidates the preview until explicitly reloaded.
        for widget in (self.separator, self.encoding, self.sheet):
            widget.currentIndexChanged.connect(self.invalidate)
        self.headers.valueChanged.connect(self.invalidate)
        for widget in (self.header_row, self.data_end, self.first_column, self.last_column):
            widget.valueChanged.connect(self.invalidate)
        self.decimal.currentIndexChanged.connect(self.invalidate)
        self.mapping.itemChanged.connect(self.clear_template)
        if snapshot:
            load.setEnabled(False)
            for widget in (self.header_row, self.data_end, self.first_column, self.last_column):
                widget.setEnabled(False)
            for i in range(form.count()):
                widget = form.itemAt(i).widget()
                if widget:
                    widget.setEnabled(False)
            self.loaded(snapshot[:3])
        add_parameter_tools(self, reset=self.restore_parameters)

    def restore_parameters(self):
        self.parameter_defaults.restore()
        preferences = load_preferences()
        self.separator.setCurrentText("TAB" if preferences.csv_delimiter == "\t" else preferences.csv_delimiter)
        self.encoding.setCurrentText(preferences.csv_encoding)
        self.decimal.setCurrentIndex(self.decimal.findData(preferences.decimal_separator))
        self.templates.combo.setCurrentIndex(0)
        self.invalidate()
        self.errors.clear()
        if self.snapshot:
            self.loaded(self.snapshot[:3])
        else:
            self.preview.setRowCount(0)
            self.mapping.setRowCount(0)

    def clear_template(self, *_):
        self.active_template = None

    def invalidate(self, *_):
        self.raw = None
        if self.preview.rowCount():
            self.preview.setRowCount(0)
            self.summary.setText(tr("Параметры изменились. Обновите предпросмотр."))
        self.preview_source = None
        self.pending_template = self.active_template = None
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def read_settings(self):
        xlsx = bool(self.path and Path(self.path).suffix.lower() == ".xlsx")
        return {"sheet": self.sheet.currentText() or None,
                "delimiter": "\t" if self.separator.currentText() == "TAB" else self.separator.currentText(),
                "encoding": self.encoding.currentText(), "header_rows": self.headers.value(),
                "header_row": self.header_row.value(), "data_end_row": self.data_end.value() or None,
                "first_column": self.first_column.value(), "last_column": self.last_column.value() or None,
                "decimal_separator": "auto" if xlsx else self.decimal.currentData()}

    def load(self, *_, template=None):
        settings = self.read_settings()
        self.settings = settings
        self.invalidate()
        self.pending_template = deepcopy(template)
        def read_snapshot():
            source = file_source(self.path, **settings)
            raw = read_file(self.path, **{key: value for key, value in settings.items() if key != "decimal_separator"})
            if file_source(self.path)["sha256"] != source["sha256"]:
                raise ValueError(tr('Файл изменился во время чтения. Обновите предпросмотр.'))
            kinds = [infer_kind((row[i] for row in raw[1]), settings["decimal_separator"]) for i in range(len(raw[0]))]
            return raw, source, kinds
        self.run_task(read_snapshot, self.file_loaded)

    def file_loaded(self, outcome):
        raw, self.preview_source, kinds = outcome
        pending = self.pending_template
        self.loaded(raw, kinds=kinds)
        self.pending_template = None
        if pending:
            try:
                from autopyro.data_io.import_templates import resolve_sources
                resolved = resolve_sources(pending, self.raw[0])
                self.mapping.blockSignals(True)
                for i in range(self.mapping.rowCount()):
                    self.mapping.cellWidget(i, 0).setChecked(False)
                for i, j in resolved.items():
                    spec = pending["columns"][i]
                    self.mapping.cellWidget(j, 0).setChecked(spec["selected"])
                    self.mapping.item(j, 2).setText(spec["name"])
                    self.mapping.cellWidget(j, 3).setCurrentIndex(self.mapping.cellWidget(j, 3).findData(spec["kind"]))
                    self.mapping.item(j, 4).setText(spec["unit"])
                    aliases = spec.get("aliases", [])
                    self.mapping.item(j, 5).setText(json.dumps(aliases, ensure_ascii=False) if any(";" in a for a in aliases) else "; ".join(aliases))
                self.mapping.blockSignals(False)
                self.mapping.resizeColumnsToContents()
                self.mapping.setColumnWidth(4, max(160, self.mapping.columnWidth(4)))
                self.active_template = deepcopy(pending)
                self.summary.setText(tr('Применён шаблон «{0}» v{1}. Проверьте типы и единицы. Новые неизвестные колонки не включены.', pending['name'], pending['revision']))
            except ValueError as exc:
                self.mapping.blockSignals(False)
                self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
                self.summary.setText(message(exc) + tr('. Прочитайте файл без шаблона для ручной настройки.'))

    def loaded(self, raw, *, kinds=None):
        if kinds is None and len(raw[0]) * len(raw[1]) >= 100_000:
            decimal = self.decimal.currentData()
            kinds = run_with_progress(self, tr("Определение типов колонок…"), lambda: [
                infer_kind((row[i] for row in raw[1]), decimal) for i in range(len(raw[0]))])
        self.raw = raw
        headers, rows, _ = raw
        self.mapping.setRowCount(len(headers))
        used = set()
        for i, header in enumerate(headers):
            check = QCheckBox()
            check.setChecked(True)
            check.toggled.connect(self.clear_template)
            self.mapping.setCellWidget(i, 0, check)
            item = QTableWidgetItem(header)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mapping.setItem(i, 1, item)
            name = header
            suffix = 2
            while name in used:
                name = f"{header} ({suffix})"
                suffix += 1
            used.add(name)
            self.mapping.setItem(i, 2, QTableWidgetItem(name))
            kind = QComboBox()
            for field_kind in FIELD_KINDS:
                kind.addItem(field_kind_label(field_kind), field_kind)
            inferred = kinds[i] if kinds is not None else infer_kind((row[i] for row in rows), self.decimal.currentData())
            kind.setCurrentIndex(kind.findData(inferred))
            kind.setToolTip(tr('Дата: ГГГГ-ММ-ДД или ДД.ММ.ГГГГ. Время: ЧЧ:ММ[:СС]. Дата и время: 2026-11-11 14:30. Логическое значение: true/false, да/нет, истина/ложь или 1/0.'))
            kind.setToolTip(kind.toolTip() + "\n" + tr('Длительность: ЧЧ:ММ:СС (часы могут превышать 23) или число секунд. Категория: текстовая метка. Идентификатор: строка с сохранением ведущих нулей. Целое число: значение без дробной части.'))
            self.mapping.setCellWidget(i, 3, kind)
            kind.currentIndexChanged.connect(self.clear_template)
            self.mapping.setItem(i, 4, QTableWidgetItem(""))
            self.mapping.setItem(i, 5, QTableWidgetItem(""))
        self.mapping.resizeColumnsToContents()
        self.mapping.setColumnWidth(4, max(160, self.mapping.columnWidth(4)))
        self.preview.setColumnCount(len(headers))
        self.preview.setHorizontalHeaderLabels(headers)
        self.preview.setRowCount(min(len(rows), 100))
        self.preview.setVerticalHeaderLabels([str(number) for number in raw[2][:100]])
        for r, row in enumerate(rows[:100]):
            for c, value in enumerate(row):
                self.preview.setItem(
                    r, c, QTableWidgetItem("" if value is None else str(value))
                )
        self.summary.setText(
            tr('{0:,} строк, {1} колонок. Проверьте типы и единицы. Числовой тип допускает ND и <LOD без подстановки чисел; методы не объединяются.', len(rows), len(headers))
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self.errors.clear()
        self.import_errors = []

    @staticmethod
    def aliases(text):
        aliases = json.loads(text) if text.strip().startswith("[") else [part.strip() for part in text.split(";") if part.strip()]
        if not isinstance(aliases, list) or not all(isinstance(a, str) and a for a in aliases):
            raise ValueError(tr('Синонимы должны быть непустыми строками'))
        return aliases

    def save_template(self, name):
        from autopyro.data_io.import_templates import create, resolve_sources
        if self.raw is None:
            self.summary.setText(tr('Сначала прочитайте файл и настройте колонки.'))
            return
        try:
            columns = [{"header": self.raw[0][i], "aliases": self.aliases(self.mapping.item(i, 5).text()),
                "selected": self.mapping.cellWidget(i, 0).isChecked(), "name": self.mapping.item(i, 2).text().strip(),
                "kind": self.mapping.cellWidget(i, 3).currentData(), "unit": self.mapping.item(i, 4).text().strip()}
                for i in range(self.mapping.rowCount())]
            template = create(name, "reader", self.templates.state, file_type="xlsx" if Path(self.path).suffix.lower() == ".xlsx" else "delimited",
                              settings=self.read_settings(), columns=columns)
            resolve_sources(template, self.raw[0])
            self.templates.put(template)
            self.active_template = template
            self.summary.setText(tr('Шаблон сохранится в проекте при импорте. JSON можно экспортировать отдельно.'))
        except (ValueError, TypeError) as exc:
            self.summary.setText(message(exc))

    def apply_template(self, template):
        try:
            from autopyro.data_io.import_templates import validate
            validate(template)
            kind = "xlsx" if Path(self.path).suffix.lower() == ".xlsx" else "delimited"
            if template["file_type"] != kind:
                raise ValueError(tr('Тип файла не соответствует шаблону'))
            settings = template["settings"]
            sheet = settings.get("sheet")
            if sheet and self.sheet.findText(sheet) < 0:
                raise ValueError(tr('Лист из шаблона отсутствует. Выберите лист и настройте импорт вручную'))
            self.templates.put(template)
            self.sheet.setCurrentIndex(self.sheet.findText(sheet) if sheet else 0)
            self.separator.setCurrentText("TAB" if settings.get("delimiter", ";") == "\t" else settings.get("delimiter", ";"))
            self.encoding.setCurrentText(settings.get("encoding", "utf-8-sig"))
            self.headers.setValue(settings.get("header_rows", 1))
            self.header_row.setValue(settings.get("header_row", 1))
            self.data_end.setValue(settings.get("data_end_row") or 0)
            self.first_column.setValue(settings.get("first_column", 1))
            self.last_column.setValue(settings.get("last_column") or 0)
            self.decimal.setCurrentIndex(self.decimal.findData(settings.get("decimal_separator", "auto")))
            self.load(template=template)
        except (ValueError, TypeError, KeyError) as exc:
            self.summary.setText(message(exc))

    def import_data(self):
        if self.raw is None:
            return
        selections = [
            (
                i,
                self.mapping.item(i, 2).text(),
                self.mapping.cellWidget(i, 3).currentData(),
                self.mapping.item(i, 4).text(),
            )
            for i in range(self.mapping.rowCount())
            if self.mapping.cellWidget(i, 0).isChecked()
        ]
        raw = self.raw
        try:
            aliases = [self.aliases(self.mapping.item(selection[0], 5).text()) for selection in selections]
        except (ValueError, TypeError) as exc:
            self.summary.setText(message(exc))
            return
        from .templates import template_state
        from autopyro.data_io.import_templates import reference, merge_views
        state = template_state(self.templates.state)
        active_template = deepcopy(self.active_template)
        decimal = "auto" if self.snapshot else self.settings["decimal_separator"]

        def build():
            source = deepcopy(self.snapshot[3] if self.snapshot else self.preview_source)
            if not self.snapshot and (source is None or file_source(self.path)["sha256"] != source["sha256"]):
                raise ValueError(tr('Файл изменился после предпросмотра. Прочитайте его заново до импорта.'))
            if active_template:
                source["import_template"] = reference(active_template)
            try:
                project = build_project(*raw, selections, source, decimal_separator=decimal)
            except ImportValidationError as exc:
                return exc
            merge_views(project.view, state, activate=True)
            for column, values in zip(project.columns, aliases, strict=True):
                column.metadata["source_aliases"] = values
            return prepare_project(project)

        self.run_task(build, self.imported)

    def imported(self, project):
        if isinstance(project, ImportValidationError):
            self.import_errors = project.errors
            self.errors.setPlainText("\n".join(tr('Строка {0}, {1}: {2!r}\n{3}\n', e['row'], e['column'], e['value'], message(e['message'])) for e in project.errors))
            self.tabs.setCurrentWidget(self.errors)
            self.summary.setText(tr('Ошибок: {0}. Изменения не применены. Проверьте типы и десятичный знак, затем повторите импорт.', len(project.errors)))
            return
        self.project = project
        self.project_prepared = True
        self.accept()
