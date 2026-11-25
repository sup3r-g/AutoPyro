"""Inspect a cell's analyses and explicitly select its working measurement."""


from ..i18n import tr, message
import json

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QTabWidget,
)

from autopyro.model.core import parse_number
from ..table import AddMeasurementCommand, SelectMeasurementCommand
from ..widgets.parameters import add_parameter_tools
from ..theme import icon


QUALIFIERS = {"exact": "Точное", "missing": "Нет результата", "non_detect": "Не обнаружено",
              "below_limit": "Ниже предела", "above_limit": "Выше предела",
              "approximate": "Приблизительное", "unparsed": "Требует разбора"}


def display(value):
    return tr('не указано') if value is None or value == "" else str(value)


class DerivedCellDialog(QDialog):
    """Show provenance for one cell without unpacking the full run protocol."""

    def __init__(self, project, row, column, parent=None, *, open_support=None):
        super().__init__(parent)
        self.setWindowTitle(tr('Производный показатель'))
        self.resize(720, 500)
        layout = QVBoxLayout(self)
        metadata = column.metadata
        proposal = next((item for item in metadata.get("proposals", ()) if item.get("row_id") == row.id), None)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlainText(self.describe(project, row, column, proposal))
        tabs = QTabWidget()
        summary = QPlainTextEdit()
        summary.setReadOnly(True)
        lines = [f"{column.name}   {display(row.values.get(column.id))} {column.unit or ''}",
                 tr('Проба: {0}', row.id)]
        if equation := metadata.get("equation"):
            lines.append(str(equation))
        if "stale" in metadata:
            lines.append(tr('Результат устарел') if metadata["stale"] else tr('Результат актуален'))
        if proposal:
            lines.extend((tr('Опорные пробы интерполяции'), display(proposal.get("left_id")),
                          display(proposal.get("right_id"))))
        summary.setPlainText("\n".join(lines))
        tabs.addTab(summary, icon("table-2"), tr('Краткая сводка'))
        tabs.addTab(self.details, icon("logs"), tr('Протокол'))
        layout.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        source_column = metadata.get("y_id")
        if open_support is not None and source_column and proposal:
            left, right = proposal.get("left_id"), proposal.get("right_id")
            for rid, label in ((left, tr('К левой опорной пробе')),
                               (right if right != left else None, tr('К правой опорной пробе'))):
                if rid:
                    button = buttons.addButton(label, QDialogButtonBox.ButtonRole.ActionRole)
                    button.clicked.connect(
                        lambda checked=False, rid=rid: self.go_to_support(open_support, rid, source_column)
                    )
        layout.addWidget(buttons)

    def go_to_support(self, open_support, row_id, column_id):
        if open_support(row_id, column_id, self):
            self.accept()

    @staticmethod
    def describe(project, row, column, proposal):
        metadata = column.metadata
        lines = [
            tr('Проба: {0}\nПоказатель: {1}', row.id, column.name),
            tr('Значение: {0}', display(row.values.get(column.id))),
            tr('Единица: {0}', display(column.unit)),
        ]
        stale = metadata.get("stale")
        if stale is not None:
            lines.append(tr('Результат устарел') if stale else tr('Результат актуален'))

        def section(title, content):
            if content:
                lines.extend(("", title, json.dumps(content, ensure_ascii=False, indent=2, default=str)))

        calculation = metadata.get("calculation")
        if calculation:
            section(tr('Расчёт'), {
                "formula": calculation.get("formula"),
                "equation": metadata.get("equation"),
                "inputs": calculation.get("inputs"),
                "scientific_source": metadata.get("scientific_source"),
            })
        if metadata.get("method"):
            section(tr('Интерполяция'), {
                key: metadata[key] for key in ("method", "x_id", "y_id", "group_ids", "max_gap",
                                               "max_extrapolation", "min_anchors", "duplicate_policy")
                if key in metadata
            })
            if proposal:
                if proposal.get("extrapolated"):
                    lines.append(tr('Экстраполяция'))
                lines.extend(("", tr('Опорные пробы интерполяции'), display(proposal.get("left_id")),
                              display(proposal.get("right_id"))))
        if curve := metadata.get("curve_fit"):
            run = project.view.get("curve_fit_runs", {}).get(curve.get("run_id"), {})
            section(tr('Подгонка кривой'), {
                **curve,
                **{key: run[key] for key in ("formula", "parameters", "source_ids") if key in run},
            })
        if analysis := metadata.get("nonpyro"):
            section(tr('Расчёт'), {"method": analysis.get("config", {}).get("method"), "output": analysis.get("output")})
        section(tr('Стратиграфия'), metadata.get("stratigraphy"))
        section(tr('Диагностика результата'), metadata.get("diagnostics", {}).get(row.id))
        section(tr('Источник пробы'), row.source)
        return "\n".join(lines)


class NewMeasurementDialog(QDialog):
    def __init__(self, model, sample_id, property_id, parent=None):
        super().__init__(parent)
        self.model, self.sample_id, self.property_id = model, sample_id, property_id
        self.setWindowTitle(tr('Новый результат анализа · исходный результат сохраняется'))
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.fields = {}
        for name, label in (("raw", tr('Значение (включая ND, <, >, ≈)')), ("method", tr('Метод')),
                            ("laboratory", tr('Лаборатория')), ("date", tr('Дата анализа')),
                            ("instrument", tr('Прибор')), ("heating_program", tr('Программа нагрева')),
                            ("preparation", tr('Подготовка')), ("fraction", tr('Фракция / навеска')),
                            ("basis", tr('Основание измерения')), ("lod", "LOD"), ("loq", "LOQ"),
                            ("uncertainty", tr('Неопределённость')), ("uncertainty_type", tr('Тип неопределённости')),
                            ("confidence_level", tr('Уровень доверия (0–1)'))):
            self.fields[name] = QLineEdit()
            form.addRow(label, self.fields[name])
        self.fields['uncertainty_type'].setPlaceholderText('absolute / standard / expanded / relative_percent')
        self.fields['uncertainty_type'].setToolTip(tr('absolute: ±u в единицах показателя; standard: ±1u; expanded: уже расширенное ±u; relative_percent: ±u процентов от значения. Множитель доверия автоматически не рассчитывается.'))
        layout.addLayout(form)
        definition = model.project.science.properties[property_id]
        layout.addWidget(QLabel(tr('Единица показателя: {0}. Новый результат не выбирается автоматически.', definition.unit or tr('не задана'))))
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('Добавить'))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('Отмена'))
        buttons.accepted.connect(self.submit)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        add_parameter_tools(self, changed=self.error.clear)

    def submit(self):
        values = {key: field.text() for key, field in self.fields.items()}
        if not values["method"].strip():
            self.error.setText(tr('Укажите метод нового анализа.'))
            return
        try:
            analysis_fields = {key: values[key].strip() for key in
                ("method", "laboratory", "date", "instrument", "heating_program", "preparation", "fraction")}
            details = {key: parse_number(values[key]) for key in ("lod", "loq", "uncertainty", "confidence_level")}
            details.update(basis=values["basis"].strip(), uncertainty_type=values["uncertainty_type"].strip())
            # Prepare without mutation; cancelling or invalid input cannot alter the project.
            measurement, analysis = self.model.project.science.prepare_result(
                self.sample_id, self.property_id, values["raw"], analysis_fields=analysis_fields, details=details)
            self.model.undo.push(AddMeasurementCommand(self.model, measurement, analysis))
            self.accept()
        except (ValueError, TypeError, OverflowError) as exc:
            self.error.setText(message(exc))


class MeasurementDialog(QDialog):
    def __init__(self, model, sample_id, property_id, parent=None, *, measurement_id=None):
        super().__init__(parent)
        self.model, self.sample_id, self.property_id = model, sample_id, property_id
        self.setWindowTitle(tr('Измерения · источник, история и рабочий результат'))
        self.resize(970, 640)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(tr('Проба: {0}', sample_id)))
        self.property_picker = QComboBox()
        self.property_picker.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.property_picker.setMinimumContentsLength(12)
        properties = model.project.science.properties
        for column in model.project.columns:
            if column.id in properties:
                self.property_picker.addItem(column.name, column.id)
        self.property_picker.setCurrentIndex(self.property_picker.findData(property_id))
        form = QFormLayout()
        form.addRow(tr('Показатель'), self.property_picker)
        layout.addLayout(form)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([tr('Рабочее'), tr('Исходная запись'), tr('Квалификатор'), tr('Единица'), tr('Метод'), tr('Лаборатория'), tr('Версия')])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.show_details)
        self.table.setMaximumHeight(240)
        layout.addWidget(self.table)
        tabs = QTabWidget()
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.source_details = QTextEdit()
        self.source_details.setReadOnly(True)
        self.history_details = QTextEdit()
        self.history_details.setReadOnly(True)
        tabs.addTab(self.details, icon('flask-conical'), tr('Измерение и анализ'))
        tabs.addTab(self.source_details, icon('database'), tr('Источник'))
        tabs.addTab(self.history_details, icon('logs'), tr('История правок'))
        layout.addWidget(tabs)
        buttons = QHBoxLayout()
        self.select_button = QPushButton(icon('check'), tr('Использовать выбранное измерение'))
        self.select_button.clicked.connect(self.select_result)
        buttons.addWidget(self.select_button)
        add = QPushButton(icon('plus'), tr('Добавить результат анализа…'))
        add.clicked.connect(self.add_result)
        buttons.addWidget(add)
        close = QPushButton(icon('x'), tr('Закрыть'))
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.property_picker.currentIndexChanged.connect(self.change_property)
        self.reload(measurement_id=measurement_id)
        add_parameter_tools(self, help_only=True)

    def change_property(self):
        self.property_id = self.property_picker.currentData()
        self.reload()

    def reload(self, *, measurement_id=None):
        store = self.model.project.science
        results = self.model.measurement_results(self.sample_id, self.property_id)
        current = store.current(self.sample_id, self.property_id)
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(results))
            selected_row = 0
            requested_row = None
            for i, measurement in enumerate(results):
                analysis = store.analyses[measurement.analysis_id]
                is_current = current is not None and measurement.id == current.id
                values = [tr('Да') if is_current else "", measurement.raw_value,
                          tr(QUALIFIERS[measurement.qualifier]), measurement.unit, analysis.method or tr('не указан'),
                          analysis.laboratory, measurement.revision]
                for j, value in enumerate(values):
                    item = QTableWidgetItem("" if value is None else str(value))
                    item.setData(Qt.ItemDataRole.UserRole, measurement.id)
                    self.table.setItem(i, j, item)
                if is_current:
                    selected_row = i
                if measurement.id == measurement_id:
                    requested_row = i
            self.table.resizeColumnsToContents()
            self.table.selectRow(selected_row if requested_row is None else requested_row)
            if requested_row is not None:
                self.table.scrollToItem(self.table.item(requested_row, 0))
        self.show_details()

    def selected_id(self):
        item = self.table.item(self.table.currentRow(), 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def show_details(self):
        mid = self.selected_id()
        self.select_button.setEnabled(mid is not None)
        if mid is None:
            self.details.clear()
            self.source_details.clear()
            self.history_details.clear()
            return
        store = self.model.project.science
        measurement = store.measurements[mid]
        analysis = store.analyses[measurement.analysis_id]
        source = store.sources.get(measurement.source_id or analysis.source_id)
        lines = [tr('Исходная запись результата: {0}', display(measurement.raw_value)),
                 tr('Квалификатор: {0}', tr(QUALIFIERS[measurement.qualifier])),
                 tr('Единица: {0}', display(measurement.unit)), tr('Основание: {0}', display(measurement.basis))]
        if measurement.limit is not None:
            lines.append(tr('Порог квалификатора: {0}{1}', measurement.comparator, measurement.limit))
        for label, value in (("LOD", measurement.lod), ("LOQ", measurement.loq),
                             (tr('Неопределённость'), measurement.uncertainty), (tr('Тип неопределённости'), measurement.uncertainty_type),
                             (tr('Уровень доверия'), measurement.confidence_level), (tr('Метод'), analysis.method),
                             (tr('Лаборатория'), analysis.laboratory), (tr('Дата анализа'), analysis.date),
                             (tr('Прибор'), analysis.instrument), (tr('Программа нагрева'), analysis.heating_program),
                             (tr('Подготовка'), analysis.preparation), (tr('Фракция / навеска'), analysis.fraction)):
            lines.append(f"{label}: {display(value)}")
        lines.extend([tr('\nUUID измерения: {0}', measurement.id), tr('UUID анализа: {0}', analysis.id), tr('Версия: {0}', measurement.revision)])
        self.details.setPlainText("\n".join(lines))
        if source:
            labels = {"path": tr('Файл'), "file": tr('Файл'), "sheet": tr('Лист'), "row": tr('Строка'),
                      "imported_at": tr('Дата импорта'), "sha256": "SHA-256", "source_record_id": tr('Запись БД'),
                      "sample_id": tr('Проба в БД'), "import_batch_id": tr('Загрузка БД')}
            origin = [tr('Значение при импорте: {0}', display(source.original.get(self.property_id)))]
            for key, value in source.provenance.items():
                value = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else display(value)
                origin.append(f"{labels.get(key, key)}: {value}")
            self.source_details.setPlainText("\n".join(origin))
        else:
            self.source_details.setPlainText(tr('Результат добавлен вручную. Внешний источник не указан.'))
        history = [tr('Создано: {0}', measurement.created_at), tr('Текущая версия: {0}', measurement.revision)]
        for entry in measurement.history:
            previous = entry["previous"]
            history.append(tr('\n{0} — изменена версия {1}:\nПредыдущая запись: {2}', entry['at'], previous['revision'], display(previous['raw_value'])))
        if not measurement.history:
            history.append(tr('Правок значения пока нет.'))
        self.history_details.setPlainText("\n".join(history))

    def select_result(self):
        mid = self.selected_id()
        current = self.model.project.science.current(self.sample_id, self.property_id)
        if mid and (current is None or current.id != mid):
            self.model.undo.push(SelectMeasurementCommand(self.model, self.sample_id, self.property_id, mid))
            self.reload()

    def add_result(self):
        dialog = NewMeasurementDialog(self.model, self.sample_id, self.property_id, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.reload()
