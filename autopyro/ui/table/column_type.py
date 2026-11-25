"""Inline column conversion preview and a single reversible schema change."""

from copy import deepcopy

from PySide6.QtGui import QUndoCommand
from PySide6.QtWidgets import QComboBox, QGroupBox, QLabel, QPushButton, QTableView, QVBoxLayout

from autopyro.model.column_types import prepare_column_conversion
from autopyro.model.field_types import FIELD_KINDS, format_duration
from autopyro.model.measurements import derived
from autopyro.model.snapshots import copy_project
from ..field_types import field_kind_label
from ..i18n import message, tr
from ..tasks import run_with_progress
from ..theme import icon
from ..widgets.results import ResultsModel
from .snapshot_policy import large_snapshot


class ChangeColumnTypeCommand(QUndoCommand):
    def __init__(self, model, before, after, column_id):
        super().__init__(tr("Изменить тип колонки"))
        self.model, self.before, self.after, self.column_id = model, before, after, column_id
        self.view_keys = {key for key in before.view.keys() | after.view.keys()
                          if before.view.get(key) != after.view.get(key)}

    def apply(self, snapshot):
        if large_snapshot(snapshot):
            state = run_with_progress(self.model.parent(), tr("Подготовка отмены изменений…"),
                                      lambda: copy_project(snapshot))
        else:
            state = copy_project(snapshot)
        project = self.model.project
        self.model.beginResetModel()
        project.columns, project.rows, project.science = state.columns, state.rows, state.science
        for key in self.view_keys:
            if key in state.view:
                project.view[key] = deepcopy(state.view[key])
            else:
                project.view.pop(key, None)
        project.event("change_column_type", column_id=self.column_id,
                      kind=next(column.kind for column in project.columns if column.id == self.column_id))
        self.model.endResetModel()
        self.model.structureChanged.emit()
        self.model.notify_changed()

    def redo(self):
        self.apply(self.after)

    def undo(self):
        self.apply(self.before)


class ColumnTypeProperties(QGroupBox):
    def __init__(self, window):
        super().__init__(tr("Преобразование типа"))
        self.window = window
        self.column_id = None
        self.prepared = None
        self.context = None
        layout = QVBoxLayout(self)
        self.type_combo = QComboBox()
        for kind in FIELD_KINDS:
            self.type_combo.addItem(field_kind_label(kind), kind)
        self.type_combo.setToolTip(tr("Проверяется вся колонка и все её повторные измерения. При ошибках преобразование не применяется."))
        self.type_combo.setWhatsThis(self.type_combo.toolTip())
        layout.addWidget(self.type_combo)
        self.preview_button = QPushButton(icon("eye"), tr("Предпросмотр преобразования"))
        layout.addWidget(self.preview_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableView()
        self.table.setMinimumHeight(150)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.hide()
        layout.addWidget(self.table)
        self.apply_button = QPushButton(icon("check"), tr("Применить тип"))
        self.apply_button.setEnabled(False)
        layout.addWidget(self.apply_button)
        self.type_combo.currentIndexChanged.connect(self.invalidate)
        self.preview_button.clicked.connect(self.preview)
        self.apply_button.clicked.connect(self.apply)

    def show_column(self, column):
        context = (id(self.window.model), column.id if column else None,
                   column.kind if column else None, self.window.model.data_revision)
        if context == self.context:
            return
        previous = self.context
        self.context = context
        self.column_id = column.id if column else None
        self.setVisible(column is not None)
        self.setEnabled(column is not None and not derived(column))
        if column and (previous is None or previous[:3] != context[:3]):
            self.type_combo.setCurrentIndex(self.type_combo.findData(column.kind))
        self.invalidate()
        if column and derived(column):
            self.status.setText(tr("Тип вычисляемой колонки задаётся её расчётом"))

    def invalidate(self, *_):
        self.prepared = None
        self.apply_button.setEnabled(False)
        self.table.hide()
        column = next((column for column in self.window.model.project.columns if column.id == self.column_id), None)
        self.preview_button.setEnabled(column is not None and column.kind != self.type_combo.currentData())
        self.status.setText(tr("Проверьте преобразование перед применением. Изменение можно отменить через Ctrl+Z."))

    def preview(self):
        model = self.window.model
        revision = model.data_revision
        column_id, kind = self.column_id, self.type_combo.currentData()
        self.invalidate()
        try:
            result = run_with_progress(self, tr("Преобразование типа"),
                                       lambda: prepare_column_conversion(model.project, column_id, kind))
            if self.window.model is not model or model.data_revision != revision or self.column_id != column_id:
                raise ValueError(tr("Данные изменились. Повторите предпросмотр."))
            before, after, rows, errors = result
            display = errors if errors else rows
            source_kind = next(column.kind for column in before.columns if column.id == column_id)
            display = [(row, format_duration(value) if source_kind == "duration" and value is not None else value,
                        format_duration(converted) if kind == "duration" and converted is not None else converted,
                        message(error)) for row, value, converted, error in display]
            old = self.table.model()
            self.table.setModel(ResultsModel([tr("Строка"), tr("До"), tr("После"), tr("Ошибка")],
                                            display, self.table))
            if old:
                old.deleteLater()
            self.table.show()
            self.status.setText(tr("Строк: {0}. Ошибок: {1}.", len(rows), len(errors)))
            if after is not None:
                self.prepared = model, revision, before, after, column_id, kind
                self.apply_button.setEnabled(True)
        except (ValueError, TypeError) as exc:
            self.status.setText(message(exc))

    def apply(self):
        if self.prepared is None:
            return
        model, revision, before, after, column_id, kind = self.prepared
        if self.window.model is not model or model.data_revision != revision or self.type_combo.currentData() != kind:
            self.invalidate()
            self.status.setText(tr("Данные изменились. Повторите предпросмотр."))
            return
        self.invalidate()
        model.undo.push(ChangeColumnTypeCommand(model, before, after, column_id))
