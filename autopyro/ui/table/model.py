from __future__ import annotations

from contextlib import contextmanager, nullcontext
from collections import OrderedDict

from ..i18n import tr, message
from autopyro.model.snapshots import copy_project

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QUndoStack
from PySide6.QtWidgets import QApplication, QStyle

from autopyro.model.core import Project
from autopyro.model.measurements import derived, parse_result, parsed_table_value
from autopyro.model.field_types import format_duration
from .commands import EditCommand
from .snapshot_policy import large_project, large_snapshot


def context_issue_text(fields):
    labels = {"method": tr('метод'), "laboratory": tr('лаборатория'),
              "preparation": tr('подготовка'), "fraction": tr('фракция'),
              "heating_program": tr('программа нагрева'), "basis": tr('база измерения')}
    return tr('Различается аналитический контекст: {0}', ", ".join(labels.get(key, key) for key in fields))


def uncertainty_tooltip(info, unit):
    if not info:
        return ""
    if info["status"] == "ok":
        contributions = ", ".join(f"{role}: {value:.6g}" for role, value in info["contributions"].items())
        tooltip = tr('Стандартная неопределённость: {0:.6g} [{1}]\nВклады: {2}',
                     info["standard"], unit, contributions)
        if info.get("unstable_denominator"):
            tooltip += "\n" + tr('Неопределённость знаменателя достигает его значения. Линейная оценка ненадёжна.')
        return tooltip
    reasons = {
        "missing": tr('нет значения'),
        "unsupported_type": tr('требуется тип standard'),
        "unit_mismatch": tr('несовместимая единица'),
        "invalid_amount": tr('недопустимый диапазон'),
    }
    details = ", ".join(f"{role}: {reasons[reason]}" for role, reason in info.get("inputs", {}).items())
    return tr('Неопределённость недоступна: {0}', details or tr('численная ошибка'))


def format_censored_range(interval):
    lower = f"{interval['lower']:.10g}"
    upper = f"{interval['upper']:.10g}" if interval["upper"] is not None else "∞"
    left = "[" if interval["lower_closed"] else "("
    right = "]" if interval["upper_closed"] else ")"
    return f"{left}{lower}, {upper}{right}"


def range_issue_text(info):
    reasons = {
        "missing": tr('нет значения'),
        "non_detect": "ND",
        "other": tr('нечисловые данные'),
        "invalid_range": tr('недопустимый диапазон'),
        "invalid_range_input": tr('нет допустимого значения или предела'),
        "missing_lod": tr('укажите числовой LOD рабочего измерения'),
        "lower_limit": tr('указан только нижний предел'),
        "qc_excluded": tr('Исключено решением QC'),
        "interpolation_input": tr('вход-интерполяция'),
        "unit_mismatch": tr('несовместимая единица'),
        "undefined_range": tr('формула не определена на этом диапазоне'),
    }
    details = []
    for issue in info.get("range_issues", (info,)):
        code = issue["range_issue"]
        details.append(f"{issue['role']}: {reasons.get(code, code)}")
    return "\n".join(details)


def snapshot_project(project, parent=None):
    """Freeze editable inputs before handing them to an isolated data operation."""
    if large_snapshot(project):
        from ..tasks import run_with_progress
        return run_with_progress(parent, tr("Подготовка снимка данных…"), lambda: copy_project(project))
    return copy_project(project)


def prepare_project(project):
    """Prepare plain Python project data before attaching it to a Qt model."""
    project.sync_measurements()
    if any(column.metadata.get("calculation") for column in project.columns):
        from autopyro.analysis.calculations import recompute

        recompute(project)
    if any(column.metadata.get("nonpyro") for column in project.columns):
        from autopyro.analysis.nonpyro import recompute as recompute_nonpyro

        recompute_nonpyro(project)
    if any(column.metadata.get("stratigraphy") for column in project.columns):
        from autopyro.geology.stratigraphy import recompute as recompute_stratigraphy

        recompute_stratigraphy(project)
    from autopyro.curves.joint_sources import update_joint_status

    update_joint_status(project)
    return project


class TableModel(QAbstractTableModel):
    changed = Signal()
    updatingData = Signal(bool)
    updatingColumns = Signal(bool)
    structureChanged = Signal()
    derivedStructureChanged = Signal(object)
    qcExclusionsChanged = Signal()
    columnContextChanged = Signal()
    error = Signal(str)

    @staticmethod
    def preserve_recovery_edit(rows, changes):
        return nullcontext()

    def __init__(self, project: Project, parent=None, *, prepared=False):
        super().__init__(parent)
        self.project = project
        if not prepared:
            prepare_project(project)
        self.data_revision = 0
        self.cell_edit_revision = None
        self.cell_edit_columns = frozenset()
        self.undo = QUndoStack(self)
        self._sort_index_column = -1
        self._sort_indexes = ()
        self._sample_measurement_cache = None

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.project.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.project.columns)

    def index(self, row, column, parent=QModelIndex()):
        if parent.isValid():
            return QModelIndex()  # The project table is flat and has no children.
        if column == self._sort_index_column and 0 <= row < len(self._sort_indexes):
            return self._sort_indexes[row]
        if 0 <= row < len(self.project.rows) and 0 <= column < len(self.project.columns):
            return self.createIndex(row, column)
        return QModelIndex()

    @contextmanager
    def cache_sort_indexes(self, column):
        """Reuse valid root indexes during one synchronous Qt sort only."""
        if not 0 <= column < len(self.project.columns):
            yield
            return
        previous = self._sort_index_column, self._sort_indexes
        # Source rows do not move when a proxy sorts. All bounds are known here.
        indexes = tuple(self.createIndex(row, column) for row in range(len(self.project.rows)))
        self._sort_index_column, self._sort_indexes = column, indexes
        try:
            yield
        finally:
            self._sort_index_column, self._sort_indexes = previous

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.project.rows[index.row()]
        column = self.project.columns[index.column()]
        value = row.values[column.id]
        if role == Qt.ItemDataRole.UserRole:
            return value
        if role == Qt.ItemDataRole.DisplayRole and column.metadata.get("calculation"):
            diagnostic = column.metadata.get("diagnostics", {}).get(row.id, {})
            if diagnostic.get("status") == "censored_range":
                return format_censored_range(diagnostic["range"])
            uncertainty = diagnostic.get("uncertainty", {})
            if value is not None and uncertainty.get("status") == "ok":
                warning = " ⚠" if uncertainty.get("unstable_denominator") else ""
                return f"{value:.6g} ± {uncertainty['standard']:.6g}{warning}"
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if column.kind == "duration" and value is not None:
                return format_duration(value)
            return "" if value is None else str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole and column.kind in ("number", "integer", "duration"):
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.ForegroundRole and derived(column):
            return QColor("#246c89")
        if role == Qt.ItemDataRole.ToolTipRole:
            original = row.original.get(column.id, "—")
            diagnostic = column.metadata.get("diagnostics", {}).get(row.id, {})
            config = column.metadata.get("calculation")
            if config and config.get("check_context") is not True and (
                value is not None or isinstance(diagnostic, dict) and diagnostic.get("status") == "censored_range"
            ):
                from autopyro.analysis.calculations import context_conflicts

                context_fields = context_conflicts(
                    self.project, row.id, (item.get("column_id") for item in config.get("inputs", {}).values())
                )
            else:
                context_fields = ()
            detail = (uncertainty_tooltip(diagnostic.get("uncertainty"), column.unit)
                      if isinstance(diagnostic, dict) else "")
            if isinstance(diagnostic, dict) and diagnostic.get("status") == "censored_range":
                detail = tr('Диапазон по пределам измерений: {0} [{1}]',
                            format_censored_range(diagnostic["range"]), column.unit)
                diagnostic = {}
            elif isinstance(diagnostic, dict) and diagnostic.get("status") == "context_conflict":
                detail = context_issue_text(diagnostic["fields"])
                diagnostic = {}
            elif isinstance(diagnostic, dict) and "range_issue" in diagnostic:
                detail = range_issue_text(diagnostic)
                diagnostic = {}
            elif detail:
                diagnostic = {k: v for k, v in diagnostic.items() if k != "uncertainty"}
            if context_fields:
                detail = "\n".join(filter(None, (context_issue_text(context_fields), detail)))
            metadata = {
                k: v
                for k, v in column.metadata.items()
                if k not in ("diagnostics", "proposals", "interpolation_protocol", "scope_row_ids")
            }
            measurement = self.project.science.current(row.id, column.id)
            working = ""
            if measurement:
                analysis = self.project.science.analyses[measurement.analysis_id]
                working = (tr('\nРабочее измерение: {0} · версия {1}\nМетод: {2}; лаборатория: {3}', measurement.id, measurement.revision, analysis.method or tr('не указан'), analysis.laboratory or tr('не указана')))
            tooltip = tr('UUID: {0}{1}\nИсходное при импорте: {2}\nИсточник: {3}\n{4}\n{5}', row.id, working, original, row.source, diagnostic, metadata)
            return f"{tooltip}\n{detail}" if detail else tooltip
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Vertical:
            return str(section + 1) if role == Qt.ItemDataRole.DisplayRole else None
        column = self.project.columns[section]
        joint = column.metadata.get("joint_curve_fit")
        source_status = column.metadata.get("source_status", "unverified") if joint else None
        stale = bool(column.metadata.get("stale") and
                     (column.metadata.get("curve_fit") or column.metadata.get("method")))
        if role == Qt.ItemDataRole.DecorationRole and (stale or source_status in ("stale", "unverified")):
            return QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        label = column.name + (f" [{column.unit}]" if column.unit else "")
        if source_status == "stale" or stale:
            label = tr("{0} · устарел", label)
        elif source_status == "unverified":
            label = tr("{0} · актуальность не подтверждена", label)
        return label

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and not derived(self.project.columns[index.column()]):
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        column = self.project.columns[index.column()]
        if derived(column):
            return False
        try:
            if column.kind != "text":
                value = parsed_table_value(parse_result(value, column.kind))
            else:
                value = str(value) if str(value) else None
        except (TypeError, ValueError) as exc:
            self.error.emit(message(exc))
            return False
        row = self.project.rows[index.row()]
        if row.values[column.id] == value:
            return True
        self.undo.push(
            EditCommand(self, [(row.id, column.id, value)], row_positions={row.id: index.row()})
        )
        return True

    def resolve_row_positions(self, row_ids, hints=None):
        """Resolve UUIDs using validated positions; scan only for missing hints.

        Hints belong to an edit command, not a full-table cache. A reset/reorder
        may invalidate them, so never use a position without checking its UUID.
        Missing UUIDs are omitted; the caller decides whether they are an error.
        """
        wanted = set(row_ids)
        rows = self.project.rows
        positions = {rid: position for rid, position in (hints or {}).items()
                     if rid in wanted and 0 <= position < len(rows) and rows[position].id == rid}
        missing = wanted - positions.keys()
        if missing:
            for position, row in enumerate(rows):
                if row.id in missing:
                    positions[row.id] = position
                    missing.remove(row.id)
                    if not missing:
                        break
        return positions

    def measurement_results(self, sample_id, property_id):
        """Reuse results while navigating recent samples; invalidate on data changes."""
        measurements = self.project.science.measurements
        key = (self.data_revision, id(measurements), len(measurements))
        cached = self._sample_measurement_cache
        if cached is None or cached[0] != key:
            cached = self._sample_measurement_cache = key, OrderedDict()
        samples = cached[1]
        if sample_id not in samples:
            by_property = {}
            for measurement in measurements.values():
                if measurement.sample_id == sample_id:
                    by_property.setdefault(measurement.property_id, []).append(measurement)
            samples[sample_id] = by_property
            if len(samples) > 8:
                samples.popitem(last=False)
        samples.move_to_end(sample_id)
        return samples[sample_id].get(property_id, ())

    def refresh(self, *, invalidate_interpolation=True, rows=None, sync_measurements=True,
                changed_columns=None, row_positions=None, stratigraphy_report=None, edited_cells=None):
        fit_status = self._fit_header_status()
        self.project.refresh(invalidate_interpolation=invalidate_interpolation, rows=rows,
                             sync_measurements=sync_measurements, changed_columns=changed_columns,
                             stratigraphy_report=stratigraphy_report, edited_cells=edited_cells)
        self._notify_fit_headers(fit_status)
        self.cell_edit_revision = self.data_revision + 1 if edited_cells else None
        self.cell_edit_columns = frozenset(cid for columns in (edited_cells or {}).values() for cid in columns)
        self.notify_changed(rows=rows, row_positions=row_positions)

    def _fit_header_status(self):
        return {c.id: (c.metadata.get("stale"), c.metadata.get("source_status"))
                for c in self.project.columns
                if c.metadata.get("curve_fit") or c.metadata.get("joint_curve_fit") or c.metadata.get("method")}

    def _notify_fit_headers(self, previous):
        changed_headers = [i for i, c in enumerate(self.project.columns)
                           if c.id in previous and (c.metadata.get("stale"), c.metadata.get("source_status")) != previous[c.id]]
        if changed_headers:
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, min(changed_headers), max(changed_headers))

    def refresh_joint_qc_status(self):
        from autopyro.curves.joint_sources import update_joint_status

        previous = self._fit_header_status()
        update_joint_status(self.project)
        self._notify_fit_headers(previous)

    def refresh_derived_structure(self, column_id):
        """Finish a derived-column insertion/removal without changing source cells."""
        previous = self._fit_header_status()
        affected = self.project.refresh(invalidate_interpolation=False, sync_measurements=False,
                                        changed_columns=[column_id])
        self._notify_fit_headers(previous)
        if affected is None or affected - {column_id}:
            self.notify_changed()  # Existing cells or other engines may have changed.
            return
        # Qt has already received columnsInserted/Removed. Only this column's
        # values changed; proxies must revisit rules that explicitly use its UUID.
        self.data_revision += 1
        self.columnContextChanged.emit()
        self.derivedStructureChanged.emit({column_id})
        self.changed.emit()

    def notify_changed(self, *, rows=None, data_changed=True, row_positions=None):
        if data_changed:
            # Qt can remove proxy rows synchronously during dataChanged.
            self.updatingData.emit(True)
            try:
                self._notify_data_changed(rows, row_positions)
            finally:
                self.updatingData.emit(False)
        self.changed.emit()

    def _notify_data_changed(self, rows, row_positions):
        self.data_revision += 1
        self.columnContextChanged.emit()
        if self.rowCount() and self.columnCount():
            first, last = 0, self.rowCount() - 1
            # Pyrolysis formulas are row-local. Other derived data may depend on
            # multiple rows and still require a full notification.
            if rows is not None and not any(any(c.metadata.get(key) for key in
                    ("method", "stratigraphy", "nonpyro")) for c in self.project.columns):
                positions = self.resolve_row_positions((row.id for row in rows), row_positions).values()
                if positions:
                    first, last = min(positions), max(positions)
            self.dataChanged.emit(self.index(first, 0), self.index(last, self.columnCount() - 1))
