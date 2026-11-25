"""Install, display and invalidate interpolation preview results."""

from __future__ import annotations

from collections import Counter

from PySide6.QtCharts import QChart
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import QDialogButtonBox

from autopyro.analysis.interpolation import DuplicateCoordinatesError, minimum_anchor_count

from ..i18n import tr, message
from ..widgets.documents import refill_combo
from ..widgets.results import ResultsModel


BACKGROUND_ROWS = 1_000


class InterpolationPreviewModel(ResultsModel):
    def __init__(self, parent=None):
        super().__init__(['Проба', "X", 'Новое Y', 'Опорные пробы', 'Ствол', 'Расчётная группа'], (), parent)
        self.extrapolation_label = tr('Экстраполяция')
        self.sample_labels = {}
        self.segment_method = False
        self.support_details = {}
        self.group_labels = {}

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return None
        proposal = self.rows[index.row()]
        column = index.column()
        sample, well = self.sample_labels.get(proposal.row_id, (proposal.row_id, ""))
        if column == 3:
            left = self.sample_labels.get(proposal.left_id, (proposal.left_id, ""))[0]
            right = self.sample_labels.get(proposal.right_id, (proposal.right_id, ""))[0]
            supports = f"{left} → {right}"
            if role == Qt.ItemDataRole.ToolTipRole:
                detail = (tr('Показаны соседние опоры. Метод использует все опоры непрерывного участка.')
                          if self.segment_method else tr('Enter или двойной щелчок открывает опорные пробы.'))
                return f"{supports}\n{detail}"
            return f"{supports} · {self.extrapolation_label}" if proposal.extrapolated else supports
        if column == 0 and role == Qt.ItemDataRole.ToolTipRole:
            return f"{sample}\n{proposal.row_id}"
        if column == 2 and role == Qt.ItemDataRole.ToolTipRole and proposal.row_id in self.support_details:
            distance, outside = self.support_details[proposal.row_id]
            detail = tr('До ближайшей опоры: {0:.6g} в единицах X.', distance)
            return detail + ("\n" + tr('Значение вне диапазона Y опор.') if outside else "")
        group = self.group_labels.get(proposal.row_id, "")
        return str((sample, proposal.x, proposal.value, "", well, group)[column])


class InterpolationDiagnosticsModel(ResultsModel):
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            # Diagnostics are already translated; sample labels are literal.
            return self.rows[index.row()][index.column()]
        return super().data(index, role)


def install_preview(sheet, result, parameters, axis_labels, token, well_columns, well_ids):
    well_id = next(iter(well_ids), None)
    bound_well_id = well_ids[1] if len(well_ids) > 1 else None
    if isinstance(result[0], DuplicateCoordinatesError):
        error, rows = result
        sheet.proposals = []
        sheet.protocol = None
        sheet.preview_model.set_rows(())
        sheet._pending_chart = None
        sheet.replace_chart(QChart())
        sheet.set_diagnostics((rows, rows, 0), parameters["x_id"])
        sheet.preview_tabs.setCurrentWidget(sheet.diagnostics.parentWidget())
        sheet.summary.setText(tr('Ошибка: {0}', message(error)) + "\n" +
                             tr('Конфликтующих проб: {0}.', len(rows)))
        sheet.announce_summary()
        return
    (sheet.proposals, sheet.protocol, diagnostics, measurements, excluded_points,
     sample_labels, sheet._support_spans, sheet.preview_model.support_details,
     sheet._chart_group_by_anchor, chart_groups, bound_wells, sheet.measurement_context, lost_anchors) = result
    sheet.parameters = parameters
    sheet.set_diagnostics(diagnostics, parameters["y_id"])
    _, _, unknown = diagnostics
    sheet.preview_model.sample_labels = sample_labels
    group_labels = [tr('Группа {0} · {1}', i, label) for i, label in enumerate(chart_groups, 1)]
    sheet.preview_model.group_labels = {
        p.row_id: group_labels[sheet._chart_group_by_anchor[p.left_id] - 1] for p in sheet.proposals
    }
    sheet.preview_model.segment_method = parameters["method"] in ("pchip", "spline")
    output_contexts = sheet.output_contexts()
    if output_contexts:
        y_column = next(c for c in sheet.project.columns if c.id == parameters["y_id"])
        units = {unit for unit, _ in output_contexts}
        unit = next(iter(units)) if len(units) == 1 else ""
        sheet.preview_model.headers[1:3] = [axis_labels[0], f"{y_column.name} [{unit}]" if unit else y_column.name]
    else:
        sheet.preview_model.headers[1:3] = axis_labels
    sheet.preview_model.set_rows(sheet.proposals)
    sheet.summary.setText(
        tr('Предлагается значений: {0}. Только значения в новой колонке; исходные не меняются.', len(sheet.proposals))
    )
    if len(output_contexts) > 1:
        sheet.summary.setText(sheet.summary.text() + "\n" + tr('Для создания колонки выберите область с одной единицей и базой Y.'))
    excluded = len(sheet._excluded_rows)
    sheet.summary.setText(sheet.summary.text() + "\n" + tr('Ограничения: исключено {0}; метод не указан у {1} проб.', excluded, unknown))
    counts = Counter(row[3] for row in sheet._excluded_rows)
    if counts:
        sheet.summary.setText(sheet.summary.text() + "\n" +
                             " · ".join(f"{label}: {count}" for label, count in counts.most_common(3)))
    if lost_anchors:
        sheet.summary.setText(sheet.summary.text() + "\n" + tr('Из-за совпадений X исключено числовых опор: {0}.', lost_anchors))
    outside = sum(outside for _, outside in sheet.preview_model.support_details.values())
    if outside:
        sheet.summary.setText(sheet.summary.text() + "\n" + tr('Значений вне диапазона Y опор: {0}.', outside))
    if parameters["method"] == "spline":
        minimum = parameters.get("min_anchors", minimum_anchor_count("spline"))
        sheet.summary.setText(sheet.summary.text() + "\n" + tr('Минимум опор на непрерывном участке: {0}.', minimum))
    if "max_extrapolation" in parameters:
        count = sum(p.extrapolated for p in sheet.proposals)
        sheet.summary.setText(sheet.summary.text() + "\n" + tr('Экстраполировано значений: {0}.', count))
    sheet._pending_chart = axis_labels, measurements, excluded_points, sheet.proposals, sample_labels, bound_wells
    sheet.chart_scope_label = sheet.scope.currentText()
    columns = {column.id: column.name for column in sheet.project.columns}
    sheet.chart_axis_names = columns[parameters["x_id"]], columns[parameters["y_id"]]
    sheet._chart_dirty = True
    wells = sorted({well for _, well in sample_labels.values() if well})
    choices = [(tr('Все стволы и группы'), None)]
    choices.extend((f"{well_columns[well_id]}: {well}" if bound_well_id else well, f"well:{well}")
                   for well in wells)
    if bound_well_id:
        choices.extend((f"{well_columns[bound_well_id]}: {well}", f"bound-well:{well}")
                       for well in sorted(set(bound_wells.values())))
    if len(group_labels) > 1:
        choices.extend((label, f"group:{i}") for i, label in enumerate(group_labels, 1))
    refill_combo(sheet.chart_well, choices)
    if (sheet.chart_well.currentData() or "").startswith("group:"):
        blocker = QSignalBlocker(sheet.chart_well)
        sheet.chart_well.setCurrentIndex(0)
        del blocker
    if sheet.preview_tabs.currentWidget() is sheet.chart_view:
        sheet.render_pending_chart()
    elif sheet.chart_view.chart().series():
        sheet.replace_chart(QChart())
    sheet.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
        bool(sheet.proposals) and len(output_contexts) == 1
    )
    sheet.validate_button.setEnabled(bool(sheet._support_spans))
    if sheet.property("workspaceTracked"):
        sheet._preview_source = token
    sheet.announce_summary()


def invalidate(sheet, *args, announce=True):
    was_current = sheet._diagnostics_current
    error = sheet.input_error()
    sheet.refresh_saved_runs()
    sheet.preview_button.setEnabled(not error)
    sheet.without_well.setVisible(sheet.selected_well() is None)
    sheet.export_diagnostics_button.setEnabled(False)
    sheet.validate_button.setEnabled(False)
    sheet.validation_status.hide()
    if error:
        previous = "\n" + tr('Показан предыдущий предпросмотр.') if sheet._diagnostic_rows else ""
        sheet.summary.setText(error + previous)
    elif sheet.preview_model.rowCount() or sheet._diagnostic_rows:
        sheet.summary.setText(tr("Данные или выборка изменились. Рассчитайте предпросмотр заново.")
                             + "\n" + tr('Показан предыдущий предпросмотр.'))
    else:
        sheet.summary.setText(tr('Рассчитайте предпросмотр. Исходный Y сохраняется.'))
    sheet._diagnostics_current = False
    sheet.update_result_accessibility()
    if was_current and sheet._pending_chart:
        sheet._chart_dirty = True
        sheet.render_pending_chart()
    if was_current and announce:
        sheet.announce_summary()
    if sheet.diagnostic_hint is not None:
        sheet.diagnostic_hint.setText(tr('Предыдущие результаты доступны только для просмотра.'))
        sheet.diagnostic_hint.setVisible(bool(sheet.diagnostic_model.rowCount()))
    sheet._preview_source = None
    sheet.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)


def update_result_accessibility(sheet):
    stale = tr('Предыдущие результаты доступны только для просмотра.')
    sheet.preview.setAccessibleDescription(
        stale if not sheet._diagnostics_current and sheet.preview_model.rowCount() else ""
    )
    sheet.diagnostics.setAccessibleDescription(stale if not sheet._diagnostics_current and sheet._diagnostic_rows else "")
    chart_help = tr('Откройте «Данные графика», чтобы прочитать значения в таблице.')
    sheet.chart_view.setAccessibleDescription(
        f"{stale} {chart_help}" if not sheet._diagnostics_current and sheet._pending_chart else chart_help
    )
