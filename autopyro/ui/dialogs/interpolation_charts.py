"""Display and inspect measured and interpolated values by depth."""

from __future__ import annotations

from operator import itemgetter

from PySide6.QtCharts import QChart, QScatterSeries
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QTableView, QVBoxLayout

from ..i18n import tr
from ..widgets.results import ResultsModel


CHART_SERIES_LIMIT = 5000


def sample_chart_points(points):
    """Keep the X endpoints and the Y extrema in each consecutive X bin."""
    points = list(points)
    total = len(points)
    if total <= CHART_SERIES_LIMIT:
        return points, total

    points.sort(key=itemgetter(0))
    interior = total - 2
    bins = (CHART_SERIES_LIMIT - 2) // 2
    shown = [points[0]]
    y_value = itemgetter(1)
    for i in range(bins):
        start = 1 + i * interior // bins
        stop = 1 + (i + 1) * interior // bins
        group = points[start:stop]
        low = min(group, key=y_value)
        high = max(reversed(group), key=y_value)
        shown.append(low)
        if high[:2] != low[:2]:
            shown.append(high)
    shown.append(points[-1])
    return shown, total


def set_chart_series_points(series, name, points, samples_at=None):
    points = list(points)
    shown, total = sample_chart_points(points)
    series.setName(tr('{0} · {1:,} из {2:,}', name, len(shown), total) if len(shown) < total else name)
    series.append([QPointF(float(x), float(y)) for x, y, _ in shown])
    if samples_at is not None:
        visible = {(float(x), float(y)) for x, y, _ in shown}
        for x, y, rid in points:
            key = float(x), float(y)
            if key in visible:
                samples_at.setdefault(key, []).append(rid)


def open_chart_sample(sheet, chart, samples_at, point):
    if chart is not sheet.chart_view.chart() or not sheet.can_open_sample():
        return
    ids = list(dict.fromkeys(samples_at.get((point.x(), point.y()), ())))
    if len(ids) == 1:
        sheet.owner.open_source_cell(ids[0], sheet._diagnostic_column_id, sheet)
    elif ids:
        sheet.open_sample_choices(ids, tr('Пробы в выбранной точке'))


def update_chart_filter(sheet):
    sheet._chart_dirty = True
    sheet.render_pending_chart()


def chart_data(sheet):
    kind, _, key = (sheet.chart_well.currentData() or "").partition(":")
    _, measurements, excluded, proposals, labels, bound_wells = sheet._pending_chart
    group_index = int(key) if kind == "group" else None

    def included(rid, anchor_id=None):
        if kind == "well":
            return labels[rid][1] == key
        if kind == "bound-well":
            return bound_wells.get(rid) == key
        if kind == "group":
            return sheet._chart_group_by_anchor.get(anchor_id or rid) == group_index
        return True

    return (
        (tr('Опорные измерения'), [(x, y, rid) for rid, x, y in measurements if included(rid)]),
        (tr('Предлагаемые значения'), [(p.x, p.value, p.row_id) for p in proposals
                                      if not p.extrapolated and included(p.row_id, p.left_id)]),
        (tr('Экстраполяция'), [(p.x, p.value, p.row_id) for p in proposals
                              if p.extrapolated and included(p.row_id, p.left_id)]),
        (tr('Исключённые измерения'), [(x, y, rid) for rid, x, y in excluded if included(rid)]
         if sheet.chart_excluded.isChecked() else []),
    )


def show_chart_data(sheet):
    if sheet._pending_chart is None:
        return
    title = tr('Данные графика…')
    dialog = QDialog(sheet)
    dialog.setWindowTitle(title)
    dialog.resize(760, 400)
    layout = QVBoxLayout(dialog)
    scope = QLabel(sheet.chart_scope_text(), dialog)
    scope.setTextFormat(Qt.TextFormat.PlainText)
    scope.setWordWrap(True)
    layout.addWidget(scope)
    labels = sheet._pending_chart[4]
    rows = [(series, *labels[rid], str(x), str(y), *sheet.measurement_context.get(rid, ("", "", "")), rid)
            for series, points in sheet.chart_data() for x, y, rid in points]
    table = QTableView(dialog)
    table.setAccessibleName(title)
    table.setModel(ResultsModel(
        [tr('Статус'), tr('Проба'), tr('Ствол'), "X", "Y", "X · " + tr('Единица'),
         "Y · " + tr('Единица'), tr('База'), tr('UUID пробы')], rows, table,
    ))
    layout.addWidget(table)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()


def chart_scope_text(sheet):
    parts = [sheet.chart_scope_label]
    if sheet.chart_well.currentData():
        parts.append(sheet.chart_well.currentText())
    if not sheet._diagnostics_current:
        parts.append(tr('Показан предыдущий предпросмотр.'))
    return " · ".join(filter(None, parts))


def render_pending_chart(sheet):
    visible = sheet.preview_tabs.currentWidget() is sheet.chart_view and sheet._pending_chart is not None
    kind, _, _ = (sheet.chart_well.currentData() or "").partition(":")
    sheet.chart_well.setVisible(visible and sheet.chart_well.count() > 1)
    sheet.chart_excluded.setVisible(visible and kind != "group" and bool(sheet._pending_chart[2]))
    sheet.chart_table_button.setVisible(visible)
    if not visible or not sheet._chart_dirty:
        return
    data = sheet.chart_data()
    contexts = {sheet.measurement_context[rid] for _, points in data for _, _, rid in points
                if rid in sheet.measurement_context}
    x_label, y_label = sheet._pending_chart[0]
    if len(contexts) == 1:
        x_unit, y_unit, _ = next(iter(contexts))
        x_label, y_label = sheet.chart_axis_names
        x_label += f" [{x_unit}]" if x_unit else ""
        y_label += f" [{y_unit}]" if y_unit else ""
    chart = QChart()
    scope = sheet.chart_scope_text()
    chart.setTitle(f"{y_label} / {x_label} · {scope}")
    if len(contexts) > 1:
        chart.setTitle(scope + " · " + tr('Разные единицы или базы. Выберите группу либо откройте данные графика.'))
        sheet.replace_chart(chart)
        sheet._chart_dirty = False
        return
    samples_at = {} if hasattr(sheet.owner, "open_source_cell") else None
    for index, (title, points) in enumerate(data):
        if index < 2 or points:
            series = QScatterSeries()
            series.setMarkerSize(9 if index == 1 else 7)
            if index == 1:
                series.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeRectangle)
            set_chart_series_points(series, title, points, samples_at)
            chart.addSeries(series)
    if samples_at is not None:
        for series in chart.series():
            series.clicked.connect(lambda point: sheet.open_chart_sample(chart, samples_at, point))
    chart.createDefaultAxes()
    for axis in chart.axes(Qt.Orientation.Horizontal):
        axis.setTitleText(x_label)
    for axis in chart.axes(Qt.Orientation.Vertical):
        axis.setTitleText(y_label)
    sheet.replace_chart(chart)
    sheet._chart_dirty = False


def replace_chart(sheet, chart):
    old_chart = sheet.chart_view.chart()
    sheet.chart_view.setChart(chart)
    if old_chart:
        old_chart.deleteLater()
