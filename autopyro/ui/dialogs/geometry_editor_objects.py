"""Edit geometry objects, paths, vertices and classification metadata."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QTableWidgetItem,
)
import numpy as np

from autopyro.model.core import parse_number
from autopyro.schemes.interpretation_thresholds import prepare_threshold
from autopyro.schemes.scheme_geometry import (
    coverage_domain,
    geometry_paths,
    object_at,
    remove_ring_or_part,
    replace_object,
    replace_vertices,
    vertices,
)

from ..i18n import tr, message


def current_entry(sheet):
    if sheet.selected is None:
        return None
    return object_at(sheet.draft, *sheet.selected)


def path_index(sheet, path):
    return next(
        (i for i in range(sheet.paths.count()) if sheet.paths.itemData(i) == path), -1
    )


def select_object(sheet, *args):
    if sheet.loading:
        return
    item = sheet.objects.currentItem()
    previous = sheet.selected
    sheet.selected = item.data(Qt.ItemDataRole.UserRole) if item else None
    entry = sheet.current_entry()
    sheet.loading = True
    sheet.paths.clear()
    rule = entry.get('threshold', {}) if entry else {}
    sheet.threshold_axis.setCurrentIndex(max(0, sheet.threshold_axis.findData(rule.get('axis'))))
    sheet.threshold_less.setText(rule.get('less', ''))
    sheet.threshold_greater.setText(rule.get('greater', ''))
    can_set_threshold = (entry is not None and entry['geometry']['type'] == 'LineString'
                         and sheet.draft.get('strategy', 'regions') == 'regions')
    for widget in (sheet.threshold_axis, sheet.threshold_less, sheet.threshold_greater, sheet.apply_threshold):
        widget.setEnabled(can_set_threshold)
    sheet.region_passport.setEnabled(entry is not None and sheet.selected[0] == 'regions')
    if entry:
        sheet.object_label.setText(entry["label"])
        sheet.object_color.setText(entry.get("color", "#0072B2"))
        for path in geometry_paths(entry):
            sheet.paths.addItem(message(path.label), path)
        index = (
            sheet.path_index(sheet.selected_path) if previous == sheet.selected else -1
        )
        sheet.paths.setCurrentIndex(max(0, index))
    else:
        sheet.object_label.clear()
        sheet.object_color.setText('')
    sheet.loading = False
    sheet.select_path()


def select_path(sheet, *args):
    if sheet.loading:
        return
    sheet.selected_path = sheet.paths.currentData()
    sheet.fill_table()
    sheet.redraw()


def fill_table(sheet, values=None):
    sheet.loading = True
    selected_row = sheet.vertex_table.currentRow()
    entry = sheet.current_entry()
    values = (
        values
        if values is not None
        else vertices(entry, sheet.selected_path)
        if entry and sheet.selected_path
        else []
    )
    sheet.vertex_table.setRowCount(len(values))
    for row, point in enumerate(values):
        for column, value in enumerate(point):
            item = QTableWidgetItem(str(value))
            item.setToolTip(str(value))
            sheet.vertex_table.setItem(row, column, item)
    if 0 <= selected_row < len(values):
        sheet.vertex_table.selectRow(selected_row)
    sheet.loading = False


def table_changed(sheet, item):
    if sheet.loading:
        return
    try:
        values = [
            [parse_number(sheet.vertex_table.item(r, c).text()) for c in range(2)]
            for r in range(sheet.vertex_table.rowCount())
        ]
        entry = replace_vertices(sheet.current_entry(), sheet.selected_path, values)
        sheet.push(
            replace_object(sheet.draft, sheet.selected[0], entry),
            tr('Изменить координату вершины'),
        )
    except (ValueError, TypeError, OverflowError) as exc:
        sheet.fill_table()
        sheet.diagnostics.setText(message(exc))


def vertex_selection_changed(sheet):
    if not sheet.loading and sheet.drag is None:
        sheet.redraw()


def choose_color(sheet):
    color = QColorDialog.getColor(parent=sheet)
    if color.isValid():
        sheet.object_color.setText(color.name())
        sheet.object_properties_changed()


def object_properties_changed(sheet):
    if sheet.loading or not sheet.selected:
        return
    from matplotlib.colors import to_rgba

    try:
        label, color = (
            sheet.object_label.text().strip(),
            sheet.object_color.text().strip(),
        )
        if not label:
            raise ValueError(tr('Подпись объекта не должна быть пустой'))
        to_rgba(color)
        entry = deepcopy(sheet.current_entry())
        entry.update(label=label, color=color)
        sheet.push(
            replace_object(sheet.draft, sheet.selected[0], entry),
            tr('Изменить подпись / цвет'),
        )
    except ValueError as exc:
        sheet.select_object()
        sheet.diagnostics.setText(message(exc))


def edit_fields(sheet, title, fields):
    dialog = QDialog(sheet)
    dialog.setWindowTitle(title)
    form = QFormLayout(dialog)
    editors = []
    for label, value in fields:
        editor = QLineEdit(value)
        form.addRow(label, editor)
        editors.append(editor)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return [editor.text().strip() for editor in editors]
    return None


def edit_region_passport(sheet):
    entry = deepcopy(sheet.current_entry())
    passport = entry.get('passport', {})
    fields = {'source': tr('Источник'), 'description': tr('Описание'), 'limitations': tr('Ограничения')}
    values = sheet.edit_fields(tr('Паспорт области'), [(label, passport.get(key, ''))
                                                      for key, label in fields.items()])
    if values is not None:
        entry['passport'] = {key: value for key, value in zip(fields, values, strict=True) if value}
        if not entry['passport']:
            del entry['passport']
        sheet.push(replace_object(sheet.draft, 'regions', entry), tr('Изменить паспорт области'))


def edit_coverage_domain(sheet):
    keys = [(axis, bound) for axis in ('x', 'y') for bound in ('minimum', 'maximum')]
    fields = []
    for axis, bound in keys:
        spec = sheet.draft[axis]
        value = spec.get(bound)
        label = tr('{0}, минимум [{1}]' if bound == 'minimum' else '{0}, максимум [{1}]',
                   axis.upper(), spec['unit'])
        fields.append((label, '' if value is None else str(value)))
    values = sheet.edit_fields(tr('Границы домена в единицах измерений'), fields)
    if values is None:
        return
    after = deepcopy(sheet.draft)
    try:
        for (axis, bound), value in zip(keys, values, strict=True):
            if value:
                parsed = parse_number(value)
                if parsed is None or not np.isfinite(parsed):
                    raise ValueError(tr('Границы должны быть конечными числами.'))
                after[axis][bound] = parsed
            else:
                after[axis].pop(bound, None)
        for axis in ('x', 'y'):
            low, high = (after[axis].get(key) for key in ('minimum', 'maximum'))
            if low is not None and high is not None and low >= high:
                raise ValueError(tr('Минимум должен быть меньше максимума.'))
        if all(values) and coverage_domain(after) is None:
            raise ValueError(tr('Границы не образуют конечный домен в координатах analysis.'))
    except (ValueError, TypeError) as exc:
        sheet.diagnostics.setText(message(exc))
        return
    sheet.push(after, tr('Изменить границы домена'))


def threshold_changed(sheet):
    if not sheet.apply_threshold.isEnabled():
        return
    from shapely.geometry import shape

    entry = deepcopy(sheet.current_entry())
    axis = sheet.threshold_axis.currentData()
    if axis is None:
        entry.pop('threshold', None)
    else:
        entry['threshold'] = {
            'axis': axis,
            'less': sheet.threshold_less.text().strip(),
            'greater': sheet.threshold_greater.text().strip(),
        }
        try:
            prepare_threshold(entry, shape(entry['geometry']), sheet.draft.get('strategy', 'regions'))
        except ValueError as exc:
            sheet.diagnostics.setText(message(exc))
            return
    sheet.push(replace_object(sheet.draft, 'lines', entry), tr('Изменить правило порога'))


def insert_vertex(sheet):
    if not sheet.selected or not sheet.selected_path:
        return
    values = vertices(sheet.current_entry(), sheet.selected_path)
    index = max(0, sheet.vertex_table.currentRow())
    if not sheet.selected_path.closed and index == len(values) - 1:
        index -= 1
    following = (index + 1) % len(values)
    values.insert(
        index + 1,
        [
            (a + b) / 2
            for a, b in zip(values[index], values[following], strict=True)
        ],
    )
    entry = replace_vertices(sheet.current_entry(), sheet.selected_path, values)
    sheet.push(
        replace_object(sheet.draft, sheet.selected[0], entry),
        tr('Вставить вершину на ребре'),
    )
    sheet.vertex_table.selectRow(index + 1)


def delete_vertex(sheet):
    if (
        not sheet.selected
        or not sheet.selected_path
        or sheet.vertex_table.currentRow() < 0
    ):
        return
    values = vertices(sheet.current_entry(), sheet.selected_path)
    values.pop(sheet.vertex_table.currentRow())
    try:
        entry = replace_vertices(sheet.current_entry(), sheet.selected_path, values)
        sheet.push(
            replace_object(sheet.draft, sheet.selected[0], entry), tr('Удалить вершину')
        )
    except ValueError as exc:
        sheet.diagnostics.setText(message(exc))


def delete_path(sheet):
    if sheet.selected and sheet.selected_path:
        try:
            entry = remove_ring_or_part(sheet.current_entry(), sheet.selected_path)
            sheet.push(
                replace_object(sheet.draft, sheet.selected[0], entry),
                tr('Удалить контур / часть'),
            )
        except ValueError as exc:
            sheet.diagnostics.setText(message(exc))


def delete_object(sheet):
    if not sheet.selected:
        return
    after = deepcopy(sheet.draft)
    collection, object_id = sheet.selected
    after[collection] = [
        entry for entry in after[collection] if entry["id"] != object_id
    ]
    sheet.push(after, tr('Удалить объект'))
