"""Handle mouse and keyboard gestures for drawing and moving vertices."""

from __future__ import annotations

from uuid import uuid4

from PySide6.QtCore import Qt

from autopyro.schemes.scheme_geometry import (
    add_geometry,
    add_ring_or_part,
    geometry_paths,
    object_at,
    replace_object,
    replace_vertices,
    vertices,
)

from ..i18n import tr, message


def mode_changed(sheet, *args):
    sheet.cancel_drawing()
    if sheet.toolbar.mode:
        if str(sheet.toolbar.mode) == "pan/zoom":
            sheet.toolbar.pan()
        else:
            sheet.toolbar.zoom()


def cancel_drawing(sheet):
    sheet.drawing_points = []
    sheet.drawing_target = None
    sheet.drag = None
    sheet.fill_table()
    sheet.redraw()


def finish_drawing(sheet):
    mode = sheet.mode.currentData()
    if not sheet.drawing_points or mode == "move":
        return
    try:
        if mode in ("polygon", "line"):
            object_id = uuid4().hex
            collection = "regions" if mode == "polygon" else "lines"
            label = (
                tr('Область') if mode == "polygon" else tr('Линия')
            ) + f" {len(sheet.draft.get(collection, [])) + 1}"
            after = add_geometry(
                sheet.draft,
                "Polygon" if mode == "polygon" else "LineString",
                sheet.drawing_points,
                object_id,
                label,
            )
            sheet.selected = (collection, object_id)
        else:
            if sheet.drawing_target is None:
                raise ValueError(tr('Сначала выберите объект и его часть'))
            collection, object_id, path = sheet.drawing_target
            entry = object_at(sheet.draft, collection, object_id)
            replacement = add_ring_or_part(
                entry, path, sheet.drawing_points, hole=mode == "hole"
            )
            after = replace_object(sheet.draft, collection, replacement)
        sheet.drawing_points = []
        sheet.drawing_target = None
        sheet.push(after, tr('Нарисовать контур'))
    except (ValueError, TypeError, StopIteration) as exc:
        sheet.diagnostics.setText(message(exc))


def on_press(sheet, event):
    if event.inaxes is not sheet.ax or sheet.toolbar.mode:
        return
    if event.button == 3:
        sheet.finish_drawing()
        return
    if event.button != 1 or event.xdata is None or event.ydata is None:
        return
    if sheet.mode.currentData() != "move":
        if not sheet.drawing_points and sheet.mode.currentData() in ("hole", "part"):
            if not sheet.selected or not sheet.selected_path:
                sheet.diagnostics.setText(
                    tr('Выберите объект и его часть перед рисованием')
                )
                return
            sheet.drawing_target = (*sheet.selected, sheet.selected_path)
        point = [float(event.xdata), float(event.ydata)]
        if not sheet.drawing_points or point != sheet.drawing_points[-1]:
            sheet.drawing_points.append(point)
        if event.dblclick:
            sheet.finish_drawing()
        else:
            sheet.redraw(preview_only=True)
        return
    nearest = None
    for i in range(sheet.objects.count()):
        collection, object_id = sheet.objects.item(i).data(Qt.ItemDataRole.UserRole)
        entry = object_at(sheet.draft, collection, object_id)
        for path in geometry_paths(entry):
            for index, point in enumerate(vertices(entry, path)):
                px, py = sheet.ax.transData.transform(point)
                distance = (px - event.x) ** 2 + (py - event.y) ** 2
                if distance <= 100 and (nearest is None or distance < nearest[0]):
                    nearest = (distance, i, path, index)
    if nearest:
        _, row, path, index = nearest
        sheet.objects.setCurrentRow(row)
        sheet.paths.setCurrentIndex(sheet.path_index(path))
        sheet.vertex_table.selectRow(index)
        sheet.drag = {
            "index": index,
            "values": vertices(sheet.current_entry(), sheet.selected_path),
        }


def on_motion(sheet, event):
    if (
        sheet.drag
        and event.inaxes is sheet.ax
        and event.xdata is not None
        and event.ydata is not None
    ):
        sheet.drag["values"][sheet.drag["index"]] = [
            float(event.xdata),
            float(event.ydata),
        ]
        sheet.fill_table(sheet.drag["values"])
        sheet.redraw(preview_only=True)


def on_release(sheet, event):
    if sheet.drag is None:
        return
    values = sheet.drag["values"]
    sheet.drag = None
    try:
        entry = replace_vertices(sheet.current_entry(), sheet.selected_path, values)
        sheet.push(
            replace_object(sheet.draft, sheet.selected[0], entry),
            tr('Переместить вершину'),
        )
    except ValueError as exc:
        sheet.diagnostics.setText(message(exc))


def on_key(sheet, event):
    if event.key in ("escape", "esc"):
        sheet.cancel_drawing()
    elif event.key in ("enter", "return"):
        sheet.finish_drawing()
