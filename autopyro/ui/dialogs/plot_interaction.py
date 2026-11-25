"""Handle plot selection, hover, picking and annotation editing."""

from __future__ import annotations

from ..i18n import tr
from ..theme import icon
from matplotlib.backend_bases import MouseButton
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QMenu, QToolTip


def install(dialog):
    dialog.hover_index = None
    dialog.hover_rows = None
    dialog.hover_results = None
    dialog.hover_texts = {}
    dialog.hover_event = None
    dialog.hover_timer = QTimer(dialog)
    dialog.hover_timer.setSingleShot(True)
    dialog.hover_timer.setInterval(60)
    dialog.hover_timer.timeout.connect(dialog.show_hover)
    dialog.selection_overlay = None
    dialog.rendered_ids = []
    dialog.selectors = []


def selection_ids(dialog):
    if dialog.interpretation_owner is not None:
        return set(dialog.interpretation_owner.result_selection)
    return dialog.owner.sample_selection.ids()


def selection_changed(dialog, *args):
    if dialog.closed or dialog.loading:
        return
    if dialog.scope.currentData() == 'selected' or dialog.result is None or dialog.timer.isActive():
        dialog.schedule()
    elif dialog.selection_overlay is not None:
        dialog.selection_overlay.set_selected(dialog.selection_ids())


def enable_brush(dialog, enabled):
    for selector in dialog.selectors:
        selector.set_active(enabled)
    if enabled and dialog.toolbar.mode:
        if str(dialog.toolbar.mode) == "pan/zoom":
            dialog.toolbar.pan()
        else:
            dialog.toolbar.zoom()


def select_rows(dialog, row_ids):
    if dialog.interpretation_owner is not None:
        dialog.interpretation_owner.select_result_rows(row_ids)
        return
    hidden = dialog.owner.sample_selection.select(row_ids)
    if hidden:
        dialog.status.setText(tr('Выбранных проб скрыто текущим фильтром таблицы: {0}', hidden))


def connect_canvas_events(dialog):
    dialog.canvas.mpl_connect('pick_event', dialog.on_pick)
    dialog.canvas.mpl_connect('button_release_event', dialog.annotation_context_menu)
    dialog.canvas.mpl_connect('motion_notify_event', dialog.on_hover)
    dialog.canvas.mpl_connect('figure_leave_event', dialog.hide_hover)


def annotation_context_menu(dialog, event):
    if (event.button != MouseButton.RIGHT or not dialog.result or dialog.toolbar.mode
            or dialog.brush.isChecked() or dialog.spec.interpretation_panels
            or event.inaxes not in dialog.result['axes'] or event.inaxes.name == '3d'):
        return
    ax = event.inaxes
    x, y = ax.transAxes.inverted().transform((event.x, event.y))
    track_id = dialog.spec.track_ids[dialog.result['axes'].index(ax)] if dialog.spec.kind == 'multi_depth' else ''
    position = (min(1., max(0., round(float(x), 4))), min(1., max(0., round(float(y), 4))), track_id)
    menu = QMenu(dialog.canvas)
    menu.setObjectName('plotAnnotationMenu')
    menu.addAction(icon('plus'), tr('Добавить подпись здесь…'), lambda: dialog.edit_annotations(position))
    menu.aboutToHide.connect(menu.deleteLater)
    menu.popup(QCursor.pos())


def edit_annotations(dialog, position=None):
    from .plot_annotations import PlotAnnotationsDialog

    project = dialog.owner.model.project
    labels = {column.id: column.name for column in project.columns}
    tracks = [(cid, labels[cid]) for cid in dialog.selected_tracks() if cid in labels
              and dialog.selected_kind() == 'multi_depth']
    current_x = dialog.x_form.fields['scale'][0].currentData()
    common_x = dialog.spec.x_axis.scale if dialog.previous_x_target else current_x
    y_scale = dialog.y_form.fields['scale'][0].currentData()
    if dialog.selected_kind() in ('box', 'violin'):
        y_scale = 'linear'
    scales = {'': (current_x, y_scale)}
    for cid, _ in tracks:
        axis = dialog.spec.track_axes.get(cid)
        x_scale = axis.scale if axis else common_x
        if cid == dialog.previous_x_target:
            x_scale = current_x
        scales[cid] = (x_scale, y_scale)
    if tracks:
        scales[''] = scales[tracks[0][0]]
    annotation_dialog = PlotAnnotationsDialog(dialog, dialog.spec.annotations, tracks, position=position,
                                   is_3d=dialog.selected_kind().startswith('mpl3d:'), axis_scales=scales)
    if annotation_dialog.exec() and not dialog.closed and project is dialog.owner.model.project:
        dialog.spec.annotations = annotation_dialog.annotations
        dialog.schedule()
    annotation_dialog.deleteLater()


def hide_hover(dialog, *args):
    dialog.hover_timer.stop()
    dialog.hover_event = None
    QToolTip.hideText()


def clear_hover(dialog, *args):
    dialog.hide_hover()
    dialog.hover_index = None
    dialog.hover_rows = None
    dialog.hover_results = None
    dialog.hover_texts.clear()


def on_hover(dialog, event):
    if (dialog.closed or dialog.result is None or event.inaxes not in dialog.result['axes']
            or event.xdata is None or event.ydata is None or event.button is not None
            or dialog.toolbar.mode):
        dialog.hover_timer.stop()
        dialog.hover_event = None
        QToolTip.hideText()
        return
    dialog.hover_event = event
    if not dialog.hover_timer.isActive():
        dialog.hover_timer.start()


def point_tooltip(dialog, row_id, coordinates, binding=None, *, track_id=None):
    from html import escape

    binding = binding if binding is not None else dialog.spec.interpretation
    x_id, y_id = (binding['x_id'], binding['y_id']) if binding else (track_id or dialog.spec.x_id, dialog.spec.y_id)
    report_key = (x_id, y_id, binding['scheme']['id'], binding['scheme']['revision'],
                  binding.get('x_unit'), binding.get('y_unit'), binding.get('text_numbers')) if binding else None
    cache_key = row_id, x_id, y_id, report_key
    if cache_key in dialog.hover_texts:
        return dialog.hover_texts[cache_key]
    if dialog.hover_rows is None:
        dialog.hover_rows = dialog.owner.model.project.row_map()
    row = dialog.hover_rows.get(row_id)
    if row is None:
        return ''
    columns = {c.id: c for c in dialog.owner.model.project.columns}
    z_id = dialog.spec.z_id if dialog.spec.kind.startswith('mpl3d:') else ''
    ids = dict.fromkeys((x_id, y_id, z_id, dialog.spec.label_by,
                         dialog.spec.color_by, dialog.spec.shape_by, dialog.spec.size_by))
    lines = [tr('Проба: {0}', row_id)]
    for cid in ids:
        if cid in columns:
            column = columns[cid]
            unit = f" [{column.unit}]" if column.unit else ''
            lines.append(f"{column.name}{unit}: {row.values.get(cid, '')}")
    if binding:
        from autopyro.schemes.interpretation import GeometricEngine, InterpretationInput, STATUSES

        scheme = binding['scheme']
        lines.append(tr('Схема: {0}', scheme['name']))
        if dialog.hover_results is None:
            dialog.hover_results = {}
        if report_key not in dialog.hover_results:
            document = dialog.interpretation_owner
            candidates = ([document.report, *document.comparison_reports.values()] if document is not None
                          else [dialog.owner.model.project.view.get('interpretation_result')])
            report = next((r for r in candidates if r and not r.get('stale') and r.get('scheme') == scheme
                           and all(r['binding'].get(key) == binding.get(key)
                                   for key in ('x_id', 'y_id', 'x_unit', 'y_unit', 'text_numbers'))), None)
            dialog.hover_results[report_key] = {r['row_id']: r for r in report['results']} if report else {}
        item = dialog.hover_results[report_key].get(row_id)
        if item is None:
            from dataclasses import asdict
            item = asdict(GeometricEngine().evaluate([InterpretationInput(row_id, *coordinates)], scheme)[0])
        labels = {e['id']: e['label'] for e in scheme.get('regions', []) + scheme.get('lines', [])}
        lines.append(tr('Интерпретация: {0}', tr(STATUSES[item['status']])))
        names = list(dict.fromkeys(labels[key] for key in item['regions'] + item['boundary_regions'] + (item.get('possible_types') or item['line_hits'])))
        lines.extend(names)
        if item.get('thresholds'):
            from autopyro.schemes.interpretation_thresholds import threshold_labels

            lines.extend(threshold_labels(scheme, item['thresholds']))
        if item.get('possible_types'):
            lines.append(tr('Возможные типы: {0}', '; '.join(labels.get(key, key) for key in item['possible_types'])))
        if item.get('uncertainty_regions'):
            lines.append(tr('Области интервала: {0}', '; '.join(labels.get(key, key) for key in item['uncertainty_regions'])))
        if item.get('value') is not None:
            lines.append(f"HIo: {item['value']:g} {scheme['y']['unit']}")
        if item.get('value_bounds'):
            lines.append(tr('Диапазон HIo: {0}–{1}', *item['value_bounds']))
        if item.get('reason'):
            lines.append(item['reason'])
    text = '<qt>' + '<br>'.join(escape(str(line)) for line in lines) + '</qt>'
    dialog.hover_texts[cache_key] = text
    return text


def show_hover(dialog):
    from html import escape
    from autopyro.visualization.plot_hover import HoverIndex

    event = dialog.hover_event
    if (event is None or dialog.closed or dialog.result is None or dialog.toolbar.mode
            or event.inaxes not in dialog.result['axes']):
        QToolTip.hideText()
        return
    if dialog.hover_index is None:
        dialog.hover_index = HoverIndex(dialog.result, dialog.spec.interpretation)
    hit = dialog.hover_index.point(event.inaxes, event.x, event.y, with_count=True)
    names = dialog.hover_index.region_labels(event.xdata, event.ydata, event.inaxes)
    details = []
    if hit is not None:
        track_id = None
        if dialog.spec.kind == 'multi_depth':
            track_id = dialog.spec.track_ids[dialog.result['axes'].index(event.inaxes)]
        text = dialog.point_tooltip(hit[0], hit[1],
                                  getattr(event.inaxes, '_autopyro_interpretation', dialog.spec.interpretation),
                                  track_id=track_id)
        if hit[3] > 1:
            details.append(tr('Проб рядом с указателем: {0}. Щёлкните для выбора.', hit[3]))
    else:
        text = ''
    if names:
        details.append(tr('Области под указателем: {0}', '; '.join(names)))
    if details:
        extra = '<br>'.join(escape(detail) for detail in details)
        text = text[:-5] + '<br>' + extra + '</qt>' if text else '<qt>' + extra + '</qt>'
    if text:
        QToolTip.showText(QCursor.pos(), text, dialog.canvas)
    else:
        QToolTip.hideText()


def on_pick(dialog, event):
    mouse = getattr(event, 'mouseevent', None)
    if mouse is not None and mouse.button != MouseButton.LEFT:
        return
    ids = getattr(event.artist, "_autopyro_row_ids", [])
    if dialog.result is not None and ids and len(event.ind) and not dialog.toolbar.mode:
        matches = list(dict.fromkeys(ids[int(index)] for index in event.ind))
        if len(matches) > 1 and getattr(event, 'mouseevent', None) is not None and not dialog.brush.isChecked():
            from ..table.selection import sample_label_column

            project = dialog.owner.model.project
            label_id = dialog.spec.label_by or sample_label_column(project.columns)
            positions = dialog.owner.model.resolve_row_positions(matches)
            menu = QMenu(dialog.canvas)
            menu.addAction(icon('check'), tr('Выделить все совпавшие пробы ({0})', len(matches)),
                           lambda: dialog.select_rows(matches))
            menu.addSeparator()
            for identifier in matches[:30]:
                position = positions.get(identifier)
                row = project.rows[position] if position is not None else None
                label = row.values.get(label_id) if row is not None and label_id else None
                text = f'{label} · {identifier[:8]}' if label else identifier
                menu.addAction(icon('table-2'), text, lambda checked=False, rid=identifier: dialog.select_rows([rid]))
            menu.exec(QCursor.pos())
            menu.deleteLater()
        else:
            dialog.select_rows(matches)


def on_rectangle(dialog, start, end):
    if (
        dialog.toolbar.mode
        or not dialog.result
        or start.xdata is None
        or end.xdata is None
        or start.ydata is None
        or end.ydata is None
    ):
        return
    xmin, xmax = sorted((start.xdata, end.xdata))
    ymin, ymax = sorted((start.ydata, end.ydata))
    ids = set()
    for artist in dialog.result["artists"]:
        if artist.axes is not start.inaxes:
            continue
        if hasattr(artist, "_autopyro_intervals"):
            for rid, (x, top, bottom) in zip(
                artist._autopyro_row_ids, artist._autopyro_intervals, strict=True
            ):
                if xmin <= x <= xmax and top <= ymax and bottom >= ymin:
                    ids.add(rid)
            continue
        for rid, (x, y) in zip(
            artist._autopyro_row_ids, artist.get_offsets(), strict=True
        ):
            if xmin <= x <= xmax and ymin <= y <= ymax:
                ids.add(rid)
    dialog.select_rows(ids)
