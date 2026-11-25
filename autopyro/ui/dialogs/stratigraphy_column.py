"""Render and navigate stratigraphy columns and sample tracks."""

from __future__ import annotations

from ..i18n import tr, message
from copy import deepcopy
from PySide6.QtCore import QItemSelectionModel, QTimer
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QInputDialog, QLabel, QVBoxLayout
from autopyro.geology.stratigraphy import render_columns
from autopyro.geology.stratigraphy_samples import sample_positions
from ..tasks import TaskCancelled, run_with_progress
from ..theme import icon
from ..widgets.stratigraphy_labels import geological_label
from ..widgets.lithology import LithologyCatalog


def initialize(editor):
    editor._has_column_preview = False
    editor._column_preview_dirty = False
    editor._preview_scheme = None
    editor._sample_track_cache = None
    editor._batch_export_running = False
    editor._column_render_running = False
    editor._column_preview_revision = 0
    editor._preview_depth_bounds = None
    editor.column_preview_timer = QTimer(editor)
    editor.column_preview_timer.setSingleShot(True)
    editor.column_preview_timer.setInterval(450)
    editor.column_preview_timer.timeout.connect(editor.refresh_column_preview)


def invalidate_column(editor, *_):
    editor._column_preview_revision += 1
    editor.column_preview_timer.stop()
    editor._preview_scheme = None
    editor.column_navigation.clear()
    editor.column_canvas.defer_navigation_draws(False)
    editor._column_preview_dirty = False
    editor._has_column_preview = False
    editor.invalidate_preview()
    editor.interval_selection.clear()
    editor.column_figure.clear()
    editor.column_canvas.draw_idle()
    editor.column_toolbar.setEnabled(False)
    editor.export_column_button.setEnabled(False)
    editor.status.setText(tr('Параметры изменены; постройте колонку или проверьте привязку заново.'))


def schedule_column_preview(editor):
    if not editor._has_column_preview and not (editor._column_render_running and editor._preview_scheme is not None):
        return
    editor._column_preview_revision += 1
    editor._column_preview_dirty = True
    editor.column_toolbar.setEnabled(False)
    editor.export_column_button.setEnabled(False)
    editor.status.setText(tr('Оформление изменено. Предпросмотр обновится после паузы в редактировании.'))
    editor.column_preview_timer.stop()
    editor.resume_column_preview()


def resume_column_preview(editor, *_):
    editor.column_preview_timer.stop()
    if editor._batch_export_running or editor._column_render_running:
        return
    if editor._column_preview_dirty and editor.isVisible() and editor.column_page.isVisible():
        editor.column_preview_timer.start()


def refresh_column_preview(editor):
    if editor._batch_export_running or editor._column_render_running:
        return
    if editor._column_preview_dirty and editor.isVisible() and editor.column_page.isVisible():
        editor.render_column_preview(automatic=True)


def show_column(editor):
    editor.column_preview_timer.stop()
    editor.render_column_preview(automatic=False)


def render_column_preview(editor, *, automatic):
    if editor._column_render_running:
        return
    try:
        # Style changes reuse the last validated draft. Draft edits invalidate it.
        options = editor.render_options()
        scheme = editor._preview_scheme if automatic and editor._preview_scheme is not None else editor.read_render_scheme()
        samples, sample_notice = editor.preview_samples(scheme, options)
        depth_bounds = (options['depth_min'], options['depth_max'], tuple(options['hierarchy_levels']),
                        tuple(options['comparison_wells']), options['correlation'],
                        tuple((well, min(row.top for row in rows),
                               max(row.top if row.bottom is None else row.bottom for row in rows))
                              for well, rows in samples.items()))
        previous_limits = (editor.column_figure.axes[0].get_ylim()
                           if automatic and editor.column_figure.axes
                           and depth_bounds == editor._preview_depth_bounds else None)
        editor.canvas_view.refresh()
        well, level = editor.column_well.currentData(), editor.level.currentData()
        wells = {well, *options['comparison_wells']}
        levels = {level, *options['hierarchy_levels']}
        count = sum(item['wellbore'] in wells and item['level'] in levels for item in scheme['intervals'])
        large = count >= editor.BACKGROUND_PREVIEW_INTERVALS or sum(map(len, samples.values())) >= editor.BACKGROUND_BINDING_ROWS
        editor.column_canvas.defer_navigation_draws(large)
        if large:
            revision = editor._column_preview_revision
            editor._preview_scheme = scheme
            editor.result_tabs.setCurrentWidget(editor.column_page)
            notices = editor.render_large_column(scheme, well, level, options, samples, previous_limits, revision)
            if notices is None:
                editor.resume_column_preview()
                return
        else:
            editor.column_navigation.clear()
            _, notices = render_columns(scheme, well, level, figure=editor.column_figure,
                                        options=options, samples=samples)
            editor.label_column_axes(editor.column_figure, scheme['depth_unit'], level)
            if previous_limits is not None:
                editor.column_figure.axes[0].set_ylim(previous_limits)
            editor.column_canvas.draw()
        notices = [editor.column_notice(notice) for notice in notices]
        if sample_notice:
            notices.append(sample_notice)
        editor.column_toolbar.update()
        editor.column_toolbar.setEnabled(True)
        editor.export_column_button.setEnabled(True)
        editor._has_column_preview = True
        editor._column_preview_dirty = False
        editor._preview_scheme = scheme
        editor._preview_depth_bounds = depth_bounds
        editor.interval_selection.set_rendered(scheme, well, level)
        if not automatic:
            editor.parameter_tabs.setCurrentIndex(1)
            editor.result_tabs.setCurrentWidget(editor.column_page)
        editor.status.setText(tr('Колонка построена. Привязка проб и замеров доступна следующим шагом.')
                            + (" " + " ".join(notices) if notices else ""))
    except TaskCancelled:
        editor._column_preview_dirty = False
        editor.column_toolbar.setEnabled(False)
        editor.export_column_button.setEnabled(False)
        editor.status.setText(tr('Подготовка отменена.'))
    except Exception as exc:
        if not automatic:
            editor._has_column_preview = False
            editor._column_preview_dirty = False
            editor._preview_scheme = None
            editor.interval_selection.clear()
            editor.column_navigation.clear()
            editor.column_canvas.defer_navigation_draws(False)
            editor.column_figure.clear()
            editor.column_canvas.draw_idle()
        editor.column_toolbar.setEnabled(False)
        editor.export_column_button.setEnabled(False)
        editor.status.setText(tr('Ошибка: {0}', message(exc)))


def render_large_column(editor, scheme, well, level, options, samples, limits, revision):
    from matplotlib.backend_bases import FigureCanvasBase
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = editor.column_canvas
    screen_dpi = canvas.figure._original_dpi
    dpi = screen_dpi * canvas.device_pixel_ratio
    size = (canvas.width() / screen_dpi, canvas.height() / screen_dpi)
    title = tr('Построить колонку')

    def render(update):
        update(title, 0)
        figure, notices = render_columns(scheme, well, level, options=options, samples=samples)
        update(title, 50)
        editor.label_column_axes(figure, scheme['depth_unit'], level)
        if limits is not None:
            figure.axes[0].set_ylim(limits)
        figure.set_size_inches(*size, forward=False)
        figure.set_dpi(dpi)
        # Qt keeps its own axes for picking and navigation history.
        navigation_scene = FigureCanvasAgg(deepcopy(figure))
        agg = FigureCanvasAgg(figure)
        agg.draw()
        return figure, notices, agg.buffer_rgba(), navigation_scene

    editor._column_render_running = True
    editor.column_preview_timer.stop()
    try:
        figure, notices, pixels, scene = run_with_progress(editor, title, render, cancellable=True)
    except Exception:
        if revision != editor._column_preview_revision:
            return None
        raise
    finally:
        editor._column_render_running = False
    if revision != editor._column_preview_revision:
        return None
    previous = editor.column_figure
    editor.column_navigation.clear()
    editor.column_navigation.scene = scene
    editor.column_figure = canvas.figure = figure
    figure.set_canvas(canvas)
    figure._original_dpi = screen_dpi
    editor.column_toolbar.reconnect_canvas()
    for axis in figure.axes:
        if hasattr(axis, '_navigate_mode'):
            axis.set_navigate_mode(editor.column_toolbar.mode.name if editor.column_toolbar.mode else None)
    editor.interval_selection.connect_canvas()
    FigureCanvasBase(previous)
    previous.clear()
    canvas.present(pixels)
    return notices


def label_column_axes(figure, unit, level):
    for axis in figure.axes:
        kind = getattr(axis, '_autopyro_kind', '')
        if kind == 'samples':
            title = tr('Пробы')
        elif kind == 'lithology':
            title = tr('Литология')
        else:
            title = geological_label(getattr(axis, '_autopyro_level', level))
        axis.set_title(title)
        if hasattr(axis, '_autopyro_depth_origin'):
            origin = axis._autopyro_depth_origin
            axis.format_coord = lambda x, y, origin=origin: (
                tr('Исходная глубина: {0:.6g} {2} · Относительная глубина: {1:.6g} {2}',
                   y + origin, y, unit))


def preview_samples(editor, scheme, options):
    if not editor.sample_track.isChecked():
        return {}, ''
    scope = editor.owner.analysis_scope
    model = editor.owner.model
    binding = editor.current_binding()
    level = editor.level.currentData()
    selected = [editor.column_well.currentData(), *options['comparison_wells']]
    wells = editor.sample_track_wells(scheme, options, level, selected)
    key = (id(model), model.data_revision, scope.revision, scheme['depth_reference'],
           scheme['depth_unit'], binding, wells)
    if editor._sample_track_cache is not None and editor._sample_track_cache[0] == key:
        return editor._sample_track_cache[1]
    try:
        samples = editor.calculate_sample_positions(scheme, binding, wells)
    except ValueError as exc:
        return {}, tr('Дорожка проб: {0}', message(exc))
    result = samples, '' if samples else tr('Нет проб с подходящими глубинами в выбранной области.')
    editor._sample_track_cache = key, result
    return result


def sample_track_wells(scheme, options, level, selected):
    selected = set(selected)
    levels = {level, *options['hierarchy_levels']}
    return {item['wellbore'] for item in scheme['intervals']
            if item['wellbore'] in selected and item['level'] in levels}


def column_samples(editor, scheme, options, level, selected):
    wells = editor.sample_track_wells(scheme, options, level, selected)
    return editor.calculate_sample_positions(scheme, editor.current_binding(), wells)


def calculate_sample_positions(editor, scheme, binding, wells):
    scope = editor.owner.analysis_scope
    scope_revision = scope.revision
    row_ids = scope.rows()
    revision = editor._column_preview_revision

    def calculate(project, checkpoint):
        return sample_positions(project, row_ids, scheme, binding, wells, checkpoint=checkpoint)

    editor._column_render_running = True
    try:
        samples = editor.calculate_binding_report(row_ids, (binding,), tr('Подготовка снимка данных…'), calculate)
    finally:
        editor._column_render_running = False
    if scope.revision != scope_revision or revision != editor._column_preview_revision:
        raise TaskCancelled()
    return samples


def open_sample_row(editor, event):
    matches = {}
    for artist, rows in event.inaxes._autopyro_sample_artists.items():
        hit, details = artist.contains(event)
        if hit:
            for index in details['ind']:
                sample = rows[index]
                matches[sample.row_id] = sample
    if not matches:
        return
    if len(matches) == 1:
        key = next(iter(matches))
    else:
        keys = list(matches)
        labels = []
        for sample in matches.values():
            depth = f'{sample.top:.8g}' if sample.bottom is None else f'{sample.top:.8g}–{sample.bottom:.8g}'
            labels.append(f'{depth} · {sample.row_id}')
        label, accepted = QInputDialog.getItem(editor, tr('Выбор пробы'), tr('Пробы на этой глубине'),
                                               labels, editable=False)
        if not accepted:
            return
        key = keys[labels.index(label)]
    owner = editor.owner
    position = owner.model.resolve_row_positions([key]).get(key)
    if position is None:
        editor.status.setText(tr('Строка пробы отсутствует в проекте.'))
        return
    index = owner.proxy.mapFromSource(owner.model.index(position, 0))
    if not index.isValid():
        editor.status.setText(tr('Проба скрыта фильтром исходной таблицы'))
        return
    owner.table.selectionModel().setCurrentIndex(index, QItemSelectionModel.SelectionFlag.ClearAndSelect
                                                 | QItemSelectionModel.SelectionFlag.Rows)
    owner.desktop.open_table()
    owner.table.scrollTo(index)


def column_notice(editor, notice):
    if notice.startswith('Нет интервалов уровня: '):
        rank, separator, well = notice.split(': ', 1)[1].partition(' · ')
        return (tr('Ствол {0}: нет интервалов уровня {1}', well, geological_label(rank))
                if separator else tr('Нет интервалов уровня: {0}', geological_label(rank)))
    return message(notice)


def show_revision_column(editor, scheme, target, parent):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from ..widgets.figure_canvas import FittedFigureCanvas, FigureNavigation
    from ..widgets.navigation import NavigationToolbar

    well, level, top, bottom = target
    dialog = figure = navigation = None
    background = False
    try:
        options = editor.render_options()
        options.update(comparison_wells=[], correlation={}, hierarchy_levels=[],
                       depth_min=None, depth_max=None, title='')
        if options['y_label']:
            options['y_label'] = tr('Глубина')
        catalog = LithologyCatalog(options['lithologies']) if options['lithologies'] else editor.lithology_catalog
        intervals = [item for item in scheme['intervals'] if item['wellbore'] == well and item['level'] == level]
        unavailable = {item['lithology'] for item in intervals
                       if item.get('lithology') and catalog.resolve(item['lithology']) is None}
        subset = {**scheme, 'intervals': [dict(item, lithology='') if item.get('lithology') in unavailable else item
                                         for item in intervals]}
        label_axes = editor.label_column_axes

        def render():
            column, _ = render_columns(subset, well, level, options=options)
            label_axes(column, subset['depth_unit'], level)
            first = column.axes[0]
            deep, shallow = first.get_ylim()
            padding = max((bottom - top) * .08, (deep - shallow) * .025)
            first.set_ylim(bottom + padding, top - padding)
            for axis in column.axes:
                if bottom > top:
                    axis.axhspan(top, bottom, color='#3879a6', alpha=.18, zorder=20)
                else:
                    axis.axhline(top, color='#3879a6', linestyle='--', zorder=20)
            return column

        large = len(intervals) >= editor.BACKGROUND_PREVIEW_INTERVALS
        figure = Figure() if large else render()
        dialog = QDialog(parent)
        title = tr('Колонка редакции {0} · {1}', scheme['revision'], well)
        dialog.setWindowTitle(title)
        dialog.resize(760, 690)
        layout = QVBoxLayout(dialog)
        if unavailable:
            note = QLabel(tr('Узоры недоступных пород пропущены: {0}', ', '.join(sorted(unavailable))))
            note.setWordWrap(True)
            layout.addWidget(note)
        canvas = FittedFigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, dialog)
        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setIcon(icon('x'))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if large:
            dialog.setModal(True)
            dialog.show()
            screen_dpi = figure._original_dpi
            size = (canvas.width() / screen_dpi, canvas.height() / screen_dpi)
            dpi = screen_dpi * canvas.device_pixel_ratio

            def render_frame(update):
                update(title, 0)
                column = render()
                update(title, 50)
                column.set_size_inches(*size, forward=False)
                column.set_dpi(dpi)
                scene = FigureCanvasAgg(deepcopy(column))
                agg = FigureCanvasAgg(column)
                agg.draw()
                return column, agg.buffer_rgba(), scene

            background = True
            editor._column_render_running = True
            editor.column_preview_timer.stop()
            figure, pixels, scene = run_with_progress(dialog, title, render_frame, cancellable=True)
            canvas.figure = figure
            figure.set_canvas(canvas)
            figure._original_dpi = screen_dpi
            toolbar.reconnect_canvas()
            toolbar.update()
            navigation = FigureNavigation(canvas)
            navigation.scene = scene
            navigation.bind_status(parent.summary)
            canvas.defer_navigation_draws(True)
            canvas.present(pixels)
        dialog.exec()
    except TaskCancelled:
        parent.summary.setText(tr('Подготовка отменена.'))
    except Exception as exc:
        parent.summary.setText(tr('Ошибка: {0}', message(exc)))
    finally:
        if navigation is not None:
            navigation.clear()
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
        if figure is not None:
            figure.clear()
        if background:
            editor._column_render_running = False
            editor.resume_column_preview()
