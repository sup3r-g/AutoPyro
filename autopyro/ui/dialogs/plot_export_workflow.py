"""Coordinate plot export requests and their desktop dialogs."""

from __future__ import annotations
from ..i18n import language, tr, message
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from autopyro.visualization.plot_data import capture_input, input_token
from autopyro.visualization.plot_export import freeze_palette, with_export_style
from ..plot_preparation import InputCapture
from ..plot_export import PlotExport, plan_batch_export
from autopyro.visualization.plotting import PlotSpec, validate_export_size


def install(dialog):
    dialog.export_capture = InputCapture(dialog)
    dialog.export_capture.ready.connect(dialog.export_input_captured)
    dialog.export_capture.failed.connect(dialog.export_capture_failed)
    dialog.export_request = None
    dialog.export_generation = 0
    dialog.export_service = PlotExport()
    dialog.export_service.progress.connect(dialog.export_progress)
    dialog.export_service.completed.connect(dialog.export_completed)
    dialog.export_service.failed.connect(dialog.export_failed)
    dialog.export_service.busyChanged.connect(dialog.export_busy_changed)


def make_export_jobs(dialog, descriptions, *, scopes=None, source=None):
    scopes = [dialog.scope_ids(spec) for _, _, spec in descriptions] if scopes is None else scopes
    ids = set().union(*(set(scope) for scope in scopes))
    specs = [spec for _, _, spec in descriptions]
    project = dialog.owner.model.project
    token = input_token(project, dialog.owner.model.data_revision, specs)
    source = capture_input(project, ids, specs, token, dialog.captured_input) if source is None else source
    jobs = []
    for (name, path, spec), scope in zip(descriptions, scopes):
        prepared = dialog.prepared
        reusable = (dialog.result is not None and prepared is not None
                    and prepared.source.token == source.token and prepared.scope == frozenset(scope)
                    and spec.to_dict() == dialog.last_config == dialog.spec.to_dict())
        jobs.append({'name': name, 'path': str(path), 'spec': deepcopy(spec), 'ids': list(scope),
                     'source': source, 'prepared': prepared if reusable else None,
                     'palette': freeze_palette(spec.cmap),
                     'revision': dialog.owner.model.data_revision})
    return jobs


def export_busy(dialog):
    return dialog.export_request is not None or dialog.export_service.worker is not None


def start_export(dialog, descriptions, manifest=None, *, report_rows=False, snapshot=False, panel_columns=None):
    if dialog.export_busy():
        return
    descriptions = [(name, path, deepcopy(spec)) for name, path, spec in descriptions]
    for _, _, spec in descriptions:
        spec.render_language = language()
    if panel_columns is not None:
        from autopyro.visualization.plot_panels import panel_layout

        panel_layout(descriptions[0][1] if descriptions else '', [spec for _, _, spec in descriptions],
                     panel_columns)
    elif manifest is None and not snapshot:
        for _, path, spec in descriptions:
            validate_export_size(path, spec)
    specs = [spec for _, _, spec in descriptions]
    scopes = [dialog.scope_ids(None if report_rows else spec) for spec in specs]
    ids = set().union(*(set(scope) for scope in scopes))
    token = input_token(dialog.owner.model.project, dialog.owner.model.data_revision, specs)
    if dialog.captured_input is not None and dialog.captured_input.covers(token, ids, specs):
        jobs = dialog.make_export_jobs(descriptions, scopes=scopes, source=dialog.captured_input)
        dialog.submit_export_jobs(jobs, manifest, snapshot, panel_columns)
        return
    dialog.export_generation += 1
    dialog.export_request = descriptions, scopes, token, manifest, snapshot, panel_columns
    dialog.export_busy_changed(True)
    dialog.export_capture.start(dialog.export_generation, dialog.owner.model.project, ids, specs, token)


def export_input_captured(dialog, generation, source):
    if dialog.closed or generation != dialog.export_generation or dialog.export_request is None:
        return
    descriptions, scopes, token, manifest, snapshot, panel_columns = dialog.export_request
    dialog.export_request = None
    specs = [spec for _, _, spec in descriptions]
    if input_token(dialog.owner.model.project, dialog.owner.model.data_revision, specs) != token:
        dialog.export_busy_changed(False)
        dialog.export_failed(tr('Данные изменились во время копирования. Повторите экспорт.'))
        return
    jobs = dialog.make_export_jobs(descriptions, scopes=scopes, source=source)
    dialog.submit_export_jobs(jobs, manifest, snapshot, panel_columns)


def submit_export_jobs(dialog, jobs, manifest, snapshot, panel_columns):
    if panel_columns is None:
        for job in jobs:
            job['snapshot'] = snapshot
    else:
        jobs = [{'name': Path(jobs[0]['path']).stem, 'path': jobs[0]['path'],
                 'panels': jobs, 'columns': panel_columns}]
    dialog.export_service.start(jobs, manifest)


def export_capture_failed(dialog, generation, error):
    if generation == dialog.export_generation:
        dialog.export_request = None
        dialog.export_busy_changed(False)
        dialog.export_failed(error)


def cancel_export(dialog):
    dialog.export_generation += 1
    dialog.export_capture.cancel()
    dialog.export_request = None
    dialog.export_service.cancel()
    if dialog.export_service.worker is not None:
        dialog.status.setText(tr('Отмена экспорта запрошена…'))
    if dialog.export_service.worker is None:
        dialog.export_busy_changed(False)
        dialog.status.setText(tr('Экспорт отменён.'))


def export_busy_changed(dialog, busy):
    dialog.cancel_export_action.setEnabled(busy)
    dialog.export_button.setEnabled(not busy and not dialog.export_busy() and dialog.result is not None)


def export_progress(dialog, done, total):
    if dialog.closed:
        return
    dialog.status.setText(tr('Экспорт: {0} из {1}', done, total))


def export_completed(dialog, outcome):
    if dialog.closed:
        return
    dialog.status.setText(tr('Экспорт завершён: {0}; ошибок: {1}', len(outcome['plots']), len(outcome['errors']))
                        + (tr(' · отменён') if outcome['cancelled'] else ''))
    if outcome['errors']:
        QMessageBox.warning(dialog, tr('Экспорт'), '\n'.join(outcome['errors']))
    warnings = [f"{plot['name']}: {warning}" for plot in outcome['plots'] for warning in plot.get('warnings', ())]
    if warnings:
        dialog.status.setText(dialog.status.text() + '\n' + '\n'.join(warnings))


def export_failed(dialog, error):
    if dialog.closed:
        return
    dialog.status.setText(tr('Экспорт не выполнен: {0}', message(error)))
    QMessageBox.warning(dialog, tr('Экспорт'), message(error))


def export_plot(dialog):
    if dialog.result is None or dialog.export_busy():
        return
    dialog.save_current_plot_image(dialog.spec, 'plot.png')


def export_track(dialog):
    if dialog.result is None or dialog.spec.kind != 'multi_depth' or dialog.export_busy():
        return
    generation = dialog.generation
    spec = dialog.spec
    columns = {column.id: column.name for column in dialog.owner.model.project.columns}
    options = [f'{index}. {columns[cid]}' for index, cid in enumerate(spec.track_ids, 1)]
    selected, accepted = QInputDialog.getItem(dialog, tr('Экспортировать отдельный трек…'),
                                              tr('Трек'), options, editable=False)
    if not accepted or dialog.closed or generation != dialog.generation or dialog.result is None:
        return
    index = options.index(selected)
    cid = spec.track_ids[index]
    annotations = [item for item in spec.annotations
                   if item.get('track_id') == cid or (index == 0 and not item.get('track_id'))]
    track = replace(spec, x_id=cid, track_ids=[cid], passport_id='',
                    track_axes={key: axis for key, axis in spec.track_axes.items() if key == cid},
                    annotations=annotations)
    dialog.save_current_plot_image(track, 'track.png')


def export_visible_limits(dialog):
    if not dialog.can_export_visible_limits():
        return
    spec = deepcopy(dialog.spec)
    dialog.copy_visible_limits(spec)
    dialog.save_current_plot_image(spec, 'plot.png')


def can_export_visible_limits(dialog):
    return (dialog.result is not None and not dialog.export_busy() and not dialog.spec.interpretation_panels
            and dialog.spec.kind not in ('box', 'violin'))


def save_current_plot_image(dialog, spec, default_path):
    generation = dialog.generation
    path, selected_format = QFileDialog.getSaveFileName(dialog, tr('Экспорт графика'), default_path,
        'PNG (*.png);;TIFF (*.tif);;SVG (*.svg);;PDF (*.pdf)')
    if path and not dialog.closed:
        if generation != dialog.generation or dialog.result is None:
            QMessageBox.information(dialog, tr('Экспорт'), tr('Дождитесь построения графика.'))
            return
        if not Path(path).suffix:
            path += {'TIFF': '.tif', 'SVG': '.svg', 'PDF': '.pdf'}.get(selected_format.split(' ')[0], '.png')
        try:
            dialog.status.setText(tr('Выполняется фоновый экспорт…'))
            job = [(spec.title or 'Plot', path, spec)]
            if dialog.interpretation_owner is None:
                dialog.start_export(job)
            else:
                dialog.start_export(job, report_rows=True)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            dialog.export_failed(str(exc))


def export_snapshot(dialog):
    if dialog.export_busy():
        return
    if dialog.result is None:
        QMessageBox.information(dialog, tr('Снимок графика'), tr('Дождитесь построения графика.'))
        return
    generation = dialog.generation
    path, _ = QFileDialog.getSaveFileName(dialog, tr('Сохранить снимок графика…'),
                                         'plot.aplot.json', 'JSON (*.aplot.json)')
    if not path or dialog.closed:
        return
    if generation != dialog.generation or dialog.result is None:
        QMessageBox.information(dialog, tr('Снимок графика'), tr('Дождитесь построения графика.'))
        return
    if not Path(path).suffix:
        path += '.aplot.json'
    try:
        dialog.status.setText(tr('Выполняется фоновый экспорт…'))
        dialog.start_export([(dialog.spec.title or 'Plot', path, dialog.spec)], snapshot=True,
                          report_rows=dialog.interpretation_owner is not None)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        dialog.export_failed(str(exc))


def replay_snapshot(dialog):
    if dialog.export_busy():
        return
    source, _ = QFileDialog.getOpenFileName(dialog, tr('Открыть снимок графика…'),
                                           '', 'JSON (*.aplot.json *.json)')
    if not source or dialog.closed:
        return
    path, selected_format = QFileDialog.getSaveFileName(dialog, tr('Экспортировать из снимка графика…'), 'plot.png',
        'PNG (*.png);;TIFF (*.tif);;SVG (*.svg);;PDF (*.pdf)')
    if not path or dialog.closed:
        return
    if not Path(path).suffix:
        path += {'TIFF': '.tif', 'SVG': '.svg', 'PDF': '.pdf'}.get(selected_format.split(' ')[0], '.png')
    if Path(source).resolve() == Path(path).resolve():
        QMessageBox.warning(dialog, tr('Снимок графика'), tr('Выберите другой файл для изображения.'))
        return
    dialog.status.setText(tr('Выполняется фоновый экспорт…'))
    dialog.export_service.start([{'name': Path(source).name, 'path': path, 'replay': source}])


def export_selected_plots(dialog, *, panels=False):
    if dialog.closed or dialog.export_busy():
        return
    project = dialog.owner.model.project
    plots = project.view.get('plots', {})
    if panels and len(plots) < 2:
        QMessageBox.information(dialog, tr('Многопанельный рисунок'),
                                tr('Сначала сохраните не менее двух именованных графиков.'))
        return
    if not plots:
        dialog.batch_export()
        return
    from .plot_export_selection import PlotExportSelectionDialog

    source_style = deepcopy(dialog.spec) if dialog.result is not None else None
    selection_dialog = PlotExportSelectionDialog(plots, dialog, panels=panels, can_use_style=source_style is not None)
    try:
        if selection_dialog.exec() and not dialog.closed and project is dialog.owner.model.project:
            style = source_style if selection_dialog.common_style.isChecked() else None
            names = selection_dialog.selected_names()
            if panels:
                dialog.export_panels(names, columns=selection_dialog.panel_columns(), style=style)
            else:
                dialog.batch_export(names=names, style=style)
    finally:
        selection_dialog.deleteLater()


def export_panels(dialog, names, *, columns, style=None):
    if dialog.closed or dialog.export_busy() or len(names) < 2:
        return
    project = dialog.owner.model.project
    path, selected_format = QFileDialog.getSaveFileName(
        dialog, tr('Многопанельный рисунок'), 'panels', 'PNG (*.png);;SVG (*.svg)')
    if not path or dialog.closed or project is not dialog.owner.model.project:
        return
    if not Path(path).suffix:
        path += '.svg' if selected_format == 'SVG (*.svg)' else '.png'
    plots = dialog.owner.model.project.view.get('plots', {})
    try:
        descriptions = [(name, path, with_export_style(PlotSpec.from_dict(plots[name]), style))
                        for name in names]
        dialog.start_export(descriptions, panel_columns=columns)
        dialog.status.setText(tr('Выполняется фоновый экспорт…'))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        dialog.export_failed(str(exc))


def batch_export(dialog, *, names=None, style=None):
    if dialog.closed or dialog.export_busy():
        return
    project = dialog.owner.model.project
    plots = project.view.get("plots", {})
    if not plots:
        QMessageBox.information(
            dialog, tr('Экспорт'), tr('Сначала сохраните именованные графики в проекте')
        )
        return
    directory = QFileDialog.getExistingDirectory(
        dialog, tr('Папка для именованных графиков')
    )
    if not directory or dialog.closed or project is not dialog.owner.model.project:
        return
    extension, ok = QInputDialog.getItem(
        dialog,
        tr('Пакетный экспорт'),
        tr('Формат'),
        ["png", "svg", "pdf", "tif"],
        editable=False,
    )
    if not ok or dialog.closed or project is not dialog.owner.model.project:
        return
    if dialog.export_service.worker is not None:
        return
    try:
        descriptions, manifest = plan_batch_export(plots, directory, extension, names=names, style=style)
        dialog.status.setText(tr('Выполняется фоновый экспорт…'))
        dialog.start_export(descriptions, str(manifest))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        dialog.export_failed(str(exc))
