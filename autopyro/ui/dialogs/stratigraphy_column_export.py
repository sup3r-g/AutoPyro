"""Export stratigraphy column images and batches with cancellation."""

from __future__ import annotations

from ..i18n import tr, message
import re
from pathlib import Path
from tempfile import mkdtemp
from PySide6.QtWidgets import QDialog, QFileDialog
from autopyro.data_io.file_output import atomic_output
from autopyro.geology.stratigraphy import render_columns
from ..tasks import TaskCancelled, run_with_progress
from ..widgets.stratigraphy_export import ColumnBatchDialog, ColumnExportDialog


def export_sample_positions(editor, scheme, options, level, selected):
    samples = editor.column_samples(scheme, options, level, selected)
    if not samples:
        raise ValueError(tr('Нет проб с подходящими глубинами в выбранной области.'))
    return samples


def export_column(editor):
    dialog = ColumnExportDialog(editor._export_include_samples, editor)
    path = dialog.selectedFiles()[0] if dialog.exec() == QDialog.DialogCode.Accepted else None
    include_samples = dialog.include_samples.isChecked()
    dialog.deleteLater()
    if not path:
        return
    editor._export_include_samples = include_samples
    try:
        path = Path(path)
        style = editor.render_options()
        scheme = editor.read_render_scheme()
        well, level = editor.column_well.currentData(), editor.level.currentData()
        samples = (editor.export_sample_positions(scheme, style, level, [well, *style['comparison_wells']])
                   if include_samples else None)
        title = tr('Экспорт колонки')

        def save(update):
            def check_cancelled():
                update(title, 0)

            check_cancelled()
            editor.save_column_figure(path, scheme, well, level, style,
                                    samples=samples, check_cancelled=check_cancelled)

        editor._batch_export_running = True
        editor.column_preview_timer.stop()
        try:
            run_with_progress(editor, title, save, cancellable=True)
        finally:
            editor._batch_export_running = False
            editor.resume_column_preview()
        editor.status.setText(tr('Колонка экспортирована: {0}', path))
    except TaskCancelled:
        editor.status.setText(tr('Экспорт отменён.'))
    except Exception as exc:
        editor.status.setText(tr('Ошибка: {0}', message(exc)))


def save_column_figure(editor, path, scheme, well, level, style, *, samples=None, check_cancelled=None):
    figure, _ = render_columns(scheme, well, level, options=style, samples=samples)
    try:
        if check_cancelled is not None:
            check_cancelled()
        editor.label_column_axes(figure, scheme['depth_unit'], level)
        with atomic_output(path) as temporary:
            figure.savefig(temporary, dpi=style['dpi'], transparent=style['transparent'])
            if check_cancelled is not None:
                check_cancelled()
    finally:
        figure.clear()


def export_column_batch(editor):
    try:
        scheme = editor.read_render_scheme()
        level = editor.level.currentData()
        style = editor.style_values()
        levels = {level, *style['hierarchy_levels']}
        wells = sorted({item['wellbore'] for item in scheme['intervals'] if item['level'] in levels})
        if not wells:
            raise ValueError(tr('Для выбранных уровней нет стволов с интервалами.'))
        defaults = {editor.column_well.currentData(), *style['comparison_wells']} & set(wells)
        templates = editor.style_templates.templates()
        dialog = ColumnBatchDialog(wells, defaults or {wells[0]}, templates, editor)
        if dialog.exec() != dialog.DialogCode.Accepted:
            dialog.deleteLater()
            return
        selected = dialog.values()
        individual = dialog.individual.isChecked()
        comparison = dialog.includes_comparison()
        include_samples = dialog.include_samples.isChecked()
        extension = dialog.format.currentText().lower()
        template = templates.get(dialog.template.currentData())
        dialog.deleteLater()
        if not selected or not (individual or comparison):
            return
        primary = editor.column_well.currentData()
        if primary in selected:
            selected.remove(primary)
            selected.insert(0, primary)
        style = editor.render_options(template)
        if style['correlation']:
            anchors = style['correlation']['anchors']
            by_id = {item['id']: item['wellbore'] for item in scheme['intervals']}
            for well in selected:
                if by_id.get(anchors.get(well)) != well:
                    raise ValueError(tr('Выберите опорный интервал для ствола: {0}', well))
        samples = editor.export_sample_positions(scheme, style, level, selected) if include_samples else None
        directory = QFileDialog.getExistingDirectory(editor, tr('Папка для комплекта колонок'))
        if directory:
            editor.write_column_batch(directory, extension, scheme, level, style, selected, individual, comparison,
                                    samples=samples)
    except TaskCancelled:
        editor.status.setText(tr('Экспорт отменён.'))
    except Exception as exc:
        editor.status.setText(tr('Ошибка: {0}', message(exc)))


def write_column_batch(editor, directory, extension, scheme, level, style, selected, individual, comparison,
                       *, samples=None):
    groups = {well: [] for well in selected}
    for item in scheme['intervals']:
        if item['wellbore'] in groups:
            groups[item['wellbore']].append(item)
    jobs = []
    if individual:
        for number, well in enumerate(selected, 1):
            name = re.sub(r'[\x00-\x1f<>:"/\\|?*]', '_', well)[:60].strip(' .')
            jobs.append((f'{number:03d}-{name or "well"}.{extension}', [well]))
    if comparison:
        jobs.append((f'comparison.{extension}', selected))
    output = Path(mkdtemp(prefix='columns-', dir=directory))
    completed = 0
    error = None
    cancelled = False

    def write_files(update):
        nonlocal completed
        for filename, wells in jobs:
            stage = tr('Экспорт: {0}', filename)
            percent = completed * 100 // len(jobs)

            def check_cancelled():
                update(stage, percent)

            check_cancelled()
            subset = {**scheme, 'intervals': [item for well in wells for item in groups[well]]}
            options = {**style, 'comparison_wells': wells[1:]}
            try:
                editor.save_column_figure(output / filename, subset, wells[0], level, options,
                                        samples=samples, check_cancelled=check_cancelled)
            except TaskCancelled:
                raise
            except Exception as exc:
                return tr('Ошибка экспорта файла {0}: {1}', filename, message(exc))
            completed += 1
        return None

    editor._batch_export_running = True
    editor.column_preview_timer.stop()
    try:
        error = run_with_progress(editor, tr('Экспорт комплекта колонок'), write_files, cancellable=True)
    except TaskCancelled:
        cancelled = True
    finally:
        editor._batch_export_running = False
        editor.resume_column_preview()
        if not completed:
            output.rmdir()
    summary = (tr('Сохранено файлов: {0} из {1}. Папка: {2}', completed, len(jobs), output)
               if completed else tr('Файлы комплекта не сохранены.'))
    prefix = error or (tr('Экспорт отменён.') if cancelled else '')
    editor.status.setText(f'{prefix} {summary}' if prefix else summary)
