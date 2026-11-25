"""Export the current table view or the selected samples from the desktop."""

from itertools import islice
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from autopyro.data_io.file_output import atomic_output
from autopyro.data_io.io import export_project_rows_csv, export_project_rows_json, export_project_rows_xlsx

from ..i18n import message, tr
from ..tasks import TaskCancelled, run_with_progress


def export_visible_rows(window):
    path, _ = QFileDialog.getSaveFileName(
        window, tr('Экспорт видимых строк и колонок'), "export.csv", "CSV (*.csv)"
    )
    if path:
        try:
            path = _export_path(window, path, ".csv")
            if path is None:
                return
            model, proxy = window.model, window.proxy
            project, columns = model.project, window.visible_ids()
            count, snapshot = proxy.visible_row_snapshot()
            revision, order_revision = model.data_revision, proxy.row_order_revision
            title = tr('Экспорт')

            def prepare():
                rows = []
                while True:
                    if (window.model is not model or model.data_revision != revision
                            or proxy.row_order_revision != order_revision):
                        raise ValueError(tr('Таблица изменилась. Повторите операцию.'))
                    batch = list(islice(snapshot, 1024))
                    rows.extend(batch)
                    if len(batch) < 1024:
                        return rows
                    yield

            def write(update, rows):
                def before_replace():
                    if update is not None:
                        update(title, 100)
                    if window.model is not model or model.data_revision != revision:
                        raise ValueError(tr('Таблица изменилась. Повторите операцию.'))

                with atomic_output(path, before_replace=before_replace) as temporary:
                    if update is None:
                        export_project_rows_csv(temporary, project, rows, columns)
                    else:
                        export_project_rows_csv(temporary, project, rows, columns,
                                                progress=lambda written, total: update(title, written * 99 // max(1, total)))

            if count >= 10_000:
                run_with_progress(window, title, write, cancellable=True, prepare=prepare())
            else:
                write(None, list(snapshot))
        except TaskCancelled:
            return
        except Exception as exc:
            QMessageBox.warning(window, tr('Экспорт'), message(exc))

def _export_path(window, filename, suffix):
    path = Path(filename)
    if path.suffix:
        if path.suffix.lower() != suffix:
            raise ValueError(tr('Расширение файла должно соответствовать выбранному формату ({0}).', suffix))
        return path
    path = path.with_suffix(suffix)
    if path.exists() and QMessageBox.question(
        window, tr('Экспорт'), tr('Заменить файл «{0}»?', path.name),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
    ) != QMessageBox.StandardButton.Yes:
        return None
    return path

def export_selected_samples(window):
    model = window.model
    count, snapshot = window.sample_selection.export_snapshot()
    if not count:
        QMessageBox.information(window, tr('Экспорт'), tr('Выделите пробы в таблице.'))
        return
    columns = window.visible_ids()
    if not columns:
        QMessageBox.information(window, tr('Экспорт'), tr('Покажите хотя бы одну колонку для экспорта.'))
        return
    revision = model.data_revision
    order_revision = window.proxy.row_order_revision
    formats = {"CSV (*.csv)": ".csv", "JSON (*.json)": ".json", "XLSX (*.xlsx)": ".xlsx"}
    path, selected_format = QFileDialog.getSaveFileName(
        window, tr('Экспорт выделенных проб…'), "selected-samples", ";;".join(formats)
    )
    if not path:
        return
    def check_snapshot():
        if (window.model is not model or model.data_revision != revision
                or window.proxy.row_order_revision != order_revision):
            raise ValueError(tr('Таблица изменилась. Повторите операцию.'))

    def prepare():
        check_snapshot()
        rows = []
        for row in snapshot:
            rows.append(row)
            if len(rows) % 1024 == 0:
                yield
                check_snapshot()
        return rows

    try:
        suffix = formats.get(selected_format, ".csv")
        path = _export_path(window, path, suffix)
        if path is None:
            return
        export_rows = {".json": export_project_rows_json, ".xlsx": export_project_rows_xlsx}.get(
            suffix, export_project_rows_csv)
        project = model.project
        title = tr('Экспорт')

        def report_rows(update, rows):
            for index, row in enumerate(rows):
                if index % 128 == 0:
                    update(title, index * 99 // len(rows))
                yield row

        def write(update, rows):
            def before_replace():
                if update is not None:
                    update(title, 100)
                if window.model is not model or model.data_revision != revision:
                    raise ValueError(tr('Таблица изменилась. Повторите операцию.'))

            with atomic_output(path, before_replace=before_replace) as temporary:
                if update is None:
                    export_rows(temporary, project, rows, columns)
                elif suffix == ".xlsx":
                    export_rows(temporary, project, report_rows(update, rows), columns,
                                packing_checkpoint=lambda: update(title, 99))
                elif export_rows is export_project_rows_csv:
                    export_rows(temporary, project, rows, columns,
                                progress=lambda written, total: update(title, written * 99 // max(1, total)))
                else:
                    export_rows(temporary, project, report_rows(update, rows), columns)

        if count >= 2_000 or (suffix == ".xlsx" and count * len(columns) >= 2_000):
            run_with_progress(window, title, write, cancellable=True, prepare=prepare())
        else:
            check_snapshot()
            write(None, list(snapshot))
    except TaskCancelled:
        return
    except Exception as exc:
        QMessageBox.warning(window, tr('Экспорт'), message(exc))
