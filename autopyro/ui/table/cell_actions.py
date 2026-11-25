"""Clipboard and batch editing actions for the project table."""

import csv
import io

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QDialog, QInputDialog, QMessageBox

from ..i18n import message, tr
from ..tasks import TaskCancelled, run_with_progress
from .batch_edit import parse_cell_change, prepare_batch_changes, read_clipboard_cells
from .clipboard import read_text as read_clipboard_text, write_text as write_clipboard_text
from .commands import EditCommand


def copy_cells(window):
    selection = window.table.selectionModel().selection()
    if not selection:
        return
    header = window.table.horizontalHeader()
    visible_columns = [header.logicalIndex(i) for i in range(header.count())
                       if not header.isSectionHidden(header.logicalIndex(i))]
    visible_positions = {column: position for position, column in enumerate(visible_columns)}
    first_row, last_row = window.proxy.rowCount(), -1
    first_column, last_column = len(visible_columns), -1
    cell_count = 0
    for area in selection:
        positions = [visible_positions[column] for column in range(area.left(), area.right() + 1)
                     if column in visible_positions]
        if not positions:
            continue
        first_row, last_row = min(first_row, area.top()), max(last_row, area.bottom())
        first_column = min(first_column, min(positions))
        last_column = max(last_column, max(positions))
        cell_count += area.height() * len(positions)
    if not cell_count:
        return
    # Qt selection ranges are disjoint. A smaller area means the rectangle has gaps.
    if cell_count != (last_row - first_row + 1) * (last_column - first_column + 1):
        QMessageBox.information(window, tr('Копировать'), tr('Для копирования выделите смежный прямоугольник ячеек.'))
        return
    model, proxy = window.model, window.proxy
    project = model.project
    revision, order_revision = model.data_revision, proxy.row_order_revision
    columns = visible_columns[first_column:last_column + 1]
    column_ids = [project.columns[column].id for column in columns]

    def check_source():
        if (window.model is not model or model.data_revision != revision
                or proxy.row_order_revision != order_revision):
            raise ValueError(tr('Таблица изменилась. Повторите операцию.'))

    def prepare():
        stream = io.StringIO()
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        batch_rows = max(1, 1024 // len(columns))
        check_source()
        for number, row in enumerate(range(first_row, last_row + 1), 1):
            source_row = proxy.mapToSource(proxy.index(row, columns[0])).row()
            values = project.rows[source_row].values
            writer.writerow([values[column_id] for column_id in column_ids])
            if number % batch_rows == 0:
                yield
                check_source()
        return stream.getvalue()

    try:
        steps = prepare()
        if cell_count >= 10_000:
            text = run_with_progress(window, tr('Копировать'), lambda update, result: result,
                                     cancellable=True, prepare=steps)
        else:
            while True:
                try:
                    next(steps)
                except StopIteration as done:
                    text = done.value
                    break
        write_clipboard_text(window, text, check_source=check_source)
    except TaskCancelled:
        return
    except (ValueError, OSError, MemoryError) as exc:
        QMessageBox.warning(window, tr('Копировать'), message(exc))

def paste_cells(window):
    start = window.table.currentIndex()
    if not start.isValid():
        return
    try:
        header = window.table.horizontalHeader()
        visual = header.visualIndex(start.column())
        available = [
            header.logicalIndex(i)
            for i in range(visual, header.count())
            if not window.table.isColumnHidden(header.logicalIndex(i))
        ]
        model, proxy = window.model, window.proxy
        revision, order_revision = model.data_revision, proxy.row_order_revision
        start_row = start.row()
        remaining_rows = proxy.rowCount() - start_row
        clipboard_text = read_clipboard_text(window)
        if (window.model is not model or model.data_revision != revision
                or proxy.row_order_revision != order_revision):
            raise ValueError(tr('Таблица изменилась. Повторите операцию.'))
        title = tr('Вставка')

        def read_cells(update=None):
            return read_clipboard_cells(clipboard_text, remaining_rows, len(available), title, update)

        if len(clipboard_text) >= 64 * 1024:
            cells, cell_count = run_with_progress(window, title, read_cells, cancellable=True)
        else:
            cells, cell_count = read_cells()
        if (window.model is not model or model.data_revision != revision
                or proxy.row_order_revision != order_revision):
            raise ValueError(tr('Таблица изменилась. Повторите операцию.'))
        targets = (
            (proxy.mapToSource(proxy.index(start_row + r, available[offset])), value)
            for r, row in enumerate(cells) for offset, value in enumerate(row)
        )
        window.preview_batch_edit(title, targets, cell_count=cell_count)
    except TaskCancelled:
        return
    except (ValueError, TypeError) as exc:
        QMessageBox.warning(window, tr('Вставка отменена'), message(exc))

def parsed_change(window, index, value):
    return parse_cell_change(window.model.project, index, value)

def fill_cells(window):
    model, proxy = window.model, window.proxy
    revision, order_revision = model.data_revision, proxy.row_order_revision
    visible_columns = {
        column for column in range(window.proxy.columnCount())
        if not window.table.isColumnHidden(column)
    }
    ranges = []
    cell_count = 0
    for area in window.table.selectionModel().selection():
        columns = tuple(column for column in range(area.left(), area.right() + 1)
                        if column in visible_columns)
        if not columns:
            continue
        rows = range(area.top(), area.bottom() + 1)
        ranges.append((rows, columns))
        cell_count += len(rows) * len(columns)
    if not cell_count:
        return

    def source_row(row, column):
        return proxy.mapToSource(proxy.index(row, column)).row()

    snapshot = None

    def remember_rows(*_):
        nonlocal snapshot
        if snapshot is None and proxy.row_order_revision == order_revision:
            snapshot = [([source_row(row, columns[0]) for row in rows], columns)
                        for rows, columns in ranges]

    # Resolve the original selection only if its order changes while the dialog is open.
    connections = [signal.connect(remember_rows) for signal in (
        proxy.layoutAboutToBeChanged, proxy.rowsAboutToBeRemoved, proxy.rowsAboutToBeInserted,
    )]
    try:
        value, ok = QInputDialog.getText(
            window, tr('Заполнение'), tr('Значение для {0} ячеек (пустое — очистить)', cell_count)
        )
    finally:
        for connection in connections:
            QObject.disconnect(connection)
    if not ok:
        return
    try:
        if (window.model is not model or model.data_revision != revision
                or (snapshot is None and proxy.row_order_revision != order_revision)):
            raise ValueError(tr('Таблица изменилась. Повторите операцию.'))

        def targets():
            for rows, columns in (ranges if snapshot is None else snapshot):
                for row in rows:
                    source = source_row(row, columns[0]) if snapshot is None else row
                    for column in columns:
                        yield model.index(source, column), value

        window.preview_batch_edit(tr('Заполнение'), targets(), cell_count=cell_count)
    except ValueError as exc:
        QMessageBox.warning(window, tr('Заполнение отменено'), message(exc))

def preview_batch_edit(window, title, targets, *, cell_count=0):
    from ..dialogs.batch_edit import BatchEditPreview

    model, revision = window.model, window.model.data_revision
    order_revision = window.proxy.row_order_revision

    def check_targets():
        if (window.model is not model or model.data_revision != revision
                or window.proxy.row_order_revision != order_revision):
            raise ValueError(tr('Таблица изменилась. Повторите операцию.'))

    steps = prepare_batch_changes(model.project, targets, window.parsed_change, check_targets)
    if cell_count >= 10_000:
        try:
            changes, positions = run_with_progress(
                window, tr('Подготовка предпросмотра…'), lambda update, result: result,
                cancellable=True, prepare=steps,
            )
        except TaskCancelled:
            return
        check_targets()
    else:
        while True:
            try:
                next(steps)
            except StopIteration as done:
                changes, positions = done.value
                break
    if not changes:
        return

    command = EditCommand(model, changes, row_positions=positions, validated=True)
    summary = tr('Будет изменено ячеек: {0}. Строк: {1}.', len(changes), len(positions))
    dialog = BatchEditPreview(title, summary, command, window)
    try:
        answer = dialog.exec()
    finally:
        dialog.deleteLater()
    if answer != QDialog.DialogCode.Accepted:
        return
    if window.model is not model or model.data_revision != revision:
        raise ValueError(tr('Таблица изменилась. Повторите операцию.'))
    model.undo.push(command)
