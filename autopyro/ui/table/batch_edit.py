"""Parse clipboard data and validate edits before creating an undo command."""

import csv
import io

from autopyro.model.measurements import derived, parse_result, parsed_table_value

from ..i18n import message, tr


def read_clipboard_cells(clipboard_text, remaining_rows, column_count, title, update=None):
    cells = []
    cell_count = 0
    stream = io.StringIO(clipboard_text, newline="")
    reader = csv.reader(stream, delimiter="\t", strict=True)
    try:
        if update is not None:
            update(title, 0)
        for row in reader:
            # CSV represents a blank line as no fields. In a pasted table it
            # occupies one empty cell, including a blank cell copied from Excel.
            if not row:
                row = [""]
            if len(cells) >= remaining_rows or len(row) > column_count:
                raise ValueError(tr('Вставка выходит за границы таблицы; изменения не применены'))
            cells.append(row)
            cell_count += len(row)
            if update is not None and len(cells) % 128 == 0:
                update(title, stream.tell() * 99 // len(clipboard_text))
    except csv.Error as exc:
        raise ValueError(tr('Не удалось прочитать табличные данные из буфера обмена.')) from exc
    if update is not None:
        update(title, 100)
    return cells, cell_count


def parse_cell_change(project, index, value):
    column = project.columns[index.column()]
    if derived(column):
        raise ValueError(tr('Производные результаты доступны только для чтения'))
    value = (
        parsed_table_value(parse_result(value, column.kind))
        if column.kind != "text"
        else (value if value else None)
    )
    return project.rows[index.row()].id, column.id, value


def prepare_batch_changes(project, targets, parse_change, check_targets):
    """Yield between groups of edits so Qt can process cancellation and table changes."""
    changes, positions = [], {}
    errors, invalid_count = [], 0
    check_targets()
    for number, (index, value) in enumerate(targets, 1):
        try:
            change = parse_change(index, value)
        except (ValueError, TypeError) as exc:
            invalid_count += 1
            if len(errors) < 5:
                column = project.columns[index.column()].name
                errors.append(tr('Строка источника {0}, {1}: {2}', index.row() + 1, column, message(exc)))
        else:
            rid, cid, new = change
            old = project.rows[index.row()].values[cid]
            if old != new:
                changes.append((rid, cid, old, new))
                positions[rid] = index.row()
        if number % 1024 == 0:
            yield
            check_targets()
    if invalid_count:
        raise ValueError(tr('Ячеек с ошибками: {0}', invalid_count) + '\n' + '\n'.join(errors))
    return changes, positions
