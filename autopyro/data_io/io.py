"""File previews and read-only source snapshots; no scientific names are merged."""

from __future__ import annotations

import csv
import hashlib
import json
from contextlib import suppress
from collections.abc import Sized
from datetime import date, datetime, time, timedelta, timezone
from io import BufferedWriter
from itertools import islice
from math import isfinite
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from autopyro.model.core import Column, Project, Row, timestamp
from autopyro.model.field_types import FIELD_KINDS, TYPED_KINDS, format_duration, parse_typed_value, typed_kind_hint
from autopyro.model.measurements import derived


def text_value(value):
    if value is None:
        return None
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return format_duration(value.total_seconds())
    return str(value)


def excel_sheet_names(path):
    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        return workbook.sheet_names


def read_excel_frame(workbook, sheet=0, *, skiprows=0, nrows=None):
    """Read untyped Excel values, including formulas and errors, into a DataFrame."""
    worksheet = workbook.book[sheet] if isinstance(sheet, str) else workbook.book.worksheets[sheet]
    worksheet.reset_dimensions()

    def literal(cell):
        value = cell.value
        # Preserve Excel's fixed-width integer identifiers through type inference.
        if type(value) is int and value >= 0:
            mask = cell.number_format or ""
            if len(mask) > 1 and not mask.strip("0"):
                return str(value).zfill(len(mask))
        return None if value == "" else value

    last = skiprows + nrows if nrows is not None else None
    records = ([literal(cell) for cell in row] for row in
               worksheet.iter_rows(min_row=skiprows + 1, max_row=last))
    return pd.DataFrame(records, dtype=object)


def read_file(path, *, sheet=None, delimiter=";", encoding="utf-8-sig", header_rows=1,
              header_row=1, data_end_row=None, first_column=1, last_column=None):
    """Return literal headers, rows, and physical row numbers. Never evaluate Excel formulas."""
    path = Path(path)
    if header_row < 1 or header_rows not in (1, 2) or first_column < 1:
        raise ValueError("Некорректное начало диапазона или число строк заголовка")
    if data_end_row is not None and data_end_row < header_row + header_rows - 1:
        raise ValueError("Конец диапазона находится выше заголовка")
    if last_column is not None and last_column < first_column:
        raise ValueError("Последняя колонка находится левее первой")
    prefix_group = ""
    if path.suffix.lower() == ".xlsx":
        if data_end_row is not None and data_end_row > 1048576:
            raise ValueError("Диапазон выходит за границы строк Excel")
        with pd.ExcelFile(path, engine="openpyxl", engine_kwargs={"data_only": False}) as workbook:
            # Styled empty cells may extend the declared range beyond the data.
            worksheet = workbook.book[sheet] if sheet else workbook.book.worksheets[0]
            max_row, max_column = worksheet.max_row, worksheet.max_column
            frame = read_excel_frame(
                workbook, sheet or 0, skiprows=header_row - 1,
                nrows=(data_end_row - header_row + 1) if data_end_row is not None else None,
            )
            if max_row is None:
                max_row = header_row - 1 + len(frame)
            if max_column is None:
                max_column = len(frame.columns)
            if header_row + header_rows - 1 > max_row:
                raise ValueError("Нет выбранных строк заголовка")
            end_column = last_column or max_column
            if first_column > max_column or end_column > 16384:
                raise ValueError("Выбранная колонка отсутствует или выходит за границы Excel")
            frame = frame.reindex(index=range(max(header_rows, len(frame))), columns=range(end_column))
            frame = frame.map(text_value).astype(object).where(frame.notna(), None)
            frame = frame.replace("", None)
            frame.index = range(header_row, header_row + len(frame))
    else:
        with path.open(encoding=encoding, newline="") as stream:
            reader = csv.reader(stream, delimiter=delimiter)
            raw, source_numbers, previous_end = [], [], 0
            for record_number, values in enumerate(reader, start=1):
                start_line, previous_end = previous_end + 1, reader.line_num
                if record_number < header_row:
                    continue
                if data_end_row is not None and record_number > data_end_row:
                    break
                raw.append(values)
                source_numbers.append(start_line)
        if raw:
            width = len(raw[0])
            if first_column > width or last_column is not None and last_column > width:
                raise ValueError("Диапазон колонок выходит за заголовок CSV")
            if last_column is None and any(len(row) > width for row in raw):
                raise ValueError("В источнике больше значений, чем заголовков; укажите диапазон или исправьте файл")
        frame = pd.DataFrame(raw, index=source_numbers, dtype=object)
        if raw:
            frame = frame.reindex(columns=range(width))
    if len(frame) < header_rows:
        raise ValueError("Нет выбранных строк заголовка")
    if header_rows == 2 and first_column > 1:
        prefix = frame.iloc[0, :first_column - 1].replace("", None).dropna()
        if not prefix.empty:
            prefix_group = prefix.iloc[-1]
    frame = frame.iloc[:, first_column - 1:last_column]
    width = len(frame.columns)
    if not width:
        raise ValueError("Пустой заголовок")
    fallback = pd.Series([f"Колонка {i + 1}" for i in range(width)], index=frame.columns)
    labels = frame.iloc[header_rows - 1].replace("", None).fillna(fallback)
    if header_rows == 2:
        groups = frame.iloc[0].replace("", None).ffill().fillna(prefix_group)
        labels = groups + " / " + labels
    data = frame.iloc[header_rows:]
    data = data.loc[(data.notna() & data.ne("")).any(axis=1)]
    data = data.astype(object).where(data.notna(), None)
    return labels.tolist(), data.values.tolist(), data.index.tolist()


def infer_kind(values, decimal_separator="auto"):
    seen_number = False
    typed_kind = None
    for value in values:
        if value is None or not str(value).strip():
            continue
        # Preserve identifiers with leading zeros as text by default.
        if isinstance(value, str) and len(value) > 1 and value[0] == "0" and value.isdigit():
            return "text"
        if not seen_number and typed_kind is None:
            typed_kind = typed_kind_hint(value)
        try:
            if typed_kind is not None:
                parse_typed_value(value, typed_kind)
                continue
            parsed = parse_import_result(value, decimal_separator)
        except (ValueError, TypeError):
            return "text"
        if not isinstance(parsed, (int, float)):
            return "text"
        seen_number = True
    return typed_kind or ("number" if seen_number else "text")


class ImportValidationError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(f"Ошибок импорта: {len(errors)}. " + "\n".join(
            f"Строка {e['row']}, {e['column']}: {e['value']!r}. {e['message']}" for e in errors[:20]))


def parse_import_result(value, decimal_separator="auto"):
    from autopyro.model.measurements import parse_result, parsed_table_value
    if decimal_separator not in ("auto", ".", ","):
        raise ValueError("Неизвестный десятичный знак")
    result = parse_result(value, "number")
    if decimal_separator != "auto" and isinstance(value, str) and result["qualifier"] not in ("missing", "non_detect", "unparsed"):
        forbidden = "," if decimal_separator == "." else "."
        if forbidden in value:
            raise ValueError(f"Значение не соответствует десятичному знаку «{decimal_separator}»; разделители тысяч не предполагаются")
    return parsed_table_value(result)


def _repeated_import_values(rows, numeric_indices, decimal_separator):
    """Preparse frequent literal strings from a bounded sample of a large import."""
    if not isinstance(rows, (list, tuple)) or len(rows) < 256:
        return {}
    sample = rows[:256]
    repeated = {}
    for index in numeric_indices:
        values = [row[index] for row in sample if index < len(row) and type(row[index]) is str]
        counts = pd.Series(values, dtype=object).value_counts(sort=False)
        readings = {}
        for literal in counts.index[counts >= 4]:
            try:
                readings[literal] = (parse_import_result(literal, decimal_separator), None)
            except (ValueError, TypeError) as exc:
                readings[literal] = (None, str(exc))
        if readings:
            repeated[index] = readings
    return repeated


def build_project(headers, rows, numbers, selections, source, *, decimal_separator="auto"):
    """selections: (source column index, display name, kind, unit)."""
    if not selections:
        raise ValueError("Выберите хотя бы одну колонку")
    names = [name.strip() for _, name, _, _ in selections]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("Названия колонок должны быть непустыми и уникальными")
    if any(kind not in FIELD_KINDS or not 0 <= i < len(headers) for i, _, kind, _ in selections):
        raise ValueError("Неизвестный тип или отсутствующая исходная колонка")
    columns = [
        Column(
            name.strip(),
            kind,
            unit=unit.strip(),
            metadata={"source_header": headers[i], "source_index": i + source.get("first_column", 1) - 1},
        )
        for i, name, kind, unit in selections
    ]
    result = Project(columns=columns, source=source)
    numeric_indices = dict.fromkeys(i for i, _, kind, _ in selections if kind == "number")
    repeated = _repeated_import_values(rows, numeric_indices, decimal_separator)
    fields = [(column, i, repeated.get(i)) for column, (i, _, _, _) in zip(columns, selections, strict=True)]
    errors = []
    for values, line in zip(rows, numbers, strict=True):
        parsed, original = {}, {}
        for column, i, repeated_values in fields:
            value = values[i]
            if isinstance(value, timedelta):
                value = format_duration(value.total_seconds())
            if isinstance(value, (date, time)):
                value = value.isoformat()
            original[column.id] = value
            try:
                if column.kind == "number":
                    reading = repeated_values.get(value) if repeated_values is not None and type(value) is str else None
                    if reading is None:
                        parsed[column.id] = parse_import_result(value, decimal_separator)
                    else:
                        number, error = reading
                        if error is not None:
                            raise ValueError(error)
                        parsed[column.id] = number
                elif column.kind in TYPED_KINDS:
                    parsed[column.id] = parse_typed_value(value, column.kind)
                else:
                    parsed[column.id] = value if value != "" else None
            except (ValueError, TypeError) as exc:
                errors.append({"row": line, "column": column.name, "value": value, "message": str(exc)})
        result.rows.append(
            Row(parsed, source={"row": line, **source}, original=original)
        )
    if errors:
        raise ImportValidationError(errors)
    result.event("import", rows=len(result.rows), source=source)
    return result


def file_source(path, **settings):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(Path(path).resolve()),
        "sha256": digest.hexdigest(),
        "imported_at": timestamp(),
        **settings,
    }


def export_table_csv(path, headers, rows, *, progress=None, row_count=None):
    """Write rows in batches. Progress receives (written, total), with total
    set to None for unsized input until the end of the stream.
    """
    headers = list(headers)
    if progress is not None and row_count is None and isinstance(rows, Sized):
        row_count = len(rows)
    values = iter(rows)
    written = 0
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        while True:
            if progress is not None:
                progress(written, row_count)
            batch = list(islice(values, 1000))
            if not batch and written:
                if progress is not None and row_count is None:
                    progress(written, written)
                break
            frame = pd.DataFrame(batch, columns=headers, dtype=object)
            export_frame_csv(stream, frame, include_header=written == 0)
            written += len(batch)
            if len(batch) < 1000:
                if progress is not None and (batch or row_count is None):
                    progress(written, written if row_count is None else row_count)
                break


def _is_spreadsheet_formula(value):
    return isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))


def spreadsheet_literal(value):
    """Keep formula-like strings literal when pasted into a spreadsheet."""
    return "'" + value if _is_spreadsheet_formula(value) else value


def export_frame_csv(path, frame, *, progress=None, include_header=True):
    """Write a prepared table without changing its cells or column order."""
    frame = frame.copy(deep=False)
    for position, (_, values) in enumerate(frame.items()):
        if progress:
            progress(0, len(frame))
        if not (pd.api.types.is_object_dtype(values.dtype) or pd.api.types.is_string_dtype(values.dtype)):
            continue
        formulas = values.map(_is_spreadsheet_formula)
        if formulas.any():
            protected = values.copy()
            protected.loc[formulas] = "'" + values.loc[formulas]
            frame.isetitem(position, protected)
    headers = [spreadsheet_literal(header) for header in frame.columns] if include_header else False
    if progress is None:
        frame.to_csv(path, sep=";", index=False, header=headers, encoding="utf-8-sig", lineterminator="\r\n")
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        for start in range(0, max(1, len(frame)), 1000):
            progress(start, len(frame))
            frame.iloc[start:start + 1000].to_csv(
                stream, sep=";", index=False, header=headers if start == 0 else False, lineterminator="\r\n"
            )
        progress(len(frame), len(frame))


def export_csv(path, project, row_ids, column_ids, *, progress=None):
    rows = project.row_map()
    export_project_rows_csv(path, project, (rows[rid] for rid in row_ids), column_ids,
                            progress=progress, row_count=len(row_ids) if isinstance(row_ids, Sized) else None)


def export_project_rows_csv(path, project, rows, column_ids, *, progress=None, row_count=None):
    """Export resolved rows in order, formatting only one batch at a time."""
    columns = {c.id: c for c in project.columns}
    column_ids = list(column_ids)
    headers = [columns[c].name + (f" [{columns[c].unit}]" if columns[c].unit else "") for c in column_ids]
    if progress is not None and row_count is None and isinstance(rows, Sized):
        row_count = len(rows)
    export_table_csv(path, headers, ([row.values[cid] for cid in column_ids] for row in rows),
                     progress=progress, row_count=row_count)


def export_project_rows_json(path, project, rows, column_ids):
    """Stream an ordered selection with its active measurements and provenance."""
    columns = {column.id: column for column in project.columns}
    selected = [columns[cid] for cid in column_ids]
    header = {
        "format": "autopyro.table_selection",
        "version": 1,
        "project_id": project.id,
        "columns": [
            {"id": column.id, "name": column.name, "kind": column.kind,
             "unit": column.unit, "derived": bool(derived(column))}
            for column in selected
        ],
    }
    column_ids = [column.id for column in selected]
    measured_ids = [entry["id"] for entry in header["columns"] if not entry["derived"]]
    encode = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encode(header)[:-1] + ',\n"rows": [')
        separator = "\n"
        for row, measurements in _selection_measurements(project, rows, measured_ids):
            stream.write(separator + encode({
                "id": row.id,
                "source": row.source,
                "values": {cid: row.values[cid] for cid in column_ids},
                "measurements": measurements,
            }))
            separator = ",\n"
        stream.write("\n]}\n")


def _selection_measurements(project, rows, measured_ids):
    store = project.science
    for row in rows:
        measurements = {}
        working = store.working.get(row.id, {})
        for cid in measured_ids:
            measurement = store.measurements.get(working.get(cid))
            if measurement is None:
                continue
            analysis = store.analyses.get(measurement.analysis_id)
            source_id = measurement.source_id or (analysis.source_id if analysis else None)
            source = store.sources.get(source_id)
            measurements[cid] = {
                "id": measurement.id,
                "revision": measurement.revision,
                "analysis_id": measurement.analysis_id,
                "method": analysis.method if analysis else "",
                "laboratory": analysis.laboratory if analysis else "",
                "raw_value": measurement.raw_value,
                "value": measurement.value,
                "qualifier": measurement.qualifier,
                "unit": measurement.unit,
                "basis": measurement.basis,
                "limit": measurement.limit,
                "comparator": measurement.comparator,
                "lod": measurement.lod,
                "loq": measurement.loq,
                "source_id": source_id,
                "source": source.provenance if source is not None else None,
            }
        yield row, measurements


_XLSX_MAX_ROWS = 1_048_576
_XLSX_MAX_COLUMNS = 16_384
_XLSX_MAX_TEXT = 32_767


class _SelectionSheet:
    def __init__(self, workbook, name, headers):
        from openpyxl.cell import WriteOnlyCell
        from openpyxl.cell.cell import ERROR_CODES

        self.workbook, self.name, self.headers = workbook, name, headers
        self.cell_factory = WriteOnlyCell
        self.error_codes = frozenset(ERROR_CODES)
        self.part = 0
        self._new_part()

    def _new_part(self):
        self.part += 1
        title = self.name if self.part == 1 else f"{self.name} {self.part}"
        self.sheet = self.workbook.create_sheet(title)
        self.sheet.append([self._cell(value) for value in self.headers])
        self.row_count = 1

    def append(self, values):
        if self.row_count == _XLSX_MAX_ROWS:
            self._new_part()
        self.sheet.append([self._cell(value) for value in values])
        self.row_count += 1

    def _cell(self, value):
        if isinstance(value, int) and abs(value) >= 10**15:
            value = str(value)
        elif isinstance(value, float):
            if not isfinite(value):
                raise ValueError("Excel export cannot represent non-finite numbers.")
            if float(format(value, ".15g")) != value:
                value = str(value)
        if isinstance(value, str):
            if len(value) > _XLSX_MAX_TEXT // 2 and len(value.encode("utf-16-le")) // 2 > _XLSX_MAX_TEXT:
                raise ValueError("Excel export cannot store text longer than 32,767 characters in one cell.")
            if value.startswith("=") or value in self.error_codes:
                cell = self.cell_factory(self.sheet, value=value)
                cell.data_type = "s"
                return cell
        return value


class _CheckedArchiveStream(BufferedWriter):
    def __init__(self, raw, checkpoint):
        super().__init__(raw)
        self.checkpoint = checkpoint

    def write(self, data):
        if self.checkpoint is not None:
            try:
                self.checkpoint()
            except Exception:
                # Let ZipFile close its partial archive while the original error propagates.
                self.checkpoint = None
                raise
        return super().write(data)


def export_project_rows_xlsx(path, project, rows, column_ids, *, packing_checkpoint=None):
    """Write selected values and their provenance as linked, streaming Excel sheets."""
    from openpyxl import Workbook

    columns = {column.id: column for column in project.columns}
    selected = [columns[cid] for cid in column_ids]
    if len(selected) + 1 > _XLSX_MAX_COLUMNS:
        raise ValueError("Too many columns for Excel. Hide some columns before exporting.")
    workbook = Workbook(write_only=True)
    workbook.properties.identifier = project.id
    try:
        _write_selection_sheets(workbook, project, rows, selected)
        if packing_checkpoint is None:
            workbook.save(path)
        else:
            from openpyxl.writer.excel import ExcelWriter

            workbook.properties.modified = datetime.now(timezone.utc).replace(tzinfo=None)
            with Path(path).open("wb", buffering=0) as raw, _CheckedArchiveStream(raw, packing_checkpoint) as stream:
                with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
                    ExcelWriter(workbook, archive).write_data()
    except Exception:
        # A failed write-only export may still own openpyxl's temporary worksheet XML.
        for sheet in workbook.worksheets:
            if sheet._writer is not None:
                with suppress(Exception):
                    sheet.close()
                with suppress(OSError, ValueError):
                    sheet._writer.cleanup()
        raise


def _write_selection_sheets(workbook, project, rows, selected):
    samples = _SelectionSheet(workbook, "Samples", ["sample_id", *(
        column.name + (f" [{column.unit}]" if column.unit else "") for column in selected)])
    schema = _SelectionSheet(workbook, "Columns", ("column_id", "name", "kind", "unit", "derived"))
    sources = _SelectionSheet(workbook, "Sources", ("sample_id", "source_json"))
    fields = ("id", "revision", "analysis_id", "method", "laboratory", "raw_value", "value",
              "qualifier", "unit", "basis", "limit", "comparator", "lod", "loq", "source_id")
    measurements = _SelectionSheet(workbook, "Measurements", ("sample_id", "column_id", *fields, "source_json"))
    encode = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode
    measured_ids = []
    for column in selected:
        is_derived = bool(derived(column))
        schema.append((column.id, column.name, column.kind, column.unit, is_derived))
        if not is_derived:
            measured_ids.append(column.id)
    for row, active in _selection_measurements(project, rows, measured_ids):
        samples.append((row.id, *(row.values[column.id] for column in selected)))
        sources.append((row.id, encode(row.source)))
        for cid, record in active.items():
            measurements.append((row.id, cid, *(record[key] for key in fields), encode(record["source"])))


# Compatibility exports; implementations live in the database package.
from autopyro.database.readers import read_postgres_snapshot, read_sqlite
