"""Coordinates of an imported record without technical source metadata."""
from pathlib import PureWindowsPath


def source_location_parts(source):
    path = source.get("path")
    endpoint = source.get("endpoint")
    database = source.get("dbname") or (endpoint.get("database") if isinstance(endpoint, dict) else None)
    if path:
        yield "file", PureWindowsPath(str(path)).name or path
    elif database:
        yield "database", database
    elif source.get("source_name"):
        yield "source", source["source_name"]
    if sheet := source.get("sheet"):
        yield "sheet", sheet
    if table := source.get("table") or source.get("relation"):
        yield "table", table
    row = source.get("row_number")
    if row is None:
        row = source.get("row")
    if row is not None:
        yield "row", row
    elif source.get("source_record_id"):
        yield "record", source["source_record_id"]
