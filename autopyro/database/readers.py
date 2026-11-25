"""Read-only SQLite and PostgreSQL source snapshots."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd


def read_sqlite(path, table, limit=100000):
    from autopyro.data_io.io import text_value

    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        available = [
            r[0]
            for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
            )
        ]
        if table not in available:
            raise ValueError("Таблица отсутствует в источнике")
        quoted = '"' + table.replace('"', '""') + '"'
        cursor = connection.execute(f"SELECT * FROM {quoted} LIMIT ?", (limit + 1,))
        # read_sql_query infers floats before applying dtype=object, rounding
        # large SQLite integers when a column also contains NULL or REAL values.
        frame = pd.DataFrame(cursor.fetchall(), columns=[c[0] for c in cursor.description], dtype=object)
    if len(frame) > limit:
        raise ValueError(
            f"Источник превышает лимит {limit}; создайте представление с ограниченной выборкой"
        )
    frame = frame.map(text_value).astype(object).where(frame.notna(), None)
    return frame.columns.tolist(), frame.values.tolist(), list(range(1, len(frame) + 1))




def read_postgres_snapshot(dsn, limit=100000):
    """Read existing geochem raw source records in a repeatable, read-only transaction."""
    from autopyro.data_io.io import text_value

    import psycopg

    with psycopg.connect(dsn, connect_timeout=10) as connection:
        connection.read_only = True
        connection.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        from .access import verify_read_only
        verify_read_only(connection)
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = '60s'")
            cursor.execute(
                """SELECT source_record_id, sample_id, raw_cells
                              FROM geochem.source_record ORDER BY source_record_id LIMIT %s""",
                (limit + 1,),
            )
            records = cursor.fetchall()
    if len(records) > limit:
        raise ValueError(
            f"Больше {limit} записей. В этой версии требуется ограниченный снимок."
        )
    if any(record[2] is None for record in records):
        raise ValueError(
            "Снимок raw_cells отсутствует у части записей. Импорт отменён без потери строк."
        )
    keys = list(
        dict.fromkeys((g, label) for _, _, cells in records for g, label, _ in cells)
    )
    headers = ["DB / source_record_id", "DB / sample_id"] + [
        f"{g} / {label}" for g, label in keys
    ]
    rows = []
    for record_id, sample_id, cells in records:
        values = {(g, label): text_value(value) for g, label, value in cells}
        rows.append(
            [str(record_id), str(sample_id)] + [values.get(key) for key in keys]
        )
    return headers, rows, [record[0] for record in records]

