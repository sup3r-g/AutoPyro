"""Compare SQLite index build order using one prepared synthetic project.

python -m tools.benchmark_sqlite_indexes --rows 30000 --directory tmp/index-order-abba
The output directory must be new. Only the database writer is timed.
"""
import argparse
from contextlib import closing
import gc
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
from time import perf_counter
from unittest.mock import patch

from autopyro.database import storage
from tools.benchmark_workspace_interpolation import dataset


def database_fingerprint(path):
    """Compare schemas and ordered rows, ignoring physical SQLite page layout."""
    digest = hashlib.sha256()

    def add(rows):
        for row in rows:
            digest.update(repr(row).encode("utf-8"))
            digest.update(b"\n")

    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        schema = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_schema ORDER BY type, name"
        ).fetchall()
        add(schema)
        for object_type, name, _, _ in schema:
            if object_type == "table":
                add([(name,)])
                quoted_name = '"' + name.replace('"', '""') + '"'
                add(connection.execute(f"SELECT * FROM {quoted_name} ORDER BY rowid"))
    return digest.hexdigest()


def benchmark(directory, rows, extra_columns):
    directory.mkdir(parents=True, exist_ok=False)
    print("Preparing synthetic project", flush=True)
    project = dataset(rows, extra_columns)
    project.sync_measurements()
    project.validate()
    report = {
        "rows": rows,
        "columns": len(project.columns),
        "measurements": len(project.science.measurements),
        "write_cache_kib": storage.WRITE_CACHE_KIB,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version,
        "runs": [],
        "verified": False,
    }
    report_path = directory / "result.json"

    def record():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for position, order in enumerate(("before", "after", "after", "before"), start=1):
        path = directory / f"{position}-{order}.apyro"
        schema, indexes = storage.SCHEMA, storage.SECONDARY_INDEXES
        if order == "before":
            schema, indexes = schema + indexes, ""
        gc.collect()
        print(f"Writing {path.name}", flush=True)
        with patch.multiple(storage, SCHEMA=schema, SECONDARY_INDEXES=indexes):
            started = perf_counter()
            storage._write_database(path, project)
            elapsed = perf_counter() - started
        run = {"file": path.name, "order": order, "seconds": elapsed, "bytes": path.stat().st_size}
        report["runs"].append(run)
        record()
        print(json.dumps(run), flush=True)

    for run in report["runs"]:
        print(f"Verifying {run['file']}", flush=True)
        run["fingerprint"] = database_fingerprint(directory / run["file"])
        record()
    if len({run["fingerprint"] for run in report["runs"]}) != 1:
        raise ValueError("Database schemas or stored rows differ")
    report["verified"] = True
    record()
    print(f"Verified: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=30000)
    parser.add_argument("--extra-columns", type=int, default=75)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1 or args.extra_columns < 0:
        parser.error("rows must be positive and extra-columns non-negative")
    benchmark(args.directory.resolve(), args.rows, args.extra_columns)


if __name__ == "__main__":
    main()
