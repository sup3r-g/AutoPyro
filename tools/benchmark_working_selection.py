"""Measure working-result selection/undo/redo in a synthetic wide Qt table.

Run with python -m tools.benchmark_working_selection --rows 30000 --output tmp/selection.json.
No project files are read or written. Fixture preparation and checks are untimed.
"""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from autopyro.calculations import make_calculated_column
from autopyro.ui.table import TableModel
from autopyro.ui.table.commands import SelectMeasurementCommand
from tools.benchmark_import_refresh import process_memory
from tools.benchmark_measurement_sync import table_digest
from tools.benchmark_workspace_interpolation import dataset


def benchmark(app, count, extra_columns):
    project = dataset(count, extra_columns)
    model = TableModel(project)
    toc, s2 = project.columns[3:5]
    config = {"formula": "HI", "version": 1, "compatible_confirmed": True,
              "analysis_context": "Synthetic performance QA; canonical units",
              "inputs": {"TOC": {"column_id": toc.id, "unit": toc.unit},
                         "S2": {"column_id": s2.id, "unit": s2.unit}}}
    hi, values = make_calculated_column(project, "HI", config)
    project.columns.append(hi)
    for row in project.rows:
        row.values[hi.id] = values[row.id]
    row = project.rows[3]  # An exact S2 value, unlike the fixture's ND/missing rows.
    original = project.science.current(row.id, s2.id)
    repeat = project.science.add_result(row.id, s2.id, "8",
        analysis_fields={"method": "Pyrol6", "laboratory": "QA repeat"},
        details={"uncertainty": .3, "uncertainty_type": "standard", "confidence_level": .95})
    source = asdict(project.science.sources[original.source_id])
    measurements = {m.id: asdict(m) for m in (original, repeat)}
    before = table_digest(project)
    stages = []
    notifications = []
    model.dataChanged.connect(lambda first, last, roles: notifications.append((first.row(), last.row())))
    actions = [("select", lambda: model.undo.push(SelectMeasurementCommand(model, row.id, s2.id, repeat.id)), repeat),
               ("undo", model.undo.undo, original), ("redo", model.undo.redo, repeat)]
    for name, action, expected in actions:
        gc.collect()
        app.processEvents()
        started = perf_counter()
        action()
        elapsed = perf_counter() - started
        metrics = {"stage": name, "seconds": elapsed, **process_memory(),
                   "notified_rows": notifications[-1]}
        stages.append(metrics)
        print(json.dumps(metrics), flush=True)
        assert project.science.current(row.id, s2.id) is expected
        assert row.values[s2.id] == expected.value
        assert row.values[hi.id] == expected.value / row.values[toc.id] * 100
        assert asdict(project.science.sources[original.source_id]) == source
        assert {m.id: asdict(m) for m in (original, repeat)} == measurements
        # Temporarily restore only the expected two projection cells to compare
        # all rows, source text and IDs against the unchanged table fingerprint.
        current = row.values[s2.id], row.values[hi.id]
        try:
            row.values[s2.id], row.values[hi.id] = original.value, values[row.id]
            assert table_digest(project) == before
        finally:
            row.values[s2.id], row.values[hi.id] = current
    project.validate()
    return {"rows": count, "source_columns": 5 + extra_columns,
            "measurements": len(project.science.measurements), "stages": stages, "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--extra-columns", type=int, default=75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 4 or args.extra_columns < 0:
        parser.error("rows must be at least 4 and extra-columns non-negative")
    app = QApplication.instance() or QApplication([])
    result = benchmark(app, args.rows, args.extra_columns)
    result.update(python=sys.version, platform=platform.platform(), processor=platform.processor(),
        note="One pass, no tracemalloc or shown desktop. Times cover command and synchronous signals; "
             "preparation, GC, checks and queued GUI work excluded. Peak includes fixture preparation. "
             "No p95 or native GUI responsiveness measurement.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
