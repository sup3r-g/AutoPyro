"""Exercise the shown Qt desktop, table, HI and interpolation on synthetic data.

python tools/benchmark_workspace_interpolation.py --rows 30000 --output tmp/workspace-30000.json
The offscreen Qt backend paints widgets but does not measure a physical display.
No source project is read or written. Each invocation measures one fresh process.
"""
import argparse
import cProfile
import gc
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QItemSelectionModel, QModelIndex, QPersistentModelIndex, QTimer, Qt
from PySide6.QtWidgets import QApplication
from autopyro.io import build_project
from autopyro.audit import MAX_PROTOCOL_BYTES, PREFIX
from autopyro.protocol_json import protocol_json_chunks
from autopyro.ui.window import MainWindow
from tools.benchmark_import_refresh import process_memory


def settle(milliseconds=50):
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def measure(app, name, function, profile_directory=None):
    gc.collect()
    settle()
    gaps = []
    timer = QTimer()
    timer.setInterval(10)
    previous = perf_counter()

    def tick():
        nonlocal previous
        now = perf_counter()
        gaps.append(now - previous)
        previous = now

    timer.timeout.connect(tick)
    timer.start()
    settle()
    started = perf_counter()
    profiler = cProfile.Profile() if profile_directory else None
    try:
        value = profiler.runcall(function) if profiler else function()
        app.processEvents()  # Include queued painting and synchronous signal callbacks.
        elapsed = perf_counter() - started
        tick()
        settle()
    finally:
        timer.stop()
    if profiler:
        profiler.dump_stats(str(profile_directory / f"{name}.prof"))
    result = {"stage": name, "seconds": elapsed, "max_qt_gap_seconds": max(gaps),
              "heartbeat_samples": len(gaps), **process_memory()}
    print(json.dumps(result), flush=True)
    return value, result


def dataset(count, extra_columns=0):
    rows = []
    for i in range(count):
        depth = i % 1000
        value = "" if i % 5 == 2 else "ND" if i % 97 == 0 else "<0.1" if i % 101 == 0 else str(2 + depth / 100)
        rows.append([f"Well {i // 1000}", f"Sample {i % 10}", str(depth), "2", value]
                    + [str((i + j) % 100) for j in range(extra_columns)])
    headers = ["Well", "Sample", "Depth", "TOC", "S2"] + [f"Property {j + 1}" for j in range(extra_columns)]
    return build_project(headers, rows, list(range(2, count + 2)),
                         [(0, "Well", "text", ""), (1, "Sample", "text", ""),
                          (2, "Depth", "number", "m"), (3, "TOC", "number", "%wt"),
                          (4, "S2", "number", "mg HC/g rock")]
                         + [(5 + j, headers[5 + j], "number", "") for j in range(extra_columns)],
                         {"type": "synthetic GUI acceptance"})


def dependent_rules(owner, project, created, hi, scope, stage, result):
    """Exercise membership and ordering changes against independent expected UUID lists."""
    depth = project.columns[2].id
    source_ids = [row.id for row in project.rows]
    base = [(i, row) for i, row in enumerate(project.rows) if row.id in scope]
    hi_values = [row.values[hi.id] for row in project.rows]
    predictions = {row.id: row.values[created.id] for row in project.rows}
    stage("dependent_show_table", lambda: owner.desktop.documents.setCurrentWidget(owner.desktop.table_page))
    assert owner.table.isVisible()
    checks = {}
    result["dependent_rule_checks"] = checks
    for mode in ("filter", "sort"):
        def configure():
            owner.proxy.sort_rules = []
            rules = [{"column": depth, "operator": "lt", "value": 900}]
            if mode == "filter":
                rules.append({"column": created.id, "operator": "empty"})
            else:
                owner.proxy.sort_rules = [(created.id, True), (depth, False)]
            owner.apply_filter(rules, "AND")
            owner.proxy.sort(2 if mode == "filter" else 0,
                             Qt.SortOrder.DescendingOrder if mode == "filter" else Qt.SortOrder.AscendingOrder)

        stage(f"dependent_{mode}_setup", configure)
        # This qualified source result never receives an interpolated value and
        # remains visible through both filter membership and order changes.
        anchor = project.rows[0]
        assert predictions[anchor.id] is None and anchor.id in scope
        index = owner.proxy.mapFromSource(owner.model.index(0, 2))
        selection = owner.table.selectionModel()
        owner.table.setCurrentIndex(index)
        selection.select(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        persistent = QPersistentModelIndex(index)
        counts = []

        def check(applied):
            if mode == "filter":
                expected = [row.id for _, row in sorted(base, key=lambda item: -item[1].values[depth])
                            if not applied or predictions[row.id] is None]
            else:
                expected = [row.id for _, row in sorted(base, key=lambda item: (
                    not applied or predictions[item[1].id] is None,
                    -predictions[item[1].id] if applied and predictions[item[1].id] is not None else 0,
                    item[1].values[depth], item[0]))]
            assert owner.row_ids() == expected
            assert persistent.isValid()
            for selected in (QModelIndex(persistent), owner.table.currentIndex(), *selection.selectedIndexes()):
                assert project.rows[owner.proxy.mapToSource(selected).row()].id == anchor.id
            assert len(selection.selectedIndexes()) == 1
            assert [row.id for row in project.rows] == source_ids
            assert [row.values[hi.id] for row in project.rows] == hi_values
            assert all(row.values.get(created.id) == (predictions[row.id] if applied else None)
                       for row in project.rows)
            counts.append(len(expected))

        check(True)
        stage(f"dependent_{mode}_undo", owner.model.undo.undo)
        check(False)
        stage(f"dependent_{mode}_redo", owner.model.undo.redo)
        check(True)
        checks[mode] = {"rows_applied_undone_redone": counts, "selected_uuid": anchor.id,
                        "uuid_order_and_values_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--extra-columns", type=int, default=0,
                        help="Additional numeric source properties, besides the five standard columns")
    parser.add_argument("--table-only", action="store_true",
                        help="Exercise table and HI without building an interpolation protocol")
    parser.add_argument("--dependent-rules", action="store_true",
                        help="Also verify interpolation-dependent filtering and multiple-key sorting through undo/redo")
    parser.add_argument("--replay", action="store_true",
                        help="Also reproduce interpolation from its saved protocol after redo")
    parser.add_argument("--plain-filter-ordering", action="store_true",
                        help="Control run: omit temporary sorting preparation for dependent filter expansion")
    args = parser.parse_args()
    if args.rows < 100:
        parser.error("--rows must be at least 100")
    if args.extra_columns < 0:
        parser.error("--extra-columns must be non-negative")
    if args.table_only and args.dependent_rules:
        parser.error("--dependent-rules requires the full interpolation scenario")
    if args.table_only and args.replay:
        parser.error("--replay requires the full interpolation scenario")
    if args.plain_filter_ordering and not args.dependent_rules:
        parser.error("--plain-filter-ordering requires --dependent-rules")
    if args.plain_filter_ordering:
        # This process only: retain normal Qt filtering/sorting and skip the
        # optional preparation, without replacing scientific code or results.
        from autopyro.ui.table import FilterProxy
        FilterProxy._filter_may_expand = lambda self, column_ids: False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    owner = MainWindow()
    owner.autosave_timer.stop()
    owner.may_replace = lambda: True
    owner.resize(1400, 900)
    owner.show()
    settle()
    project = dataset(args.rows, args.extra_columns)
    ids = [row.id for row in project.rows]
    source_columns = len(project.columns)
    result = {"rows": args.rows, "source_columns": source_columns, "python": sys.version,
              "platform": platform.platform(), "processor": platform.processor(),
              "qt_backend": app.platformName(), "heartbeat_interval_ms": 10,
              "profiled": args.profile,
              "dependent_rules": args.dependent_rules,
              "filter_ordering": "plain_control" if args.plain_filter_ordering else "prepared_on_expansion",
              "scenario": "table_and_hi" if args.table_only else "table_hi_and_interpolation",
              "note": "One shown desktop; synthetic data; no tracemalloc; native working-set peak is cumulative. Initial window construction and dataset construction are excluded. Offscreen painting is not physical-display acceptance.",
              "stages": []}

    def stage(name, function):
        value, metrics = measure(app, name, function, args.output.parent if args.profile else None)
        result["stages"].append(metrics)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return value

    try:
        stage("attach_project", lambda: owner.set_project(project))
        for analysis in project.science.analyses.values():
            analysis.method, analysis.laboratory = "Synthetic Pyrol6", "QA"
        stage("open_calculations", owner.show_calculations)
        calculation = owner.calculation_window
        calculation.context.setText("Synthetic whole-rock acceptance")
        calculation.confirmed.setChecked(True)
        for i in range(calculation.outputs.rowCount()):
            calculation.outputs.cellWidget(i, 0).setChecked(i == 0)  # HI only.
        stage("calculate_hi", calculation.calculate)
        assert len(calculation.prepared) == 1, calculation.summary.text()
        stage("apply_hi", calculation.apply)
        hi = project.columns[-1]
        assert hi.name == "HI"
        stage("show_table", lambda: owner.desktop.documents.setCurrentWidget(owner.desktop.table_page))
        assert owner.table.isVisible()
        stage("filter_depth", lambda: owner.apply_filter([
            {"column": project.columns[2].id, "operator": "lt", "value": 900}], "AND"))
        stage("sort_depth", lambda: owner.proxy.sort(2, Qt.SortOrder.DescendingOrder))
        stage("scroll_table", owner.table.scrollToBottom)
        scope = set(owner.row_ids())
        assert [row.id for row in project.rows] == ids
        target = project.row_map()[owner.row_ids()[0]]
        original = target.values[project.columns[4].id]
        original_hi = target.values[hi.id]
        owner.table.setCurrentIndex(owner.proxy.index(0, 4))
        app.clipboard().setText("8.5")
        stage("paste_sorted", owner.paste_cells)
        assert target.values[hi.id] == 425
        stage("undo_edit", owner.model.undo.undo)
        assert target.values[project.columns[4].id] == original and target.values[hi.id] == original_hi
        assert [row.id for row in project.rows] == ids
        result.update(scope_rows=len(scope), measurements=len(project.science.measurements))
        if args.table_only:
            result["verified"] = True
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return
        stage("open_interpolation", owner.interpolate)
        dialog = owner.interpolation_window
        dialog.x.setCurrentIndex(dialog.x.findData(project.columns[2].id))
        dialog.y.setCurrentIndex(dialog.y.findData(project.columns[4].id))
        dialog.well.setCurrentIndex(dialog.well.findData(project.columns[0].id))
        dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
        dialog.method.setCurrentIndex(dialog.method.findData("pchip"))
        dialog.gap.setValue(10)
        stage("interpolation_preview", dialog.calculate)
        assert dialog.proposals, dialog.summary.text()
        assert all(p.row_id in scope and math.isclose(p.value, 2 + p.x / 100, abs_tol=1e-10)
                   for p in dialog.proposals)
        result.update(scope_rows=len(scope), proposals=len(dialog.proposals),
                      measurements=len(project.science.measurements),
                      snapshot_archived=isinstance(dialog.protocol, str))
        # Stream size accounting outside the timed preview; never allocate a
        # second whole-document JSON string just to report its length.
        protocol_bytes = (int(dialog.protocol[len(PREFIX):].split(":", 1)[0])
                          if isinstance(dialog.protocol, str) else
                          sum(map(len, protocol_json_chunks(dialog.protocol))))
        result.update(protocol_bytes=protocol_bytes, archive_limit_bytes=MAX_PROTOCOL_BYTES,
                      protocol_over_archive_limit=protocol_bytes > MAX_PROTOCOL_BYTES)
        stage("interpolation_chart", lambda: dialog.preview_tabs.setCurrentWidget(dialog.chart_view))
        assert dialog.chart_view.isVisible()
        stage("interpolation_diagnostics", lambda: dialog.preview_tabs.setCurrentWidget(dialog.diagnostics))
        expected_values = {p.row_id: p.value for p in dialog.proposals}
        hi_values = [row.values[hi.id] for row in project.rows]
        stage("interpolation_apply", dialog.apply)
        created = project.columns[-1]
        assert created.metadata.get("method") == "pchip"
        assert all(row.values[created.id] == expected_values.get(row.id) for row in project.rows)
        first_protocol = project.history[-1]["metadata"]["interpolation_protocol"]
        if isinstance(dialog.protocol, dict):
            assert first_protocol is not dialog.protocol
            assert first_protocol["checksum"] == dialog.protocol["checksum"]
        stage("undo_interpolation", owner.model.undo.undo)
        assert len(project.columns) == source_columns + 1
        assert all(created.id not in row.values for row in project.rows)
        stage("redo_interpolation", owner.model.undo.redo)
        assert len(project.columns) == source_columns + 2
        assert all(row.values[created.id] == expected_values.get(row.id) for row in project.rows)
        if isinstance(dialog.protocol, dict):
            repeated = project.history[-1]["metadata"]["interpolation_protocol"]
            assert repeated is not first_protocol and repeated is not dialog.protocol
            assert repeated["checksum"] == first_protocol["checksum"]
        assert [row.id for row in project.rows] == ids
        assert [row.values[hi.id] for row in project.rows] == hi_values
        assert target.values[hi.id] == original_hi
        if args.replay:
            from autopyro.interpolation import replay
            from autopyro.ui.tasks import run_with_progress
            reproduced = stage("replay_interpolation", lambda: run_with_progress(
                owner, "Replay synthetic interpolation", lambda: replay(created.metadata["interpolation_protocol"])))
            assert reproduced == dialog.proposals
            result["replay_verified"] = True
        if args.dependent_rules:
            dependent_rules(owner, project, created, hi, scope, stage, result)
        result["verified"] = True
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        owner.dirty = False
        owner.close()
        app.processEvents()


if __name__ == "__main__":
    main()
