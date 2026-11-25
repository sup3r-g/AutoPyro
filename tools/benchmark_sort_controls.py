"""Measure actual table sort controls on a synthetic shown desktop, offscreen."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.benchmark_workspace_interpolation import dataset, measure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from autopyro.ui.window import MainWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=30000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    owner = MainWindow()
    owner.autosave_timer.stop()
    project = dataset(args.rows)
    source_ids = [row.id for row in project.rows]
    stages = []
    try:
        owner.set_project(project)
        owner.resize(1400, 900)
        owner.show()
        app.processEvents()
        editor = owner.desktop.table_properties
        editor.load_sort()
        for i in range(editor.sort_columns.count()):
            item = editor.sort_columns.item(i)
            item.setCheckState(Qt.CheckState.Checked if i in (2, 3) else Qt.CheckState.Unchecked)
        _, result = measure(app, "apply_multiple_sort", editor.apply_sort)
        stages.append(result)
        expected = [r.id for r in sorted(project.rows, key=lambda r: (
            r.values[project.columns[2].id], r.values[project.columns[3].id]))]
        assert owner.row_ids() == expected
        # Change the direction without changing the native Qt sort column/order.
        item = next(editor.sort_columns.item(i) for i in range(editor.sort_columns.count())
                    if editor.sort_columns.item(i).data(Qt.ItemDataRole.UserRole) == project.columns[2].id)
        editor.reverse_sort(item)
        _, result = measure(app, "change_multiple_sort", editor.apply_sort)
        stages.append(result)
        assert owner.row_ids() == [r.id for r in sorted(project.rows, key=lambda r: (
            -r.values[project.columns[2].id], r.values[project.columns[3].id]))]
        _, result = measure(app, "reset_sort", editor.reset_sort)
        stages.append(result)
        assert owner.row_ids() == source_ids
        assert [r.id for r in project.rows] == source_ids
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"rows": args.rows, "backend": "offscreen",
                                           "verified": True, "stages": stages}, indent=2), encoding="utf-8")
    finally:
        owner.autosave_timer.stop()
        owner.dirty = False
        owner.close()
        app.processEvents()


if __name__ == "__main__":
    main()
