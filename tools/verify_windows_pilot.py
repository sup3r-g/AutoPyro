"""Run the portable control case or compare two reports from a pilot series."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from hashlib import file_digest
from io import BytesIO
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from zipfile import ZipFile


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_install(archive_path, candidate):
    import autopyro

    require(sys.platform == "win32" and sys.version_info[:2] == (3, 12)
            and platform.machine().lower() in {"amd64", "x86_64"} and sys.maxsize > 2**32,
            "Use 64-bit CPython 3.12 on Windows.")
    require(sys.prefix != sys.base_prefix, "Run the verifier using the installed AutoPyro venv.")
    require(sha256(archive_path) == candidate["archive_sha256"], "The candidate ZIP SHA-256 differs.")
    package = Path(autopyro.__file__).resolve().parent
    require(package.is_relative_to(Path(sys.prefix).resolve()), "AutoPyro was imported outside the venv.")
    with ZipFile(archive_path) as archive:
        for line in archive.read("install.lock").decode("utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                name, version = line.split()[0].split("==", 1)
                require(importlib.metadata.version(name) == version, f"Installed version differs for {name}.")
        wheel, = (name for name in archive.namelist() if Path(name).name.startswith("autopyro-")
                  and name.endswith(".whl"))
        with ZipFile(BytesIO(archive.read(wheel))) as app:
            for name in app.namelist():
                if name.startswith("autopyro/") and not name.endswith("/"):
                    installed = package / name.removeprefix("autopyro/")
                    require(installed.is_file() and installed.read_bytes() == app.read(name),
                            f"Installed file differs from the candidate wheel: {name}")


def comparable(report):
    # Only timestamps generated on each report build can differ. Measurement,
    # method, interpretation, QC and project dates remain part of the comparison.
    require(report.get("format") == "autopyro.report" and report.get("version") == 1,
            "Expected an AutoPyro report JSON, format version 1.")
    core = {key: value for key, value in report.items() if key != "created_at"}
    core["statistics"] = {key: value for key, value in report["statistics"].items() if key != "created_at"}
    return core


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() | right.keys()):
            pointer = f"{path}/{key.replace('~', '~0').replace('/', '~1')}"
            if key not in left or key not in right:
                yield pointer
            else:
                yield from differences(left[key], right[key], pointer)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            yield f"{path} (length)"
        for index, (old, new) in enumerate(zip(left, right)):
            yield from differences(old, new, f"{path}/{index}")
    elif left != right or (type(left) is bool or type(right) is bool) and type(left) is not type(right):
        yield path or "/"


def compare_reports(reference, actual):
    reports = [comparable(json.loads(path.read_text(encoding="utf-8-sig"))) for path in (reference, actual)]
    first = []
    count = 0
    for count, path in enumerate(differences(*reports), 1):
        if count <= 30:
            first.append(path)
    return {
        "matches": count == 0,
        "reference_sha256": sha256(reference),
        "actual_sha256": sha256(actual),
        "difference_count": count,
        "first_differences": first,
    }


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def check_recovery(app, output, source, expected_hash):
    from PySide6.QtCore import QSettings
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from autopyro.model.core import Project
    from autopyro.ui import i18n
    from autopyro.ui.window import MainWindow

    settings = QSettings(str(output / "control.ini"), QSettings.Format.IniFormat)
    settings.setValue("general/autosave_enabled", True)
    i18n.settings = lambda: settings
    i18n.initialize(settings)
    working = output / "working.apyro"
    restored = output / "restored.apyro"
    shutil.copyfile(source, working)
    window = MainWindow()
    window.autosave_timer.stop()
    window.resize(1024, 700)
    window.show()
    try:
        app.processEvents()
        require(QTest.qWaitForWindowExposed(window, 10000), "The control window was not shown.")
        warnings = []
        with patch.object(QMessageBox, "warning", side_effect=lambda *args: warnings.append(str(args[2]))):
            with patch.object(QFileDialog, "getOpenFileName", return_value=(str(working), "")):
                window.open_project()
            project = window.model.project
            require(window.path == working and len(project.rows) == 6, "The control project failed to open.")
            position, column = next((i, column) for i, column in enumerate(project.columns) if column.name == "S2")
            row_id = project.rows[0].id
            measurement = project.science.current(row_id, column.id)
            require(measurement.value == 3.6, "Unexpected working repeat in the control project.")
            measurement_id, history_size = measurement.id, len(measurement.history)
            cell = window.model.index(0, position)
            require(window.model.setData(cell, "3.7"), "Could not edit the working repeat.")
            window.autosave()
            job = window._background_autosave
            publications = []
            job.saved.connect(publications.append)
            job.failed.connect(warnings.append)
            require(window.model.setData(cell, "3.8"), "Could not edit during autosave.")
            deadline = perf_counter() + 90
            while perf_counter() < deadline and not warnings and (len(publications) < 2 or job.worker):
                QTest.qWait(10)
            require(len(publications) == 2 and not warnings, f"Expected two autosave copies. {warnings}")
            recovery = window.autosave_path()
            recovery_hash = sha256(recovery)
            require(sha256(working) == expected_hash, "Autosave altered the original control file.")

            # Only the disposable copy is damaged. The kit project stays intact.
            working.write_bytes(b"Pilot control: deliberately damaged disposable copy")
            working_hash = sha256(working)
            choices = iter((QMessageBox.StandardButton.Discard, QMessageBox.StandardButton.Yes))
            with (patch.object(QFileDialog, "getOpenFileName", return_value=(str(working), "")),
                  patch.object(QMessageBox, "question", side_effect=lambda *args: next(choices))):
                window.open_project()
            require(window.path is None and window.recovered_from == recovery, "The recovery copy did not open.")
            with patch.object(QFileDialog, "getSaveFileName", return_value=(str(restored), "")):
                require(window.save(), f"Saving under a new name failed. {warnings}")
            reopened = Project.load(restored).science.current(row_id, column.id)
            require((reopened.id, reopened.value, len(reopened.history)) ==
                    (measurement_id, 3.8, history_size + 2), "Recovered working repeat or history differs.")
            require(sha256(working) == working_hash and sha256(recovery) == recovery_hash,
                    "Recovery altered its source files.")
        return {"publications": len(publications), "working_repeat": reopened.value,
                "restored_project_sha256": sha256(restored), "preserved_recovery_sha256": recovery_hash}
    finally:
        window.dirty = False
        window.close()
        app.aboutToQuit.emit()
        app.processEvents()


def reference(archive_path, output):
    kit = Path(__file__).resolve().parent
    candidate = json.loads((kit / "candidate.json").read_text(encoding="utf-8"))
    verify_install(archive_path, candidate)
    check_reference_case(kit, candidate, output)


def check_reference_case(kit, candidate, output):
    from PySide6.QtPdf import QPdfDocument
    from PySide6.QtWidgets import QApplication

    from autopyro.model.core import Project
    from autopyro.reports.reporting import ReportOptions, build_report, export_report

    source = kit / "control.apyro"
    project_hash = sha256(source)
    require(project_hash == candidate["control_project_sha256"], "The control project SHA-256 differs.")
    expected_report = kit / "reference.json"
    require(sha256(expected_report) == candidate["control_report_sha256"], "The reference report SHA-256 differs.")
    output = output.resolve()
    output.mkdir()
    app = QApplication([])
    require(app.platformName() == "windows", "The reference check requires native Windows Qt.")
    project = Project.load(source)
    column = next(column for column in project.columns if column.name == "S2")
    options = ReportOptions((column.id,), sections=("summary", "source", "method", "quality", "statistics", "data"))
    report = build_report(project, [row.id for row in project.rows], options)
    control_row = project.rows[-1].id
    science = report["project"]["science"]
    columns = {column.name: column.id for column in project.columns}
    for name, raw, qualifier, limit in (("Tmax", "ND", "non_detect", None), ("TOC", "<0.1", "below_limit", 0.1)):
        result_id = science["working"][control_row][columns[name]]
        record = science["measurements"][result_id]
        require((record["raw_value"], record["qualifier"], record["limit"], record["value"]) ==
                (raw, qualifier, limit, None), f"The control qualifier changed for {name}.")
    for kind in ("json", "pdf", "docx"):
        export_report(output / f"report.{kind}", report, kind)
    comparison = compare_reports(expected_report, output / "report.json")
    write_json(output / "comparison.json", comparison)
    require(comparison["matches"], "Control report differs. See comparison.json.")
    pdf = QPdfDocument()
    require(pdf.load(str(output / "report.pdf")) == QPdfDocument.Error.None_ and pdf.pageCount() > 0,
            "Cannot read the generated PDF.")
    pages = pdf.pageCount()
    pdf.close()
    with ZipFile(output / "report.docx") as document:
        require(document.testzip() is None and "word/document.xml" in document.namelist(), "Invalid generated DOCX.")
    recovery = check_recovery(app, output, source, project_hash)
    require(sha256(source) == project_hash, "The kit control file was altered.")
    write_json(output / "result.json", {
        "passed": True, "windows": platform.platform(), "python": platform.python_version(),
        "qt_platform": app.platformName(), "archive_sha256": candidate["archive_sha256"],
        "control_project_sha256": project_hash, "pdf_pages": pages,
        "control_qualifiers": {"Tmax": "ND", "TOC": "<0.1"},
        "report_comparison": comparison, "recovery": recovery,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    control = commands.add_parser("reference", help="Check candidate installation, control report and recovery.")
    control.add_argument("--zip", type=Path, required=True)
    control.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="Compare two JSON reports made from the same saved project.")
    compare.add_argument("reference", type=Path)
    compare.add_argument("actual", type=Path)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "reference":
            reference(args.zip, args.output)
            print(args.output / "result.json")
            return 0
        result = compare_reports(args.reference, args.actual)
        if args.output is not None:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["matches"] else 1
    except (OSError, ValueError, KeyError, importlib.metadata.PackageNotFoundError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    sys.exit(main())
