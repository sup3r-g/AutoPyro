import gc
import json
import weakref
from pathlib import Path
from threading import Event, get_ident

import pytest
from docx import Document
from matplotlib import get_data_path
from pypdf import PdfReader
from PySide6.QtCore import QCoreApplication, QEvent, QTimer, Qt
from PySide6.QtGui import QAccessible, QDesktopServices, QFont, QFontDatabase
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QInputDialog, QLabel, QListWidget, QPushButton, QScrollArea, QSpinBox, QTextEdit

from autopyro.report_dialog import ReportDialog
from autopyro.data_io.io import build_project
from autopyro.reports.history import compare_snapshots, load_snapshot
from autopyro.reports.reporting import report_blocks
from autopyro.ui.dialogs import report_batch, report_export, report_preview
from autopyro.ui.tasks import TaskDialog
from autopyro.visualization.plotting import PlotSpec
from autopyro.window import MainWindow


def test_report_dialog_releases_project_snapshots_after_twenty_closes(qtbot):
    from autopyro.ui.i18n import tr

    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    bindings = len(window.analysis_scope._bindings)
    snapshots = []

    def build_and_close():
        dialog = QApplication.activeModalWidget()
        try:
            assert isinstance(dialog, ReportDialog)
            dialog.build()
            assert dialog.report is not None, dialog.status.text()
            source, _, _ = dialog.report_source
            assert source is not window.model.project and len(source.rows) == 5
            snapshots.append(weakref.ref(source))
        finally:
            dialog.close()

    for cycle in range(1, 21):
        QTimer.singleShot(0, build_and_close)
        window.actions[tr('Отчёт…')].trigger()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        gc.collect()
        assert len(snapshots) == cycle
        assert not window.findChildren(ReportDialog)
        assert len(window.analysis_scope._bindings) == bindings
        assert all(snapshot() is None for snapshot in snapshots)


def test_report_build_invalidates_and_keeps_empty_selection(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.may_replace = lambda: True
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.scope.setCurrentIndex(dialog.scope.findData("selected"))
    assert dialog.scope_summary.text() == "Проб в выборке: 0"
    dialog.build()
    assert dialog.report is not None, dialog.status.text()
    assert dialog.report["row_ids"] == []
    source, _, _ = dialog.report_source
    assert source.rows == []
    assert not source.science.measurements
    assert all(b.isEnabled() for b in dialog.exports)
    dialog.title.setText("Другой отчёт")
    assert dialog.report is None
    assert not any(b.isEnabled() for b in dialog.exports)
    dialog.scope.setCurrentIndex(dialog.scope.findData("all"))
    assert dialog.scope_summary.text() == "Проб в выборке: 5"
    dialog.build()
    assert len(dialog.report["row_ids"]) == len(window.model.project.rows)
    assert "Отчёт…" in window.actions
    dialog.columns.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert dialog.report is None
    window.dirty = False


def test_report_dialog_fits_compact_window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.resize(800, 600)
    dialog.show()
    assert dialog.width() <= 800 and dialog.height() <= 600
    settings = dialog.findChild(QScrollArea)
    assert settings.verticalScrollBar().maximum() > 0
    window.dirty = False


def test_report_keyboard_workflow_with_larger_font(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    font_path = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    font = QFont(QFontDatabase.applicationFontFamilies(font_id)[0])
    font.setPointSize(15)
    dialog.setFont(font)
    dialog.resize(800, 600)
    dialog.show()
    dialog.activateWindow()
    assert dialog.width() <= 800 and dialog.height() <= 600
    template = next(button for button in dialog.findChildren(QPushButton)
                    if button.text() == "Применить шаблон…")
    template.setFocus()
    qtbot.waitUntil(template.hasFocus)
    qtbot.keyClick(template, Qt.Key.Key_Tab)
    assert dialog.scope.hasFocus(), QApplication.focusWidget()
    dialog.column_search.setFocus()
    qtbot.keyClick(dialog.column_search, Qt.Key.Key_Return)
    assert dialog.report is None
    qtbot.keyClick(dialog.column_search, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert dialog.report is not None, dialog.status.text()
    dialog.activateWindow()
    dialog.column_search.setFocus()
    qtbot.waitUntil(dialog.column_search.hasFocus)
    qtbot.keyClick(dialog.column_search, Qt.Key.Key_L, Qt.KeyboardModifier.ControlModifier)
    assert dialog.preview_sections.hasFocus()
    export = next(button for button in dialog.findChildren(QPushButton) if button.text() == "Экспорт")
    opened = []

    def close_export_menu():
        opened.append(True)
        QTimer.singleShot(0, export.menu().close)

    export.menu().aboutToShow.connect(close_export_menu)
    qtbot.keyClick(dialog.preview_sections, Qt.Key.Key_E,
                   Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert opened == [True]
    assert QAccessible.queryAccessibleInterface(dialog.preview).text(QAccessible.Text.Name) == "Предпросмотр отчёта"
    dialog.title.setText("Изменённый заголовок")
    assert "Устарел" in QAccessible.queryAccessibleInterface(dialog.preview).text(QAccessible.Text.Description)
    window.dirty = False


def test_report_preview_uses_pdf_pages_and_opens_temporary_docx(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.section_order = ("conclusions",)
    dialog.conclusions.setPlainText("\n".join(f"Абзац {i}" for i in range(100)))
    dialog.build()
    assert dialog.page_preview.isEnabled()
    original_exec = QDialog.exec
    visited = []

    def inspect(popup):
        if popup.windowTitle() != "Предпросмотр PDF":
            return original_exec(popup)
        view = popup.findChild(QPdfView)
        page = popup.findChild(QSpinBox)
        assert page.maximum() == view.document().pageCount() > 1
        assert popup.focusWidget() is page
        assert page.accessibleDescription() == f"Всего страниц: {page.maximum()}."
        assert view.accessibleName() == "Страницы отчёта PDF"
        page.setValue(2)
        assert view.pageNavigator().currentPage() == 1
        visited.append(page.value())
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", inspect)
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(Path(url.toLocalFile())) or True)
    dialog.preview_pages("pdf")
    dialog.preview_pages("docx")
    assert visited == [2]
    path, = opened
    assert path.suffix == ".docx" and path.is_file()
    assert any(p.text == "Абзац 99" for p in Document(path).paragraphs)
    dialog.title.setText("Устаревший снимок")
    assert not dialog.page_preview.isEnabled()
    dialog.preview_pages("docx")
    assert len(opened) == 1
    window.dirty = False


def test_report_search_bulk_selection_and_keyboard_section_order(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    project = window.model.project
    first, second = project.columns[3:5]
    second.name = first.name
    project.view["plots"] = {
        "Пиролиз А": PlotSpec(kind="hist", x_id=second.id).to_dict(),
        "Другой график": PlotSpec(kind="hist", x_id=second.id).to_dict(),
    }
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.group.itemText(dialog.group.findData(first.id)) != dialog.group.itemText(dialog.group.findData(second.id))
    dialog.build()
    report = dialog.report
    dialog.column_search.setText("toc колонка")
    assert dialog.report is report
    assert [not dialog.columns.item(i).isHidden() for i in range(dialog.columns.count())].count(True) == 2
    clear = next(button for button in dialog.findChildren(QPushButton)
                 if button.accessibleName().startswith("Поиск колонок") and button.text() == "Снять")
    clear.click()
    assert dialog.report is None
    assert set(dialog.composition()[0].column_ids) == {column.id for column in project.columns if column not in (first, second)}
    dialog.plot_search.setText("пиролиз")
    select = next(button for button in dialog.findChildren(QPushButton)
                  if button.accessibleName().startswith("Поиск графиков") and button.text() == "Отметить")
    select.click()
    dialog.plot_search.clear()
    assert [name for name, _ in dialog.composition()[1]] == ["Сохранённый: Пиролиз А"]

    original_exec = QDialog.exec

    def reorder(popup):
        if popup.windowTitle() != "Разделы отчёта":
            return original_exec(popup)
        items = popup.findChild(QListWidget)
        items.setCurrentRow(1)
        up = next(button for button in popup.findChildren(QPushButton) if button.text() == "Выше")
        qtbot.keyClick(up, Qt.Key.Key_Space)
        assert not up.isEnabled()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", reorder)
    dialog.edit_sections()
    assert dialog.section_order[:2] == ("source", "summary")
    window.dirty = False


def test_preview_stays_navigable_but_cannot_export_until_rebuilt(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.resize(800, 600)
    dialog.show()
    dialog.section_titles = {"data": "Мои данные"}
    dialog.build()
    document = dialog.preview.toPlainText()
    index = dialog.preview_sections.findData("report-data")
    assert index > 0 and dialog.preview_sections.itemText(index) == "Мои данные"
    dialog.preview_sections.activated.emit(index)
    cursor = dialog.preview.textCursor()
    assert cursor.block().text() == "Мои данные"
    assert cursor.positionInBlock() == 0
    dialog.preview.setFocus()
    qtbot.keyClick(dialog.preview, Qt.Key.Key_Right)
    assert dialog.preview.textCursor().position() == cursor.position() + 1
    scrollbar = dialog.preview.verticalScrollBar()
    qtbot.waitUntil(lambda: scrollbar.value() > 0)
    position = scrollbar.value()

    dialog.title.setText("Новый паспорт")
    assert dialog.preview_stale.isVisible()
    assert dialog.preview.toPlainText() == document
    assert scrollbar.value() == position
    assert dialog.preview_sections.isEnabled()
    assert not any(action.isEnabled() for action in dialog.exports)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: pytest.fail("Stale export"))
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: pytest.fail("Stale batch"))
    dialog.export("pdf")
    dialog.export_batch()

    with monkeypatch.context() as error:
        announcements = []
        error.setattr(dialog, "announce", announcements.append)
        def fail(*args, **kwargs):
            raise ValueError("test failure")
        error.setattr(report_preview, "build_report", fail)
        dialog.build()
        assert announcements[-1] == dialog.status.text()
        assert announcements[-1].startswith("Ошибка сборки")
    assert dialog.preview_stale.isVisible() and dialog.preview.toPlainText() == document
    dialog.build()
    assert "Новый паспорт" in dialog.preview.toPlainText()
    assert not dialog.preview_stale.isVisible()
    assert all(action.isEnabled() for action in dialog.exports)
    window.dirty = False


def test_release_checks_explain_text_numbers_and_block_report_without_sections(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.build()
    toc = next(column for column in window.model.project.columns if column.name == "TOC")
    check = next(check for check in dialog.report["release_checks"]
                 if check["code"] == "no_numeric_data" and check["column_id"] == toc.id)
    assert check["severity"] == "warning"
    assert "Использовать числа в текстовых колонках" in check["hint"]

    dialog.text_numbers.setChecked(True)
    dialog.build()
    assert not any(check["code"] == "no_numeric_data" and check["column_id"] == toc.id
                   for check in dialog.report["release_checks"])

    dialog.section_order = ("conclusions",)
    dialog.build()
    assert "Блокирует выпуск" in dialog.preview.toPlainText()
    assert not any(action.isEnabled() for action in dialog.exports)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: pytest.fail("Blocked report"))
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **kw: pytest.fail("Blocked package"))
    dialog.export("pdf")
    dialog.export_batch()
    dialog.conclusions.setPlainText("Проверка завершена.")
    dialog.build()
    assert all(action.isEnabled() for action in dialog.exports)
    window.dirty = False


def test_report_cancellation_discards_background_result(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.title.setText("Сохранённый заголовок")
    main_thread = get_ident()
    released = Event()
    worker_threads = []

    def assemble(*args, progress, **kwargs):
        worker_threads.append(get_ident())
        assert released.wait(timeout=5)
        progress("Следующий этап", 50)
        raise AssertionError("The cancellation checkpoint must stop the build")

    def cancel():
        if not worker_threads:
            QTimer.singleShot(5, cancel)
            return
        task = dialog.findChild(TaskDialog)
        task.findChild(QPushButton).click()
        released.set()

    monkeypatch.setattr(report_preview, "build_report", assemble)
    announcements = []
    monkeypatch.setattr(dialog, "announce", announcements.append)
    QTimer.singleShot(0, cancel)
    dialog.build()
    assert worker_threads and worker_threads[0] != main_thread
    assert dialog.report is None
    assert "Сборка отменена" in dialog.status.text()
    assert announcements == [dialog.status.text()]
    assert dialog.title.text() == "Сохранённый заголовок"
    assert not any(button.isEnabled() for button in dialog.exports)
    window.dirty = False


def test_batch_export_stops_when_scope_changes_during_group_choice(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.scope.setCurrentIndex(dialog.scope.findData("all"))
    dialog.build()
    assert dialog.report["row_ids"]

    def change_scope(*args, **kwargs):
        window.analysis_scope.set_mode("selected")
        return args[3][0], True

    monkeypatch.setattr(QInputDialog, "getItem", change_scope)
    monkeypatch.setattr(dialog, "choose_batch_formats", lambda: pytest.fail("Stale report reached format choice"))
    dialog.export_batch()
    assert dialog.report is None
    assert "Соберите актуальный отчёт" in dialog.status.text()


def test_report_template_and_batch_keep_composition_and_scope(qtbot, monkeypatch, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.scope.setCurrentIndex(dialog.scope.findData("all"))
    dialog.section_order = ("data", "summary")
    dialog.qc_open_only.setChecked(True)
    assert not dialog.detailed_correlations.isChecked()
    dialog.detailed_correlations.setChecked(True)
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("Well report", True))
    dialog.save_template()
    dialog.section_order = None
    dialog.qc_open_only.setChecked(False)
    dialog.detailed_correlations.setChecked(False)
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: (args[3][0], True))
    dialog.load_template()
    assert dialog.qc_open_only.isChecked()
    assert dialog.detailed_correlations.isChecked()
    dialog.build()
    html = dialog.preview.toPlainText()
    assert html.index("Приложение: данные") < html.index("Краткая сводка")
    assert "Методика" not in html
    # A live change that did not invalidate the dialog must not leak into the package.
    project = window.model.project
    project.rows[0].values[project.columns[0].id] = "Изменено после предпросмотра"

    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("1. Скважина", True))
    monkeypatch.setattr(dialog, "choose_batch_formats", lambda: ("json",))
    real_exec = QDialog.exec
    monkeypatch.setattr(QDialog, "exec", lambda self: (
        QDialog.DialogCode.Accepted if self.windowTitle() in {"Файлы пакета", "Готовность пакета"} else real_exec(self)
    ))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    dialog.export_batch()
    snapshots = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(tmp_path.glob("*/*.json"))]
    assert [len(report["row_ids"]) for report in snapshots] == [3, 2], dialog.status.text()
    assert [report["selection_context"]["batch_group"]["value"] for report in snapshots] == ["A", "B"]
    assert all(report["options"]["sections"] == ["data", "summary"] for report in snapshots)
    assert all(report["options"]["qc_open_only"] for report in snapshots)
    assert all(report["options"]["detailed_correlations"] for report in snapshots)
    window.dirty = False


def test_report_template_repairs_removed_column_and_renamed_plot(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    project = window.model.project
    well, removed, depth, _, indicator, _ = project.columns
    project.view["plots"] = {"Старый график": PlotSpec(x_id=indicator.id, y_id=depth.id).to_dict()}
    original = ReportDialog(window)
    qtbot.addWidget(original)
    original.group.setCurrentIndex(original.group.findData(removed.id))
    original.plots.item(0).setCheckState(Qt.CheckState.Checked)
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("Repair", True))
    original.save_template()
    saved = project.view["report_templates"]["Repair"]
    del saved["options"]["detailed_correlations"]  # A template saved before this control existed.
    assert saved["column_labels"][removed.id] == removed.name
    indicator.name = "S2 переименован"
    project.columns.remove(removed)
    for row in project.rows:
        row.values.pop(removed.id)
        row.original.pop(removed.id, None)
    project.view["plots"]["Новый график"] = project.view["plots"].pop("Старый график")
    project.view["plots"]["График удалённой колонки"] = PlotSpec(x_id=removed.id, y_id=depth.id).to_dict()
    window.set_project(project)
    repaired = ReportDialog(window)
    qtbot.addWidget(repaired)
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("Repair", True))
    real_exec = QDialog.exec
    cancelled = True

    def respond(dialog):
        if dialog.windowTitle() != "Восстановление ссылок шаблона":
            return real_exec(dialog)
        if cancelled:
            return QDialog.DialogCode.Rejected
        combos = dialog.findChildren(QComboBox)
        assert len(combos) == 2
        assert any(removed.name in label.text() and "группировка" in label.text()
                   for label in dialog.findChildren(QLabel))
        assert any("Старый график" in label.text() for label in dialog.findChildren(QLabel))
        assert combos[1].findData("saved:График удалённой колонки") == -1
        combos[0].setCurrentIndex(combos[0].findData(well.id))
        combos[1].setCurrentIndex(combos[1].findData("saved:Новый график"))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", respond)
    repaired.load_template()
    assert project.view["report_templates"]["Repair"] is saved
    cancelled = False
    repaired.load_template()
    options, plots = repaired.composition()
    assert options.detailed_correlations
    assert removed.id not in options.column_ids
    assert indicator.id in options.column_ids
    assert options.group_ids == (well.id,)
    assert [name for name, _ in plots] == ["Сохранённый: Новый график"]
    assert "saved:Новый график" in project.view["report_templates"]["Repair"]["plot_refs"]
    repaired.build()
    assert repaired.report is not None, repaired.status.text()
    window.dirty = False


@pytest.mark.parametrize("release_ready", [False, True])
def test_batch_format_error_requires_explicit_partial_release(qtbot, monkeypatch, tmp_path, release_ready):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.build()
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("1. Скважина", True))
    monkeypatch.setattr(dialog, "choose_batch_formats", lambda: ("pdf",))

    staged_paths = []

    def stage(path, report, kind):
        staged_paths.append(path)
        if report["selection_context"]["batch_group"]["value"] == "B":
            raise ValueError("Проверочная ошибка PDF")
        path.write_bytes(b"prepared")

    monkeypatch.setattr(report_batch, "export_report", stage)
    original = QDialog.exec

    def respond(popup):
        if popup.windowTitle() == "Файлы пакета":
            return QDialog.DialogCode.Accepted
        if popup.windowTitle() == "Готовность пакета":
            assert staged_paths[0].is_file()
            assert not list(tmp_path.iterdir())
            details = popup.findChild(QTextEdit).toPlainText()
            assert "001-A.pdf" in details and "002-B.pdf" in details
            assert "Проверочная ошибка PDF" in details
            save = next(button for button in popup.findChild(QDialogButtonBox).buttons()
                        if "Записать" in button.text())
            assert not save.isEnabled()
            if release_ready:
                check = next(check for check in popup.findChildren(QCheckBox) if "Исключить" in check.text())
                check.setChecked(True)
                assert save.isEnabled()
            return QDialog.DialogCode.Accepted
        return original(popup)

    monkeypatch.setattr(QDialog, "exec", respond)
    def choose_folder(*args):
        assert release_ready, "Blocked package must not open a destination picker"
        return str(tmp_path)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", choose_folder)
    dialog.export_batch()
    if release_ready:
        assert [path.name for path in tmp_path.glob("*/*.pdf")] == ["001-A.pdf"]
    else:
        assert "Пакет не записан" in dialog.status.text()
        assert not list(tmp_path.iterdir())
    assert all(not path.parent.exists() for path in staged_paths)
    window.dirty = False


def test_batch_multiple_formats_merge_missing_and_retry_only_failed_file(qtbot, monkeypatch, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    project = window.model.project
    well_id = project.columns[0].id
    for row, value in zip(project.rows, (None, "", "  ")):
        row.values[well_id] = value
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.build()
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("1. Скважина", True))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    original_exec = QDialog.exec
    original_write = report_batch.write_report_file
    original_export = report_batch.export_report
    attempts = []
    staged_paths = []

    def stage(path, report, kind):
        staged_paths.append(path)
        original_export(path, report, kind)

    def write(path, payload):
        assert not isinstance(payload, bytes)
        attempts.append(path)
        if path.name == "002-B.json" and attempts.count(path) == 1:
            raise PermissionError("Проверочная ошибка записи")
        original_write(path, payload)

    def respond(popup):
        title = popup.windowTitle()
        if title == "Форматы пакета":
            for check in popup.findChildren(QCheckBox):
                check.setChecked(check.text().startswith(("HTML", "JSON")))
        elif title == "Файлы пакета":
            files = popup.findChild(QListWidget)
            assert files.count() == 4
            assert sum("проб: 3" in files.item(i).text() for i in range(files.count())) == 2
        elif title == "Готовность пакета":
            assert not attempts and not list(tmp_path.iterdir())
            assert len(staged_paths) == 4 and all(path.is_file() for path in staged_paths)
        elif title == "Не удалось записать файлы":
            details = popup.findChild(QTextEdit).toPlainText()
            assert "002-B.json" in details and "001-" not in details
            assert len(list(tmp_path.glob("reports-*/*"))) == 3
        else:
            return original_exec(popup)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(report_batch, "write_report_file", write)
    monkeypatch.setattr(report_batch, "export_report", stage)
    monkeypatch.setattr(QDialog, "exec", respond)
    dialog.export_batch()
    assert len(attempts) == 5 and attempts[-1] == attempts[-2]
    assert len({path.parent for path in attempts}) == 1
    assert all(not path.parent.exists() for path in staged_paths)
    missing = next(tmp_path.glob("reports-*/001-*.json"))
    snapshot = json.loads(missing.read_text(encoding="utf-8"))
    assert snapshot["selection_context"]["batch_group"]["value"] is None
    assert len(snapshot["row_ids"]) == 3
    assert "Сохранено файлов: 4" in dialog.status.text()
    window.dirty = False


def test_builtin_templates_and_two_level_comparison(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    well, formation = window.model.project.columns[:2]
    formation.name = "Пласт"
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.scope.setCurrentIndex(dialog.scope.findData("all"))
    choices = iter(("Паспорт скважины", "Характеристика пласта", "Контроль качества"))

    def choose_template(*args, **kwargs):
        name = next(choices)
        return next(label for label in args[3] if label.endswith(name)), True

    monkeypatch.setattr(QInputDialog, "getItem", choose_template)
    dialog.load_template()
    assert dialog.composition()[0].group_ids == (well.id, formation.id)
    dialog.build()
    assert dialog.report is not None, dialog.status.text()
    assert "Сопоставление объектов" in dialog.preview.toPlainText()
    groups = dialog.report["statistics"]["summary"]
    assert {(tuple(row["group"]), row["rows"]) for row in groups} == {
        (("A", "Shale"), 3), (("B", "Shale"), 2),
    }
    assert dialog.report["overview"]["depth"]["minimum"] == 1000
    assert dialog.report["overview"]["depth"]["maximum"] == 1020
    assert {row["name"] for row in dialog.report["overview"]["indicators"]} == {"S2"}
    chart = next(chart for chart in dialog.report["comparison_figures"] if chart["name"].endswith("S2"))
    assert chart["groups"] == [["A", "Shale"], ["B", "Shale"]]
    key = next(key for section, key, kind, _, _ in report_blocks(dialog.report)
               if section == "comparison_figures" and kind == "figure")
    dialog.section_titles = {"summary": "Обзор скважин"}
    dialog.block_comments = {key: "Комментарий к сопоставлению"}
    dialog.build()
    assert "Обзор скважин" in dialog.preview.toPlainText()
    assert "Комментарий к сопоставлению" in dialog.preview.toPlainText()

    dialog.load_template()
    assert dialog.composition()[0].group_ids == (formation.id, well.id)
    dialog.load_template()
    options, _ = dialog.composition()
    assert not options.group_ids
    assert "quality" in options.sections
    assert "comparison" not in options.sections
    formation.name = "Произвольная колонка"
    announcements = []
    monkeypatch.setattr(dialog, "announce", announcements.append)
    dialog.apply_builtin_template("Паспорт скважины")
    assert dialog.composition()[0].group_ids == (well.id,)
    assert announcements == [dialog.status.text()]
    assert announcements[0].endswith("Укажите колонки группировки вручную: пласт.")
    window.dirty = False


@pytest.mark.parametrize("interface_language", ["ru", "en"])
def test_report_history_archives_and_compares_snapshots(qtbot, monkeypatch, tmp_path, interface_language):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.path = tmp_path / "project.apyro"
    project = window.model.project
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    paths = iter((tmp_path / "before.json", tmp_path / "after.json"))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(next(paths)), "JSON"))
    dialog.scope.setCurrentIndex(dialog.scope.findData("selected"))
    dialog.build()
    dialog.archive_report()
    dialog.scope.setCurrentIndex(dialog.scope.findData("all"))
    dialog.title.setText("Обновлённый выпуск")
    dialog.limit.setValue(20)
    dialog.section_order = ("summary", "data")
    dialog.build()
    dialog.archive_report()
    entries = project.view["report_history"]
    before, after = [load_snapshot(entry, project.id) for entry in entries]
    diff = compare_snapshots(before, after)
    assert diff["added"] == [row.id for row in project.rows]
    assert not diff["removed"] and not diff["changed"]
    assert diff["statistics"]
    assert {"Заголовок", "Разделы и порядок", "Лимит строк приложения"} <= {
        name for name, _, _ in diff["composition"]
    }
    assert entries[0]["relative_path"] == "before.json"
    moved = tmp_path / "moved"
    moved.mkdir()
    path = (tmp_path / "before.json").rename(moved / "before.json")
    assert load_snapshot(entries[0], project.id, moved / "project.apyro") == before
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    choices = iter((invalid, path))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(next(choices)), "JSON"))
    real_exec = QDialog.exec
    main = get_ident()
    threads = []

    def load_in_background(*args, **kwargs):
        threads.append(get_ident())
        return load_snapshot(*args, **kwargs)

    def interact(popup):
        if popup.windowTitle() != "История отчётов":
            return real_exec(popup)
        result = popup.findChild(QTextEdit)
        controls = popup.findChildren(QPushButton)
        locate = next(button for button in controls if button.text() == "Указать файл…")
        locate.click()
        assert "не перепривязан" in result.toPlainText()
        assert entries[0]["relative_path"] == "before.json"
        locate.click()
        assert entries[0]["relative_path"] == "moved/before.json"
        next(button for button in controls if button.text() == "Сравнить").click()
        text = result.toPlainText()
        if interface_language == "en":
            assert "Content and layout changes" in text
            assert "Title · " in text and "Appendix row limit · " in text
            assert "Statistics changes" in text and "Numeric N · " in text
            assert "Изменения состава и оформления" not in text
        else:
            assert "Изменения состава и оформления" in text
        assert "Обновлённый выпуск" in text
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(report_export, "load_snapshot", load_in_background)
    monkeypatch.setattr(report_export, "language", lambda: interface_language)
    monkeypatch.setattr(QDialog, "exec", interact)
    dialog.show_history()
    assert len(threads) == 4 and all(thread != main for thread in threads)
    window.dirty = False


@pytest.mark.parametrize(("template", "section"), [
    ("Характеристика пласта", "Сопоставление объектов"),
    ("Контроль качества", "Замечания QC"),
])
def test_demo_other_builtin_templates_render(qtbot, monkeypatch, template, section):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: (
        next(label for label in args[3] if label.endswith(template)), True
    ))
    dialog.load_template()
    dialog.build()
    assert dialog.report is not None, dialog.status.text()
    assert len(dialog.report["row_ids"]) == 5
    preview = dialog.preview.toPlainText()
    assert template in preview and section in preview
    window.dirty = False


def test_passport_recognizes_real_import_headers_and_separates_depth_datums(qtbot):
    headers = ["Well / Well Name", "Sample / Top MD [m]", "Sample / Base MD [m]",
               "Sample / MD datum", "Geol / Alt Top Lstrat(Fm)", "Geol / Alt Base Lstrat(Fm)",
               "Pyrol6 / TOC [%wt]"]
    data = [
        ["A", 100, 110, "RKB", "F1", "F2", 1],
        ["A", 90, 95, "MSL", "F2", "F2", 2],
        ["B", 140, 150, "RKB", "F1", "F1", 3],
    ]
    settings = [(i, name, "number" if i in (1, 2, 6) else "text",
                 {1: "m", 2: "m", 6: "%wt"}.get(i, "")) for i, name in enumerate(headers)]
    project = build_project(headers, data, [1, 2, 3], settings, {})
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    dialog.apply_builtin_template("Паспорт скважины")
    assert dialog.composition()[0].group_ids == (project.columns[0].id, project.columns[4].id)
    dialog.build()
    assert dialog.report is not None, dialog.status.text()
    profiles = dialog.report["well_profiles"]
    assert [profile["depth_range"] for profile in profiles] == [[100, 110], [90, 95], [140, 150]]
    assert all(profile["interval_samples"] == 1 for profile in profiles)
    assert all(profile["spec"]["track_ids"] == [project.columns[6].id] for profile in profiles)
    assert dialog.report["overview"]["mixed_depth_datums"]
    assert dialog.report["overview"]["depth"] is None
    assert dialog.report["overview"]["indicators"][0]["n"] == 3
    assert "квадрат — на подошве" in dialog.preview.toPlainText()
    window.dirty = False


@pytest.mark.parametrize("kind", ["HTML", "PDF", "DOCX", "JSON"])
def test_demo_passport_preview_and_batch_exports(qtbot, monkeypatch, tmp_path, kind):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    dialog = ReportDialog(window)
    qtbot.addWidget(dialog)
    choices = iter((None, "1. Скважина"))
    monkeypatch.setattr(dialog, "choose_batch_formats", lambda: (kind.lower(),))

    def choose(*args, **kwargs):
        value = next(choices)
        return (value or next(label for label in args[3] if label.endswith("Паспорт скважины"))), True

    monkeypatch.setattr(QInputDialog, "getItem", choose)
    dialog.load_template()
    dialog.build()
    preview = dialog.preview.toPlainText()
    assert "Глубинный профиль: A" in preview and "Глубинный профиль: B" in preview
    assert len(dialog.report["row_ids"]) == 5
    assert "графиков: 4" in dialog.status.text()
    assert preview.count("\ufffc") == 4
    snapshot = dialog.report

    def accept_preview(popup):
        if popup.windowTitle() == "Готовность пакета":
            assert not list(tmp_path.iterdir())
            details = popup.findChild(QTextEdit)
            assert details.accessibleName() == "Результаты проверки файлов пакета"
            assert details.accessibleDescription().startswith("Файлов: 2.")
            return QDialog.DialogCode.Accepted
        if popup.windowTitle() != "Файлы пакета":
            return real_exec(popup)
        files = popup.findChild(QListWidget)
        assert files.accessibleName() == "Файлы для экспорта"
        assert [files.item(i).text() for i in range(files.count())] == [
            f"001-A.{kind.lower()} · проб: 3", f"002-B.{kind.lower()} · проб: 2",
        ]
        ok = popup.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok)
        assert ok.isEnabled()
        for i in range(files.count()):
            files.item(i).setCheckState(Qt.CheckState.Unchecked)
        assert not ok.isEnabled()
        for i in range(files.count()):
            files.item(i).setCheckState(Qt.CheckState.Checked)
        assert ok.isEnabled()
        return QDialog.DialogCode.Accepted

    real_exec = QDialog.exec
    monkeypatch.setattr(QDialog, "exec", accept_preview)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    announcements = []
    monkeypatch.setattr(dialog, "announce", announcements.append)
    dialog.export_batch()
    assert dialog.report is snapshot
    assert announcements == [dialog.status.text()]
    assert announcements[0].startswith("Сохранено файлов: 2.")
    files = sorted(tmp_path.glob(f"reports-*/*.{kind.lower()}"))
    assert len(files) == 2, dialog.status.text()
    for path, well, count in zip(files, ("A", "B"), (3, 2)):
        expected = f"Глубинный профиль: {well}"
        if kind == "JSON":
            report = json.loads(path.read_text(encoding="utf-8"))
            well_id = window.model.project.columns[0].id
            assert report["row_ids"] == [row.id for row in window.model.project.rows if row.values[well_id] == well]
            assert report["selection_context"]["batch_group"]["value"] == well
            assert [profile["name"] for profile in report["well_profiles"]] == [expected]
            assert "empty_comparison" not in {item["code"] for item in report["release_checks"]}
        else:
            if kind == "HTML":
                body = path.read_text(encoding="utf-8")
                assert "data:image/png;base64," in body
            elif kind == "PDF":
                body = "\n".join(page.extract_text() for page in PdfReader(path).pages)
            else:
                document = Document(path)
                body = "\n".join(paragraph.text for paragraph in document.paragraphs)
                assert document.inline_shapes
            assert expected in body
            assert "Проб в выборке" in body and str(count) in body
            assert "Графики сопоставления требуют" not in body
    window.dirty = False
