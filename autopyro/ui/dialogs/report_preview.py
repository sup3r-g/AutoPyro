"""Build frozen report snapshots and display HTML and paginated previews."""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4
import base64

from PySide6.QtCore import QBuffer, QIODevice, QPointF, QSignalBlocker, QTemporaryDir, QUrl, Qt
from PySide6.QtGui import QDesktopServices, QImage, QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from autopyro.reports.reporting import (
    build_report,
    encode_report,
    freeze_report_source,
    render_html,
    release_blockers,
    write_report_file,
)

from ..i18n import message, tr
from ..tasks import TaskCancelled, run_with_progress


def resize_to_screen(dialog, width, height):
    screen = dialog.screen().availableGeometry()
    dialog.resize(min(width, screen.width() - 32), min(height, screen.height() - 48))


class ReportPreview(QTextBrowser):
    def go_to_section(self, anchor):
        block = self.document().begin()
        while block.isValid():
            fragments = block.begin()
            while not fragments.atEnd():
                fragment = fragments.fragment()
                if anchor in fragment.charFormat().anchorNames():
                    cursor = QTextCursor(block)
                    cursor.setPosition(fragment.position())
                    self.setTextCursor(cursor)
                    self.scrollToAnchor(anchor)
                    return
                fragments += 1
            block = block.next()

    def loadResource(self, type, url):
        value = url.toString()
        if type == QTextDocument.ResourceType.ImageResource and value.startswith(
            "data:image/png;base64,"
        ):
            image = QImage.fromData(base64.b64decode(value.split(",", 1)[1]))
            return (
                image.scaledToWidth(720, Qt.TransformationMode.SmoothTransformation)
                if image.width() > 720
                else image
            )
        # Reports are self-contained; do not fetch file or network resources.
        return None


def focus_preview_sections(sheet):
    if sheet.preview_sections.isEnabled():
        sheet.preview_sections.setFocus(Qt.FocusReason.ShortcutFocusReason)


def invalidate(sheet, *_):
    was_current = sheet.report is not None
    sheet.report = None
    sheet.report_source = None
    sheet.check_actions.hide()
    sheet.page_preview.setEnabled(False)
    for button in sheet.exports:
        button.setEnabled(False)
    sheet.preview_stale.setVisible(sheet.preview_report is not None)
    sheet.preview.setAccessibleDescription(sheet.preview_stale.text() if sheet.preview_report is not None else "")
    sheet.status.setText(tr('Настройки изменены. Соберите новый снимок отчёта.'))
    if was_current:
        sheet.announce(sheet.preview_stale.text())


def build(sheet):
    sheet.invalidate()
    try:
        project = sheet.owner.model.project
        scope = sheet.scope.currentData()
        ids = sheet.owner.analysis_scope.rows()
        options, plots = deepcopy(sheet.composition())
        context = sheet.selection_context(scope, ids)
        def assemble(update):
            update(tr('Фиксация источника отчёта…'), 0)
            frozen = freeze_report_source(project, ids, options, plots)
            report = build_report(
                frozen, ids, options, plots, selection_context=context, progress=update,
            )
            update(tr('Подготовка предпросмотра…'), 98)
            navigation = []
            return report, render_html(report, navigation=navigation), navigation, frozen

        report, preview, navigation, frozen = run_with_progress(
            sheet, tr('Сборка отчёта'), assemble, cancellable=True,
        )
        sheet.preview.setHtml(preview)
        sheet.preview_sections.clear()
        sheet.preview_sections.addItem(tr('Начало отчёта'), "report-top")
        for anchor, label in navigation:
            sheet.preview_sections.addItem(label, anchor)
        sheet.preview_sections.setCurrentIndex(-1)
        sheet.preview_sections.setEnabled(True)
        sheet.preview_stale.hide()
        sheet.preview.setAccessibleDescription("")
        sheet.report = sheet.preview_report = report
        sheet.report_source = (frozen, options, plots)
        sheet.update_check_actions(report)
        figure_count = sum(len(sheet.report.get(key, ())) for key in (
            "well_profiles", "comparison_figures", "joint_curve_figures",
        ))
        if options.sections is None or "figures" in options.sections:
            figure_count += len(sheet.report["figures"]) + len(sheet.report.get("map_figures", ()))
        blockers = release_blockers(sheet.report)
        warnings = sum(check.get("severity", "warning") == "warning"
                       for check in sheet.report["release_checks"])
        sheet.status.setText(tr(
            'Снимок: {0} проб; графиков: {1}. Замечаний: {2}. Блокирующих проверок: {3}.',
            len(ids), figure_count, warnings, len(blockers),
        ))
        for button in sheet.exports:
            button.setEnabled(not blockers)
        sheet.page_preview.setEnabled(not blockers)
        sheet.announce(sheet.status.text())
    except TaskCancelled:
        sheet.announce_status(tr('Сборка отменена. Настройки отчёта сохранены.'))
    except Exception as exc:
        sheet.report = None
        sheet.announce_status(tr('Ошибка сборки: {0}', message(exc)))


def update_check_actions(sheet, report):
    targets = {
        "empty_selection": sheet.scope,
        "empty_profile": sheet.profile_depth,
        "empty_comparison": sheet.group,
        "empty_figure": sheet.plots,
        "no_numeric_data": sheet.columns,
        "truncated_appendix": sheet.limit,
        "truncated_qc": sheet.qc_limit,
        "truncated_protocol": sheet.protocol_limit,
        "joint_fit_snapshot_missing": sheet.joint_snapshot,
        "joint_fit_outside_scope": sheet.joint_snapshot,
        "stale_joint_fit": sheet.joint_snapshot,
    }
    sheet.check_menu.clear()
    for check in report["release_checks"]:
        if (target := targets.get(check["code"])) is not None and target.isEnabled():
            sheet.check_menu.addAction(
                check["message"], lambda checked=False, widget=target: widget.setFocus()
            )
    if any(check["code"] == "no_report_sections" for check in report["release_checks"]):
        sheet.check_menu.addAction(tr('Выбрать разделы отчёта'), sheet.edit_sections)
    sheet.check_actions.setVisible(bool(sheet.check_menu.actions()))


def preview_pages(sheet, kind):
    snapshot = sheet.report
    if snapshot is None or release_blockers(snapshot):
        return
    try:
        payload = run_with_progress(
            sheet, tr('Подготовка предпросмотра {0}…', kind.upper()),
            lambda: encode_report(snapshot, kind),
        )
        if sheet.report is not snapshot:
            return
        if kind == "pdf":
            sheet.show_pdf_pages(payload)
        else:
            app = QApplication.instance()
            directory = getattr(app, "_report_preview_directory", None)
            if directory is None:
                directory = app._report_preview_directory = QTemporaryDir()
            if not directory.isValid():
                raise OSError(directory.errorString())
            path = directory.filePath(f"report-preview-{uuid4().hex}.docx")
            write_report_file(path, payload)
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
                raise OSError(tr('Не удалось открыть DOCX. Проверьте приложение по умолчанию для этого формата.'))
            sheet.announce_status(tr('В редакторе открыта временная копия DOCX. Для выпуска используйте «Экспорт».'))
    except Exception as exc:
        QMessageBox.warning(sheet, tr('Предпросмотр отчёта'), message(exc))


def show_pdf_pages(sheet, payload):
    from PySide6.QtPdf import QPdfDocument
    from PySide6.QtPdfWidgets import QPdfView

    dialog = QDialog(sheet)
    dialog.setWindowTitle(tr('Предпросмотр PDF'))
    resize_to_screen(dialog, 900, 800)
    layout = QVBoxLayout(dialog)
    document = QPdfDocument(dialog)
    buffer = QBuffer(document)
    buffer.setData(payload)
    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
    document.load(buffer)
    if document.status() != QPdfDocument.Status.Ready or not document.pageCount():
        raise ValueError(tr('Не удалось загрузить страницы PDF.'))
    view = QPdfView()
    view.setAccessibleName(tr('Страницы отчёта PDF'))
    view.setAccessibleDescription(tr('Для чтения текста используйте основной предпросмотр отчёта.'))
    view.setDocument(document)
    view.setPageMode(QPdfView.PageMode.MultiPage)
    view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
    navigator = view.pageNavigator()
    controls = QHBoxLayout()
    page = QSpinBox()
    page.setRange(1, document.pageCount())
    page.setAccessibleName(tr('Номер страницы'))
    page.setAccessibleDescription(tr('Всего страниц: {0}.', document.pageCount()))
    controls.addWidget(QLabel(tr('Страница')))
    controls.addWidget(page)
    controls.addWidget(QLabel(tr('из {0}', document.pageCount())))
    zoom = QComboBox()
    zoom.addItem(tr('По ширине'), QPdfView.ZoomMode.FitToWidth)
    zoom.addItem(tr('Страница целиком'), QPdfView.ZoomMode.FitInView)
    zoom.setAccessibleName(tr('Масштаб PDF'))
    controls.addWidget(zoom)
    controls.addStretch()
    layout.addLayout(controls)
    layout.addWidget(view, 1)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    QWidget.setTabOrder(page, zoom)
    QWidget.setTabOrder(zoom, view)
    page.setFocus(Qt.FocusReason.OtherFocusReason)
    page.valueChanged.connect(lambda number: navigator.jump(number - 1, QPointF()))
    zoom.currentIndexChanged.connect(lambda: view.setZoomMode(zoom.currentData()))

    def sync_page(index):
        with QSignalBlocker(page):
            page.setValue(index + 1)

    navigator.currentPageChanged.connect(sync_page)
    try:
        dialog.exec()
    finally:
        view.setDocument(None)
        document.close()
