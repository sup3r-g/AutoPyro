"""Own the cached analysis documents for the current project."""

from PySide6.QtCore import QObject, Qt
from shiboken6 import isValid

from .document_descriptors import DOCUMENTS, document_descriptor


class AnalysisDocuments(QObject):
    """Reuse closed worksheets within a project and retire them on replacement.

    Documents with background work can implement ``prepare_project_close`` to
    cancel work or suppress its results. They report whether closing must wait
    through ``task_running`` and notify us via ``project_tasks_finished``.
    ``project_close_blocker`` optionally vetoes an interactive replacement.
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        for descriptor in DOCUMENTS.values():
            setattr(self, descriptor.slot, None)
        self.geochemistry_windows = {}
        self._retiring = set()

    def cached(self, descriptor):
        if descriptor.profile:
            return self.geochemistry_windows.get(descriptor.profile)
        return getattr(self, descriptor.slot)

    def _documents(self):
        cached = (*self.geochemistry_windows.values(), *(self.cached(item) for item in DOCUMENTS.values()))
        return tuple(dict.fromkeys(document for document in cached if document is not None))

    def project_close_blocker(self):
        for document in self._documents():
            blocker = getattr(document, 'project_close_blocker', None)
            if blocker is not None and (title := blocker()):
                return title
        return None

    def shutdown_blocker(self):
        for document in (*self._documents(), *self._retiring):
            if getattr(document, 'task_running', False):
                return document.windowTitle()
        return None

    def close_for_project_replacement(self):
        """Retire the old worksheets while they still belong to the old project."""
        self.window.desktop.clear_documents()
        for document in self._documents():
            self._close_project_document(document)
        self.geochemistry_windows.clear()
        for descriptor in DOCUMENTS.values():
            setattr(self, descriptor.slot, None)

    def _close_project_document(self, document):
        prepare = getattr(document, 'prepare_project_close', None)
        if prepare is not None:
            prepare()

        def close_when_ready():
            if not isValid(document) or getattr(document, 'task_running', False):
                return
            self._retiring.discard(document)
            # Let Qt delete the widget only after it accepts a close event.
            document.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            if not document.close():
                document.hide()

        if getattr(document, 'task_running', False):
            self._retiring.add(document)
            document.project_tasks_finished.connect(close_when_ready)
            document.hide()
        else:
            close_when_ready()

    def invalidate_after_edit(self):
        for document in (self.interpolation_window, self.calculation_window, self.curve_fitting_window):
            if document is not None:
                document.invalidate()
        if self.stratigraphy_window is not None:
            self.stratigraphy_window.invalidate_preview()

    def open(self, kind, profile=''):
        descriptor = document_descriptor(kind, profile)
        if descriptor is None:
            raise KeyError((kind, profile))
        window = self.window
        document = self.cached(descriptor)
        if document is None:
            dialog_class = descriptor.dialog_class()
            if descriptor.profile:
                document = self.geochemistry_windows[descriptor.profile] = dialog_class(window, descriptor.profile)
            elif kind == 'interpolation':
                document = dialog_class(window.model.project, set(window.row_ids()), window)
                window.model.columnsInserted.connect(document.refresh_columns)
                window.model.columnsRemoved.connect(document.refresh_columns)
                window.model.modelReset.connect(document.refresh_columns)
            else:
                document = dialog_class(window)
                if kind == 'interpretation':
                    document.scheme_saved.connect(window.interpretation_scheme_saved)
            if not descriptor.profile:
                setattr(self, descriptor.slot, document)
        elif kind == 'interpolation':
            document.refresh_columns()

        if kind in ('statistics', 'plot', 'map', 'interpretation'):
            document.closed = False
        if kind == 'plot':
            document.preview.warmup()
        window.desktop.open_document(kind, document, descriptor.title(), profile=descriptor.profile, icon_name=descriptor.icon)
        if kind == 'statistics':
            document.request_refresh()
        elif kind in ('plot', 'map', 'interpretation'):
            document.schedule()

    def show_report(self):
        from .dialogs.report import ReportDialog

        dialog = ReportDialog(self.window)
        try:
            dialog.exec()
        finally:
            self.window.analysis_scope.unbind(dialog.scope)
            dialog.deleteLater()

    def interpretation_scheme_saved(self):
        self.window.model_changed()
        if self.plot_window is not None:
            self.plot_window.schedule()

    def show_history(self):
        from .dialogs.history import HistoryDialog
        dialog = HistoryDialog(self.window.model.project, self.window)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()
