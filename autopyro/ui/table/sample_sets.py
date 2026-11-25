"""Manage named sets of samples in the project table."""

from PySide6.QtWidgets import QInputDialog, QMessageBox

from autopyro.model.sample_sets import create_sample_set

from ..i18n import message, tr


class SampleSetActions:
    """Own the applied set and the filter to restore when it closes."""

    def __init__(self, window):
        self.window = window
        self.reset()

    def reset(self):
        self.active_name = None
        self.return_filter = None

    @property
    def saved_sets(self):
        return self.window.model.project.view.get("sample_sets", {})

    def row_ids(self):
        return self.saved_sets.get(self.active_name, ()) if self.active_name else None

    def capture_into(self, view):
        if self.active_name is not None:
            view["sample_set"] = self.active_name
            if self.return_filter is not None:
                view["sample_set_return_filter"] = self.return_filter

    def restore_from(self, view):
        name = view.get("sample_set")
        self.active_name = name if isinstance(name, str) and name else None
        self.return_filter = view.get("sample_set_return_filter") if self.active_name else None
        return self.row_ids()

    def save_sample_set(self):
        row_ids = self.window.sample_selection.ids(ordered=True)
        if not row_ids:
            QMessageBox.information(self.window, tr('Наборы проб'), tr('Выделите пробы в таблице.'))
            return
        name, accepted = QInputDialog.getText(self.window, tr('Наборы проб'), tr('Название набора'))
        if not accepted:
            return
        try:
            create_sample_set(self.window.model.project.view, name, row_ids)
        except ValueError as exc:
            QMessageBox.warning(self.window, tr('Наборы проб'), message(exc))
            return
        self.window.analysis_scope.refresh()
        self.window.view_changed()

    def choose_sample_set(self, title):
        sets = self.saved_sets
        if not sets:
            QMessageBox.information(self.window, tr('Наборы проб'), tr('Сохранённых наборов проб пока нет.'))
            return None
        name, accepted = QInputDialog.getItem(
            self.window, title, tr('Набор проб'), sorted(sets), editable=False,
        )
        return name if accepted else None

    def open_sample_set(self):
        name = self.window.choose_sample_set(tr('Открыть набор проб…'))
        if name is None:
            return
        row_ids = self.saved_sets[name]
        if self.active_name is None:
            self.return_filter = {"rules": self.window.proxy.rules, "combine": self.window.proxy.combine}
        self.active_name = name
        self.window.table_filter_sort.restore_filter([], "AND", row_ids=row_ids)
        self.window.view_changed()
        missing = len(row_ids) - self.window.proxy.rowCount()
        if missing:
            QMessageBox.information(self.window, tr('Наборы проб'), tr('Проб в наборе, отсутствующих в проекте: {0}', missing))

    def close_sample_set(self):
        previous = self.return_filter
        self.reset()
        rules = previous["rules"] if previous is not None else self.window.proxy.rules
        combine = previous["combine"] if previous is not None else self.window.proxy.combine
        self.window.table_filter_sort.restore_filter(rules, combine, row_ids=None)
        self.window.view_changed()

    def compare_sample_sets(self):
        from ..dialogs.sample_sets import SampleSetComparisonDialog

        sets = self.saved_sets
        if len(sets) < 2:
            QMessageBox.information(self.window, tr('Сравнить наборы проб…'), tr('Сохраните хотя бы два набора проб.'))
            return
        dialog = SampleSetComparisonDialog(self.window, sets)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def rename_sample_set(self):
        title = tr('Переименовать набор проб…')
        name = self.window.choose_sample_set(title)
        if name is None:
            return
        new_name, accepted = QInputDialog.getText(self.window, title, tr('Новое название набора'), text=name)
        if accepted:
            try:
                self.window.analysis_scope.rename_sample_set(name, new_name)
            except ValueError as exc:
                QMessageBox.warning(self.window, title, message(exc))

    def delete_sample_set(self):
        title = tr('Удалить набор проб…')
        name = self.window.choose_sample_set(title)
        if name is None:
            return
        try:
            self.window.analysis_scope.check_sample_set_deletion(name)
        except ValueError as exc:
            QMessageBox.warning(self.window, title, message(exc))
            return
        prompt = tr('Удалить набор «{0}»? Связанные настройки анализа переключатся на текущую таблицу.', name)
        dependencies = self.window.analysis_scope.sample_set_dependencies(name)
        if dependencies:
            prompt += "\n\n" + tr('Затронутые настройки') + "\n" + "\n".join(dependencies)
        answer = QMessageBox.question(
            self.window, title, prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            try:
                self.window.analysis_scope.delete_sample_set(name)
            except ValueError as exc:
                QMessageBox.warning(self.window, title, message(exc))
