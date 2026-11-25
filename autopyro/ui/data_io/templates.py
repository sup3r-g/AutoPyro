"""Reusable import templates stored with projects and exchanged as declarative JSON."""


from ..i18n import tr, message
from copy import deepcopy

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QInputDialog,
    QFileDialog,
    QMessageBox,
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QDialogButtonBox,
)

from autopyro.data_io.import_templates import library_key, load_template, save_template, store
from ..widgets.parameters import add_parameter_tools
from ..theme.controls import apply_control_icons
from ..theme import icon


def template_state(view):
    return {key: deepcopy(view.get(key, {})) for key in ("import_templates", "import_template_revisions")}


def export_template(parent, template):
    path, _ = QFileDialog.getSaveFileName(parent, tr('Экспорт шаблона'), "import-template.json", "JSON (*.json)")
    if path:
        from pathlib import Path
        if Path(path).suffix.lower() != ".json":
            path += ".json"
        save_template(path, template)


class TemplateBar(QWidget):
    applyRequested = Signal(object)
    saveRequested = Signal(str)

    def __init__(self, stage, view, parent=None):
        super().__init__(parent)
        self.stage, self.state = stage, template_state(view)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(12)
        layout.addWidget(QLabel(tr('Шаблон:')))
        layout.addWidget(self.combo, 1)
        for label, callback, image in ((tr('Применить'), self.apply, 'check'), (tr('Сохранить…'), self.save, 'save'),
                                (tr('Загрузить JSON…'), self.load, 'import'), (tr('Экспорт JSON…'), self.export, 'log-out')):
            button = QPushButton(icon(image), label)
            button.clicked.connect(callback)
            layout.addWidget(button)
        self.refresh()
        apply_control_icons(self)

    def refresh(self, selected=None):
        selected = selected or self.combo.currentData()
        self.combo.clear()
        self.combo.addItem(tr('Без шаблона'), None)
        for key, template in self.state["import_templates"].items():
            if template["stage"] == self.stage:
                self.combo.addItem(tr('{0} · редакция {1}', template['name'], template['revision']), key)
        self.combo.setCurrentIndex(max(0, self.combo.findData(selected)))

    def put(self, template):
        store(self.state, template)
        self.refresh(library_key(template))

    def selected(self):
        return self.state["import_templates"].get(self.combo.currentData())

    def apply(self):
        if self.selected():
            self.applyRequested.emit(deepcopy(self.selected()))

    def save(self):
        name, ok = QInputDialog.getText(self, tr('Сохранить шаблон'), tr('Название (совпадающее имя создаёт новую редакцию)'),
                                       text=self.selected()["name"] if self.selected() else "")
        if ok and name.strip():
            self.saveRequested.emit(name.strip())

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('Загрузить шаблон'), "", "JSON (*.json)")
        if path:
            try:
                template = load_template(path)
                if template["stage"] != self.stage:
                    raise ValueError(tr('Шаблон относится к другому этапу импорта'))
                self.put(template)
            except (ValueError, OSError, TypeError) as exc:
                QMessageBox.warning(self, tr('Шаблон'), message(exc))

    def export(self):
        if self.selected():
            try:
                export_template(self, self.selected())
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self, tr('Шаблон'), message(exc))


class TemplateLibraryDialog(QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner, self.state = owner, template_state(owner.model.project.view)
        self.setWindowTitle(tr('Шаблоны импорта · библиотека проекта'))
        self.resize(850, 520)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(tr('JSON содержит только настройки. Старые редакции сохраняются для воспроизведения импортов.')))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([tr('Название'), tr('Этап'), tr('Редакция')])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)
        actions = QHBoxLayout()
        for label, callback, image in ((tr('Загрузить JSON…'), self.load, 'import'), (tr('Экспорт JSON…'), self.export, 'log-out'), (tr('Удалить из списка'), self.remove, 'trash')):
            button = QPushButton(icon(image), label)
            button.clicked.connect(callback)
            actions.addWidget(button)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('Применить библиотеку'))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('Отмена'))
        buttons.accepted.connect(self.apply_library)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.reload()
        add_parameter_tools(self, help_only=True)

    def reload(self):
        self.table.setRowCount(len(self.state["import_templates"]))
        for i, (key, template) in enumerate(self.state["import_templates"].items()):
            for column, value in enumerate((template["name"], tr('Чтение файла') if template["stage"] == "reader" else tr('Добавление в проект'), template["revision"])):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, key)
                self.table.setItem(i, column, item)
        self.table.resizeColumnsToContents()

    def selected(self):
        item = self.table.item(self.table.currentRow(), 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('Загрузить шаблон'), "", "JSON (*.json)")
        if path:
            try:
                store(self.state, load_template(path))
                self.reload()
            except (ValueError, OSError, TypeError) as exc:
                QMessageBox.warning(self, tr('Шаблон'), message(exc))

    def export(self):
        if self.selected():
            try:
                export_template(self, self.state["import_templates"][self.selected()])
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self, tr('Шаблон'), message(exc))

    def remove(self):
        if self.selected():
            self.state["import_templates"].pop(self.selected())
            self.reload()

    def apply_library(self):
        from ..table import TemplateLibraryCommand
        try:
            self.owner.model.undo.push(TemplateLibraryCommand(self.owner.model, self.state))
            self.accept()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, tr('Шаблон'), message(exc))
