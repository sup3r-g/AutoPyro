"""Named appearance presets stored in the project, separate from interval data."""

from copy import deepcopy

from PySide6.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox,
    QPushButton, QVBoxLayout,
)

from autopyro.geology.stratigraphy import column_style
from ..i18n import message, tr
from ..theme import icon


TEMPLATES_KEY = 'stratigraphy_style_templates'
# These fields describe the current section and its rocks, not a reusable appearance.
CONTEXT_FIELDS = {'title', 'depth_min', 'depth_max', 'hierarchy_levels', 'correlation',
                  'comparison_wells', 'lithologies'}


def template_style(options):
    """Validate a preset and complete older presets with current style defaults."""
    return {key: value for key, value in column_style(options).items() if key not in CONTEXT_FIELDS}


class ColumnStyleTemplates(QGroupBox):
    def __init__(self, dialog):
        super().__init__(tr('Шаблоны оформления'), dialog)
        self.dialog = dialog
        layout = QVBoxLayout(self)
        self.combo = QComboBox()
        self.combo.setMinimumContentsLength(12)
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        layout.addWidget(self.combo)
        row = QHBoxLayout()
        self.apply_button = QPushButton(icon('check'), tr('Применить'))
        self.save_button = QPushButton(icon('save'), tr('Сохранить как…'))
        self.delete_button = QPushButton(icon('trash'), tr('Удалить'))
        for button in (self.apply_button, self.save_button, self.delete_button):
            button.setAutoDefault(False)
            row.addWidget(button)
        layout.addLayout(row)
        hint = QLabel(tr('Шаблоны хранятся в проекте. Применяется только оформление; стволы, уровни, пределы глубин, заголовок и свои породы остаются текущими.'))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.feedback = QLabel()
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.combo.currentIndexChanged.connect(self.update_buttons)
        self.apply_button.clicked.connect(self.apply_template)
        self.save_button.clicked.connect(self.save_template)
        self.delete_button.clicked.connect(self.delete_template)
        self.refresh()

    def templates(self):
        return self.dialog.owner.model.project.view.get(TEMPLATES_KEY, {})

    def refresh(self, selected=None):
        self.combo.clear()
        self.combo.addItem(tr('Выберите шаблон оформления'), None)
        for name in sorted(self.templates(), key=str.casefold):
            self.combo.addItem(name, name)
        self.combo.setCurrentIndex(max(0, self.combo.findData(selected)))
        self.update_buttons()

    def update_buttons(self):
        selected = self.combo.currentData() in self.templates()
        self.apply_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)

    def store(self, templates, selected=None):
        self.dialog.owner.model.project.view[TEMPLATES_KEY] = templates
        self.dialog.owner.view_changed()
        self.refresh(selected)

    def save_template(self):
        try:
            style = template_style(self.dialog.style_values())
            name, ok = QInputDialog.getText(
                self, tr('Сохранить шаблон оформления'), tr('Название шаблона'),
                text=self.combo.currentData() or '')
            if not ok:
                return
            name = name.strip()
            if not name:
                raise ValueError(tr('Укажите название шаблона.'))
            templates = deepcopy(self.templates())
            if name in templates and QMessageBox.question(
                    self, tr('Заменить шаблон?'),
                    tr('Заменить оформление в шаблоне «{0}»?', name),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
            templates[name] = style
            self.store(templates, name)
            self.feedback.setText(tr('Шаблон «{0}» сохранён. Сохраните проект, чтобы записать его на диск.', name))
        except Exception as exc:
            self.feedback.setText(tr('Ошибка: {0}', message(exc)))

    def apply_template(self):
        name = self.combo.currentData()
        if name not in self.templates():
            return
        try:
            style = template_style(self.templates()[name])
            form = self.dialog.column_style_form
            form.load(style)
            # One signal schedules a single debounced preview after all controls are loaded.
            form.edited.emit()
            self.feedback.setText(tr('Шаблон «{0}» применён к текущему оформлению.', name))
        except Exception as exc:
            self.feedback.setText(tr('Ошибка: {0}', message(exc)))

    def delete_template(self):
        name = self.combo.currentData()
        if name not in self.templates():
            return
        if QMessageBox.question(self, tr('Удалить шаблон?'),
                                tr('Удалить шаблон «{0}»? Текущее оформление сохранится.', name),
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        templates = deepcopy(self.templates())
        del templates[name]
        self.store(templates)
        self.feedback.setText(tr('Шаблон «{0}» удалён.', name))
