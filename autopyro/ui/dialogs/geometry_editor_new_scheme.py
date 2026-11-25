"""Create new interpretation schemes and collect their initial metadata."""

from __future__ import annotations

from uuid import uuid4

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from ..i18n import tr
from ..widgets.parameters import add_parameter_tools


def new_scheme(x_property="X", x_unit="1", y_property="Y", y_unit="1"):
    return {
        "format": "autopyro.interpretation",
        "version": 1,
        "id": f"user-{uuid4().hex[:12]}",
        "revision": "1",
        "name": tr('Новая схема'),
        "source": "",
        "scientific_status": "user_supplied",
        "coordinate_space": "analysis",
        "boundary_policy": "flag",
        "tolerance": 0,
        "x": {
            "property": x_property,
            "unit": x_unit,
            "transform": "identity",
            "origin": 0,
            "scale": 1,
        },
        "y": {
            "property": y_property,
            "unit": y_unit,
            "transform": "identity",
            "origin": 0,
            "scale": 1,
        },
        "regions": [],
        "lines": [],
    }


class NewSchemeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('Новая интерпретационная схема'))
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.fields = {}
        for key, label in [
            ("name", tr('Название')),
            ("source", tr('Источник / автор / обоснование')),
            ("x_property", tr('Показатель X')),
            ("x_unit", tr('Единица X')),
            ("y_property", tr('Показатель Y')),
            ("y_unit", tr('Единица Y')),
        ]:
            self.fields[key] = QLineEdit()
            form.addRow(label, self.fields[key])
        self.transforms = {}
        for key in ("x", "y"):
            combo = QComboBox()
            combo.addItem(tr('Без преобразования'), "identity")
            combo.addItem("log10(value / reference)", "log10")
            self.transforms[key] = combo
            form.addRow(tr('Преобразование {0}', key.upper()), combo)
        layout.addLayout(form)
        note = QLabel(
            tr('Контуры задаются в analysis: при log10 — в log10(value / 1), при identity — в исходных единицах. Начальные origin=0, scale=1. Дополнительные преобразования доступны в JSON-редакторе.')
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        add_parameter_tools(self)

    def apply(self):
        values = {key: widget.text().strip() for key, widget in self.fields.items()}
        if not all(values.values()):
            QMessageBox.warning(
                self, tr('Схема'), tr('Заполните название, источник, показатели и единицы')
            )
            return
        self.scheme = new_scheme(
            **{
                key: values[key]
                for key in ("x_property", "x_unit", "y_property", "y_unit")
            }
        )
        self.scheme.update(name=values["name"], source=values["source"])
        for key, combo in self.transforms.items():
            self.scheme[key]["transform"] = combo.currentData()
        self.accept()
