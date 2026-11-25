"""Edit positioned annotations as a single undoable change."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHeaderView, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from autopyro.model.core import parse_number
from autopyro.visualization.plot_annotations import validate_annotations
from ..i18n import tr, message


class PlotAnnotationsDialog(QDialog):
    def __init__(self, parent, annotations, tracks, *, position=None, is_3d=False, axis_scales=None):
        super().__init__(parent)
        self.setWindowTitle(tr('Аннотации на графике'))
        self.resize(850, 360)
        self.tracks = tracks
        self.is_3d = is_3d
        self.axis_scales = axis_scales or {}
        self.annotations = []
        layout = QVBoxLayout(self)
        help_text = QLabel(tr(r'Область осей: X и Y от 0 до 1, начало внизу слева. Для новой строки введите \n.'))
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        shape_help = QLabel(tr('Для стрелки X₂ и Y₂ задают наконечник, для фигуры — противоположный угол рамки.'))
        shape_help.setWordWrap(True)
        layout.addWidget(shape_help)
        if is_3d:
            shape_help.setText(tr('В 3D доступны текстовые подписи в области осей.'))
        self.table = QTableWidget(0, 8, self)
        self.table.setHorizontalHeaderLabels(
            [tr('Текст'), tr('Координаты'), 'X', 'Y', tr('Трек'), tr('Тип'), 'X₂', 'Y₂'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(4, 116)
        self.table.setColumnHidden(4, not tracks and not any(row.get('track_id') for row in annotations))
        layout.addWidget(self.table)
        for annotation in annotations:
            self.add_row(annotation)
        if position is not None:
            x, y, track_id = position
            self.add_row({'text': '', 'x': x, 'y': y, 'track_id': track_id, 'coordinates': 'axes'})
            self.table.editItem(self.table.item(self.table.rowCount() - 1, 0))
        controls = QHBoxLayout()
        add = QPushButton(tr('Добавить аннотацию'))
        add.clicked.connect(lambda: self.add_row())
        remove = QPushButton(tr('Удалить аннотацию'))
        remove.clicked.connect(self.remove_row)
        controls.addWidget(add)
        controls.addWidget(remove)
        controls.addStretch()
        layout.addLayout(controls)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_row(self, annotation=None):
        annotation = annotation or {'text': '', 'coordinates': 'axes', 'x': .5, 'y': .5}
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in ((0, annotation.get('text', '').replace('\n', r'\n')), (2, annotation['x']), (3, annotation['y'])):
            self.table.setItem(row, column, QTableWidgetItem(str(value)))
        mode = QComboBox()
        mode.addItem(tr('Область осей'), 'axes')
        mode.addItem(tr('Значения осей'), 'data')
        if self.is_3d:
            mode.model().item(mode.findData('data')).setEnabled(False)
        mode.setCurrentIndex(mode.findData(annotation['coordinates']))
        self.table.setCellWidget(row, 1, mode)
        target = QComboBox()
        target.addItem(tr('Первый трек'), '')
        for identifier, name in self.tracks:
            target.addItem(name, identifier)
        identifier = annotation.get('track_id', '')
        if target.findData(identifier) < 0:
            target.addItem(tr('Недоступный трек'), identifier)
            target.model().item(target.count() - 1).setEnabled(False)
        target.setCurrentIndex(target.findData(identifier))
        target.setToolTip(target.currentText())
        target.currentTextChanged.connect(target.setToolTip)
        self.table.setCellWidget(row, 4, target)
        kind = QComboBox()
        for label, key in ((tr('Подпись'), 'text'), (tr('Стрелка'), 'arrow'),
                           (tr('Прямоугольник'), 'rectangle'), (tr('Эллипс'), 'ellipse')):
            kind.addItem(label, key)
        kind.setCurrentIndex(kind.findData(annotation.get('kind', 'text')))
        if self.is_3d:
            for index in range(1, kind.count()):
                kind.model().item(index).setEnabled(False)
        self.table.setCellWidget(row, 5, kind)
        endpoints = [QTableWidgetItem(str(annotation.get(key, ''))) for key in ('x2', 'y2')]
        for column, item in enumerate(endpoints, 6):
            self.table.setItem(row, column, item)

        def update_endpoints():
            flags = Qt.ItemFlag.ItemIsSelectable
            if kind.currentData() != 'text':
                flags |= Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable
            for item in endpoints:
                item.setFlags(flags)

        kind.currentIndexChanged.connect(update_endpoints)
        update_endpoints()
        self.table.setCurrentCell(row, 0)

    def remove_row(self):
        if self.table.currentRow() >= 0:
            self.table.removeRow(self.table.currentRow())

    def accept(self):
        annotations = []
        for row in range(self.table.rowCount()):
            try:
                annotation = {
                    'text': self.table.item(row, 0).text().replace(r'\n', '\n'),
                    'coordinates': self.table.cellWidget(row, 1).currentData(),
                    'x': parse_number(self.table.item(row, 2).text()),
                    'y': parse_number(self.table.item(row, 3).text()),
                    'track_id': self.table.cellWidget(row, 4).currentData(),
                }
                kind = self.table.cellWidget(row, 5).currentData()
                if self.is_3d and (kind != 'text' or annotation['coordinates'] != 'axes'):
                    self.error.setText(tr('В 3D доступны текстовые подписи в области осей.'))
                    self.table.setCurrentCell(row, 5 if kind != 'text' else 1)
                    return
                if kind != 'text':
                    annotation.update(kind=kind, x2=parse_number(self.table.item(row, 6).text()),
                                      y2=parse_number(self.table.item(row, 7).text()))
                annotations.append(annotation)
            except ValueError:
                self.error.setText(tr('Проверьте координаты аннотации №{0}.', row + 1))
                self.table.setCurrentCell(row, 2)
                return
        try:
            validate_annotations(annotations)
        except ValueError as exc:
            self.error.setText(message(exc))
            return
        track_ids = {identifier for identifier, _ in self.tracks}
        for row, annotation in enumerate(annotations):
            if annotation['track_id'] and annotation['track_id'] not in track_ids:
                self.error.setText(tr('Трек аннотации №{0} недоступен. Выберите другой трек или удалите аннотацию.', row + 1))
                self.table.setCurrentCell(row, 4)
                return
            if annotation['coordinates'] != 'data':
                continue
            scales = self.axis_scales.get(annotation['track_id'], ('linear', 'linear'))
            for dimension, scale in zip(('x', 'y'), scales, strict=True):
                endpoints = (dimension, dimension + '2') if annotation.get('kind', 'text') != 'text' else (dimension,)
                if scale == 'log' and any(annotation[key] <= 0 for key in endpoints):
                    self.error.setText(tr('Координаты аннотации №{0} по логарифмической оси {1} должны быть положительными.',
                                          row + 1, dimension.upper()))
                    column = 2 if dimension == 'x' else 3
                    if annotation[dimension] > 0:
                        column += 4
                    self.table.setCurrentCell(row, column)
                    return
        self.annotations = annotations
        super().accept()
