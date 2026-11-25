"""Lazy plots for inspecting the joint fit and its vertical residuals."""

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QHeaderView, QLabel, QPushButton, QTableView, QTabWidget, QVBoxLayout, QWidget

from ..i18n import tr
from ..theme import icon
from ..widgets.figure_canvas import FittedFigureCanvas
from ..widgets.results import ResultsModel


class _ResidualRows:
    """Read only visible rows from the fit arrays."""

    def __init__(self, result, row_ids, indices):
        self.result, self.row_ids, self.indices = result, row_ids, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        fitted = self.result.fitted[index]
        return self.row_ids[index], fitted, self.result.y[index] - fitted


class JointPlots(QTabWidget):
    source_selected = Signal(object)

    def __init__(self, fit_canvas, parent=None):
        super().__init__(parent)
        fit_canvas.setAccessibleName(tr("График совместной подгонки"))
        self.addTab(fit_canvas, icon("chart-no-axes-combined"), tr("График"))
        self.overview = FittedFigureCanvas(Figure(layout="constrained"))
        self.overview.setAccessibleName(tr("Обзор серий"))
        overview_page = QWidget()
        overview_layout = QVBoxLayout(overview_page)
        self.overview_warning = QLabel()
        self.overview_warning.setTextFormat(Qt.TextFormat.PlainText)
        self.overview_warning.setWordWrap(True)
        overview_layout.addWidget(self.overview_warning)
        overview_layout.addWidget(self.overview, 1)
        self.overview_tab = self.addTab(overview_page, icon("layers"), tr("Обзор серий"))
        self.residuals = FittedFigureCanvas(Figure(layout="constrained"))
        self.residuals.setAccessibleName(tr("График остатков по серии"))
        self.residuals.setAccessibleDescription(tr("Значения точек доступны в таблице остатков."))
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.residuals, 2)
        self.residual_table = QTableView()
        self.residual_table.setAccessibleName(tr("Остатки по серии"))
        self.residual_table.setAccessibleDescription(tr("Нажмите Enter, чтобы перейти к выбранной пробе в исходной таблице."))
        self.residual_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.residual_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.residual_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.residual_table.activated.connect(self.open_residual_row)
        layout.addWidget(self.residual_table, 1)
        self.open_residual_button = QPushButton(icon("table-2"), tr("Показать выбранную пробу"))
        self.open_residual_button.clicked.connect(lambda: self.open_residual_row(self.residual_table.currentIndex()))
        layout.addWidget(self.open_residual_button)
        self.residual_tab = self.addTab(page, icon("chart-column"), tr("Остатки по серии"))
        self.currentChanged.connect(self.refresh)
        self.residuals.mpl_connect("pick_event", self.pick_residual)
        self.clear()

    def clear(self):
        self.result = None
        self.series = None
        self.row_ids = None
        self.segments = None
        self.highlight = None
        self.residual_points = None
        self.residual_series = None
        self.residual_indices = None
        self.set_residual_model(None)
        self.overview.figure.clear()
        self.residuals.figure.clear()
        self.setCurrentIndex(0)

    def set_result(self, result, row_ids, axes):
        self.result, self.row_ids, self.axes = result, row_ids, axes

    def select_series(self, series):
        self.series = series
        self.refresh(self.currentIndex())

    def refresh(self, index):
        if self.result is None or self.series is None:
            return
        if index == self.overview_tab:
            self.draw_overview()
        elif index == self.residual_tab and self.residual_series != self.series:
            self.draw_residuals()

    def draw_overview(self):
        if self.segments is None:
            segments = []
            for series in range(len(self.result.labels)):
                x = self.result.x[self.result.indices(series)]
                grid = np.linspace(x.min(), x.max(), 100)
                try:
                    predicted = self.result.predict(series, grid)
                except ValueError:
                    segments.append(np.empty((0, 2)))
                else:
                    segments.append(np.column_stack((grid, predicted)))
            self.segments = segments
            axis = self.overview.figure.subplots()
            axis.add_collection(LineCollection(segments, colors="tab:gray", linewidths=1, alpha=.45))
            axis.autoscale_view()
            axis.set_xlabel(self.axes[0], parse_math=False)
            axis.set_ylabel(self.axes[1], parse_math=False)
            if missing := sum(len(segment) == 0 for segment in segments):
                axis.set_title(tr("Кривые вне области определения: {0}", missing))
            self.highlight, = axis.plot([], [], color="tab:blue", linewidth=2)
        self.highlight.set_data(*self.segments[self.series].T)
        warning = (tr("Кривая серии «{0}» недоступна на сетке графика.", self.result.labels[self.series])
                   if not len(self.segments[self.series]) else "")
        self.overview_warning.setText(warning)
        self.overview_warning.setVisible(bool(warning))
        self.overview.setAccessibleDescription(warning)
        self.overview.draw_idle()

    def draw_residuals(self):
        self.residual_series = self.series
        self.residual_indices = indices = self.result.indices(self.series)
        fitted = self.result.fitted[indices]
        self.residuals.figure.clear()
        axis = self.residuals.figure.subplots()
        axis.axhline(0, color="tab:gray")
        self.residual_points = axis.scatter(fitted, self.result.y[indices]-fitted, s=12,
                                            linewidths=0, picker=5, rasterized=True)
        axis.set(xlabel=tr("Прогноз Y"), ylabel=tr("Остаток Y"))
        rows = _ResidualRows(self.result, self.row_ids, indices)
        self.set_residual_model(ResultsModel(["UUID", "Прогноз Y", "Остаток Y"], rows, self.residual_table))
        self.residuals.draw_idle()

    def set_residual_model(self, model):
        old = self.residual_table.model()
        self.residual_table.setModel(model)
        self.open_residual_button.setEnabled(False)
        if old is not None:
            old.deleteLater()
        if model is not None:
            self.residual_table.selectionModel().currentChanged.connect(
                lambda current, _: self.open_residual_button.setEnabled(current.isValid()))

    def open_residual_row(self, index):
        if index.isValid() and self.residual_indices is not None:
            self.source_selected.emit(self.row_ids[self.residual_indices[index.row()]])

    def pick_residual(self, event):
        if self.result is not None and event.artist is self.residual_points and len(event.ind):
            self.source_selected.emit(self.row_ids[self.residual_indices[event.ind[0]]])
