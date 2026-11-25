"""Modal preview of sample-level changes before committing a scheme revision."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QAbstractItemView, QDialog, QDialogButtonBox, QLabel, QSplitter, QTableView, QVBoxLayout
from PySide6.QtCore import Qt

from autopyro.analysis.statistics import classify
from autopyro.schemes.interpretation import STATUSES
from autopyro.schemes.revision_comparison import changed_samples
from autopyro.visualization.plotting import PlotSpec, draw_interpretation_layer
from ..i18n import tr
from ..widgets.results import SampleResultsModel, sample_labels


class SchemeRevisionDialog(QDialog):
    def __init__(self, before, after, prepared, parent):
        super().__init__(parent)
        self.setWindowTitle(tr('Сравнение редакций схемы'))
        self.resize(1050, 750)
        layout = QVBoxLayout(self)
        schemes = [before['scheme'], after['scheme']]
        rows, before_coordinates, after_coordinates = prepared
        layout.addWidget(QLabel(tr('Редакции {0} → {1}. Изменения у {2} из {3} проб текущей области данных.',
                                  schemes[0]['revision'], schemes[1]['revision'], len(rows), len(after['results']))))
        case_count = len(schemes[1].get('reference_cases', []))
        if case_count:
            layout.addWidget(QLabel(tr('Эталонные пробы прошли проверку ({0}).', case_count)))
        columns = {column.id: column.name for column in parent.owner.model.project.columns}
        titles = tr('До'), tr('После')
        for title, report in zip(titles, (before, after), strict=True):
            axes = [tr('{0}: {1} [{2}]', report['scheme'][key]['property'],
                       columns.get(report['binding'][f'{key}_id'], '—'), report['binding'][f'{key}_unit'])
                    for key in ('x', 'y')]
            label = QLabel(f"{title} · {' / '.join(axes)}")
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            layout.addWidget(label)
        splitter = QSplitter(Qt.Orientation.Vertical)
        figure = Figure(layout='tight')
        self.canvas = FigureCanvasQTAgg(figure)
        axes = figure.subplots(1, 2)
        self.table = QTableView()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        labels = sample_labels(parent.owner.model.project, (row[0] for row in rows)) if rows else {}
        self.table.setModel(SampleResultsModel([
            tr('Проба'), tr('Тип / область до'), tr('Тип / область после'),
            tr('Статус до'), tr('Статус после'), tr('Диапазон HIo до'), tr('Диапазон HIo после'),
            tr('Пороги до'), tr('Пороги после'),
        ], rows, labels, self.table))
        self.table.horizontalHeader().setResizeContentsPrecision(100)
        self.table.resizeColumnsToContents()
        point_rows, highlights = {}, []
        for ax, title, report, coordinates in zip(axes, titles, (before, after),
                                                  (before_coordinates, after_coordinates), strict=True):
            scheme = report['scheme']
            missing = len(rows) - len(coordinates)
            ax.set_title(tr('{0} · без координат: {1}', title, missing) if missing else title)
            x, y = zip(*coordinates.values()) if coordinates else ([], [])
            point_rows[ax.scatter(x, y, picker=5)] = list(coordinates)
            highlights.append((ax.scatter([], [], s=90, facecolors='none', edgecolors='#d55e00', linewidths=2),
                               coordinates))
            for key, set_label in (('x', ax.set_xlabel), ('y', ax.set_ylabel)):
                axis = scheme[key]
                set_label(f"{axis['property']} [{axis['unit']}]")
                if axis.get('transform') == 'log10':
                    getattr(ax, f'set_{key}scale')('log')
            binding = report['binding']
            spec = PlotSpec(x_id=binding['x_id'], y_id=binding['y_id'],
                            interpretation=dict(binding, scheme=scheme), interpretation_labels=False)
            draw_interpretation_layer(ax, parent.owner.model.project, spec)
            ax.grid(alpha=.25)

        def select_point(event):
            if event.artist in point_rows and len(event.ind) and not toolbar.mode:
                position = point_rows[event.artist][int(event.ind[0])]
                self.table.selectRow(position)
                self.table.scrollTo(self.table.model().index(position, 0))

        def highlight_row(*_):
            selection = self.table.selectionModel().selectedRows()
            for highlight, coordinates in highlights:
                point = coordinates.get(selection[0].row()) if selection else None
                highlight.set_visible(point is not None)
                if point is not None:
                    highlight.set_offsets([point])
            self.canvas.draw_idle()

        toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=False)
        layout.addWidget(toolbar)
        self.canvas.mpl_connect('pick_event', select_point)
        self.table.selectionModel().selectionChanged.connect(highlight_row)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.setSizes([420, 250])
        layout.addWidget(splitter, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('Применить редакцию'))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @classmethod
    def prepare(cls, before, after, checkpoint):
        rows, positions = [], {}
        for row_id, old, new in changed_samples(before, after, checkpoint=checkpoint):
            row = [row_id]
            for left, right in zip(cls.describe(old), cls.describe(new), strict=True):
                row.extend((left, right))
            positions[row_id] = len(rows)
            rows.append(row)
        return (rows, cls.coordinates(before, positions, checkpoint), cls.coordinates(after, positions, checkpoint))

    @staticmethod
    def coordinates(report, positions, checkpoint):
        coordinates = {}
        if positions:
            logarithmic = tuple(report['scheme'][key].get('transform') == 'log10' for key in ('x', 'y'))
            for index, sample in enumerate(report['inputs']):
                if index % 256 == 0:
                    checkpoint()
                position = positions.get(sample['row_id'])
                if position is not None and sample['statuses'] == ['numeric', 'numeric']:
                    x, y = (classify(sample[key], numeric=True)[1] for key in ('x', 'y'))
                    if all(value is not None and (not log or value > 0)
                           for value, log in zip((x, y), logarithmic, strict=True)):
                        coordinates[position] = (x, y)
        return coordinates

    @staticmethod
    def describe(signature):
        types, interval_regions, status, bounds, (_, thresholds), value = signature
        names = ', '.join(name for _, name in sorted(types)) or '—'
        if interval_regions and interval_regions != types:
            interval = ', '.join(name for _, name in sorted(interval_regions))
            names += ' ' + tr('(интервал: {0})', interval)
        estimate = ' – '.join(format(bound, '.12g') for bound in bounds)
        if value is not None:
            estimate = f'{value:.12g} [{estimate}]' if estimate else f'{value:.12g}'
        return names, tr(STATUSES[status]), estimate or '—', '; '.join(thresholds) or '—'
