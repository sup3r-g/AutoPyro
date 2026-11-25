"""Well profile and interval drill-down for the current interpretation report."""

from collections import Counter, defaultdict
from html import escape
from textwrap import fill

from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QLabel,
    QLineEdit, QMessageBox, QSplitter, QTableView, QVBoxLayout,
)
from PySide6.QtCore import QLocale, QSignalBlocker, Qt, Signal

from autopyro.data_io.file_output import write_excel_csv
from autopyro.geology.stratigraphy import LEVELS, STATUSES as STRATIGRAPHY_STATUSES, UNITS, validate_binding
from autopyro.schemes.interpretation import STATUSES
from autopyro.schemes.interpretation_summary import group_intervals, prepare_summary
from autopyro.schemes.interpretation_thresholds import threshold_labels
from ..i18n import message, tr
from .figure_canvas import FittedFigureCanvas
from .results import ResultsModel
from .stratigraphy_labels import geological_label
from ..theme.controls import add_command_button


class SummaryBindingDialog(QDialog):
    def __init__(self, parent, project, scheme, initial):
        super().__init__(parent)
        self.setWindowTitle(tr('Привязка проб для сводки'))
        self.project, self.scheme = project, scheme
        self.binding = None
        form = QFormLayout(self)
        self.fields = {}
        for key, title in (('well_id', 'Ствол'), ('depth_id', 'Глубина / кровля пробы'),
                           ('bottom_id', 'Подошва пробы')):
            combo = QComboBox()
            combo.addItem(tr('Выберите колонку'), None)
            for column in project.columns:
                if not column.metadata.get('stratigraphy') and not column.metadata.get('method'):
                    label = f'{column.name} [{column.unit}]' if column.unit else column.name
                    combo.addItem(label, column.id)
            combo.setCurrentIndex(max(0, combo.findData(initial.get(key))))
            self.fields[key] = combo
            form.addRow(tr(title), combo)
        self.mode = QComboBox()
        self.mode.addItem(tr('Точечные пробы'), 'point')
        self.mode.addItem(tr('Интервальные пробы'), 'interval')
        self.mode.setCurrentIndex(max(0, self.mode.findData(initial.get('sample_mode', 'point'))))
        self.mode.currentIndexChanged.connect(self.update_mode)
        form.insertRow(0, tr('Тип проб'), self.mode)
        self.unit = QComboBox()
        self.unit.addItems(list(UNITS))
        self.unit.setCurrentText(initial.get('depth_unit', scheme['depth_unit']))
        form.addRow(tr('Единица глубины проб'), self.unit)
        self.reference = QLineEdit(initial.get('depth_reference', ''))
        self.reference.setPlaceholderText(scheme['depth_reference'])
        form.addRow(tr('Тип и начало отсчёта глубины проб'), self.reference)
        reference_note = QLabel(tr('Тип и начало отсчёта в схеме: {0}', scheme['depth_reference']))
        reference_note.setTextFormat(Qt.TextFormat.PlainText)
        reference_note.setWordWrap(True)
        form.addRow(reference_note)
        self.text_numbers = QCheckBox(tr('Читать числа в текстовых колонках'))
        self.text_numbers.setChecked(initial.get('text_numbers', False))
        form.addRow(self.text_numbers)
        self.error = QLabel()
        self.error.setTextFormat(Qt.TextFormat.PlainText)
        self.error.setWordWrap(True)
        form.addRow(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply_binding)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.update_mode()

    def update_mode(self):
        self.fields['bottom_id'].setEnabled(self.mode.currentData() == 'interval')

    def apply_binding(self):
        binding = dict(well_id=self.fields['well_id'].currentData(),
                       depth_id=self.fields['depth_id'].currentData(), depth_unit=self.unit.currentText(),
                       depth_reference=self.reference.text().strip(), text_numbers=self.text_numbers.isChecked())
        if self.mode.currentData() == 'interval':
            binding.update(sample_mode='interval', bottom_id=self.fields['bottom_id'].currentData())
        try:
            validate_binding(self.project, self.scheme, binding)
        except ValueError as exc:
            self.error.setText(message(exc))
            return
        self.binding = binding
        self.accept()


def edit_summary_binding(parent, project, scheme, initial):
    dialog = SummaryBindingDialog(parent, project, scheme, initial)
    accepted = dialog.exec()
    binding = dialog.binding if accepted else None
    dialog.deleteLater()
    return binding


class InterpretationSummaryDialog(QDialog):
    profile_selected = Signal(str)

    def __init__(self, parent, project, report, scheme, binding):
        super().__init__(parent)
        self.project, self.report, self.binding = project, report, binding
        self.wells, self.positions = prepare_summary(project, report, scheme, binding)
        self.scheme = scheme
        self.interpretation = report['scheme']
        self.intervals = {item['id']: item for item in scheme['intervals']}
        self.labels = {item['id']: item['label']
                       for item in self.interpretation.get('regions', []) + self.interpretation.get('lines', [])}
        self.statuses = {code: tr(label) for code, label in STATUSES.items()}
        self.statuses['excluded'] = tr('Исключено по входам')
        self.quality_by_id = {item['row_id']: item.get('quality', ()) for item in report['inputs']}
        table = getattr(parent, 'table', None)
        self.sample_names = getattr(table.model(), 'labels', {}) if table is not None else {}
        self.profile_well = object()
        self.profile_row = None
        self.selected_ids = ()
        self.setWindowTitle(tr('Интерпретация по стволам и интервалам'))
        self.resize(1020, 760)
        layout = QVBoxLayout(self)
        note = QLabel(tr('Основная схема: {0}. Стратиграфия: {1}, редакция {2}. '
                         'Доли рассчитаны от всех проб в строке. Пересечения, перекрытия '
                         'и неполное покрытие показаны отдельно.',
                         self.interpretation['name'], scheme['name'], scheme['revision']))
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.well = QComboBox()
        self.comparison_well = QComboBox()
        self.fill_wells()
        self.level = QComboBox()
        for level in LEVELS:
            self.level.addItem(geological_label(level), level)
        self.level.setCurrentIndex(self.level.findData('FORMATION'))
        form.addRow(tr('Ствол'), self.well)
        form.addRow(tr('Второй ствол'), self.comparison_well)
        form.addRow(tr('Уровень'), self.level)
        layout.addLayout(form)

        split = QSplitter(Qt.Orientation.Vertical)
        self.figure = Figure(layout='constrained')
        self.canvas = FittedFigureCanvas(self.figure)
        self.canvas.setMinimumHeight(220)
        split.addWidget(self.canvas)
        self.table = QTableView()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.model = ResultsModel(
            [tr('Интервал / привязка'), tr('Проб'), tr('Распределение оценок'),
             tr('Неоднозначные'), tr('Исключённые по входам')], [], self.table)
        self.summary_headers = tuple(self.model.headers)
        self.table.setModel(self.model)
        split.addWidget(self.table)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        self.profile_label = QLabel()
        self.profile_label.setTextFormat(Qt.TextFormat.PlainText)
        form.addRow(self.profile_label)
        self.profile_note = QLabel()
        self.profile_note.setTextFormat(Qt.TextFormat.PlainText)
        self.profile_note.setWordWrap(True)
        layout.addWidget(self.profile_note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        binding_button = buttons.addButton(tr('Настроить привязку…'), QDialogButtonBox.ButtonRole.ActionRole)
        binding_button.clicked.connect(self.change_binding)
        self.export_button = buttons.addButton(tr('Экспорт CSV…'), QDialogButtonBox.ButtonRole.ActionRole)
        self.export_button.setToolTip(tr('Выгрузить таблицу выбранных стволов и стратиграфического уровня.'))
        self.export_button.clicked.connect(self.export_csv)
        self.show_samples = add_command_button(buttons, 'table-2', tr('Показать пробы'), QDialogButtonBox.ButtonRole.ActionRole)
        self.show_samples.clicked.connect(lambda: self.open_samples(self.table.currentIndex()))
        self.show_profile_sample = buttons.addButton(tr('Перейти к пробе'), QDialogButtonBox.ButtonRole.ActionRole)
        self.show_profile_sample.setEnabled(False)
        self.show_profile_sample.clicked.connect(self.open_profile_sample)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.table.doubleClicked.connect(self.open_samples)
        self.table.selectionModel().currentRowChanged.connect(self.show_selected_well_profile)
        self.well.currentIndexChanged.connect(self.primary_well_changed)
        self.comparison_well.currentIndexChanged.connect(self.refresh)
        self.level.currentIndexChanged.connect(self.refresh)
        self.canvas.mpl_connect('pick_event', self.pick_profile)
        self.refresh()

    def change_binding(self):
        binding = edit_summary_binding(self, self.project, self.scheme, self.binding)
        if binding is None or binding == self.binding:
            return
        try:
            wells, positions = prepare_summary(self.project, self.report, self.scheme, binding)
        except (KeyError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, tr('Привязка проб для сводки'), message(exc))
            return
        self.binding, self.wells, self.positions = binding, wells, positions
        self.fill_wells()
        self.profile_well = object()
        self.refresh()

    def fill_wells(self):
        selected = self.well.currentData()
        has_selection = self.well.currentIndex() >= 0
        with QSignalBlocker(self.well):
            self.well.clear()
            for value in sorted(self.wells, key=lambda value: (value is None, (value or '').casefold())):
                self.well.addItem(value if value is not None else tr('Ствол не указан'), value)
            if has_selection:
                self.well.setCurrentIndex(max(0, self.well.findData(selected)))
        self.fill_comparison_wells()

    def primary_well_changed(self):
        self.fill_comparison_wells()
        self.refresh()

    def fill_comparison_wells(self):
        primary = self.well.currentData()
        selected = self.comparison_well.currentData()
        with QSignalBlocker(self.comparison_well):
            self.comparison_well.clear()
            self.comparison_well.addItem(tr('Без сравнения'), None)
            if primary is not None:
                for well in sorted((name for name in self.wells if name is not None and name != primary),
                                   key=str.casefold):
                    self.comparison_well.addItem(well, well)
            self.comparison_well.setCurrentIndex(max(0, self.comparison_well.findData(selected)))
        self.comparison_well.setEnabled(self.comparison_well.count() > 1)

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, tr('Сводка по интервалам'), 'interpretation-intervals.csv', 'CSV (*.csv)')
        if not path:
            return
        prefix = (self.level.currentData(), self.interpretation['name'], self.interpretation['revision'],
                  self.scheme['name'], self.scheme['revision'], self.scheme['depth_reference'], self.scheme['depth_unit'])
        headers = [
            'wellbore', 'level', 'interpretation_scheme', 'interpretation_revision', 'stratigraphy_scheme',
            'stratigraphy_revision', 'depth_reference', 'depth_unit', 'interval_or_status', 'sample_count',
            'outcomes', 'ambiguous_count', 'ambiguous_percent', 'excluded_count', 'excluded_percent',
        ]
        try:
            rows = ((well, *prefix, *row) for well, *row in self.export_rows)
            write_excel_csv(path, headers, rows, QLocale.system().decimalPoint())
        except OSError as exc:
            QMessageBox.warning(self, tr('Экспорт'), message(exc))

    def open_samples(self, index):
        if index.isValid():
            self.selected_ids = self.group_rows[index.row()]
            self.accept()

    def refresh(self):
        primary = self.well.currentData()
        comparison = self.comparison_well.currentData()
        wells = (primary, comparison) if comparison is not None else (primary,)

        def group_order(key):
            status, interval_id = key
            return (interval_id is None, self.intervals[interval_id]['top'] if interval_id else 0, status)

        rows, self.group_rows, self.export_rows = [], [], []
        for well in wells:
            samples = self.wells.get(well, [])
            if not samples:
                continue
            groups = group_intervals(samples, self.level.currentData())
            rows.append(self.summary_row(well, tr('Итого по стволу'), samples))
            for status, interval_id in sorted(groups, key=group_order):
                if interval_id is None:
                    label = tr(STRATIGRAPHY_STATUSES[status])
                else:
                    item = self.intervals[interval_id]
                    label = f"{item['name']} [{item['top']:g}, {item['bottom']:g}) {self.scheme['depth_unit']}"
                rows.append(self.summary_row(well, label, groups[status, interval_id]))
        headers = list(self.summary_headers)
        if comparison is not None:
            headers.insert(0, tr('Ствол'))
            rows = [(row[0], *cells) for row, cells in zip(self.export_rows, rows, strict=True)]
        self.model.set_rows(rows, headers=headers)
        self.table.horizontalHeader().setResizeContentsPrecision(100)
        self.table.resizeColumnsToContents()
        distribution_column = 2 if comparison is None else 3
        self.table.setColumnWidth(distribution_column, min(530, self.table.columnWidth(distribution_column)))
        self.show_samples.setEnabled(bool(rows))
        self.export_button.setEnabled(bool(rows))
        if rows:
            self.table.selectRow(0)
        self.show_well_profile(primary)

    def show_selected_well_profile(self, index, _previous):
        if index.isValid() and index.row() < len(self.export_rows):
            self.show_well_profile(self.export_rows[index.row()][0])

    def show_well_profile(self, well):
        if well != self.profile_well:
            self.profile_well = well
            self.profile_label.setText(tr('Ствол профиля: {0}', well or tr('Ствол не указан')))
            self.draw_profile(self.wells.get(well, []), self.positions.get(well, ()))

    def outcome_labels(self, item):
        names = item['regions'] + item['boundary_regions'] + (item.get('possible_types') or item['line_hits'])
        labels = [self.labels.get(key, key) for key in names if key not in item.get('thresholds', {})]
        labels.extend(threshold_labels(self.interpretation, item.get('thresholds', {})))
        return list(dict.fromkeys(labels))

    def summary_row(self, well, label, samples):
        counts = Counter()
        ambiguous = excluded = 0
        ids = []
        for _, item, blocked in samples:
            ids.append(item['row_id'])
            status = item['status']
            ambiguous += status == 'ambiguous'
            excluded += blocked
            labels = self.outcome_labels(item)
            outcome = self.statuses['excluded' if blocked else status]
            if labels:
                outcome += ' · ' + ' / '.join(dict.fromkeys(labels))
            counts[outcome] += 1
        self.group_rows.append(tuple(ids))
        total = len(samples)
        distribution = ' | '.join(f'{outcome} ({n})' for outcome, n in counts.most_common())
        self.export_rows.append((well, label, total, distribution, ambiguous, ambiguous / total * 100,
                                 excluded, excluded / total * 100))
        return label, total, distribution, f'{ambiguous} ({ambiguous / total:.1%})', f'{excluded} ({excluded / total:.1%})'

    def draw_profile(self, samples, positions):
        self.figure.clear()
        self.profile_artists = {}
        self.profile_points = defaultdict(dict)
        self.profile_outcomes = {}
        self.highlights = {}
        if not positions:
            axis = self.figure.add_subplot()
            axis.set_axis_off()
            axis.text(.5, .5, tr('Нет проб с корректной глубиной'), ha='center', va='center',
                      transform=axis.transAxes)
            self.select_profile(None)
            return

        outcomes = {item['row_id']: (item, blocked) for _, item, blocked in samples}
        statuses, types, quality, hio = {}, {}, {}, {}
        for sample in positions:
            row_id = sample.row_id
            item, blocked = outcomes[row_id]
            statuses[row_id] = 'excluded' if blocked else item['status']
            types[row_id] = ' / '.join(self.outcome_labels(item)) or tr('Нет оценки')
            quality[row_id] = len(self.quality_by_id[row_id])
            bounds = item.get('value_bounds')
            if bounds is None and item.get('value') is not None:
                bounds = (item['value'], item['value'])
            if bounds is not None:
                hio[row_id] = bounds
            self.profile_outcomes[row_id] = (sample, statuses[row_id], types[row_id], bounds, quality[row_id])

        has_hio = bool(hio) or self.interpretation.get('strategy') == 'initial_hi'
        axes = self.figure.subplots(1, 4 if has_hio else 3, sharey=True, squeeze=False)[0]
        present_statuses = set(statuses.values())
        status_order = [status for status in self.statuses if status in present_statuses]
        self.draw_categories(axes[0], positions, statuses, status_order, self.statuses, tr('Статус'))
        type_order = sorted(set(types.values()), key=str.casefold)
        self.draw_categories(axes[1], positions, types, type_order, {}, tr('Тип / область'))
        self.draw_depth_track(axes[2], positions, quality, '#a65300')
        axes[2].set_title(tr('Контроль качества'))
        axes[2].set_xlabel(tr('Число замечаний'))
        axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[2].set_xlim(left=-.5, right=max(quality.values()) + .5)
        if has_hio:
            self.draw_hio(axes[3], positions, hio)

        axes[0].invert_yaxis()
        axes[0].set_ylabel(tr('Глубина {0} [{1}]', self.scheme['depth_reference'], self.scheme['depth_unit']))
        for axis in axes:
            axis.grid(axis='y', alpha=.25)
            self.highlights[axis] = axis.scatter([], [], s=75, facecolors='none',
                                                 edgecolors='#cc3311', linewidths=1.5, zorder=5)
        if len(positions) != len(samples):
            self.figure.suptitle(tr('Проб без корректной глубины: {0}', len(samples) - len(positions)))
        self.select_profile(self.profile_row)

    def draw_categories(self, axis, positions, values, order, labels, title):
        grouped = defaultdict(list)
        for sample in positions:
            grouped[values[sample.row_id]].append(sample)
        for lane, key in enumerate(order):
            samples = grouped[key]
            self.draw_depth_track(axis, samples, {sample.row_id: lane for sample in samples}, f'C{lane % 10}')
        axis.set_xlim(-.5, len(order) - .5)
        step = max(1, (len(order) + 5) // 6)
        ticks = range(0, len(order), step)
        axis.set_xticks(list(ticks), [fill(labels.get(order[i], order[i]), 14) for i in ticks],
                        rotation=20 if len(order) > 2 else 0, fontsize=8)
        axis.set_title(title)

    def draw_depth_track(self, axis, samples, values, color):
        points, point_ids, segments, segment_ids = [], [], [], []
        for sample in samples:
            row_id = sample.row_id
            x = values[row_id]
            if sample.bottom is None:
                points.append((x, sample.top))
                point_ids.append(row_id)
            else:
                segments.append(((x, sample.top), (x, sample.bottom)))
                segment_ids.append(row_id)
            depth = sample.top if sample.bottom is None else (sample.top + sample.bottom) / 2
            self.profile_points[row_id][axis] = (x, depth)
        self.draw_profile_artists(axis, points, point_ids, segments, segment_ids, color)

    def draw_hio(self, axis, positions, bounds):
        points, point_ids, segments, segment_ids = [], [], [], []
        for sample in positions:
            row_id = sample.row_id
            if row_id not in bounds:
                continue
            low, high = bounds[row_id]
            depth = sample.top if sample.bottom is None else (sample.top + sample.bottom) / 2
            if low == high:
                points.append((low, depth))
                point_ids.append(row_id)
            else:
                segments.append(((low, depth), (high, depth)))
                segment_ids.append(row_id)
            self.profile_points[row_id][axis] = ((low + high) / 2, depth)
        self.draw_profile_artists(axis, points, point_ids, segments, segment_ids, '#0072b2')
        axis.set_title(tr('Диапазон HIo'))
        axis.set_xlabel(self.interpretation['y']['unit'])
        if not bounds:
            axis.text(.5, .5, tr('Нет оценок HIo'), ha='center', va='center', transform=axis.transAxes)

    def draw_profile_artists(self, axis, points, point_ids, segments, segment_ids, color):
        if points:
            x, depth = zip(*points)
            artist = axis.scatter(x, depth, color=color, s=24, picker=True)
            self.profile_artists[artist] = point_ids
        if segments:
            artist = LineCollection(segments, colors=color, linewidths=2, picker=True, pickradius=5)
            axis.add_collection(artist)
            axis.autoscale_view()
            self.profile_artists[artist] = segment_ids

    def pick_profile(self, event):
        ids = self.profile_artists.get(event.artist)
        if ids and len(event.ind):
            self.select_profile(ids[int(event.ind[0])])
            self.profile_selected.emit(self.profile_row)

    def select_profile(self, row_id):
        self.profile_row = row_id if row_id in self.profile_outcomes else None
        for axis, artist in self.highlights.items():
            point = self.profile_points.get(self.profile_row, {}).get(axis)
            artist.set_visible(point is not None)
            if point is not None:
                artist.set_offsets([point])
        self.show_profile_sample.setEnabled(self.profile_row is not None)
        self.profile_note.setToolTip('')
        if self.profile_row is None:
            self.profile_note.setText(tr('Выберите пробу на профиле.'))
        else:
            sample, status, outcome, bounds, quality = self.profile_outcomes[self.profile_row]
            depth = f'{sample.top:g}' if sample.bottom is None else f'{sample.top:g}–{sample.bottom:g}'
            text = tr('Проба {0} · {1} {2} · {3} · {4} · замечаний QC: {5}',
                      self.sample_names.get(self.profile_row, self.profile_row), depth,
                      self.scheme['depth_unit'], self.statuses[status], outcome, quality)
            if bounds is not None:
                text += tr(' · HIo {0}–{1} {2}', f'{bounds[0]:g}', f'{bounds[1]:g}', self.interpretation['y']['unit'])
            self.profile_note.setText(text)
            issues = self.quality_by_id[self.profile_row]
            if issues:
                self.profile_note.setToolTip('<qt>' + '<br>'.join(escape(issue['message']) for issue in issues) + '</qt>')
        self.canvas.draw_idle()

    def show_profile_row(self, row_id):
        for well, positions in self.positions.items():
            if any(sample.row_id == row_id for sample in positions):
                if well != self.well.currentData() and (well is None or well != self.comparison_well.currentData()):
                    self.well.setCurrentIndex(self.well.findData(well))
                self.show_well_profile(well)
                self.select_profile(row_id)
                return

    def open_profile_sample(self):
        if self.profile_row is not None:
            self.selected_ids = (self.profile_row,)
            self.accept()
