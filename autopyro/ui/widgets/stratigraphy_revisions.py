"""Compare two snapshots and open their columns without editing the draft."""

from PySide6.QtCore import QLocale, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFormLayout, QHeaderView, QHBoxLayout,
    QFileDialog, QLabel, QMessageBox, QPushButton, QTableView, QTabWidget, QVBoxLayout,
)

from autopyro.data_io.file_output import write_excel_csv
from autopyro.model.core import parse_number
from autopyro.geology.stratigraphy import STATUSES, UNITS
from autopyro.geology.stratigraphy_revisions import interval_changes, sample_changes
from ..i18n import message, tr
from ..tasks import TaskCancelled
from ..theme import icon
from .results import ResultsModel
from .stratigraphy_labels import geological_label


def interval_label(item, scheme):
    if item is None:
        return None
    return (f"{item['wellbore']} · {geological_label(item['level'])} · {item['name']} · "
            f"{item['top']:.10g}–{item['bottom']:.10g} {scheme['depth_unit']} · {scheme['depth_reference']}")


def membership_label(result, unit):
    names = result['names']
    overlaps = result.get('overlaps', ())
    if overlaps:
        names = [f"{name} ({part['length']:.6g} {unit})"
                 for name, part in zip(names, overlaps, strict=True)]
    status = tr(STATUSES[result['status']])
    if result.get('coverage_fraction') is not None:
        status += tr(' · покрытие {0:.1%}', result['coverage_fraction'])
        if result.get('ambiguous_length'):
            status += tr(' · перекрытие {0:.6g} {1}', result['ambiguous_length'], unit)
    return ' · '.join([', '.join(names), status]) if names else status


def interval_target(item, fields):
    if item is None:
        return None
    top, bottom = item['top'], item['bottom']
    if 'top' in fields and 'bottom' not in fields:
        bottom = top
    elif 'bottom' in fields and 'top' not in fields:
        top = bottom
    return item['wellbore'], item['level'], top, bottom


class RevisionComparisonDialog(QDialog):
    def __init__(self, owner, revisions, draft, binding):
        super().__init__(owner)
        self.owner, self.binding = owner, binding
        self.schemes = [scheme for _, scheme in revisions]
        self.setWindowTitle(tr('Сравнение редакций стратиграфии'))
        self.resize(1050, 640)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.before, self.after = QComboBox(), QComboBox()
        for combo in (self.before, self.after):
            for index, (revision, scheme) in enumerate(revisions):
                combo.addItem(tr('{0} · редакция {1}', scheme['name'], revision), index)
                combo.setItemData(index, scheme['id'], Qt.ItemDataRole.ToolTipRole)
        self.before.setCurrentIndex(len(revisions) - (1 if draft else 2))
        if draft:
            self.after.addItem(tr('Текущий черновик'), len(self.schemes))
            self.schemes.append(draft)
        self.after.setCurrentIndex(self.after.count() - 1)
        form.addRow(tr('До'), self.before)
        form.addRow(tr('После'), self.after)
        layout.addLayout(form)
        self.samples = QCheckBox(tr('Сравнить принадлежность проб по текущим параметрам привязки и области проб'))
        self.samples.setChecked(bool(binding.get('well_id') and binding.get('depth_id')))
        layout.addWidget(self.samples)
        compare = QPushButton(icon('files'), tr('Сравнить редакции'))
        compare.clicked.connect(self.compare)
        layout.addWidget(compare)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.tabs = QTabWidget()
        self.tables = []
        for title, headers in (
            (tr('Изменения интервалов'), [tr('Изменение'), tr('Сопоставление'), tr('До'), tr('После')]),
            (tr('Изменения привязки проб'), [tr('UUID пробы'), tr('Ствол'), tr('Глубина'),
                                            tr('Уровень'), tr('До'), tr('После')]),
        ):
            table = QTableView()
            table.setModel(ResultsModel(headers, [], table))
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            for column in (len(headers) - 2, len(headers) - 1):
                table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
            table.selectionModel().selectionChanged.connect(self.update_navigation)
            table.doubleClicked.connect(lambda index: self.open_column(1 if self.target(1) else 0))
            self.tables.append(table)
            self.tabs.addTab(table, icon('layers' if len(self.tables) == 1 else 'table-2'), title)
        layout.addWidget(self.tabs, 1)
        self.tables[0].setColumnWidth(1, 185)
        navigation = QHBoxLayout()
        self.navigation_buttons = []
        for side, text in enumerate((tr('Колонка до изменения'), tr('Колонка после изменения'))):
            button = QPushButton(icon('zoom-in'), text)
            button.clicked.connect(lambda checked=False, side=side: self.open_column(side))
            navigation.addWidget(button)
            self.navigation_buttons.append(button)
        layout.addLayout(navigation)
        self.export_button = QPushButton(icon('log-out'), tr('Экспорт CSV…'))
        self.export_button.setToolTip(tr('Сохранить все строки текущей вкладки сравнения.'))
        self.export_button.clicked.connect(self.export_csv)
        navigation.addWidget(self.export_button)
        self.targets = [[], []]
        self.csv_results = [None, None]
        self.tabs.currentChanged.connect(self.update_navigation)
        for combo in (self.before, self.after):
            combo.currentIndexChanged.connect(self.clear_results)
        self.samples.toggled.connect(self.clear_results)
        self.update_navigation()

    def clear_results(self):
        self.csv_results = [None, None]
        self.targets = [[], []]
        for table in self.tables:
            table.model().set_rows([])
        self.summary.clear()
        self.update_navigation()

    def compare(self):
        self.clear_results()
        before, after = [self.schemes[combo.currentData()] for combo in (self.before, self.after)]
        self.compared = (before, after)
        labels = {'wellbore': tr('Ствол'), 'level': tr('Уровень'), 'name': tr('Подразделение'),
                  'top': tr('Кровля'), 'bottom': tr('Подошва'), 'lithology': tr('Литология'),
                  'added': tr('Добавление'), 'removed': tr('Удаление')}
        matching = {'id': tr('По идентификатору'), 'name_bounds': tr('По названию и границам'),
                    'bounds': tr('По границам'), 'name_overlap': tr('По названию и перекрытию'), None: None}
        changes, identities = interval_changes(before, after)
        self.csv_results[0] = [(', '.join(labels[key] for key in fields), matching[method], left, right)
                               for fields, left, right, method in changes]
        self.targets[0] = [
            (interval_target(left, fields), interval_target(right, fields))
            for fields, left, right, _ in changes]
        catalog = self.owner.lithology_catalog

        def describe(item, scheme, other):
            label = interval_label(item, scheme)
            if item is None:
                return label
            code = item.get('lithology', '').strip()
            if not code and not (other or {}).get('lithology', '').strip():
                return label
            rock = catalog.resolve(code)
            return f'{rock.label if rock else code} · {label}'

        self.tables[0].model().set_rows([
            (label, method, describe(left, before, right), describe(right, after, left))
            for label, method, left, right in self.csv_results[0]])
        summary = tr('Изменений интервалов: {0}.', len(changes))
        if self.samples.isChecked():
            try:
                wells = {item['wellbore'] for fields, left, right, _ in changes
                         if fields != ('lithology',) for item in (left, right) if item}
                self.compare_samples(before, after, identities, wells)
                summary += ' ' + tr('Изменений привязки по уровням: {0}.', len(self.targets[1]))
            except TaskCancelled:
                summary += ' ' + tr('Привязка проб: {0}', tr('Подготовка отменена.'))
            except ValueError as exc:
                summary += ' ' + tr('Привязка проб: {0}', message(exc))
        self.summary.setText(summary)
        for table in self.tables:
            if table.model().rowCount():
                table.selectRow(0)
        self.update_navigation()

    def compare_samples(self, before, after, identities, wells):
        scope = self.owner.owner.analysis_scope
        scope_revision = scope.revision
        row_ids = scope.rows()
        binding = self.binding
        format_rows = self.sample_comparison_rows

        def calculate(project, checkpoint):
            rows_by_id = project.row_map()
            well_column = binding.get('well_id')
            ids = []
            for index, key in enumerate(row_ids):
                if checkpoint and index % 256 == 0:
                    checkpoint()
                well = rows_by_id[key].values.get(well_column)
                if isinstance(well, str) and well in wells:
                    ids.append(key)
            changes = sample_changes(project, ids, before, after, binding, identities, checkpoint=checkpoint)
            rows, targets = format_rows(changes, before, after, binding['depth_unit'], checkpoint)
            return changes, rows, targets

        changes, rows, targets = self.owner.calculate_binding_report(
            row_ids, (binding,), tr('Изменения привязки проб'), calculate, parent=self)
        if scope.revision != scope_revision:
            raise TaskCancelled()
        self.targets[1] = targets
        self.tables[1].model().set_rows(rows)
        self.csv_results[1] = changes

    @staticmethod
    def sample_comparison_rows(changes, before, after, unit, checkpoint):
        tracks = [{(item['wellbore'], item['level']) for item in scheme['intervals']}
                  for scheme in (before, after)]
        rows, targets = [], []
        for index, (item, level, left, right) in enumerate(changes):
            if checkpoint and index % 256 == 0:
                checkpoint()
            top = parse_number(item['depth'])
            bottom = parse_number(item['bottom']) if item['bottom'] is not None else top
            depth = item['depth'] if item['bottom'] is None else f"[{item['depth']}, {item['bottom']})"
            rows.append((item['row_id'], item['wellbore'], depth, geological_label(level),
                         membership_label(left, unit), membership_label(right, unit)))
            pair = []
            for scheme, available in zip((before, after), tracks, strict=True):
                well = item['wellbore']
                factor = UNITS[unit] / UNITS[scheme['depth_unit']]
                pair.append((well, level, top * factor, bottom * factor) if (well, level) in available else None)
            targets.append(tuple(pair))
        return rows, targets

    def csv_table(self, tab):
        before, after = self.compared
        revisions = (before['revision'], after['revision'])
        revision_headers = [tr('Редакция до'), tr('Редакция после')]
        if tab == 0:
            fields = ('id', 'wellbore', 'level', 'name', 'top', 'bottom')
            columns = [tr('Идентификатор интервала'), tr('Ствол'), tr('Уровень'), tr('Подразделение'),
                       tr('Кровля'), tr('Подошва'), tr('Литология'), tr('Единица глубины'), tr('Система глубин')]
            headers = [*revision_headers, tr('Изменение'), tr('Сопоставление'),
                       *(f'{column} · {side}' for side in (tr('До'), tr('После')) for column in columns)]

            def values(item, scheme):
                if item is None:
                    return (None,) * len(columns)
                return (*(item[key] for key in fields), item.get('lithology'),
                        scheme['depth_unit'], scheme['depth_reference'])

            return headers, ((*revisions, label, method, *values(left, before), *values(right, after))
                             for label, method, left, right in self.csv_results[0])

        headers = [*revision_headers, tr('UUID пробы'), tr('Ствол'), tr('Уровень'), tr('Верх отбора'),
                   tr('Низ отбора'), tr('Единица глубины'), tr('Система глубин'), tr('Сторона сравнения'),
                   tr('Тип строки'), tr('Статус'), tr('Подразделение'), tr('Идентификатор интервала'),
                   tr('Длина отбора в подразделении'), tr('Покрытие пробы, %')]

        def sample_rows():
            total, subdivision = tr('Итог по пробе'), tr('Подразделение')
            for item, level, left, right in self.csv_results[1]:
                sample = (*revisions, item['row_id'], item['wellbore'], level,
                          parse_number(item['depth']), parse_number(item['bottom']),
                          self.binding['depth_unit'], self.binding['depth_reference'])
                for side, result in ((tr('До'), left), (tr('После'), right)):
                    status = tr(STATUSES[result['status']])
                    coverage = result.get('coverage_fraction')
                    percent = None if coverage is None else coverage * 100
                    yield (*sample, side, total, status, None, None, None, percent)
                    lengths = {part['interval_id']: part['length'] for part in result.get('overlaps', ())}
                    for name, key in zip(result['names'], result['interval_ids'], strict=True):
                        yield (*sample, side, subdivision, None, name, key, lengths.get(key), None)

        return headers, sample_rows()

    def export_csv(self):
        tab = self.tabs.currentIndex()
        if self.csv_results[tab] is None:
            return
        filename = 'interval-changes.csv' if tab == 0 else 'sample-assignment-changes.csv'
        dialog = QFileDialog(self, tr('Экспорт CSV…'), filename, 'CSV (*.csv)')
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix('csv')
        path = dialog.selectedFiles()[0] if dialog.exec() == QDialog.DialogCode.Accepted else None
        dialog.deleteLater()
        if path:
            try:
                headers, rows = self.csv_table(tab)
                write_excel_csv(path, headers, rows, QLocale.system().decimalPoint())
            except OSError as exc:
                QMessageBox.warning(self, tr('Ошибка'), tr('Ошибка экспорта: {0}', str(exc)))

    def target(self, side):
        tab = self.tabs.currentIndex()
        rows = self.tables[tab].selectionModel().selectedRows()
        if rows and rows[0].row() < len(self.targets[tab]):
            return self.targets[tab][rows[0].row()][side]
        return None

    def update_navigation(self):
        self.export_button.setEnabled(self.csv_results[self.tabs.currentIndex()] is not None)
        for side, button in enumerate(self.navigation_buttons):
            button.setEnabled(self.target(side) is not None)

    def open_column(self, side):
        target = self.target(side)
        if target:
            self.owner.show_revision_column(self.compared[side], target, self)
