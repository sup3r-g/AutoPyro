"""Read-only thickness and lithology tables."""

from itertools import chain

from PySide6.QtCore import QLocale, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QLabel, QLineEdit, QMessageBox, QTableView, QTabWidget, QVBoxLayout,
)

from autopyro.data_io.file_output import write_excel_csv
from autopyro.geology.stratigraphy_summary import summarize_intervals
from ..i18n import tr
from ..theme import icon
from ..theme.controls import apply_control_icons
from .results import ResultsModel
from .stratigraphy_labels import geological_label
from ..theme.controls import add_search_icon


class StratigraphySummaryDialog(QDialog):
    def __init__(self, scheme, catalog, parent):
        super().__init__(parent)
        self.setWindowTitle(tr('Сводка мощностей и литологии'))
        self.resize(1050, 590)
        layout = QVBoxLayout(self)
        note = QLabel(tr('Мощности измеряются по оси глубины схемы. Доли рассчитаны от покрытой мощности. '
                         'Разрывы указаны между интервалами одного ствола и уровня. '
                         'Перекрытия показаны отдельно и не распределяются по породам.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        search = QLineEdit()
        search.setPlaceholderText(tr('Поиск…'))
        add_search_icon(search)
        layout.addWidget(search)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)
        missing = ('missing', '')
        labels = {missing: tr('Без литологии')}
        rock_keys = {}
        for raw in {item.get('lithology', '') for item in scheme['intervals']}:
            entry = catalog.resolve(raw)
            if not raw.strip() or (entry and entry.code in ('', 'no')):
                key = missing
            elif entry:
                base = catalog.by_code.get(entry.code.removesuffix('*'))
                if base and base.name == entry.name:
                    entry = base
                key = ('rock', entry.code)
                labels[key] = entry.label
            else:
                key = ('unknown', raw.strip())
                labels[key] = tr('Литология недоступна: {0}', raw.strip())
            rock_keys[raw] = key
        thickness, composition = [], []
        for (well, level, name), result in summarize_intervals(scheme['intervals'], rock_keys).items():
            group = (well, geological_label(level), name)
            unknown = sum(length for (kind, _), length in result.rocks.items() if kind == 'unknown')
            thickness.append((*group, result.covered, result.gap, result.overlap,
                              result.rocks.get(missing, 0), unknown))
            for rock, length in sorted(result.rocks.items(), key=lambda pair: labels[pair[0]].casefold()):
                composition.append((*group, labels[rock], length, length / result.covered))
            if result.overlap:
                composition.append((*group, tr('Перекрытие'), result.overlap,
                                    result.overlap / result.covered))
        group_headers = [tr('Ствол'), tr('Уровень'), tr('Подразделение')]
        unit = scheme['depth_unit']
        thickness_headers = [tr('Покрытая мощность [{0}]', unit), tr('Разрывы [{0}]', unit),
                             tr('Перекрытия [{0}]', unit), tr('Без литологии [{0}]', unit),
                             tr('Не распознано [{0}]', unit)]
        rock_headers = [tr('Литология'), tr('Мощность [{0}]', unit)]
        metrics = [*thickness_headers, rock_headers[1], tr('Доля, %')]
        self.csv_headers = [tr('Раздел'), tr('Тип строки'), *group_headers, rock_headers[0], *metrics,
                            *(f"{metric} · {tr('Подразделение')}" for metric in metrics)]
        self.thickness_rows = thickness
        self.composition_rows = composition

        def display_group(row):
            return (*row[:2], row[2] if row[2] is not None else tr('Итого по уровню'), *row[3:])

        tabs.addTab(self.make_table(
            [*group_headers, *thickness_headers],
            [display_group(row) for row in thickness], search), icon('table-2'), tr('Мощность'))
        tabs.addTab(self.make_table(
            [*group_headers, *rock_headers, tr('Доля')],
            [(*display_group(row[:-1]), f'{row[-1]:.2%}') for row in composition], search),
            icon('chart-column'), tr('Состав пород'))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        export = buttons.addButton(tr('Экспорт CSV…'), QDialogButtonBox.ButtonRole.ActionRole)
        export.setIcon(icon('log-out'))
        export.setToolTip(tr('Сохранить полную сводку, включая строки, скрытые поиском.'))
        export.clicked.connect(self.export_csv)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        apply_control_icons(self)

    def export_csv(self):
        dialog = QFileDialog(self, tr('Экспорт CSV…'), 'stratigraphy-summary.csv', 'CSV (*.csv)')
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix('csv')
        path = dialog.selectedFiles()[0] if dialog.exec() == QDialog.DialogCode.Accepted else None
        dialog.deleteLater()
        if not path:
            return
        try:
            self.write_csv(path, QLocale.system().decimalPoint())
        except OSError as exc:
            QMessageBox.warning(self, tr('Ошибка'), tr('Ошибка экспорта: {0}', str(exc)))

    def write_csv(self, path, decimal):
        level_total, subdivision = tr('Итого по уровню'), tr('Подразделение')
        thickness_section, composition_section = tr('Мощность'), tr('Состав пород')
        empty_thickness = (None,) * 5

        def csv_row(section, group, rock, metrics):
            empty = (None,) * len(metrics)
            if group[2] is None:
                return (section, level_total, *group, rock, *metrics, *empty)
            return (section, subdivision, *group, rock, *empty, *metrics)

        thickness = (csv_row(thickness_section, row[:3], None, (*row[3:], None, None))
                     for row in self.thickness_rows)
        composition = (
            csv_row(composition_section, row[:3], row[3], (*empty_thickness, row[4], row[5] * 100))
            for row in self.composition_rows
        )
        write_excel_csv(path, self.csv_headers, chain(thickness, composition), decimal)

    @staticmethod
    def make_table(headers, rows, search):
        table = QTableView()
        model = ResultsModel(headers, rows, table)
        proxy = QSortFilterProxyModel(table)
        proxy.setSourceModel(model)
        proxy.setFilterKeyColumn(-1)
        proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        search.textChanged.connect(proxy.setFilterFixedString)
        table.setModel(proxy)
        for column, title in enumerate(headers):
            table.setColumnWidth(column, max(105, table.fontMetrics().horizontalAdvance(title) + 24))
        table.horizontalHeader().setStretchLastSection(True)
        return table


