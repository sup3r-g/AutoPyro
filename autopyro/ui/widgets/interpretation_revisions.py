"""Compare the previous and current calculation snapshots."""

from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QLabel, QTableView, QTextBrowser, QVBoxLayout,
)

from autopyro.schemes.interpretation import STATUSES
from autopyro.schemes.interpretation_revisions import input_differences, revision_changes
from ..i18n import tr
from .results import ResultsModel


FLAGS = {
    'added': 'Вошла в выборку',
    'removed': 'Выбыла из выборки',
    'inputs': 'Изменились входные данные',
    'outcome': 'Изменился вывод',
    'ambiguity_added': 'Появилась неоднозначность',
    'ambiguity_removed': 'Исчезла неоднозначность',
    'agreement': 'Изменилось согласие диаграмм',
    'review': 'Требуется повторная экспертная проверка',
}

INPUT_FIELDS = {
    'binding': 'Привязка', 'value': 'Исходное значение', 'status': 'Статус входа',
    'measurement': 'Рабочее измерение', 'uncertainty': 'Неопределённость',
    'bounds': 'Границы неопределённости', 'definition': 'Контрольная сумма определения осей',
    'text_numbers': 'Чтение чисел из текста', 'quality': 'Замечания контроля качества',
    'quality_blocked': 'Расчёт заблокирован контролем качества',
}
INPUT_STATUSES = {
    'numeric': 'Число', 'missing': 'Нет значения', 'other': 'Нечисловое значение',
    'non_detect': 'Не обнаружено', 'censored': 'Значение с квалификатором',
    'range_excluded': 'Вне допустимого диапазона',
    'qc_excluded': 'Исключено контролем качества', 'interpolation_excluded': 'Интерполированное значение',
}


def input_text(field, value, column_name=None):
    if value is None or value == () or value == [] or value == '':
        return '—'
    if field == 'binding':
        prop, column_id, unit = value
        column = f'{column_name} ({column_id})' if column_name else column_id
        return f'{prop} [{unit}] · {column}'
    if field == 'measurement':
        return tr('{0}, версия {1}', value['id'], value['revision'])
    if field == 'uncertainty':
        text = f"{value['uncertainty']:g} {value.get('unit', '')}".strip()
        if value.get('type'):
            text += f" · {value['type']}"
        if value.get('confidence_level') is not None:
            text += ' · ' + tr('Уровень доверия {0}', value['confidence_level'])
        return text
    if field == 'bounds':
        return ' – '.join(f'{number:g}' for number in value)
    if field == 'quality':
        return ' | '.join(issue['message'] for issue in value)
    if field == 'status':
        return tr(INPUT_STATUSES.get(value, value))
    if isinstance(value, bool):
        return tr('Да') if value else tr('Нет')
    return str(value)


def outcome_text(outcomes):
    lines = []
    for name, (status, labels, value, bounds, uncertain) in outcomes.items():
        line = tr(STATUSES[status])
        if name:
            line = f'{name} · {line}'
        if labels:
            line += ' · ' + ' / '.join(labels)
        if value is not None:
            line += f' · {value:g}'
        if bounds:
            line += f' [{bounds[0]:g}, {bounds[1]:g}]'
        if uncertain:
            line += ' · ' + tr('Области интервала: {0}', ' / '.join(uncertain))
        lines.append(line)
    return '\n'.join(lines) or '—'


class InterpretationRevisionsDialog(QDialog):
    def __init__(self, parent, project, previous, current, *, pinned=False):
        super().__init__(parent)
        self.setWindowTitle(tr('Сравнение расчётов интерпретации'))
        self.resize(1160, 630)
        self.row_id = None
        self.previous, self.current = previous, current
        self.changes = list(revision_changes(project, previous, current))
        layout = QVBoxLayout(self)
        for label, report in [(tr('Закреплённая база') if pinned else tr('До'), previous), (tr('После'), current)]:
            scheme = report['scheme']
            heading = QLabel(tr('{0} · {1} · редакция {2} · проб: {3}',
                                label, scheme['name'], scheme['revision'], len(report['results'])))
            heading.setWordWrap(True)
            layout.addWidget(heading)
        layout.addWidget(QLabel(tr('Проб с изменениями или повторной проверкой: {0}', len(self.changes))))
        self.table = QTableView()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        rows = [
            (change.row_id, '\n'.join(tr(FLAGS[flag]) for flag in change.flags),
             outcome_text(change.before), outcome_text(change.after))
            for change in self.changes
        ]
        self.table.setModel(ResultsModel(
            [tr('UUID пробы'), tr('Изменения'), tr('До'), tr('После')], rows, self.table))
        for column, width in enumerate((265, 250, 300, 300)):
            self.table.setColumnWidth(column, width)
        self.table.verticalHeader().setDefaultSectionSize(74)
        self.table.doubleClicked.connect(self.open_sample)
        layout.addWidget(self.table, 1)
        layout.addWidget(QLabel(tr('Входные данные выбранной пробы')))
        self.input_details = QTextBrowser()
        self.input_details.setAccessibleName(tr('Изменения входных данных'))
        self.input_details.setFixedHeight(125)
        layout.addWidget(self.input_details)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.show_sample = buttons.addButton(tr('Показать пробу'), QDialogButtonBox.ButtonRole.ActionRole)
        self.show_sample.clicked.connect(lambda: self.open_sample(self.table.currentIndex()))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.table.selectionModel().currentRowChanged.connect(self.update_action)
        self.show_sample.setEnabled(False)
        if self.changes:
            self.table.selectRow(0)
        else:
            self.update_action(self.table.currentIndex(), None)

    def update_action(self, index, _previous):
        if not index.isValid():
            self.show_sample.setEnabled(False)
            self.input_details.setPlainText(tr('Выберите пробу для сравнения входных данных.'))
            return
        change = self.changes[index.row()]
        self.show_sample.setEnabled(bool(change.after))
        lines = []
        current_group = None
        for name, axis, field, left, right in input_differences(self.previous, self.current, change):
            before = self.previous if name is None else self.previous['comparisons'][name]
            after = self.current if name is None else self.current['comparisons'][name]
            old_column = before.get('axis_column_names', {}).get(axis)
            new_column = after.get('axis_column_names', {}).get(axis)
            group = (name, axis)
            if group != current_group:
                current_group = group
                heading = name or tr('Основная схема')
                if axis:
                    columns = [column for column in dict.fromkeys((old_column, new_column)) if column]
                    heading += f" · {axis.upper()} · " + (' → '.join(columns) or after['scheme'][axis]['property'])
                if lines:
                    lines.append('')
                lines.append(heading)
            lines.append(tr('{0}: {1} → {2}', tr(INPUT_FIELDS[field]),
                            input_text(field, left, old_column), input_text(field, right, new_column)))
        if not lines:
            if None in change.positions:
                lines.append(tr('Проба присутствует только в одном расчёте.'))
            else:
                lines.append(tr('Изменений входных данных на общих диаграммах нет.'))
        self.input_details.setPlainText('\n'.join(lines))

    def open_sample(self, index):
        if index.isValid() and self.changes[index.row()].after:
            self.row_id = self.changes[index.row()].row_id
            self.accept()
