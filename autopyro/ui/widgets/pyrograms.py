"""Review attached source signals without deriving analytical measurements."""

from collections import defaultdict
from pathlib import Path
import numpy as np

from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QUndoCommand
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout,
)

from autopyro.curves.pyrograms import mapped_pyrogram, read_pyrogram, read_pyrogram_table, verify_pyrogram_source
from ..i18n import message, tr
from ..tasks import TaskCancelled, run_with_progress
from .figure_canvas import FittedFigureCanvas
from .navigation import NavigationToolbar


class PyrogramColumnsDialog(QDialog):
    def __init__(self, parent, table):
        super().__init__(parent)
        self.source = self.table = table
        self.setWindowTitle(tr('Колонки пирограммы'))
        layout = QFormLayout(self)
        self.header_line = QSpinBox()
        self.header_line.setRange(1, max(table.header_line, len(table.text.splitlines())))
        self.header_line.setValue(table.header_line)
        self.delimiter = QComboBox()
        for label, value in ((tr('Автоматически'), None), (tr('Табуляция'), '\t'),
                             (tr('Точка с запятой'), ';'), (tr('Запятая'), ',')):
            self.delimiter.addItem(label, value)
        if Path(table.file).suffix.lower() == '.tsv':
            self.delimiter.setCurrentIndex(self.delimiter.findData('\t'))
        self.layout_notice = QLabel()
        self.layout_notice.setTextFormat(Qt.TextFormat.PlainText)
        self.layout_notice.setWordWrap(True)
        layout.addRow(tr('Строка заголовка'), self.header_line)
        layout.addRow(tr('Разделитель'), self.delimiter)
        layout.addRow(self.layout_notice)
        self.columns = {}
        for key, label in (
            ('time', tr('Время')), ('signal', tr('Сигнал')), ('temperature', tr('Температура [°C]')),
        ):
            combo = self.columns[key] = QComboBox()
            layout.addRow(label, combo)
        self.time_unit = QComboBox()
        self.time_unit.addItem(tr('Секунды'), 's')
        self.time_unit.addItem(tr('Минуты'), 'min')
        self.time_unit.addItem(tr('ЧЧ:ММ:СС'), 'clock')
        layout.addRow(tr('Формат времени'), self.time_unit)
        self.first_line, self.last_line = QSpinBox(), QSpinBox()
        for field, label, default in (
            (self.first_line, tr('Первая строка запуска'), tr('Сразу после заголовка')),
            (self.last_line, tr('Последняя строка запуска'), tr('До конца файла')),
        ):
            field.setRange(0, self.header_line.maximum())
            field.setSpecialValueText(default)
            layout.addRow(label, field)
        self.fields = {}
        for key, label in (
            ('label', tr('Название запуска')), ('channel', tr('Канал')),
            ('stage', tr('Стадия')), ('unit', tr('Единица сигнала')),
        ):
            field = self.fields[key] = QLineEdit(Path(table.file).stem if key == 'label' else '')
            layout.addRow(label, field)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.import_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.import_button.setText(tr('Импортировать'))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        for combo in self.columns.values():
            combo.currentIndexChanged.connect(self.update_import_button)
        for field in self.fields.values():
            field.textChanged.connect(self.update_import_button)
        self.header_line.valueChanged.connect(self.update_layout)
        self.delimiter.currentIndexChanged.connect(self.update_layout)
        self.first_line.valueChanged.connect(self.update_import_button)
        self.last_line.valueChanged.connect(self.update_import_button)
        self.update_layout()

    def update_layout(self):
        try:
            self.table = self.source.with_layout(self.header_line.value(), self.delimiter.currentData())
        except ValueError as exc:
            self.table = None
            self.layout_notice.setText(message(exc))
        else:
            self.layout_notice.setText(tr('Распознано колонок: {0}', len(self.table.headers)))
        headers = self.table.headers if self.table else ()
        for key, combo in self.columns.items():
            previous = combo.currentText() if combo.currentData() is not None else ''
            with QSignalBlocker(combo):
                combo.clear()
                combo.addItem(tr('Не импортировать') if key == 'temperature' else tr('Выберите колонку'), None)
                for index, header in enumerate(headers):
                    combo.addItem(header, index)
                combo.setCurrentIndex(max(0, combo.findText(previous)))
        self.update_import_button()

    def update_import_button(self):
        indices = [combo.currentData() for combo in self.columns.values() if combo.currentData() is not None]
        first, last = self.first_line.value(), self.last_line.value()
        header = self.header_line.value()
        self.import_button.setEnabled(
            self.table is not None
            and self.columns['time'].currentData() is not None and self.columns['signal'].currentData() is not None
            and len(indices) == len(set(indices)) and all(field.text().strip() for field in self.fields.values())
            and (not first or first > header) and (not last or last >= max(header + 1, first))
        )

    def mapping(self):
        return {**{key: combo.currentData() for key, combo in self.columns.items()},
                **{key: field.text().strip() for key, field in self.fields.items()},
                'time_unit': self.time_unit.currentData(), 'first_line': self.first_line.value() or None,
                'last_line': self.last_line.value() or None}


class PyrogramMetadataCommand(QUndoCommand):
    def __init__(self, project, sample_id, key, value, description, notify):
        super().__init__(description)
        self.project, self.sample_id, self.key = project, sample_id, key
        self.before, self.after = project.science.samples[sample_id].metadata.get(key), value
        self.notify = notify

    def audit_value(self, value):
        if self.key != "pyrograms" or value is None:
            return value
        return [{key: entry[key] for key in ("file", "sha256", "table_mapping") if key in entry}
                for entry in value]

    def apply(self, value, action):
        metadata = self.project.science.samples[self.sample_id].metadata
        previous = self.audit_value(metadata.get(self.key))
        if value is None:
            metadata.pop(self.key, None)
        else:
            metadata[self.key] = value
        self.project.event(action, sample_id=self.sample_id, field=self.key,
                           previous=previous, value=self.audit_value(value))
        if self.notify is not None:
            self.notify()

    def redo(self):
        self.apply(self.after, "pyrogram_metadata")

    def undo(self):
        self.apply(self.before, "undo_pyrogram_metadata")


def preview_signal(x, y, *, limit=6000, bounds=None):
    """Keep coordinate and signal extrema in acquisition order for a bounded preview."""
    x, y = np.asarray(x), np.asarray(y)
    if bounds is not None:
        low, high = sorted(bounds)
        visible = (x >= low) & (x <= high)
        # Include crossing segments even when neither endpoint lies in the viewport.
        crossing = (np.minimum(x[:-1], x[1:]) <= high) & (np.maximum(x[:-1], x[1:]) >= low)
        visible[:-1] |= crossing
        visible[1:] |= crossing
        x, y = x[visible], y[visible]
    if len(x) <= limit:
        return x, y
    bins = max(1, (limit - 2) // 4)
    block = (len(x) + bins - 1) // bins
    indices = [0, len(x) - 1]
    for start in range(0, len(x), block):
        stop = min(start + block, len(x))
        for values in (x[start:stop], y[start:stop]):
            indices.extend((start + int(values.argmin()), start + int(values.argmax())))
    selected = np.unique(indices)
    return x[selected], y[selected]


class PyrogramDialog(QDialog):
    changed = Signal()

    def __init__(self, parent, project, sample_id, *, undo_stack=None, notify=None):
        super().__init__(parent)
        self.project, self.sample_id = project, sample_id
        self.undo_stack, self.notify = undo_stack, notify
        self.metadata = project.science.samples[sample_id].metadata
        self.attachments = self.metadata.get('pyrograms', [])
        self.setWindowTitle(tr('Пирограммы пробы {0}', sample_id))
        self.resize(1080, 810)
        layout = QVBoxLayout(self)
        import_button = QPushButton(tr('Добавить пирограмму…'))
        import_button.clicked.connect(self.add_run)
        self.detach_button = QPushButton(tr('Открепить выбранный запуск…'))
        self.detach_button.setEnabled(False)
        self.detach_button.clicked.connect(self.detach_run)
        run_actions = QHBoxLayout()
        run_actions.addWidget(import_button)
        run_actions.addWidget(self.detach_button)
        self.verify_source_button = QPushButton(tr('Проверить исходный файл…'))
        self.verify_source_button.setEnabled(False)
        self.verify_source_button.clicked.connect(self.verify_source)
        run_actions.addWidget(self.verify_source_button)
        layout.addLayout(run_actions)
        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.runs = QListWidget()
        self.runs.setAccessibleName(tr('Запуски для наложения'))
        self.runs.setMaximumHeight(126)
        controls.addWidget(self.runs, 1)
        form = QFormLayout()
        self.trace = QComboBox()
        self.axis = QComboBox()
        self.axis.addItem(tr('Время [с]'), 'time')
        self.axis.addItem(tr('Температура [°C]'), 'temperature')
        self.normalize = QCheckBox(tr('Делить сигнал на его максимальный модуль'))
        form.addRow(tr('Канал / стадия'), self.trace)
        form.addRow(tr('Горизонтальная ось'), self.axis)
        form.addRow(self.normalize)
        controls.addLayout(form, 1)
        self.notice = QLabel()
        self.notice.setTextFormat(Qt.TextFormat.PlainText)
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        self.figure = Figure(layout='constrained')
        self.canvas = FittedFigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.observation = QPlainTextEdit(self.metadata.get('pyrogram_observation', ''))
        self.observation.setAccessibleName(tr('Наблюдение по пирограммам'))
        self.observation.setPlaceholderText(tr('Наблюдение по пирограммам для карточки пробы'))
        self.observation.setMaximumHeight(78)
        layout.addWidget(self.observation)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.save_observation = buttons.addButton(tr('Сохранить наблюдение'), QDialogButtonBox.ButtonRole.ActionRole)
        self.save_observation.setEnabled(False)
        self.save_observation.clicked.connect(self.store_observation)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for attachment in self.attachments:
            self.add_run_item(attachment)
        self.refresh_traces()
        self.runs.itemChanged.connect(self.draw_signals)
        self.runs.currentRowChanged.connect(self.update_detach_button)
        self.trace.currentIndexChanged.connect(self.draw_signals)
        self.axis.currentIndexChanged.connect(self.draw_signals)
        self.normalize.toggled.connect(self.draw_signals)
        self.observation.textChanged.connect(self.update_observation_button)
        if self.runs.count() == 1:
            self.runs.setCurrentRow(0)
        self.draw_signals()

    def add_run_item(self, attachment):
        run = attachment['run']
        text = f"{run['label']} · {attachment['file']}"
        item = QListWidgetItem(text)
        details = [text, run.get('heating_program', '')]
        if attachment.get('table_mapping'):
            mapping = attachment['table_mapping']
            columns = mapping['columns']
            details.append(tr('Колонки: {0}', ' / '.join(columns.values())))
            if mapping.get('first_line') is not None or mapping.get('last_line') is not None:
                details.append(tr('Строки данных: {0}–{1}', mapping.get('first_line') or tr('после заголовка'),
                                  mapping.get('last_line') or tr('конец файла')))
        if run.get('analysis_id'):
            details.append(tr('Анализ: {0}', run['analysis_id']))
        item.setToolTip('\n'.join(filter(None, details)))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        self.runs.addItem(item)

    def refresh_traces(self):
        previous = self.trace.currentData()
        choices = sorted({(trace['channel'], trace['stage'])
                          for entry in self.attachments for trace in entry['run']['traces']})
        with QSignalBlocker(self.trace):
            self.trace.clear()
            for channel, stage in choices:
                self.trace.addItem(f'{channel} / {stage}', (channel, stage))
            index = choices.index(previous) if previous in choices else 0
            self.trace.setCurrentIndex(index if choices else -1)

    def add_run(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr('Исходная пирограмма'), '', tr('Пирограммы (*.json *.csv *.tsv *.s00 *.S00)'),
        )
        if not path:
            return
        try:
            if Path(path).suffix.lower() in ('.csv', '.tsv'):
                table = run_with_progress(self, tr('Исходная пирограмма'),
                                          lambda update: read_pyrogram_table(path, allow_unmapped=True), cancellable=True)
                dialog = PyrogramColumnsDialog(self, table)
                accepted = dialog.exec()
                table, mapping = dialog.table, dialog.mapping()
                dialog.deleteLater()
                if not accepted:
                    return
                if table is None:
                    raise ValueError(tr('Укажите корректную строку заголовка.'))
                attachment = run_with_progress(
                    self, tr('Исходная пирограмма'),
                    lambda update: mapped_pyrogram(table, self.project, self.sample_id, **mapping), cancellable=True)
            else:
                attachment = run_with_progress(
                    self, tr('Исходная пирограмма'),
                    lambda update: read_pyrogram(path, self.project, self.sample_id), cancellable=True)
            if any(self.source_key(entry) == self.source_key(attachment) for entry in self.attachments):
                raise ValueError(tr('Этот ряд уже прикреплён к пробе.'))
        except TaskCancelled:
            return
        except (OSError, ValueError, OverflowError, RecursionError) as exc:
            QMessageBox.warning(self, tr('Исходная пирограмма'), message(exc))
            return
        self.attachments = [*self.attachments, attachment]
        self.store_metadata('pyrograms', self.attachments, tr('Добавить пирограмму…'))
        with QSignalBlocker(self.runs):
            self.add_run_item(attachment)
        self.runs.setCurrentRow(self.runs.count() - 1)
        self.refresh_traces()
        self.draw_signals()

    def store_metadata(self, key, value, description):
        if self.metadata.get(key) == value:
            return
        command = PyrogramMetadataCommand(
            self.project, self.sample_id, key, value, description, self.notify or self.changed.emit)
        if self.undo_stack is not None:
            self.undo_stack.push(command)
        else:
            command.redo()

    @staticmethod
    def source_key(attachment):
        mapping = attachment.get('table_mapping')
        if mapping is not None:
            mapping = {'header_line': 1, **mapping}
        return attachment['sha256'], mapping

    def update_detach_button(self, row):
        self.detach_button.setEnabled(0 <= row < len(self.attachments))
        self.verify_source_button.setEnabled(0 <= row < len(self.attachments))

    def verify_source(self):
        row = self.runs.currentRow()
        if not 0 <= row < len(self.attachments):
            return
        attachment = self.attachments[row]
        path, _ = QFileDialog.getOpenFileName(
            self, tr('Проверить исходный файл…'), attachment['file'], tr('Все файлы (*)'))
        if not path:
            return
        try:
            matches = run_with_progress(
                self, tr('Проверить исходный файл…'),
                lambda update: verify_pyrogram_source(path, attachment), cancellable=True)
        except TaskCancelled:
            return
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, tr('Исходная пирограмма'), message(exc))
            return
        text = (tr('Файл совпадает с исходником прикреплённой пирограммы.')
                if matches else tr('Файл отличается от исходника прикреплённой пирограммы.'))
        QMessageBox.information(self, tr('Исходная пирограмма'), text)

    def detach_run(self):
        row = self.runs.currentRow()
        if not 0 <= row < len(self.attachments):
            return
        answer = QMessageBox.question(
            self, tr('Открепить пирограмму'),
            tr('Убрать запуск «{0}» из выбранной пробы?', self.runs.item(row).text()),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.attachments = self.attachments[:row] + self.attachments[row + 1:]
        self.store_metadata('pyrograms', self.attachments or None, tr('Открепить пирограмму'))
        with QSignalBlocker(self.runs):
            self.runs.takeItem(row)
            self.runs.setCurrentRow(min(row, self.runs.count() - 1))
        self.update_detach_button(self.runs.currentRow())
        self.refresh_traces()
        self.draw_signals()

    def observation_changed(self):
        return self.observation.toPlainText().strip() != self.metadata.get('pyrogram_observation', '')

    def update_observation_button(self):
        self.save_observation.setEnabled(self.observation_changed())

    def store_observation(self):
        text = self.observation.toPlainText().strip()
        self.store_metadata('pyrogram_observation', text or None, tr('Наблюдение по пирограммам'))
        self.update_observation_button()

    def reject(self):
        if self.observation_changed():
            answer = QMessageBox.question(
                self, tr('Несохранённое наблюдение'), tr('Сохранить изменения наблюдения по пирограммам?'),
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Save:
                self.store_observation()
            elif answer == QMessageBox.StandardButton.Discard:
                self.observation.setPlainText(self.metadata.get('pyrogram_observation', ''))
            else:
                return
        super().reject()

    def draw_signals(self, *_):
        self.figure.clear()
        self.toolbar.update()
        key = self.trace.currentData()
        coordinate = self.axis.currentData()
        normalized = self.normalize.isChecked()
        groups = defaultdict(list)
        skipped = repeated_time = 0
        programs = []
        for i, entry in enumerate(self.attachments):
            if self.runs.item(i).checkState() != Qt.CheckState.Checked:
                continue
            run = entry['run']
            label = f"{run['label']} · {entry['file']}"
            for trace in run['traces']:
                if (trace['channel'], trace['stage']) != key:
                    continue
                if coordinate not in trace:
                    skipped += 1
                    continue
                unit = tr('Нормированный сигнал') if normalized else trace['unit']
                groups[unit].append((label, trace))
                repeated_time += trace.get('time_order') == 'nondecreasing'
                if run.get('heating_program'):
                    programs.append(f"{run['label']} · {run['heating_program']}")
        notices = programs
        if skipped:
            notices.append(tr('Без ряда температуры: {0}', skipped))
        if repeated_time:
            notices.append(tr('Рядов с повторяющимися отметками времени: {0}. Исходные точки сохранены.', repeated_time))
        if not groups:
            notices.append(tr('Добавьте пирограмму выбранной пробы (Rock-Eval S00, JSON, CSV или TSV).') if not self.attachments
                           else tr('Для выбранных запусков, канала и оси нет сигнала.'))
        self.notice.setText(' | '.join(notices))
        for panel, (unit, traces) in enumerate(groups.items(), 1):
            axis = self.figure.add_subplot(len(groups), 1, panel)
            for label, trace in traces:
                x, y = np.asarray(trace[coordinate]), np.asarray(trace['signal'])
                if normalized:
                    scale = np.max(np.abs(y)) or 1
                    y = y / scale
                line, = axis.plot(*preview_signal(x, y), label=label)
                line._pyrogram_source = (x, y)
                color = line.get_color()
                for peak in trace.get('peaks', ()):
                    i = peak['index']
                    axis.scatter(x[i], y[i], marker='v', color=color, zorder=3)
                    axis.annotate(peak['label'], (x[i], y[i]), xytext=(3, 5),
                                  textcoords='offset points', parse_math=False)
                for interval in trace.get('intervals', ()):
                    start, end = interval['start'], interval['end']
                    interval_x, interval_y = preview_signal(x[start:end + 1], y[start:end + 1])
                    interval_line, = axis.plot(interval_x, interval_y, color=color, lw=5, alpha=.3)
                    interval_line._pyrogram_source = (x[start:end + 1], y[start:end + 1])
                    axis.scatter([x[start], x[end]], [y[start], y[end]], color=color, s=9)
                    middle = (start + end) // 2
                    axis.annotate(interval['label'], (x[middle], y[middle]), xytext=(3, -11),
                                  textcoords='offset points', parse_math=False)
            axis.set_xlabel(self.axis.currentText())
            axis.set_ylabel(unit, parse_math=False)
            axis.grid(alpha=.25)
            for label in axis.legend(loc='upper right').get_texts():
                label.set_parse_math(False)
            axis.callbacks.connect('xlim_changed', self.refresh_visible_signal)
        self.canvas.draw_idle()

    def refresh_visible_signal(self, axis):
        for line in axis.lines:
            source = getattr(line, '_pyrogram_source', None)
            if source is not None:
                line.set_data(*preview_signal(*source, bounds=axis.get_xlim()))
        self.canvas.draw_idle()
