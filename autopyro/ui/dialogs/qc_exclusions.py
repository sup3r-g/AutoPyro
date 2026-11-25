"""Manage analysis-specific QC exclusions and their undoable updates."""

from PySide6.QtCore import QItemSelection, QItemSelectionModel, QSignalBlocker, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTableView,
    QVBoxLayout,
)

from autopyro.analysis.qc import SCOPES, active_exclusions, exclusion_time, parse_exclusion_time
from autopyro.model.core import timestamp

from ..i18n import tr
from ..table import QCCommand
from ..table.selection import add_row_ranges
from ..widgets.results import ResultsModel, sample_labels
from ..theme.controls import add_command_button


def exclude(editor, excluded):
    issue = editor.action_issue()
    if not issue or not issue["measurement_id"]:
        return
    reason = editor.exclusion_comment.text().strip()
    if excluded and not reason:
        editor.status.setText(tr('Укажите причину исключения.'))
        return
    scope = editor.scope.currentData()
    editor.exclusion_comment.clear()
    editor._exclusion_drafts.pop((issue["id"], scope), None)
    editor.update_exclusion(scope, issue["measurement_id"],
                          {"revision": issue["revision"], "comment": reason, "at": timestamp()}
                          if excluded else None)


def update_exclusion(editor, scope, mid, record, *, refresh=True):
    records = editor.owner.model.project.view.get("qc_exclusions", {}).get(scope, {})
    if record is None and mid not in records:
        return
    editor.owner.model.undo.push(QCCommand(editor.owner.model, ("qc_exclusions", scope), record,
                                         measurement_id=mid))
    if refresh:
        editor.reload()


def manage_exclusions(editor):
    project = editor.owner.model.project
    dialog = QDialog(editor)
    dialog.setWindowTitle(tr('Действующие исключения'))
    dialog.resize(950, 410)
    layout = QVBoxLayout(dialog)
    description = QLabel(tr('Действующие исключения по всему проекту'))
    layout.addWidget(description)
    search = QLineEdit()
    search.setPlaceholderText(tr('Найти исключение…'))
    search.setAccessibleName(search.placeholderText())
    search.setClearButtonEnabled(True)
    scopes = QComboBox()
    scopes.addItem(tr('Все виды анализа'), None)
    for scope, label in SCOPES.items():
        scopes.addItem(tr(label), scope)
    order = QComboBox()
    order.setAccessibleName(tr('Порядок исключений'))
    order.addItem(tr('Без сортировки по дате'), "original")
    order.addItem(tr('Сначала новые'), "newest")
    order.addItem(tr('Сначала старые'), "oldest")
    filters = QHBoxLayout()
    filters.addWidget(search, 1)
    filters.addWidget(scopes)
    filters.addWidget(order)
    layout.addLayout(filters)
    table = QTableView(dialog)
    model = ResultsModel(['Проба', 'Показатель', 'Исходная запись', 'Вид анализа', 'Измерение',
                          'Установлено', 'Причина'], [], table)
    table.setModel(model)
    table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
    table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
    for index, width in enumerate((140, 95, 110, 125, 150, 170)):
        table.setColumnWidth(index, width)
    layout.addWidget(table)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.button(QDialogButtonBox.StandardButton.Close).setText(tr('Закрыть'))
    remove = add_command_button(buttons, 'eye', tr('Снять исключение'), QDialogButtonBox.ButtonRole.ActionRole)
    open_card = buttons.addButton(tr('Измерение…'), QDialogButtonBox.ButtonRole.ActionRole)
    layout.addWidget(buttons)
    buttons.rejected.connect(dialog.reject)
    def selected_rows():
        return sorted({row for area in table.selectionModel().selection()
                       for row in range(area.top(), area.bottom() + 1)})

    def selection_changed(*_):
        count = len(selected_rows())
        remove.setEnabled(count > 0)
        open_card.setEnabled(count == 1)
        remove.setText(tr('Снять исключения ({0})', count) if count > 1 else tr('Снять исключение'))

    table.selectionModel().selectionChanged.connect(selection_changed)
    entries = []
    visible = []
    modified = False
    entries_revision = None

    def read_entries():
        nonlocal entries_revision
        exclusions = list(active_exclusions(project))
        labels = sample_labels(project, (m.sample_id for _, m, _ in exclusions))
        columns = {col.id: col.name for col in project.columns}
        analysis_names = {scope: tr(label) for scope, label in SCOPES.items()}
        entries.clear()
        working = project.science.working
        for scope, m, record in exclusions:
            is_working = working.get(m.sample_id, {}).get(m.property_id) == m.id
            values = (labels.get(m.sample_id, m.sample_id), columns.get(m.property_id, m.property_id),
                      "—" if m.raw_value is None else str(m.raw_value),
                      analysis_names.get(scope, scope),
                      tr('Рабочий результат') if is_working else tr('Не выбран рабочим'),
                      exclusion_time(record.get("at")),
                      record.get("comment", ""))
            searchable = " ".join((*values, m.sample_id, m.id)).casefold()
            instant = parse_exclusion_time(record.get("at"))
            entries.append((scope, m.id, values, searchable,
                            instant if instant is not None and instant.tzinfo is not None else None))
        entries_revision = editor.owner.analysis_revision

    def populate(*, preserve_selection=False):
        selection_model = table.selectionModel()
        selected = {visible[row][:2] for row in selected_rows()} if preserve_selection else set()
        current = selection_model.currentIndex()
        current_key = visible[current.row()][:2] if selected and current.isValid() else None
        scope = scopes.currentData()
        query = search.text().strip().casefold()
        visible[:] = [entry for entry in entries
                      if (scope is None or entry[0] == scope) and query in entry[3]]
        direction = order.currentData()
        if direction != "original":
            dated = sorted((entry for entry in visible if entry[4] is not None),
                           key=lambda entry: entry[4], reverse=direction == "newest")
            visible[:] = dated + [entry for entry in visible if entry[4] is None]
        with QSignalBlocker(selection_model):
            model.set_rows([entry[2] for entry in visible])
            if selected:
                rows = [row for row, entry in enumerate(visible) if entry[:2] in selected]
                selection = QItemSelection()
                add_row_ranges(selection, model, rows, 0, 0)
                flags = QItemSelectionModel.SelectionFlag
                selection_model.select(selection, flags.ClearAndSelect | flags.Rows)
                if rows:
                    focused = next((row for row in rows if visible[row][:2] == current_key), rows[0])
                    index = model.index(focused, 0)
                    selection_model.setCurrentIndex(index, flags.NoUpdate)
                    table.scrollTo(index)
        selection_changed()
        description.setText(tr('Действующие исключения: {0} из {1}', len(visible), len(entries)))

    def refresh_if_changed():
        nonlocal modified
        if project is not editor.owner.model.project:
            dialog.reject()
            return True
        if entries_revision == editor.owner.analysis_revision:
            return False
        read_entries()
        populate()
        modified = True
        description.setText(tr('Данные изменились. Проверьте замечание перед применением действия.'))
        return True

    def remove_selected():
        nonlocal modified
        if refresh_if_changed():
            return
        selected = [visible[row][:2] for row in selected_rows()]
        if not selected:
            return
        if len(selected) == 1:
            scope, mid = selected[0]
            editor.update_exclusion(scope, mid, None, refresh=False)
        else:
            table_model = editor.owner.model
            confirmation_revision = editor.owner.analysis_revision
            answer = QMessageBox.question(
                dialog, tr('Снять исключения'), tr('Снять выбранные исключения ({0})?', len(selected)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            if refresh_if_changed():
                return
            if confirmation_revision != editor.owner.analysis_revision:
                populate()
                description.setText(tr('Данные изменились. Проверьте замечание перед применением действия.'))
                return
            by_scope = {}
            for scope, mid in selected:
                by_scope.setdefault(scope, []).append(mid)
            table_model.undo.beginMacro(tr('Снять исключения'))
            try:
                for scope, ids in by_scope.items():
                    table_model.undo.push(QCCommand(table_model, ("qc_exclusions", scope), None,
                                                    measurement_ids=ids))
            finally:
                table_model.undo.endMacro()
        editor.timer.stop()
        modified = True
        read_entries()
        populate()

    def open_selected():
        nonlocal modified
        if refresh_if_changed():
            return
        rows = selected_rows()
        if len(rows) != 1:
            return
        mid = visible[rows[0]][1]
        measurement = project.science.measurements.get(mid)
        table_model = editor.owner.model
        if (measurement is None or not table_model.resolve_row_positions([measurement.sample_id])
                or not any(column.id == measurement.property_id for column in project.columns)):
            description.setText(tr('Данные изменились. Проверьте замечание перед применением действия.'))
            return
        revision = editor.owner.analysis_revision
        editor.show_measurement_card(measurement, dialog)
        if editor.owner.analysis_revision != revision:
            editor.timer.stop()
            read_entries()
            populate()
            modified = True

    open_card.clicked.connect(open_selected)
    table.doubleClicked.connect(lambda _: open_selected())
    remove.clicked.connect(remove_selected)
    search.textChanged.connect(lambda _: populate(preserve_selection=True))
    scopes.currentIndexChanged.connect(lambda _: populate(preserve_selection=True))
    order.currentIndexChanged.connect(lambda _: populate(preserve_selection=True))

    def refresh_entries():
        nonlocal modified
        if project is editor.owner.model.project and entries_revision != editor.owner.analysis_revision:
            read_entries()
            populate(preserve_selection=True)
            modified = True

    refresh_timer = QTimer(dialog)
    refresh_timer.setSingleShot(True)
    refresh_timer.setInterval(100)
    refresh_timer.timeout.connect(refresh_entries)
    source_model = editor.owner.model
    source_model.changed.connect(refresh_timer.start)
    read_entries()
    populate()
    try:
        dialog.exec()
    finally:
        source_model.changed.disconnect(refresh_timer.start)
        refresh_timer.stop()
        dialog.deleteLater()
    if modified and not editor._check_error:
        editor.reload()
