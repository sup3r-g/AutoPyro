"""Review interval findings, schema merges and revisions."""

from __future__ import annotations

from ..i18n import tr, message
from PySide6.QtWidgets import QCheckBox, QDialogButtonBox, QHeaderView
from autopyro.model.core import parse_number
from autopyro.geology.stratigraphy import interval_field_errors
from autopyro.geology.stratigraphy_review import database_source, diagnose, plan_merge
from ..theme import icon
from ..widgets.stratigraphy_revisions import RevisionComparisonDialog
from ..widgets.stratigraphy_summary import StratigraphySummaryDialog
from ..widgets.stratigraphy_labels import geological_label
from ..widgets.lithology import LithologyCatalog
from ..widgets.interval_history import MergeIntervals
from ..widgets.interval_review import IntervalReviewDialog


def compare_revisions(editor):
    project = editor.owner.model.project
    scheme_id = editor.draft['id']
    history = {(key, revision): scheme
               for key, revisions in project.view.get('stratigraphy_revisions', {}).items()
               for revision, scheme in revisions.items()}
    for scheme in (project.view.get('stratigraphy_scheme'),
                   (project.view.get('stratigraphy') or {}).get('scheme')):
        if scheme:
            history.setdefault((scheme['id'], scheme['revision']), scheme)
    try:
        draft = editor.read_scheme()
    except ValueError:
        draft = None
    if len(history) + bool(draft) < 2:
        editor.status.setText(tr('Для сравнения нужны две редакции или сохранённая редакция и корректный черновик.'))
        return
    editor.refresh_columns()
    revisions = [(revision, history[key, revision])
                 for key, revision in sorted(history, key=lambda pair: pair[0] == scheme_id)]
    dialog = RevisionComparisonDialog(editor, revisions, draft, editor.current_binding())
    dialog.exec()
    dialog.deleteLater()


def show_thickness_summary(editor):
    try:
        scheme = editor.read_scheme()
        custom = editor.column_style_form.rocks_editor.values()
        catalog = LithologyCatalog(custom) if custom else editor.lithology_catalog
        dialog = StratigraphySummaryDialog(scheme, catalog, editor)
        dialog.exec()
        dialog.deleteLater()
    except ValueError as exc:
        editor.status.setText(tr('Ошибка: {0}', message(exc)))


def review_intervals(editor):
    try:
        scheme = editor.read_scheme()
    except ValueError as exc:
        if not editor.review_interval_errors():
            editor.status.setText(tr('Ошибка: {0}', message(exc)))
        return
    findings = diagnose(scheme['intervals'])
    names = {item['id']: item['name'] for item in scheme['intervals']}
    labels = {'gap': tr('Разрыв'), 'overlap': tr('Перекрытие'), 'duplicate': tr('Дубль'),
              'missing': tr('Без литологии')}
    dialog = IntervalReviewDialog(
        editor, tr('Диагностика интервалов'),
        tr('Замечаний: {0}. Разрывы и перекрытия проверяются отдельно для каждого ствола и уровня.',
           len(findings)) if findings else tr('Замечаний нет.'),
        [tr('Статус'), tr('Ствол колонки'), tr('Уровень'), tr('Глубина от'), tr('Глубина до'), tr('Подразделение')],
        [(labels[item.kind], item.well, geological_label(item.level), item.top, item.bottom,
          ' · '.join(names[key] for key in item.ids))
         for item in findings])
    dialog.add_search()
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    locate = buttons.addButton(tr('Показать строки'), QDialogButtonBox.ButtonRole.ActionRole)
    locate.setIcon(icon('table-2'))
    plot = buttons.addButton(tr('Показать на колонке'), QDialogButtonBox.ButtonRole.ActionRole)
    plot.setIcon(icon('zoom-in'))
    buttons.rejected.connect(dialog.reject)
    dialog.layout_box.addWidget(buttons)

    def jump(show_plot):
        row = dialog.source_row()
        if row >= 0:
            finding = findings[row]
            dialog.accept()
            editor.locate_finding(finding, show_plot)

    locate.clicked.connect(lambda: jump(False))
    plot.clicked.connect(lambda: jump(True))
    dialog.table.doubleClicked.connect(lambda index: jump(True))
    if findings:
        dialog.table.selectRow(0)
    locate.setEnabled(bool(findings))
    plot.setEnabled(bool(findings))
    dialog.exec()
    dialog.deleteLater()


def review_interval_errors(editor):
    metadata = [(label, widget) for label, widget in (
        (tr('Название схемы'), editor.name),
        (tr('Редакция'), editor.revision),
        (tr('Тип / начало отсчёта интервалов'), editor.reference),
    ) if not widget.text().strip()]
    fields = ('wellbore', 'name', 'level', 'top', 'bottom', 'color', 'id', 'lithology')
    columns = {key: column for column, key in enumerate(fields)}
    errors = []
    seen = set()
    for row in range(editor.intervals.rowCount()):
        item = {key: editor.intervals.item(row, column).text() for key, column in columns.items()}
        for key in ('top', 'bottom'):
            try:
                item[key] = parse_number(item[key])
            except ValueError:
                item[key] = None
        if not item['id'] or item['id'] in seen:
            errors.append((row, columns['name'], tr('Каждому интервалу нужен уникальный ID')))
        seen.add(item['id'])
        for key, error in interval_field_errors(item):
            details = (tr('Выберите допустимый стратиграфический уровень.')
                       if key == 'level' else message(error))
            errors.append((row, columns[key], details))
    if not metadata and not errors:
        return False
    dialog = IntervalReviewDialog(
        editor, tr('Диагностика интервалов'),
        tr('Ошибок: {0}. Исправьте их, затем повторите проверку разрывов и перекрытий.',
           len(metadata) + len(errors)),
        [tr('Строка'), tr('Поле'), tr('Ошибка')],
        [(None, label, tr('Заполните обязательное поле.')) for label, _ in metadata]
        + [(row + 1, editor.intervals.horizontalHeaderItem(column).text(), error)
           for row, column, error in errors])
    header = dialog.table.horizontalHeader()
    for column, width in enumerate((90, 240)):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(column, width)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    locate = buttons.addButton(tr('Перейти к ячейке'), QDialogButtonBox.ButtonRole.ActionRole)
    locate.setIcon(icon('table-2'))
    buttons.rejected.connect(dialog.reject)
    dialog.layout_box.addWidget(buttons)

    def update_locate(current, previous):
        is_metadata = current.isValid() and current.row() < len(metadata)
        locate.setText(tr('Перейти к полю') if is_metadata else tr('Перейти к ячейке'))

    dialog.table.selectionModel().currentChanged.connect(update_locate)
    dialog.table.selectRow(0)

    locate.clicked.connect(dialog.accept)
    dialog.table.doubleClicked.connect(dialog.accept)
    if dialog.exec() == dialog.DialogCode.Accepted and (index := dialog.table.currentIndex()).isValid():
        editor.result_tabs.setCurrentWidget(editor.intervals)
        if index.row() < len(metadata):
            _, field = metadata[index.row()]
            editor.parameter_tabs.setCurrentIndex(0)
            field.setFocus()
            field.selectAll()
        else:
            row, column, _ = errors[index.row() - len(metadata)]
            editor.intervals.setCurrentCell(row, column)
            editor.intervals.scrollToItem(editor.intervals.item(row, column))
            editor.intervals.setFocus()
    dialog.deleteLater()
    return True


def locate_finding(editor, finding, show_plot):
    selection = editor.interval_selection
    ids = set(finding.ids)
    if show_plot:
        if not editor._has_column_preview or not ids <= selection.rendered.keys():
            editor.column_well.setCurrentIndex(editor.column_well.findData(finding.well))
            editor.level.setCurrentIndex(editor.level.findData(finding.level))
            editor.show_column()
        if editor._has_column_preview and ids <= selection.rendered.keys():
            editor.parameter_tabs.setCurrentIndex(1)
            editor.result_tabs.setCurrentWidget(editor.column_page)
            selection.select_ids(finding.ids)
            origin = selection.origins.get(finding.well, 0)
            selection.zoom_to_depth(finding.top - origin, finding.bottom - origin)
            return
    editor.parameter_tabs.setCurrentIndex(0)
    editor.result_tabs.setCurrentWidget(editor.intervals)
    selection.select_ids(finding.ids)


def review_merge(editor, current, incoming):
    from_database = database_source(incoming.get('source'))
    entries = plan_merge(current, incoming, update_database=False)
    labels = {'add': tr('Добавление'), 'update': tr('Обновление'), 'duplicate': tr('Дубль'),
              'conflict': tr('Конфликт')}
    reasons = {'identity': tr('Неоднозначный идентификатор источника'),
               'bounds': tr('Совпадающие границы'),
               'overlap': tr('Перекрытие'), '': ''}

    def field_value(entry, key):
        value = entry.item[key]
        display = geological_label(value) if key == 'level' else value
        if from_database and entry.previous is not None and entry.previous[key] != value:
            previous = entry.previous[key]
            if key == 'level':
                previous = geological_label(previous)
            return f'{previous} → {display}'
        return display

    dialog = IntervalReviewDialog(
        editor, tr('Обновление интервалов из PostgreSQL') if from_database else tr('Объединение интервалов'),
        '', [tr('Статус'), tr('Ствол колонки'), tr('Уровень'), tr('Название подразделения'),
             tr('Кровля'), tr('Подошва'), tr('Причина'), tr('Конфликтующий интервал')], [])
    model = dialog.table.model()
    dialog.add_search()
    replace = QCheckBox(tr('Обновить существующие интервалы') if from_database
                        else tr('Обновить интервалы с точно совпадающими границами'))
    replace.setEnabled(any(entry.kind == 'update' for entry in entries))
    dialog.layout_box.addWidget(replace)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    apply = buttons.button(QDialogButtonBox.StandardButton.Ok)
    apply.setText(tr('Применить'))
    apply.setIcon(icon('check'))
    plans = {False: entries}
    selected = []
    current_by_id = {item['id']: item for item in current['intervals']}

    def geometry_findings(intervals):
        return {(finding.kind, finding.well, finding.level, finding.top, finding.bottom)
                for finding in diagnose(intervals) if finding.kind in ('gap', 'overlap')}

    baseline = geometry_findings(current['intervals']) if from_database else set()
    locate = buttons.addButton(tr('Показать строки'), QDialogButtonBox.ButtonRole.ActionRole)
    locate.setIcon(icon('table-2'))

    def conflict_target():
        row = dialog.source_row()
        if 0 <= row < len(active_entries):
            entry = active_entries[row]
            other = entry.conflict_with
            if other is None and entry.kind == 'conflict':
                other = entry.previous
            if other is not None and other['id'] in current_by_id:
                return other['id']
        return None

    def jump():
        key = conflict_target()
        if key is not None:
            dialog.reject()
            editor.result_tabs.setCurrentWidget(editor.intervals)
            editor.interval_selection.select_ids([key])

    locate.clicked.connect(jump)
    dialog.table.selectionModel().currentChanged.connect(lambda *_: locate.setEnabled(conflict_target() is not None))
    active_entries = entries

    def refresh_preview(checked):
        nonlocal active_entries, selected
        if checked not in plans:
            plans[checked] = plan_merge(current, incoming) if from_database else entries
        active_entries = plans[checked]
        blocked = from_database and checked and any(entry.kind == 'conflict' for entry in active_entries)
        selected = [entry for entry in active_entries
                    if entry.kind == 'add' or (checked and entry.kind == 'update')]
        rows = []
        for entry in active_entries:
            other = entry.conflict_with or (entry.previous if entry.kind == 'conflict' else None)
            detail = (f"{other['wellbore']} · {other['name']} · {other['top']}–{other['bottom']}"
                      if other else '')
            rows.append((labels[entry.kind], *(field_value(entry, key) for key in
                         ('wellbore', 'level', 'name', 'top', 'bottom')), reasons[entry.reason], detail))
        note = tr('Добавления: {0}. Возможные обновления: {1}. Дубли: {2}. Конфликты: {3}.',
                  *(sum(entry.kind == kind for entry in active_entries) for kind in labels))
        if blocked:
            note += ' ' + tr('Обновление из БД заблокировано, чтобы не применить связанную правку частично.')
            selected = []
        elif from_database:
            replacements = {entry.target: entry.item for entry in selected if entry.target}
            proposed = [replacements.get(key, item) for key, item in current_by_id.items()]
            proposed.extend(entry.item for entry in selected if not entry.target)
            labels_by_finding = {'gap': tr('Разрыв'), 'overlap': tr('Перекрытие')}
            for kind, well, level, top, bottom in sorted(geometry_findings(proposed) - baseline):
                rows.append((labels_by_finding[kind], well, geological_label(level), '', top, bottom, '', ''))
        model.set_rows(rows)
        locate.setEnabled(False)
        dialog.summary.setText(note)
        apply.setEnabled(bool(selected))

    replace.toggled.connect(refresh_preview)
    refresh_preview(False)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    dialog.layout_box.addWidget(buttons)
    accepted = dialog.exec() == dialog.DialogCode.Accepted
    dialog.deleteLater()
    if not accepted or not selected:
        return False
    editor.draft_undo.push(MergeIntervals(editor.interval_history, selected))
    editor.status.setText(tr('Объединение выполнено. Изменено интервалов: {0}.', len(selected)))
    return True
