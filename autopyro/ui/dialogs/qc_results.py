"""Display, filter, sort and select quality-control findings."""

from PySide6.QtCore import (
    QAbstractTableModel,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QSignalBlocker,
    Qt,
)

from autopyro.analysis.qc import (
    PROFILE_UNITS,
    QC_SEVERITY_ORDER,
    RULES,
    SCOPES,
    active_exclusions,
    exclusion_time,
    pyrolysis_profile,
    role,
)
from autopyro.model.measurements import derived
from autopyro.model.source_location import source_location_parts

from ..i18n import tr, message
from ..table.model import context_issue_text
from ..table.selection import add_row_ranges
from ..theme import icon


def source_search_text(source):
    return " · ".join(f"{key}: {value}" for key, value in source.items() if value is not None)


def source_location_labels():
    return {
        "file": tr('Файл: {0}'), "database": tr('База: {0}'), "source": tr('Источник: {0}'),
        "sheet": tr('Лист: {0}'), "table": tr('Таблица: {0}'), "row": tr('Строка: {0}'),
        "record": tr('Запись источника: {0}'),
    }


def source_location(source, labels=None):
    if labels is None:
        labels = source_location_labels()
    return " · ".join(labels[key].format(value) for key, value in source_location_parts(source))


class QCIssuesModel(QAbstractTableModel):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.issues = []
        self.columns = {}
        self.samples = {}
        self.headers = [tr('Уровень'), tr('Показатель'), tr('Проба'), tr('Проверка'), tr('Описание'),
                        tr('Решение'), tr('Комментарий'), tr('Источник')]

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.issues)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return self.headers[section] if orientation == Qt.Orientation.Horizontal else str(section + 1)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        issue = self.issues[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return issue
        if role == Qt.ItemDataRole.DecorationRole:
            if index.column() == 0:
                return icon('x' if issue['severity'] in ('error', 'critical') else 'circle-question-mark')
            if index.column() == 5:
                return icon('shield-check' if issue['decision'] == 'acknowledged' else 'circle-question-mark')
            return None
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return None
        column = index.column()
        if column == 0:
            return {"error": tr('Ошибка'), "critical": tr('Критично'),
                    "warning": tr('Предупреждение') if role == Qt.ItemDataRole.ToolTipRole
                    else tr('Предупр.')}[issue["severity"]]
        if column == 1:
            return self.columns[issue['property_id']].name
        if column == 2:
            label = self.samples.get(issue['sample_id'], issue['sample_id'])
            return f"{label}\n{issue['sample_id']}" if role == Qt.ItemDataRole.ToolTipRole and label != issue['sample_id'] else label
        if column == 3:
            return tr(RULES[issue['rule_id']]["description"])
        if column == 4:
            if "profile" in issue:
                data = issue["profile"]
                return f"{data['role']} {data['display_value']} · {tr(RULES[issue['rule_id']]['description'])}"
            return message(issue['message'])
        if column == 5:
            return tr('Принято') if issue['decision'] == 'acknowledged' else tr('Открыто')
        if column == 6:
            return issue['comment']
        return source_location(issue['source']) or "—"

    def replace(self, issues):
        self.beginResetModel()
        self.columns = {column.id: column for column in self.project.columns}
        self.issues = issues
        self.endResetModel()


def current_issue(editor):
    return editor.table.currentIndex().data(Qt.ItemDataRole.UserRole)


def select_issue_row(editor, row):
    # QTableView.selectRow follows the current keyboard modifiers.
    flags = QItemSelectionModel.SelectionFlag
    index = editor.results_model.index(row, 0)
    editor.table.selectionModel().setCurrentIndex(index, flags.ClearAndSelect | flags.Rows)


def selected_spans(editor):
    spans = []
    for top, bottom in sorted((area.top(), area.bottom()) for area in editor.table.selectionModel().selection()):
        if spans and top <= spans[-1][1] + 1:
            spans[-1] = spans[-1][0], max(spans[-1][1], bottom)
        else:
            spans.append((top, bottom))
    return spans


def selected_issue(editor):
    spans = editor.selected_spans()
    if len(spans) == 1 and spans[0][0] == spans[0][1]:
        return editor.results_model.issues[spans[0][0]]
    return None


def render(editor, *_):
    editor.retry.setVisible(bool(editor._check_error))
    if editor._check_error:
        editor.coverage_button.setEnabled(False)
        editor._render_key = None
        editor.results_model.replace([])
        editor.show_details()
        editor.status.setText(editor._check_error)
        return
    profile = pyrolysis_profile(editor.owner.model.project)
    sample_scope = profile.get("sample_scope")
    editor.profile_status.setText(tr('Пороговый профиль включён') if profile["enabled"]
                                else tr('Пороговый профиль выключен'))
    selected = editor.current_issue()
    selected_id = selected["id"] if selected else None
    selected_ids = {editor.results_model.issues[row]["id"]
                    for top, bottom in editor.selected_spans() for row in range(top, bottom + 1)}
    columns = {c.id: c for c in editor.owner.model.project.columns}
    property_query, sample_query, source_query = (widget.text().strip().casefold() for widget in
                                                 (editor.property_filter, editor.sample, editor.source))
    severity, state = editor.severity.currentData(), editor.state.currentData()
    rule = editor.rule_filter.currentData()
    names = {cid: column.name.casefold() for cid, column in columns.items()}
    scope = editor.owner.analysis_scope
    scope_key = (id(editor.issues), len(editor.issues), scope.revision)
    order = QC_SEVERITY_ORDER
    if editor._scope_cache is None or editor._scope_cache[0] != scope_key:
        try:
            row_ids = set(scope.rows())
        except ValueError as exc:
            editor.coverage_button.setEnabled(False)
            editor._render_key = None
            editor.results_model.replace([])
            editor.show_details()
            editor.status.setText(message(exc))
            return
        scoped = []
        levels = dict.fromkeys(order, 0)
        for issue in editor.issues:
            if issue["property_id"] in columns and issue["sample_id"] in row_ids:
                scoped.append(issue)
                levels[issue["severity"]] += 1
        coverage = {}
        recorded = {}
        for sid, cid, outcome in editor.profile_coverage:
            if sid in row_ids:
                counts = coverage.setdefault(cid, {})
                counts[outcome] = counts.get(outcome, 0) + 1
                recorded.setdefault(cid, set()).add(sid)
        profile_scope_count = (sum(rid in row_ids for rid in sample_scope["row_ids"])
                               if sample_scope is not None else len(row_ids))
        absent = {cid: profile_scope_count - len(recorded.get(cid, ()))
                  for cid, column in columns.items()
                  if profile["enabled"] and role(column) in PROFILE_UNITS
                  and not derived(column)}
        editor._scope_cache = (scope_key, len(row_ids), scoped, levels, profile_scope_count, coverage, absent)
    _, row_count, scoped, levels, profile_scope_count, coverage, absent = editor._scope_cache
    has_rows = bool(row_count)
    editor.coverage_button.setEnabled(editor._checked_revision is not None and has_rows)
    if profile["enabled"]:
        checked = sum(counts.get("checked", 0) for counts in coverage.values())
        total = sum(sum(counts.values()) for counts in coverage.values())
        profile_text = tr('Записи профиля: проверено {0}, пропущено {1}.', checked, total - checked)
        if missing := sum(absent.values()):
            profile_text += "\n" + tr('Пар проба–показатель без записи: {0}.', missing)
        if sample_scope is not None:
            profile_text += "\n" + tr('Область профиля: {0}. Проб в текущей области: {1} из {2}.',
                                      sample_scope["name"], profile_scope_count, len(sample_scope["row_ids"]))
        editor.profile_status.setText(profile_text)
    labels = editor.results_model.samples
    header = editor.table.horizontalHeader()
    sort_column = header.sortIndicatorSection()
    descending = header.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
    sort_key = (scope_key, sort_column, descending, editor._decision_revision if sort_column in (5, 6) else None)
    excluded_only = editor.exclusion_filter.currentData()
    render_key = (sort_key, editor._decision_revision, excluded_only, severity, state, rule,
                  property_query, sample_query, source_query)
    if editor._render_key == render_key:
        return
    first_render = editor._render_key is None
    editor._render_key = render_key
    source_labels = source_location_labels() if source_query or sort_column == 7 else None
    rule_labels = ({key: tr(rule["description"]).casefold() for key, rule in RULES.items()}
                   if sort_column == 3 else {})

    def source_matches(issue):
        if not source_query:
            return True
        key = issue["id"]
        search_text = editor._source_search_cache.get(key)
        if search_text is None:
            source = issue["source"]
            search_text = (source_search_text(source).casefold(), source_location(source, source_labels).casefold())
            editor._source_search_cache[key] = search_text
        return any(source_query in text for text in search_text)

    keys = {
        0: lambda i: order[i["severity"]],
        1: lambda i: names[i["property_id"]],
        2: lambda i: labels.get(i["sample_id"], i["sample_id"]).casefold(),
        3: lambda i: rule_labels[i["rule_id"]],
        4: lambda i: i["message"].casefold(),
        5: lambda i: i["decision"] != "open",
        6: lambda i: i["comment"].casefold(),
        7: lambda i: source_location(i["source"], source_labels).casefold(),
    }
    if editor._sort_cache is None or editor._sort_cache[0] != sort_key:
        editor._sort_cache = (sort_key, sorted(scoped, key=keys[sort_column], reverse=descending))
    ordered = editor._sort_cache[1]
    exclusions = {m.id for _, m, _ in active_exclusions(editor.owner.model.project)} if excluded_only else None
    sample_matches = ({sid for sid, label in labels.items()
                       if sample_query in sid.casefold() or sample_query in label.casefold()}
                      if sample_query else None)
    if severity == state == rule == "all" and exclusions is None and not any(
            (property_query, sample_query, source_query)):
        visible = ordered
    else:
        visible = [i for i in ordered if (severity == "all" or i["severity"] == severity)
                   and (state == "all" or i["decision"] == state)
                   and (rule == "all" or i["rule_id"] == rule)
                   and (exclusions is None or i["measurement_id"] in exclusions)
                   and (not property_query or property_query in names[i["property_id"]])
                   and (sample_matches is None or i["sample_id"] in sample_matches)
                   and source_matches(i)]
    with QSignalBlocker(editor.table.selectionModel()):
        editor.results_model.replace(visible)
        selected_rows = [row for row, issue in enumerate(visible) if issue["id"] in selected_ids]
        if selected_rows:
            selection = QItemSelection()
            add_row_ranges(selection, editor.results_model, selected_rows, 0, 0)
            flags = QItemSelectionModel.SelectionFlag
            editor.table.selectionModel().select(selection, flags.ClearAndSelect | flags.Rows)
            focused = next((row for row in selected_rows if visible[row]["id"] == selected_id), selected_rows[0])
            index = editor.results_model.index(focused, 0)
            editor.table.selectionModel().setCurrentIndex(index, flags.NoUpdate)
            editor.table.scrollTo(index)
        elif first_render and visible and not selected_ids and selected_id is None and editor._draft_issue is None:
            editor.select_issue_row(0)
    editor.show_details()
    counts = tr('Показано: {0} · В области: {1} · В проекте: {2}',
                len(visible), len(scoped), len(editor.issues))
    review_key = (scope_key, editor._decision_revision)
    if editor._review_cache is None or editor._review_cache[0] != review_key:
        acknowledged = sum(issue["decision"] == "acknowledged" for issue in scoped)
        editor._review_cache = review_key, acknowledged
    acknowledged = editor._review_cache[1]
    review = tr('В области: к рассмотрению {0} · принято {1}', len(scoped) - acknowledged, acknowledged)
    if editor._checked_revision is None:
        editor.status.setText(tr('Проверка ещё не завершена.'))
        return
    elif not has_rows:
        detail = tr('В выбранной области нет проб.')
    elif not scoped:
        detail = tr('В выбранной области нет замечаний по выполненным проверкам.')
    elif not visible:
        detail = tr('По текущим фильтрам совпадений нет.')
    else:
        visible_levels = levels
        if visible is not ordered:
            visible_levels = dict.fromkeys(order, 0)
            if severity != "all":
                visible_levels[severity] = len(visible)
            else:
                for issue in visible:
                    visible_levels[issue["severity"]] += 1
        detail = tr('В выдаче: ошибок {0} · критических {1} · предупреждений {2}', *visible_levels.values())
    editor.status.setText(f"{counts}\n{review}\n{detail}")


def show_details(editor, *_):
    spans = editor.selected_spans()
    count = sum(bottom - top + 1 for top, bottom in spans)
    issue = editor.results_model.issues[spans[0][0]] if count == 1 else None
    issue_id = issue["id"] if issue else None
    editor.batch_button.setEnabled(count > 1)
    editor.batch_button.setText(tr('Групповое решение ({0})…', count) if count > 1 else tr('Групповое решение…'))
    if editor._draft_issue != issue_id:
        editor.switch_draft(editor.comment, editor._decision_drafts, editor._draft_issue, issue_id)
        editor._draft_issue = issue_id
    exclusion_key = (issue_id, editor.scope.currentData()) if issue_id else None
    if editor._draft_exclusion != exclusion_key:
        editor.switch_draft(editor.exclusion_comment, editor._exclusion_drafts, editor._draft_exclusion, exclusion_key)
        editor._draft_exclusion = exclusion_key
    editor.acknowledge.setEnabled(bool(issue))
    editor.reopen.setEnabled(bool(issue and issue["decision"] == "acknowledged"))
    editor.go.setEnabled(bool(issue))
    editor.source_button.setEnabled(bool(issue and any(value is not None for value in issue["source"].values())))
    editor.comment.setEnabled(bool(issue))
    project = editor.owner.model.project
    measurement = project.science.measurements.get(issue["measurement_id"]) if issue else None
    records = project.view.get("qc_exclusions", {})
    active = records.get(editor.scope.currentData(), {}).get(measurement.id) if measurement else None
    has_exclusion = bool(active and active.get("revision") == measurement.revision)
    can_exclude = bool(measurement and measurement.revision == issue["revision"])
    editor.measurement_button.setVisible(can_exclude)
    editor.exclude_button.setEnabled(can_exclude and not has_exclusion)
    editor.include_button.setEnabled(can_exclude and has_exclusion)
    editor.exclusion_comment.setEnabled(can_exclude and not has_exclusion)
    editor.scope.setEnabled(can_exclude)
    if not issue:
        editor.details.setPlainText(tr('Выбрано замечаний: {0}. Групповое решение применяется только к выбранным строкам.', count)
                                  if count > 1 else "")
        return
    if issue:
        rule = RULES[issue["rule_id"]]
        exclusions = []
        if measurement:
            current = editor.owner.model.project.science.current(issue["sample_id"], issue["property_id"])
            for scope, label in SCOPES.items():
                record = records.get(scope, {}).get(measurement.id)
                if record and record.get("revision") == measurement.revision:
                    exclusions.append(f"{tr(label)} — {record.get('comment', '')}".rstrip(" —"))
        if measurement:
            details = tr('{0} · версия {1}\nОбоснование: {2}\nИзмерение: {3} · версия {4}\n{5}; исключение действует при использовании этого результата.\nИсключено из: {6}', tr(rule['description']), rule['version'], tr(rule['source']), issue['measurement_id'], issue['revision'], tr('Рабочий результат') if current and current.id == measurement.id else tr('Не выбран рабочим'), '\n'.join(exclusions) or tr('нет исключений'))
        else:
            details = tr('Вычисляемый показатель · {0}', tr(rule["description"]))
        sample = editor.results_model.samples.get(issue["sample_id"], issue["sample_id"])
        intro = [tr('Проба: {0}', sample), message(issue['message'])]
        if measurement:
            raw = repr(measurement.raw_value) if isinstance(measurement.raw_value, str) else measurement.raw_value
            intro.append(tr('Исходное значение: {0} {1}', raw, measurement.unit).rstrip())
        location = source_location(issue["source"])
        if location:
            intro.append(location)
        elif not editor.source_button.isEnabled():
            intro.append(tr('Нет источника'))
        details = "\n".join((*intro, details))
        if issue.get("context_fields"):
            details += "\n" + context_issue_text(issue["context_fields"])
        if issue["comment"]:
            details += "\n" + tr('Комментарий к решению: {0}', issue["comment"])
        if issue["decision"] == "acknowledged" or issue.get("decision_at"):
            details += "\n" + tr('Время решения: {0}', exclusion_time(issue.get("decision_at")))
        peers = issue.get("peers", ())
        if peers:
            details += "\n\n" + tr('Связанные измерения')
            if issue["rule_id"] == "repeat":
                details += "\n" + (tr('Сравнение по общей части заявленных интервалов в совместимых единицах.')
                                   if issue.get("repeat_comparison") == "interval" else
                                   tr('Сравнение точных значений. Для группы нет сопоставимых интервалов неопределённости.'))
            elif issue["rule_id"] == "repeat_censored":
                details += "\n" + tr('Строгие границы исключаются. Для ND с числовым LOD верхняя граница включается.')
            context_labels = {
                "method": tr('Метод'), "laboratory": tr('Лаборатория'),
                "preparation": tr('Подготовка'), "fraction": tr('Фракция'),
                "heating_program": tr('Программа нагрева'),
            }
            for peer in peers:
                analysis = peer["analysis"]
                other = project.science.measurements.get(peer["id"])
                context = " · ".join(f"{label}: {analysis.get(key) or '—'}"
                                     for key, label in context_labels.items())
                details += f"\n{peer['raw']} {other.unit if other else ''} · {peer['id']}\n  {context}"
                if issue["rule_id"] in ("repeat", "repeat_censored") and peer.get("uncertainty") is not None:
                    details += "\n  " + tr('Неопределённость: {0} ({1})', peer["uncertainty"],
                                            peer["uncertainty_type"] or "—")
                    if peer.get("confidence_level") is not None:
                        details += " · " + tr('Уровень доверия: {0}', peer["confidence_level"])
                if peer.get("lod") is not None:
                    details += "\n  " + tr('LOD: {0}', peer["lod"])
        if "profile" in issue:
            profile = issue["profile"]
            amin, tmin, tmax, amax = profile["thresholds"]
            details += "\n" + tr('Типичный диапазон [{0:g}, {1:g}]. Абсолютный диапазон [{2:g}, {3:g}].',
                                  tmin, tmax, amin, amax)
            for violation in profile.get("violations", ()):
                kind = tr('Абсолютный порог') if violation["rule"] == "profile_absolute" else tr('Типичный порог')
                side = tr('нижний') if violation["direction"] == "below" else tr('верхний')
                state = tr('выход за порог') if violation["whole"] else tr('пересечение порога')
                details += "\n" + tr('{0}, {1} {2:g} [{3}] · {4}', kind, side,
                                      violation["threshold"], profile["unit"], state)
            provenance = profile.get("provenance", {})
            for key, label in (("source", tr('Источник порогов')), ("version", tr('Версия методики')),
                               ("applicability", tr('Область применимости'))):
                if provenance.get(key):
                    details += f"\n{label}: {provenance[key]}"
        editor.details.setPlainText(details)


def switch_draft(field, drafts, previous, current):
    if previous is not None:
        if field.text():
            drafts[previous] = field.text()
        else:
            drafts.pop(previous, None)
    field.setText(drafts.get(current, ""))
