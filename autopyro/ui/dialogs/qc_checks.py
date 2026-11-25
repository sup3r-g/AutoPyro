"""Run QC checks and manage the pyrolysis threshold profile and coverage."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from autopyro.analysis.qc import (
    PROFILE_KEY,
    PROFILE_UNITS,
    ProfileThresholdError,
    profile_thresholds,
    pyrolysis_profile,
    role,
    run_checks,
    snapshot_for_checks,
)

from ..i18n import tr, message
from ..table import QCCommand
from ..table.model import large_snapshot
from ..tasks import TaskCancelled, run_with_progress
from ..widgets.results import ResultsModel, sample_labels


def on_activated(editor):
    if not editor._check_error:
        editor.reload()


def refresh_checks(editor):
    if not editor._checking:
        editor._checked_revision = None
        editor._check_error = ""
        editor.reload()


def schedule(editor):
    if editor.isVisible() and not editor._check_error:
        editor.timer.start()


def edit_profile(editor):
    project = editor.owner.model.project
    profile = pyrolysis_profile(project)
    dialog = QDialog(editor)
    dialog.setWindowTitle(tr('Профиль порогов пиролиза'))
    dialog.resize(790, 520)
    layout = QVBoxLayout(dialog)
    enabled = QCheckBox(tr('Применять пороги в этом проекте'))
    enabled.setChecked(profile["enabled"])
    layout.addWidget(enabled)
    scope_form = QFormLayout()
    profile_scope = QComboBox()
    profile_scope.addItem(tr('Все пробы проекта'), None)
    saved_scope = profile.get("sample_scope")
    if saved_scope is not None:
        profile_scope.addItem(tr('Зафиксировано: {0} ({1})', saved_scope["name"], len(saved_scope["row_ids"])),
                              "saved")
        profile_scope.setCurrentIndex(1)
    sample_sets = project.view.get("sample_sets", {})
    for name, row_ids in sample_sets.items():
        profile_scope.addItem(tr('Набор проб: {0} ({1})', name, len(row_ids)), f"set:{name}")
    scope_form.addRow(tr('Применять к пробам'), profile_scope)
    layout.addLayout(scope_form)
    table = QTableWidget(len(PROFILE_UNITS), 4)
    table.setHorizontalHeaderLabels([
        tr('Абсолютный\nминимум'), tr('Типичный\nминимум'),
        tr('Типичный\nмаксимум'), tr('Абсолютный\nмаксимум'),
    ])
    table.setVerticalHeaderLabels([f"{name} [{unit}]" for name, unit in PROFILE_UNITS.items()])
    for row, name in enumerate(PROFILE_UNITS):
        for col, value in enumerate(profile["thresholds"][name]):
            table.setItem(row, col, QTableWidgetItem(str(value)))
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    layout.addWidget(table)
    provenance = QFormLayout()
    fields = {}
    for key, label in (("source", tr('Источник порогов')), ("version", tr('Версия методики')),
                       ("applicability", tr('Описание применимости'))):
        fields[key] = QLineEdit(profile[key])
        provenance.addRow(label, fields[key])
    layout.addLayout(provenance)
    applicability = QLabel(tr('Выбор набора фиксирует его состав. Для обновления состава выберите набор заново. Основные проверки QC действуют во всей области данных.'))
    applicability.setWordWrap(True)
    layout.addWidget(applicability)
    status = QLabel()
    status.setWordWrap(True)
    layout.addWidget(status)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    layout.addWidget(buttons)

    def save():
        thresholds = {}
        for row, name in enumerate(PROFILE_UNITS):
            try:
                thresholds[name] = profile_thresholds(table.item(row, col).text() for col in range(4))
            except ProfileThresholdError as exc:
                status.setText(tr('{0}: введите четыре конечных порога в порядке возрастания', name))
                table.setCurrentCell(row, exc.column)
                table.setFocus()
                return
        new = {"enabled": enabled.isChecked(), "thresholds": thresholds,
               **{key: field.text().strip() for key, field in fields.items()}}
        selection = profile_scope.currentData()
        if selection is not None:
            if selection == "saved":
                new["sample_scope"] = saved_scope
            else:
                name = selection[4:]
                existing = project.row_map()
                row_ids = list(dict.fromkeys(rid for rid in sample_sets.get(name, ()) if rid in existing))
                if not row_ids:
                    status.setText(tr('В выбранном наборе нет доступных проб.'))
                    profile_scope.setFocus()
                    return
                new["sample_scope"] = {"name": name, "row_ids": row_ids}
        if new != {**profile, "thresholds": {key: list(value) for key, value in profile["thresholds"].items()}}:
            command = QCCommand(editor.owner.model, PROFILE_KEY, new)
            command.setText(tr('Изменить профиль порогов пиролиза'))
            editor.owner.model.undo.push(command)
        dialog.accept()

    buttons.accepted.connect(save)
    buttons.rejected.connect(dialog.reject)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        editor.reload()


def reload(editor):
    if editor._checking:
        return
    project = editor.owner.model.project
    revision = editor.owner.model.data_revision, pyrolysis_profile(project)
    if editor._checked_revision != revision:
        def check(source, checkpoint=None):
            coverage = []
            issues = run_checks(source, coverage_details=coverage,
                                checkpoint=checkpoint, include_peers=True)
            return issues, coverage, sample_labels(source)

        editor._checking = True
        editor._check_error = ""
        editor.timer.stop()
        try:
            if large_snapshot(project) or len(project.science.measurements) >= 10_000:
                def check_snapshot(update):
                    stage = tr("Подготовка снимка данных…")
                    update(stage, 0)
                    snapshot = snapshot_for_checks(project, lambda: update(stage, 0))
                    return check(snapshot, lambda stage, percent: update(tr(stage), percent))

                issues, coverage, labels = run_with_progress(
                    editor, tr("Проверка качества данных…"), check_snapshot, cancellable=True)
            else:
                issues, coverage, labels = check(project)
        except (TaskCancelled, ValueError) as exc:
            editor._check_error = (tr('Проверка отменена. Обновите результаты для продолжения.')
                                 if isinstance(exc, TaskCancelled) else message(exc))
            editor.render()
            return
        finally:
            editor._checking = False
        if (project is not editor.owner.model.project
                or revision != (editor.owner.model.data_revision, pyrolysis_profile(project))):
            editor.schedule()
            return
        editor.issues = issues
        editor._source_search_cache.clear()
        editor.profile_coverage = coverage
        editor.results_model.samples = labels
        editor._checked_revision = revision
        editor._decision_revision = None
    if editor._decision_revision != editor.owner.analysis_revision:
        decisions = editor.owner.model.project.view.get("qc_decisions", {})
        for issue in editor.issues:
            decision = decisions.get(issue["id"], {})
            issue["decision"] = decision.get("action", "open")
            issue["comment"] = decision.get("comment", "")
            if decision.get("at"):
                issue["decision_at"] = decision["at"]
            else:
                issue.pop("decision_at", None)
        editor._decision_revision = editor.owner.analysis_revision
    editor.render()


def show_coverage(editor):
    project = editor.owner.model.project
    if editor._checked_revision != (editor.owner.model.data_revision, pyrolysis_profile(project)):
        editor.reload()
    if not editor.coverage_button.isEnabled():
        return
    profile = pyrolysis_profile(project)
    _, row_count, _, _, profile_scope_count, coverage, absent = editor._scope_cache
    headers = ['Показатель', 'Роль для QC', 'Пороговый профиль']
    if profile["enabled"]:
        headers += ['Проверено', 'Причины пропуска', 'Проб без записи']
    headers.append('Базовые проверки')
    reasons = {
        "missing": tr('Пустые записи'),
        "unit": tr('Неподдерживаемая единица'),
        "range": tr('Недопустимый предел'),
        "no_bounds": tr('Нет числа или диапазона'),
        "excluded": tr('Исключено из расчёта'),
    }
    rows = []
    for column in project.columns:
        formula = column.metadata.get("calculation", {}).get("formula")
        name = formula if formula in ("HI", "OI") else role(column)
        if not name and column.kind != "number":
            continue
        counts = coverage.get(column.id, {})
        if name not in PROFILE_UNITS:
            applicability = tr('Не предусмотрен')
        elif not profile["enabled"]:
            applicability = tr('Выключен')
        else:
            applicability = tr('Есть записи') if counts else tr('Нет записей')
        row = [column.name, name or tr('Не распознана'), applicability]
        if profile["enabled"]:
            skipped = "\n".join(f"{label}: {counts[key]}" for key, label in reasons.items() if counts.get(key))
            row.extend((counts.get("checked", 0), skipped or "—", absent.get(column.id, "—")))
        if formula:
            row.append(tr('Контекст расчёта'))
        elif role(column):
            row.append(tr('Формат, единицы, диапазон, контекст и повторы'))
        else:
            row.append(tr('Формат записи'))
        rows.append(row)
    dialog = QDialog(editor)
    dialog.setWindowTitle(tr('Охват проверок'))
    dialog.resize(860, 420)
    layout = QVBoxLayout(dialog)
    note_text = tr('Проб в области: {0}. Охват показан до фильтров списка замечаний.', row_count)
    if profile["enabled"]:
        note_text += "\n" + tr('Проб в области профиля: {0}. Проверено и пропущено считают записи и диагностики HI/OI, включая повторы.', profile_scope_count)
    note = QLabel(note_text)
    note.setWordWrap(True)
    layout.addWidget(note)
    table = QTableView()
    table.setModel(ResultsModel(headers, rows, table))
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    if profile["enabled"]:
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    layout.addWidget(table)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.button(QDialogButtonBox.StandardButton.Close).setText(tr('Закрыть'))
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()
