"""Explain and review interpretation decisions, including summaries and revision comparisons."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import QTimer, Qt, QSignalBlocker
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QTextBrowser,
)

from autopyro.schemes.interpretation import STATUSES
from autopyro.schemes.interpretation_review import (
    review_changes,
    review_history,
    review_reasons,
    review_state,
    save_review,
)
from autopyro.schemes.interpretation_revisions import comparison_base

from ..i18n import tr, message, settings
from ..theme import icon
from .interpretation_results import AGREEMENT_LABELS, RESULT_FILTERS, REVIEW_LABELS, REVIEW_QUEUE_STATES


REVIEW_REASONS = {
    'disagreement': 'Расхождение оценок HI–Tmax и HI–OI',
    'ambiguous': 'Неоднозначный результат на одной или нескольких диаграммах',
    'boundary': 'Номинальное измерение на границе области или порога',
    'quality': 'Предупреждения контроля качества',
}


REVIEW_CHANGE_LABELS = {
    'scheme': 'правила схемы',
    'definition': 'определения параметров',
    'binding': 'привязка осей',
    'input': 'измерения или их неопределённость',
    'quality': 'результаты контроля качества',
    'result': 'автоматический результат',
}


def update_explanation(sheet, *args):
    sheet.review_button.setEnabled(sheet.report is not None and len(sheet.result_selection) == 1)
    sheet.pyrogram_button.setEnabled(sheet.review_button.isEnabled())
    sheet.summary_button.setEnabled(sheet.report is not None and bool(sheet.report['results']))
    sheet.update_comparison_actions()
    if sheet.report is None:
        sheet.explanation.setPlainText(tr('Результат не актуален. Ожидается расчёт интерпретации.'))
        return
    if not sheet.result_selection:
        count = sheet.quality_count
        sheet.explanation.setPlainText(tr('Выберите пробу на графике или в таблице результатов. Проб с предупреждениями: {0}.', count))
        return
    from autopyro.schemes.interpretation_explanation import explain_sample
    current = sheet.table.currentIndex().row()
    positions = {sheet.result_rows[rid] for rid in sheet.result_selection}
    visible_position = current if current in positions else min(positions)
    position = sheet.visible_positions[visible_position]
    prefix = tr('Выделено проб: {0}. Показано объяснение текущей пробы.\n\n', len(positions)) if len(positions) > 1 else ''
    model = sheet.table.model()
    label = model.index(visible_position, 0).data()
    state, decision = review_state(sheet.owner.model.project, sheet.report, position)
    highlights = []
    if state:
        highlights.append(tr('Экспертная проверка — {0}', tr(REVIEW_LABELS[state])))
    if state == 'stale':
        changes = review_changes(sheet.report, position, decision)
        for diagram, fields in changes:
            labels = ', '.join(tr(REVIEW_CHANGE_LABELS[field]) for field in fields)
            highlights.append(tr('Причина повторной проверки — {0}: {1}', diagram, labels))
        if not changes:
            highlights.append(tr('В сохранённом решении нет детализации изменившихся данных.'))
    reasons = review_reasons(sheet.report, position)
    if reasons:
        highlights.append(tr('Причины проверки — {0}', ', '.join(tr(REVIEW_REASONS[reason]) for reason in reasons)))
    if decision and decision['conclusion']:
        title = 'Предыдущий экспертный вывод — {0}' if state == 'stale' else 'Экспертный вывод — {0}'
        highlights.append(tr(title, decision['conclusion']))
    if sheet.comparison_panels:
        agreement = tr(AGREEMENT_LABELS[sheet.report['type_agreement'][position]])
        highlights.append(tr('Согласие стандартных HI–Tmax и HI–OI — {0}', agreement))
    for mode in sheet.active_filters():
        if not mode.startswith('review') and mode != 'disagreement':
            diagrams = [name for name, report in sheet.filter_diagrams()
                        if sheet.matches_filter(report, position, mode)]
            if diagrams:
                highlights.append(tr('Условие «{0}» — {1}', tr(RESULT_FILTERS[mode]),
                                     ', '.join(dict.fromkeys(diagrams))))
    text = prefix + explain_sample(sheet.report, position, sample_label=label, highlights=highlights)
    if sheet.comparison_panels:
        text += '\n\n' + tr('Сравнение диаграмм для этой пробы:')
        for panel in sheet.comparison_panels:
            companion = sheet.comparison_reports.get(panel['name'])
            if companion is None:
                text += '\n' + panel['name'] + ': ' + panel['error']
                continue
            item = companion['results'][position]
            text += '\n' + panel['name'] + ': ' + tr(STATUSES[item['status']])
            labels = sheet.comparison_labels[panel['name']]
            names = item['regions'] + item['boundary_regions'] + (item.get('possible_types') or item['line_hits'])
            text += '; ' + '; '.join(labels.get(key, key) for key in names)
            if item.get('reason'):
                text += '\n' + item['reason']
            for issue in companion['inputs'][position].get('quality', []):
                text += '\n' + issue['message']
    if decision:
        text += '\n'
        if state == 'stale':
            text += '\n' + tr('Предыдущее решение — {0}', tr(REVIEW_LABELS[decision['status']]))
        if decision['note']:
            text += '\n' + tr('Комментарий — {0}', decision['note'])
        text += '\n' + tr('Автор решения — {0}', decision.get('author') or tr('Не указан'))
    sample = sheet.owner.model.project.science.samples.get(sheet.report['results'][position]['row_id'])
    if sample is not None:
        observation = sample.metadata.get('pyrogram_observation')
        if observation:
            text += '\n\n' + tr('Наблюдение по пирограммам — {0}', observation)
    sheet.explanation.setPlainText(text)


def show_pyrograms(sheet):
    from ..widgets.pyrograms import PyrogramDialog

    if sheet.report is None or len(sheet.result_selection) != 1:
        return
    project = sheet.owner.model.project
    sample_id = next(iter(sheet.result_selection))
    if sample_id not in project.science.samples:
        return
    dialog = PyrogramDialog(sheet, project, sample_id, undo_stack=sheet.owner.model.undo,
                            notify=sheet.owner.pyrograms_changed)
    dialog.exec()
    dialog.deleteLater()
    sheet.update_explanation()


def edit_review(sheet):
    report = sheet.report
    if report is None or len(sheet.result_selection) != 1:
        return
    row_id = next(iter(sheet.result_selection))
    position = sheet.visible_positions[sheet.result_rows[row_id]]
    project = sheet.owner.model.project
    _, decision = review_state(project, report, position)
    decision = decision or {}

    dialog = QDialog(sheet)
    dialog.setWindowTitle(tr('Экспертная проверка пробы'))
    dialog.resize(480, 300)
    form = QFormLayout(dialog)
    status = QComboBox()
    for label, value in [
        (tr('Оставить вопрос открытым'), 'open'),
        (tr('Принять автоматическую оценку'), 'accepted'),
        (tr('Записать экспертный вывод'), 'expert'),
    ]:
        status.addItem(label, value)
    status.setCurrentIndex(status.findData(decision.get('status', 'open')))
    conclusion = QLineEdit(decision.get('conclusion', ''))
    conclusion.setObjectName('expert_conclusion')
    note = QPlainTextEdit(decision.get('note', ''))
    author = QLineEdit(project.view.get('interpretation_review_author', ''))
    author.setObjectName('review_author')
    form.addRow(tr('Решение'), status)
    form.addRow(tr('Экспертный вывод'), conclusion)
    form.addRow(tr('Комментарий / обоснование'), note)
    form.addRow(tr('Автор решения'), author)
    entries = review_history(project, report['scheme']['id'], row_id)
    if entries:
        history = QTextBrowser(dialog)
        history.setAccessibleName(tr('История экспертных решений'))
        records = []
        for entry in reversed(entries):
            heading = ' · '.join((entry.get('updated_at', '—'), entry.get('author') or tr('Не указан'),
                                  tr(REVIEW_LABELS[entry['status']])))
            revision = entry.get('scheme_revision')
            if revision:
                heading += ' · ' + tr('Редакция {0}', revision)
            records.append('\n'.join(filter(None, (heading, entry.get('conclusion'), entry.get('note')))))
        history.setPlainText('\n\n'.join(records))
        form.addRow(tr('История решений'), history)
        dialog.resize(540, 470)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
    form.addRow(buttons)
    next_probe = None
    if any(mode.startswith('review') for mode in sheet.active_filters()):
        next_probe = next((
            report['results'][i]['row_id']
            for i in sheet.visible_positions[sheet.result_rows[row_id] + 1:]
            if sheet.review_states[i] in REVIEW_QUEUE_STATES
        ), None)
    next_button = (buttons.addButton(tr('Сохранить и открыть следующую пробу'),
                                    QDialogButtonBox.ButtonRole.ActionRole) if next_probe else None)
    open_next = False

    def save_and_continue():
        nonlocal open_next
        open_next = True
        dialog.accept()

    def validate():
        expert = status.currentData() == 'expert'
        conclusion.setEnabled(expert)
        valid = bool(author.text().strip()) and (
            not expert or bool(conclusion.text().strip() and note.toPlainText().strip()))
        buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(valid)
        if next_button is not None:
            next_button.setEnabled(valid)

    if next_button is not None:
        next_button.clicked.connect(save_and_continue)
    status.currentIndexChanged.connect(validate)
    conclusion.textChanged.connect(validate)
    note.textChanged.connect(validate)
    author.textChanged.connect(validate)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    validate()
    accepted = dialog.exec()
    values = status.currentData(), conclusion.text(), note.toPlainText()
    reviewer = author.text().strip()
    dialog.deleteLater()
    if not accepted:
        return
    if sheet.report is not report or sheet.owner.model.project is not project:
        QMessageBox.warning(sheet, tr('Экспертная проверка пробы'),
                            tr('Результаты изменились. Откройте пробу для повторной проверки.'))
        return
    previous_state = sheet.review_states[position]
    if not reviewer:
        return
    decision = save_review(project, report, position, *values, author=reviewer)
    project.view['interpretation_review_author'] = reviewer
    state = decision['status']
    sheet.review_states[position] = state
    if previous_state in REVIEW_QUEUE_STATES:
        sheet.review_counts[previous_state] -= 1
    if state in REVIEW_QUEUE_STATES:
        sheet.review_counts[state] += 1
    sheet.update_review_queue_count()
    sheet.report_table_rows[position][-1] = tr(REVIEW_LABELS[state])
    sheet.owner.view_changed()
    if any(mode.startswith('review') and
           sheet.matches_review_filter(previous_state, mode) != sheet.matches_review_filter(state, mode)
           for mode in sheet.active_filters()):
        old_position = sheet.result_rows.get(row_id, 0)
        sheet.filter_results()
        if sheet.result_rows:
            next_position = min(old_position, len(sheet.result_rows) - 1)
            sheet.select_result_rows([sheet.table.model().rows[next_position][0]])
    else:
        model = sheet.table.model()
        if row_id in sheet.result_rows:
            index = model.index(sheet.result_rows[row_id], model.columnCount() - 1)
            model.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole])
        sheet.update_explanation()
    if open_next and next_probe in sheet.result_rows:
        sheet.select_result_rows([next_probe])
        QTimer.singleShot(0, sheet.edit_review)


def show_summary(sheet):
    from autopyro.geology.stratigraphy import validate_binding as validate_stratigraphy_binding
    from ..widgets.interpretation_summary import InterpretationSummaryDialog, edit_summary_binding

    report = sheet.report
    if report is None:
        return
    project = sheet.owner.model.project
    settings = project.view.get('stratigraphy') or {}
    title = tr('Сводка по стволам и интервалам')
    scheme = settings.get('scheme') or project.view.get('stratigraphy_scheme')
    if not scheme:
        QMessageBox.information(sheet, title, tr('Сохраните схему интервалов в модуле «Стратиграфия».'))
        return
    saved = project.view.get('interpretation_summary_bindings', {}).get(scheme['id'])
    initial = saved or settings.get('binding') or {}
    binding = initial
    try:
        validate_stratigraphy_binding(project, scheme, binding)
    except (KeyError, TypeError, ValueError):
        binding = edit_summary_binding(sheet, project, scheme, initial)
        if binding is None or sheet.owner.model.project is not project or sheet.report is not report:
            return
    try:
        dialog = InterpretationSummaryDialog(sheet, project, report, scheme, binding)
    except (KeyError, TypeError, ValueError) as exc:
        QMessageBox.warning(sheet, title, message(exc))
        return

    def select_in_interpretation(ids):
        if sheet.report is report and sheet.owner.model.project is project:
            if any(row_id not in sheet.result_rows for row_id in ids):
                sheet.clear_result_filters()
            sheet.select_result_rows(ids)

    dialog.profile_selected.connect(lambda row_id: select_in_interpretation([row_id]))
    if len(sheet.result_selection) == 1:
        dialog.show_profile_row(next(iter(sheet.result_selection)))
    accepted = dialog.exec()
    ids = dialog.selected_ids
    if sheet.owner.model.project is project and dialog.binding != initial:
        project.view.setdefault('interpretation_summary_bindings', {})[scheme['id']] = dialog.binding
        sheet.owner.view_changed()
    dialog.deleteLater()
    if accepted:
        select_in_interpretation(ids)


def update_review_queue_count(sheet):
    counts = sheet.review_counts
    sheet.review_queue_count.setText(tr(
        'Экспертная очередь · новые: {0} · открытые: {1} · повторные: {2}',
        counts['pending'], counts['open'], counts['stale'],
    ))
    sheet.review_queue_count.show()


def update_comparison_actions(sheet):
    view = sheet.owner.model.project.view
    pinned = view.get('interpretation_comparison_base')
    with QSignalBlocker(sheet.pin_base):
        sheet.pin_base.setChecked(bool(pinned))
    sheet.pin_base.setEnabled(sheet.report is not None or bool(pinned))
    sheet.pin_base.setText(tr('Открепить базу сравнения') if pinned else tr('Закрепить текущий расчёт как базу'))
    sheet.pin_base.setIcon(icon('x' if pinned else 'save'))
    sheet.pin_base.setToolTip(
        tr('Закреплена схема {0}, редакция {1}, проб: {2}.', pinned['scheme']['name'],
           pinned['scheme']['revision'], len(pinned['results'])) if pinned else
        tr('Сохранить снимок текущего расчёта для сравнения после фильтрации и пересчёта.'))
    sheet.revisions_button.setEnabled(sheet.report is not None and bool(comparison_base(view)))


def set_comparison_base(sheet, checked):
    view = sheet.owner.model.project.view
    if checked and sheet.report is not None:
        view['interpretation_comparison_base'] = deepcopy(sheet.report)
    elif not checked and 'interpretation_comparison_base' in view:
        del view['interpretation_comparison_base']
    else:
        sheet.update_comparison_actions()
        return
    sheet.owner.view_changed()
    sheet.update_comparison_actions()


def show_revisions(sheet):
    from ..widgets.interpretation_revisions import InterpretationRevisionsDialog

    report = sheet.report
    project = sheet.owner.model.project
    previous = comparison_base(project.view)
    if report is None or not previous:
        return
    dialog = InterpretationRevisionsDialog(sheet, project, previous, report,
                                           pinned=bool(project.view.get('interpretation_comparison_base')))
    accepted = dialog.exec()
    row_id = dialog.row_id
    dialog.deleteLater()
    if accepted and row_id and sheet.report is report and sheet.owner.model.project is project:
        sheet.clear_result_filters()
        sheet.select_result_rows([row_id])
