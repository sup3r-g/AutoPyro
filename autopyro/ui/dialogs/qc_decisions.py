"""Apply individual and batch QC review decisions."""

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QLineEdit, QVBoxLayout

from autopyro.analysis.qc import pyrolysis_profile
from autopyro.model.core import timestamp

from ..i18n import tr
from ..table import QCCommand
from ..table.commands import QCBatchDecisionCommand


def decide(editor, action):
    issue = editor.action_issue()
    if not issue:
        return
    if action == "acknowledged" and not editor.comment.text().strip():
        editor.status.setText(tr('Укажите комментарий к решению.'))
        return
    previous_row = editor.selected_spans()[0][0]
    comment = editor.comment.text().strip() if action == "acknowledged" else ""
    editor.comment.clear()
    editor._decision_drafts.pop(issue["id"], None)
    editor.owner.model.undo.push(QCCommand(editor.owner.model, ("qc_decisions", issue["id"]),
        {"action": action, "comment": comment, "at": timestamp()}))
    editor.reload()
    if editor.current_issue() is None and editor.results_model.rowCount():
        next_row = min(previous_row, editor.results_model.rowCount() - 1)
        editor.select_issue_row(next_row)
        editor.table.scrollTo(editor.results_model.index(next_row, 0))


def batch_decision(editor):
    if editor.filter_timer.isActive():
        editor.filter_timer.stop()
        editor.render()
        editor.status.setText(tr('Фильтры изменились. Выберите замечания заново.'))
        return
    model = editor.owner.model
    project = model.project
    if editor._check_error or editor._checked_revision != (model.data_revision, pyrolysis_profile(project)):
        editor.reload()
        editor.status.setText(tr('Данные изменились. Проверьте замечание перед применением действия.'))
        return
    revision = editor.owner.analysis_revision, editor._render_key
    opened, accepted = [], []
    issues = editor.results_model.issues
    for top, bottom in editor.selected_spans():
        for row in range(top, bottom + 1):
            issue = issues[row]
            target = accepted if issue["decision"] == "acknowledged" else opened
            target.append(issue["id"])
    count = len(opened) + len(accepted)
    if count < 2:
        return

    dialog = QDialog(editor)
    dialog.setWindowTitle(tr('Групповое решение QC'))
    layout = QVBoxLayout(dialog)
    summary = QLabel(tr('Выбрано: {0}. К рассмотрению: {1}. Принято: {2}.', count, len(opened), len(accepted)))
    summary.setWordWrap(True)
    layout.addWidget(summary)
    comment = QLineEdit()
    comment.setPlaceholderText(tr('Общий комментарий для принятия'))
    comment.setAccessibleName(comment.placeholderText())
    layout.addWidget(comment)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
    buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('Отмена'))
    accept = buttons.addButton(tr('Принять ({0})', len(opened)), QDialogButtonBox.ButtonRole.ActionRole)
    reopen = buttons.addButton(tr('Вернуть ({0})', len(accepted)), QDialogButtonBox.ButtonRole.ActionRole)
    layout.addWidget(buttons)
    accept.setEnabled(False)
    reopen.setEnabled(bool(accepted))
    comment.textChanged.connect(lambda text: accept.setEnabled(bool(opened and text.strip())))
    result = {}

    def choose(action, ids):
        result.update(action=action, ids=ids)
        dialog.accept()

    accept.clicked.connect(lambda: choose("acknowledged", opened))
    reopen.clicked.connect(lambda: choose("open", accepted))
    buttons.rejected.connect(dialog.reject)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return
    if editor.filter_timer.isActive():
        editor.filter_timer.stop()
        editor.render()
        editor.status.setText(tr('Фильтры изменились. Выберите замечания заново.'))
        return
    if (model is not editor.owner.model or revision != (editor.owner.analysis_revision, editor._render_key)
            or editor._checked_revision != (model.data_revision, pyrolysis_profile(project))):
        editor.reload()
        editor.status.setText(tr('Данные изменились. Проверьте замечание перед применением действия.'))
        return
    action = result["action"]
    model.undo.push(QCBatchDecisionCommand(model, result["ids"], action,
                                          comment.text().strip() if action == "acknowledged" else ""))
    editor.reload()


def action_issue(editor):
    if editor.filter_timer.isActive():
        editor.filter_timer.stop()
        selected = editor.selected_issue()
        editor.render()
        if editor.selected_issue() is not selected:
            return None
    project = editor.owner.model.project
    if editor._checked_revision != (editor.owner.model.data_revision, pyrolysis_profile(project)):
        editor.reload()
        editor.status.setText(tr('Данные изменились. Проверьте замечание перед применением действия.'))
        return None
    return editor.selected_issue()
