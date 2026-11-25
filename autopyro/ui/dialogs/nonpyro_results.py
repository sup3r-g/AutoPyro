"""Track report freshness and apply or select calculated results."""

from __future__ import annotations

from PySide6.QtCore import Qt

from autopyro.analysis.nonpyro import check_saved_reports, result_columns

from ..geochemistry import status_label
from ..i18n import tr, message
from ..table import AddColumnCommand
from ..widgets.documents import source_token
from ..widgets.results import ResultsModel


class ProfileResultsModel(ResultsModel):
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return super().data(index, role)
        row = self.rows[index.row()]
        column = index.column()
        if column == 2:
            value = status_label(row[column])
        elif column == len(self.headers) - 2:
            value = message(row[column])
        elif column == len(self.headers) - 1:
            value = "; ".join(message(warning) for warning in row[column])
        else:
            value = super().data(index, role)
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"{value}\nUUID: {self.rows[index.row()][0]}"
        return value


def report_context(editor):
    return source_token(editor.owner), tuple(editor.row_ids())


def current_report(editor):
    if editor.report is not None and editor._report_context is not None:
        context = editor.report_context()
        if editor._report_context != context:
            check_saved_reports(editor.owner.model.project, [editor.report])
            if editor.report["stale"] or context[1] != editor._report_context[1]:
                editor.invalidate()
            else:
                editor._report_context = context
    return editor.report is not None


def invalidate(editor, *_):
    if editor.loading:
        return
    editor.report = None
    editor._report_context = None
    for canvas in (editor.alkane_canvas, editor.sara_canvas, editor.sara_profiles_canvas,
                   editor.homohopane_canvas, editor.chung_canvas):
        if canvas is not None:
            editor.result_tabs.setTabVisible(editor.result_tabs.indexOf(canvas), False)
    editor.status.setText(tr('Настройки или входы изменены. Выполните пересчёт.'))
    editor.update_workflow()


def update_result_actions(editor):
    ready = editor.report is not None
    editor.history_button.setEnabled(bool(editor.history_reports()))
    editor.calculate_button.setText(tr('Пересчитать') if ready else tr('Рассчитать'))
    if ready:
        editor.workflow_hint.setText(tr('Расчёт готов. Выберите действие с результатом ниже.'))
    for key, button in editor.result_actions.items():
        button.setEnabled(ready and (key != 'select' or editor.results.selectionModel().hasSelection()))
    alkane_method = editor.method.currentData() == 'alkane_indices'
    editor.alkane_profile_button.setVisible(alkane_method)
    editor.alkane_profile_button.setEnabled(alkane_method and bool(editor.selected_alkane_distribution()))
    sara_method = editor.method.currentData() == 'sara'
    sterane_method = editor.method.currentData() in ('sterane_distribution', 'rallakis_steranes')
    editor.sara_plot_button.setText(tr('Треугольная диаграмма стеранов') if sterane_method
                                  else tr('Треугольная диаграмма SARA'))
    editor.sara_plot_button.setToolTip(tr('Показывает доли трёх пиков ααα 20R.') if sterane_method
                                    else tr('Показывает пробы с полным составом SARA и балансом в пределах допуска.'))
    editor.sara_plot_button.setVisible(sara_method or sterane_method)
    editor.sara_plot_button.setEnabled(ready and (sara_method or sterane_method))
    editor.sara_profiles_button.setVisible(sara_method)
    editor.sara_profiles_button.setEnabled(ready and sara_method and editor.results.selectionModel().hasSelection())
    homohopanes = editor.method.currentData() == 'homohopane_distribution'
    editor.homohopane_button.setVisible(homohopanes)
    editor.homohopane_button.setEnabled(ready and homohopanes and editor.results.selectionModel().hasSelection())
    gas_isotopes = editor.method.currentData() == 'gas_isotopes'
    editor.chung_button.setVisible(gas_isotopes)
    editor.chung_button.setEnabled(ready and gas_isotopes and editor.results.selectionModel().hasSelection())


def promote(editor):
    if not editor.current_report():
        editor.status.setText(tr('Сначала выполните актуальный расчёт.'))
        return
    report = editor.report
    config = report["config"]
    model = editor.owner.model
    editor.promoting = True
    model.undo.beginMacro(tr('Результаты геохимической интерпретации'))
    try:
        for column in result_columns(config):
            if any(
                c.metadata.get("nonpyro") == column.metadata["nonpyro"]
                for c in model.project.columns
            ):
                continue
            names = {c.name for c in model.project.columns}
            base, index = column.name, 2
            while column.name in names:
                column.name = f"{base} ({index})"
                index += 1
            model.undo.push(AddColumnCommand(model, column, {}))
    finally:
        model.undo.endMacro()
        editor.promoting = False
    model.project.view.setdefault("nonpyro_results", {})[config["method"]] = dict(report)
    editor._report_context = editor.report_context()
    editor.update_result_actions()
    editor.status.setText(
        tr('Защищённые колонки добавлены; пересчитываются при редактировании входов. Доступна отмена.')
    )


def select_samples(editor):
    if not editor.current_report():
        return
    ids = {
        editor.report['results'][index.row()]['row_id']
        for index in editor.results.selectionModel().selectedRows()
    }
    selection = editor.owner.table.selectionModel()
    from PySide6.QtCore import QItemSelectionModel

    selection.clearSelection()
    for i, rid in enumerate(editor.owner.row_ids()):
        if rid in ids:
            selection.select(
                editor.owner.proxy.index(i, 0),
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows,
            )
    if editor.property("workspaceTracked"):
        editor.owner.desktop.open_table()
