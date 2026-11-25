"""Present project and selection status in the main window."""

from PySide6.QtWidgets import QLabel

from .i18n import tr


def install(window):
    window.selection_status = QLabel()
    window.statusBar().addPermanentWidget(window.selection_status)


def update_status(window):
    stale_columns = [c for c in window.model.project.columns if c.metadata.get("stale")]
    stale = sum(bool(c.metadata.get("method")) for c in stale_columns)
    stale_fit = sum(bool(c.metadata.get("curve_fit")) for c in stale_columns)
    window.setWindowTitle(
        f"AutoPyro · {window.path.name if window.path else tr('Новый проект')}{' *' if window.dirty else ''}"
    )
    snapshot = window.model.project.view.get("database_snapshot")
    checked = tr('   Снимок проверен: {0} UTC', snapshot['checked_at'][:19].replace('T', ' ')) if snapshot else ""
    scope = tr('   Набор проб: {0}', window.active_sample_set) if window.active_sample_set else ""
    if window.active_sample_set and window.active_sample_set not in window.sample_sets.saved_sets:
        scope += "   " + tr('Сохранённый набор проб отсутствует.')
    if window.table_filter_sort.missing_filter_column:
        scope += "   " + tr('Колонка сохранённого фильтра удалена; обновите представление')
    fit_status = "   " + tr('Устаревших колонок подгонки: {0}', stale_fit) if stale_fit else ""
    window.statusBar().showMessage(
        tr(
            'Строк: {0:,} / {1:,}   Колонок: {2}   Условия: {3}   Устаревших интерполяций: {4}{5}',
            window.proxy.rowCount(), window.model.rowCount(), window.model.columnCount(),
            window.table_filter_sort.condition_count(), stale, checked,
        ) + fit_status + scope
    )
    window.close_sample_set_action.setEnabled(window.active_sample_set is not None)
    window.desktop.schedule_refresh()


def update_selection_status(window):
    visible, hidden = window.sample_selection.counts()
    total = visible + hidden
    window.clear_selection_action.setEnabled(bool(total))
    window.selection_status.setVisible(bool(total))
    window.selection_status.setText(tr('Выбрано проб: {0:,} · скрыто: {1:,}', total, hidden))
