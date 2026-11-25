"""Prepare, install, filter and select interpretation results."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from PySide6.QtCore import Qt, QItemSelection, QItemSelectionModel, QSignalBlocker

from autopyro.schemes.interpretation import STATUSES
from autopyro.schemes.interpretation_review import (
    is_boundary_result,
    migrate_review_evidence,
    review_state,
)
from autopyro.schemes.interpretation_thresholds import threshold_labels

from ..i18n import tr, message, settings
from ..widgets.results import SampleResultsModel, sample_label_steps


AGREEMENT_LABELS = {
    'equal': 'Совпадают наборы типов',
    'partial': 'Частичное совпадение',
    'different': 'Нет общих типов',
    'unavailable': 'Сравнение недоступно',
}


REVIEW_LABELS = {
    'pending': 'Требует проверки',
    'accepted': 'Автоматическая оценка принята',
    'expert': 'Сохранён экспертный вывод',
    'open': 'Вопрос открыт',
    'stale': 'Данные изменились · требуется повторная проверка',
}


REVIEW_QUEUE_STATES = {'pending', 'open', 'stale'}


RESULT_FILTERS = {
    'review': 'Очередь экспертной проверки',
    'review_pending': 'Ожидают экспертной проверки',
    'review_open': 'Вопрос оставлен открытым',
    'review_stale': 'Требуют повторной проверки',
    'ambiguous': 'Неоднозначные пробы',
    'boundary': 'На границе области или порога',
    'quality': 'С предупреждениями контроля качества',
    'outside': 'Вне областей схемы',
    'not_applicable': 'Вне области применимости',
    'disagreement': 'Расхождения стандартных HI–Tmax и HI–OI',
}


def initialize(sheet):
    sheet.report = None
    sheet.review_states = []
    sheet.review_counts = Counter()
    sheet.previous_stale_notice = ""
    sheet.result_selection = set()
    sheet.result_rows = {}
    sheet.report_table_rows = []
    sheet.visible_positions = ()
    sheet.quality_count = 0
    sheet.comparison_panels = []
    sheet.comparison_reports = {}
    sheet.comparison_labels = {}


def clear_result_view(sheet):
    sheet.previous_result_button.hide()
    sheet.report = None
    sheet.result_rows = {}
    sheet.report_table_rows = []
    sheet.report_positions = {}
    sheet.visible_positions = ()
    model = sheet.table.model()
    if model is not None and model.rowCount():
        # Keep UUID selection for a successful refresh, independent of Qt's model reset.
        with QSignalBlocker(sheet.table.selectionModel()):
            model.set_rows([])
    sheet.result_filter.setEnabled(False)
    sheet.extra_filters.setEnabled(False)
    sheet.comparison_context.hide()
    sheet.filtered_count.clear()
    sheet.review_queue_count.hide()
    sheet.update_explanation()
    sheet.export_json.setEnabled(False)
    sheet.export_csv.setEnabled(False)
    sheet.update_plot()


def select_result_rows(sheet, row_ids, *, scroll=True):
    """Select report UUIDs without changing the source-table calculation scope."""
    sheet.result_selection = set(row_ids).intersection(sheet.result_rows)
    selection_model = sheet.table.selectionModel()
    if selection_model is not None:
        positions = sorted(sheet.result_rows[rid] for rid in sheet.result_selection)
        selection = QItemSelection()
        model = sheet.table.model()
        # Coalesce contiguous rows into ranges, keeping large selections cheap.
        start = end = None
        for position in positions:
            if start is None:
                start = end = position
            elif position == end + 1:
                end = position
            else:
                selection.select(model.index(start, 0), model.index(end, 0))
                start = end = position
        if start is not None:
            selection.select(model.index(start, 0), model.index(end, 0))
        blocker = QSignalBlocker(selection_model)
        flags = QItemSelectionModel.SelectionFlag
        selection_model.select(selection, flags.ClearAndSelect | flags.Rows)
        if positions:
            index = model.index(positions[0], 0)
            selection_model.setCurrentIndex(index, flags.NoUpdate)
            if scroll:
                sheet.table.scrollTo(index)
        del blocker
    sheet.highlight_result_selection()


def results_selection_changed(sheet, *args):
    model = sheet.table.model()
    sheet.result_selection = {
        model.rows[row][0]
        for area in sheet.table.selectionModel().selection()
        for row in range(area.top(), area.bottom() + 1)
    }
    sheet.highlight_result_selection()


def highlight_result_selection(sheet):
    sheet.update_explanation()
    if hasattr(sheet, 'plot') and sheet.plot.selection_overlay is not None:
        sheet.plot.selection_overlay.set_selected(sheet.result_selection)


def comparison_cells(sheet, position):
    return sheet.format_comparison_cells(position, sheet.comparison_panels,
                                        sheet.comparison_reports, sheet.comparison_labels)


def format_comparison_cells(position, panels, reports, labels_by_panel):
    cells = []
    for panel in panels:
        report = reports.get(panel['name'])
        if report is None:
            cells.append(panel['error'])
            continue
        item = report['results'][position]
        labels = labels_by_panel[panel['name']]
        names = item['regions'] + item['boundary_regions'] + (item.get('possible_types') or item['line_hits'])
        cells.append(tr(STATUSES[item['status']]) + '; ' + '; '.join(labels.get(key, key) for key in names))
    return cells


def sync_agreement_filter(sheet):
    index = sheet.result_filter.findData('disagreement')
    with QSignalBlocker(sheet.result_filter):
        if sheet.comparison_panels and index < 0:
            sheet.result_filter.addItem(tr('Расхождения стандартных HI–Tmax и HI–OI'), 'disagreement')
        elif not sheet.comparison_panels and index >= 0:
            if sheet.result_filter.currentIndex() == index:
                sheet.result_filter.setCurrentIndex(0)
            sheet.result_filter.removeItem(index)
    extra = sheet.extra_filter_actions['disagreement']
    extra.setVisible(bool(sheet.comparison_panels))
    if not sheet.comparison_panels:
        extra.setChecked(False)
    choice = sheet.comparison_view.currentData()
    with QSignalBlocker(sheet.comparison_view):
        sheet.comparison_view.clear()
        sheet.comparison_view.addItem(tr('Предпросмотр: автоматически'), 'auto')
        sheet.comparison_view.addItem(tr('Предпросмотр: все диаграммы'), 'all')
        for panel in sheet.comparison_panels:
            sheet.comparison_view.addItem(panel['name'], panel['name'])
        sheet.comparison_view.setCurrentIndex(max(0, sheet.comparison_view.findData(choice)))
    sheet.comparison_view.setVisible(bool(sheet.comparison_panels))


def prepare_report_steps(sheet, project, report):
    scheme = report['scheme']
    panels, comparisons = report['comparison_panels'], report['comparisons']
    panel_labels = {
        name: {entry['id']: entry['label'] for field in ('regions', 'lines')
               for entry in diagram['scheme'].get(field, ())}
        for name, diagram in comparisons.items()
    }
    labels = {entry['id']: entry['label'] for entry in scheme.get('regions', ())}
    lines = {entry['id']: entry['label'] for entry in scheme.get('lines', ())}
    show_hio = scheme.get('strategy') == 'initial_hi'
    view = project.view
    decisions = view.get('interpretation_reviews', {}).get(scheme['id'], {})
    legacy = {key: dict(value) for key, value in decisions.items() if value.get('evidence_version') != 2}
    review_project = project
    if legacy:
        # A cancelled preparation must not migrate decisions in the live project.
        review_view = dict(view, interpretation_reviews={scheme['id']: {**decisions, **legacy}})
        review_project = SimpleNamespace(view=review_view)
        migrate_review_evidence(review_project, view.get('interpretation_result'),
                                view.get('interpretation_previous_result'),
                                view.get('interpretation_comparison_base'), report)
    rows, review_states, counts, positions = [], [], Counter(), {}
    review_counts = Counter()
    quality_count = 0
    for position, (item, sample) in enumerate(zip(report['results'], report['inputs'], strict=True)):
        if position and position % 256 == 0:
            yield
        line_ids = set(item.get('possible_types') or item['line_hits'])
        names = [label for key, label in lines.items()
                 if key in line_ids and key not in item.get('thresholds', {})]
        names.extend(threshold_labels(scheme, item.get('thresholds', {})))
        issues = sample.get('quality', ())
        row = [
            item['row_id'], tr(STATUSES[item['status']]),
            ', '.join(labels[key] for key in item['regions']),
            ', '.join(item['boundary_regions']),
            '; '.join(names), message(item['reason']),
            '\n'.join(issue['message'] for issue in issues),
            '; '.join(labels.get(key, key) for key in item['uncertainty_regions']),
        ]
        row.extend(sheet.format_comparison_cells(position, panels, comparisons, panel_labels))
        if panels:
            row.append(tr(AGREEMENT_LABELS[report['type_agreement'][position]]))
        if show_hio:
            bounds = item.get('value_bounds')
            row.append(' – '.join(f'{bound:.6g}' for bound in bounds) if bounds is not None else '—')
        state, _ = review_state(review_project, report, position)
        review_states.append(state)
        if state in REVIEW_QUEUE_STATES:
            review_counts[state] += 1
        row.append(tr(REVIEW_LABELS[state]) if state else '')
        rows.append(row)
        positions[item['row_id']] = position
        counts[item['status']] += 1
        quality_count += bool(issues)
    names = yield from sample_label_steps(project, (row[0] for row in rows))
    migrated = {key: value for key, value in legacy.items() if value.get('evidence_version') == 2}
    return SimpleNamespace(rows=rows, review_states=review_states, counts=counts, names=names,
                           quality_count=quality_count, labels=panel_labels, migrated=migrated,
                           positions=positions, review_counts=review_counts)


def install_report(sheet, report, prepared, saved, predecessor):
    sheet.cancel_calculation_button.hide()
    try:
        sheet.comparison_panels = report['comparison_panels']
        sheet.comparison_reports = report['comparisons']
        sheet.comparison_labels = prepared.labels
        project = sheet.owner.model.project
        if prepared.migrated:
            project.view['interpretation_reviews'][report['scheme']['id']].update(prepared.migrated)
        sheet.sync_agreement_filter()
        sheet.report = report
        sheet.quality_count = prepared.quality_count
        sheet.review_states = prepared.review_states
        sheet.review_counts = prepared.review_counts
        sheet.update_review_queue_count()
        show_hio = sheet.scheme.get('strategy') == 'initial_hi'
        rows = prepared.rows
        sheet.report_table_rows = rows
        sheet.report_positions = prepared.positions
        old = sheet.table.model()
        old_selection = sheet.table.selectionModel()
        if old_selection is not None:
            old_selection.selectionChanged.disconnect(sheet.results_selection_changed)
            old_selection.currentRowChanged.disconnect(sheet.update_explanation)
        sheet.table.setModel(
            SampleResultsModel(
                [
                    tr('Проба'),
                    tr('Статус · {0}', sheet.scheme['name']) if sheet.comparison_panels else tr('Статус'),
                    tr('Области'),
                    tr('Границы'),
                    tr('Линии / пороги') if any('threshold' in e for e in sheet.scheme.get('lines', []))
                    else tr('Кривые / возможные типы'),
                    tr('Обоснование'),
                    tr('Контроль качества'),
                    tr('Области интервала'),
                ] + [panel['name'] for panel in sheet.comparison_panels]
                + ([tr('Согласие стандартных HI–Tmax и HI–OI')] if sheet.comparison_panels else [])
                + ([tr('Диапазон HIo')] if show_hio else [])
                + [tr('Экспертная проверка')],
                [],
                prepared.names,
                sheet.table,
            )
        )
        sheet.table.selectionModel().selectionChanged.connect(sheet.results_selection_changed)
        sheet.table.selectionModel().currentRowChanged.connect(sheet.update_explanation)
        sheet.update_result_columns()
        if old_selection is not None:
            old_selection.deleteLater()
        if old:
            old.deleteLater()
        sheet.apply_result_filter()
        sheet.result_filter.setEnabled(True)
        sheet.extra_filters.setEnabled(True)
        sheet.table.horizontalHeader().setResizeContentsPrecision(100)
        sheet.table.resizeColumnsToContents()
        header = sheet.table.horizontalHeader()
        review_column = sheet.table.model().columnCount() - 1
        curve = sheet.scheme.get('strategy') in ('nearest_curve', 'initial_hi')
        has_thresholds = any('threshold' in entry for entry in sheet.scheme.get('lines', ()))
        result_column = 4 if curve or has_thresholds else 2
        for visual, logical in enumerate((0, 1, result_column, review_column)):
            header.moveSection(header.visualIndex(logical), visual)
        if show_hio:
            header.moveSection(header.visualIndex(review_column - 1), 2)
        for column in range(header.count()):
            limit = 170 if column == 0 else 210
            sheet.table.setColumnWidth(column, min(limit, sheet.table.columnWidth(column)))
        counts = prepared.counts
        sheet.status.setText(
            tr('Проб: {0} · ', len(rows))
            + "; ".join(
                f"{tr(STATUSES[key])}: {count}" for key, count in counts.items()
            )
            + tr(' · Проб с предупреждениями: {0}', sheet.quality_count)
            + tr('\nГеометрическая принадлежность не является вероятностью или самостоятельным геологическим заключением.')
        )
        settings = {
            **report["binding"],
            "scope": sheet.scope.currentData(),
            "scheme_id": sheet.scheme["id"],
        }
        view = sheet.owner.model.project.view
        comparison = sheet.comparison.isChecked()
        changed = (saved is not None or bool(prepared.migrated)
                   or view.get('interpretation_current') != settings
                   or view.get('interpretation_comparison', False) != comparison)
        if changed:
            view['interpretation_current'] = settings
            view['interpretation_comparison'] = comparison
        if predecessor is not None:
            view['interpretation_previous_result'] = predecessor
        if saved is not None:
            view['interpretation_result'] = saved
        sheet.update_comparison_actions()
        if sheet.previous_stale_notice:
            sheet.stale_notice.setText(tr('{0} Результаты пересчитаны.', sheet.previous_stale_notice))
        if changed:
            sheet.owner.view_changed()
        sheet.export_json.setEnabled(True)
        sheet.export_csv.setEnabled(True)
        sheet.update_plot()
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        sheet.calculation_error(exc)


def update_result_columns(sheet):
    if sheet.table.model() is not None:
        compact = sheet.compact_results.isChecked()
        for column in (3, 5, 6, 7):
            sheet.table.setColumnHidden(column, compact)


def active_filters(sheet):
    main = sheet.result_filter.currentData()
    extra = tuple(mode for mode, action in sheet.extra_filter_actions.items() if action.isChecked())
    if main not in ('all', None) and main not in extra:
        return (main, *extra)
    return extra


def clear_result_filters(sheet):
    with QSignalBlocker(sheet.result_filter):
        sheet.result_filter.setCurrentIndex(0)
    for action in sheet.extra_filter_actions.values():
        action.setChecked(False)
    sheet.filter_results()


def filter_diagrams(sheet):
    return ((sheet.report['scheme']['name'], sheet.report), *sheet.comparison_reports.items())


def matches_filter(report, position, mode):
    if mode == 'quality':
        return bool(report['inputs'][position].get('quality'))
    item = report['results'][position]
    if mode == 'boundary':
        return is_boundary_result(item)
    return item['status'] == mode


def matches_review_filter(state, mode):
    return state in REVIEW_QUEUE_STATES if mode == 'review' else state == mode.removeprefix('review_')


def apply_result_filter(sheet):
    modes = sheet.active_filters()
    rows = sheet.report_table_rows
    if not modes:
        sheet.visible_positions = range(len(rows))
        visible_rows = rows
    else:
        candidates = range(len(rows))
        diagrams = sheet.filter_diagrams()
        for mode in modes:
            if mode.startswith('review'):
                candidates = [i for i in candidates if sheet.matches_review_filter(sheet.review_states[i], mode)]
            elif mode == 'disagreement':
                agreement = sheet.report['type_agreement']
                candidates = [i for i in candidates if agreement[i] in ('partial', 'different')]
            else:
                candidates = [i for i in candidates if any(
                    sheet.matches_filter(report, i, mode) for _, report in diagrams)]
        sheet.visible_positions = candidates
        visible_rows = [rows[i] for i in sheet.visible_positions]

    header = sheet.table.horizontalHeader()
    column = header.sortIndicatorSection()
    if 0 <= column < sheet.table.model().columnCount() and sheet.visible_positions:
        labels = sheet.table.model().labels
        sheet.visible_positions = sorted(
            sheet.visible_positions,
            key=lambda i: str(labels.get(rows[i][0], rows[i][0]) if column == 0 else rows[i][column]).casefold(),
            reverse=header.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder,
        )
        visible_rows = [rows[i] for i in sheet.visible_positions]
    sheet.result_rows = (sheet.report_positions if not modes and column < 0
                        else {row[0]: i for i, row in enumerate(visible_rows)})
    with QSignalBlocker(sheet.table.selectionModel()):
        sheet.table.model().set_rows(visible_rows)
    sheet.select_result_rows(sheet.result_selection, scroll=False)
    extra_count = sum(action.isChecked() for action in sheet.extra_filter_actions.values())
    title = tr('Дополнительные условия')
    sheet.extra_filters.setText(f'{title} ({extra_count})' if extra_count else title)
    sheet.filtered_count.setText(tr('Показано проб: {0} из {1}', len(visible_rows), len(rows)))


def filter_results(sheet, *_):
    if sheet.report is not None:
        sheet.apply_result_filter()
        sheet.update_plot()
