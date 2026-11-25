"""Export geochemistry passports, comparisons and correlation results."""

import csv
import json
from PySide6.QtWidgets import QFileDialog
from autopyro.analysis.geochemistry_workspace import (
    PROFILE_REVIEWS_KEY,
    GeochemistryIndex,
    comparison_is_current,
    freeze_comparison,
    settings,
)
from autopyro.data_io.file_output import atomic_output
from ..i18n import tr
from ..tasks import run_with_progress
from .geochemistry_workspace_display import _status


def export_passport(workspace):
    ids = workspace.selected_samples()
    if not ids:
        return
    project = workspace.owner.model.project
    path, _ = QFileDialog.getSaveFileName(workspace, tr('Экспорт паспорта'), 'geochemistry-passport.json', 'JSON (*.json)')
    if path:
        def write():
            model = workspace.owner.model
            if model.project is not project:
                raise ValueError(tr('Данные изменились. Пересчитайте сравнение перед экспортом.'))
            context = (model, model.data_revision, settings(project))
            index = workspace.index if workspace._index_context == context else None

            def prepare():
                return (index or GeochemistryIndex(project)).passport(ids[0])

            if index is None and (len(project.rows) >= 10000 or len(project.science.measurements) >= 100000):
                passport = run_with_progress(workspace, tr('Подготовка измерений…'), prepare)
            else:
                passport = prepare()
            current = workspace.owner.model
            if current.project is not project or context != (current, current.data_revision, settings(project)):
                raise ValueError(tr('Данные изменились. Пересчитайте сравнение перед экспортом.'))
            with atomic_output(path) as temporary:
                temporary.write_text(json.dumps(passport, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        workspace.attempt(write)


def export_correlation(workspace):
    if (workspace.profile_match is None or 'correlation' not in workspace.profile_match or workspace.report is None
            or not comparison_is_current(workspace.owner.model.project, workspace.report)):
        workspace.status.setText(tr('Сначала выполните актуальную корреляцию.'))
        return
    report, match = workspace.report, workspace.profile_match
    path, _ = QFileDialog.getSaveFileName(workspace, tr('Экспорт корреляции'), 'geochemistry-correlation.json',
                                         tr('Протокол корреляции JSON (*.json);;Таблица показателей CSV (*.csv)'))
    if not path:
        return
    if not workspace.check_comparison_export(report, match=match):
        return
    review = workspace.owner.model.project.view.get(PROFILE_REVIEWS_KEY, {}).get(match['review_key'])

    def write():
        with atomic_output(path) as temporary:
            if path.lower().endswith('.csv'):
                with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
                    writer = csv.writer(stream)
                    writer.writerow([tr('Показатель'), tr('Единица'), 'A', tr('Статус A'),
                                     'B', tr('Статус B'), tr('Разность B−A'),
                                     tr('Проба A'), tr('Проба B'), tr('Перекрытие н-алканов, %')])
                    context = (*report['request']['names'], match['overlap_percent'])
                    for item in match['evidence']:
                        a, b = item['a'], item['b']
                        writer.writerow([item['field']['name'], item['field']['unit'], a['value'],
                                         _status(a['status']), b['value'], _status(b['status']), item['difference'],
                                         *context])
                    if not match['evidence']:
                        writer.writerow([None] * 7 + list(context))
            else:
                temporary.write_text(json.dumps({'match': match, 'expert_review': review,
                                                'snapshot': freeze_comparison(report)},
                                               ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    workspace.attempt(write)


def export_comparison(workspace):
    if workspace.report is None or not comparison_is_current(workspace.owner.model.project, workspace.report):
        workspace.invalidate_comparison()
        workspace.status.setText(tr('Данные изменились. Пересчитайте сравнение перед экспортом.'))
        return
    source_report = workspace.report
    path, _ = QFileDialog.getSaveFileName(workspace, tr('Экспорт сравнения'), 'geochemistry-comparison.json',
                                         tr('Полный протокол JSON (*.json);;Таблица групп CSV (*.csv)'))
    if not path or not workspace.check_comparison_export(source_report):
        return
    def write():
        with atomic_output(path) as temporary:
            if path.lower().endswith('.csv'):
                with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
                    writer = csv.writer(stream)
                    writer.writerow([workspace.comparison_table.horizontalHeaderItem(i).text() for i in range(workspace.comparison_table.columnCount())])
                    for row in range(workspace.comparison_table.rowCount()):
                        writer.writerow([workspace.comparison_table.item(row, col).text() for col in range(workspace.comparison_table.columnCount())])
            else:
                report = dict(workspace.report)
                if workspace.pca_run is not None:
                    report['pca'] = workspace.pca_run
                if workspace.clustering_run is not None:
                    report['clustering'] = workspace.clustering_run
                if workspace.kmeans_run is not None:
                    report['kmeans'] = workspace.kmeans_run
                if workspace.dbscan_run is not None:
                    report['dbscan'] = workspace.dbscan_run
                if workspace.prediction_run is not None:
                    report['prediction'] = workspace.prediction_run
                if workspace.profile_match is not None:
                    report['alkane_profile_match'] = workspace.profile_match
                    review_key = workspace.profile_match.get('review_key', workspace.profile_match['input_digest'])
                    review = workspace.owner.model.project.view.get(PROFILE_REVIEWS_KEY, {}).get(review_key)
                    if review is not None:
                        report['expert_review'] = review
                temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    workspace.attempt(write)


def check_comparison_export(workspace, report, *, match=None):
    if (workspace.report is report and (match is None or workspace.profile_match is match)
            and comparison_is_current(workspace.owner.model.project, report)):
        return True
    workspace.status.setText(tr('Данные изменились. Пересчитайте сравнение перед экспортом.'))
    return False
