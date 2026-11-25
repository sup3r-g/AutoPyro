"""Manage saved model runs, archived reviews and prediction applications."""

import json
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import QFileDialog, QTableWidgetItem
from autopyro.analysis.geochemistry_workspace import (
    MODEL_RUNS_KEY,
    PROFILE_REVIEWS_KEY,
    apply_prediction,
    archive_model_run,
    comparison_is_current,
    replay_model_run,
    install_model_run,
    verify_model_run_package,
    validate_model_run,
)
from autopyro.data_io.file_output import atomic_output
from ..i18n import tr, message
from ..tasks import run_with_progress
from ..widgets.documents import source_token
from .geochemistry_workspace_display import _number, _biodegradation_labels, _biodegradation_indicators, _observation_labels, _status


def save_model_run(workspace, report, result):
    project = workspace.owner.model.project
    record = archive_model_run(project, report, result)
    project.event('geochemistry_model_run', run_id=record['id'], format=result['format'])
    workspace.owner.view_changed()
    if workspace.passport_tabs.isTabVisible(workspace.model_archive_tab):
        workspace.refresh_model_archive()


def model_title(workspace, result):
    return {'ward': tr('Иерархическая кластеризация'), 'kmeans': tr('Кластеризация K-средних'),
            'dbscan': tr('Кластеризация DBSCAN'), 'well_holdout_ols': tr('Линейный прогноз A → B'),
            'oil_mixing': tr('Бинарное смешение нефтей')}.get(result.get('method'), 'PCA')


def show_model_archive(workspace):
    count = workspace.refresh_model_archive()
    workspace.passport_tabs.setTabVisible(workspace.model_archive_tab, True)
    workspace.passport_tabs.setCurrentWidget(workspace.model_archive_page)
    workspace.status.setText(tr('Сохранено запусков: {0}.', count))


def show_review_archive(workspace):
    workspace.refresh_review_archive()
    workspace.passport_tabs.setTabVisible(workspace.review_archive_tab, True)
    workspace.passport_tabs.setCurrentWidget(workspace.review_archive_page)


def refresh_review_archive(workspace):
    records = workspace.owner.model.project.view.get(PROFILE_REVIEWS_KEY, {})
    ordered = sorted(records.items(), key=lambda item: item[1]['created_at'], reverse=True)
    with QSignalBlocker(workspace.review_archive):
        workspace.review_archive.setRowCount(len(ordered))
        for row, (digest, record) in enumerate(ordered):
            for col, value in enumerate((record['created_at'], *record['request']['names'])):
                cell = QTableWidgetItem(str(value))
                cell.setData(Qt.ItemDataRole.UserRole, digest)
                workspace.review_archive.setItem(row, col, cell)
        if ordered:
            workspace.review_archive.selectRow(0)
    workspace.review_archive.resizeColumnsToContents()
    workspace.display_archived_review()


def selected_archived_review(workspace):
    rows = workspace.review_archive.selectionModel().selectedRows()
    if rows:
        digest = workspace.review_archive.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        return workspace.owner.model.project.view.get(PROFILE_REVIEWS_KEY, {}).get(digest)
    return None


def display_archived_review(workspace):
    record = workspace.selected_archived_review()
    workspace.export_review_button.setEnabled(record is not None)
    if record is None:
        workspace.archived_review_text.setPlainText(tr('Сохранённых заключений нет.'))
        return
    match = record['match']
    snapshot = record.get('snapshot') or {
        'version': 1, 'input_digest': match['input_digest'], 'request': record['request'],
    }
    state = (tr('Входы совпадают с текущим проектом.') if comparison_is_current(workspace.owner.model.project, snapshot)
             else tr('Входы изменились. Показано архивное заключение.'))
    labels = _biodegradation_labels()
    biodegradation = record.get('biodegradation', {})
    overlap = match['overlap_percent']
    details = [state, (tr('Перекрытие профилей: {0}%.', _number(overlap)) if overlap is not None
                       else tr('Сравнение показателей без расчёта перекрытия профилей.')),
               tr('Признаки биодеградации в A') + ' — ' + labels.get(biodegradation.get('a'), labels['not_assessed']),
               tr('Признаки биодеградации в B') + ' — ' + labels.get(biodegradation.get('b'), labels['not_assessed']),
               '', record['note']]
    if match.get('correlation', {}).get('isotope_basis'):
        details.insert(2, tr('Подтверждены фракции C15+ и шкала VPDB.'))
    indicators, observations = _biodegradation_indicators(), _observation_labels()
    saved_observations = record.get('biodegradation_observations', [])
    if saved_observations:
        details.extend(('', tr('Наблюдения биодеградации')))
    for item in saved_observations:
        label = indicators.get(item['indicator'], item['indicator'])
        status = observations.get(item['status'], item['status'])
        source = f" · {item['source']}" if item['source'] else ''
        details.append(f"{item['side'].upper()} · {label} · {status}{source}")
    evidence_notes = record.get('evidence_notes', [])
    if evidence_notes:
        details.extend(('', tr('Экспертные пометки к показателям')))
    fields = {item['field']['id']: item['field']['name'] for item in match.get('evidence', [])}
    for item in evidence_notes:
        label = indicators.get(item['indicator'], item['indicator'])
        name = fields.get(item['field_id'], item['field_id'])
        details.append(f"{name} · {item['side'].upper()} · {label} · {item['reason']}")
    workspace.archived_review_text.setPlainText('\n'.join(details))


def export_archived_review(workspace):
    record = workspace.selected_archived_review()
    if record is None:
        return
    path, _ = QFileDialog.getSaveFileName(workspace, tr('Экспорт заключения JSON…'),
                                         'geochemistry-expert-review.json', 'JSON (*.json)')
    if path:
        def write():
            with atomic_output(path) as temporary:
                temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        workspace.attempt(write)


def refresh_model_archive(workspace):
    archive = workspace.owner.model.project.view.get(MODEL_RUNS_KEY, {})
    runs = archive.get('runs', [])
    with QSignalBlocker(workspace.model_runs):
        workspace.model_runs.setRowCount(len(runs))
        for row, record in enumerate(reversed(runs)):
            result = record['result']
            values = (record['created_at'], workspace.model_title(result), workspace.preprocessing_label(result),
                      len(result.get('scores', result.get('membership', result.get('predictions',
                                                                                  result.get('sample_ids', []))))))
            for col, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setData(Qt.ItemDataRole.UserRole, record['id'])
                cell.setToolTip(record['id'] + '\n' + str(record['engine']))
                workspace.model_runs.setItem(row, col, cell)
        if runs:
            workspace.model_runs.selectRow(0)
    workspace.model_runs.resizeColumnsToContents()
    workspace.display_model_passport()
    return len(runs)


def preprocessing_label(workspace, result):
    if result.get('method') == 'oil_mixing':
        return tr('Массовый баланс')
    return {'clr': 'CLR', 'train_standardized': tr('Стандартизация только на A')}.get(
        result['preprocessing'], tr('Стандартизация'))


def display_model_passport(workspace):
    if not workspace.model_runs.selectionModel().selectedRows():
        workspace.model_passport.setPlainText(tr('Выберите запуск модели в архиве.'))
        return
    try:
        record, snapshot = workspace.selected_model_run()
    except (ValueError, KeyError) as exc:
        workspace.model_passport.setPlainText(message(exc))
        return
    result = record['result']
    method = result.get('method', 'pca')
    limitations = {
        'pca': tr('PCA описывает вариацию. Нужны полные пробы и два независимых направления вариации.'),
        'ward': tr('Метод Уорда строит иерархию по евклидовым расстояниям. Состав групп зависит от выбранного числа кластеров.'),
        'kmeans': tr('K-средних группирует пробы по расстоянию до центра. Результат зависит от числа кластеров и инициализации.'),
        'dbscan': tr('DBSCAN группирует пробы по плотности. Радиус и минимальный размер окрестности влияют на долю шума.'),
        'well_holdout_ols': tr('Линейный прогноз обучается на A и проверяется на других скважинах B. Прогнозы за пределами диапазона обучения отмечаются в таблице.'),
        'oil_mixing': tr('Оценка условна для выбранных конечных компонентов, концентраций и допусков.'),
    }
    fields = {field['id']: field for field in snapshot['inputs']['fields']}
    predictors = result.get('predictors', result.get('fields', list(result.get('tolerances', {}))))
    field_names = [f"{fields[key]['name']} [{fields[key]['unit']}]" if fields[key]['unit'] else fields[key]['name']
                   for key in predictors]
    lines = [workspace.model_title(result), tr('Версия расчёта: {0}.', result['version']),
             tr('Подготовка: {0}.', workspace.preprocessing_label(result)), tr('Входы: {0}.', ', '.join(field_names))]
    if method == 'well_holdout_ols':
        lines.append(tr('Целевой показатель: {0}.', fields[result['target']]['name']))
    if method in ('ward', 'kmeans'):
        lines.append(tr('Число кластеров: {0}.', result['cluster_count']))
    if method == 'kmeans':
        lines.append(tr('Начальное число генератора: {0}.', result['seed']))
    if method == 'dbscan':
        lines.append(tr('Радиус: {0}. Минимум проб в окрестности: {1}.', _number(result['radius']), result['min_samples']))
    if result.get('preprocessing') == 'clr':
        lines.append(tr('CLR применяется к частям одного состава. Неположительные доли исключают пробу целиком.'))
    if 'excluded' in result:
        lines.append(tr('Исключено проб: {0}.', len(result['excluded'])))
    lines.extend(('', limitations.get(method, ''),
                  '', ', '.join(f'{library} {version}' for library, version in record['engine'].items())))
    workspace.model_passport.setPlainText('\n'.join(lines))


def selected_model_run(workspace):
    rows = workspace.model_runs.selectionModel().selectedRows()
    if not rows:
        raise ValueError(tr('Выберите запуск модели в архиве.'))
    run_id = workspace.model_runs.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
    archive = workspace.owner.model.project.view[MODEL_RUNS_KEY]
    record = next((item for item in archive['runs'] if item['id'] == run_id), None)
    if record is None:
        raise ValueError(tr('Выберите запуск модели в архиве.'))
    snapshot = archive['snapshots'][record['snapshot_id']]
    validate_model_run(record, snapshot)
    return record, snapshot


def apply_archived_prediction(workspace):
    def apply():
        record, snapshot = workspace.selected_model_run()
        project = workspace.owner.model.project
        sample_ids = workspace.selected_samples()
        context = source_token(workspace.owner)
        result = run_with_progress(workspace, tr('Применение сохранённой модели…'),
                                   lambda: apply_prediction(project, record, snapshot, sample_ids))
        if project is not workspace.owner.model.project or context != source_token(workspace.owner):
            raise ValueError(tr('Данные изменились во время расчёта. Повторите сравнение.'))
        fields = {field['id']: field['name'] for field in result['predictors']}
        rows = result['results']
        workspace.application_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            outside = ', '.join(fields[field] for field in item['outside_train_range'])
            reasons = [f'{fields[field]}: {_status(status)}' for field, status in item['excluded'].items()]
            if item['status'] == 'material_mismatch':
                reasons.append(tr('Состояние материала вне обучающей выборки'))
            elif item['status'] == 'non_finite':
                reasons.append(_status('non_finite'))
            values = (item['name'], _number(item['predicted']), result['target']['unit'], outside, '; '.join(reasons))
            for col, value in enumerate(values):
                workspace.application_table.setItem(row, col, QTableWidgetItem(value))
        workspace.application_table.resizeColumnsToContents()
        workspace.prediction_application = result
        workspace.passport_tabs.setTabVisible(workspace.application_tab, True)
        workspace.passport_tabs.setCurrentWidget(workspace.application_page)
        workspace.status.setText(tr('Цель: {0}. Прогноз получен для {1} из {2} проб.', result['target']['name'],
                               sum(item['status'] == 'numeric' for item in rows), len(rows)))
    workspace.attempt(apply)


def export_prediction_application(workspace):
    result = workspace.prediction_application
    if result is None:
        return
    path, _ = QFileDialog.getSaveFileName(workspace, tr('Экспорт прогнозов JSON…'),
                                         'geochemistry-predictions.json', 'JSON (*.json)')
    if path:
        def write():
            with atomic_output(path) as temporary:
                temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        workspace.attempt(write)


def open_model_run(workspace, *_):
    def show():
        record, snapshot = workspace.selected_model_run()
        run = record['result']
        if run['format'] == 'autopyro.comparison_pca':
            workspace.render_pca(snapshot, run)
            tab = workspace.pca_tab
        elif run['format'] == 'autopyro.comparison_prediction':
            workspace.render_prediction(snapshot, run)
            tab = workspace.prediction_tab
        elif run['format'] == 'autopyro.comparison_oil_mixing':
            workspace.render_oil_mixing(snapshot, run)
            tab = workspace.oil_mixing_tab
        else:
            workspace.render_clustering(snapshot, run, workspace.model_title(run))
            tab = workspace.clustering_tab
        workspace.passport_tabs.setTabText(tab, tr('Архив: {0}', workspace.passport_tabs.tabText(tab)))
        state = (tr('Входы совпадают с текущим проектом.') if comparison_is_current(workspace.owner.model.project, snapshot)
                 else tr('Входы изменились. Показан сохранённый расчёт.'))
        workspace.status.setText(tr('Запуск от {0}.', record['created_at']) + ' ' + state + '\n' + workspace.status.text())
    workspace.attempt(show)


def check_model_run(workspace):
    def check():
        record, snapshot = workspace.selected_model_run()
        matches = run_with_progress(workspace, tr('Повторный расчёт по архивному снимку…'),
                                    lambda: replay_model_run(record, snapshot))
        workspace.status.setText(tr('Архивный запуск воспроизведён.') if matches else
                            tr('Повторный расчёт отличается от сохранённого результата.'))
    workspace.attempt(check)


def export_model_run(workspace):
    def write():
        record, snapshot = workspace.selected_model_run()
        path, _ = QFileDialog.getSaveFileName(workspace, tr('Экспорт запуска JSON…'), 'geochemistry-model-run.json', 'JSON (*.json)')
        if path:
            with atomic_output(path) as temporary:
                temporary.write_text(json.dumps({'run': record, 'snapshot': snapshot}, ensure_ascii=False,
                                                indent=2, allow_nan=False), encoding='utf-8')
    workspace.attempt(write)


def import_model_run(workspace):
    path, _ = QFileDialog.getOpenFileName(workspace, tr('Импорт запуска JSON…'), '', 'JSON (*.json)')
    if not path:
        return
    project = workspace.owner.model.project

    def read():
        with open(path, encoding='utf-8-sig') as stream:
            return verify_model_run_package(json.load(stream))

    def install():
        record, snapshot = run_with_progress(workspace, tr('Проверка импортируемого запуска…'), read)
        if project is not workspace.owner.model.project:
            raise ValueError(tr('Проект изменился во время импорта запуска.'))
        if not install_model_run(project, record, snapshot):
            workspace.status.setText(tr('Этот запуск уже есть в архиве.'))
            return
        project.event('geochemistry_model_run_imported', run_id=record['id'])
        workspace.owner.view_changed()
        workspace.show_model_archive()
        workspace.status.setText(tr('Запуск проверен и добавлен в архив.'))

    workspace.attempt(install)
