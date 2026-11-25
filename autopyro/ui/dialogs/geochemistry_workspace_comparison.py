"""Compare selected samples and groups, including profile correlation and review."""

from copy import deepcopy
from textwrap import fill
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QTableWidgetItem,
    QVBoxLayout,
)
from autopyro.analysis.geochemistry_workspace import (
    PROFILE_REVIEWS_KEY,
    compare_alkane_profiles,
    compare_samples,
    correlate_oil_profiles,
    correlation_sensitivity,
    comparison_is_current,
    freeze_comparison,
    settings,
)
from autopyro.model.core import Column, timestamp
from autopyro.analysis.nonpyro import METHODS, field_matches, source_identity
from ..geochemistry import group_label
from ..i18n import tr, message
from ..tasks import run_with_progress
from ..widgets.documents import source_token
from ..widgets.tables import readonly_table
from .geochemistry_workspace_display import _number, _biodegradation_indicators, _status


def restore_saved_comparison(workspace):
    owner = workspace.owner
    saved = owner.model.project.view.get('geochemistry_comparison')
    if saved and workspace.index is not None and comparison_is_current(owner.model.project, saved):
        workspace._loading = True
        request = saved['request']
        for side, (combo, key) in enumerate(((workspace.a, 'group_a'), (workspace.b, 'group_b'))):
            sample_ids = request[key]
            data = 'sample:' + sample_ids[0] if len(sample_ids) == 1 else None
            if 'saved_groups' in request:
                if (name := request['saved_groups'][side]) is not None:
                    data = 'group:' + name
            elif data is None:
                for name, members in workspace.index.state.get('groups', {}).items():
                    if members == sample_ids:
                        data = 'group:' + name
                        break
            combo.setCurrentIndex(combo.findData(data))
        workspace.mixed.setChecked(request['allow_mixed_material'])
        for i in range(workspace.fields.count()):
            item = workspace.fields.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole)
                               in request['column_ids'] else Qt.CheckState.Unchecked)
        workspace._loading = False
        workspace.report = deepcopy(saved)
        workspace.display_comparison()
    elif saved:
        workspace.status.setText(tr('Данные изменились. Пересчитайте сравнение перед экспортом.'))


def invalidate_oil_correlation(workspace, *_):
    if workspace.profile_match is not None and 'correlation' in workspace.profile_match:
        workspace.profile_match = None
        workspace.export_correlation_button.setEnabled(False)
        workspace.passport_tabs.setTabVisible(workspace.profile_tab, False)
        workspace.status.setText(tr('Условия δ¹³C изменились. Повторите корреляцию.'))


def invalidate_comparison(workspace, *_):
    if workspace._loading:
        return
    workspace.correlation_readiness.clear()
    workspace.report = None
    workspace.profile_match = None
    workspace.export_correlation_button.setEnabled(False)
    workspace.prediction_run = None
    workspace.prediction_application = None
    workspace.passport_tabs.setTabVisible(workspace.application_tab, False)
    workspace.prediction_button.setEnabled(False)
    workspace.oil_mixing_button.setEnabled(False)
    workspace.passport_tabs.setTabVisible(workspace.oil_mixing_tab, False)
    workspace.prediction_cv.setEnabled(False)
    workspace.passport_tabs.setTabVisible(workspace.prediction_tab, False)
    workspace.profile_match_button.setEnabled(False)
    workspace.oil_correlation_button.setEnabled(False)
    workspace.correlation_isotope_basis.setChecked(False)
    workspace.correlation_isotope_basis.setEnabled(False)
    workspace.passport_tabs.setTabVisible(workspace.profile_tab, False)
    workspace.clear_multivariate_results()
    with QSignalBlocker(workspace.clr):
        workspace.clr.setChecked(False)
    workspace.clr.setEnabled(False)
    workspace.pca_button.setEnabled(False)
    workspace.clustering_button.setEnabled(False)
    workspace.kmeans_button.setEnabled(False)
    workspace.dbscan_button.setEnabled(False)
    workspace._report_context = None
    workspace.comparison_table.setRowCount(0)
    workspace.export_comparison_button.setEnabled(False)
    workspace.figure.clear()
    workspace.canvas.draw_idle()


def clear_multivariate_results(workspace, *_):
    workspace.pca_run = None
    workspace.clustering_run = None
    workspace.kmeans_run = None
    workspace.dbscan_run = None
    workspace.passport_tabs.setTabVisible(workspace.pca_tab, False)
    workspace.passport_tabs.setTabVisible(workspace.clustering_tab, False)


def preprocessing(workspace):
    return 'clr' if workspace.clr.isChecked() else 'standardized'


def change_preprocessing(workspace):
    workspace.clear_multivariate_results()
    workspace.status.setText(tr('Режим подготовки изменён. Запустите PCA или кластеризацию заново.'))


def resolve_group(workspace, combo):
    value = combo.currentData()
    if not value:
        raise ValueError(tr('Выберите пробы или группы сравнения'))
    kind, key = value.split(':', 1)
    ids = [key] if kind == 'sample' else workspace.index.state.get('groups', {}).get(key, [])
    missing = [sid for sid in ids if sid not in workspace.index.samples]
    if missing:
        raise ValueError(tr('В группе «{0}» отсутствуют пробы {1}. Сохраните группу заново с актуальными пробами.',
                            key, ', '.join(missing)))
    if not ids or not set(ids) <= workspace.available_ids:
        raise ValueError(tr('Группа содержит пробы вне текущей выборки. Измените область данных.'))
    return ids


def comparison_selection(workspace):
    fields = [workspace.fields.item(i).data(Qt.ItemDataRole.UserRole) for i in range(workspace.fields.count())
              if workspace.fields.item(i).checkState() == Qt.CheckState.Checked]
    return workspace.resolve_group(workspace.a), workspace.resolve_group(workspace.b), fields


def preview_oil_correlation(workspace):
    def preview():
        a, b, fields = workspace.comparison_selection()
        if len(a) != 1 or len(b) != 1:
            raise ValueError(tr('Выберите по одной пробе в A и B.'))
        project = workspace.owner.model.project
        context = source_token(workspace.owner), settings(project)
        mixed, names = workspace.mixed.isChecked(), (workspace.a.currentText(), workspace.b.currentText())
        confirmed = workspace.correlation_isotope_basis.isChecked()

        def check():
            report = compare_samples(project, a, b, fields, allow_mixed_material=mixed, names=names)
            return correlate_oil_profiles(report, confirm_c15_vpdb=confirmed)

        match = run_with_progress(workspace, tr('Проверка пары для корреляции…'), check)
        if workspace.owner.model.project is not project or context != (source_token(workspace.owner), settings(project)):
            raise ValueError(tr('Данные изменились во время расчёта. Повторите сравнение.'))
        workspace.show_correlation_coverage(match, names)
    workspace.attempt(preview)


def show_correlation_coverage(workspace, match, names):
    dialog = QDialog(workspace)
    dialog.setWindowTitle(tr('Готовность пары к корреляции'))
    layout = QVBoxLayout(dialog)
    paired = sum(item['a']['status'] == item['b']['status'] == 'numeric' for item in match['evidence'])
    if match['peaks']:
        overlap = tr('Перекрытие н-алканов доступно.')
    elif match.get('overlap_issue'):
        overlap = tr('Перекрытие н-алканов недоступно. {0}', message(match['overlap_issue']))
    else:
        overlap = tr('Пики н-алканов nC24–nC34 не выбраны.')
    lines = [
        tr('A: {0}. B: {1}.', *names),
        tr('Сопоставимые показатели в обеих пробах: {0} из {1}.', paired, len(match['evidence'])),
        overlap,
    ]
    for item in match['evidence']:
        lines.append(tr('{0} — A: {1}, B: {2}', item['field']['name'],
                        _status(item['a']['status']), _status(item['b']['status'])))
        if item.get('missing_peaks'):
            lines.append(tr('Недостающие пики: {0}', ', '.join(item['missing_peaks'])))
    details = QPlainTextEdit()
    details.setReadOnly(True)
    details.setPlainText('\n'.join(lines))
    layout.addWidget(details)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.resize(680, 420)
    dialog.exec()


def calculate_comparison(workspace):
    def calculate():
        a, b, fields = workspace.comparison_selection()
        project = workspace.owner.model.project
        context = source_token(workspace.owner), settings(project)
        mixed, names = workspace.mixed.isChecked(), (workspace.a.currentText(), workspace.b.currentText())
        saved_groups = tuple(value[6:] if value.startswith('group:') else None
                             for value in (workspace.a.currentData(), workspace.b.currentData()))
        report = run_with_progress(workspace, tr('Сравнение геохимических проб…'),
            lambda: compare_samples(project, a, b, fields, allow_mixed_material=mixed, names=names,
                                    saved_groups=saved_groups))
        if workspace.owner.model.project is not project or context != (source_token(workspace.owner), settings(project)):
            raise ValueError(tr('Данные изменились во время расчёта. Повторите сравнение.'))
        workspace.report, workspace._report_context = report, context
        project.view['geochemistry_comparison'] = deepcopy(report)
        project.event('geochemistry_comparison', sample_ids=[*a, *b], column_ids=fields)
        workspace.owner.view_changed()
        workspace._report_context = source_token(workspace.owner), settings(project)
        workspace.display_comparison()
    workspace.attempt(calculate)


def display_comparison(workspace):
    workspace.profile_match = None
    workspace.export_correlation_button.setEnabled(False)
    workspace.prediction_run = None
    workspace.passport_tabs.setTabVisible(workspace.prediction_tab, False)
    workspace.passport_tabs.setTabVisible(workspace.oil_mixing_tab, False)
    workspace.passport_tabs.setTabVisible(workspace.profile_tab, False)
    request = workspace.report['request']
    pair = len(request['group_a']) == len(request['group_b']) == 1
    alkane_ready = False
    if pair:
        try:
            compare_alkane_profiles(workspace.report)
        except ValueError as exc:
            workspace.profile_match_button.setToolTip(message(exc))
        else:
            alkane_ready = True
            workspace.profile_match_button.setToolTip(tr('Сравните две пробы по полному набору пиков nC24–nC34.'))
    else:
        workspace.profile_match_button.setToolTip(tr('Выберите по одной пробе в A и B.'))
    workspace.profile_match_button.setEnabled(alkane_ready)
    with QSignalBlocker(workspace.correlation_isotope_basis):
        workspace.correlation_isotope_basis.setChecked(False)
    workspace.update_correlation_readiness()
    workspace.clear_multivariate_results()
    multiple_fields = len(workspace.report['inputs']['fields']) >= 2
    workspace.prediction_button.setEnabled(multiple_fields and len(request['group_a']) >= 3 and len(request['group_b']) >= 2)
    workspace.oil_mixing_button.setEnabled(multiple_fields and len(request['group_a']) == 2 and len(request['group_b']) == 1)
    workspace.prediction_cv.setEnabled(workspace.prediction_button.isEnabled())
    workspace.clr.setEnabled(multiple_fields)
    if not multiple_fields:
        workspace.clr.setChecked(False)
    workspace.pca_button.setEnabled(multiple_fields)
    workspace.clustering_button.setEnabled(len(workspace.report['inputs']['samples']) >= 3)
    workspace.kmeans_button.setEnabled(workspace.clustering_button.isEnabled())
    workspace.dbscan_button.setEnabled(workspace.clustering_button.isEnabled())
    workspace.comparison_table.setRowCount(len(workspace.report['results']))
    for row, item in enumerate(workspace.report['results']):
        a, b = item['a'], item['b']
        excluded = '; '.join(f'{name}: {_status(key)}={count}' for name, group in (('A', a), ('B', b))
                             for key, count in group['excluded'].items())
        values = [item['field']['name'], item['field']['unit'], f"{a['n']} / {a['total']}", a['mean'], a['median'], a['sd'],
                  f"{b['n']} / {b['total']}", b['mean'], b['median'], b['sd'], item['difference'], excluded]
        for col, value in enumerate(values):
            cell = QTableWidgetItem(_number(value))
            if (point := workspace.comparison_measurement(row, col)) is not None:
                _, record = point
                analysis = record.get('analysis_run_id') or record.get('analysis_id')
                cell.setToolTip(
                    f"{tr('Анализ')}: {_number(analysis)}\n"
                    f"{tr('Повтор')}: {_number(record.get('repeat_number'))}\n"
                    + tr('Дважды щёлкните, чтобы открыть измерение в паспорте.')
                )
            workspace.comparison_table.setItem(row, col, cell)
    workspace.comparison_table.resizeColumnsToContents()
    workspace.export_comparison_button.setEnabled(True)
    workspace.passport_tabs.setCurrentWidget(workspace.comparison_table)
    workspace.status.setText(tr('A: {0}. B: {1}.', *workspace.report['request']['names']) + '\n' +
        tr('Описательное сравнение: одна выбранная точка на пробу. ND, пределы, пропуски и исключения QC не заменяются нулями.'))
    if workspace.report['results']:
        workspace.comparison_table.selectRow(0)


def comparison_measurement(workspace, row, column):
    if workspace.report is None or not (2 <= column <= 9):
        return None
    group = workspace.report['request']['group_a' if column < 6 else 'group_b']
    if len(group) != 1:
        return None
    field_id = workspace.report['results'][row]['field']['id']
    record = workspace.report['inputs']['points'][field_id][group[0]].get('measurement')
    return (group[0], record) if record is not None else None


def open_comparison_measurement(workspace, row, column):
    if (point := workspace.comparison_measurement(row, column)) is None:
        return
    sample_id, record = point
    for index in range(workspace.samples.count()):
        sample = workspace.samples.item(index)
        if sample.data(Qt.ItemDataRole.UserRole) == sample_id:
            if sample.isHidden():
                workspace.search.clear()
            workspace.samples.clearSelection()
            workspace.samples.setCurrentItem(sample)
            break
    else:
        return
    for index in range(workspace.measurements.rowCount()):
        candidate = workspace.measurements.item(index, 0).data(Qt.ItemDataRole.UserRole)
        if candidate['id'] == record['id']:
            workspace.passport_tabs.setCurrentWidget(workspace.measurements)
            workspace.measurements.selectRow(index)
            workspace.measurements.scrollToItem(workspace.measurements.item(index, 0))
            break


def update_correlation_readiness(workspace):
    if workspace.report is None:
        return
    ready = isotope_available = False
    try:
        preview = correlate_oil_profiles(workspace.report, confirm_c15_vpdb=True)
        actual = (preview if workspace.correlation_isotope_basis.isChecked()
                  else correlate_oil_profiles(workspace.report))
    except ValueError as exc:
        status = message(exc)
    else:
        count = sum(item['difference'] is not None for item in actual['evidence'])
        isotope_available = any(
            item['difference'] is not None and any(field_matches(
                *source_identity(Column(**item['field'], kind='number')), METHODS['sofer']['roles'][key]
            ) for key in ('sat', 'arom')) for item in preview['evidence']
        )
        ready = bool(actual['peaks']) or count > 0
        status = tr('Сопоставимые показатели: {0}', count)
        if actual['peaks']:
            status += '\n' + tr('Перекрытие н-алканов доступно.')
        if isotope_available and not workspace.correlation_isotope_basis.isChecked():
            status += '\n' + tr('Для сравнения изотопов подтвердите фракции C15+ и шкалу VPDB.')
    workspace.correlation_isotope_basis.setEnabled(isotope_available)
    workspace.oil_correlation_button.setEnabled(ready)
    workspace.oil_correlation_button.setToolTip(status)
    workspace.correlation_readiness.setText(status)


def show_profile_match(workspace, *, oil_correlation=False):
    def render():
        if workspace.report is None or not comparison_is_current(workspace.owner.model.project, workspace.report):
            raise ValueError(tr('Данные изменились. Пересчитайте сравнение перед экспортом.'))
        match = (correlate_oil_profiles(workspace.report, confirm_c15_vpdb=workspace.correlation_isotope_basis.isChecked())
                 if oil_correlation else compare_alkane_profiles(workspace.report))
        workspace.profile_match = match
        workspace.sensitivity_button.setVisible(oil_correlation)
        workspace.export_correlation_button.setEnabled(oil_correlation)
        figure = workspace.profile_canvas.figure
        figure.clear()
        if match['peaks']:
            axes = figure.add_subplot()
            carbons = [peak['carbon'] for peak in match['peaks']]
            for key, name in zip(('a_percent', 'b_percent'), workspace.report['request']['names']):
                axes.plot(carbons, [peak[key] for peak in match['peaks']], marker='o', label=name)
            axes.set_xticks(carbons, [f'nC{carbon}' for carbon in carbons])
            axes.set_ylabel(tr('Доля сигнала C24–C34, %'))
            axes.set_ylim(bottom=0)
            for label in axes.legend().get_texts():
                label.set_parse_math(False)
            axes.grid(axis='y', color='0.85')
        reviews = workspace.owner.model.project.view.get(PROFILE_REVIEWS_KEY, {})
        review = reviews.get(match.get('review_key', match['input_digest']), {})
        notes = {item['field_id']: item for item in review.get('evidence_notes', [])}
        workspace.profile_evidence_notes.clear()
        workspace.profile_evidence.setRowCount(0)
        workspace.profile_evidence.setRowCount(len(match['evidence']))
        for column in (5, 6):
            workspace.profile_evidence.setColumnHidden(column, not oil_correlation)
        indicators = _biodegradation_indicators()
        for row, evidence in enumerate(match['evidence']):
            a, b = evidence['a'], evidence['b']
            values = (evidence['field']['name'], evidence['field']['unit'],
                      *(_number(point['value']) if point['status'] == 'numeric' else _status(point['status'])
                        for point in (a, b)), _number(evidence['difference']))
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if evidence.get('missing_peaks'):
                    cell.setToolTip(tr('Недостающие пики: {0}', ', '.join(evidence['missing_peaks'])))
                workspace.profile_evidence.setItem(row, col, cell)
            if oil_correlation and evidence['difference'] is not None:
                field_id = evidence['field']['id']
                observation = QComboBox()
                observation.addItem(tr('Без пометки'), '')
                for side in ('a', 'b'):
                    for key, label in indicators.items():
                        observation.addItem(f'{side.upper()} · {label}', f'{side}:{key}')
                reason = QLineEdit()
                reason.setPlaceholderText(tr('Поясните, как признак ограничивает сравнение'))
                saved = notes.get(field_id)
                if saved:
                    observation.setCurrentIndex(max(0, observation.findData(f"{saved['side']}:{saved['indicator']}")))
                    reason.setText(saved['reason'])
                workspace.profile_evidence.setCellWidget(row, 5, observation)
                workspace.profile_evidence.setCellWidget(row, 6, reason)
                workspace.profile_evidence_notes[field_id] = observation, reason
        workspace.profile_evidence.resizeColumnsToContents()
        workspace.profile_views.setTabEnabled(1, bool(match['evidence']))
        workspace.profile_views.setTabEnabled(0, bool(match['peaks']))
        workspace.profile_views.setCurrentWidget(workspace.profile_canvas if match['peaks'] else workspace.profile_evidence)
        workspace.profile_note.setPlainText(review.get('note', ''))
        biodegradation = review.get('biodegradation', {})
        for side, combo in workspace.biodegradation.items():
            combo.setCurrentIndex(max(0, combo.findData(biodegradation.get(side))))
        observations = {(item['side'], item['indicator']): item
                        for item in review.get('biodegradation_observations', [])}
        nor25 = METHODS['hopane_indicators']['roles']['nor25']
        nor25_evidence = [
            item for item in match['evidence']
            if oil_correlation and field_matches(
                *source_identity(Column(**item['field'], kind='number')), nor25
            )
        ]
        for row, (key, (status, source)) in enumerate(workspace.biodegradation_observations.items()):
            observation = observations.get(key, {})
            status.setCurrentIndex(max(0, status.findData(observation.get('status'))))
            source.setText(observation.get('source', ''))
            side, indicator = key
            measured = [
                f"{item['field']['name']}: {_number(item[side]['value'])} [{item['field']['unit']}]"
                for item in nor25_evidence if item[side]['status'] == 'numeric'
            ] if indicator == 'hopane_changes' else []
            workspace.biodegradation_table.setItem(row, 4, QTableWidgetItem('\n'.join(measured)))
        workspace.biodegradation_table.resizeRowsToContents()
        workspace.passport_tabs.setTabVisible(workspace.profile_tab, True)
        workspace.passport_tabs.setCurrentWidget(workspace.profile_page)
        workspace.profile_canvas.draw_idle()
        overlap = match['overlap_percent']
        status = (tr('Перекрытие профилей: {0}%. Показатель описывает сходство формы двух распределений.',
                     _number(overlap)) if overlap is not None
                  else tr('Сравнение показателей без расчёта перекрытия профилей.'))
        if oil_correlation:
            status += '\n' + tr('Вывод об общем источнике требует экспертного заключения.')
        if match.get('overlap_issue'):
            status += '\n' + tr('Перекрытие пропущено: {0}', message(match['overlap_issue']))
        workspace.status.setText(status)

    workspace.attempt(render)


def show_correlation_sensitivity(workspace):
    def show():
        report, match = workspace.report, workspace.profile_match
        if report is None or match is None or 'correlation' not in match or not comparison_is_current(
            workspace.owner.model.project, report
        ):
            raise ValueError(tr('Данные изменились. Пересчитайте сравнение перед экспортом.'))
        scenarios = correlation_sensitivity(report, match)
        dialog = QDialog(workspace)
        dialog.setWindowTitle(tr('Чувствительность к составу показателей'))
        layout = QVBoxLayout(dialog)
        note = QLabel(tr('Сравните исходный набор со сценариями исключения группы или отдельного измерения. Вывод об общем источнике требует экспертного заключения.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        selector = QComboBox()
        selector.setToolTip(tr('Пробное исключение применяется только к этому расчёту.'))
        selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        selector.setMinimumContentsLength(24)
        selector.addItem(tr('Пробное исключение QC…'), None)
        for field in report['inputs']['fields']:
            for side in ('a', 'b'):
                sample_id, = report['request']['group_' + side]
                point = report['inputs']['points'][field['id']][sample_id]
                if point['status'] == 'numeric':
                    selector.addItem(f"{side.upper()} · {field['name']}", (field['id'], sample_id))
        selector.setEnabled(selector.count() > 1)
        layout.addWidget(selector)
        table = readonly_table([tr('Сценарий'), tr('Сопоставимые показатели'),
                                tr('Перекрытие н-алканов, %'), tr('Потерянные показатели')])
        table.resizeColumnsToContents()
        baseline = sum(item['difference'] is not None for item in match['evidence'])
        rows = [(tr('Исходный набор'), baseline, match['overlap_percent'], None)]
        rows.extend((group_label(row['group']), row['comparable'], row['overlap_percent'],
                     ', '.join(row['lost']) or None) for row in scenarios)
        def set_row(row, values):
            for col, value in enumerate(values):
                cell = QTableWidgetItem(_number(value))
                cell.setToolTip(cell.text())
                table.setItem(row, col, cell)

        def update_trial():
            table.setRowCount(len(rows))
            if (point := selector.currentData()) is None:
                return
            trial, = correlation_sensitivity(report, match, exclude_point=point)
            table.insertRow(len(rows))
            set_row(len(rows), (selector.currentText(), trial['comparable'], trial['overlap_percent'],
                                ', '.join(trial['lost']) or None))
            table.scrollToItem(table.item(len(rows), 0))

        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            set_row(row, values)
        selector.currentIndexChanged.connect(update_trial)
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(800, 400)
        dialog.exec()

    workspace.attempt(show)


def save_profile_review(workspace):
    def save():
        report, match = workspace.report, workspace.profile_match
        project = workspace.owner.model.project
        if report is None or match is None or not comparison_is_current(project, report):
            raise ValueError(tr('Данные изменились. Пересчитайте сравнение перед экспортом.'))
        note = workspace.profile_note.toPlainText().strip()
        if not note:
            raise ValueError(tr('Введите текст экспертного заключения.'))
        observations = []
        for (side, indicator), (status, source) in workspace.biodegradation_observations.items():
            value, reference = status.currentData(), source.text().strip()
            if value != 'insufficient_data' and not reference:
                workspace.profile_views.setCurrentWidget(workspace.biodegradation_table)
                source.setFocus()
                raise ValueError(tr('Укажите источник анализа для оценённого признака биодеградации.'))
            observations.append({
                'side': side, 'sample_id': report['request']['group_' + side][0],
                'indicator': indicator, 'status': value, 'source': reference,
            })
        observed = {(item['side'], item['indicator']) for item in observations if item['status'] == 'observed'}
        evidence_notes = []
        for field_id, (observation, reason) in workspace.profile_evidence_notes.items():
            selected, explanation = observation.currentData(), reason.text().strip()
            if not selected and not explanation:
                continue
            if not selected or tuple(selected.split(':')) not in observed:
                workspace.profile_views.setCurrentWidget(workspace.profile_evidence)
                raise ValueError(tr('Свяжите пометку с признаком, отмеченным как «Наблюдается».'))
            if not explanation:
                workspace.profile_views.setCurrentWidget(workspace.profile_evidence)
                raise ValueError(tr('Поясните влияние наблюдения на интерпретацию показателя.'))
            side, indicator = selected.split(':')
            evidence_notes.append({'field_id': field_id, 'side': side, 'indicator': indicator,
                                   'reason': explanation})
        project.view.setdefault(PROFILE_REVIEWS_KEY, {})[match.get('review_key', match['input_digest'])] = {
            'created_at': timestamp(), 'note': note, 'match': deepcopy(match), 'request': deepcopy(report['request']),
            'biodegradation': {side: combo.currentData() for side, combo in workspace.biodegradation.items()},
            'biodegradation_observations': observations,
            'evidence_notes': evidence_notes,
            'snapshot': freeze_comparison(report),
        }
        project.event('geochemistry_profile_review', input_digest=match['input_digest'])
        workspace.owner.view_changed()
        if workspace.passport_tabs.isTabVisible(workspace.review_archive_tab):
            workspace.refresh_review_archive()
        workspace.status.setText(tr('Экспертное заключение сохранено для текущих входов.'))

    workspace.attempt(save)


def draw_comparison(workspace):
    if workspace.report is None:
        return
    rows = workspace.comparison_table.selectionModel().selectedRows()
    if not rows:
        return
    item = workspace.report['results'][rows[0].row()]
    workspace.figure.clear()
    axis = workspace.figure.add_subplot(111)
    field = item['field']
    values = workspace.report['inputs']['points'][field['id']]
    for position, key in enumerate(('group_a', 'group_b')):
        numbers = [values[sid]['value'] for sid in workspace.report['request'][key] if values[sid]['status'] == 'numeric']
        offsets = [position + (((i * 37) % 101) / 100 - .5) * .28
                   for i in range(len(numbers))] if len(numbers) > 1 else [position] * len(numbers)
        axis.scatter(offsets, numbers, alpha=.7)
        if numbers:
            axis.plot([position - .12, position + .12], [sum(numbers) / len(numbers)] * 2, color='black')
    axis.set_xticks([0, 1], [f"{key.upper()}\nN = {item[key]['n']}" for key in ('a', 'b')])
    axis.set_ylabel(field['unit'])
    axis.set_title(fill(field['name'], width=36, max_lines=2, placeholder='…'), parse_math=False)
    axis.set_xlim(-.5, 1.5)
    workspace.canvas.draw_idle()
