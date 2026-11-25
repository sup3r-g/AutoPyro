"""Run and display geochemistry models using the current comparison."""

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTableWidgetItem,
)
from autopyro.analysis.geochemistry_workspace import (
    comparison_clustering,
    comparison_is_current,
    comparison_pca,
    comparison_prediction,
    comparison_dbscan,
    comparison_kmeans,
    comparison_oil_mixing,
)
from autopyro.model.measurements import parse_number
from ..i18n import tr
from ..tasks import run_with_progress
from .geochemistry_workspace_display import _number, _status


def calculate_oil_mixing(workspace):
    report = workspace.report
    if report is None:
        return
    request, inputs = report['request'], report['inputs']
    sample_names = [inputs['samples'][sid]['name'] for sid in (*request['group_a'], *request['group_b'])]
    dialog = QDialog(workspace)
    dialog.setWindowTitle(tr('Допуски бинарного смешения'))
    form = QFormLayout(dialog)
    form.addRow(QLabel(tr('Компонент A — {0}. Компонент B — {1}. Смесь — {2}.', *sample_names)))
    form.addRow(QLabel(tr('Введите положительный допуск в единицах каждого маркера.')))
    edits = {}
    for field in inputs['fields']:
        edit = QLineEdit(dialog)
        form.addRow(f"{field['name']} ({field['unit']})", edit)
        edits[field['id']] = edit
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return

    def calculate():
        if not comparison_is_current(workspace.owner.model.project, report):
            raise ValueError(tr('Данные изменились. Пересчитайте сравнение перед экспортом.'))
        run = comparison_oil_mixing(report, {cid: parse_number(edit.text()) for cid, edit in edits.items()})
        workspace.render_oil_mixing(report, run)
        workspace.save_model_run(report, run)

    workspace.attempt(calculate)


def render_oil_mixing(workspace, report, run):
    samples = report['inputs']['samples']
    names = [samples[sid]['name'] for sid in run['sample_ids']]
    workspace.oil_mixing_table.setRowCount(len(run['markers']))
    for index, marker in enumerate(run['markers']):
        field = marker['field']
        observations = [_number(value) if status == 'numeric' else _status(status)
                        for value, status in zip(marker['values'], marker['statuses'])]
        yes_no = {True: tr('Да'), False: tr('Нет'), None: '—'}
        values = [field['name'], field['unit'], *observations, run['tolerances'][field['id']],
                  marker['predicted'], marker['residual'], yes_no[marker['used_for_fit']],
                  yes_no[marker['within_tolerance']]]
        for column, value in enumerate(values):
            workspace.oil_mixing_table.setItem(index, column, QTableWidgetItem(_number(value)))
    workspace.oil_mixing_table.resizeColumnsToContents()
    workspace.passport_tabs.setTabText(workspace.oil_mixing_tab, tr('Бинарное смешение нефтей'))
    workspace.passport_tabs.setTabVisible(workspace.oil_mixing_tab, True)
    workspace.passport_tabs.setCurrentWidget(workspace.oil_mixing_table)
    status = {
        'consistent': tr('Согласуется с заданными допусками.'),
        'inconsistent': tr('Остатки превышают допуск. Доля приведена для диагностики.'),
        'endpoint': tr('Совместимо с одним конечным компонентом.'),
        'underdetermined': tr('Недостаточно маркеров, различающих конечные компоненты.'),
    }[run['status']]
    if run['outside_interval']:
        status += ' ' + tr('Свободное решение вне интервала 0–1.')
    fraction = run['fraction_b']
    fraction_label = _number(100 * fraction) + '%' if fraction is not None else '—'
    workspace.status.setText(tr('Компонент A — {0}. Компонент B — {1}. Смесь — {2}.', *names)
                        + '\n' + tr('Массовая доля компонента B — {0}.', fraction_label) + ' ' + status)


def calculate_prediction(workspace):
    report = workspace.report
    if report is None:
        return
    fields = report['inputs']['fields']
    labels = [f"{i}. {field['name']}" for i, field in enumerate(fields, 1)]
    label, ok = QInputDialog.getItem(workspace, tr('Линейный прогноз A → B'), tr('Целевой показатель'), labels, 0, False)
    if not ok:
        return
    target = fields[labels.index(label)]['id']
    well_cv = workspace.prediction_cv.isChecked()

    def calculate():
        run = run_with_progress(workspace, tr('Линейный прогноз A → B…'),
                                lambda: comparison_prediction(report, target, well_cv=well_cv))
        if report is not workspace.report or not comparison_is_current(workspace.owner.model.project, report):
            raise ValueError(tr('Данные изменились во время расчёта. Повторите сравнение.'))
        workspace.prediction_run = run
        workspace.save_model_run(report, run)
        workspace.render_prediction(report, run)

    workspace.attempt(calculate)


def render_prediction(workspace, report, run):
    fields = {field['id']: field for field in report['inputs']['fields']}
    target = fields[run['target']]
    predictions = run['predictions']
    figure = workspace.prediction_canvas.figure
    figure.clear()
    axes = figure.add_subplot()
    observed = [item['observed'] for item in predictions]
    estimated = [item['predicted'] for item in predictions]
    lower, upper = min(*observed, *estimated), max(*observed, *estimated)
    axes.plot([lower, upper], [lower, upper], color='0.6')
    axes.scatter(observed, estimated)
    baseline = run.get('baseline')
    if baseline is not None:
        axes.axhline(baseline['value'], color='tab:orange', linestyle='--', label=tr('Среднее по обучающей выборке'))
        axes.legend()
    axes.set_xlabel(tr('Измерено') + f" [{target['unit']}]")
    axes.set_ylabel(tr('Прогноз') + f" [{target['unit']}]")
    axes.set_title(target['name'], parse_math=False)
    workspace.prediction_table.setRowCount(len(predictions))
    for row, item in enumerate(predictions):
        sample = report['inputs']['samples'][item['sample_id']]
        outside = ', '.join(fields[key]['name'] for key in item['outside_train_range'])
        for col, value in enumerate((sample['name'], _number(item['observed']), _number(item['predicted']), outside)):
            workspace.prediction_table.setItem(row, col, QTableWidgetItem(value))
    workspace.prediction_table.resizeColumnsToContents()
    cross_validation = run.get('well_cv')
    folds = cross_validation['folds'] if cross_validation else []
    workspace.prediction_cv_table.setRowCount(len(folds))
    for row, fold in enumerate(folds):
        values = (fold['well'], str(len(fold['predictions'])),
                  *(_number(fold['metrics'][name]) for name in ('RMSE', 'MAE', 'R2')),
                  _number(fold.get('baseline', {}).get('metrics', {}).get('RMSE')))
        for col, value in enumerate(values):
            workspace.prediction_cv_table.setItem(row, col, QTableWidgetItem(value))
    workspace.prediction_cv_table.resizeColumnsToContents()
    workspace.prediction_details.setTabVisible(1, bool(folds))
    by_well = run.get('validation_by_well')
    workspace.prediction_details.setTabVisible(2, by_well is not None)
    if by_well is not None:
        rows = [(item['well'], len(item['sample_ids']), item['metrics'], item['baseline_metrics'])
                for item in by_well['wells']]
        rows.append((tr('Все скважины с равным весом'), len(predictions), by_well['equal_weight_metrics'],
                     by_well['equal_weight_baseline_metrics']))
        workspace.prediction_well_table.setRowCount(len(rows))
        for row, (name, count, model, reference) in enumerate(rows):
            values = (name, str(count), *(_number(metrics[key]) for metrics in (model, reference) for key in ('RMSE', 'MAE')))
            for col, value in enumerate(values):
                workspace.prediction_well_table.setItem(row, col, QTableWidgetItem(value))
        workspace.prediction_well_table.resizeColumnsToContents()
    workspace.prediction_details.setCurrentIndex(0)
    workspace.passport_tabs.setTabText(workspace.prediction_tab, tr('Проверка прогноза'))
    workspace.passport_tabs.setTabVisible(workspace.prediction_tab, True)
    workspace.passport_tabs.setCurrentWidget(workspace.prediction_page)
    workspace.prediction_canvas.draw_idle()
    metrics = ', '.join(f'{name}={_number(run["metrics"][name])}' for name in ('RMSE', 'MAE', 'R2'))
    status = tr('Обучение A: {0} проб. Проверка B: {1} проб. Исключено: {2}.',
                len(run['train_ids']), len(predictions), len(run['excluded'])) + ' ' + metrics
    if baseline is not None:
        status += '\n' + tr('Базовый прогноз на B: RMSE {0}, MAE {1}.',
                             _number(baseline['metrics']['RMSE']), _number(baseline['metrics']['MAE']))
    if cross_validation:
        status += '\n' + tr('Проверка внутри A по {0} скважинам. RMSE: {1}.',
                             len(folds), _number(cross_validation['metrics']['RMSE']))
        if 'baseline_metrics' in cross_validation:
            status += ' ' + tr('RMSE базового прогноза: {0}.', _number(cross_validation['baseline_metrics']['RMSE']))
    workspace.status.setText(status)


def calculate_pca(workspace):
    report = workspace.report
    if report is None:
        return

    preprocessing = workspace.preprocessing()

    def calculate():
        run = run_with_progress(workspace, tr('Расчёт PCA…'), lambda: comparison_pca(report, preprocessing=preprocessing))
        if (report is not workspace.report or preprocessing != workspace.preprocessing()
                or not comparison_is_current(workspace.owner.model.project, report)):
            raise ValueError(tr('Данные изменились. Пересчитайте сравнение перед экспортом.'))
        workspace.pca_run = run
        workspace.save_model_run(report, run)
        workspace.render_pca(report, run)

    workspace.attempt(calculate)


def render_pca(workspace, report, run):
    preprocessing = run['preprocessing']
    figure = workspace.pca_canvas.figure
    figure.clear()
    axes = figure.add_subplot()
    if preprocessing == 'clr':
        axes.set_title('PCA · CLR')
    for group, name in zip(('group_a', 'group_b'), report['request']['names']):
        coordinates = [score['coordinates'] for score in run['scores'] if score['group'] == group]
        if coordinates:
            xs, ys = zip(*coordinates)
            axes.scatter(xs, ys, label=name)
    for index, setter in enumerate((axes.set_xlabel, axes.set_ylabel)):
        setter(f'PC{index + 1} ({run["explained_variance_ratio"][index]:.1%})')
    for label in axes.legend().get_texts():
        label.set_parse_math(False)
    axes.grid(color='0.85')
    fields = {field['id']: field['name'] for field in report['inputs']['fields']}
    workspace.pca_coefficients.setRowCount(len(run['fields']))
    for row, (field, pc1, pc2) in enumerate(zip(run['fields'], *run['coefficients'])):
        for col, value in enumerate((fields[field], _number(pc1), _number(pc2))):
            workspace.pca_coefficients.setItem(row, col, QTableWidgetItem(value))
    workspace.pca_coefficients.resizeColumnsToContents()
    workspace.passport_tabs.setTabText(workspace.pca_tab, 'PCA')
    workspace.passport_tabs.setTabVisible(workspace.pca_tab, True)
    workspace.passport_tabs.setCurrentWidget(workspace.pca_page)
    workspace.pca_canvas.draw_idle()
    workspace.status.setText(tr('PCA: включено проб {0}, исключено неполных проб {1}, постоянных показателей {2}. Показаны коэффициенты PC1 и PC2.',
                           len(run['scores']), len(run['excluded']), len(run['constant_fields'])))


def dbscan_parameters(workspace, title, sample_count):
    dialog = QDialog(workspace)
    dialog.setWindowTitle(title)
    form = QFormLayout(dialog)
    radius = QDoubleSpinBox(dialog)
    radius.setDecimals(4)
    radius.setRange(0.0001, 1000000)
    radius.setValue(0.5)
    neighbors = QSpinBox(dialog)
    neighbors.setRange(2, min(sample_count, 2000))
    neighbors.setValue(min(5, sample_count))
    form.addRow(tr('Радиус в координатах CLR') if workspace.clr.isChecked() else
                tr('Радиус в стандартизированных единицах'), radius)
    form.addRow(tr('Минимум проб в окрестности, включая саму пробу'), neighbors)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)
    options = (radius.value(), neighbors.value()) if dialog.exec() == QDialog.DialogCode.Accepted else None
    dialog.deleteLater()
    return options


def calculate_clustering(workspace, *, method='ward'):
    report = workspace.report
    if report is None or len(report['inputs']['samples']) < 3:
        return
    sample_count = len(report['inputs']['samples'])
    preprocessing = workspace.preprocessing()
    if method == 'dbscan':
        title, fit = tr('Кластеризация DBSCAN'), comparison_dbscan
        options = workspace.dbscan_parameters(title, sample_count)
        if options is None:
            return
    else:
        title, fit = ((tr('Кластеризация K-средних'), comparison_kmeans) if method == 'kmeans' else
                      (tr('Иерархическая кластеризация'), comparison_clustering))
        limit = min(sample_count - 1, 20 if method == 'kmeans' else 1999)
        count, ok = QInputDialog.getInt(workspace, title, tr('Число кластеров'), 2, 2, limit)
        if not ok:
            return
        options = (count,)

    def calculate():
        run = run_with_progress(workspace, title, lambda: fit(report, *options, preprocessing=preprocessing))
        if (report is not workspace.report or preprocessing != workspace.preprocessing()
                or not comparison_is_current(workspace.owner.model.project, report)):
            raise ValueError(tr('Данные изменились. Пересчитайте сравнение перед экспортом.'))
        if method == 'dbscan':
            workspace.dbscan_run = run
        elif method == 'kmeans':
            workspace.kmeans_run = run
        else:
            workspace.clustering_run = run
        workspace.save_model_run(report, run)
        workspace.render_clustering(report, run, title)

    workspace.attempt(calculate)


def render_clustering(workspace, report, run, title):
    method = run['method']
    preprocessing = run['preprocessing']
    members = run['membership']
    count = run['cluster_count']
    figure = workspace.clustering_canvas.figure
    figure.clear()
    axes = figure.add_subplot()
    roles = {'core': tr('Ядро'), 'border': tr('Граница'), 'noise': tr('Шум')}
    if method == 'dbscan':
        from collections import Counter

        totals = Counter(item['role'] for item in members)
        axes.bar(list(roles.values()), [totals[key] for key in roles])
        axes.set_ylabel(tr('Число проб'))
    elif method == 'kmeans':
        fields = {field['id']: field['name'] for field in report['inputs']['fields']}
        positions = range(len(run['fields']))
        for cluster, center in enumerate(run['centers'], 1):
            axes.plot(positions, center, marker='o', label=str(cluster))
        axes.set_xticks(positions, [fields[field] for field in run['fields']], rotation=30, ha='right')
        axes.set_ylabel(tr('Центр кластера в координатах CLR') if preprocessing == 'clr' else
                        tr('Стандартизированный центр кластера'))
        axes.legend(title=tr('Кластер'), ncols=min(5, count))
        axes.grid(axis='y', color='0.85')
    else:
        from scipy.cluster.hierarchy import dendrogram

        dendrogram(run['linkage'], ax=axes, labels=[str(i) for i in range(1, len(members) + 1)],
                   truncate_mode='lastp', p=30, color_threshold=0)
        axes.set_xlabel(tr('Номера проб или число проб в свёрнутой ветви (в скобках)'))
        axes.set_ylabel(tr('Расстояние Уорда'))
    names = dict(zip(('group_a', 'group_b'), report['request']['names']))
    samples = report['inputs']['samples']
    workspace.cluster_members.setRowCount(len(members))
    workspace.cluster_members.setColumnHidden(4, method != 'dbscan')
    for row, item in enumerate(members):
        sample = samples[item['sample_id']]
        for col, value in enumerate((row + 1, sample['name'], names[item['group']],
                                     item['cluster'] or tr('Шум'), roles.get(item.get('role'), ''))):
            cell = QTableWidgetItem(str(value))
            cell.setToolTip(item['sample_id'])
            workspace.cluster_members.setItem(row, col, cell)
    workspace.cluster_members.resizeColumnsToContents()
    workspace.passport_tabs.setTabText(workspace.clustering_tab, title + ' · CLR' if preprocessing == 'clr' else title)
    workspace.passport_tabs.setTabVisible(workspace.clustering_tab, True)
    workspace.passport_tabs.setCurrentWidget(workspace.clustering_page)
    workspace.clustering_canvas.draw_idle()
    status = tr('Кластеров {0}, включено проб {1}, исключено неполных проб {2}, постоянных показателей {3}.',
                count, len(members), len(run['excluded']), len(run['constant_fields']))
    if method == 'dbscan':
        status += ' ' + tr('Шумовых проб: {0}.', run['noise_count'])
    workspace.status.setText(status)
