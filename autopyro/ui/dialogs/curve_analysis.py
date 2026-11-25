"""Additional curve-analysis views; numerical fitting remains in core modules."""
from operator import itemgetter

import numpy as np
from scipy.stats import probplot
from matplotlib.figure import Figure
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTableView, QPushButton, QListWidget,
                              QListWidgetItem, QInputDialog, QFileDialog, QAbstractItemView, QLabel, QComboBox)

from autopyro.curves.curve_numerics import residual_autocorrelation
from autopyro.curves.curve_validation import prediction_metrics
from ..i18n import tr, message
from ..theme import icon
from ..widgets.figure_canvas import FittedFigureCanvas
from ..widgets.navigation import NavigationToolbar
from ..widgets.results import ResultsModel
from ..widgets.documents import parameter_page
from ..table.selection import select_row_ids


def _pair_interval_note(pair):
    interval = pair.interval
    if interval is None:
        return tr("Интервал ΔMAE недоступен: {0}", message(pair.interval_error)) if pair.interval_error else ""
    methods = {"rows": tr("По строкам"), "groups": tr("По группам"), "moving_blocks": tr("Скользящие блоки")}
    directions = {"a": tr("Интервал указывает на меньшую MAE у A."),
                  "b": tr("Интервал указывает на меньшую MAE у B."),
                  "uncertain": tr("Интервал включает ноль. Направление различия неопределённо.")}
    lines = [tr("Интервал ΔMAE {0:g}%: [{1:.6g}, {2:.6g}] · {3}", interval["confidence"]*100,
                interval["lower"], interval["upper"], methods[interval["method"]]), directions[interval["direction"]]]
    practical = interval.get("practical")
    if practical is not None:
        conclusions = {"a": tr("Преимущество A превышает порог."),
                       "b": tr("Преимущество B превышает порог."),
                       "within_margin": tr("Весь интервал в пределах допуска."),
                       "uncertain": tr("Вывод относительно порога неопределён.")}
        lines.append(tr("Порог ΔMAE: ±{0:.6g}. {1}", practical["margin"], conclusions[practical["conclusion"]]))
    return "\n".join(lines)


class _PairedTestTableModel(ResultsModel):
    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if not 0 <= column < len(self.headers):
            return
        present, missing = [], []
        for row in self.rows:
            (missing if row[column] is None else present).append(row)
        present.sort(key=itemgetter(column), reverse=order == Qt.SortOrder.DescendingOrder)
        self.set_rows(present+missing)


class CurveAnalysisTools:
    def build_analysis_tools(self):
        self._archive_records = None
        page = QWidget()
        layout = QVBoxLayout(page)
        self.diagnostic_figure = Figure(layout="constrained")
        self.diagnostic_canvas = FittedFigureCanvas(self.diagnostic_figure)
        layout.addWidget(NavigationToolbar(self.diagnostic_canvas, page))
        layout.addWidget(self.diagnostic_canvas, 2)
        self.residual_table = QTableView()
        self.residual_table.setMaximumHeight(230)
        self.residual_table.doubleClicked.connect(self.jump_to_sample)
        layout.addWidget(self.residual_table)
        jump = QPushButton(icon('table-2'), tr("Показать выбранную пробу"))
        jump.clicked.connect(self.jump_to_sample)
        layout.addWidget(jump)
        self.tabs.addTab(page, icon('shield-check'), tr("Диагностика"))
        self.diagnostic_page = page
        self._diagnostic_source = None
        self._diagnostic_drawn = False
        self.tabs.currentChanged.connect(self.refresh_diagnostic_view)
        self.autocorrelation_page = QWidget()
        acf_layout = QVBoxLayout(self.autocorrelation_page)
        self.autocorrelation_figure = Figure(layout="constrained")
        self.autocorrelation_canvas = FittedFigureCanvas(self.autocorrelation_figure)
        acf_layout.addWidget(self.autocorrelation_canvas, 1)
        self.autocorrelation_note = QLabel()
        self.autocorrelation_note.setWordWrap(True)
        acf_layout.addWidget(self.autocorrelation_note)
        self.autocorrelation_tab = self.tabs.addTab(self.autocorrelation_page, icon("chart-column"), tr("Автокорреляция остатков"))
        self.tabs.setTabEnabled(self.autocorrelation_tab, False)
        self._autocorrelation_drawn = False
        self.tabs.currentChanged.connect(self.refresh_autocorrelation)
        self.influence_page = QWidget()
        influence_layout = QVBoxLayout(self.influence_page)
        self.influence_summary = QLabel()
        self.influence_summary.setWordWrap(True)
        self.influence_summary.setToolTip(tr("Метрики LOO рассчитываются без весов по исходным значениям Y."))
        influence_layout.addWidget(self.influence_summary)
        self.influence_table = QTableView()
        influence_layout.addWidget(self.influence_table)
        self.influence_table.doubleClicked.connect(self.jump_to_influential_sample)
        self.influence_tab = self.tabs.addTab(self.influence_page, icon("chart-column"), tr("Влияние точек"))
        self.tabs.setTabToolTip(self.influence_tab, tr("Диагностика влияния доступна для линейного OLS/WLS без ограничений"))
        self.tabs.setTabEnabled(self.influence_tab, False)
        self.tabs.currentChanged.connect(self.refresh_influence)
        self.holdout_pair_page = QWidget()
        pair_layout = QVBoxLayout(self.holdout_pair_page)
        self.holdout_pair_summary = QLabel()
        self.holdout_pair_summary.setWordWrap(True)
        self.holdout_pair_summary.setTextFormat(Qt.TextFormat.PlainText)
        pair_layout.addWidget(self.holdout_pair_summary)
        self.holdout_pair_table = QTableView()
        self.holdout_pair_table.doubleClicked.connect(self.jump_to_test_sample)
        pair_layout.addWidget(self.holdout_pair_table)
        self.holdout_pair_tab = self.tabs.addTab(self.holdout_pair_page, icon("table-2"), tr("Попарный тест"))
        self.tabs.setTabEnabled(self.holdout_pair_tab, False)
        self.tabs.currentChanged.connect(self.refresh_holdout_pair)
        self.archive_diff = None
        self._archive_diff_mode = None
        self.archive_diff_page = QWidget()
        archive_layout = QVBoxLayout(self.archive_diff_page)
        self.archive_diff_summary = QLabel()
        self.archive_diff_summary.setWordWrap(True)
        self.archive_diff_summary.setTextFormat(Qt.TextFormat.PlainText)
        archive_layout.addWidget(self.archive_diff_summary)
        self.archive_diff_choice = QComboBox()
        for label, mode, image in ((tr("Изменения данных"), "changes", "table-2"),
                                   (tr("Изменения коэффициентов"), "parameters", "chart-column"),
                                   (tr("Разница прогнозов"), "curve", "chart-no-axes-combined")):
            self.archive_diff_choice.addItem(icon(image), label, mode)
        self.archive_diff_choice.currentIndexChanged.connect(self.refresh_archive_diff)
        archive_layout.addWidget(self.archive_diff_choice)
        self.archive_diff_table = QTableView()
        self.archive_diff_table.doubleClicked.connect(self.jump_to_changed_sample)
        archive_layout.addWidget(self.archive_diff_table)
        self.archive_diff_tab = self.tabs.addTab(self.archive_diff_page, icon("files"), tr("Сравнение с архивом"))
        self.tabs.setTabEnabled(self.archive_diff_tab, False)
        self.tabs.currentChanged.connect(self.refresh_archive_diff)
        parameter_page_widget = QWidget()
        parameter_layout = QVBoxLayout(parameter_page_widget)
        self.parameter_figure = Figure(layout="constrained")
        self.parameter_canvas = FittedFigureCanvas(self.parameter_figure)
        parameter_layout.addWidget(self.parameter_canvas)
        self.parameter_table = QTableView()
        self.parameter_table.clicked.connect(self.draw_parameter_histogram)
        parameter_layout.addWidget(self.parameter_table)
        self.tabs.addTab(parameter_page_widget, icon('chart-column'), tr("Параметры bootstrap"))
        self._parameter_cache = None
        self.diagnostic_canvas.mpl_connect("pick_event", self.pick_sample)
        self.diagnostic_rows = []
        self.diagnostic_artists = {}
        controls = parameter_page(self.parameter_tabs, tr("Результаты модели"), 'files')
        self.result_controls = controls
        self.apply_fit_button = QPushButton(icon('columns-3'), tr("Создать колонки прогноза и полос"))
        self.apply_fit_button.setEnabled(False)
        self.apply_fit_button.clicked.connect(self.apply_current_fit)
        controls.addWidget(self.apply_fit_button)
        self.save_fit_button = QPushButton(icon('save'), tr("Сохранить результат…"))
        self.save_fit_button.setEnabled(False)
        self.save_fit_button.clicked.connect(self.save_current_result)
        controls.addWidget(self.save_fit_button)
        self.current_fit_button = QPushButton(icon('chart-no-axes-combined'), tr("Показать текущий расчёт"))
        self.current_fit_button.setEnabled(False)
        self.current_fit_button.clicked.connect(self.show_current_fit)
        controls.addWidget(self.current_fit_button)
        self.saved_results = QListWidget()
        self.saved_results.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        controls.addWidget(self.saved_results)
        self._saved_signature = None
        self.refresh_saved_results()
        for text, callback, image in ((tr("Восстановить параметры результата"), self.restore_result_settings, 'rotate-ccw'),
                               (tr("Пересчитать по сохранённым строкам"), self.replay_saved_result, 'rotate-ccw'),
                               (tr("Показать выбранные кривые"), self.show_saved_curves, 'chart-no-axes-combined'),
                               (tr("Экспорт отчёта HTML…"), self.export_fit_report, 'file-text'),
                               (tr("Удалить сохранённый результат"), self.remove_saved_results, 'trash')):
            button = QPushButton(icon(image), text)
            button.clicked.connect(callback)
            controls.addWidget(button)
        controls.addStretch()

    def clear_diagnostics(self):
        if not hasattr(self, "diagnostic_figure"):
            return
        self.diagnostic_figure.clear()
        self.diagnostic_artists.clear()
        old = self.residual_table.model()
        self.residual_table.setModel(None)
        if old is not None:
            old.deleteLater()
        self.diagnostic_canvas.draw_idle()

    def clear_analysis(self):
        if not hasattr(self, "diagnostic_figure"):
            return
        self._diagnostic_source = None
        self._diagnostic_drawn = False
        self.clear_diagnostics()
        self.archive_diff = None
        self._archive_diff_mode = None
        self.tabs.setTabEnabled(self.archive_diff_tab, False)
        self.archive_diff_summary.clear()
        old = self.archive_diff_table.model()
        self.archive_diff_table.setModel(None)
        if old is not None:
            old.deleteLater()
        self.tabs.setTabEnabled(self.autocorrelation_tab, False)
        self.autocorrelation_figure.clear()
        self.autocorrelation_note.clear()
        self._autocorrelation_drawn = False
        self.tabs.setTabEnabled(self.influence_tab, False)
        self.influence_summary.clear()
        old = self.influence_table.model()
        self.influence_table.setModel(None)
        if old is not None:
            old.deleteLater()
        self._archive_records = None
        self.apply_fit_button.setEnabled(False)
        self.save_fit_button.setEnabled(False)
        self.current_fit_button.setEnabled(False)
        self.parameter_figure.clear()
        self.parameter_canvas.draw_idle()
        old = self.parameter_table.model()
        self.parameter_table.setModel(None)
        if old is not None:
            old.deleteLater()
        self._parameter_cache = None
        self.refresh_saved_results()

    def display_diagnostics(self):
        r = self.result
        self._diagnostic_source = (r.y, r.fitted, self.row_ids, False)
        self._diagnostic_drawn = False
        self.refresh_diagnostic_view()
        self.tabs.setTabEnabled(self.autocorrelation_tab, True)
        self.refresh_autocorrelation()
        self.tabs.setTabEnabled(self.influence_tab, r.supports_influence)
        self.refresh_influence()
        self.apply_fit_button.setEnabled(True)
        self.save_fit_button.setEnabled(True)
        self.current_fit_button.setEnabled(True)

    def show_current_fit(self):
        if self.result is None:
            return
        self._archive_records = None
        self.confidence.setEnabled(True)
        self.export_button.setEnabled(True)
        self.apply_fit_button.setEnabled(True)
        self.save_fit_button.setEnabled(True)
        bands_available = self.current_bands_available()
        self.show_confidence.setEnabled(bands_available)
        self.show_prediction.setEnabled(bands_available)
        self.redraw()
        self.tabs.setCurrentWidget(self.plot_page)

    def refresh_diagnostic_view(self, *args):
        if self.tabs.currentWidget() is self.diagnostic_page and self._diagnostic_source is not None and not self._diagnostic_drawn:
            observed, predicted, ids, oof = self._diagnostic_source
            self.draw_diagnostics(observed, predicted, ids, oof=oof)
            self._diagnostic_drawn = True

    def refresh_autocorrelation(self, *args):
        if (self.tabs.currentWidget() is not self.autocorrelation_page
                or self.result is None or self._autocorrelation_drawn):
            return
        self._autocorrelation_drawn = True
        result = self.result
        try:
            values = residual_autocorrelation(result.x, result.y-result.fitted)
        except ValueError as exc:
            self.autocorrelation_note.setText(message(exc))
        else:
            axis = self.autocorrelation_figure.subplots()
            axis.bar(np.arange(1, len(values)+1), values)
            axis.axhline(0, color="gray", linewidth=.8)
            axis.set(xlabel=tr("Лаг в точках"), ylabel=tr("Автокорреляция остатков"), ylim=(-1, 1))
            self.autocorrelation_note.setText(tr("Остатки упорядочены по X. Лаг измеряется в точках. Для интерпретации по времени или глубине нужен примерно равный шаг X."))
        self.autocorrelation_canvas.draw_idle()

    def refresh_influence(self, *args):
        result = self.result
        if (self.tabs.currentWidget() is not self.influence_page or self.influence_table.model() is not None
                or result is None or not result.supports_influence):
            return
        leverage, cook = result.influence()
        predicted = result.leave_one_out(leverage=leverage)
        with np.errstate(over="ignore"):
            errors = result.y-predicted
        valid = np.isfinite(errors)
        self.influence_summary.setText(self.leave_one_out_summary(result.y, predicted, errors, valid))
        order = np.lexsort((-leverage, -np.nan_to_num(cook, nan=-np.inf)))
        rows = [[self.row_ids[i], float(result.x[i]), float(result.y[i]),
                 float(predicted[i]) if valid[i] else None, float(errors[i]) if valid[i] else None, float(leverage[i]),
                 float(cook[i]) if np.isfinite(cook[i]) else None] for i in order]
        headers = ["UUID", "X", "Y", tr("Прогноз LOO"), tr("Ошибка LOO"), tr("Рычаг"), tr("Расстояние Кука")]
        self.influence_table.setModel(ResultsModel(headers, rows, self.influence_table))

    @staticmethod
    def leave_one_out_summary(y, predicted, errors, valid):
        if not np.all(valid):
            return tr("Прогнозы LOO: {0} из {1}. Для сводки нужны все прогнозы.", int(valid.sum()), len(y))
        try:
            metrics = prediction_metrics(y, predicted)
        except ValueError as exc:
            return message(exc)
        press = float(errors @ errors)
        r2 = f"{metrics['R2']:.6g}" if np.isfinite(metrics["R2"]) else "—"
        return f"LOOCV · PRESS {press:.6g} · RMSE {metrics['RMSE']:.6g} · MAE {metrics['MAE']:.6g} · R² {r2}"

    def jump_to_influential_sample(self, index):
        if index.isValid():
            self.select_source_sample(self.influence_table.model().rows[index.row()][0])

    def clear_holdout_pair(self):
        self.tabs.setTabEnabled(self.holdout_pair_tab, False)
        self.holdout_pair_summary.clear()
        old = self.holdout_pair_table.model()
        self.holdout_pair_table.setModel(None)
        if old is not None:
            old.deleteLater()

    def refresh_holdout_pair(self, *args):
        if (self.tabs.currentWidget() is not self.holdout_pair_page or self.holdout_pair_table.model() is not None
                or self.comparison_result is None or self.comparison_result.holdout is None):
            return
        test = self.comparison_result.holdout
        pair = test.pair
        if pair is None:
            return
        first, second = pair.models
        metrics = pair.metrics
        lines = [f"A · {first.name}\nB · {second.name}",
                 tr("Общих прогнозов: {0} из {1}", metrics["N_common"], metrics["N_test"])]
        if metrics["N_common"]:
            lines.append(f"MAE A {metrics['MAE_A']:.6g} · MAE B {metrics['MAE_B']:.6g} · Δ {metrics['mean_difference']:.6g}")
            lines.append(tr("A точнее: {0} · B точнее: {1} · Одинаково: {2}",
                            metrics["wins_A"], metrics["wins_B"], metrics["ties"]))
        interval_note = _pair_interval_note(pair)
        if interval_note:
            lines.append(interval_note)
        lines.extend(f"{model.name} · {message(model.error)}" for model in pair.models if model.error)
        self.holdout_pair_summary.setText("\n".join(lines))

        def finite_values(values):
            cells = values.astype(object)
            cells[~np.isfinite(values)] = None
            return cells.tolist()

        prediction_a, prediction_b = (model.predictions if model.predictions is not None else np.full_like(test.y, np.nan)
                                      for model in pair.models)
        values = (prediction_a, prediction_b, *pair.absolute_errors, pair.difference)
        rows = list(zip(self.holdout_row_ids, test.x.tolist(), test.y.tolist(), *(finite_values(column) for column in values)))
        headers = ["UUID", "X", "Y", tr("Прогноз A"), tr("Прогноз B"), "|e A|", "|e B|", "Δ (B − A)"]
        table = self.holdout_pair_table
        table.setSortingEnabled(False)
        table.setModel(_PairedTestTableModel(headers, rows, table))
        table.horizontalHeader().setSortIndicator(len(headers)-1, Qt.SortOrder.DescendingOrder)
        table.setSortingEnabled(True)

    def jump_to_test_sample(self, index):
        if index.isValid():
            self.select_source_sample(self.holdout_pair_table.model().rows[index.row()][0])

    def refresh_archive_diff(self, *args):
        if self.tabs.currentWidget() is not self.archive_diff_page or self.archive_diff is None:
            return
        mode = self.archive_diff_choice.currentData()
        if mode == self._archive_diff_mode:
            return
        self._archive_diff_mode = mode
        diff = self.archive_diff
        lines = [diff["archive_name"], tr("Изменилось обучающих строк: {0}", diff["changed_rows"])]
        if peak := diff["peak"]:
            lines.append(tr("Максимальное |ΔY| при X = {0:.6g}. ΔY = {1:.6g}", peak["x"], peak["delta"]))
        if diff["curve_error"]:
            lines.append(message(diff["curve_error"]))
        self.archive_diff_summary.setText("\n".join(lines))
        before, after = tr("Архивное значение"), tr("Текущее значение")
        headers = {"changes": ["UUID", tr("Поле"), before, after],
                   "parameters": [tr("Параметр"), before, after, "Δ"],
                   "curve": ["X", before, after, "ΔY"]}[mode]
        old = self.archive_diff_table.model()
        self.archive_diff_table.setModel(ResultsModel(headers, diff[mode], self.archive_diff_table))
        if old is not None:
            old.deleteLater()

    def jump_to_changed_sample(self, index):
        if index.isValid() and self.archive_diff_choice.currentData() == "changes":
            self.select_source_sample(self.archive_diff_table.model().rows[index.row()][0])

    def display_bootstrap_parameters(self):
        bootstrap = self.bootstrap_result
        if bootstrap is None:
            return
        key = id(bootstrap), self.confidence.value()
        if key == self._parameter_cache:
            return
        self._parameter_cache = key
        records = bootstrap.parameter_intervals(self.result.model, confidence=self.confidence.value()/100,
                                                bounds=self.result.bounds)
        rows = [[d[k] for k in ("parameter", "median", "lower", "upper", "sd", "bound_frequency")] for d in records]
        old = self.parameter_table.model()
        self.parameter_table.setModel(ResultsModel(
            [tr("Параметр"), tr("Медиана"), tr("Нижняя граница"), tr("Верхняя граница"), "Bootstrap SD",
             tr("Доля у границы")], rows, self.parameter_table))
        if old is not None:
            old.deleteLater()
        self.parameter_table.setCurrentIndex(self.parameter_table.model().index(0, 0))
        self.draw_parameter_histogram()

    def draw_parameter_histogram(self, *args):
        if self.bootstrap_result is None:
            return
        from autopyro.curves.curve_constraints import parameter_samples
        samples = parameter_samples(self.result.model, self.bootstrap_result.parameters)
        index = max(0, self.parameter_table.currentIndex().row())
        name = list(samples)[index]
        self.parameter_figure.clear()
        axis = self.parameter_figure.subplots()
        axis.hist(samples[name], bins=min(40, max(5, int(np.sqrt(len(samples[name]))))))
        axis.set_xlabel(name)
        axis.set_ylabel(tr("Число повторов"))
        self.parameter_canvas.draw_idle()

    def show_oof_diagnostics(self, index=None):
        result = self.comparison_result
        if result is None:
            return
        if result.nested is not None:
            predictions = result.nested.predictions
            if index is None and np.any(np.isfinite(predictions)):
                self._diagnostic_source = (result.y, predictions, self.row_ids, True)
                self._diagnostic_drawn = False
                self.refresh_diagnostic_view()
            return
        if not result.ranking:
            return
        scores = [*result.ranking, *(s for s in result.scores if s.errors)]
        row = index.row() if index is not None else 0
        score = scores[row]
        if score.predictions is not None:
            self._diagnostic_source = (result.y, score.predictions, self.row_ids, True)
            self._diagnostic_drawn = False
            self.refresh_diagnostic_view()

    def apply_current_fit(self):
        if self.result is None or self._archive_records is not None or (self.worker is not None and self.worker.isRunning()):
            return
        from copy import deepcopy
        from autopyro.model.core import Column, Row, Project
        from autopyro.curves.curve_outputs import fit_source_ids, prepare_output_columns
        from ..table.commands import AddColumnCommand
        result, bootstrap, settings = self.result, self.bootstrap_result, deepcopy(self.fit_settings)
        row_ids, generation = list(self.row_ids), self.generation
        project = self.owner.model.project
        targets = set(self.owner.analysis_scope.rows())
        wanted = targets | set(row_ids)
        sources = fit_source_ids(settings, bootstrap)
        data = Project(columns=[Column(id=c.id, name=c.name, kind=c.kind, unit=c.unit)
                                for c in project.columns if c.id in sources],
                       rows=[Row(id=r.id, values={cid: deepcopy(r.values.get(cid)) for cid in sources})
                             for r in project.rows if r.id in wanted])
        rows = [r for r in data.rows if r.id in targets]
        confidence = self.confidence.value()/100
        labels = dict(prediction=tr("Прогноз Y"), residual=tr("Остаток Y"),
                      confidence_lower=tr("Confidence: нижняя граница"), confidence_upper=tr("Confidence: верхняя граница"),
                      prediction_lower=tr("Prediction: нижняя граница"), prediction_upper=tr("Prediction: верхняя граница"))
        prefix = tr("Модель")

        def commit(payload):
            if generation != self.generation:
                return
            record, columns = payload
            names = {c.name for c in project.columns}
            commands = []
            for column, values in columns:
                original, counter = column.name, 2
                while column.name in names:
                    column.name = f"{original} ({counter})"
                    counter += 1
                names.add(column.name)
                commands.append(AddColumnCommand(self.owner.model, column, values))
            project.view.setdefault("curve_fit_runs", {})[record["id"]] = record
            self.owner.model.undo.beginMacro(tr("Колонки подгонки кривой"))
            try:
                for command in commands:
                    self.owner.model.undo.push(command)
            finally:
                self.owner.model.undo.endMacro()
            self.status.setText(tr("Создано колонок: {0}", len(columns)))

        self.run_task(lambda: prepare_output_columns(data, rows, result, settings, row_ids,
                                                     bootstrap=bootstrap, confidence=confidence, labels=labels, prefix=prefix), commit)

    def draw_diagnostics(self, observed, predicted, row_ids, *, oof=False):
        self.clear_diagnostics()
        valid = np.isfinite(predicted)
        observed, predicted = np.asarray(observed)[valid], np.asarray(predicted)[valid]
        ids = np.asarray(row_ids, dtype=object)[valid]
        residual = observed-predicted
        left, right = self.diagnostic_figure.subplots(1, 2)
        if oof:
            points = left.scatter(observed, predicted, s=12, picker=5, rasterized=True)
            limits = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
            left.plot(limits, limits, color="gray")
            left.set_xlabel(tr("Фактическое Y"))
            left.set_ylabel(tr("Отложенный прогноз Y"))
            other = right.scatter(predicted, residual, s=10, picker=5, rasterized=True)
            self.diagnostic_artists[other] = ids.tolist()
            right.axhline(0, color="gray")
            right.set_xlabel(tr("Прогноз Y"))
            right.set_ylabel(tr("Остаток Y"))
        else:
            points = left.scatter(predicted, residual, s=12, picker=5, rasterized=True)
            left.axhline(0, color="gray")
            left.set_xlabel(tr("Прогноз Y"))
            left.set_ylabel(tr("Остаток Y"))
            (theoretical, ordered), (slope, intercept, _) = probplot(residual)
            other = right.scatter(theoretical, ordered, s=10, picker=5, rasterized=True)
            self.diagnostic_artists[other] = ids[np.argsort(residual, kind="stable")].tolist()
            right.plot(theoretical, slope*theoretical+intercept, color="gray")
            right.set_xlabel(tr("Нормальные квантили"))
            right.set_ylabel(tr("Квантили остатков"))
        self.diagnostic_artists[points] = ids.tolist()
        center = np.median(residual)
        scale = 1.4826*np.median(np.abs(residual-center))
        if scale == 0:
            scale = np.std(residual)
        suspect = np.abs(residual-center) > 3.5*scale if scale > 0 else np.zeros(len(residual), bool)
        order = np.argsort(-np.abs(residual), kind="stable")
        self.diagnostic_rows = ids[order].tolist()
        rows = [[str(ids[i]), float(observed[i]), float(predicted[i]), float(residual[i]),
                 tr("Проверить") if suspect[i] else ""] for i in order]
        self.residual_table.setModel(ResultsModel(
            ["UUID", tr("Фактическое Y"), tr("Прогноз Y"), tr("Остаток Y"), tr("Диагностика")], rows, self.residual_table))
        self.diagnostic_canvas.draw_idle()

    def jump_to_sample(self, *args):
        index = self.residual_table.currentIndex()
        if index.isValid() and index.row() < len(self.diagnostic_rows):
            self.select_source_sample(self.diagnostic_rows[index.row()])

    def pick_sample(self, event):
        ids = self.diagnostic_artists.get(event.artist)
        if ids is not None and len(event.ind):
            self.select_source_sample(ids[int(event.ind[0])])

    def announce_status(self, text):
        self.status.setText(text)
        if QAccessible.isActive():
            QAccessible.updateAccessibility(QAccessibleAnnouncementEvent(self.status, text))

    def select_source_sample(self, row_id):
        if not select_row_ids(self.owner, [row_id], preserve_if_empty=True):
            self.announce_status(tr("Проба скрыта фильтром исходной таблицы"))
            return
        table = self.owner.table
        selection = table.selectionModel()
        selected = selection.selectedRows()
        if selected:
            self.owner.desktop.open_table()
            selection.setCurrentIndex(selected[0], QItemSelectionModel.SelectionFlag.NoUpdate)
            table.scrollTo(selected[0])
            table.setFocus()

    def refresh_saved_results(self):
        records = self.owner.model.project.view.get("curve_fit_results", [])
        signature = tuple((r["id"], r["name"], r.get("stale", False)) for r in records)
        if signature == self._saved_signature:
            return
        self._saved_signature = signature
        selected = {i.data(Qt.ItemDataRole.UserRole) for i in self.saved_results.selectedItems()}
        self.saved_results.clear()
        for record in records:
            label = record["name"] + (" · "+tr("Устарел") if record.get("stale") else "")
            item = QListWidgetItem(icon("chart-no-axes-combined"), label)
            item.setData(Qt.ItemDataRole.UserRole, record["id"])
            self.saved_results.addItem(item)
            item.setSelected(record["id"] in selected)

    def selected_records(self):
        ids = {i.data(Qt.ItemDataRole.UserRole) for i in self.saved_results.selectedItems()}
        return [r for r in self.owner.model.project.view.get("curve_fit_results", []) if r["id"] in ids]

    def save_current_result(self):
        if self.result is None or self._archive_records is not None or (self.worker is not None and self.worker.isRunning()):
            return
        from copy import deepcopy
        from autopyro.model.core import new_id, timestamp, Project, Column, Row
        from autopyro.curves.curve_outputs import fit_source_ids, source_digest
        from autopyro.curves.curve_archive import fit_payload
        name, accepted = QInputDialog.getText(self, tr("Сохранение результата"), tr("Название"))
        if not accepted or not name.strip() or self.result is None:
            return
        result, bootstrap = self.result, self.bootstrap_result
        axis_labels = list(self.axis_labels)
        error, settings, row_ids = self.bootstrap_error, deepcopy(self.fit_settings), list(self.row_ids)
        settings["scope"] = self.scope.currentData()
        confidence = self.confidence.value()/100
        grid, bands = self.band_data()
        generation = self.generation
        sources = fit_source_ids(settings, bootstrap)
        wanted = set(row_ids)
        project = self.owner.model.project
        columns = {column.id: column for column in project.columns}
        axis_units = [columns[settings[key]].unit for key in ('x', 'y')]
        data = Project(columns=[Column(id=c.id, name=c.name, kind=c.kind, unit=c.unit)
                                for c in project.columns if c.id in sources],
                       rows=[Row(id=r.id, values={cid: deepcopy(r.values.get(cid)) for cid in sources})
                             for r in project.rows if r.id in wanted])

        def prepare():
            record = fit_payload(result, settings, row_ids, grid, bands, bootstrap=bootstrap,
                                 bootstrap_error=error, confidence=confidence)
            record.update(id=new_id(), name=name.strip(), at=timestamp(), source_ids=sources,
                          source_rows=row_ids, source_digest=source_digest(data, sources, row_ids), source_digest_version=2, stale=False,
                          axis_labels=axis_labels, axis_units=axis_units)
            return record

        def save(record):
            if generation != self.generation:
                return
            project.view.setdefault("curve_fit_results", []).append(record)
            self.owner.view_changed()
            self.refresh_saved_results()
            self.saved_results.item(self.saved_results.count()-1).setSelected(True)

        self.run_task(prepare, save)

    def restore_result_settings(self):
        records = self.selected_records()
        if len(records) != 1:
            self.status.setText(tr("Выберите один сохранённый результат"))
            return
        self.restore_settings(records[0]["settings"])
        self.confidence.setValue(records[0]["confidence"]*100)
        if records[0]["settings"].get("scope"):
            self.owner.analysis_scope.set_mode(records[0]["settings"]["scope"])
        self.invalidate()

    def replay_saved_result(self):
        if self.task_running:
            return
        records = self.selected_records()
        if len(records) != 1:
            self.status.setText(tr("Выберите один сохранённый результат"))
            return
        self.calculate(replay=records[0])

    def show_saved_curves(self):
        records = self.selected_records()
        if not records:
            self.status.setText(tr("Выберите сохранённые результаты"))
            return
        if len({(r["settings"]["x"], r["settings"]["y"]) for r in records}) != 1:
            self.status.setText(tr("Для наложения кривых выберите одинаковые показатели X и Y"))
            return
        self._archive_records = list(records)
        self.confidence.setEnabled(False)
        self.export_button.setEnabled(False)
        self.apply_fit_button.setEnabled(False)
        self.save_fit_button.setEnabled(False)
        bands_available = any(r.get("bands") for r in records)
        self.show_confidence.setEnabled(bands_available)
        self.show_prediction.setEnabled(bands_available)
        self.draw_saved_curves(records)

    def draw_saved_curves(self, records):
        self.figure.clear()
        axis = self.figure.subplots()
        for record in records:
            label = record["name"]+(" · "+tr("Устарел") if record.get("stale") else "")
            if record.get("bands"):
                label += f" · {100*record['confidence']:g}%"
                if (record.get("bootstrap") or {}).get("confidence_method") == "grid_max_deviation":
                    label += " · " + tr("Одновременная доверительная полоса")
            line, = axis.plot(record["grid"], record["curve"], label=label)
            if record.get("bands") and self.show_prediction.isChecked():
                b = record["bands"]
                axis.fill_between(record["grid"], b["prediction_lower"], b["prediction_upper"], color=line.get_color(), alpha=.06)
            if record.get("bands") and self.show_confidence.isChecked():
                b = record["bands"]
                axis.fill_between(record["grid"], b["confidence_lower"], b["confidence_upper"], color=line.get_color(), alpha=.15)
        names = records[0].get("axis_labels", ["X", "Y"])
        axis.set(xlabel=names[0], ylabel=names[1])
        axis.legend()
        self.canvas.draw_idle()
        self.export_plot_button.setEnabled(True)
        self.status.setText(tr("Архивные кривые: полосы используют сохранённые уровни доверия"))
        self.tabs.setCurrentWidget(self.plot_page)

    def remove_saved_results(self):
        ids = {r["id"] for r in self.selected_records()}
        if ids:
            project = self.owner.model.project
            project.view["curve_fit_results"] = [r for r in project.view["curve_fit_results"] if r["id"] not in ids]
            plot = self.owner.plot_window
            if plot is not None and plot.spec.curve_archive_id in ids:
                plot.schedule()
            self.owner.view_changed()
            self.refresh_saved_results()
            if self._archive_records is not None and any(r["id"] in ids for r in self._archive_records):
                remaining = [r for r in self._archive_records if r["id"] not in ids]
                if remaining:
                    self._archive_records = remaining
                    self.draw_saved_curves(remaining)
                elif self.result is not None:
                    self.show_current_fit()
                else:
                    self._archive_records = None
                    self.confidence.setEnabled(True)
                    self.figure.clear()
                    self.canvas.draw_idle()
                    self.export_plot_button.setEnabled(False)

    def export_fit_report(self):
        if self.worker is not None and self.worker.isRunning():
            return
        records = self.selected_records()
        comparison = self.comparison_result
        if not records and comparison is None:
            self.status.setText(tr("Выберите сохранённые результаты"))
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Отчёт по моделям"), "curve_report.html", "HTML (*.html)")
        if not path:
            return
        from copy import deepcopy
        from autopyro.curves.curve_archive import report_html
        from autopyro.data_io.file_output import atomic_output
        labels = dict(title=tr("Подгонка кривых"), parameter=tr("Параметр"), estimate=tr("Оценка"), holdout=tr("Независимый тест"),
                      holdout_pair=tr("Попарный тест"),
                      nested_cv=tr("Вложенная CV"), outer_fold=tr("Внешний блок"), selected_model=tr("Выбранная модель"),
                      inner_rmse=tr("Внутренний RMSE"), outer_rmse=tr("Внешний RMSE"), outer_mae=tr("Внешний MAE"),
                      fold_errors=tr("Ошибки блоков"),
                      train_groups=tr("Групп в обучении"), test_groups=tr("Групп в проверке"),
                      train_periods=tr("Моментов в обучении"), test_periods=tr("Моментов в проверке"),
                      gap_rows=tr("Строк в разрыве"),
                      metric=tr("Метрика"), value=tr("Значение"), stale=tr("Устарел"), current=tr("Актуален"),
                      prediction=tr("Прогноз Y"), residual=tr("Остаток Y"), normal=tr("Нормальные квантили"),
                      cv=tr("Кросс-валидация"), model=tr("Модель"), observed=tr("Фактическое Y"),
                      oof=tr("Отложенный прогноз Y"), bound=tr("Доля у границы"),
                      lower=tr("Нижняя граница"), upper=tr("Верхняя граница"))
        if comparison is not None and comparison.holdout is not None and comparison.holdout.pair is not None:
            labels["holdout_interval_note"] = _pair_interval_note(comparison.holdout.pair)
        records = deepcopy(records)
        for record in records:
            record["warnings"] = [message(w) for w in record.get("warnings", [])]
            if record.get("bootstrap"):
                record["bootstrap"]["warnings"] = [message(w) for w in record["bootstrap"].get("warnings", [])]

        def export():
            text = report_html(records, comparison=comparison, labels=labels)
            with atomic_output(path) as temporary:
                temporary.write_text(text, encoding="utf-8")

        self.run_task(export, lambda _: self.status.setText(tr("Отчёт сохранён")))
