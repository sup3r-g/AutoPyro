"""Controls and results for the joint curve-fitting workflow."""
import json
from threading import Event

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QLabel, QListWidget,
                               QHBoxLayout, QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QStyle, QTableView,
                               QToolButton, QVBoxLayout, QWidget)

from autopyro.curves.joint_fitting import fit_joint_curve
from autopyro.curves.curve_bootstrap import BOOTSTRAP_BAND_METHODS, BootstrapCancelled
from autopyro.curves.joint_bootstrap import bootstrap_joint_fit
from autopyro.curves.joint_validation import compare_joint_sharing
from autopyro.curves.curve_numerics import curve_axis_scale
from autopyro.data_io.file_output import write_json
from autopyro.model.core import Column, new_id, timestamp
from autopyro.model.tabular import excluded_sample_ids, project_frame
from autopyro.curves.joint_sources import joint_source_digest, source_schema
from autopyro.reports.joint_fit_figures import capture_joint_plots
from ..i18n import message, tr
from ..table.commands import AddColumnCommand
from ..theme import icon
from ..widgets.documents import parameter_page, refill_combo
from ..widgets.figure_canvas import FittedFigureCanvas
from ..widgets.results import ResultsModel
from .joint_bounds import JointBoundsDialog
from .joint_band_cache import JointBandCache
from .joint_diagnostics import JointPlots
from .joint_exclusions import JointExclusionModel, prepare_joint_rows


def _series_bound_labels(overrides, series):
    if not overrides:
        return overrides
    labels = {}
    for value in dict.fromkeys(series):
        labels.setdefault(str(value), []).append(value)
    resolved = {}
    for label, limits in overrides.items():
        matches = labels.get(label, [])
        if len(matches) != 1:
            raise ValueError("В границах серий указано неизвестное или неоднозначное имя серии")
        resolved[matches[0]] = limits
    return resolved


class JointCurveTools:
    def build_joint_controls(self, model_options):
        self._joint_model_options = model_options
        self.joint_result = None
        self.joint_bootstrap = None
        self._joint_band_cache = JointBandCache(self)
        self._joint_band_cache.ready.connect(self.show_joint_bands)
        self._joint_band_view = None
        self._joint_band_error_status = None
        self._joint_curve_error_status = None
        self._joint_cancel = Event()
        self._joint_cancel.set()
        controls = parameter_page(self.parameter_tabs, tr("Совместная подгонка"), 'layers')
        form = QFormLayout()
        controls.addLayout(form)
        self.joint_column = QComboBox()
        self.joint_parameters = QListWidget()
        self.joint_parameters.setMaximumHeight(160)
        self.joint_bounds_button = QPushButton(icon("settings-2"), tr("Изменить границы по сериям…"))
        self.joint_bounds_button.clicked.connect(self.edit_joint_bounds)
        form.addRow(tr("Колонка серий"), self.joint_column)
        form.addRow(tr("Общие коэффициенты"), self.joint_parameters)
        form.addRow(tr("Границы по сериям"), self.joint_bounds_button)
        note = QLabel(tr("Отметьте общие коэффициенты. Остальные будут рассчитаны отдельно для каждой серии. X, Y, σY и модель берутся с вкладки «Параметры»."))
        note.setWordWrap(True)
        controls.addWidget(note)
        bootstrap_note = QLabel(tr("Bootstrap совместной подгонки пересэмплирует точки внутри каждой серии. Группы и блоки bootstrap относятся к обычной подгонке."))
        bootstrap_note.setWordWrap(True)
        controls.addWidget(bootstrap_note)
        ordinary_only = tr("Несколько начальных приближений и абсолютные погрешности доступны только в обычной подгонке.")
        note = QLabel(ordinary_only)
        note.setWordWrap(True)
        controls.addWidget(note)
        for widget in (self.starts, self.start_seed, self.absolute_sigma):
            widget.setToolTip(ordinary_only)
        self.joint_band_note = QLabel()
        self.joint_band_note.setWordWrap(True)
        controls.addWidget(self.joint_band_note)
        self.joint_compatibility = QLabel()
        self.joint_compatibility.setWordWrap(True)
        self.joint_compatibility_icon = QLabel()
        self.joint_compatibility_icon.setPixmap(icon("settings-2").pixmap(16, 16))
        compatibility = QHBoxLayout()
        compatibility.addWidget(self.joint_compatibility_icon)
        compatibility.addWidget(self.joint_compatibility, 1)
        controls.addLayout(compatibility)
        self.joint_run_button = QPushButton(icon('layers'), tr("Подогнать серии"))
        self.joint_run_button.clicked.connect(self.calculate_joint)
        controls.addWidget(self.joint_run_button)
        self.joint_cv_folds = QSpinBox()
        self.joint_cv_folds.setRange(2, 20)
        self.joint_cv_folds.setValue(5)
        self.joint_cv_folds.valueChanged.connect(self.clear_joint_comparison)
        form.addRow(tr("Блоки CV по сериям"), self.joint_cv_folds)
        cv_note = QLabel(tr("CV сравнивает текущий набор общих коэффициентов с вариантами по одному общему коэффициенту. Проверяются прогнозы для известных серий."))
        cv_note.setText(cv_note.text()+" "+tr("В рейтинге RMSE каждая отложенная точка имеет одинаковый вес."))
        cv_note.setWordWrap(True)
        controls.addWidget(cv_note)
        self.joint_cv_button = QPushButton(icon("files"), tr("Сравнить варианты совместной подгонки"))
        self.joint_cv_button.clicked.connect(lambda: self.calculate_joint(comparison=True))
        controls.addWidget(self.joint_cv_button)
        self.joint_cv_cancel_button = QPushButton(icon("x"), tr("Отменить сравнение"))
        self.joint_cv_cancel_button.setEnabled(False)
        self.joint_cv_cancel_button.clicked.connect(self.cancel_joint_comparison)
        self.joint_cv_folds.valueChanged.connect(self.cancel_joint_comparison)
        self.joint_cv_progress.connect(self.show_joint_cv_progress)
        controls.addWidget(self.joint_cv_cancel_button)
        self.joint_export_button = QPushButton(icon('file-text'), tr("Экспорт совместной подгонки JSON…"))
        self.joint_export_button.setEnabled(False)
        self.joint_export_button.clicked.connect(self.export_joint)
        controls.addWidget(self.joint_export_button)
        warning = QHBoxLayout()
        self.joint_snapshot_icon = QLabel()
        self.joint_snapshot_icon.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning).pixmap(16, 16))
        self.joint_snapshot_warning = QLabel()
        self.joint_snapshot_warning.setWordWrap(True)
        warning.addWidget(self.joint_snapshot_icon)
        warning.addWidget(self.joint_snapshot_warning, 1)
        controls.addLayout(warning)
        self.owner.model.changed.connect(self.refresh_joint_snapshot_warning)
        self.refresh_joint_snapshot_warning()
        controls.addStretch()
        stored = self.owner.model.project.view.get("joint_curve_fitting", {})
        self._joint_saved_shared = stored.get("shared")
        self._joint_bounds_text = stored.get("series_bounds", "")
        self.sync_joint_columns(stored.get("column"))
        self.joint_column.currentIndexChanged.connect(self.clear_joint_fit)
        self.joint_parameters.itemChanged.connect(self.clear_joint_fit)
        self.method.currentIndexChanged.connect(self.update_joint_controls)
        self.parameter_rules.textChanged.connect(self.update_joint_controls)
        self.band_method.currentIndexChanged.connect(self.update_joint_controls)
        self.update_joint_controls()

    def joint_shared_parameters(self):
        return tuple(self.joint_parameters.item(i).text() for i in range(self.joint_parameters.count())
                     if self.joint_parameters.item(i).checkState() == Qt.CheckState.Checked)

    def edit_joint_bounds(self):
        if self.task_running:
            return
        try:
            series_id = self.joint_column.currentData()
            if not series_id:
                raise ValueError("Выберите колонку серий")
            model, _, bounds = self._joint_model_options(self.settings())
            shared = self.joint_shared_parameters()
            lower, upper = (np.broadcast_to(side, (len(model.parameters),)) for side in bounds)
            if np.any(np.isnan(lower) | np.isnan(upper) | (lower >= upper)):
                raise ValueError("Некорректные границы параметров")
            defaults = {name: [float(lo) if np.isfinite(lo) else None, float(hi) if np.isfinite(hi) else None]
                        for name, lo, hi in zip(model.parameters, lower, upper) if name not in shared}
            frame = project_frame(self.owner.model.project, self.owner.analysis_scope.rows(), [series_id])
            labels = [str(value) for value in frame[series_id].dropna().unique() if str(value).strip()]
            if len(set(labels)) != len(labels):
                raise ValueError("В границах серий указана неизвестная или неоднозначная серия")
        except (ValueError, TypeError) as exc:
            self.status.setText(message(exc))
            return
        notice = ""
        try:
            overrides = self.read_joint_bounds()
            editable = {}
            for label, limits in overrides.items():
                if not isinstance(limits, dict):
                    raise ValueError
                for name, interval in limits.items():
                    if name not in defaults:
                        notice = tr("Сохранение удалит границы коэффициентов, которые больше не являются отдельными.")
                        continue
                    if not isinstance(interval, list) or len(interval) != 2:
                        raise ValueError
                    lower, upper = (None if value is None else float(value) for value in interval)
                    editable.setdefault(label, {})[name] = [None if lower == -np.inf else lower,
                                                            None if upper == np.inf else upper]
        except (ValueError, TypeError):
            overrides, editable = None, {}
            notice = tr("Сохранённые границы не удалось прочитать. Подтвердите изменения, чтобы заменить их.")
        if not defaults or not (labels or editable):
            self.status.setText(tr("Для редактора нужны серии и отдельные коэффициенты."))
            return
        editor = JointBoundsDialog(self, labels, defaults, editable, notice)
        if editor.exec() != QDialog.DialogCode.Accepted or editor.overrides == overrides:
            return
        self._joint_bounds_text = json.dumps(editor.overrides, ensure_ascii=False) if editor.overrides else ""
        project = self.owner.model.project
        project.view["joint_curve_fitting"] = {
            "column": series_id, "shared": list(shared), "series_bounds": self._joint_bounds_text,
        }
        project.view["curve_fitting"] = self.settings()
        self.owner.view_changed()
        self.clear_joint_fit()

    def read_joint_bounds(self):
        try:
            overrides = json.loads(self._joint_bounds_text) if self._joint_bounds_text else {}
        except json.JSONDecodeError as exc:
            raise ValueError("Границы серий должны быть объектом JSON") from exc
        if not isinstance(overrides, dict):
            raise ValueError("Границы серий должны быть объектом JSON")
        return overrides

    def joint_compatibility_error(self):
        if self.method.currentData() != "ols" or self.parameter_rules.text().strip():
            return tr("Для совместной подгонки выберите OLS/WLS без связей коэффициентов")
        return ""

    def update_joint_controls(self, *args):
        error = self.joint_compatibility_error()
        self.joint_run_button.setEnabled(not error)
        self.joint_cv_button.setEnabled(not error)
        self.joint_run_button.setToolTip(error)
        self.joint_compatibility.setText(error)
        self.joint_compatibility.setVisible(bool(error))
        self.joint_compatibility_icon.setVisible(bool(error))
        method = self.band_method.currentData()
        if method == "analytic":
            note = tr("Аналитические полосы недоступны для совместной подгонки. Выберите bootstrap.")
        elif method == "auto":
            note = tr("В режиме «Авто» совместная подгонка строит полосы при робастной регрессии или активных границах. Для остальных расчётов выберите bootstrap.")
        else:
            note = ""
        self.joint_band_note.setText(note)
        self.joint_band_note.setVisible(bool(note))

    def refresh_joint_snapshot_warning(self):
        snapshot = self.owner.model.project.view.get("joint_curve_report")
        detail = ""
        if snapshot:
            status = snapshot.get("source_status", "unverified")
            if status == "stale":
                detail = tr("Снимок совместной подгонки устарел. Исходные данные или исключения QC изменились.")
            elif status == "unverified":
                detail = tr("Актуальность снимка совместной подгонки не подтверждена.")
        self.joint_snapshot_warning.setText(detail)
        self.joint_snapshot_warning.setVisible(bool(detail))
        self.joint_snapshot_icon.setVisible(bool(detail))

    def build_joint_view(self):
        self.joint_page = QWidget()
        layout = QVBoxLayout(self.joint_page)
        self.joint_summary = QLabel()
        self.joint_summary.setWordWrap(True)
        self.joint_summary.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.joint_summary)
        self.joint_tail_warning = QLabel()
        self.joint_tail_warning.setWordWrap(True)
        self.joint_tail_warning.hide()
        self.confidence.valueChanged.connect(self.update_joint_tail_warning)
        layout.addWidget(self.joint_tail_warning)
        self.joint_series = QComboBox()
        self.joint_series.setAccessibleName(tr("Серия на графике"))
        self.joint_series.setToolTip(tr("Серия на графике"))
        self.joint_series.currentIndexChanged.connect(self.select_joint_series)
        layout.addWidget(self.joint_series)
        actions = QHBoxLayout()
        layout.addLayout(actions)
        self.joint_columns_button = self.joint_action(
            actions, "columns-3", tr("Создать колонки прогноза и остатков"), self.apply_joint_fit)
        self.joint_report_button = self.joint_action(
            actions, "file-text", tr("Сохранить совместную подгонку для отчёта"), self.save_joint_for_report)
        self.joint_exclusion_button = self.joint_action(
            actions, "table-2", tr("Исключённые строки"), self.focus_joint_exclusions)
        actions.addStretch()
        self.joint_figure = Figure(layout="constrained")
        self.joint_canvas = FittedFigureCanvas(self.joint_figure)
        self.joint_plots = JointPlots(self.joint_canvas)
        self.joint_plots.currentChanged.connect(self.draw_joint_series)
        self.joint_plots.source_selected.connect(self.select_source_sample)
        layout.addWidget(self.joint_plots, 2)
        self.joint_table = QTableView()
        self.joint_table.setAccessibleName(tr("Коэффициенты и метрики"))
        self.joint_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.joint_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        layout.addWidget(self.joint_table, 1)
        self.joint_tab = self.tabs.addTab(self.joint_page, icon("layers"), tr("Совместная подгонка"))
        self.tabs.setTabEnabled(self.joint_tab, False)
        page = QWidget()
        layout = QVBoxLayout(page)
        self.joint_exclusion_table = QTableView()
        self.joint_exclusion_table.setAccessibleName(tr("Исключённые строки"))
        self.joint_exclusion_table.setAccessibleDescription(
            tr("Нажмите Enter, чтобы перейти к выбранной пробе в исходной таблице."))
        self.joint_exclusion_table.horizontalHeader().setStretchLastSection(True)
        self.joint_exclusion_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.joint_exclusion_table.activated.connect(self.open_joint_excluded_row)
        layout.addWidget(self.joint_exclusion_table)
        self.joint_exclusion_open = QPushButton(icon("table-2"), tr("Показать выбранную пробу"))
        self.joint_exclusion_open.setEnabled(False)
        self.joint_exclusion_open.clicked.connect(lambda: self.open_joint_excluded_row(self.joint_exclusion_table.currentIndex()))
        layout.addWidget(self.joint_exclusion_open)
        self.joint_exclusion_tab = self.tabs.addTab(page, icon("table-2"), tr("Исключённые строки"))
        self.tabs.setTabEnabled(self.joint_exclusion_tab, False)
        self.joint_exclusions = None
        self.joint_cv_table = QTableView()
        self.joint_cv_table.setAccessibleName(tr("CV совместной подгонки"))
        self.joint_cv_tab = self.tabs.addTab(self.joint_cv_table, icon("files"), tr("CV совместной подгонки"))
        self.tabs.setTabEnabled(self.joint_cv_tab, False)
        self.tabs.currentChanged.connect(self.show_joint_exclusions)
        self.confidence.valueChanged.connect(self.draw_joint_series)
        self.show_confidence.toggled.connect(self.draw_joint_series)
        self.show_prediction.toggled.connect(self.draw_joint_series)

    def joint_action(self, layout, icon_name, label, callback):
        button = QToolButton()
        button.setIcon(icon(icon_name))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setText(label)
        button.setToolTip(label)
        button.setAccessibleName(label)
        button.setEnabled(False)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def show_joint_exclusions(self, tab):
        table = self.joint_exclusion_table
        if tab != self.joint_exclusion_tab or self.joint_exclusions is None or table.model() is not None:
            return
        table.setModel(JointExclusionModel(*self.joint_exclusions, parent=table))
        table.setColumnWidth(0, 120)
        table.selectionModel().currentChanged.connect(
            lambda current, _: self.joint_exclusion_open.setEnabled(current.isValid()))

    def focus_joint_exclusions(self):
        self.tabs.setCurrentIndex(self.joint_exclusion_tab)
        table = self.joint_exclusion_table
        if not table.currentIndex().isValid() and table.model() is not None:
            table.setCurrentIndex(table.model().index(0, 0))
        table.setFocus()

    def open_joint_excluded_row(self, index):
        if index.isValid() and self.joint_exclusions is not None:
            self.select_source_sample(self.joint_exclusions[0][index.row()])

    def select_joint_series(self, *args):
        series = self.joint_series.currentData()
        if self.joint_result is None or series is None:
            return
        table = self.joint_table
        with QSignalBlocker(table.selectionModel()):
            table.selectRow(series)
        self.joint_plots.select_series(series)
        self.draw_joint_series()

    def sync_joint_columns(self, selected=None):
        selected = selected or self.joint_column.currentData()
        choices = [(tr("Выберите колонку серий"), None)]
        choices.extend((column.name, column.id) for column in self.owner.model.project.columns
                       if not column.metadata.get("method") and not column.metadata.get("curve_fit"))
        refill_combo(self.joint_column, choices)
        self.joint_column.setCurrentIndex(max(0, self.joint_column.findData(selected)))

    def refresh_joint_parameters(self, model):
        names = model.parameters if model is not None and len(model.parameters) > 1 else ()
        old_names = tuple(self.joint_parameters.item(i).text() for i in range(self.joint_parameters.count()))
        if names == old_names:
            return
        shared = self._joint_saved_shared if self._joint_saved_shared is not None else names[:1]
        if names:
            self._joint_saved_shared = None
        with QSignalBlocker(self.joint_parameters):
            self.joint_parameters.clear()
            for name in names:
                item = QListWidgetItem(name, self.joint_parameters)
                item.setCheckState(Qt.CheckState.Checked if name in shared else Qt.CheckState.Unchecked)

    def clear_joint_fit(self, *args):
        self.clear_joint_comparison()
        had_result = self.joint_result is not None or not self._joint_cancel.is_set()
        self._joint_cancel.set()
        if had_result:
            self.status.setText(tr("Настройки совместной подгонки изменились. Нажмите «Подогнать серии»."))
        if self.joint_exclusions is not None:
            self.joint_exclusions = None
            self.tabs.setTabEnabled(self.joint_exclusion_tab, False)
            self.joint_exclusion_button.setEnabled(False)
            self.joint_exclusion_open.setEnabled(False)
            table = self.joint_exclusion_table
            old = table.model()
            table.setModel(None)
            if old is not None:
                old.deleteLater()
        if self.joint_result is None:
            return
        self.joint_result = None
        self.joint_bootstrap = None
        self._joint_band_cache.clear()
        self._joint_band_view = None
        self._joint_band_error_status = None
        self._joint_curve_error_status = None
        self.joint_export_button.setEnabled(False)
        self.joint_columns_button.setEnabled(False)
        self.joint_report_button.setEnabled(False)
        self.tabs.setTabEnabled(self.joint_tab, False)
        self.joint_summary.clear()
        old = self.joint_table.model()
        self.joint_table.setModel(None)
        if old is not None:
            old.deleteLater()
        self.joint_figure.clear()
        self.joint_plots.clear()

    def clear_joint_comparison(self, *args):
        if not hasattr(self, "joint_cv_table"):
            return
        old = self.joint_cv_table.model()
        if old is not None:
            self.joint_cv_table.setModel(None)
            old.deleteLater()
            self.tabs.setTabEnabled(self.joint_cv_tab, False)

    def show_joint_comparison(self, scores):
        metrics = ("RMSE", "MAE", "MedAE", "R2")
        rows = []
        for score in scores:
            values = [(score.metrics or {}).get(key) for key in metrics]
            rows.append([", ".join(score.shared),
                         *(value if value is not None and np.isfinite(value) else None for value in values),
                         message(score.error)])
        self.joint_cv_table.setModel(ResultsModel(
            [tr("Общие коэффициенты"), *metrics, tr("Ошибка")], rows, self.joint_cv_table))
        self.tabs.setTabEnabled(self.joint_cv_tab, True)
        self.tabs.setCurrentIndex(self.joint_cv_tab)
        successful = sum(score.metrics is not None for score in scores)
        status = tr("Вариантов совместной подгонки проверено: {0}", len(scores))
        outcome = (tr("Успешно: {0} · С ошибками: {1}", successful, len(scores)-successful)
                   if successful else tr("Рейтинг недоступен: все варианты завершились с ошибкой."))
        self.announce_status(f"{status}\n{outcome}")

    def cancel_joint_comparison(self, *args):
        if self.joint_cv_cancel_button.isEnabled():
            self._joint_cancel.set()
            self.joint_cv_cancel_button.setEnabled(False)
            self.announce_status(tr("Сравнение отменено"))

    def show_joint_cv_progress(self, cancel, done, total):
        if cancel is self._joint_cancel and not cancel.is_set():
            self.status.setText(tr("Совместная CV: блок {0} из {1}", done, total))

    def calculate_joint(self, *, comparison=False):
        if self.task_running:
            return
        if error := self.joint_compatibility_error():
            self.status.setText(error)
            return
        settings = self.settings()
        series_id = self.joint_column.currentData()
        shared = self.joint_shared_parameters()
        bounds_text = self._joint_bounds_text
        try:
            model, initial, bounds = self._joint_model_options(settings)
            overrides = self.read_joint_bounds()
            if not shared or len(shared) == len(model.parameters):
                raise ValueError("Выберите хотя бы один общий и один отдельный коэффициент")
            if not series_id:
                raise ValueError("Выберите колонку серий")
            x_id, y_id, sy_id = (settings[key] for key in ("x", "y", "sy"))
            if not x_id or not y_id:
                raise ValueError("Выберите числовые колонки X и Y")
            project = self.owner.model.project
            numeric = {cid for cid in (x_id, y_id, sy_id) if cid}
            source_ids = sorted(numeric | {series_id})
            frame = project_frame(project, self.owner.analysis_scope.rows(), source_ids)
            schema = source_schema(project, source_ids)
            excluded = frame.index.isin(excluded_sample_ids(project, numeric, "statistics"))
            if comparison:
                if any(not isinstance(limits, dict) for limits in overrides.values()):
                    raise ValueError("В границах серий указана неизвестная серия или неверный формат")
                local_bounds = {name for limits in overrides.values() for name in limits}
                singletons = [(name,) for name in model.parameters if name not in local_bounds]
                candidates = tuple(dict.fromkeys([shared, *singletons]))
                if len(candidates) < 2:
                    raise ValueError("Для CV нужны хотя бы два варианта общих коэффициентов")
                folds = self.joint_cv_folds.value()
        except (KeyError, ValueError) as exc:
            self.status.setText(message(exc))
            return
        if comparison:
            self.clear_joint_comparison()
        else:
            self.invalidate()
        labels = self.x_column.currentText(), self.y_column.currentText()
        cancel = self._joint_cancel = Event()
        bootstrap_cancel = self._bootstrap_cancel = Event()
        generation = self.generation
        stored = {"column": series_id, "shared": list(shared), "series_bounds": bounds_text}
        if project.view.get("joint_curve_fitting") != stored or project.view.get("curve_fitting") != settings:
            project.view["joint_curve_fitting"] = stored
            project.view["curve_fitting"] = settings
            self.owner.view_changed()

        def compute():
            arrays, valid, rejected_ids, reasons = prepare_joint_rows(
                frame, x_id, y_id, sy_id, series_id, excluded, model.positive_x)
            diagnostics = rejected_ids, reasons
            sigma = arrays[sy_id][valid] if sy_id else None
            series = frame[series_id].to_numpy()[valid]
            if comparison:
                return compare_joint_sharing(
                    arrays[x_id][valid], arrays[y_id][valid], series, model, candidates, folds=folds,
                    sy=sigma, initial=initial, bounds=bounds, series_bounds=_series_bound_labels(overrides, series),
                    loss=settings["loss"], f_scale=settings["f_scale"], cancelled=cancel.is_set,
                    progress=lambda done: self.joint_cv_progress.emit(cancel, done, len(candidates)*folds),
                )
            try:
                result = fit_joint_curve(arrays[x_id][valid], arrays[y_id][valid], series, model, shared=shared,
                                         sy=sigma, initial=initial, bounds=bounds, series_bounds=_series_bound_labels(overrides, series),
                                         loss=settings["loss"], f_scale=settings["f_scale"],
                                         cancelled=cancel.is_set)
            except ValueError as exc:
                return diagnostics, exc
            if result is None:
                return None
            bootstrap = error = None
            band_method = settings["band_method"]
            active = result.active_bounds is not None and np.any(result.active_bounds)
            needs_bootstrap = (band_method in BOOTSTRAP_BAND_METHODS
                               or band_method == "auto" and (settings["loss"] != "linear" or active))
            if needs_bootstrap:
                try:
                    bootstrap = bootstrap_joint_fit(
                        result, resamples=settings["bootstrap_resamples"], seed=settings["bootstrap_seed"],
                        prediction_sigma=settings["prediction_sigma"] or None,
                        progress=lambda done, total: self.bootstrap_progress.emit(generation, done, total),
                        cancelled=lambda: cancel.is_set() or bootstrap_cancel.is_set())
                except (BootstrapCancelled, ValueError) as exc:
                    error = str(exc)
            scope_rows = frame.index.tolist()
            digest = joint_source_digest(schema, frame.itertuples(name=None), frame.index[excluded])
            provenance = {"version": 1, "source_ids": source_ids, "numeric_ids": sorted(numeric),
                          "row_ids": scope_rows, "digest": digest}
            return diagnostics, (result, frame.index[valid].tolist(), bootstrap, error, provenance)

        def display(payload):
            if cancel.is_set() or payload is None:
                return
            if comparison:
                self.show_joint_comparison(payload)
                return
            diagnostics, outcome = payload
            dropped = len(diagnostics[0])
            if dropped:
                self.joint_exclusions = diagnostics
                self.tabs.setTabEnabled(self.joint_exclusion_tab, True)
                self.joint_exclusion_button.setEnabled(True)
            if isinstance(outcome, ValueError):
                detail = message(outcome)
                self.status.setText(detail)
                if dropped:
                    self.tabs.setCurrentIndex(self.joint_exclusion_tab)
                QMessageBox.warning(self, tr("Ошибка"), detail)
                return
            (self.joint_result, self.joint_row_ids, self.joint_bootstrap,
             self.joint_bootstrap_error, self.joint_provenance) = outcome
            self.joint_run_id = new_id()
            ordinary_only = {"sx", "starts", "start_seed", "absolute_sigma", "parameter_rules",
                             "bootstrap_group", "bootstrap_block_length"}
            self.joint_settings = {key: value for key, value in settings.items() if key not in ordinary_only} | stored
            self.joint_band_method_actual = None
            if self.joint_bootstrap is not None:
                self.joint_band_method_actual = ("bootstrap_simultaneous" if settings["band_method"] == "bootstrap_simultaneous"
                                                 else "bootstrap")
            self.joint_axis_labels = labels
            self.show_joint_result(dropped)

        self.status.setText(tr("Выполняется расчёт…"))
        def finished():
            cancel.set()
            self.cancel_bootstrap.setEnabled(False)
            self.joint_cv_cancel_button.setEnabled(False)

        self.run_task(compute, display, on_finished=finished)
        self.setEnabled(True)
        self.joint_cv_cancel_button.setEnabled(comparison)

    def show_joint_result(self, dropped):
        result = self.joint_result
        parameters = result.model.parameters
        common = [parameters.index(name) for name in result.shared]
        local = [i for i, name in enumerate(parameters) if name not in result.shared]
        labels = [f"{i+1}. {name}" for i, name in enumerate(result.labels)]
        method = "WLS" if result.sy is not None else "OLS"
        summary = [tr("Общие коэффициенты")+" · " + ", ".join(
            f"{parameters[i]} = {result.coefficients[0, i]:.8g}" for i in common),
            f"{method} · {self.loss.currentText()} · RMSE {result.metrics['RMSE']:.6g} · MAE {result.metrics['MAE']:.6g}"]
        if self.joint_bootstrap is not None:
            counts = self.joint_bootstrap.diagnostics()
            summary.append(tr("Bootstrap с одновременной полосой") if self.joint_band_method_actual == "bootstrap_simultaneous"
                           else tr("Bootstrap (percentile)"))
            summary.append(tr("Bootstrap: успешно {0} из {1}", counts["successful"], counts["resamples"]))
        elif not self.joint_bootstrap_error:
            summary.append(tr("Полосы не рассчитаны. Чтобы построить полосы, выберите bootstrap и повторите подгонку."))
        bootstrap_warning = message(self.joint_bootstrap_error) if self.joint_bootstrap_error else ""
        if bootstrap_warning:
            summary.append(bootstrap_warning)
        self.joint_summary.setText("\n".join(summary))
        self.update_joint_tail_warning()
        rows = [[label, *result.coefficients[i, local].tolist(), int(result.boundaries[i+1]-result.boundaries[i]),
                 float(result.series_rmse[i]), float(result.series_mae[i])] for i, label in enumerate(labels)]
        headers = [tr("Серия"), *(parameters[i] for i in local), "N", "RMSE", "MAE"]
        if result.bounds is not None:
            headers.append(tr("Активные границы"))
            for row, mask in zip(rows, result.active_bounds):
                row.append(", ".join(f"{name} {'↑' if active > 0 else '↓'}" for name, active in zip(parameters, mask) if active))
        self.joint_table.setModel(ResultsModel(headers, rows, self.joint_table))
        self.joint_table.selectionModel().currentRowChanged.connect(
            lambda current, _: self.joint_series.setCurrentIndex(current.row()) if current.isValid() else None)
        self.joint_plots.set_result(result, self.joint_row_ids, self.joint_axis_labels)
        with QSignalBlocker(self.joint_series):
            refill_combo(self.joint_series, [(label, i) for i, label in enumerate(labels)])
        self.tabs.setTabEnabled(self.joint_tab, True)
        self.tabs.setCurrentWidget(self.joint_page)
        self.joint_export_button.setEnabled(True)
        self.joint_columns_button.setEnabled(True)
        self.joint_report_button.setEnabled(True)
        for widget in (self.show_confidence, self.show_prediction):
            widget.setEnabled(self.joint_bootstrap is not None)
        status = tr("Серий: {0} · Точек: {1} · Исключено: {2}", len(labels), len(result.x), dropped)
        if bootstrap_warning:
            status = f"{status}\n{bootstrap_warning}"
        self.announce_status(status)
        self.select_joint_series()

    def update_joint_tail_warning(self, *args):
        bootstrap = self.joint_bootstrap
        warnings = bootstrap.diagnostics(confidence=self.confidence.value()/100)["warnings"] if bootstrap else []
        self.joint_tail_warning.setText("\n".join(message(warning) for warning in warnings))
        self.joint_tail_warning.setVisible(bool(warnings))

    def draw_joint_series(self, *args):
        self._joint_band_view = None
        self._joint_band_cache.discard_pending()
        result, series = self.joint_result, self.joint_series.currentData()
        if result is None or series is None or self.joint_plots.currentIndex() != 0:
            return
        indices = result.indices(series)
        x = result.x[indices]
        grid = np.linspace(x.min(), x.max(), 200)
        self.joint_figure.clear()
        axis = self.joint_figure.subplots()
        axis.set_xlabel(self.joint_axis_labels[0], parse_math=False)
        axis.set_ylabel(self.joint_axis_labels[1], parse_math=False)
        axis.scatter(x, result.y[indices], s=12, linewidths=0, rasterized=True)
        try:
            mean = result.predict(series, grid)
            axis.plot(grid, mean, color="tab:red")
            if self._joint_curve_error_status is not None:
                previous, error = self._joint_curve_error_status
                if self.status.text() == error:
                    self.announce_status(previous)
                self._joint_curve_error_status = None
            bootstrap = self.joint_bootstrap
            if bootstrap is not None and (self.show_confidence.isChecked() or self.show_prediction.isChecked()):
                confidence = self.confidence.value()/100
                key = series, confidence
                simultaneous = self.joint_settings["band_method"] == "bootstrap_simultaneous"
                bands = self._joint_band_cache.request(
                    key, lambda: bootstrap.bands(result, series, grid, confidence=confidence, simultaneous=simultaneous),
                )
                pending = None if bands is not None else axis.text(
                    .02, .98, tr("Расчёт полос…"), transform=axis.transAxes, va="top")
                self._joint_band_view = key, axis, grid, pending
                if bands is not None:
                    self.show_joint_bands(key, bands)
            else:
                self.clear_joint_band_error()
        except ValueError as exc:
            error = message(exc)
            previous = self.status.text()
            if self._joint_curve_error_status is not None:
                saved_status, saved_error = self._joint_curve_error_status
                if previous == saved_error:
                    previous = saved_status
            self._joint_curve_error_status = previous, error
            self.announce_status(error)
        self.joint_canvas.draw_idle()

    def show_joint_bands(self, key, bands):
        view = self._joint_band_view
        if view is None or view[0] != key:
            return
        _, axis, grid, pending = view
        if pending is not None:
            pending.remove()
        if isinstance(bands, str):
            error = message(bands)
            previous = self.status.text()
            if self._joint_band_error_status is not None:
                saved_status, saved_error = self._joint_band_error_status
                if previous == saved_error:
                    previous = saved_status
            self._joint_band_error_status = previous, error
            self.announce_status(error)
        else:
            mean, lower, upper, pred_lower, pred_upper = bands
            show_prediction = self.show_prediction.isChecked()
            show_confidence = self.show_confidence.isChecked()
            visible_bands = (bands[3:] if show_prediction else ()) + (bands[1:3] if show_confidence else ())
            scale = curve_axis_scale(*visible_bands)
            if scale != 1.:
                result = self.joint_result
                indices = result.indices(key[0])
                axis.clear()
                axis.scatter(result.x[indices], result.y[indices]/scale, s=12, linewidths=0, rasterized=True)
                axis.plot(grid, mean/scale, color="tab:red")
                axis.set_xlabel(self.joint_axis_labels[0], parse_math=False)
                axis.set_ylabel(f"{self.joint_axis_labels[1]} (× {scale:.0e})", parse_math=False)
                lower, upper = lower/scale, upper/scale
                pred_lower, pred_upper = pred_lower/scale, pred_upper/scale
            level = f" {key[1]*100:g}%"
            if show_prediction:
                axis.fill_between(grid, pred_lower, pred_upper, alpha=.12, label=tr("Предсказательная полоса")+level)
            if show_confidence:
                label = (tr("Одновременная доверительная полоса") if self.joint_settings["band_method"] == "bootstrap_simultaneous"
                         else tr("Доверительная полоса"))
                axis.fill_between(grid, lower, upper, alpha=.24, label=label+level)
            axis.legend(loc="upper left")
            self.clear_joint_band_error()
        self._joint_band_view = None
        self.joint_canvas.draw_idle()

    def clear_joint_band_error(self):
        if self._joint_band_error_status is None:
            return
        previous, error = self._joint_band_error_status
        if self.status.text() == error:
            self.announce_status(previous)
        self._joint_band_error_status = None

    def apply_joint_fit(self):
        result = self.joint_result
        if result is None or self.task_running:
            return
        model = self.owner.model
        project = model.project
        settings = self.joint_settings.copy()
        row_ids = self.joint_row_ids
        unit = next(column.unit for column in project.columns if column.id == settings["y"])
        labels = (tr("Прогноз Y"), tr("Остаток Y"))
        prefix = tr("Совместная подгонка")

        def prepare():
            return (dict(zip(row_ids, result.fitted.tolist())),
                    dict(zip(row_ids, (result.y-result.fitted).tolist())))

        def commit(outputs):
            if self.joint_result is not result or model.project is not project:
                return
            run_id = self.joint_run_id
            project.view.setdefault("joint_curve_runs", {})[run_id] = self.joint_provenance
            names = {column.name for column in project.columns}
            commands = []
            for label, output, values in zip(labels, ("prediction", "residual"), outputs):
                base = name = f"{prefix}: {label}"
                suffix = 2
                while name in names:
                    name = f"{base} ({suffix})"
                    suffix += 1
                names.add(name)
                column = Column(name, kind="number", unit=unit,
                                metadata={"joint_curve_fit": {"settings": settings, "output": output, "run_id": run_id},
                                          "source_status": "current"})
                commands.append(AddColumnCommand(model, column, values))
            model.undo.beginMacro(tr("Колонки подгонки кривой"))
            try:
                for command in commands:
                    model.undo.push(command)
            finally:
                model.undo.endMacro()
            self.status.setText(tr("Создано колонок: {0}", len(commands)))

        self.run_task(prepare, commit)

    def save_joint_for_report(self):
        result = self.joint_result
        if result is None or self.task_running:
            return
        project = self.owner.model.project
        row_ids, settings = self.joint_row_ids, self.joint_settings.copy()
        axes = self.joint_axis_labels
        bootstrap, bootstrap_failed = self.joint_bootstrap, bool(self.joint_bootstrap_error)
        confidence = self.confidence.value()/100
        band_method_actual = self.joint_band_method_actual

        def prepare():
            return {
                "saved_at": timestamp(), "row_ids": list(row_ids), "settings": settings,
                "axes": list(axes), "formula": result.model.formula,
                "parameters": list(result.model.parameters), "series": [str(label) for label in result.labels],
                "coefficients": result.coefficients.tolist(), "counts": np.diff(result.boundaries).tolist(),
                "metrics": {key: value if np.isfinite(value) else None for key, value in result.metrics.items()},
                "rmse": result.series_rmse.tolist(), "mae": result.series_mae.tolist(),
                "active_bounds": result.active_bounds.tolist() if result.active_bounds is not None else None,
                "bootstrap": bootstrap.diagnostics(confidence=confidence) if bootstrap is not None else None,
                "bootstrap_failed": bootstrap_failed,
                "band_method_actual": band_method_actual,
                "plots": capture_joint_plots(result, bootstrap, confidence=confidence,
                                             simultaneous=settings["band_method"] == "bootstrap_simultaneous"),
            }

        def commit(snapshot):
            if self.joint_result is result and self.owner.model.project is project:
                project.view.setdefault("joint_curve_runs", {})[self.joint_run_id] = self.joint_provenance
                if previous := project.view.get("joint_curve_report"):
                    project.view.setdefault("joint_curve_reports", {})[previous.get("id", "legacy")] = previous
                snapshot.update(id=new_id(), run_id=self.joint_run_id, source_status="current")
                project.view["joint_curve_report"] = snapshot
                self.owner.view_changed()
                self.refresh_joint_snapshot_warning()
                self.status.setText(tr("Совместная подгонка сохранена для отчёта"))

        self.run_task(prepare, commit)

    def export_joint(self):
        result = self.joint_result
        if result is None or self.task_running:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Экспорт совместной подгонки JSON…"), "joint_fit.json", "JSON (*.json)")
        if not path or self.joint_result is not result:
            return
        settings, row_ids = self.joint_settings, self.joint_row_ids
        bootstrap, bootstrap_error = self.joint_bootstrap, self.joint_bootstrap_error
        series, confidence = self.joint_series.currentData(), self.confidence.value()/100
        band_method_actual = self.joint_band_method_actual

        def export():
            payload = {"settings": settings, "formula": result.model.formula, "band_method_actual": band_method_actual,
                       "parameter_names": list(result.model.parameters), "series_labels": [str(label) for label in result.labels],
                       "coefficients": result.coefficients.tolist(), "metrics": {k: v if np.isfinite(v) else None
                                                                                 for k, v in result.metrics.items()},
                       "series_rmse": result.series_rmse.tolist(), "series_mae": result.series_mae.tolist(),
                       "row_ids": row_ids, "series_ids": result.codes.tolist(), "x": result.x.tolist(),
                       "y": result.y.tolist(), "sy": result.sy.tolist() if result.sy is not None else None,
                       "fitted": result.fitted.tolist()}
            if result.bounds is not None:
                payload["bounds"] = [np.where(np.isfinite(side), side, None).tolist() for side in result.bounds]
                payload["active_bounds"] = result.active_bounds.tolist()
            if result.evaluations is not None:
                payload["evaluations"] = result.evaluations
            if bootstrap is not None:
                payload["bootstrap"] = bootstrap.diagnostics(confidence=confidence)
                payload["bootstrap_coefficients"] = bootstrap.coefficients.tolist()
                payload["bootstrap_prediction_draws"] = bootstrap.prediction_draws.tolist()
                indices = result.indices(series)
                grid = np.linspace(result.x[indices].min(), result.x[indices].max(), 200)
                try:
                    bands = bootstrap.bands(result, series, grid, confidence=confidence,
                                            simultaneous=settings["band_method"] == "bootstrap_simultaneous")
                    payload["bands"] = {"series": series, "confidence": confidence, "grid": grid.tolist(),
                                        **{name: values.tolist() for name, values in zip(("mean", "confidence_lower", "confidence_upper",
                                                                                        "prediction_lower", "prediction_upper"), bands)}}
                except ValueError as exc:
                    payload["bands_error"] = str(exc)
            if bootstrap_error:
                payload["bootstrap_error"] = bootstrap_error
            write_json(path, payload)

        self.run_task(export, lambda _: None)
