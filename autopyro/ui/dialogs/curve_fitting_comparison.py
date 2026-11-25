"""Configure validation, compare candidates and export comparison results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
)
import numpy as np

from autopyro.curves.curve_validation import NESTED_STRATEGIES

from ..i18n import tr, message
from ..theme import icon
from ..widgets.documents import parameter_page, refill_combo
from ..widgets.results import ResultsModel
from .curve_fitting_inputs import _model_options


@dataclass(frozen=True)
class _ComparisonFit:
    settings: dict
    cv: dict
    row_ids: tuple[str, ...]
    winner: str | None
    rmse: float | None

    def resolve(self, winner, rmse):
        definition = next(candidate["settings"] for candidate in self.cv["candidates"] if candidate["name"] == winner)
        source_strategy = self.cv["strategy"]
        strategy = NESTED_STRATEGIES.get(source_strategy, source_strategy)
        provenance = {"source": "cross_validation", "candidate": winner, "source_strategy": source_strategy,
                      "strategy": strategy, "rmse": rmse, "seed": self.cv["seed"],
                      "folds": self.cv["inner_folds"] if source_strategy in NESTED_STRATEGIES else self.cv["folds"],
                      "shuffle": self.cv["shuffle"], "gap": self.cv["gap"] if strategy == "time" else 0,
                      "holdout_column": self.cv.get("holdout_key"),
                      "split_column": self.cv["key"] if strategy in ("group", "time") else None}
        return self.settings | definition | {"selection": provenance}


def build_comparison_controls(editor):
    controls = parameter_page(editor.parameter_tabs, tr("Кросс-валидация"), 'files')
    editor.candidate_list = QListWidget()
    controls.addWidget(editor.candidate_list)
    add = QPushButton(icon('plus'), tr("Добавить текущую модель"))
    add.clicked.connect(editor.add_candidate)
    controls.addWidget(add)
    remove = QPushButton(icon('trash'), tr("Удалить выбранную модель"))
    remove.clicked.connect(editor.remove_candidate)
    controls.addWidget(remove)
    form = QFormLayout()
    controls.addLayout(form)
    editor.cv_folds = QSpinBox()
    editor.cv_folds.setRange(2, 20)
    editor.cv_inner_folds = QSpinBox()
    editor.cv_inner_folds.setRange(2, 20)
    editor.cv_seed = QSpinBox()
    editor.cv_seed.setRange(0, 2**31-1)
    form.addRow(tr("Число блоков K"), editor.cv_folds)
    form.addRow(tr("Число внутренних блоков"), editor.cv_inner_folds)
    form.addRow("Seed", editor.cv_seed)
    editor.cv_strategy = QComboBox()
    for label, code, image in ((tr("Независимые строки"), "kfold", 'table-2'), (tr("Разделение по группам"), "group", 'layers'),
                        (tr("Прошлое → будущее"), "time", 'play'), (tr("По одной точке (LOO)"), "loo", 'table-2'),
                        (tr("Вложенная CV"), "nested", 'layers'), (tr("Вложенная CV по группам"), "nested_group", 'files'),
                        (tr("Вложенная временная CV"), "nested_time", 'clock')):
        editor.cv_strategy.addItem(icon(image), label, code)
    editor.cv_key = QComboBox()
    editor.cv_holdout = QComboBox()
    editor.cv_pair = (QComboBox(), QComboBox())
    editor.cv_pair_margin = QDoubleSpinBox()
    editor.cv_pair_margin.setDecimals(12)
    editor.cv_pair_margin.setRange(0, 1e12)
    editor.cv_pair_margin.setSpecialValueText(tr("Без порога"))
    editor.cv_pair_margin.setToolTip(tr("Задайте допустимую разницу MAE в единицах Y до просмотра тестовых результатов. Ноль отключает порог."))
    for combo in editor.cv_pair:
        combo.setToolTip(tr("Интервал ΔMAE использует уровень доверия и настройки bootstrap с вкладки «Параметры»."))
    editor.cv_holdout.setToolTip(tr("Задайте роли в текстовой колонке. Используйте «Обучение» / Train и «Тест» / Test. Пустые строки исключаются."))
    editor.cv_gap = QSpinBox()
    editor.cv_gap.setRange(0, 1000000)
    form.addRow(tr("Стратегия CV"), editor.cv_strategy)
    form.addRow(tr("Колонка группы / времени"), editor.cv_key)
    form.addRow(tr("Колонка ролей выборки"), editor.cv_holdout)
    form.addRow(tr("Тестовая модель A"), editor.cv_pair[0])
    form.addRow(tr("Тестовая модель B"), editor.cv_pair[1])
    form.addRow(tr("Практический порог ΔMAE"), editor.cv_pair_margin)
    form.addRow(tr("Разрыв (различных моментов)"), editor.cv_gap)
    for c in editor.owner.model.project.columns:
        editor.cv_key.addItem(c.name, c.id)
    editor.cv_shuffle = QCheckBox(tr("Перемешать строки перед разбиением"))
    controls.addWidget(editor.cv_shuffle)
    stored = editor.owner.model.project.view.get("curve_fitting_comparison", {})
    editor.refill_holdout_columns(stored.get("holdout_key"))
    editor.cv_folds.setValue(stored.get("folds", 5))
    editor.cv_inner_folds.setValue(stored.get("inner_folds", 4))
    editor.cv_seed.setValue(stored.get("seed", 0))
    editor.cv_strategy.setCurrentIndex(max(0, editor.cv_strategy.findData(stored.get("strategy", "kfold"))))
    editor.cv_key.setCurrentIndex(max(0, editor.cv_key.findData(stored.get("key"))))
    editor.cv_gap.setValue(stored.get("gap", 0))
    editor.cv_shuffle.setChecked(stored.get("shuffle", True))
    editor.candidates = deepcopy(stored.get("candidates", []))
    editor.candidate_list.addItems([c["name"] for c in editor.candidates])
    editor.refill_test_models()
    for combo, name in zip(editor.cv_pair, stored.get("holdout_pair") or ()):
        combo.setCurrentIndex(max(0, combo.findData(name)))
    editor.cv_pair_margin.setValue(stored.get("holdout_margin", 0.))
    editor.cv_pair_margin.valueChanged.connect(editor.invalidate_comparison)
    for combo in editor.cv_pair:
        combo.currentIndexChanged.connect(editor.invalidate_comparison)
        combo.currentIndexChanged.connect(editor.update_bootstrap_controls)
    editor.cv_folds.valueChanged.connect(editor.invalidate_comparison)
    editor.cv_inner_folds.valueChanged.connect(editor.invalidate_comparison)
    editor.cv_seed.valueChanged.connect(editor.invalidate_comparison)
    editor.cv_shuffle.toggled.connect(editor.invalidate_comparison)
    editor.cv_strategy.currentIndexChanged.connect(editor.invalidate_comparison)
    editor.cv_strategy.currentIndexChanged.connect(editor.update_cv_controls)
    editor.cv_key.currentIndexChanged.connect(editor.invalidate_comparison)
    editor.cv_holdout.currentIndexChanged.connect(editor.invalidate_comparison)
    editor.cv_holdout.currentIndexChanged.connect(editor.update_cv_controls)
    editor.cv_gap.valueChanged.connect(editor.invalidate_comparison)
    run = QPushButton(icon('files'), tr("Сравнить модели"))
    run.clicked.connect(lambda: editor.calculate(comparison=True))
    controls.addWidget(run)
    editor.export_comparison_button = QPushButton(icon('file-text'), tr("Экспорт сравнения JSON…"))
    editor.export_comparison_button.setEnabled(False)
    editor.export_comparison_button.clicked.connect(editor.export_comparison)
    controls.addWidget(editor.export_comparison_button)
    editor.fit_selected_button = QPushButton(icon('chart-no-axes-combined'), tr("Подогнать выбранную модель"))
    editor.fit_selected_button.setEnabled(False)
    editor.fit_selected_button.clicked.connect(editor.fit_selected_model)
    controls.addWidget(editor.fit_selected_button)
    note = QLabel(tr("Добавляйте модели с вкладки «Параметры»: формула, начальные значения и границы сохраняются отдельно для каждой. Метод, функция потерь и σ общие. Рейтинг — по RMSE отложенных прогнозов; все модели используют одинаковые блоки. K-fold предполагает независимые строки; группы и временные ряды требуют другого разбиения."))
    note.setWordWrap(True)
    controls.addWidget(note)
    controls.addStretch()
    editor.update_cv_controls()


def refill_holdout_columns(editor, selected):
    choices = [(tr("Без тестовой выборки"), None)]
    choices.extend((c.name, c.id) for c in editor.owner.model.project.columns if c.kind == "text")
    if selected and selected not in {cid for _, cid in choices}:
        choices.append((tr("Колонка ролей недоступна"), selected))
    if refill_combo(editor.cv_holdout, choices):
        role_icon = icon("table-2")
        for index in range(1, editor.cv_holdout.count()):
            editor.cv_holdout.setItemIcon(index, role_icon)
    editor.cv_holdout.setCurrentIndex(max(0, editor.cv_holdout.findData(selected)))


def update_cv_controls(editor, *args):
    strategy = editor.cv_strategy.currentData()
    base_strategy = NESTED_STRATEGIES.get(strategy, strategy)
    editor.cv_key.setEnabled(base_strategy in ("group", "time"))
    editor.cv_gap.setEnabled(base_strategy == "time")
    editor.cv_folds.setEnabled(strategy != "loo")
    editor.cv_inner_folds.setEnabled(strategy in NESTED_STRATEGIES)
    editor.cv_seed.setEnabled(base_strategy not in ("loo", "time"))
    editor.cv_shuffle.setEnabled(base_strategy not in ("loo", "time"))
    for combo in editor.cv_pair:
        combo.setEnabled(strategy not in NESTED_STRATEGIES and editor.cv_holdout.currentData() is not None and len(editor.candidates) >= 2)
    editor.update_bootstrap_controls()
    if base_strategy == "time":
        editor.cv_shuffle.setChecked(False)


def comparison_settings(editor):
    return {"candidates": deepcopy(editor.candidates), "folds": editor.cv_folds.value(),
            "inner_folds": editor.cv_inner_folds.value(),
            "seed": editor.cv_seed.value(), "shuffle": editor.cv_shuffle.isChecked(),
            "holdout_key": editor.cv_holdout.currentData(),
            "holdout_pair": [combo.currentData() for combo in editor.cv_pair],
            "holdout_margin": editor.cv_pair_margin.value(),
            "strategy": editor.cv_strategy.currentData(), "key": editor.cv_key.currentData(), "gap": editor.cv_gap.value()}


def save_comparison_settings(editor):
    editor.owner.model.project.view["curve_fitting_comparison"] = editor.comparison_settings()
    editor.owner.view_changed()


def refill_test_models(editor):
    choices = [(tr("Без попарного сравнения"), None), *((c["name"], c["name"]) for c in editor.candidates)]
    for combo in editor.cv_pair:
        if refill_combo(combo, choices):
            model_icon = icon("chart-no-axes-combined")
            for index in range(1, combo.count()):
                combo.setItemIcon(index, model_icon)
    editor.update_cv_controls()


def add_candidate(editor):
    settings = editor.settings()
    try:
        model, _, _ = _model_options(settings)
    except ValueError as exc:
        editor.status.setText(message(exc))
        return
    # Store only model-specific settings; data and estimator options are common.
    definition = {key: settings[key] for key in ("model", "degree", "expression", "parameters",
                                                 "initial", "lower_bounds", "upper_bounds", "parameter_rules")}
    number = max((c.get("number", 0) for c in editor.candidates), default=0)+1
    name = f"{number}. {model.name}: {model.formula}"
    editor.candidates.append({"name": name, "number": number, "settings": definition})
    editor.candidate_list.addItem(name)
    editor.refill_test_models()
    editor.invalidate_comparison()
    editor.save_comparison_settings()


def remove_candidate(editor):
    index = editor.candidate_list.currentRow()
    if index >= 0:
        editor.candidates.pop(index)
        editor.candidate_list.takeItem(index)
        editor.refill_test_models()
        editor.invalidate_comparison()
        editor.save_comparison_settings()


def invalidate_comparison(editor, *args):
    editor._comparison_cancel.set()
    editor.comparison_result = None
    editor.comparison_generation += 1
    if hasattr(editor, "holdout_pair_table"):
        editor.clear_holdout_pair()
    if hasattr(editor, "export_comparison_button"):
        editor.export_comparison_button.setEnabled(False)
        editor.fit_selected_button.setEnabled(False)
    if hasattr(editor, "comparison_table"):
        old = editor.comparison_table.model()
        editor.comparison_table.setModel(None)
        if old is not None:
            old.deleteLater()
    if getattr(editor, "_diagnostic_source", None) is not None and editor._diagnostic_source[-1]:
        editor._diagnostic_source = None
        editor._diagnostic_drawn = False
        editor.clear_diagnostics()
    if editor._running_comparison and hasattr(editor, "status"):
        editor.status.setText(tr("Настройки сравнения изменились; нажмите «Сравнить модели»"))


def show_comparison(editor):
    result = editor.comparison_result
    if result.nested is not None:
        editor.show_nested_comparison(result.nested)
        return
    metric_keys = ["RMSE", "MAE", "MedAE", "R2", "RMSE_fold_std"]
    headers = [tr("Модель"), "OOF RMSE", "OOF MAE", "OOF MedAE", "OOF R²", "SD RMSE"]
    if result.plan.strategy == "loo":
        metric_keys.append("PRESS")
        headers.append("PRESS")
    test_keys = ()
    if result.holdout is not None:
        test_keys = ("RMSE", "MAE", "R2")
        headers.extend(tr("Тест {0}", name) for name in ("RMSE", "MAE", "R²"))
    headers.append(tr("Ошибки блоков"))
    rows = []
    for score in [*result.ranking, *(s for s in result.scores if s.errors)]:
        test_metrics = (result.holdout.metrics or {}) if test_keys and score.name == result.holdout.winner else {}
        test_values = [test_metrics[key] if key in test_metrics and np.isfinite(test_metrics[key]) else None
                       for key in test_keys]
        rows.append([score.name, *(score.metrics.get(key) for key in metric_keys),
                     *test_values,
                     "\n".join(message(error) for error in score.errors)])
    editor.set_comparison_table(headers, rows)
    editor.tabs.setTabEnabled(editor.holdout_pair_tab, result.holdout is not None and result.holdout.pair is not None)
    if not result.ranking:
        editor.status.setText(tr("Ни одна модель не прошла все блоки CV; проверьте ошибки в таблице"))
    elif result.holdout is not None:
        status = tr("Обучение: {0} · Тест: {1} · Исключено: {2}", len(result.y), len(result.holdout.y), editor.excluded_count)
        if result.holdout.error:
            status += "\n" + tr("Ошибка независимого теста: {0}", message(result.holdout.error))
        editor.status.setText(status)


def show_nested_comparison(editor, nested):
    strategy = editor.comparison_result.plan.strategy
    headers = [tr("Внешний блок"), tr("Выбранная модель"), "N train", "N test"]
    extra_fields = ()
    if strategy == "nested_group":
        headers.extend([tr("Групп в обучении"), tr("Групп в проверке")])
        extra_fields = ("train_groups", "test_groups")
    elif strategy == "nested_time":
        headers.extend([tr("Моментов в обучении"), tr("Моментов в проверке"), tr("Строк в разрыве")])
        extra_fields = ("train_periods", "test_periods", "gap_rows")
    headers.extend([tr("Внутренний RMSE"), tr("Внешний RMSE"), tr("Внешний MAE"), tr("Ошибки блоков")])
    rows = [[fold.fold, fold.winner, fold.train_n, fold.test_n,
             *(getattr(fold, key) for key in extra_fields), fold.inner_rmse,
             fold.metrics.get("RMSE"), fold.metrics.get("MAE"), message(fold.error) if fold.error else ""]
            for fold in nested.folds]
    editor.set_comparison_table(headers, rows)
    if nested.metrics:
        editor.status.setText(tr("Внешние прогнозы: {0} · RMSE {1:.6g} · MAE {2:.6g}",
                               nested.metrics["N_evaluated"], nested.metrics["RMSE"], nested.metrics["MAE"]))
    else:
        editor.status.setText(tr("Вложенная CV: общая оценка недоступна из-за ошибок во внешних блоках"))


def set_comparison_table(editor, headers, rows):
    old = editor.comparison_table.model()
    editor.comparison_table.setModel(ResultsModel(headers, rows, editor.comparison_table))
    if old is not None:
        old.deleteLater()
    editor.comparison_table.horizontalHeader().setResizeContentsPrecision(100)
    editor.comparison_table.resizeColumnsToContents()
    editor.tabs.setCurrentWidget(editor.comparison_table)
    editor.export_comparison_button.setEnabled(True)
    result = editor.comparison_result
    editor.fit_selected_button.setEnabled(bool(result.nested.metrics if result.nested is not None else result.ranking))
    editor.show_oof_diagnostics()


def fit_selected_model(editor):
    result = editor.comparison_result
    if result is None or editor.task_running:
        return
    winner = rmse = None
    if result.nested is not None:
        if not result.nested.metrics:
            return
    else:
        ranking = result.ranking
        if not ranking:
            return
        index = editor.comparison_table.currentIndex()
        row = index.row() if index.isValid() else 0
        if row >= len(ranking):
            editor.status.setText(tr("Выберите модель, которая прошла все блоки CV"))
            return
        winner, rmse = ranking[row].name, ranking[row].metrics["RMSE"]
    selection = _ComparisonFit(deepcopy(editor.fit_settings), deepcopy(editor.cv_snapshot), tuple(editor.row_ids), winner, rmse)
    editor.calculate(selection=selection)


def export_comparison(editor):
    if editor.comparison_result is None:
        return
    path, _ = QFileDialog.getSaveFileName(editor, tr("Экспорт сравнения"), "model_comparison.json", "JSON (*.json)")
    if not path:
        return
    result = editor.comparison_result
    finite_metrics = lambda values: {k: v if np.isfinite(v) else None for k, v in values.items()}
    def finite_values(array):
        return [float(value) if np.isfinite(value) else None for value in array]

    payload = {"settings": editor.fit_settings, "cross_validation": editor.cv_snapshot,
               "x": result.x.tolist(), "y": result.y.tolist(),
               "row_ids": editor.row_ids, "fold_ids": result.fold_ids.tolist(),
               "ranking": [s.name for s in result.ranking], "scores": [
                   {"name": s.name, "metrics": finite_metrics(s.metrics),
                    "fold_metrics": [finite_metrics(m) for m in s.fold_metrics], "errors": s.errors,
                    "predictions": [float(v) if np.isfinite(v) else None for v in s.predictions]
                    if s.predictions is not None else None}
                   for s in result.scores]}
    if result.nested is not None:
        nested = result.nested
        payload["nested"] = {"inner_folds": nested.inner_folds, "metrics": finite_metrics(nested.metrics),
                             "predictions": finite_values(nested.predictions),
                             "folds": [asdict(fold) | {"metrics": finite_metrics(fold.metrics)} for fold in nested.folds]}
    if result.holdout is not None:
        test = result.holdout
        payload["holdout"] = {"winner": test.winner, "row_ids": editor.holdout_row_ids,
                              "x": test.x.tolist(), "y": test.y.tolist(),
                              "parameters": test.parameters.tolist() if test.parameters is not None else None,
                              "predictions": test.predictions.tolist() if test.predictions is not None else None,
                              "metrics": finite_metrics(test.metrics) if test.metrics is not None else None,
                              "error": test.error}
        if test.pair is not None:
            pair = test.pair
            payload["holdout"]["pair"] = {
                "models": [{"name": model.name, "error": model.error,
                            "parameters": model.parameters.tolist() if model.parameters is not None else None,
                            "predictions": finite_values(model.predictions) if model.predictions is not None else None}
                           for model in pair.models],
                "absolute_errors": [finite_values(errors) for errors in pair.absolute_errors],
                "difference_B_minus_A": finite_values(pair.difference), "metrics": pair.metrics,
                "interval": pair.interval, "interval_error": pair.interval_error,
            }
    try:
        from autopyro.data_io.file_output import write_json
        write_json(path, payload)
    except (OSError, ValueError) as exc:
        QMessageBox.warning(editor, tr("Ошибка"), message(exc))
