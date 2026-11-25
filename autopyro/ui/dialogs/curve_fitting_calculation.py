"""Capture fitting inputs and run cancellable fit, validation and bootstrap tasks."""

from __future__ import annotations

from copy import deepcopy
from threading import Event

import numpy as np

from autopyro.curves.curve_archive import compare_archived_fit
from autopyro.curves.curve_bootstrap import BOOTSTRAP_BAND_METHODS, BootstrapCancelled, bootstrap_bands
from autopyro.curves.curve_fitting import fit_curve
from autopyro.curves.curve_validation import NESTED_STRATEGIES, holdout_masks
from autopyro.curves.model_comparison import ComparisonCancelled, FitCandidate, compare_models
from autopyro.model.tabular import classify_series, exclusion_mask, project_frame

from ..i18n import tr, message
from .curve_fitting_inputs import _FIT_DEFAULTS, _column_label, _model_options


def initialize(editor):
    editor.result = None
    editor.bootstrap_result = None
    editor.bootstrap_error = None
    editor._bootstrap_cancel = Event()
    editor.bootstrap_progress.connect(editor.show_bootstrap_progress)
    editor.comparison_result = None
    editor.comparison_generation = 0
    editor._comparison_cancel = Event()
    editor._running_comparison = False
    editor._restoring_fit_settings = False
    editor.candidates = []
    editor.generation = 0


def invalidate(editor, *args):
    if editor._restoring_fit_settings:
        return
    editor.clear_analysis()
    editor.clear_joint_fit()
    editor._bootstrap_cancel.set()
    editor.generation += 1
    editor.result = None
    editor.bootstrap_result = None
    editor.bootstrap_error = None
    editor.invalidate_comparison()
    if hasattr(editor, "status"):
        editor.status.setText(tr("Данные или параметры изменились; нажмите «Рассчитать»"))
        editor.export_button.setEnabled(False)
        editor.export_plot_button.setEnabled(False)
        editor.summary.clear()
        editor.figure.clear()
        editor.canvas.draw_idle()
        editor.show_confidence.setEnabled(True)
        editor.show_prediction.setEnabled(True)
        editor.confidence.setEnabled(True)


def calculate(editor, *, comparison=False, selection=None, replay=None):
    if editor.task_running:
        return
    if replay is None:
        editor.invalidate()
    reselect = selection is not None and selection.winner is None
    run_cv = comparison or reselect
    try:
        replay_rows = None
        replay_confidence = None
        if replay is not None:
            replay_rows = tuple(replay.get("row_ids") or replay.get("source_rows") or ())
            if not replay_rows or len(set(replay_rows)) != len(replay_rows):
                raise ValueError("В архиве нет пригодного списка обучающих строк")
            settings = _FIT_DEFAULTS | deepcopy(replay["settings"])
            settings["replay"] = {"archive_id": replay["id"], "archive_name": replay["name"],
                                  "source_was_stale": bool(replay.get("stale"))}
            replay_confidence = replay["confidence"]
            cv_settings = None
        elif selection is not None:
            settings, cv_settings = selection.settings, selection.cv
        else:
            settings = editor.settings()
            cv_settings = editor.comparison_settings() if comparison else None
        model_options = None
        if run_cv:
            candidates = [FitCandidate(c["name"], *_model_options(c["settings"])) for c in cv_settings["candidates"]]
            if not candidates:
                raise ValueError("Добавьте модели в список сравнения")
        elif selection is None:
            model_options = _model_options(settings)
        fields = [settings["x"], settings["y"]]
        if not all(fields):
            raise ValueError("Выберите числовые колонки X и Y")
        for key in ("sx", "sy"):
            if settings[key] and (key != "sx" or settings["method"] == "odr"):
                fields.append(settings[key])
        project = editor.owner.model.project
        columns = {column.id: column for column in project.columns}
        if any(cid not in columns or columns[cid].kind != "number" or columns[cid].metadata.get("method")
               or columns[cid].metadata.get("curve_fit") for cid in fields):
            raise ValueError("Выбранные колонки отсутствуют или интерполированы")
        group_id = settings["bootstrap_group"] if not comparison and settings["band_method"] != "analytic" else None
        snapshot_fields = set(fields)
        if group_id in columns:
            snapshot_fields.add(group_id)
        replay_role = (settings.get("selection") or {}).get("holdout_column") if replay is not None else None
        if replay_role:
            if replay_role not in columns or columns[replay_role].kind != "text":
                raise ValueError("Колонка ролей недоступна")
            snapshot_fields.add(replay_role)
        holdout_key = cv_settings["holdout_key"] if comparison else None
        cv_strategy = cv_settings["strategy"] if run_cv else None
        base_cv_strategy = NESTED_STRATEGIES.get(cv_strategy, cv_strategy)
        if holdout_key and cv_strategy in NESTED_STRATEGIES:
            raise ValueError("Для вложенной CV выберите «Без тестовой выборки»")
        pair = cv_settings["holdout_pair"] if holdout_key else None
        pair = tuple(pair) if pair and any(pair) else None
        if pair is not None and (None in pair or pair[0] == pair[1]):
            raise ValueError("Для попарного теста выберите две разные модели и тестовую выборку")
        pair_options = None
        pair_group_id = None
        if pair is not None:
            pair_group_id = settings["bootstrap_group"]
            if pair_group_id and pair_group_id not in columns:
                raise ValueError("Колонка групп отсутствует")
            if pair_group_id:
                snapshot_fields.add(pair_group_id)
            block_length = settings["bootstrap_block_length"]
            pair_options = dict(confidence=editor.confidence.value()/100, resamples=settings["bootstrap_resamples"],
                                seed=settings["bootstrap_seed"],
                                margin=cv_settings["holdout_margin"],
                                block_length=block_length if not pair_group_id and block_length > 1 else None)
        if holdout_key:
            if holdout_key not in columns or columns[holdout_key].kind != "text":
                raise ValueError("Колонка ролей недоступна")
            snapshot_fields.add(holdout_key)
        split_key = None
        if base_cv_strategy in ("group", "time"):
            split_key = cv_settings["key"]
            if split_key not in columns:
                raise ValueError("Выберите существующую колонку группы или времени")
            snapshot_fields.add(split_key)
        if replay_rows is not None:
            rows = replay_rows
        else:
            rows = selection.row_ids if selection is not None else editor.owner.analysis_scope.rows()
        frame = project_frame(project, rows, snapshot_fields)
        if replay_rows is not None:
            missing = len(replay_rows)-len(frame)
            if missing:
                raise ValueError(tr("В проекте отсутствуют сохранённые обучающие строки: {0}", missing))
            if tuple(frame.index) != replay_rows:
                frame = frame.reindex(replay_rows)
            if replay_role:
                train, _ = holdout_masks(frame[replay_role].to_numpy())
                if not np.all(train):
                    raise ValueError("У сохранённых обучающих строк изменились роли выборки")
        if selection is not None and tuple(frame.index) != selection.row_ids:
            raise ValueError("Обучающая выборка изменилась. Выполните сравнение заново")
        cv_values = frame[split_key].to_numpy() if split_key else None
        excluded = np.zeros(len(frame), dtype=bool)
        for cid in set(fields[:2]):
            excluded |= exclusion_mask(project, frame.index, cid, "statistics").to_numpy()
        excluded_test = excluded.copy() if holdout_key else None
        for cid in set(fields[2:]) - set(fields[:2]):
            excluded |= exclusion_mask(project, frame.index, cid, "statistics").to_numpy()
        if replay is not None:
            editor.invalidate()
        generation = editor.generation
        comparison_generation = editor.comparison_generation
        labels = tuple(_column_label(columns[settings[key]]) for key in ("x", "y"))
    except (ValueError, KeyError) as exc:
        editor.status.setText(message(exc))
        return
    if selection is None and replay is None and project.view.get("curve_fitting") != settings:
        project.view["curve_fitting"] = settings
        editor.owner.view_changed()
    if comparison:
        editor.save_comparison_settings()
    editor.status.setText(tr("Выполняется расчёт…"))
    cancel_event = Event()
    editor._bootstrap_cancel = cancel_event
    comparison_cancel = Event()
    editor._comparison_cancel = comparison_cancel
    editor._running_comparison = run_cv

    def compute():
        arrays = {cid: classify_series(frame[cid], numeric=True)["number"].to_numpy(dtype=float)
                  for cid in set(fields)}
        valid = ~excluded
        for values in arrays.values():
            valid &= np.isfinite(values)
        if selection is not None and not np.all(valid):
            raise ValueError("Обучающая выборка изменилась. Выполните сравнение заново")
        if replay is not None and not np.all(valid):
            raise ValueError("Часть сохранённых обучающих строк непригодна для расчёта")
        holdout = None
        pair_bootstrap = pair_options
        test_ids = []
        included = int(valid.sum())
        if holdout_key:
            train_role, test_role = holdout_masks(frame[holdout_key].to_numpy())
            x, y = arrays[settings["x"]], arrays[settings["y"]]
            test = test_role & ~excluded_test & np.isfinite(x) & np.isfinite(y)
            valid &= train_role
            if not np.any(valid) or not np.any(test):
                raise ValueError("Для независимого теста нужны пригодные строки в обеих выборках")
            holdout = x[test], y[test]
            if pair_group_id:
                pair_bootstrap = dict(pair_options, groups=frame[pair_group_id].to_numpy()[test])
            test_ids = frame.index[test].tolist()
            included = int(valid.sum())+len(test_ids)
        # Invalid positive uncertainties are errors, never silently dropped.
        sx = arrays[settings["sx"]][valid] if settings["sx"] and settings["method"] == "odr" else None
        sy = arrays[settings["sy"]][valid] if settings["sy"] else None
        options = dict(method=settings["method"], loss=settings["loss"], f_scale=settings["f_scale"],
                       sx=sx, sy=sy, absolute_sigma=settings["absolute_sigma"], starts=settings["starts"], start_seed=settings["start_seed"])
        x, y = arrays[settings["x"]][valid], arrays[settings["y"]][valid]
        if run_cv:
            try:
                result = compare_models(x, y, candidates,
                                        folds=cv_settings["inner_folds"] if reselect else cv_settings["folds"],
                                        shuffle=cv_settings["shuffle"], seed=cv_settings["seed"],
                                        strategy=base_cv_strategy if reselect else cv_strategy, gap=cv_settings["gap"],
                                        inner_folds=cv_settings["inner_folds"],
                                        groups=cv_values[valid] if base_cv_strategy == "group" else None,
                                        times=cv_values[valid] if base_cv_strategy == "time" else None,
                                        cancelled=comparison_cancel.is_set, holdout=holdout, holdout_pair=pair,
                                        pair_bootstrap=pair_bootstrap, **options)
            except ComparisonCancelled:
                return None
            if comparison:
                return result, frame.index[valid].tolist(), len(frame)-included, None, None, test_ids, settings, None
        resolved = settings
        if selection is not None:
            if reselect:
                if not result.ranking:
                    raise ValueError("Ни одна модель не прошла итоговую CV")
                score = result.ranking[0]
                resolved = selection.resolve(score.name, score.metrics["RMSE"])
            else:
                resolved = selection.resolve(selection.winner, selection.rmse)
        model, initial, bounds = model_options if selection is None else _model_options(resolved)
        result = fit_curve(x, y, model, initial=initial, bounds=bounds, **options)
        bootstrap = error = None
        needs_bootstrap = (settings["band_method"] in BOOTSTRAP_BAND_METHODS or
                           (settings["band_method"] == "auto" and result.covariance is None))
        if needs_bootstrap:
            try:
                if group_id and group_id not in frame:
                    raise ValueError("Колонка групп отсутствует")
                grid = np.linspace(np.min(result.x), np.max(result.x), 400)
                block_length = settings["bootstrap_block_length"]
                bootstrap = bootstrap_bands(
                    result, grid, resamples=settings["bootstrap_resamples"], seed=settings["bootstrap_seed"],
                    prediction_sigma=settings["prediction_sigma"] or None,
                    groups=frame[group_id].to_numpy()[valid] if group_id else None,
                    block_length=block_length if not group_id and block_length > 1 else None,
                    simultaneous=settings["band_method"] == "bootstrap_simultaneous",
                    progress=lambda done, total: editor.bootstrap_progress.emit(generation, done, total),
                    cancelled=lambda: cancel_event.is_set() or (reselect and comparison_cancel.is_set()))
            except BootstrapCancelled:
                return None
            except ValueError as exc:
                error = str(exc)  # Keep the valid fit; do not publish partial bootstrap bands.
        row_ids = frame.index[valid].tolist()
        archive_diff = compare_archived_fit(replay, result, row_ids) if replay is not None else None
        return result, row_ids, len(frame)-included, bootstrap, error, [], resolved, archive_diff

    def display(payload):
        if generation != editor.generation or (run_cv and comparison_generation != editor.comparison_generation):
            return
        if payload is None:
            editor.status.setText(tr("Сравнение отменено") if comparison else
                                tr("Bootstrap отменён; нажмите «Рассчитать» для нового расчёта"))
            return
        result, editor.row_ids, dropped, editor.bootstrap_result, editor.bootstrap_error, test_ids, resolved, archive_diff = payload
        editor.fit_settings, editor.axis_labels = resolved, labels
        editor.excluded_count = dropped
        editor.status.setText(tr("Точек: {0}; исключено: {1}", len(editor.row_ids), dropped))
        if comparison:
            editor.comparison_result = result
            editor.holdout_row_ids = test_ids
            editor.cv_snapshot = cv_settings | {"folds": result.folds, "shuffle": result.shuffle}
            if pair_options is not None:
                editor.cv_snapshot["pair_bootstrap"] = dict(pair_options, group_column=pair_group_id)
            editor.show_comparison()
            return
        if selection is not None or replay is not None:
            editor._restoring_fit_settings = True
            try:
                editor.restore_settings(resolved)
                if replay_confidence is not None:
                    editor.confidence.setValue(replay_confidence*100)
            finally:
                editor._restoring_fit_settings = False
            project.view["curve_fitting"] = resolved
            editor.owner.view_changed()
        editor.result = result
        editor.archive_diff = archive_diff
        editor.tabs.setTabEnabled(editor.archive_diff_tab, archive_diff is not None)
        editor.display_diagnostics()
        for widget in (editor.show_confidence, editor.show_prediction):
            widget.setEnabled(editor.current_bands_available())
        editor.export_button.setEnabled(True)
        editor.export_plot_button.setEnabled(True)
        editor.show_summary()
        editor.tabs.setCurrentWidget(editor.plot_page)
        editor.redraw()

    editor.run_task(compute, display)
    # Inputs live in the workspace sidebar. Allow edits/cancellation while the
    # worker uses its snapshot; generation checks discard any stale result.
    editor.setEnabled(True)
    editor.cancel_bootstrap.setEnabled(False)
    editor.worker.finished.connect(lambda: editor.cancel_bootstrap.setEnabled(False))
    editor.worker.finished.connect(lambda: setattr(editor, "_running_comparison", False))


def show_bootstrap_progress(editor, generation, done, total):
    if generation == editor.generation and editor.worker is not None and editor.worker.isRunning():
        editor.status.setText(tr("Bootstrap: {0} / {1}", done, total))
        editor.cancel_bootstrap.setEnabled(True)
