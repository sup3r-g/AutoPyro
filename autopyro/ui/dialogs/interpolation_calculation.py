"""Prepare depth interpolation and its diagnostics from a consistent project input."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from PySide6.QtCore import Qt

from autopyro.analysis.interpolation import DuplicateCoordinatesError, prepare
from autopyro.analysis.interpolation_constraints import well_column_ids
from autopyro.analysis.statistics import classify
from autopyro.model.audit import archive_protocol
from autopyro.model.tabular import excluded_sample_ids

from ..i18n import tr, message
from ..table.selection import sample_label_column
from ..tasks import TaskCancelled, run_with_progress
from ..widgets.documents import source_token
from ..widgets.stratigraphy_labels import geological_label
from .interpolation_diagnostics import diagnostic_labels, diagnostic_rows, duplicate_y_role
from .interpolation_results import BACKGROUND_ROWS, install_preview


def describe_calculation_groups(groups):
    fields = {
        "method": tr('Метод'), "laboratory": tr('Лаборатория'), "instrument": tr('Прибор'),
        "heating_program": tr('Программа нагрева'), "preparation": tr('Подготовка'),
        "fraction": tr('Фракция'), "unit": tr('Единица'), "basis": tr('База'),
    }
    values = defaultdict(lambda: defaultdict(set))
    for name, context in groups:
        for field in fields:
            values[name][field].add(context.get(field))
    return [
        " · ".join(part for part in [
            name, *(f"{label}: {context.get(field) or tr('не указано')}"
                    for field, label in fields.items() if len(values[name][field]) > 1),
        ] if part)
        for name, context in groups
    ]


def calculate(sheet):
    sheet.refresh_columns()
    sheet.invalidate(announce=False)
    if sheet.input_error():
        return
    if getattr(sheet.scope, '_analysis_scope', None) is not None:
        try:
            sheet.row_ids = set(sheet.owner.analysis_scope.rows())
        except ValueError as exc:
            sheet.summary.setText(tr('Ошибка: {0}', message(exc)))
            return
    group_ids = sheet.selected_group_ids() or [sheet.selected_well()]
    sheet.parameters = {
        "x_id": sheet.x.currentData(),
        "y_id": sheet.y.currentData(),
        "group_ids": group_ids,
        "method": sheet.method.currentData(),
        "max_gap": sheet.gap.value(),
        "constraints": {
            "well_id": sheet.well.currentData(), "analytical_context": True,
            "stratigraphy": deepcopy(sheet.project.view.get("stratigraphy", {})),
            "levels": [sheet.boundaries.item(i).data(Qt.ItemDataRole.UserRole)
                       for i in range(sheet.boundaries.count())
                       if sheet.boundaries.item(i).checkState() == Qt.CheckState.Checked],
        },
    }
    if sheet.extrapolate.isChecked():
        sheet.parameters["max_extrapolation"] = sheet.extrapolation_limit.value()
    if sheet.min_anchors.value() > sheet.min_anchors.minimum():
        sheet.parameters["min_anchors"] = sheet.min_anchors.value()
    if sheet.duplicates.currentData() == "exclude":
        sheet.parameters["duplicate_policy"] = "exclude"
    if bottom_id := sheet.interval_bottom.currentData():
        sheet.parameters["constraints"]["exclude_intervals_by"] = bottom_id
    token = source_token(sheet.owner) if sheet.property("workspaceTracked") else None
    axis_labels = sheet.x.currentText(), sheet.y.currentText()
    project, row_ids, parameters = sheet.project, set(sheet.row_ids), deepcopy(sheet.parameters)
    sample_id = sample_label_column(project.columns)
    well_ids = well_column_ids(parameters["constraints"])
    well_id = next(iter(well_ids), None)
    bound_well_id = well_ids[1] if len(well_ids) > 1 else None
    row_label = tr('Строка {0}')
    from autopyro.geology.stratigraphy import STATUSES
    labels = diagnostic_labels()
    reasons = {key: tr(label) for key, label in STATUSES.items()}
    group_columns = {c.id: c.name for c in project.columns if c.id in group_ids}
    well_columns = {c.id: c.name for c in project.columns if c.id in well_ids}

    def sample_details(row, number):
        sample, well = row.values.get(sample_id), row.values.get(well_id)
        return (row_label.format(number) if sample is None or sample == "" else str(sample),
                "" if well is None else str(well))

    try:
        def calculate(progress=None):
            stage, percent = tr('Расчёт значений'), 10

            def checkpoint():
                if progress:
                    progress(stage, percent)

            check = checkpoint if progress else None
            checkpoint()
            parameters["constraints"]["excluded_row_ids"] = sorted(
                row_ids & excluded_sample_ids(project, {parameters["x_id"], parameters["y_id"]}, "interpolation")
            )
            try:
                proposals, protocol = prepare(project, row_ids=row_ids, checkpoint=check, **parameters)
            except DuplicateCoordinatesError as exc:
                rows = []
                for number, row in enumerate(project.rows, 1):
                    if check and number % 256 == 0:
                        check()
                    if row.id not in exc.coordinates:
                        continue
                    sample, well = sample_details(row, number)
                    group = " · ".join(f"{group_columns[cid]} = {row.values[cid]}" for cid in group_ids)
                    group += f" · {duplicate_y_role(row.values[parameters['y_id']], labels)}"
                    rows.append([sample, well, str(exc.coordinates[row.id]),
                                 labels["duplicate_coordinate"], group, row.id])
                return exc, rows
            stage, percent = tr('Подготовка диагностики'), 65
            checkpoint()
            x_id, y_id = parameters["x_id"], parameters["y_id"]
            coordinates = {}
            measurements = []
            eligibility = protocol.get("eligibility", {})
            duplicate_details = {}
            for number, row in enumerate(protocol["rows"]):
                if check and number % 256 == 0:
                    check()
                x, y = row["values"][x_id], row["values"][y_id]
                coordinates[row["id"]] = x
                if x is not None and y is not None:
                    measurements.append((row["id"], x, y))
                if eligibility.get(row["id"], {}).get("status") == "duplicate_coordinate":
                    duplicate_details[row["id"]] = duplicate_y_role(y, labels)
            columns = {column["id"]: column for column in protocol["columns"]}
            basis = columns[y_id].get("metadata", {}).get("basis", "")
            measurement_context = {}
            for number, row in enumerate(protocol["rows"]):
                if check and number % 256 == 0:
                    check()
                x_meta = protocol["measurements"].get(row["working"].get(x_id), {})
                y_meta = protocol["measurements"].get(row["working"].get(y_id), {})
                measurement_context[row["id"]] = (
                    x_meta.get("unit", columns[x_id]["unit"]),
                    y_meta.get("unit", columns[y_id]["unit"]),
                    y_meta.get("basis", basis),
                )
            sample_labels = {}
            bound_wells = {}
            for number, row in enumerate(project.rows, 1):
                if check and number % 256 == 0:
                    check()
                if row.id not in eligibility:
                    continue
                sample_labels[row.id] = sample_details(row, number)
                if bound_well_id:
                    well = row.values.get(bound_well_id)
                    if well is not None and str(well).strip():
                        bound_wells[row.id] = str(well)
            support_spans = {}
            support_values = {}
            chart_group_by_anchor = {}
            chart_groups = []
            constraints = parameters["constraints"]
            levels = constraints["levels"]
            scheme = constraints["stratigraphy"].get("scheme", {})
            intervals = {item["id"]: item for item in scheme.get("intervals", ())} if levels else {}
            nonlinear = parameters["method"] in ("pchip", "spline")
            for group in protocol["supports"]:
                if not group["anchors"]:
                    continue
                group_index = len(chart_groups) + 1
                representative = group["anchors"][0]["row_id"]
                well = sample_labels[representative][1]
                if len(well_ids) > 1:
                    group_values = [f"{well_columns[cid]}: {value}"
                                    for cid, value in zip(well_ids, group["partition"])]
                    group_values.extend(str(value) for cid, value in zip(group_ids, group["group"], strict=True)
                                        if cid not in well_columns)
                else:
                    group_values = list(dict.fromkeys([well, *(str(value) for value in group["group"])]))
                if levels:
                    # The partition ends with interval IDs in the selected level order.
                    for interval_id in group["partition"][-len(levels):]:
                        interval = intervals[interval_id]
                        group_values.append(tr('{0}: {1} [{2}–{3} {4}]', geological_label(interval["level"]),
                                               interval["name"], interval["top"], interval["bottom"],
                                               scheme["depth_unit"]))
                context = eligibility[representative].get("analytical_context", {})
                for field, label in (("unit", tr('Единица')), ("basis", tr('База'))):
                    if value := context.get(field):
                        group_values.append(f"{label}: {value}")
                chart_groups.append((" · ".join(filter(None, group_values)), context))
                span = []
                last_x = None
                for number, anchor in enumerate(group["anchors"]):
                    if check and number % 256 == 0:
                        check()
                    x = anchor["x"]
                    if nonlinear and last_x is not None and x - last_x > parameters["max_gap"]:
                        span = []
                    rid = anchor["row_id"]
                    span.append(rid)
                    support_spans[rid] = span
                    support_values[rid] = (x, anchor["y"])
                    chart_group_by_anchor[rid] = group_index
                    last_x = x
            support_details = {}
            bounds = {}
            for number, proposal in enumerate(proposals):
                if check and number % 256 == 0:
                    check()
                left_x, left_y = support_values[proposal.left_id]
                right_x, right_y = support_values[proposal.right_id]
                low, high = sorted((left_y, right_y))
                if nonlinear:
                    span = support_spans[proposal.left_id]
                    if id(span) not in bounds:
                        values = [support_values[rid][1] for rid in span]
                        bounds[id(span)] = min(values), max(values)
                    low, high = bounds[id(span)]
                distance = min(abs(proposal.x - left_x), abs(proposal.x - right_x))
                support_details[proposal.row_id] = distance, not low <= proposal.value <= high
            proposal_roles = {p.row_id: "extrapolated" if p.extrapolated else "proposed" for p in proposals}
            diagnostics = diagnostic_rows(eligibility, labels, reasons, coordinates, sample_labels,
                                          support_spans, proposal_roles, duplicate_details=duplicate_details,
                                          checkpoint=check)
            chart_points, excluded_points = [], []
            for number, (rid, x, y) in enumerate(measurements):
                if check and number % 256 == 0:
                    check()
                _, x = classify(x, numeric=True)
                _, y = classify(y, numeric=True)
                if x is not None and y is not None:
                    points = chart_points if rid in support_spans else excluded_points
                    points.append((rid, x, y))
            stage, percent = tr('Сохранение снимка входов'), 90
            checkpoint()
            numeric_anchor = tr('Числовая опора')
            return (proposals, archive_protocol(protocol, checkpoint=check), diagnostics, chart_points, excluded_points,
                    sample_labels, support_spans, support_details, chart_group_by_anchor,
                    describe_calculation_groups(chart_groups), bound_wells, measurement_context,
                    sum(role == numeric_anchor for role in duplicate_details.values()))

        if max(len(row_ids), len(project.rows)) >= BACKGROUND_ROWS:
            result = run_with_progress(sheet, tr('Рассчитать предпросмотр'), calculate, cancellable=True)
        else:
            result = calculate()
    except TaskCancelled:
        previous = "\n" + tr('Показан предыдущий предпросмотр.') if sheet._diagnostic_rows else ""
        sheet.summary.setText(tr('Расчёт отменён.') + previous)
        sheet.announce_summary()
        return
    except (ValueError, TypeError, KeyError) as exc:
        previous = "\n" + tr('Показан предыдущий предпросмотр.') if sheet._diagnostic_column_id else ""
        detail = "\n".join(message(line) for line in str(exc).splitlines())
        sheet.summary.setText(tr('Ошибка: {0}', detail) + previous)
        sheet.announce_summary()
        return
    if sheet.property("workspaceTracked") and token != source_token(sheet.owner):
        sheet.invalidate()
        sheet.summary.setText(tr("Данные или выборка изменились. Рассчитайте предпросмотр заново."))
        return
    install_preview(sheet, result, parameters, axis_labels, token, well_columns, well_ids)
