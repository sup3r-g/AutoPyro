"""Export interpretation results and expert reviews with aligned comparative rows."""

import json

from autopyro.data_io.file_output import atomic_output, write_json
from autopyro.data_io.io import export_table_csv
from autopyro.model.sample_labels import sample_label, sample_label_fields

from .interpretation_review import review_history, review_state


def review_export(project, report, position):
    row_id = report['results'][position]['row_id']
    state, decision = review_state(project, report, position)
    sample = project.science.samples.get(row_id)
    return {
        'row_id': row_id,
        'state': state,
        'decision': decision,
        'history': review_history(project, report['scheme']['id'], row_id),
        'pyrogram_observation': sample.metadata.get('pyrogram_observation', '') if sample else '',
    }


def filtered_report(report, positions):
    snapshot = dict(report)
    for key in ('inputs', 'results', 'type_agreement'):
        if key in report:
            snapshot[key] = [report[key][i] for i in positions]
    if 'comparisons' in report:
        snapshot['comparisons'] = {name: filtered_report(diagram, positions)
                                   for name, diagram in report['comparisons'].items()}
    return snapshot


def write_result_snapshot(path, project, report, positions, *, scope, filter_mode, visible_count,
                          additional_filters=()):
    payload = {
        "format": "autopyro.interpretation-result",
        "version": 1,
        "report": report if scope == 'all' else filtered_report(report, positions),
        "selection": {"scope": scope, "filter": filter_mode, "visible_count": visible_count,
                      "exported_count": len(positions), "calculation_count": len(report['results'])},
        "expert_reviews": [review_export(project, report, position) for position in positions],
    }
    if additional_filters:
        payload["selection"]["additional_filters"] = list(additional_filters)
    write_json(path, payload)


def write_result_csv(path, project, report, positions, comparison_names, comparison_cells, *, sample_labels=None):
    """Write the selected rows using the dialog's localized comparison cells."""
    scheme = report['scheme']
    binding = report['binding']
    names = report.get('axis_column_names', {})
    x_name = names.get('x', scheme['x']['property'])
    y_name = names.get('y', scheme['y']['property'])
    if sample_labels is None:
        label_fields = sample_label_fields((column.id, column.name) for column in project.columns)
        sample_labels = {row.id: sample_label(row.id, row.values, label_fields) for row in project.rows}
    has_thresholds = any('threshold' in entry for entry in scheme.get('lines', []))

    def rows():
        for position in positions:
            item, sample = report['results'][position], report['inputs'][position]
            review = review_export(project, report, position)
            decision = review['decision'] or {}
            yield ([
                item["row_id"], item["status"],
                json.dumps(item["regions"], ensure_ascii=False),
                json.dumps(item["boundary_regions"], ensure_ascii=False),
                scheme["id"], scheme["revision"], report["scheme_hash"],
                json.dumps(item["line_hits"], ensure_ascii=False), item.get("value"),
                json.dumps(item.get("value_bounds"), ensure_ascii=False),
                json.dumps(sample.get("quality", []), ensure_ascii=False),
                json.dumps(sample.get("x_bounds")), json.dumps(sample.get("y_bounds")),
                json.dumps(item["uncertainty_regions"], ensure_ascii=False),
                json.dumps(item.get("possible_types", []), ensure_ascii=False),
                json.dumps(item.get("interval_details", {}), ensure_ascii=False),
            ] + comparison_cells(position)
               + ([report['type_agreement'][position]] if comparison_names else [])
               + ([json.dumps(item.get('thresholds', {}), ensure_ascii=False)] if has_thresholds else [])
               + [review['state'], decision.get('status'), decision.get('author'), decision.get('conclusion'),
                  decision.get('note'), decision.get('updated_at'), decision.get('scheme_revision'),
                  review['pyrogram_observation']]
               + [sample_labels.get(item['row_id'], item['row_id']),
                  sample.get('x'), x_name, binding['x_unit'],
                  sample.get('y'), y_name, binding['y_unit'], item.get('reason')])

    headers = [
        'row_id', 'status', 'regions', 'boundaries', 'scheme_id', 'revision', 'scheme_hash',
        'lines', 'value', 'value_bounds', 'quality', 'x_bounds', 'y_bounds',
        'uncertainty_regions', 'possible_types', 'interval_details',
    ] + comparison_names
    if comparison_names:
        headers.append('type_agreement')
    if has_thresholds:
        headers.append('thresholds')
    headers.extend([
        'review_state', 'review_decision', 'review_author', 'review_conclusion', 'review_note',
        'review_updated_at', 'review_scheme_revision', 'pyrogram_observation',
        'sample_label', 'x', 'x_column', 'x_unit', 'y', 'y_column', 'y_unit', 'reason',
    ])
    with atomic_output(path) as temporary:
        export_table_csv(temporary, headers, rows())
