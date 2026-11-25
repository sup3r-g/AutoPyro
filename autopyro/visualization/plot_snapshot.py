"""Versioned, data-only input for repeating a graph export."""
from dataclasses import asdict
import json
from pathlib import Path

from autopyro.model.core import Column, Project
from .plot_data import PlotInput, PlotRows, check_cancelled, inputs_for
from .plotting import PlotSpec


FORMAT = 'autopyro.plot-snapshot'


def write_snapshot(path, job, cancelled=None):
    spec, source = job['spec'], job['source']
    local_ids, global_ids = inputs_for(spec)
    scope = set(job['ids'])
    rows = source.project.rows
    positions = None if scope == source.row_ids else [
        index for index, row_id in enumerate(rows.ids) if row_id in scope
    ]
    ids = rows.ids if positions is None else [rows.ids[index] for index in positions]
    values = {}
    for cid, column in rows.values_by_column.items():
        if cid in local_ids:
            check_cancelled(cancelled)
            values[cid] = column if positions is None else [column[index] for index in positions]
    view = dict(source.project.view)
    view['measurement_uncertainty'] = {
        cid: {row_id: item for row_id, item in metadata.items() if row_id in scope}
        for cid, metadata in view.get('measurement_uncertainty', {}).items() if cid in local_ids
    }
    if 'interval_ambiguity' in view:
        view['interval_ambiguity'] = {
            key: [sorted(covered & scope), {rid: value for rid, value in markers.items() if rid in scope}]
            for key, (covered, markers) in view['interval_ambiguity'].items()
        }
    if 'curve_fit_results' in view:
        view['curve_fit_results'] = [record for record in view['curve_fit_results']
                                     if record['id'] == spec.curve_archive_id]
    if not spec.stratigraphy:
        view.pop('stratigraphy', None)
    payload = {
        'format': FORMAT, 'version': 1, 'project_id': source.project.id,
        'data_revision': job['revision'], 'spec': spec.to_dict(), 'palette': job.get('palette'),
        'columns': [asdict(column) for column in source.project.columns if column.id in local_ids],
        'ids': ids, 'values': values, 'view': view,
        'global_values': {cid: column for cid, column in source.global_values.items() if cid in global_ids},
        'excluded': {cid: sorted(row_ids if cid in global_ids else row_ids & scope)
                     for cid, row_ids in source.excluded.items() if cid in local_ids},
    }
    check_cancelled(cancelled)
    with Path(path).open('w', encoding='utf-8') as stream:
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        for index, chunk in enumerate(encoder.iterencode(payload)):
            if index % 2048 == 0:
                check_cancelled(cancelled)
            stream.write(chunk)


def read_snapshot(path):
    with Path(path).open(encoding='utf-8') as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or payload.get('format') != FORMAT or payload.get('version') != 1:
        raise ValueError('Неизвестный формат снимка графика')
    spec = PlotSpec.from_dict(payload['spec'])
    columns = [Column(**column) for column in payload['columns']]
    ids, values = payload['ids'], payload['values']
    column_ids = {column.id for column in columns}
    if (not isinstance(ids, list) or not all(isinstance(rid, str) for rid in ids) or len(set(ids)) != len(ids)
            or not isinstance(values, dict) or values.keys() != column_ids or len(column_ids) != len(columns)
            or any(not isinstance(column, list) or len(column) != len(ids) for column in values.values())):
        raise ValueError('Некорректные строки или колонки снимка графика')
    global_values = payload['global_values']
    if (not isinstance(global_values, dict) or not global_values.keys() <= column_ids
            or any(not isinstance(column, dict) for column in global_values.values())):
        raise ValueError('Некорректные глобальные значения снимка графика')
    view = payload['view']
    if 'interval_ambiguity' in view:
        view['interval_ambiguity'] = {key: (set(covered), markers)
                                      for key, (covered, markers) in view['interval_ambiguity'].items()}
    project = Project(columns=columns, rows=PlotRows(ids, values), view=view, id=payload['project_id'])
    excluded = {cid: set(row_ids) for cid, row_ids in payload['excluded'].items()}
    source = PlotInput((object(),), project, frozenset(values), global_values, excluded, frozenset(ids))
    return spec, source, ids, payload.get('palette'), payload.get('data_revision')
