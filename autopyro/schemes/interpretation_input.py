"""Capture only the working data needed by an interpretation worker."""

from copy import copy, deepcopy

from autopyro.model.core import Project, Row
from .interpretation_quality import InterpretationQuality


def capture_interpretation(project, ids, bindings):
    steps = capture_interpretation_steps(project, ids, bindings)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def capture_interpretation_steps(project, ids, bindings, *, chunk_size=512):
    wanted = set(ids)
    columns = set()
    axes = set()
    for binding in bindings:
        quality = InterpretationQuality(project, **binding)
        axes.update((binding['x_id'], binding['y_id']))
        columns.update(column.id for column, *_ in quality.columns)
    columns.update(axes)
    columns.discard('')
    snapshot = Project(id=project.id, columns=deepcopy(project.columns))
    exclusions = project.view.get('qc_exclusions', {}).get('interpretation', {})
    selected_exclusions = {}
    for index, row in enumerate(project.rows):
        if index and index % chunk_size == 0:
            yield
        if row.id not in wanted:
            continue
        snapshot.rows.append(Row(id=row.id, values=deepcopy({key: row.values.get(key) for key in columns})))
        working = project.science.working.get(row.id, {})
        for column_id in columns.intersection(working):
            mid = working[column_id]
            if column_id not in axes and mid not in exclusions:
                continue
            measurement = project.science.measurements[mid]
            record = copy(measurement)
            record.history = []
            snapshot.science.measurements[record.id] = record
            snapshot.science.working.setdefault(row.id, {})[column_id] = record.id
            if record.id in exclusions:
                selected_exclusions[record.id] = deepcopy(exclusions[record.id])
    snapshot.view['qc_exclusions'] = {'interpretation': selected_exclusions}
    return snapshot
