"""Small immutable plot inputs and bounded, reusable numeric preparation.

Capture runs on the owning thread. Preparation reads only the captured values;
neither scientific stores nor Qt models are shared with a worker.
"""
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import pandas as pd

from autopyro.model.core import Column, Project
from autopyro.model.tabular import classify_series, project_frame


class PreparationCancelled(Exception):
    pass


@dataclass(slots=True)
class PlotRow:
    """Renderer-only row. Provenance/history remain in the owning document.

    Snapshot Projects are read-only render inputs, not editable documents.
    Avoid a per-row __dict__ and two unused provenance dictionaries.
    """
    values: dict
    id: str


@dataclass(slots=True)
class PlotRows(Sequence):
    """Read-only renderer rows backed by columns, not N retained row objects.

    Individual rows are created on access for existing renderers. Preparation
    builds its DataFrame directly from the columns without materializing rows.
    """
    ids: list[str]
    values_by_column: dict[str, list]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        rid = self.ids[index]
        return PlotRow({cid: values[index] for cid, values in self.values_by_column.items()}, rid)

    def to_frame(self, column_ids, row_ids=None):
        if row_ids is None:
            ids = self.ids
            data = {cid: self.values_by_column[cid] for cid in column_ids}
        else:
            wanted = set(row_ids)
            positions = [i for i, rid in enumerate(self.ids) if rid in wanted]
            ids = [self.ids[i] for i in positions]
            data = {cid: [self.values_by_column[cid][i] for i in positions] for cid in column_ids}
        return pd.DataFrame(data, index=ids, dtype=object)

    @classmethod
    def from_frame(cls, frame):
        return cls(frame.index.tolist(), {cid: frame[cid].tolist() for cid in frame.columns})


def check_cancelled(cancelled):
    if cancelled is not None and cancelled.is_set():
        raise PreparationCancelled()


def inputs_for(spec):
    from autopyro.model.filtering import leaf_rules

    local = {spec.x_id, spec.y_id, spec.z_id, spec.depth_bottom_id, spec.label_by,
             spec.color_by, spec.shape_by, spec.size_by, spec.line_group_by, *spec.track_ids, *spec.library_bindings.values()}
    local.update(rule['column'] for rule in leaf_rules(spec.filter_rules))
    for panel in spec.interpretation_panels:
        binding = panel.get('binding', {})
        local.update(binding.get(key) for key in ('x_id', 'y_id'))
    if spec.stratigraphy:
        binding = spec.stratigraphy['binding']
        local.update(binding.get(key) for key in ('well_id', 'depth_id', 'bottom_id'))
    global_ids = {cid for cid in (spec.color_by, spec.shape_by, spec.size_by) if cid}
    return local - {'', None}, global_ids


def _cell(value):
    return value if value is None or isinstance(value, (str, int, float, bool)) else deepcopy(value)


LIBRARY_MEASUREMENT_ERROR_KINDS = frozenset({
    'mpl:scatter', 'mpl:plot', 'mpl:step', 'mpl:errorbar', 'mpl:semilogx', 'mpl:semilogy', 'mpl:loglog',
    'mpl3d:scatter', 'mpl3d:errorbar',
})


def measurement_error_columns(spec):
    """Columns used by opt-in measurement intervals on ordinary plots."""
    return measurement_plot_columns(spec, uncertainty=True) if spec.measurement_error_bars else set()


def measurement_plot_columns(spec, *, uncertainty=False):
    if spec.interpretation or spec.interpretation_panels:
        return set()
    library_kinds = LIBRARY_MEASUREMENT_ERROR_KINDS if uncertainty else {'mpl:scatter'}
    if spec.kind in library_kinds and not spec.depth_bottom_id:
        from .plot_controls import controls_for

        return {spec.library_bindings.get(key, getattr(spec, dimension + '_id'))
                for dimension, key in controls_for(spec.kind).axes} - {'', None}
    if spec.kind not in ('scatter', 'depth', 'multi_depth'):
        return set()
    x_ids = spec.track_ids if spec.kind == 'multi_depth' else [spec.x_id]
    if spec.depth_bottom_id:
        return set(x_ids) - {'', None} if uncertainty and spec.kind in ('depth', 'multi_depth') else set()
    return {*x_ids, spec.y_id} - {'', None}


def input_token(project, revision, specs):
    """QC decisions can change without TableModel.data_revision changing."""
    qc = []
    for mid, decision in project.view.get('qc_exclusions', {}).get('plots', {}).items():
        m = project.science.measurements.get(mid)
        qc.append((mid, decision.get('revision'), m.revision if m else None,
                   project.science.working.get(m.sample_id, {}).get(m.property_id) if m else None))
    layers = []
    if any(spec.filter_rules for spec in specs):
        from autopyro.model.filtering import leaf_rules

        layers.append(project.view.get('qc_exclusions', {}))
        layers.append(sorted({rule['column'] for spec in specs for rule in leaf_rules(spec.filter_rules)}))
    for spec in specs:
        if spec.curve_archive_id:
            from .curve_layer import archived_curve
            record = archived_curve(project, spec.curve_archive_id) or {}
            layers.append(('curve_archive', spec.curve_archive_id, record.get('name'), record.get('at'),
                           record.get('source_digest'), record.get('stale')))
        if spec.stratigraphy:
            layers.append(project.view.get('stratigraphy', {}).get('scheme'))
        for binding in [spec.interpretation, *(p.get('binding') for p in spec.interpretation_panels)]:
            if binding:
                layers.append(project.view.get('interpretation_schemes', {}).get(binding['scheme']['id']))
    layers.append(sorted(set().union(*(measurement_error_columns(spec) for spec in specs))))
    layers.append(sorted(set().union(*(measurement_plot_columns(spec) for spec in specs if spec.qc_outline))))
    passport_axes = sorted({cid for spec in specs if spec.passport_id for cid in (spec.x_id, spec.y_id, spec.depth_bottom_id, *spec.track_ids) if cid})
    if passport_axes:
        layers.append(('passport_axes', passport_axes))
    report = project.view.get('interpretation_result')
    if report and any(spec.interpretation or spec.interpretation_panels for spec in specs):
        layers.append(('interval_report', report.get('created_at'), report.get('scheme_hash'), report.get('stale')))
    layer_hash = hashlib.sha256(json.dumps(layers, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return id(project), revision, tuple(sorted(qc)), layer_hash


@dataclass
class PlotInput:
    token: tuple
    project: Project
    local_ids: frozenset
    global_values: dict
    excluded: dict
    row_ids: frozenset

    def covers(self, token, ids, specs):
        local, global_ids = set(), set()
        for spec in specs:
            a, b = inputs_for(spec)
            local.update(a)
            global_ids.update(b)
        return (self.token == token and set(ids) <= self.row_ids
                and local <= self.local_ids and global_ids <= self.global_values.keys())


def capture_steps(project, row_ids, specs, token, previous=None, chunk_size=1024):
    """Yield between bounded chunks. Callers must verify the token before use."""
    ids = frozenset(row_ids)
    if previous is not None and previous.covers(token, ids, specs):
        return previous
    local, global_ids = set(), set()
    for spec in specs:
        a, b = inputs_for(spec)
        local.update(a)
        global_ids.update(b)
    columns = [Column(c.name, c.kind, c.id, c.unit, {
        'method': bool(c.metadata.get('method')), 'stale': bool(c.metadata.get('stale')),
        'calculation': bool(c.metadata.get('calculation')),
        'nonpyro': bool(c.metadata.get('nonpyro')),
        'stratigraphy': bool(c.metadata.get('stratigraphy')), 'basis': _cell(c.metadata.get('basis', '')),
    }) for c in project.columns]
    if any(spec.passport_id for spec in specs):
        from autopyro.schemes.interpretation_quality import column_property_name
        for column, original in zip(columns, project.columns, strict=True):
            if column.id in local:
                column.metadata['property_role'] = column_property_name(original)
    existing = {c.id for c in columns}
    # Invalid bindings are reported by the renderer, not by an unsafe dict lookup.
    local &= existing
    global_ids &= existing
    uncertainty = {}
    from .plot_qc import has_qc_status

    measurement_qc = {}
    qc_ids = set().union(*(measurement_plot_columns(spec) for spec in specs if spec.qc_outline))
    uncertainty_ids = set().union(*(measurement_error_columns(spec) for spec in specs))
    for spec in specs:
        for binding in [spec.interpretation, *(p.get('binding') for p in spec.interpretation_panels)]:
            if binding:
                uncertainty_ids.update(binding[key] for key in ('x_id', 'y_id'))
    captured_ids, values = [], {cid: [] for cid in local}
    from autopyro.model.filtering import leaf_rules
    from autopyro.analysis.qc import is_excluded

    filter_ids = {rule['column'] for spec in specs for rule in leaf_rules(spec.filter_rules)}
    filter_measurements = {}
    globals_ = {cid: {} for cid in global_ids}
    for position, row in enumerate(project.rows):
        if position and position % chunk_size == 0:
            yield
        if row.id in ids:
            captured_ids.append(row.id)
            for cid in local:
                values[cid].append(_cell(row.values.get(cid)))
                if cid in filter_ids:
                    m = project.science.current(row.id, cid)
                    info = dict(unit=m.unit, qualifier=m.qualifier, raw=m.raw_value) if m else {}
                    info['excluded'] = any(is_excluded(project, row.id, cid, scope)
                                           for scope in project.view.get('qc_exclusions', {}))
                    filter_measurements.setdefault(cid, {})[row.id] = info
                if cid in uncertainty_ids or cid in qc_ids:
                    m = project.science.current(row.id, cid)
                    if m is not None and cid in uncertainty_ids and m.uncertainty is not None:
                        uncertainty.setdefault(cid, {})[row.id] = {
                            'uncertainty': m.uncertainty, 'uncertainty_type': m.uncertainty_type,
                            'confidence_level': m.confidence_level, 'unit': m.unit,
                            'id': m.id, 'revision': m.revision}
                    if m is not None and cid in qc_ids and has_qc_status(m.qc_status):
                        measurement_qc.setdefault(cid, {})[row.id] = (m.id, m.revision, m.qc_status)
        for cid in global_ids:
            globals_[cid][row.id] = _cell(row.values.get(cid))
    excluded = {}
    for position, (mid, decision) in enumerate(project.view.get('qc_exclusions', {}).get('plots', {}).items()):
        if position and position % chunk_size == 0:
            yield
        measurement = project.science.measurements.get(mid)
        if (measurement is not None and decision.get('revision') == measurement.revision
                and project.science.working.get(measurement.sample_id, {}).get(measurement.property_id) == mid):
            excluded.setdefault(measurement.property_id, set()).add(measurement.sample_id)
    view = {'measurement_uncertainty': uncertainty, 'plot_measurement_qc': measurement_qc}
    view['plot_filter_measurements'] = filter_measurements
    view['plot_filter_stale'] = {column.id: column.metadata.get('stale')
                                for column in project.columns if column.id in filter_ids}
    archive_ids = {spec.curve_archive_id for spec in specs if spec.curve_archive_id}
    if archive_ids:
        from .curve_layer import curve_snapshot
        view['curve_fit_results'] = [curve_snapshot(record) for record in project.view.get('curve_fit_results', [])
                                     if record.get('id') in archive_ids]
    for spec in specs:
        if spec.stratigraphy:
            view['stratigraphy'] = {'scheme': deepcopy(project.view.get('stratigraphy', {}).get('scheme'))}
        for binding in [spec.interpretation, *(p.get('binding') for p in spec.interpretation_panels)]:
            if binding:
                sid = binding['scheme']['id']
                current = project.view.get('interpretation_schemes', {}).get(sid)
                if current:
                    view.setdefault('interpretation_schemes', {})[sid] = deepcopy(current)
    report = project.view.get('interpretation_result')
    if report and uncertainty:
        from autopyro.schemes.interpretation import interpretation_binding_key, saved_interval_markers

        marker_cache = {}
        for spec in specs:
            for binding in [spec.interpretation, *(p.get('binding') for p in spec.interpretation_panels)]:
                if not binding:
                    continue
                key = interpretation_binding_key(binding)
                if key in marker_cache:
                    continue
                uncertain_ids = (uncertainty.get(binding['x_id'], {}).keys()
                                 | uncertainty.get(binding['y_id'], {}).keys())
                marker_cache[key] = saved_interval_markers(project, binding, uncertain_ids)
        view['interval_ambiguity'] = marker_cache
    frozen = Project(columns=columns, rows=PlotRows(captured_ids, values), view=view, id=project.id)
    return PlotInput(token, frozen, frozenset(local), globals_, excluded, frozenset(captured_ids))


def capture_input(project, row_ids, specs, token, previous=None):
    """Synchronous capture for core APIs and explicitly requested immediate renders."""
    steps = capture_steps(project, row_ids, specs, token, previous)
    while True:
        try:
            next(steps)
        except StopIteration as finished:
            return finished.value


@dataclass
class PreparedPlot:
    source: PlotInput
    scope: frozenset
    policy: tuple
    local: pd.DataFrame
    global_frame: pd.DataFrame
    parsed: dict
    category_values: dict

    @property
    def project(self):
        return self.source.project

    def frame(self, row_ids=None, column_ids=None):
        data = self.global_frame if row_ids is None else self.local
        if row_ids is not None:
            wanted = set(row_ids)
            if wanted != self.scope:
                data = data.loc[data.index.isin(wanted)]
        return data.loc[:, list(column_ids)] if column_ids is not None else data

    def columns(self, frame, column_ids):
        result = {}
        for cid in dict.fromkeys(column_ids):
            data = self.parsed[(cid, cid in self.source.global_values, *self.policy)]
            result[cid] = data if data.index.equals(frame.index) else data.reindex(frame.index)
        return result

    def categories(self, cid):
        return self.category_values[cid]


class PlotDataCache:
    """One raw snapshot plus LRU parsed columns, with an explicit numeric budget.

    The budget covers retained classifications, not the active snapshot/renderer.
    A prepared result owns its frames even when an entry is too large to retain.
    """
    def __init__(self, max_bytes=128 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.sizes = {}
        self.bytes = 0
        self.token = None
        self.categories = {}
        self.snapshot = None
        self.hits = self.misses = 0
        self.frame_source = self.local_scope = self.local_frame = self.global_frame = None

    def fork(self):
        other = PlotDataCache(self.max_bytes)
        other.entries, other.sizes = self.entries.copy(), self.sizes.copy()
        other.bytes, other.token = self.bytes, self.token
        other.categories, other.snapshot = self.categories.copy(), self.snapshot
        other.hits, other.misses = self.hits, self.misses
        other.frame_source, other.local_scope = self.frame_source, self.local_scope
        other.local_frame, other.global_frame = self.local_frame, self.global_frame
        return other

    def prepare(self, source, row_ids, spec, cancelled=None):
        from .plotting import category_key, excluded_column_status
        check_cancelled(cancelled)
        if self.token != source.token:
            self.entries.clear()
            self.sizes.clear()
            self.categories.clear()
            self.bytes = 0
            self.token = source.token
        self.snapshot = source
        scope = frozenset(row_ids)
        policy = spec.text_numbers, spec.include_interpolated, spec.include_calculated, spec.include_original
        if self.frame_source is not source or self.local_scope != scope:
            rows = source.project.rows
            if isinstance(rows, PlotRows):
                selected = None if scope == source.row_ids else scope
                self.local_frame = rows.to_frame(source.local_ids, selected)
            else:
                self.local_frame = project_frame(source.project, scope, source.local_ids)
            self.local_scope = scope
        if self.frame_source is not source:
            self.global_frame = pd.DataFrame(source.global_values, dtype=object)
        self.frame_source = source
        local, global_frame = self.local_frame, self.global_frame
        columns = {c.id: c for c in source.project.columns}
        parsed = {}
        local_ids, global_ids = inputs_for(spec)
        for cid in local_ids & columns.keys():
            check_cancelled(cancelled)
            global_ = cid in source.global_values
            key = cid, None if global_ else scope, *policy
            data = self.entries.pop(key, None)
            if data is None:
                self.misses += 1
                values = global_frame[cid] if global_ else local[cid]
                column = columns[cid]
                excluded = excluded_column_status(column, spec)
                if excluded is not None:
                    data = pd.DataFrame({'status': excluded, 'number': None}, index=values.index, dtype=object)
                else:
                    data = classify_series(values, numeric=column.kind == 'number' or spec.text_numbers)
                mask = data.index.isin(source.excluded.get(cid, ()))
                data.loc[mask, ['status', 'number']] = ['qc_excluded', None]
                self.sizes[key] = int(data.memory_usage(index=True, deep=True).sum())
                self.bytes += self.sizes[key]
            else:
                self.hits += 1
            self.entries[key] = data
            parsed[(cid, global_, *policy)] = data
        category_ids = {cid for cid, enabled in ((spec.color_by, spec.color_mode == 'category'),
                        (spec.shape_by, True), (spec.size_by, spec.size_mode == 'category')) if cid and enabled}
        for cid in category_ids:
            if cid not in source.global_values:
                continue
            if cid not in self.categories:
                check_cancelled(cancelled)
                values = global_frame[cid]
                keys = values.map(category_key)
                unique = ~keys.duplicated()
                self.categories[cid] = list(zip(values.loc[unique].head(101), keys.loc[unique].head(101)))
        while self.bytes > self.max_bytes and self.entries:
            key, _ = self.entries.popitem(last=False)
            self.bytes -= self.sizes.pop(key)
        check_cancelled(cancelled)
        return PreparedPlot(source, scope, policy, local, global_frame, parsed,
                            {cid: self.categories[cid] for cid in category_ids if cid in self.categories})
