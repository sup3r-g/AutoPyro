"""User-supplied screening ranges, separate from geological classification."""
from collections import defaultdict

import pandas as pd

from autopyro.analysis.calculations import FORMULAS, UNITS
from autopyro.analysis.statistics import classify
from autopyro.model.tabular import classify_series, exclusion_mask

RANGES = {
    'TOC': (0, .1, 84, 100), 'S1': (0, .2, 2, 100),
    'S2': (0, .5, 100, 1000), 'TMAX': (350, 400, 500, 600),
    'HI': (0, 10, 900, 1200), 'OI': (0, 2, 5, 500),
}
CANONICAL_UNITS = {'TOC': '%wt', 'S1': 'mg HC/g rock', 'S2': 'mg HC/g rock',
                   'TMAX': '°C', 'HI': 'mg HC/g TOC', 'OI': 'mg CO2/g TOC'}


def property_name(value):
    name = value.rsplit(' / ', 1)[-1].split(' [', 1)[0].strip().upper().replace('-', '')
    if name.replace(' ', '') == 'S1+S2':
        return 'GP'
    return {'SI': 'S1'}.get(name, name)


def column_property_name(column):
    calculation = column.metadata.get('calculation')
    formula = calculation.get('formula') if isinstance(calculation, dict) else None
    if isinstance(formula, str) and formula in FORMULAS:
        return property_name(formula)
    role = column.metadata.get('property_role')
    return property_name(role if isinstance(role, str) and role.strip() else column.name)


class InterpretationQuality:
    """Resolve columns and unit factors once; check only selected row values."""

    def __init__(self, project, x_id, y_id, x_unit, y_unit, scheme, text_numbers):
        self.project = project
        self.text_numbers = text_numbers
        axes = {x_id: (property_name(scheme['x']['property']), x_unit),
                y_id: (property_name(scheme['y']['property']), y_unit)}
        self.axis_roles = {role for role, _ in axes.values()}
        axis_groups = {c.name.rsplit(' / ', 1)[0] if ' / ' in c.name else ''
                       for c in project.columns if c.id in axes}
        groups = defaultdict(list)
        for column in project.columns:
            group = column.name.rsplit(' / ', 1)[0] if ' / ' in column.name else ''
            if column.id not in axes and group not in axis_groups:
                continue
            role = axes[column.id][0] if column.id in axes else column_property_name(column)
            if role in RANGES:
                groups[role].append(column)
        required = set()
        if self.axis_roles <= RANGES.keys() | {'PI', 'GP'}:
            if 'TMAX' in self.axis_roles:
                required.add('S2')
            if self.axis_roles & {'HI', 'OI'}:
                required.add('TOC')
        self.coverage_issues = []
        for role in sorted(required):
            columns = groups.get(role, ())
            if any(column.id in axes for column in columns) or len(columns) == 1:
                continue
            reason = ('нет колонки показателя' if not columns
                      else 'несколько подходящих колонок, источник измерения не определён')
            self.coverage_issues.append(dict(column_id=None, property=role, code='coverage_unchecked',
                                             severity='warning', message=f'{role}: диапазон не проверен — {reason}'))
        self.columns = []
        for role, columns in groups.items():
            # Bound axes are authoritative; do not guess between duplicate ancillary properties.
            candidates = [c for c in columns if c.id in axes] or (columns if len(columns) == 1 else [])
            for column in candidates:
                unit = axes[column.id][1] if column.id in axes else column.unit
                unit = {'%': '%wt', '% wt': '%wt', 'C': '°C', 'degC': '°C'}.get(unit, unit)
                factors = UNITS.get(role, {CANONICAL_UNITS[role]: 1})
                self.columns.append((column, role, factors.get(unit), column.id in axes))

    def check(self, row):
        from autopyro.analysis.qc import is_excluded
        issues, blocked = [], set()
        for column, role, factor, axis in self.columns:
            def issue(code, text, severity='warning'):
                issues.append(dict(column_id=column.id, property=role, code=code,
                                   severity=severity, message=f'{column.name}: {text}'))
            if is_excluded(self.project, row.id, column.id, 'interpretation'):
                issue('qc_excluded', 'измерение исключено контролем качества')
                continue
            if column.metadata.get('method'):
                issue('interpolation_excluded', 'интерполированное значение не используется')
                continue
            status, value = classify(row.values.get(column.id), numeric=column.kind == 'number' or self.text_numbers)
            if status != 'numeric':
                issue('unavailable', f'нет допустимого числового значения ({status}); запись не подменяется числом')
                continue
            if factor is None:
                issue('unit_unchecked', 'диапазон не проверен: не объявлена поддерживаемая единица')
                continue
            value *= factor
            low, typical_low, typical_high, high = RANGES[role]
            unit = CANONICAL_UNITS[role]
            if not low <= value <= high:
                issue('absolute_range', f'{value:g} {unit} вне абсолютного диапазона {low:g}–{high:g}', 'error')
                if axis:
                    blocked.add(column.id)
            elif not typical_low <= value <= typical_high:
                note = ''
                if role == 'S2' and value < typical_low and self.axis_roles & {'TMAX', 'HI'}:
                    note = '; S2 ниже заданного типичного минимума: проверьте форму пика и надёжность Tmax/HI'
                if role == 'TOC' and value < typical_low and self.axis_roles & {'HI', 'OI'}:
                    note = '; низкий TOC: проверьте надёжность индексов, нормированных на TOC'
                issue('typical_range', f'{value:g} {unit} вне типичного диапазона {typical_low:g}–{typical_high:g}; требуется проверка' + note)
        issues.extend(self.coverage_issues)
        return issues, blocked

    def check_many(self, rows, *, axis_data=None):
        """Check columns in bulk and fill axis_data with iterators for screened axes."""
        rows = list(rows)
        if len(rows) < 256:
            return {row.id: self.check(row) for row in rows}
        frame = pd.DataFrame([row.values for row in rows], index=pd.Index([row.id for row in rows], dtype=object),
                             columns=[column.id for column, _, _, _ in self.columns], dtype=object)
        results = {row.id: ([], set()) for row in rows}
        for column, role, factor, axis in self.columns:
            def record(index, code, text, severity='warning'):
                for rid in index:
                    issues, blocked = results[rid]
                    message = text(rid) if callable(text) else text
                    issues.append(dict(column_id=column.id, property=role, code=code,
                                       severity=severity, message=f'{column.name}: {message}'))
                    if axis and code == 'absolute_range':
                        blocked.add(column.id)

            excluded = exclusion_mask(self.project, frame.index, column.id, 'interpretation')
            record(frame.index[excluded], 'qc_excluded', 'измерение исключено контролем качества')
            active = frame.loc[~excluded, column.id]
            if column.metadata.get('method'):
                record(active.index, 'interpolation_excluded', 'интерполированное значение не используется')
                if axis and axis_data is not None:
                    statuses = excluded.map({True: 'qc_excluded', False: 'interpolation_excluded'})
                    axis_data[column.id] = zip(statuses, [None] * len(rows), strict=True)
                continue
            data = classify_series(active, numeric=column.kind == 'number' or self.text_numbers)
            if axis and axis_data is not None:
                statuses, raw_numbers = data['status'], data['number']
                if excluded.any():
                    statuses = statuses.reindex(frame.index, fill_value='qc_excluded')
                    raw_numbers = raw_numbers.reindex(frame.index)
                    raw_numbers.loc[excluded] = None
                axis_data[column.id] = zip(statuses, raw_numbers, strict=True)
            missing = data['status'].ne('numeric')
            record(data.index[missing], 'unavailable',
                   lambda rid: f"нет допустимого числового значения ({data.at[rid, 'status']}); запись не подменяется числом")
            numbers = data.loc[~missing, 'number']
            if factor is None:
                record(numbers.index, 'unit_unchecked', 'диапазон не проверен: не объявлена поддерживаемая единица')
                continue
            values = numbers * factor
            low, typical_low, typical_high, high = RANGES[role]
            unit = CANONICAL_UNITS[role]
            absolute = ~values.between(low, high)
            record(values.index[absolute], 'absolute_range',
                   lambda rid: f'{values.at[rid]:g} {unit} вне абсолютного диапазона {low:g}–{high:g}', 'error')
            typical = ~absolute & ~values.between(typical_low, typical_high)
            note = ''
            if role == 'S2' and self.axis_roles & {'TMAX', 'HI'}:
                note = '; S2 ниже заданного типичного минимума: проверьте форму пика и надёжность Tmax/HI'
            if role == 'TOC' and self.axis_roles & {'HI', 'OI'}:
                note = '; низкий TOC: проверьте надёжность индексов, нормированных на TOC'
            record(values.index[typical], 'typical_range', lambda rid:
                   f'{values.at[rid]:g} {unit} вне типичного диапазона {typical_low:g}–{typical_high:g}; требуется проверка'
                   + (note if values.at[rid] < typical_low else ''))
        if self.coverage_issues:
            for issues, _ in results.values():
                issues.extend(self.coverage_issues)
        return results


def apply_quality(results, inputs):
    for result, sample in zip(results, inputs, strict=True):
        if sample.get('quality_blocked'):
            result.regions = []
            result.boundary_regions = []
            result.line_hits = []
            result.thresholds = {}
            result.uncertainty_regions = []
            result.distances = {}
            result.possible_types = []
            result.interval_details = {}
            result.value = result.value_bounds = None
            result.status = 'not_applicable'
            result.reason = 'Интерпретация заблокирована: значение оси вне абсолютного диапазона'
    return results
