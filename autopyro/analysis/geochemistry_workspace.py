"""Identity-based geochemical passports and descriptive comparisons; no Qt.

Links are non-destructive: original rows, analysis events and repeats survive.
One logical sample contributes at most one explicitly selected value per field.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from statistics import fmean, median, stdev

from autopyro.model.core import Column, new_id, timestamp
from autopyro.model.measurements import derived
from .nonpyro import (
    METHODS, REPORTED_BIOMARKER_METHODS, UNITS, alkane_distribution, field_matches, source_identity,
    STER_HOP_HOPANES, STER_HOP_STERANES, ster_hop_ratio, variant_tag,
)
from autopyro.model.tabular import classify


STATE_KEY = 'geochemistry_workspace'
MATERIAL_FIELDS = ('sample_type', 'preparation', 'rock_type', 'liquid_type', 'gas_type')
CATALOG_COLUMNS = ('Sample / Name', 'Sample / Type', 'Sample / Preparation', 'Well / Well Name',
                   'Sample / Top MD [m]', 'Sample / Base MD [m]', 'Sample / MD datum')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str, allow_nan=False).encode()).hexdigest()


def settings(project):
    return deepcopy(project.view.get(STATE_KEY, {}))


def _material(row, columns):
    labels = {'sample_type': 'Sample / Type', 'preparation': 'Sample / Preparation'}
    material = {}
    for key in MATERIAL_FIELDS:
        column_id = columns.get(labels.get(key))
        value = row.values.get(column_id) if column_id is not None else row.source.get(key)
        material[key] = str(value or '')
    return material


def _compatible(materials):
    return all(len({item[key] for item in materials if item.get(key)}) <= 1 for key in MATERIAL_FIELDS)


class GeochemistryIndex:
    def __init__(self, project, *, checkpoint=None, row_records=None):
        self.project = project
        self.rows = project.row_map()
        self.columns = {column.id: column for column in project.columns}
        self.by_name = {column.name: column.id for column in project.columns}
        self.well_column = next((self.by_name[name] for name in
                                 ('Well / Well Name', 'Скважина', 'Well') if name in self.by_name), None)
        self.depth_column = self.by_name.get('Sample / Top MD [m]')
        if self.depth_column is None:
            self.depth_column = next((column.id for column in project.columns
                                      if column.name == 'MD' and column.unit == 'm'), None)
        self.state = project.view.get(STATE_KEY, {})
        self.row_records = defaultdict(list) if row_records is None else row_records
        self.records_by_property = {}
        self.invalid_links = []
        measurements = project.science.measurements.values() if row_records is None else ()
        for position, measurement in enumerate(measurements):
            if checkpoint is not None and position % 16384 == 0:
                checkpoint()
            if measurement.sample_id in self.rows and measurement.property_id in self.columns:
                self.row_records[measurement.sample_id].append(measurement)
        self.samples = self._catalog(checkpoint)

    def _catalog(self, checkpoint=None):
        grouped = defaultdict(list)
        manual = {}
        memberships = Counter(rid for link in self.state.get('links', []) for rid in link.get('row_ids', ()))
        for link in self.state.get('links', []):
            ids = link.get('row_ids', [])
            if (not ids or any(rid not in self.rows or memberships[rid] > 1 for rid in ids)
                    or not _compatible([_material(self.rows[rid], self.by_name) for rid in ids])):
                self.invalid_links.append(link)
                continue
            for rid in ids:
                manual[rid] = link
        for position, row in enumerate(self.project.rows):
            if checkpoint is not None and position % 1024 == 0:
                checkpoint()
            material = _material(row, self.by_name)
            external = row.source.get('sample_id')
            endpoint = row.source.get('geochemistry_endpoint') or row.source.get('endpoint')
            if endpoint is None:
                # Appended source events retain their own endpoint in provenance.
                for measurement in self.records_for_row(row.id):
                    source = self.project.science.sources.get(measurement.source_id)
                    if source and source.provenance.get('endpoint'):
                        endpoint = source.provenance['endpoint']
                        break
            endpoint = endpoint or self.project.source.get('endpoint')
            if row.id in manual:
                key = 'link:' + manual[row.id]['id']
            elif external is not None and external != '' and endpoint:
                namespace = {k: str(endpoint.get(k, '')) for k in ('host', 'port', 'database')}
                namespace['host'] = namespace['host'].casefold()
                key = 'sample:' + _hash([namespace, str(external), material])
            else:
                key = 'row:' + row.id
            grouped[key].append(row)
        result = {}
        for position, (key, rows) in enumerate(grouped.items()):
            if checkpoint is not None and position % 1024 == 0:
                checkpoint()
            names = []
            for row in rows:
                name = row.values.get(self.by_name.get('Sample / Name'))
                well = row.values.get(self.well_column)
                depth = row.values.get(self.depth_column)
                location = ' · '.join(str(part) for part in (well, depth) if part is not None and part != '')
                names.append(str(name or location or row.id))
            names = list(dict.fromkeys(names))
            label = manual[rows[0].id].get('name') if key.startswith('link:') else None
            materials = [_material(row, self.by_name) for row in rows]
            material = {field: next((values[field] for values in materials if values[field]), '')
                        for field in MATERIAL_FIELDS}
            result[key] = {'id': key, 'name': label or names[0], 'row_ids': [row.id for row in rows],
                           'source_names': names, 'material': material, 'manual': key.startswith('link:'),
                           'source_record_ids': [row.source.get('source_record_id') for row in rows],
                           'source_sample_ids': list(dict.fromkeys(str(row.source['sample_id']) for row in rows
                                                                  if row.source.get('sample_id') is not None)),
                           'well': list(dict.fromkeys(str(row.values.get(self.well_column) or '')
                                                      for row in rows)),
                           'intervals': [{'top_md': row.values.get(self.depth_column),
                                          'base_md': row.values.get(self.by_name.get('Sample / Base MD [m]')),
                                          'datum': row.values.get(self.by_name.get('Sample / MD datum'))}
                                         for row in rows]}
        return result

    def records_for_row(self, row_id):
        return iter(self.row_records.get(row_id, ()))

    def records_for_property(self, row_id, column_id):
        # Passports and comparisons only need the property index for rows they inspect.
        if row_id not in self.records_by_property:
            by_property = defaultdict(list)
            for record in self.records_for_row(row_id):
                by_property[record.property_id].append(record)
            self.records_by_property[row_id] = by_property
        return self.records_by_property[row_id].get(column_id, ())

    def iter_catalog(self, row_ids=None):
        """Yield catalog entries for read-only consumers."""
        scope = None if row_ids is None else set(row_ids)
        return (item for item in self.samples.values()
                if scope is None or scope.intersection(item['row_ids']))

    def catalog(self, row_ids=None):
        return [deepcopy(item) for item in self.iter_catalog(row_ids)]

    def candidates(self, sample_id, column_id):
        sample = self.samples[sample_id]
        column = self.columns[column_id]
        result = []
        for rid in sample['row_ids']:
            row = self.rows[rid]
            records = self.records_for_property(rid, column_id)
            for record in records:
                analysis = self.project.science.analyses[record.analysis_id]
                provenance = self.project.science.sources.get(record.source_id or analysis.source_id)
                group, _ = source_identity(column)
                run = row.source.get('analysis_runs', {}).get(group, {})
                result.append({
                    'id': record.id, 'revision': record.revision, 'row_id': rid, 'column_id': column_id,
                    'raw': deepcopy(record.raw_value), 'value': record.table_value(), 'qualifier': record.qualifier,
                    'unit': record.unit, 'basis': record.basis, 'analysis_id': record.analysis_id,
                    'method': analysis.method, 'laboratory': analysis.laboratory,
                    'fraction': analysis.fraction, 'preparation': analysis.preparation,
                    'source_id': record.source_id or analysis.source_id,
                    'source_record_id': row.source.get('source_record_id'),
                    'analysis_run_id': analysis.metadata.get('analysis_run_id', run.get('analysis_run_id')),
                    'repeat_number': analysis.metadata.get('repeat_number', run.get('repeat_number')),
                    'source': deepcopy(provenance.provenance) if provenance else deepcopy(row.source),
                    'working': self.project.science.working.get(rid, {}).get(column_id) == record.id,
                    'qc_status': record.qc_status,
                })
            if not records and not derived(column):
                raw = row.values.get(column_id)
                if raw is not None:
                    result.append({'id': f'cell:{rid}:{column_id}', 'revision': 0, 'row_id': rid,
                                   'column_id': column_id, 'raw': deepcopy(raw), 'value': raw,
                                   'unit': column.unit, 'basis': column.metadata.get('basis', ''),
                                   'qualifier': 'exact', 'working': True, 'source': deepcopy(row.source),
                                   'source_record_id': row.source.get('source_record_id'), 'qc_status': 'unchecked'})
        return result

    def value(self, sample_id, column_id):
        column = self.columns[column_id]
        if derived(column):
            return {'status': 'derived_excluded', 'value': None, 'candidates': []}
        candidates = self.candidates(sample_id, column_id)
        choice = self.state.get('selections', {}).get(sample_id, {}).get(column_id)
        if choice:
            chosen = [item for item in candidates if item['id'] == choice['measurement_id']]
            if not chosen or chosen[0]['revision'] != choice['revision']:
                return {'status': 'stale_selection', 'value': None, 'candidates': candidates}
        else:
            chosen = [item for item in candidates if item['working'] and item['value'] is not None]
        if len(chosen) != 1:
            return {'status': 'ambiguous' if len(chosen) > 1 else 'missing', 'value': None, 'candidates': candidates}
        item = chosen[0]
        base = {'measurement': item, 'candidates': candidates, 'value': None}
        if item['qualifier'] != 'exact':
            return {**base, 'status': item['qualifier']}
        status, number = classify(item['value'], numeric=column.kind == 'number')
        if status != 'numeric':
            return {**base, 'status': status}
        if self.statistics_exclusion(item) is not None:
            return {**base, 'status': 'qc_excluded'}
        target_basis = column.metadata.get('basis', '')
        if target_basis and item['basis'] and target_basis != item['basis']:
            return {**base, 'status': 'basis_mismatch'}
        family = field_family(column)
        if item['unit'] != column.unit:
            units = UNITS.get(family, {})
            if family == 'alkane_signal' or item['unit'] not in units or column.unit not in units:
                return {**base, 'status': 'unit_mismatch'}
            number = float(number) * units[item['unit']] / units[column.unit]
        if not math.isfinite(float(number)):
            return {**base, 'status': 'non_finite'}
        return {**base, 'status': 'numeric', 'value': number}

    def statistics_exclusion(self, measurement):
        exclusion = self.project.view.get('qc_exclusions', {}).get('statistics', {}).get(measurement['id'])
        return exclusion if exclusion is not None and exclusion.get('revision') == measurement['revision'] else None

    def passport(self, sample_id):
        sample = deepcopy(self.samples[sample_id])
        measurements = []
        populated = {record.property_id for rid in sample['row_ids'] for record in self.records_for_row(rid)}
        populated.update(cid for rid in sample['row_ids'] for cid, value in self.rows[rid].values.items()
                         if value is not None)
        present_by_group = defaultdict(list)
        present_by_id = {}
        for column in self.columns.values():
            if column.id not in populated or derived(column):
                continue
            state = self.value(sample_id, column.id)
            group, label = source_identity(column)
            if any(candidate['value'] is not None for candidate in state['candidates']):
                present_by_id[column.id] = (column, state)
                present_by_group[group].append((label, column, state))
            for candidate in state['candidates']:
                chosen = state.get('measurement', {}).get('id') == candidate['id']
                exclusion = self.statistics_exclusion(candidate)
                status = state['status'] if chosen else 'qc_excluded' if exclusion is not None else 'not_selected'
                measurements.append({**candidate, 'name': column.name,
                                     'group': group,
                                     'chosen': chosen, 'comparison_status': status,
                                     **({'statistics_exclusion': deepcopy(exclusion)} if exclusion is not None else {})})
        availability = []
        for method_id, definition in METHODS.items():
            roles = {}
            saved = self.project.view.get('nonpyro_configs', {}).get(method_id, {}).get('inputs', {})
            for key, role in definition['roles'].items():
                if role['optional']:
                    continue
                bound = saved.get(key, {}).get('column_id')
                if bound in self.columns:
                    present = [present_by_id[bound]] if bound in present_by_id else []
                else:
                    present = [(column, value) for group in role['groups']
                               for label, column, value in present_by_group[group]
                               if field_matches(group, label, role)]
                if len(present) == 1:
                    column, value = present[0]
                    status = value['status']
                    if status == 'numeric' and column.unit not in UNITS[role['unit_family']]:
                        status = 'unit_mismatch'
                    roles[key] = {'status': status, 'label': role['label'], 'column_id': column.id}
                else:
                    roles[key] = {'status': 'ambiguous' if len(present) > 1 else 'missing', 'label': role['label']}
            identities = [source_identity(self.columns[item['column_id']]) for item in roles.values()
                          if item.get('column_id')]
            groups = {group for group, _ in identities if group}
            variants = {variant_tag(label) for _, label in identities if variant_tag(label)}
            units = {self.columns[item['column_id']].unit for item in roles.values() if item.get('column_id')}
            if ((definition['same_group'] and len(groups) > 1) or len(variants) > 1
                    or (method_id == 'alkane_indices' and len(units) > 1)):
                for item in roles.values():
                    if item['status'] == 'numeric':
                        item['status'] = 'incompatible_inputs'
            ready = all(item['status'] == 'numeric' for item in roles.values())
            availability.append({'method_id': method_id, 'name': definition['name'], 'roles': roles,
                                 'status': 'inputs_ready' if ready else 'needs_inputs'})
        results = []
        for method_id, report in self.project.view.get('nonpyro_results', {}).items():
            for result in report.get('results', []):
                if result.get('row_id') in sample['row_ids']:
                    results.append({'method_id': method_id, 'stale': bool(report.get('stale')),
                                    **deepcopy(result)})
        relations = [deepcopy(link) for link in self.state.get('relationships', [])
                     if sample_id in (link['parent'], link['child'])]
        for relation in relations:
            missing = [sid for sid in (relation['parent'], relation['child']) if sid not in self.samples]
            if missing:
                relation['missing_sample_ids'] = missing
        return {'version': 1, 'sample': sample, 'measurements': measurements,
                'availability': availability, 'results': results, 'relationships': relations}


def field_family(column):
    families = {role['unit_family'] for definition in METHODS.values() for role in definition['roles'].values()
                if field_matches(*source_identity(column), role)}
    return next(iter(families)) if len(families) == 1 else None


def link_samples(project, sample_ids, name):
    index = GeochemistryIndex(project)
    ids = list(sample_ids)
    if len(ids) < 2 or len(ids) != len(set(ids)) or any(sid not in index.samples for sid in ids):
        raise ValueError('Выберите не менее двух разных проб')
    rows = [rid for sid in ids for rid in index.samples[sid]['row_ids']]
    if not _compatible([_material(index.rows[rid], index.by_name) for rid in rows]):
        raise ValueError('Разные состояния материала нельзя объединить в одну пробу')
    if not name.strip():
        raise ValueError('Укажите название объединённой пробы')
    state = settings(project)
    if any(sid in (link['parent'], link['child']) for link in state.get('relationships', []) for sid in ids):
        raise ValueError('Сначала удалите связи исходного и производного материала')
    state['links'] = [link for link in state.get('links', []) if not set(rows).intersection(link['row_ids'])]
    state['links'].append({'id': new_id(), 'name': name.strip(), 'row_ids': rows})
    return state


def unlink_sample(project, sample_id):
    state = settings(project)
    if not sample_id.startswith('link:'):
        raise ValueError('Эта проба не имеет ручного объединения')
    if any(sample_id in (link['parent'], link['child']) for link in state.get('relationships', [])):
        raise ValueError('Сначала удалите связи исходного и производного материала')
    state['links'] = [link for link in state.get('links', []) if 'link:' + link['id'] != sample_id]
    return state


def select_measurement(project, sample_id, column_id, measurement_id):
    index = GeochemistryIndex(project)
    state = settings(project)
    chosen = state.setdefault('selections', {}).setdefault(sample_id, {})
    if measurement_id is None:
        chosen.pop(column_id, None)
        return state
    candidates = index.candidates(sample_id, column_id)
    item = next((item for item in candidates if item['id'] == measurement_id), None)
    if item is None:
        raise ValueError('Измерение относится к другой пробе или показателю')
    chosen[column_id] = {'measurement_id': item['id'], 'revision': item['revision']}
    return state


def relate_materials(project, parent, child, kind):
    index = GeochemistryIndex(project)
    if parent == child or parent not in index.samples or child not in index.samples or kind not in ('extract', 'residue'):
        raise ValueError('Укажите разные исходный и производный материалы')
    state = settings(project)
    links = state.setdefault('relationships', [])
    if any(link['child'] == child for link in links):
        raise ValueError('Для производного материала уже задана исходная проба')
    descendants, pending = set(), [child]
    while pending:
        current = pending.pop()
        if current not in descendants:
            descendants.add(current)
            pending.extend(link['child'] for link in links if link['parent'] == current)
    if parent in descendants:
        raise ValueError('Связь материалов создаёт цикл')
    links.append({'parent': parent, 'child': child, 'kind': kind})
    return state


def save_group(project, name, sample_ids):
    index = GeochemistryIndex(project)
    ids = list(sample_ids)
    if not name.strip() or not ids or len(set(ids)) != len(ids) or any(sid not in index.samples for sid in ids):
        raise ValueError('Укажите название и существующие пробы группы')
    state = settings(project)
    state.setdefault('groups', {})[name.strip()] = ids
    return state


def _summary(values, total, excluded):
    return {'total': total, 'n': len(values), 'excluded': dict(excluded),
            'mean': fmean(values) if values else None, 'median': median(values) if values else None,
            'sd': stdev(values) if len(values) > 1 else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def _comparison_inputs(project, a, b, column_ids, allow_mixed_material):
    index = GeochemistryIndex(project)
    if not a or not b or len(set(a)) != len(a) or len(set(b)) != len(b) or set(a).intersection(b):
        raise ValueError('Выберите две непересекающиеся группы разных проб')
    if any(sid not in index.samples for sid in [*a, *b]):
        raise ValueError('Состав группы устарел: обновите выбор проб')
    if not _compatible([index.samples[sid]['material'] for sid in [*a, *b]]) and not allow_mixed_material:
        raise ValueError('Подтвердите сравнение разных состояний материала')
    if not column_ids or len(set(column_ids)) != len(column_ids):
        raise ValueError('Выберите показатели сравнения')
    fields, points = [], {}
    for cid in column_ids:
        if cid not in index.columns or derived(index.columns[cid]) or index.columns[cid].kind != 'number':
            raise ValueError('Для сравнения нужны исходные числовые показатели')
        column = index.columns[cid]
        fields.append({'id': cid, 'name': column.name, 'unit': column.unit, 'metadata': deepcopy(column.metadata)})
        points[cid] = {sid: index.value(sid, cid) for sid in [*a, *b]}
        bases = {value['measurement']['basis'] for value in points[cid].values()
                 if value['status'] == 'numeric' and value['measurement'].get('basis')}
        if len(bases) > 1:
            for value in points[cid].values():
                if value['status'] == 'numeric':
                    value.update(status='basis_mismatch', value=None)
    return {'samples': {sid: deepcopy(index.samples[sid]) for sid in [*a, *b]},
            'fields': fields, 'points': points, 'settings': settings(project)}


def compare_samples(project, group_a, group_b, column_ids, *, allow_mixed_material=False, names=('A', 'B'),
                    saved_groups=(None, None)):
    request = {'group_a': list(group_a), 'group_b': list(group_b), 'column_ids': list(column_ids),
               'allow_mixed_material': bool(allow_mixed_material), 'names': list(names)}
    if any(name is not None for name in saved_groups):
        request['saved_groups'] = list(saved_groups)
    inputs = _comparison_inputs(project, **{key: request[key] for key in
                                ('column_ids', 'allow_mixed_material')}, a=request['group_a'], b=request['group_b'])
    exclusions = project.view.get('qc_exclusions', {}).get('statistics', {})
    qc = {}
    results = []
    for field in inputs['fields']:
        values = inputs['points'][field['id']]
        if exclusions:
            for point in values.values():
                if point['status'] == 'qc_excluded':
                    measurement = point['measurement']
                    mid = measurement['id']
                    if comment := exclusions.get(mid, {}).get('comment'):
                        qc[mid] = {'revision': measurement['revision'], 'comment': comment}
        summaries = []
        for ids in (group_a, group_b):
            numbers = [values[sid]['value'] for sid in ids if values[sid]['status'] == 'numeric']
            excluded = Counter(values[sid]['status'] for sid in ids if values[sid]['status'] != 'numeric')
            summaries.append(_summary(numbers, len(ids), excluded))
        a, b = summaries
        difference = b['mean'] - a['mean'] if a['n'] and b['n'] else None
        results.append({'field': field, 'a': a, 'b': b, 'difference': difference})
    return {'version': 1, 'created_at': timestamp(), 'request': request, 'inputs': inputs,
            'input_digest': _comparison_digest(inputs), 'results': results, **({'qc_exclusions': qc} if qc else {})}


def _comparison_digest(inputs):
    return _hash({key: inputs[key] for key in ('samples', 'fields', 'points')})


def comparison_is_current(project, report):
    try:
        request = report['request']
        saved_inputs = report.get('inputs')
        current_groups = settings(project).get('groups', {})
        for suffix, name in zip(('a', 'b'), request.get('saved_groups', (None, None)), strict=True):
            if name is not None and current_groups.get(name) != request['group_' + suffix]:
                return False
        inputs = _comparison_inputs(project, request['group_a'], request['group_b'], request['column_ids'],
                                    request['allow_mixed_material'])
        if report.get('version') != 1:
            return False
        digest = report['input_digest']
        if digest in (_comparison_digest(inputs), _hash(inputs)):
            return True
        if saved_inputs is None:
            return False
        if 'settings' not in saved_inputs:
            current_snapshot = freeze_comparison({**report, 'inputs': inputs})
            return current_snapshot['inputs'] == saved_inputs
        saved_digest = _comparison_digest(saved_inputs)
        return digest in (saved_digest, _hash(saved_inputs)) and _comparison_digest(inputs) == saved_digest
    except (ValueError, KeyError, TypeError):
        return False


PROFILE_REVIEWS_KEY = 'geochemistry_profile_reviews'


def _pair_evidence(report, fields):
    request = report['request']
    evidence = []
    for field in fields:
        points = report['inputs']['points'][field['id']]
        a, b = points[request['group_a'][0]], points[request['group_b'][0]]
        difference = b['value'] - a['value'] if a['status'] == b['status'] == 'numeric' else None
        evidence.append({'field': deepcopy(field), 'a': {'value': a['value'], 'status': a['status']},
                         'b': {'value': b['value'], 'status': b['status']}, 'difference': difference})
    return evidence


def compare_alkane_profiles(report):
    """Compare two complete GC profiles on the same fraction and measurement basis."""
    request = report['request']
    if len(request['group_a']) != 1 or len(request['group_b']) != 1:
        raise ValueError('Для сопоставления профилей выберите по одной пробе в A и B')
    roles = METHODS['alkane_indices']['roles']
    bindings, identities, supplementary = {}, set(), []
    for field in report['inputs']['fields']:
        column = Column(**field, kind='number')
        group, label = source_identity(column)
        role = next((key for key, definition in roles.items() if field_matches(group, label, definition)), None)
        basis = variant_tag(label)
        expected_unit = {'a': 'count', 'ca': 'ppm (µg/g)'}.get(basis)
        if role is None:
            supplementary.append(field)
            continue
        if role in bindings or column.unit != expected_unit:
            raise ValueError('Выберите один набор пиков nC24–nC34 с одной основой измерения')
        bindings[role] = field['id']
        identities.add((group, basis))
    if bindings.keys() != roles.keys() or len(identities) != 1:
        raise ValueError('Выберите один набор пиков nC24–nC34 с одной основой измерения')
    group, basis = identities.pop()
    profiles, measurement_bases = [], set()
    for sample in (request['group_a'][0], request['group_b'][0]):
        observations = {role: report['inputs']['points'][field][sample] for role, field in bindings.items()}
        if any(point['status'] != 'numeric' or point['value'] < 0 for point in observations.values()):
            raise ValueError('Для сопоставления нужны полные профили без недопущенных или отрицательных пиков')
        measurements = [point['measurement'] for point in observations.values()]
        analyses = {item.get('analysis_id') or item['row_id'] for item in measurements}
        runs = {item.get('analysis_run_id') for item in measurements if item.get('analysis_run_id') is not None}
        if len(analyses) != 1 or len(runs) > 1:
            raise ValueError('Пики каждой пробы должны относиться к одному анализу')
        measurement_bases.update(item['basis'] for item in measurements if item.get('basis'))
        distribution = alkane_distribution({'canonical_inputs': {key: point['value'] for key, point in observations.items()}})
        if not distribution:
            raise ValueError('Для сопоставления сумма пиков каждой пробы должна быть положительной')
        profiles.append(dict(distribution))
    if len(measurement_bases) > 1:
        raise ValueError('Выберите один набор пиков nC24–nC34 с одной основой измерения')
    a, b = profiles
    return {'version': 1, 'input_digest': report['input_digest'], 'group': group, 'basis': basis,
            'evidence': _pair_evidence(report, supplementary),
            'overlap_percent': min(100.0, math.fsum(min(a[carbon], b[carbon]) for carbon in a)),
            'peaks': [{'carbon': carbon, 'a_percent': a[carbon], 'b_percent': b[carbon],
                       'difference_pp': b[carbon] - a[carbon]} for carbon in a]}


def _ster_hop_evidence(inputs, sample_ids):
    roles = METHODS['ster_hop_from_peaks']['roles']
    by_basis = {'h': {}, 'a': {}}
    for field in inputs['fields']:
        group, label = source_identity(Column(**field, kind='number'))
        peak, _, _ = label.partition('[')
        if peak in roles and field_matches(group, label, roles[peak]):
            by_basis[variant_tag(label)].setdefault(peak, []).append(field)
    for basis, peaks in by_basis.items():
        if peaks.keys() != roles.keys():
            if peaks.keys() & set(STER_HOP_STERANES) and peaks.keys() & set(STER_HOP_HOPANES):
                yield {
                    'field': {'id': f'ster_hop_from_peaks:{basis}', 'name': f'Ster/Hop · 17/16 [{basis}]',
                              'unit': '1', 'metadata': {}},
                    'input_field_ids': [],
                    'missing_peaks': [peak for peak in roles if peak not in peaks],
                    'a': {'status': 'missing', 'value': None},
                    'b': {'status': 'missing', 'value': None},
                    'difference': None,
                }
            continue
        field_ids = [field['id'] for candidates in peaks.values() for field in candidates]
        sides, measurement_bases = [], set()
        for sample in sample_ids:
            status, value = 'numeric', None
            if any(len(candidates) != 1 for candidates in peaks.values()):
                status = 'ambiguous'
            elif any(candidates[0]['unit'] != 'count' for candidates in peaks.values()):
                status = 'unit_mismatch'
            else:
                observations = {peak: inputs['points'][fields[0]['id']][sample]
                                for peak, fields in peaks.items()}
                unavailable = next((point['status'] for point in observations.values()
                                    if point['status'] != 'numeric'), None)
                if unavailable:
                    status = unavailable
                elif any(point['value'] < 0 for point in observations.values()):
                    status = 'invalid_range'
                else:
                    measurements = [point['measurement'] for point in observations.values()]
                    analyses = {item.get('analysis_id') or item['row_id'] for item in measurements}
                    runs = {item['analysis_run_id'] for item in measurements if item.get('analysis_run_id') is not None}
                    if len(analyses) != 1 or len(runs) > 1:
                        status = 'incompatible_inputs'
                    else:
                        measurement_bases.update(item['basis'] for item in measurements if item.get('basis'))
                        value = ster_hop_ratio({peak: point['value'] for peak, point in observations.items()})
                        if value is None:
                            status = 'zero_denominator'
                        elif not math.isfinite(value):
                            status, value = 'non_finite', None
            sides.append({'status': status, 'value': value})
        if len(measurement_bases) > 1:
            sides = [{'status': 'basis_mismatch', 'value': None} if point['status'] == 'numeric' else point
                     for point in sides]
        a, b = sides
        yield {
            'field': {'id': f'ster_hop_from_peaks:{basis}', 'name': f'Ster/Hop · 17/16 [{basis}]',
                      'unit': '1', 'metadata': {}},
            'input_field_ids': field_ids,
            'a': a, 'b': b,
            'difference': b['value'] - a['value'] if a['status'] == b['status'] == 'numeric' else None,
        }


def correlate_oil_profiles(report, *, confirm_c15_vpdb=False):
    """Build a descriptive evidence sheet for oils and source-rock extracts."""
    request, inputs = report['request'], report['inputs']
    if len(request['group_a']) != 1 or len(request['group_b']) != 1:
        raise ValueError('Для корреляции выберите по одной пробе в A и B')
    sample_ids = (request['group_a'][0], request['group_b'][0])
    kinds = []
    for sample_id in sample_ids:
        material = inputs['samples'][sample_id]['material']
        sample_type, preparation = (material[key].strip().casefold() for key in ('sample_type', 'preparation'))
        if sample_type in ('oil', 'нефть'):
            kinds.append('oil')
        elif sample_type in ('rock', 'порода') and preparation in ('extract', 'экстракт'):
            kinds.append('rock_extract')
        else:
            raise ValueError('Укажите тип пробы «нефть» или тип «порода» с подготовкой «экстракт»')
    if 'oil' not in kinds:
        raise ValueError('Для корреляции нужна хотя бы одна проба нефти')

    identities = [source_identity(Column(**field, kind='number')) for field in inputs['fields']]
    peak_roles = METHODS['alkane_indices']['roles'].values()
    overlap_issue = None
    match = {'version': 1, 'input_digest': report['input_digest'], 'group': None, 'basis': None,
             'evidence': _pair_evidence(report, inputs['fields']), 'overlap_percent': None, 'peaks': []}
    if any(field_matches(group, label, role) for group, label in identities for role in peak_roles):
        try:
            candidate = compare_alkane_profiles(report)
            if candidate['group'] != 'Sat-GC':
                raise ValueError('Для корреляции нужны профили насыщенной фракции Sat-GC')
            match = candidate
        except ValueError as exc:
            overlap_issue = str(exc)
    if overlap_issue:
        match['overlap_issue'] = overlap_issue
    roles = [role for method in ('gc_profile', 'stereo_profile', 'aromatic_profile')
             for role in METHODS[method]['roles'].values()]
    roles.extend(role for definition in REPORTED_BIOMARKER_METHODS.values()
                 for key, role in definition['roles'].items() if key != 'ster_hop')
    isotope_roles = [METHODS['sofer']['roles'][key] for key in ('sat', 'arom')]
    sat_gcms_fraction_prefixes = (
        tuple(f'H{carbon} 22S/R[' for carbon in range(32, 36))
        + tuple(f'{carbon}HomoHop[' for carbon in range(31, 36))
        + ('H29ab/H29ba[', 'H30ab/H30ba[')
    )
    for evidence in match['evidence']:
        field = evidence['field']
        group, label = source_identity(Column(**field, kind='number'))
        isotope = any(field_matches(group, label, role) for role in isotope_roles)
        family = next((role['unit_family'] for role in roles if field_matches(group, label, role)), None)
        if group == 'Sat-GC' and label.startswith(('iC18/nC18[', 'Pr/nC19[')):
            family = 'ratio'
        if group == 'Sat-GCMS':
            if label.startswith(sat_gcms_fraction_prefixes):
                family = 'fraction'
            elif label.startswith((
                't19/t23[', 't23/tet24[', 'H27Ts/H27Tm[',
                't20/t21[', 't22/t21[',
                'H35ab/H31ab[', 'H35ab/H34ab[', 'H35abS/H34abS[',
                'H31abR/H30ab[', 'Gamm/31abR[',
                'H30ab/H29ab[',
            )):
                family = 'ratio'
        factor = UNITS.get(family, {}).get(field['unit'])
        supported = group != 'WO-GC' and variant_tag(label) in ('a', 'h', 'ca', 'ch') and factor is not None
        if isotope:
            supported = field['unit'] in UNITS['delta']
        for side in ('a', 'b'):
            point = evidence[side]
            if point['status'] != 'numeric':
                continue
            if not supported:
                point['status'] = 'unsupported_correlation_indicator'
            elif isotope and not confirm_c15_vpdb:
                point['status'] = 'unconfirmed_isotope_basis'
            elif not isotope and (point['value'] < 0 or (family == 'fraction' and point['value'] * factor > 1)):
                point['status'] = 'invalid_range'
        if any(evidence[side]['status'] != 'numeric' for side in ('a', 'b')):
            evidence['difference'] = None
    ster_hop = list(_ster_hop_evidence(inputs, sample_ids))
    if ster_hop:
        consumed = {field_id for item in ster_hop for field_id in item['input_field_ids']}
        match['evidence'] = [item for item in match['evidence'] if item['field']['id'] not in consumed] + ster_hop
    match['correlation'] = {'version': 1, 'materials': kinds, 'expert_review_required': True}
    match['review_key'] = 'oil-correlation:' + match['input_digest']
    if confirm_c15_vpdb:
        match['correlation']['isotope_basis'] = {'fraction': 'C15+', 'reference': 'VPDB'}
        match['review_key'] += ':c15-vpdb'
    return match


def correlation_sensitivity(report, baseline, *, exclude_point=None):
    """Recalculate after omitting groups or hypothetically excluding one measurement."""
    inputs = report['inputs']
    if exclude_point is None:
        groups = {}
        for field in inputs['fields']:
            group, _ = source_identity(Column(**field, kind='number'))
            if group:
                groups.setdefault(group, set()).add(field['id'])
        cases = [(group, {**inputs, 'fields': [field for field in inputs['fields'] if field['id'] not in omitted]})
                 for group, omitted in sorted(groups.items())]
    else:
        field_id, sample_id = exclude_point
        points = inputs['points']
        values = points[field_id]
        point = values[sample_id]
        if point['status'] != 'numeric':
            raise ValueError('Для пробного исключения выберите допущенное числовое измерение.')
        excluded = {**point, 'status': 'qc_excluded', 'value': None}
        cases = [(None, {**inputs, 'points': {**points, field_id: {**values, sample_id: excluded}}})]
    confirmed = bool(baseline['correlation'].get('isotope_basis'))
    comparable = {item['field']['id']: item['field']['name'] for item in baseline['evidence']
                  if item['difference'] is not None}
    scenarios = []
    for group, scenario_inputs in cases:
        match = correlate_oil_profiles({**report, 'inputs': scenario_inputs},
                                       confirm_c15_vpdb=confirmed)
        remaining = {item['field']['id'] for item in match['evidence'] if item['difference'] is not None}
        scenarios.append({'group': group, 'comparable': len(remaining),
                          'lost': [name for key, name in comparable.items() if key not in remaining],
                          'overlap_percent': match['overlap_percent']})
    return scenarios


def _comparison_matrix(report, preprocessing):
    """Use the same complete cases and scaling for PCA and clustering."""
    import numpy as np

    fields = report['inputs']['fields']
    if preprocessing not in ('standardized', 'clr'):
        raise ValueError('Неизвестное преобразование показателей')
    clr = preprocessing == 'clr'
    if clr and (len(fields) < 2 or len({field['unit'] for field in fields}) != 1):
        raise ValueError('Для CLR нужны не менее двух частей состава в одинаковых единицах')
    if clr:
        compositions = []
        source_bases = set()
        for field in fields:
            column = Column(**field, kind='number')
            group, label = source_identity(column)
            peak = label.partition('[')[0]
            composition = None
            if group == 'Sat-GCMS' and peak in {f'{carbon}HomoHop' for carbon in range(31, 36)}:
                composition = 'homohopanes'
            elif group == 'Arom-GCMS' and peak in ('27MAS', '28MAS', '29MAS'):
                composition = 'MAS'
            elif group == 'Arom-GCMS' and peak in ('26TAS', '27TAS', '28TAS'):
                composition = 'TAS'
            elif (field_family(column) in ('ratio', 'percent_ratio', 'fraction')
                  or group == 'Sat-GCMS' and peak in {f'H{carbon} 22S/R' for carbon in range(31, 36)}):
                raise ValueError('Отдельные отношения биомаркеров не являются частями одного состава для CLR')
            compositions.append(composition)
            if composition and (tag := variant_tag(label)):
                source_bases.add(tag)
        if any(compositions) and len(set(compositions)) != 1:
            raise ValueError('Для CLR выберите части одного известного распределения')
        if len(source_bases) > 1 or 'un' in source_bases:
            raise ValueError('Для CLR нужна известная единая аналитическая основа')
    points = report['inputs']['points']
    included, excluded, rows = [], {}, []
    bases = set()
    for group in ('group_a', 'group_b'):
        for sample in report['request'][group]:
            values = [points[field['id']][sample] for field in fields]
            reasons = {field['id']: value['status'] for field, value in zip(fields, values)
                       if value['status'] != 'numeric'}
            if clr:
                reasons.update({field['id']: 'nonpositive_composition' for field, value in zip(fields, values)
                                if value['status'] == 'numeric' and value['value'] <= 0})
            if reasons:
                excluded[sample] = reasons
            else:
                included.append({'sample_id': sample, 'group': group})
                rows.append([value['value'] for value in values])
                if clr:
                    bases.update(basis for field, value in zip(fields, values)
                                 if (basis := value['measurement'].get('basis') or field['metadata'].get('basis')))
    if len(rows) < 3:
        raise ValueError('Нужны не менее трёх проб с полным набором показателей')

    matrix = np.asarray(rows, dtype=float)
    metadata = {'input_digest': report['input_digest'], 'excluded': excluded, 'preprocessing': preprocessing}
    if clr:
        if len(bases) > 1:
            raise ValueError('Для CLR нужна единая основа измерения всех частей состава')
        logs = np.log(matrix)
        transformed = logs - logs.mean(axis=1, keepdims=True)
        if np.allclose(transformed, transformed[0], rtol=0, atol=1e-12):
            raise ValueError('После CLR в выборке нет различий между составами')
        return transformed, included, {
            **metadata, 'fields': [field['id'] for field in fields], 'constant_fields': [],
            'composition': {'transform': 'clr', 'log_base': 'natural', 'nonpositive_policy': 'exclude_sample'},
        }
    active = np.any(matrix != matrix[0], axis=0)
    matrix = matrix[:, active]
    # Work in scaled units first to avoid overflow when centering large values.
    scale = np.max(np.abs(matrix), axis=0)
    matrix /= scale
    centers = matrix.mean(axis=0)
    deviations = matrix.std(axis=0, ddof=1)
    standardized = (matrix - centers) / deviations
    return standardized, included, {
        **metadata,
        'constant_fields': [field['id'] for field, keep in zip(fields, active) if not keep],
        'fields': [field['id'] for field, keep in zip(fields, active) if keep],
        'standardization': {'scale': scale.tolist(), 'scaled_means': centers.tolist(),
                            'scaled_sample_sd': deviations.tolist()},
    }


def comparison_pca(report, *, preprocessing='standardized'):
    """Fit a descriptive, standardized PCA on complete comparison observations."""
    import numpy as np

    if len(report['inputs']['fields']) < 2:
        raise ValueError('Для PCA выберите не менее двух показателей')
    matrix, included, metadata = _comparison_matrix(report, preprocessing)
    if matrix.shape[1] < (3 if preprocessing == 'clr' else 2):
        raise ValueError('Для PCA нужны два независимых направления вариации')
    if preprocessing == 'clr':
        center = matrix.mean(axis=0)
        matrix -= center
        metadata['composition']['pca_center'] = center.tolist()
    try:
        _, singular, vectors = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Не удалось разложить матрицу PCA') from exc
    threshold = np.finfo(float).eps * max(matrix.shape) * singular[0]
    if preprocessing == 'clr':
        threshold = max(threshold, 1e-12)
    if singular[1] <= threshold:
        raise ValueError('Для PCA нужны два независимых направления вариации')
    components = vectors[:2]
    # Orient each axis by its largest coefficient for reproducible signs.
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    scores = matrix @ components.T
    variance = singular ** 2
    return {
        'format': 'autopyro.comparison_pca', 'version': 1, **metadata,
        'explained_variance_ratio': (variance[:2] / variance.sum()).tolist(),
        'coefficients': components.tolist(),
        'scores': [{**item, 'coordinates': score.tolist()} for item, score in zip(included, scores)],
    }


def comparison_clustering(report, cluster_count, *, preprocessing='standardized'):
    """Build a Ward hierarchy in the standardized measurement space."""
    from scipy.cluster.hierarchy import cut_tree, linkage

    standardized, included, metadata = _comparison_matrix(report, preprocessing)
    if len(included) > 2000:
        raise ValueError('Для кластеризации сократите выборку до 2000 полных проб')
    if not 2 <= cluster_count < len(included):
        raise ValueError('Число кластеров должно быть не меньше двух и меньше числа полных проб')
    if not standardized.shape[1]:
        raise ValueError('Для кластеризации нужен хотя бы один непостоянный показатель')
    tree = linkage(standardized, method='ward', metric='euclidean')
    labels = cut_tree(tree, n_clusters=[cluster_count]).ravel()
    return {
        'format': 'autopyro.comparison_clustering', 'version': 1, **metadata,
        'method': 'ward', 'metric': 'euclidean', 'cluster_count': cluster_count,
        'linkage': tree.tolist(),
        'membership': [{**sample, 'cluster': int(label) + 1} for sample, label in zip(included, labels)],
    }


def comparison_kmeans(report, cluster_count, *, seed=42, preprocessing='standardized'):
    """Run seeded K-means++ starts and retain the smallest within-cluster sum of squares."""
    import numpy as np
    from scipy.cluster.vq import ClusterError, kmeans2, vq

    matrix, included, metadata = _comparison_matrix(report, preprocessing)
    if len(included) > 2000:
        raise ValueError('Для кластеризации сократите выборку до 2000 полных проб')
    if not 2 <= cluster_count <= 20 or cluster_count >= len(included):
        raise ValueError('Для K-средних задайте от 2 до 20 кластеров и меньше числа полных проб')
    if not matrix.shape[1]:
        raise ValueError('Для кластеризации нужен хотя бы один непостоянный показатель')
    if len(np.unique(matrix, axis=0)) < cluster_count:
        raise ValueError('Число кластеров превышает число различных профилей')

    starts, iterations = 10, 100
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        try:
            centers, _ = kmeans2(matrix, cluster_count, iter=iterations, minit='++', missing='raise', seed=rng)
        except ClusterError:
            continue
        labels, _ = vq(matrix, centers)
        order = list(dict.fromkeys(labels))
        if len(order) != cluster_count:
            continue
        # Keep the final assignment and its means consistent even after the last iteration.
        centers = np.stack([matrix[labels == label].mean(axis=0) for label in order])
        remap = {label: index for index, label in enumerate(order)}
        membership = [remap[label] for label in labels]
        residuals = matrix - centers[membership]
        squared = np.sum(residuals ** 2, axis=1)
        _, nearest = vq(matrix, centers)
        if not np.allclose(squared, nearest ** 2, rtol=1e-12, atol=1e-14):
            continue
        inertia = float(squared.sum())
        if best is None or inertia < best[0]:
            best = inertia, centers, membership
    if best is None:
        raise ValueError('K-средних не сошёлся для выбранного числа кластеров')
    inertia, centers, membership = best
    return {
        'format': 'autopyro.comparison_kmeans', 'version': 1, **metadata,
        'method': 'kmeans', 'metric': 'euclidean', 'cluster_count': cluster_count,
        'seed': seed, 'initialization': 'kmeans++', 'starts': starts, 'max_iterations': iterations,
        'inertia': inertia, 'centers': centers.tolist(),
        'membership': [{**sample, 'cluster': label + 1} for sample, label in zip(included, membership)],
    }


def comparison_dbscan(report, radius, min_samples, *, preprocessing='standardized'):
    """Find density-connected cores and attach border samples in comparison order."""
    from scipy.spatial import cKDTree

    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Радиус DBSCAN должен быть положительным и конечным')
    matrix, included, metadata = _comparison_matrix(report, preprocessing)
    if len(included) > 2000:
        raise ValueError('Для кластеризации сократите выборку до 2000 полных проб')
    if type(min_samples) is not int or not 2 <= min_samples <= len(included):
        raise ValueError('Минимум соседей DBSCAN должен быть от двух до числа полных проб')
    if not matrix.shape[1]:
        raise ValueError('Для кластеризации нужен хотя бы один непостоянный показатель')

    tree = cKDTree(matrix)
    core = tree.query_ball_point(matrix, radius, return_length=True) >= min_samples
    labels = [0] * len(included)
    count = 0
    for index, is_core in enumerate(core):
        if not is_core or labels[index]:
            continue
        count += 1
        labels[index] = count
        pending = [index]
        while pending:
            point = pending.pop()
            for neighbor in tree.query_ball_point(matrix[point], radius):
                if labels[neighbor]:
                    continue
                labels[neighbor] = count
                if core[neighbor]:
                    pending.append(neighbor)
    return {
        'format': 'autopyro.comparison_dbscan', 'version': 1, **metadata,
        'method': 'dbscan', 'metric': 'euclidean', 'radius': radius, 'min_samples': min_samples,
        'cluster_count': count, 'noise_count': labels.count(0),
        'membership': [{**sample, 'cluster': label or None,
                        'role': 'core' if is_core else 'border' if label else 'noise'}
                       for sample, label, is_core in zip(included, labels, core)],
    }


MODEL_RUNS_KEY = 'geochemistry_model_runs'


def comparison_oil_mixing(report, tolerances):
    """Fit the mass fraction of the second end member against measured concentrations."""
    request, inputs = report['request'], report['inputs']
    if len(request['group_a']) != 2 or len(request['group_b']) != 1:
        raise ValueError('Выберите два конечных компонента в группе A и одну смесь в группе B')
    sample_ids = [*request['group_a'], *request['group_b']]
    if len(set(sample_ids)) != 3 or any(
        inputs['samples'][sample]['material']['sample_type'].strip().casefold() not in ('oil', 'нефть')
        for sample in sample_ids
    ):
        raise ValueError('Для оценки смешения нужны три разные пробы нефти')
    fields = inputs['fields']
    if set(tolerances) != {field['id'] for field in fields}:
        raise ValueError('Укажите допуск для каждого выбранного маркера')
    rows, fit, analytes = [], [], set()
    for field in fields:
        cid, unit, metadata = field['id'], field['unit'], field['metadata']
        tolerance = tolerances[cid]
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('Допуски должны быть положительными конечными числами')
        points = [inputs['points'][cid][sample] for sample in sample_ids]
        bases = {point['measurement']['basis'] for point in points
                 if point.get('measurement', {}).get('basis')}
        whole_oil_unit = unit in ('mg/kg oil', 'µg/g oil', 'mg/g oil')
        mass_unit = unit in ('mg/kg', 'µg/g', 'mg/g', 'ppm (µg/g)')
        whole_oil_basis = metadata.get('basis') == 'whole_oil' or all(
            point.get('measurement', {}).get('basis') == 'whole_oil'
            for point in points if point['status'] == 'numeric'
        )
        if (bases - {'whole_oil'} or metadata.get('basis') not in (None, '', 'whole_oil')
                or not (whole_oil_unit or (mass_unit and whole_oil_basis))):
            raise ValueError('Для смешения нужны концентрации на массу исходной нефти')
        _, label = source_identity(Column(**field, kind='number'))
        analyte = str(metadata.get('analyte') or label.split('[', 1)[0]).strip().casefold()
        if not analyte or analyte in analytes:
            raise ValueError('Выберите по одному показателю на маркер')
        analytes.add(analyte)
        statuses = [point['status'] if point['status'] != 'numeric' or point['value'] >= 0 else 'invalid_range'
                    for point in points]
        values = [point['value'] for point in points]
        row = {'field': deepcopy(field), 'values': values, 'statuses': statuses,
               'used_for_fit': False, 'predicted': None, 'residual': None, 'within_tolerance': None}
        rows.append(row)
        if statuses == ['numeric'] * 3:
            a, b, measured = values
            if abs(b - a) > tolerance:
                fit.append(((b - a) / tolerance, (measured - a) / tolerance))
                row['used_for_fit'] = True

    result = {'format': 'autopyro.comparison_oil_mixing', 'version': 1, 'method': 'oil_mixing',
              'input_digest': report['input_digest'], 'sample_ids': sample_ids,
              'tolerances': dict(tolerances), 'markers': rows, 'status': 'underdetermined',
              'fraction_b': None, 'unbounded_fraction_b': None, 'outside_interval': None}
    if len(fit) < 2:
        return result
    scale = max(abs(delta) for delta, _ in fit)
    scaled = [(delta / scale, target / scale) for delta, target in fit]
    if any(not math.isfinite(target) or not math.isfinite(delta) for delta, target in scaled):
        raise ValueError('Численный диапазон концентраций и допусков слишком велик')
    try:
        free = math.fsum(delta * target for delta, target in scaled) / math.fsum(delta ** 2 for delta, _ in scaled)
    except OverflowError as exc:
        raise ValueError('Численный диапазон концентраций и допусков слишком велик') from exc
    fraction = min(1.0, max(0.0, free))
    for row in rows:
        if row['statuses'] != ['numeric'] * 3:
            continue
        a, b, measured = row['values']
        predicted = (1 - fraction) * a + fraction * b
        residual = measured - predicted
        row.update(predicted=predicted, residual=residual,
                   within_tolerance=abs(residual) <= tolerances[row['field']['id']])
    status = ('inconsistent' if any(row['within_tolerance'] is False for row in rows)
              else 'endpoint' if fraction in (0, 1) else 'consistent')
    result.update(status=status, fraction_b=fraction, unbounded_fraction_b=free,
                  outside_interval=not 0 <= free <= 1)
    return result


def comparison_prediction(report, target_id, *, well_cv=False, version=3):
    run = _well_holdout_prediction(report, target_id, version=version)
    if version >= 3:
        run['validation_by_well'] = _validation_by_well(run)
    if well_cv:
        run['well_cv'] = _leave_one_well_out(report, run)
    return run


def _well_holdout_prediction(report, target_id, *, min_test=2, version=2):
    """Fit OLS on group A and evaluate group B on explicitly disjoint wells."""
    import numpy as np
    inputs, request = report['inputs'], report['request']
    fields = [field['id'] for field in inputs['fields']]
    if target_id not in fields or len(fields) < 2:
        raise ValueError('Для прогноза выберите цель и хотя бы один предиктор')
    predictors = [field for field in fields if field != target_id]
    wells = {}
    for group in ('group_a', 'group_b'):
        wells[group] = {}
        for sample in request[group]:
            names = {' '.join(name.casefold().split()) for name in inputs['samples'][sample]['well']}
            if len(names) != 1 or '' in names:
                raise ValueError('Для каждой пробы укажите одну однозначную скважину')
            wells[group][sample] = names.pop()
    if set(wells['group_a'].values()) & set(wells['group_b'].values()):
        raise ValueError('Скважины обучения A и проверки B не должны пересекаться')

    datasets, excluded = {}, {}
    for group in wells:
        ids, x, y = [], [], []
        for sample in request[group]:
            points = {field: inputs['points'][field][sample] for field in fields}
            reasons = {field: point['status'] for field, point in points.items() if point['status'] != 'numeric'}
            if reasons:
                excluded[sample] = reasons
                continue
            ids.append(sample)
            x.append([points[field]['value'] for field in predictors])
            y.append(points[target_id]['value'])
        datasets[group] = ids, np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    train_ids, train_x, train_y = datasets['group_a']
    test_ids, test_x, test_y = datasets['group_b']
    if len(train_ids) <= len(predictors) + 1 or len(test_ids) < min_test:
        raise ValueError('Для прогноза нужно больше обучающих проб, чем коэффициентов, и хотя бы две проверочные пробы')
    with np.errstate(over='ignore', invalid='ignore'):
        centers, scales = train_x.mean(axis=0), train_x.std(axis=0, ddof=1)
        if not np.isfinite(centers).all() or not np.isfinite(scales).all():
            raise ValueError('Численное переполнение; измените масштаб данных или начальные параметры')
        if np.any(scales <= 0):
            raise ValueError('Один из предикторов постоянен в обучающей выборке')
        design = np.column_stack((np.ones(len(train_ids)), (train_x - centers) / scales))
        validation = np.column_stack((np.ones(len(test_ids)), (test_x - centers) / scales))
        if not np.isfinite(design).all() or not np.isfinite(validation).all():
            raise ValueError('Численное переполнение; измените масштаб данных или начальные параметры')
        try:
            coefficients, _, rank, _ = np.linalg.lstsq(design, train_y, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Не удалось решить задачу линейного прогноза') from exc
        if rank != design.shape[1]:
            raise ValueError('Предикторы линейно зависимы в обучающей выборке')
        predictions = validation @ coefficients
        if not np.isfinite(predictions).all() or not np.isfinite(coefficients).all():
            raise ValueError('Численное переполнение; измените масштаб данных или начальные параметры')
    metrics = _finite_prediction_metrics(test_y, predictions)
    outside = (test_x < train_x.min(axis=0)) | (test_x > train_x.max(axis=0))
    result = {
        'format': 'autopyro.comparison_prediction', 'version': version, 'input_digest': report['input_digest'],
        'method': 'well_holdout_ols', 'preprocessing': 'train_standardized', 'target': target_id,
        'predictors': predictors, 'train_ids': train_ids, 'wells': wells, 'excluded': excluded,
        'train_means': centers.tolist(), 'train_sample_sd': scales.tolist(),
        'coefficients': coefficients.tolist(), 'metrics': metrics,
        'predictions': [{'sample_id': sample, 'observed': float(y), 'predicted': float(predicted),
                         'outside_train_range': [field for field, flag in zip(predictors, flags) if flag]}
                        for sample, y, predicted, flags in zip(test_ids, test_y, predictions, outside)],
    }
    if version >= 2:
        baseline = float(train_y.mean())
        result['baseline'] = {'method': 'training_mean', 'value': baseline,
                              'metrics': _finite_prediction_metrics(test_y, np.full_like(test_y, baseline))}
    return result


def _finite_prediction_metrics(observed, estimated):
    from autopyro.curves.curve_validation import prediction_metrics

    return {key: value if math.isfinite(value) else None
            for key, value in prediction_metrics(observed, estimated).items()}


def _validation_by_well(run):
    import numpy as np

    groups = defaultdict(list)
    for prediction in run['predictions']:
        groups[run['wells']['group_b'][prediction['sample_id']]].append(prediction)
    wells = []
    for well, predictions in groups.items():
        observed = np.asarray([item['observed'] for item in predictions])
        estimated = np.asarray([item['predicted'] for item in predictions])
        baseline = np.full_like(observed, run['baseline']['value'])
        wells.append({'well': well, 'sample_ids': [item['sample_id'] for item in predictions],
                      'metrics': _finite_prediction_metrics(observed, estimated),
                      'baseline_metrics': _finite_prediction_metrics(observed, baseline)})

    def equal_weight(key):
        values = [item[key] for item in wells]
        return {'RMSE': math.hypot(*(item['RMSE'] for item in values)) / math.sqrt(len(values)),
                'MAE': fmean(item['MAE'] for item in values)}

    return {'wells': wells, 'equal_weight_metrics': equal_weight('metrics'),
            'equal_weight_baseline_metrics': equal_weight('baseline_metrics')}


def _leave_one_well_out(report, run):
    import numpy as np

    by_well = defaultdict(list)
    for sample in run['train_ids']:
        by_well[run['wells']['group_a'][sample]].append(sample)
    if not 2 <= len(by_well) <= 50:
        raise ValueError('Для перекрёстной проверки нужны от 2 до 50 скважин в группе A')
    folds, observations, baseline_values = [], [], []
    for well, held_out in by_well.items():
        training = [sample for sample in run['train_ids'] if run['wells']['group_a'][sample] != well]
        if len(training) <= len(run['predictors']) + 1:
            raise ValueError('После исключения одной скважины в A недостаточно обучающих проб')
        fold_report = {**report, 'request': {**report['request'], 'group_a': training, 'group_b': held_out}}
        fold = _well_holdout_prediction(fold_report, run['target'], min_test=1, version=run['version'])
        folds.append({'well': well, 'train_ids': fold['train_ids'], 'train_means': fold['train_means'],
                      'train_sample_sd': fold['train_sample_sd'], 'metrics': fold['metrics'],
                      'predictions': fold['predictions']})
        observations.extend(fold['predictions'])
        if 'baseline' in fold:
            folds[-1]['baseline'] = fold['baseline']
            baseline_values.extend([fold['baseline']['value']] * len(held_out))
    observed = np.asarray([item['observed'] for item in observations])
    metrics = _finite_prediction_metrics(observed, np.asarray([item['predicted'] for item in observations]))
    result = {'method': 'leave_one_well_out', 'folds': folds, 'metrics': metrics}
    if baseline_values:
        result['baseline_metrics'] = _finite_prediction_metrics(observed, np.asarray(baseline_values))
    return result


def freeze_comparison(report):
    """Freeze the selected inputs without copying unused measurement candidates."""
    inputs = report['inputs']
    points = {}
    for field, values in inputs['points'].items():
        points[field] = {}
        for sample, value in values.items():
            point = {'status': value['status'], 'value': value['value']}
            if value.get('measurement'):
                measurement = value['measurement']
                point['measurement'] = {key: deepcopy(measurement[key]) for key in (
                    'id', 'revision', 'unit', 'basis', 'row_id', 'analysis_id', 'source_id', 'repeat_number',
                ) if key in measurement}
            points[field][sample] = point
    return {'version': report['version'], 'input_digest': report['input_digest'], 'request': deepcopy(report['request']),
            'inputs': {'fields': deepcopy(inputs['fields']), 'points': points, 'samples': deepcopy(inputs['samples'])},
            **({'qc_exclusions': deepcopy(report['qc_exclusions'])} if report.get('qc_exclusions') else {})}


def apply_prediction(project, record, snapshot, sample_ids):
    validate_model_run(record, snapshot)
    run = record['result']
    if run['format'] != 'autopyro.comparison_prediction':
        raise ValueError('Выберите сохранённую модель линейного прогноза')
    index = GeochemistryIndex(project)
    if not sample_ids or len(set(sample_ids)) != len(sample_ids) or set(sample_ids) - index.samples.keys():
        raise ValueError('Выберите пробы для применения прогноза')
    inputs = snapshot['inputs']
    fields = {field['id']: field for field in inputs['fields']}
    predictors = run['predictors']
    bases, ranges = {}, {}
    for field in predictors:
        column = index.columns.get(field)
        saved = fields[field]
        if (column is None or derived(column) or column.kind != 'number' or column.unit != saved['unit']
                or column.metadata.get('basis') != saved['metadata'].get('basis')
                or column.metadata.get('geochem_field') != saved['metadata'].get('geochem_field')):
            raise ValueError('Определение предиктора изменилось или колонка удалена')
        training = [inputs['points'][field][sample] for sample in run['train_ids']]
        bases[field] = {point['measurement'].get('basis') or '' for point in training}
        values = [point['value'] for point in training]
        ranges[field] = min(values), max(values)
    training_materials = [inputs['samples'][sample]['material'] for sample in run['train_ids']]
    rows = []
    for sample in sample_ids:
        descriptor = index.samples[sample]
        points = {field: index.value(sample, field) for field in predictors}
        reasons = {field: point['status'] for field, point in points.items() if point['status'] != 'numeric'}
        for field, point in points.items():
            if field not in reasons and (point['measurement'].get('basis') or '') not in bases[field]:
                reasons[field] = 'basis_mismatch'
        compatible = any(_compatible([descriptor['material'], material]) for material in training_materials)
        prediction, outside = None, []
        if not reasons and compatible:
            values = [points[field]['value'] for field in predictors]
            outside = [field for field, value in zip(predictors, values) if not ranges[field][0] <= value <= ranges[field][1]]
            prediction = run['coefficients'][0] + sum(
                weight * ((value - mean) / scale) for value, mean, scale, weight in
                zip(values, run['train_means'], run['train_sample_sd'], run['coefficients'][1:]))
        status = ('material_mismatch' if not compatible else 'excluded' if reasons else
                  'numeric' if math.isfinite(prediction) else 'non_finite')
        rows.append({'sample_id': sample, 'name': descriptor['name'], 'status': status,
                     'predicted': prediction if status == 'numeric' else None, 'excluded': reasons,
                     'outside_train_range': outside,
                     'inputs': {field: {'value': point['value'], 'status': point['status'],
                                        'measurement_id': point.get('measurement', {}).get('id'),
                                        'revision': point.get('measurement', {}).get('revision')}
                                for field, point in points.items()}})
    return {'format': 'autopyro.prediction_application', 'version': 1, 'model_run_id': record['id'],
            'snapshot_id': record['snapshot_id'], 'target': deepcopy(fields[run['target']]),
            'predictors': [deepcopy(fields[field]) for field in predictors], 'results': rows}


def archive_model_run(project, report, result):
    """Share a frozen comparison across runs without copying unused measurement candidates."""
    import numpy
    import scipy
    from autopyro import __version__

    snapshot = freeze_comparison(report)
    snapshot_id = _hash(snapshot)
    archive = project.view.setdefault(MODEL_RUNS_KEY, {'snapshots': {}, 'runs': []})
    archive['snapshots'].setdefault(snapshot_id, snapshot)
    record = {
        'format': 'autopyro.geochemistry_model_run', 'version': 1, 'id': new_id(),
        'created_at': timestamp(), 'snapshot_id': snapshot_id, 'result': deepcopy(result),
        'engine': {'autopyro': __version__, 'numpy': numpy.__version__, 'scipy': scipy.__version__},
    }
    archive['runs'].append(record)
    return record


def validate_model_run(record, snapshot):
    if (record.get('format') != 'autopyro.geochemistry_model_run' or record.get('version') != 1
            or record.get('snapshot_id') != _hash(snapshot)):
        raise ValueError('Версия запуска не поддерживается или снимок входов повреждён')
    saved = record['result']
    supported = (1, 2, 3) if saved.get('format') == 'autopyro.comparison_prediction' else (1,)
    if saved.get('version') not in supported or saved.get('input_digest') != snapshot.get('input_digest'):
        raise ValueError('Версия запуска не поддерживается или снимок входов повреждён')


def replay_model_run(record, snapshot):
    """Recompute from the archived inputs and report whether the result agrees."""
    validate_model_run(record, snapshot)
    saved = record['result']
    if saved['format'] == 'autopyro.comparison_oil_mixing':
        return _model_results_match(saved, comparison_oil_mixing(snapshot, saved['tolerances']))
    options = {'preprocessing': saved['preprocessing']}
    if saved['format'] == 'autopyro.comparison_pca':
        replayed = comparison_pca(snapshot, **options)
    elif saved['format'] == 'autopyro.comparison_clustering':
        replayed = comparison_clustering(snapshot, saved['cluster_count'], **options)
    elif saved['format'] == 'autopyro.comparison_kmeans':
        replayed = comparison_kmeans(snapshot, saved['cluster_count'], seed=saved['seed'], **options)
    elif saved['format'] == 'autopyro.comparison_dbscan':
        replayed = comparison_dbscan(snapshot, saved['radius'], saved['min_samples'], **options)
    elif saved['format'] == 'autopyro.comparison_prediction':
        replayed = comparison_prediction(snapshot, saved['target'], well_cv='well_cv' in saved, version=saved['version'])
    else:
        raise ValueError('Версия запуска не поддерживается или снимок входов повреждён')
    return _model_results_match(saved, replayed)


def _model_results_match(saved, replayed):
    if isinstance(saved, dict) and isinstance(replayed, dict):
        return saved.keys() == replayed.keys() and all(_model_results_match(saved[key], replayed[key]) for key in saved)
    if isinstance(saved, list) and isinstance(replayed, list):
        return len(saved) == len(replayed) and all(_model_results_match(a, b) for a, b in zip(saved, replayed))
    if isinstance(saved, float) and isinstance(replayed, float):
        return math.isclose(saved, replayed, rel_tol=1e-9, abs_tol=1e-12)
    return saved == replayed


def verify_model_run_package(payload):
    """Validate and replay a portable archive before it can modify a project."""
    try:
        record, snapshot = payload['run'], payload['snapshot']
        if (not isinstance(record['id'], str) or not record['id']
                or not isinstance(record['created_at'], str) or not record['created_at']
                or not isinstance(record['engine'], dict)
                or not all(isinstance(version, str) for version in record['engine'].values())):
            raise TypeError('Invalid run metadata')
        matches = replay_model_run(record, snapshot)
    except (AttributeError, KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError('Файл не содержит корректного запуска модели и снимка входов') from exc
    if not matches:
        raise ValueError('Повторный расчёт отличается от сохранённого результата.')
    return record, snapshot


def install_model_run(project, record, snapshot):
    """Add a verified run and share any identical snapshot already in the archive."""
    validate_model_run(record, snapshot)
    archive = project.view.get(MODEL_RUNS_KEY, {'snapshots': {}, 'runs': []})
    existing = next((item for item in archive['runs'] if item['id'] == record['id']), None)
    if existing is not None:
        if existing != record:
            raise ValueError('В архиве уже есть другой запуск с этим идентификатором')
        return False
    archive['snapshots'].setdefault(record['snapshot_id'], deepcopy(snapshot))
    archive['runs'].append(deepcopy(record))
    project.view[MODEL_RUNS_KEY] = archive
    return True
