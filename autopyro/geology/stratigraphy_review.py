"""Quality checks and a previewable merge for interval drafts."""

from collections import defaultdict
from dataclasses import dataclass, replace

from autopyro.model.core import new_id
from .stratigraphy import STRATAPY_FIELDS


def track(item):
    return item['wellbore'], item['level']


def bounds(item):
    return item['top'], item['bottom']


def content(item):
    return (*track(item), *bounds(item), item['name'], str(item.get('color') or ''),
            bool(item.get('color_manual')),
            *(str(item.get(key, '')) for key in ('lithology', *STRATAPY_FIELDS)))


@dataclass(frozen=True)
class Finding:
    kind: str
    well: str
    level: str
    top: float
    bottom: float
    ids: tuple[str, ...]


def diagnose(intervals):
    findings = []
    tracks = defaultdict(list)
    seen = {}
    for item in intervals:
        well, level = track(item)
        top, bottom = bounds(item)
        key = content(item)
        if key in seen:
            findings.append(Finding('duplicate', well, level, top, bottom, (seen[key], item['id'])))
        else:
            seen[key] = item['id']
            tracks[well, level].append(item)
        if not item.get('lithology', '').strip() or item['lithology'].strip().casefold() == 'no':
            findings.append(Finding('missing', well, level, top, bottom, (item['id'],)))
    for (well, level), items in tracks.items():
        covering = None
        for item in sorted(items, key=bounds):
            top, bottom = bounds(item)
            if covering is not None:
                edge = covering['bottom']
                ids = (covering['id'], item['id'])
                if top > edge:
                    findings.append(Finding('gap', well, level, edge, top, ids))
                elif top < edge:
                    findings.append(Finding('overlap', well, level, top, min(bottom, edge), ids))
            if covering is None or bottom > covering['bottom']:
                covering = item
    return findings


@dataclass(frozen=True)
class MergeEntry:
    kind: str
    item: dict
    target: str | None = None
    previous: dict | None = None
    reason: str = ''
    conflict_with: dict | None = None


def database_source(source):
    return isinstance(source, dict) and source.get('type') == 'postgresql_lithostratigraphy'


def database_interval_key(item, scheme_source):
    source = item.get('import_source', scheme_source)
    if not database_source(source):
        return None
    endpoint = source.get('endpoint')
    if not isinstance(endpoint, dict) or not isinstance(endpoint.get('port'), int):
        return None
    if not all(isinstance(endpoint.get(key), str) and endpoint[key] for key in ('host', 'database')):
        return None
    raw = item.get('raw_cells')
    if not isinstance(raw, dict):
        return None
    wellbore = raw.get('wellbore_id')
    source_id = raw.get('source_interval_id')
    row_id = raw.get('interpretation_interval_id')
    if not isinstance(wellbore, int) or not isinstance(source_id or row_id, int):
        return None
    # Revision rows get new IDs when cloned. Rows with no source ID can
    # only be matched within the same revision.
    return (endpoint['host'], endpoint['port'], endpoint['database'], wellbore,
            ('source', source_id) if source_id else ('row', row_id))


def preserve_local_geology(item, previous, *, by_id=False):
    return {**item, **{key: previous[key] for key in (
        'color', 'lithology', 'lithology_source', 'lithology_source_field', *STRATAPY_FIELDS
    ) if key in previous and (by_id or key == 'color' or not item.get(key))}}


def plan_merge(current, incoming, *, update_database=True):
    """Classify same-track intervals without enumerating all overlapping pairs."""
    if any(current[key] != incoming[key] for key in ('depth_unit', 'depth_reference')):
        raise ValueError('Для объединения должны совпадать единицы и система глубин.')
    existing = defaultdict(list)
    tracks = defaultdict(list)
    seen = set()
    ids = set()
    source = incoming.get('source')
    db_import = database_source(source)
    current_source = current.get('source')
    database_matches = defaultdict(list)
    incoming_keys = {}
    key_counts = defaultdict(int)
    if db_import:
        for index, item in enumerate(incoming['intervals']):
            key = database_interval_key(item, source)
            if key is not None:
                incoming_keys[index] = key
                key_counts[key] += 1
    for item in current['intervals']:
        existing[(*track(item), *bounds(item))].append(item)
        tracks[track(item)].append((*bounds(item), None, item))
        seen.add(content(item))
        ids.add(item['id'])
        if db_import and (key := database_interval_key(item, current_source)) in key_counts:
            database_matches[key].append(item)
    entries = []
    candidates = defaultdict(list)
    replacements = {}
    for index, item in enumerate(incoming['intervals']):
        geometry = (*track(item), *bounds(item))
        db_key = incoming_keys.get(index)
        db_matches = database_matches.get(db_key, ())
        if db_key is not None and (key_counts[db_key] > 1 or len(db_matches) > 1):
            entries.append(MergeEntry('conflict', item, reason='identity',
                                      conflict_with=db_matches[0] if db_matches else None))
            continue
        if db_matches:
            previous = db_matches[0]
            item = preserve_local_geology(item, previous, by_id=True)
            if (content(item) == content(previous) and item.get('raw_cells') == previous.get('raw_cells')
                    and item.get('import_source', source) == previous.get('import_source', current_source)):
                entries.append(MergeEntry('duplicate', item))
            else:
                entries.append(MergeEntry('update', item, previous['id'], previous))
                if update_database:
                    replacements[previous['id']] = len(entries) - 1
            continue
        matches = existing.get(geometry, ())
        if db_key is not None and matches:
            entries.append(MergeEntry('conflict', item, previous=matches[0], reason='bounds'))
            continue
        if db_import and len(matches) == 1:
            if database_interval_key(matches[0], current_source) is not None:
                entries.append(MergeEntry('conflict', item, previous=matches[0], reason='identity'))
                continue
            item = preserve_local_geology(item, matches[0])
        key = content(item)
        if key in seen:
            entries.append(MergeEntry('duplicate', item))
            continue
        seen.add(key)
        candidates[geometry].append(len(entries))
        entries.append(MergeEntry('add', item))
    addition_tracks = set()
    for geometry, indexes in candidates.items():
        matches = existing.get(geometry, [])
        if len(indexes) > 1 or len(matches) > 1:
            for index in indexes:
                entries[index] = replace(entries[index], kind='conflict', reason='bounds')
            # Ambiguous incoming geometry also blocks overlapping additions.
            if not matches:
                tracks[geometry[:2]].append((*geometry[2:], None, entries[indexes[0]].item))
        elif matches:
            index = indexes[0]
            entries[index] = MergeEntry('update', entries[index].item, matches[0]['id'], matches[0])
        else:
            tracks[geometry[:2]].append((*geometry[2:], indexes[0], entries[indexes[0]].item))
            addition_tracks.add(geometry[:2])
    if replacements:
        check_database_geometry(current, entries, replacements)
    else:
        mark_overlapping_entries(entries, (tracks[key] for key in addition_tracks))
    for index, entry in enumerate(entries):
        if entry.kind in ('add', 'update'):
            item = dict(entry.item)
            item.setdefault('import_source', source)
            if entry.target:
                item['id'] = entry.target
            elif item['id'] in ids:
                item['id'] = new_id()
            ids.add(item['id'])
            entries[index] = replace(entry, item=item)
    return entries


def check_database_geometry(current, entries, replacements):
    """Check the proposed scheme with old versions of updated rows removed."""
    tracks = defaultdict(list)
    for item in current['intervals']:
        if item['id'] not in replacements:
            tracks[track(item)].append((*bounds(item), None, item))
    for index, entry in enumerate(entries):
        if entry.kind == 'add':
            tracks[track(entry.item)].append((*bounds(entry.item), index, entry.item))
        elif entry.target in replacements:
            changed = (track(entry.item), bounds(entry.item)) != (track(entry.previous), bounds(entry.previous))
            tracks[track(entry.item)].append((*bounds(entry.item), index if changed else None, entry.item))
    mark_overlapping_entries(entries, tracks.values())


def mark_overlapping_entries(entries, tracks):
    for segments in tracks:
        segments.sort(key=lambda segment: segment[:2])
        covering = None
        for position, segment in enumerate(segments):
            top, bottom, index, _ = segment
            if index is not None:
                other = covering if covering and covering[1] > top else None
                if other is None and position + 1 < len(segments) and segments[position + 1][0] < bottom:
                    other = segments[position + 1]
                if other is not None:
                    entries[index] = replace(entries[index], kind='conflict', reason='overlap',
                                             conflict_with=other[3])
            if covering is None or bottom > covering[1]:
                covering = segment
