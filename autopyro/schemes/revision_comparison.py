"""Differences between revisions evaluated on the same project inputs."""

from collections import Counter

from .interpretation_thresholds import threshold_labels


def threshold_signature(item, scheme):
    thresholds = item.get('thresholds', {})
    relations = tuple(sorted((key, value['relation'], frozenset(value['possible_relations']))
                             for key, value in thresholds.items()))
    return relations, tuple(sorted(threshold_labels(scheme, thresholds)))


def outcome_labels(scheme):
    return {entry['id']: entry['label']
            for field in ('regions', 'lines') for entry in scheme.get(field, ())}


def comparison_labels(before, after):
    left, right = outcome_labels(before), outcome_labels(after)
    counts = Counter(label for _, label in left.items() | right.items())
    return tuple({key: f'{label} [{key}]' if counts[label] > 1 else label
                  for key, label in labels.items()} for labels in (left, right))


def outcome_signature(item, scheme, names):
    if scheme.get('strategy') in ('nearest_curve', 'initial_hi'):
        types = item.get('possible_types') or item.get('line_hits', ())
    else:
        types = [*item.get('regions', ()), *item.get('boundary_regions', ()),
                 *(key for key in item.get('line_hits', ()) if key not in item.get('thresholds', {}))]
    interval_regions = item.get('uncertainty_regions', ())
    return (frozenset((key, names.get(key, key)) for key in types),
            frozenset((key, names.get(key, key)) for key in interval_regions),
            item['status'], tuple(item.get('value_bounds') or ()), threshold_signature(item, scheme),
            item.get('value'))


def changed_samples(before, after, *, checkpoint=None):
    old_rows = {item['row_id']: item for item in before['results']}
    new_rows = {item['row_id']: item for item in after['results']}
    if old_rows.keys() != new_rows.keys():
        raise ValueError('Для сравнения редакций нужен один набор проб')

    old_labels, new_labels = comparison_labels(before['scheme'], after['scheme'])
    for index, (row_id, new) in enumerate(new_rows.items()):
        if checkpoint is not None and index % 256 == 0:
            checkpoint()
        old = old_rows[row_id]
        previous = outcome_signature(old, before['scheme'], old_labels)
        proposed = outcome_signature(new, after['scheme'], new_labels)
        if previous != proposed:
            yield row_id, previous, proposed
