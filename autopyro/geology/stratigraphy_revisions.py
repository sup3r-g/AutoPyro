"""Read-only comparison of saved interval and sample interpretations."""

from collections import defaultdict

from .stratigraphy import LEVELS, evaluate_project


def interval_changes(before, after):
    old = {item['id']: item for item in before['intervals']}
    new = {item['id']: item for item in after['intervals']}
    pairs = [(old.pop(key), new.pop(key), 'id') for key in tuple(old) if key in new]
    comparable_depth = all(before[key] == after[key] for key in ('depth_unit', 'depth_reference'))
    if comparable_depth:
        # Match only unique identities after re-imports that assigned fresh IDs.
        for method, fields in (
            ('name_bounds', ('wellbore', 'level', 'name', 'top', 'bottom')),
            ('bounds', ('wellbore', 'level', 'top', 'bottom')),
            ('name_overlap', ('wellbore', 'level', 'name')),
        ):
            left, right = defaultdict(list), defaultdict(list)
            for source, groups in ((old, left), (new, right)):
                for key, item in source.items():
                    groups[tuple(item[field] for field in fields)].append(key)
            for key, matches in left.items():
                if len(matches) != 1 or len(right.get(key, ())) != 1:
                    continue
                old_id, new_id = matches[0], right[key][0]
                first, second = old[old_id], new[new_id]
                if method == 'name_overlap' and max(first['top'], second['top']) >= min(first['bottom'], second['bottom']):
                    continue
                pairs.append((old.pop(old_id), new.pop(new_id), method))
    changes = []
    identities = {}
    for left, right, method in pairs:
        identities[right['id']] = left['id']
        fields = tuple(key for key in ('wellbore', 'level', 'name', 'top', 'bottom')
                       if left[key] != right[key] or (key in ('top', 'bottom') and not comparable_depth))
        if left.get('lithology', '').strip() != right.get('lithology', '').strip():
            fields += ('lithology',)
        if fields:
            changes.append((fields, left, right, method))
    changes.extend((('removed',), item, None, None) for item in old.values())
    changes.extend((('added',), None, item, None) for item in new.values())
    return changes, identities


def membership_key(result, identities):
    members = sorted((identities.get(key, key), name)
                     for key, name in zip(result['interval_ids'], result['names'], strict=True))
    overlaps = sorted((identities.get(item['interval_id'], item['interval_id']), round(item['fraction'], 10))
                      for item in result.get('overlaps', ()))
    coverage = tuple(round(result[field], 10) if result.get(field) is not None else None
                     for field in ('coverage_fraction', 'ambiguous_length'))
    return result['status'], members, overlaps, coverage


def sample_changes(project, row_ids, before, after, binding, identities, *, before_binding=None,
                   checkpoint=None, after_report=None):
    first = evaluate_project(project, row_ids, before,
                             binding if before_binding is None else before_binding,
                             checkpoint=checkpoint)['results']
    if after_report is None:
        after_report = evaluate_project(project, row_ids, after, binding, checkpoint=checkpoint)
    second = after_report['results']
    changes = []
    for number, (old, new) in enumerate(zip(first, second, strict=True)):
        if checkpoint and number % 256 == 0:
            checkpoint()
        for level in LEVELS:
            left, right = old['levels'][level], new['levels'][level]
            if left != right and membership_key(left, {}) != membership_key(right, identities):
                changes.append((new, level, left, right))
    return changes
