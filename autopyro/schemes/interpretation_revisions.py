"""Store scheme revisions and compare calculated outcomes by sample UUID."""

from copy import deepcopy
from dataclasses import dataclass

from .interpretation import check_saved_result, scheme_hash
from .interpretation_review import review_state
from .interpretation_thresholds import threshold_labels
from .revision_comparison import comparison_labels, outcome_labels, outcome_signature


def validate_scheme_revision(project, scheme):
    previous = project.view.get("interpretation_schemes", {}).get(scheme["id"])
    digest = scheme_hash(scheme)
    if previous and previous["revision"] == scheme["revision"] and scheme_hash(previous) != digest:
        raise ValueError('Изменённая схема должна иметь новую revision')
    archived = project.view.get("interpretation_scheme_revisions", {}).get(scheme["id"], {})
    same_revision = archived.get(scheme["revision"])
    if same_revision is not None and scheme_hash(same_revision) != digest:
        raise ValueError('Такая revision уже использовалась для другого содержимого; задайте новую')
    return previous


def store_scheme_revision(project, scheme):
    """Archive and audit a changed scheme and invalidate any stale result."""
    previous = validate_scheme_revision(project, scheme)
    if previous == scheme:
        return False
    scheme_id, revision = scheme["id"], scheme["revision"]
    history = project.view.setdefault("interpretation_scheme_revisions", {}).setdefault(scheme_id, {})
    if previous:
        history[previous["revision"]] = deepcopy(previous)
    history[revision] = deepcopy(scheme)
    project.view.setdefault("interpretation_schemes", {})[scheme_id] = deepcopy(scheme)
    check_saved_result(project)
    project.event("interpretation_scheme", id=scheme_id, revision=revision, sha256=scheme_hash(scheme))
    return True


def _diagrams(report):
    return {None: report, **report.get('comparisons', {})}


def _revision_data(report):
    return {
        name: tuple(diagram.get(key) for key in (
            'scheme_hash', 'definition_hash', 'binding', 'inputs', 'results', 'type_agreement',
        ))
        for name, diagram in _diagrams(report).items()
    }


def retain_previous_revision(view, report):
    previous = view.get('interpretation_result')
    if previous and _revision_data(previous) != _revision_data(report):
        view['interpretation_previous_result'] = previous


def same_snapshot_metadata(previous, current):
    """Compare persisted metadata of reports with identical calculation data."""
    if not previous:
        return False

    def metadata(report):
        ignored = {'created_at', 'inputs', 'results', 'comparisons'}
        return {name: {key: value for key, value in diagram.items() if key not in ignored}
                for name, diagram in _diagrams(report).items()}

    old, new = metadata(previous), metadata(current)
    return old.keys() == new.keys() and all(
        old[name] == new[name] or scheme_hash(old[name]) == scheme_hash(new[name])
        for name in old
    )


def comparison_base(view):
    return view.get('interpretation_comparison_base') or view.get('interpretation_previous_result')


@dataclass
class RevisionChange:
    row_id: str
    before: dict
    after: dict
    flags: list[str]
    positions: tuple[int | None, int | None]


def input_differences(previous, current, change):
    """Yield differences for one sample on diagrams present in both snapshots."""
    a, b = change.positions
    if a is None or b is None:
        return
    old, new = _diagrams(previous), _diagrams(current)

    def axis_binding(diagram, axis):
        binding = diagram['binding']
        return (diagram['scheme'][axis]['property'], binding.get(f'{axis}_id'),
                binding.get(f'{axis}_unit'))

    def axis_input(sample, axis, index):
        statuses = sample.get('statuses', ())
        uncertainty = sample.get('uncertainty', {}).get(axis, {})
        measurement = sample.get('measurements', {}).get(axis)
        if measurement is None and 'measurement_id' in uncertainty:
            measurement = {'id': uncertainty['measurement_id'], 'revision': uncertainty['revision']}
        return {
            'value': sample.get(axis),
            'status': statuses[index] if index < len(statuses) else None,
            'measurement': measurement,
            'uncertainty': {key: value for key, value in uncertainty.items()
                            if key not in ('measurement_id', 'revision')} or None,
            'bounds': tuple(sample.get(f'{axis}_bounds') or ()),
        }

    for name, before in old.items():
        after = new.get(name)
        if after is None:
            continue
        left, right = before['inputs'][a], after['inputs'][b]
        for index, axis in enumerate(('x', 'y')):
            old_binding, new_binding = axis_binding(before, axis), axis_binding(after, axis)
            old_fields, new_fields = axis_input(left, axis, index), axis_input(right, axis, index)
            if old_binding != new_binding:
                yield name, axis, 'binding', old_binding, new_binding
            for key, value in old_fields.items():
                if value != new_fields[key]:
                    yield name, axis, key, value, new_fields[key]
        for field, old_value, new_value in (
            ('definition', before.get('definition_hash'), after.get('definition_hash')),
            ('text_numbers', before['binding'].get('text_numbers'), after['binding'].get('text_numbers')),
            ('quality', left.get('quality', ()), right.get('quality', ())),
            ('quality_blocked', bool(left.get('quality_blocked')), bool(right.get('quality_blocked'))),
        ):
            if old_value != new_value:
                yield name, None, field, old_value, new_value


def _outcome(diagram, labels, position):
    item = diagram['results'][position]
    thresholds = item.get('thresholds', {})
    ids = (item['regions'] + item['boundary_regions']
           + (item.get('possible_types') or item['line_hits']))
    names = [labels.get(key, key) for key in ids if key not in thresholds]
    names.extend(threshold_labels(diagram['scheme'], thresholds))
    return (item['status'], tuple(dict.fromkeys(names)), item.get('value'), item.get('value_bounds'),
            tuple(labels.get(key, key) for key in item.get('uncertainty_regions', ())))


def revision_changes(project, previous, current):
    old, new = _diagrams(previous), _diagrams(current)
    old_rows = {item['row_id']: i for i, item in enumerate(previous['results'])}
    new_rows = {item['row_id']: i for i, item in enumerate(current['results'])}

    old_labels = {name: outcome_labels(diagram['scheme']) for name, diagram in old.items()}
    new_labels = {name: outcome_labels(diagram['scheme']) for name, diagram in new.items()}
    old_agreement = previous.get('type_agreement', ())
    new_agreement = current.get('type_agreement', ())
    shared = old.keys() & new.keys()
    for name in shared:
        old_labels[name], new_labels[name] = comparison_labels(old[name]['scheme'], new[name]['scheme'])
    binding_changed = any(old[name].get(key) != new[name].get(key)
                          for name in shared for key in ('binding', 'definition_hash'))

    for row_id in dict.fromkeys([*new_rows, *old_rows]):
        a, b = old_rows.get(row_id), new_rows.get(row_id)
        flags = []
        if a is None:
            flags.append('added')
        elif b is None:
            flags.append('removed')
        else:
            if binding_changed or any(old[name]['inputs'][a] != new[name]['inputs'][b] for name in shared):
                flags.append('inputs')
            if old.keys() != new.keys() or any(
                outcome_signature(old[name]['results'][a], old[name]['scheme'], old_labels[name])
                != outcome_signature(new[name]['results'][b], new[name]['scheme'], new_labels[name])
                for name in shared
            ):
                flags.append('outcome')
            old_uncertain = {name for name, diagram in old.items() if diagram['results'][a]['status'] == 'ambiguous'}
            new_uncertain = {name for name, diagram in new.items() if diagram['results'][b]['status'] == 'ambiguous'}
            if new_uncertain - old_uncertain:
                flags.append('ambiguity_added')
            if old_uncertain - new_uncertain:
                flags.append('ambiguity_removed')
            old_type = old_agreement[a] if old_agreement else None
            new_type = new_agreement[b] if new_agreement else None
            if old_type != new_type:
                flags.append('agreement')
        if b is not None and review_state(project, current, b)[0] == 'stale':
            flags.append('review')
        if flags:
            before = {name: _outcome(diagram, old_labels[name], a)
                      for name, diagram in old.items()} if a is not None else {}
            after = {name: _outcome(diagram, new_labels[name], b)
                     for name, diagram in new.items()} if b is not None else {}
            yield RevisionChange(row_id, before, after, flags, (a, b))
