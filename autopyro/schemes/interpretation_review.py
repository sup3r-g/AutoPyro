"""Expert review of a sample against a specific interpretation result."""

from datetime import datetime, timezone
from copy import deepcopy

from .interpretation import scheme_hash


def review_scheme_hash(scheme):
    """Identify interpretation rules independently of revision and presentation."""
    rules = {key: scheme[key] for key in (
        'strategy', 'boundary_policy', 'tolerance', 'distance_metric',
        'distance_tolerance', 'max_distance', 'interval_max_evaluations', 'hi_resolution',
        'curve_alignment', 'intermediate_curves', 'no_extrapolation',
    ) if key in scheme}
    for dimension in ('x', 'y'):
        axis = scheme[dimension]
        rules[dimension] = {key: axis[key] for key in (
            'property', 'unit', 'minimum', 'maximum', 'transform', 'reference', 'origin', 'scale',
        ) if key in axis}
    for field in ('regions', 'lines'):
        rules[field] = [
            {key: entry[key] for key in ('id', 'label', 'geometry', 'threshold', 'reference_only')
             if key in entry}
            for entry in scheme.get(field, ())
        ]
    return scheme_hash(rules)


def is_boundary_result(item):
    return item['status'] == 'boundary' or any(
        threshold['relation'] == 'boundary' for threshold in item.get('thresholds', {}).values())


def review_reasons(report, position):
    reasons = []
    agreement = report.get('type_agreement')
    if agreement is not None and agreement[position] in ('partial', 'different'):
        reasons.append('disagreement')
    diagrams = (report, *report.get('comparisons', {}).values())
    if any(diagram['results'][position]['status'] == 'ambiguous' for diagram in diagrams):
        reasons.append('ambiguous')
    if any(is_boundary_result(diagram['results'][position]) for diagram in diagrams):
        reasons.append('boundary')
    if any(diagram['inputs'][position].get('quality') for diagram in diagrams):
        reasons.append('quality')
    return reasons


def _evidence(report, position, *, semantic=False):
    # Scope and display order can change without changing a sample's evidence.
    def evidence(diagram):
        digest = diagram['scheme_hash']
        if semantic:
            digest = diagram.get('review_scheme_hash') or review_scheme_hash(diagram['scheme'])
        return {
            'scheme': digest,
            'definition': diagram['definition_hash'],
            'binding': diagram['binding'],
            'input': diagram['inputs'][position],
            'result': diagram['results'][position],
        }

    return {
        'primary': evidence(report),
        'comparisons': {name: evidence(diagram)
                        for name, diagram in report.get('comparisons', {}).items()},
    }


def _fingerprints(evidence):
    return {
        'primary': scheme_hash(evidence['primary']),
        'comparisons': {name: scheme_hash(value) for name, value in evidence['comparisons'].items()},
    }


def _evidence_parts(evidence):
    def parts(diagram):
        sample = {**diagram['input']}
        quality = {key: sample.pop(key, None) for key in ('quality', 'quality_blocked')}
        return {
            'scheme': diagram['scheme'],
            'definition': diagram['definition'],
            'binding': scheme_hash(diagram['binding']),
            'input': scheme_hash(sample),
            'quality': scheme_hash(quality),
            'result': scheme_hash(diagram['result']),
        }

    return {
        'primary': parts(evidence['primary']),
        'comparisons': {name: parts(diagram) for name, diagram in evidence['comparisons'].items()},
    }


def review_changes(report, position, decision):
    """Explain changed evidence for a decision already marked stale."""
    saved = decision.get('evidence_parts')
    if saved is None:
        return []
    current = _evidence_parts(_evidence(report, position, semantic=True))
    pairs = [(report['scheme']['name'], saved['primary'], current['primary'])]
    for name, previous in saved['comparisons'].items():
        if name in current['comparisons']:
            pairs.append((name, previous, current['comparisons'][name]))
    return [(name, [key for key, value in new.items() if old.get(key) != value])
            for name, old, new in pairs if old != new]


def _changed(saved, current):
    old, new = saved['comparisons'], current['comparisons']
    return (saved['primary'] != current['primary']
            or any(old[name] != new[name] for name in old.keys() & new.keys()))


def _legacy_fingerprints(project, row_id, digest):
    # A retained report can recover the evidence for an older single-hash decision.
    for key in ('interpretation_result', 'interpretation_previous_result', 'interpretation_comparison_base'):
        snapshot = project.view.get(key)
        if not snapshot:
            continue
        position = next((index for index, item in enumerate(snapshot['results'])
                         if item['row_id'] == row_id), None)
        if position is None:
            continue
        evidence = _evidence(snapshot, position)
        if scheme_hash(evidence) == digest:
            return _fingerprints(evidence)
    return None


def migrate_review_evidence(project, *reports):
    """Recover old decisions from retained snapshots with matching evidence."""
    reviews = project.view.get('interpretation_reviews', {})
    for report in reports:
        if not report:
            continue
        decisions = reviews.get(report['scheme']['id'], {})
        legacy = {row_id for row_id, decision in decisions.items() if decision.get('evidence_version') != 2}
        if not legacy:
            continue
        for position, item in enumerate(report['results']):
            row_id = item['row_id']
            if row_id not in legacy:
                continue
            decision = decisions[row_id]
            evidence = _evidence(report, position)
            previous = decision.get('evidence')
            matches = (previous == _fingerprints(evidence) if previous is not None
                       else decision.get('evidence_hash') == scheme_hash(evidence))
            if matches:
                semantic = _evidence(report, position, semantic=True)
                decision.update(evidence_version=2, evidence=_fingerprints(semantic),
                                evidence_parts=_evidence_parts(semantic))
                decision.pop('evidence_hash', None)


def review_state(project, report, position):
    row_id = report['results'][position]['row_id']
    reviews = project.view.get('interpretation_reviews', {}).get(report['scheme']['id'], {})
    decision = reviews.get(row_id)
    if decision is None:
        return 'pending' if review_reasons(report, position) else None, None
    evidence = _evidence(report, position, semantic=decision.get('evidence_version') == 2)
    saved = decision.get('evidence')
    if saved is None:
        digest = decision['evidence_hash']
        if digest in (scheme_hash(evidence), scheme_hash(dict(evidence, comparisons={}))):
            return decision['status'], decision
        saved = _legacy_fingerprints(project, row_id, digest)
    if saved is None or _changed(saved, _fingerprints(evidence)):
        return 'stale', decision
    return decision['status'], decision


def review_history(project, scheme_id, row_id):
    view = project.view
    previous = view.get('interpretation_review_history', {}).get(scheme_id, {}).get(row_id, ())
    current = view.get('interpretation_reviews', {}).get(scheme_id, {}).get(row_id)
    return [*previous, current] if current is not None else list(previous)


def save_review(project, report, position, status, conclusion='', note='', *, author=''):
    conclusion, note = conclusion.strip(), note.strip()
    if status not in ('accepted', 'expert', 'open'):
        raise ValueError('Неизвестный статус экспертной проверки')
    if status == 'expert' and (not conclusion or not note):
        raise ValueError('Введите экспертный вывод и его обоснование')
    row_id = report['results'][position]['row_id']
    scheme = report['scheme']
    evidence = _evidence(report, position, semantic=True)
    decision = {
        'status': status,
        'author': author.strip(),
        'conclusion': conclusion if status == 'expert' else '',
        'note': note,
        'scheme_revision': scheme['revision'],
        'scheme_hash': report['scheme_hash'],
        'evidence_version': 2,
        'evidence': _fingerprints(evidence),
        'evidence_parts': _evidence_parts(evidence),
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }
    reviews = project.view.setdefault('interpretation_reviews', {}).setdefault(scheme['id'], {})
    if row_id in reviews:
        history = project.view.setdefault('interpretation_review_history', {}).setdefault(scheme['id'], {})
        history.setdefault(row_id, []).append(deepcopy(reviews[row_id]))
    reviews[row_id] = decision
    return decision
