"""Piecewise linear thresholds with an explicitly declared dependent axis."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Threshold:
    id: str
    independent: int
    coordinates: np.ndarray
    values: np.ndarray

    def relations(self, bounds, tolerance):
        """Return every relation attained by a point of the analysis rectangle."""
        axis = self.independent
        low, high = bounds[axis], bounds[axis + 2]
        domain_low, domain_high = self.coordinates[0], self.coordinates[-1]
        if high < domain_low or low > domain_high:
            return ['outside_domain']
        start, end = max(low, domain_low), min(high, domain_high)
        endpoints = np.interp([start, end], self.coordinates, self.values)
        minimum, maximum = min(endpoints), max(endpoints)
        if start != end:
            first = self.coordinates.searchsorted(start, side='right')
            last = self.coordinates.searchsorted(end, side='left')
            interior = self.values[first:last]
            if interior.size:
                minimum = min(minimum, interior.min())
                maximum = max(maximum, interior.max())
        dependent = 1 - axis
        smallest = bounds[dependent] - maximum
        largest = bounds[dependent + 2] - minimum
        relations = []
        if smallest < -tolerance:
            relations.append('less')
        if smallest <= tolerance and largest >= -tolerance:
            relations.append('boundary')
        if largest > tolerance:
            relations.append('greater')
        if low < domain_low or high > domain_high:
            relations.append('outside_domain')
        return relations


def prepare_threshold(entry, geometry, strategy):
    rule = entry['threshold']
    if (strategy != 'regions' or geometry.geom_type != 'LineString'
            or not isinstance(rule, dict) or rule.get('axis') not in ('x', 'y')
            or any(not isinstance(rule.get(side), str) or not rule[side].strip()
                   for side in ('less', 'greater'))):
        raise ValueError(f"Порог {entry['id']} требует LineString, ось x или y и подписи less/greater")
    independent = 0 if rule['axis'] == 'y' else 1
    coordinates = np.asarray(geometry.coords)
    if coordinates[0, independent] > coordinates[-1, independent]:
        coordinates = coordinates[::-1]
    if not np.all(np.diff(coordinates[:, independent]) > 0):
        raise ValueError(f"Порог {entry['id']} должен быть однозначной функцией по независимой оси")
    return Threshold(entry['id'], independent, coordinates[:, independent], coordinates[:, 1 - independent])


def threshold_labels(scheme, results):
    """Present the declared classes and explicit boundary/domain outcomes."""
    if not results:
        return []
    labels = []
    for entry in scheme['lines']:
        result = results.get(entry['id'])
        if result is None:
            continue
        names = {**entry['threshold'], 'boundary': 'на пороге', 'outside_domain': 'вне области порога'}
        label = f"{entry['label']}: {names[result['relation']]}"
        if len(result['possible_relations']) > 1:
            label += ' (интервал: ' + ', '.join(names[r] for r in result['possible_relations']) + ')'
        labels.append(label)
    return labels
