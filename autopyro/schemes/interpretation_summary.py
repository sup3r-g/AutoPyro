"""Group interpretation results using the saved stratigraphic binding."""

from collections import defaultdict

from autopyro.geology.stratigraphy import evaluate_project
from autopyro.geology.stratigraphy_samples import sample_positions


EXCLUDED_AXES = {'qc_excluded', 'range_excluded', 'interpolation_excluded'}


def prepare_summary(project, report, scheme, binding):
    row_ids = [item['row_id'] for item in report['results']]
    stratigraphy = evaluate_project(project, row_ids, scheme, binding)
    wells = defaultdict(list)
    for location, result, sample in zip(stratigraphy['results'], report['results'], report['inputs'], strict=True):
        well = location['wellbore']
        well = well if isinstance(well, str) and well.strip() else None
        excluded = bool(EXCLUDED_AXES.intersection(sample['statuses']))
        wells[well].append((location, result, excluded))
    positions = sample_positions(project, row_ids, scheme, binding, wells)
    return dict(wells), positions


def group_intervals(samples, level):
    """Partition a well's samples, keeping unresolved memberships separate."""
    groups = defaultdict(list)
    for entry in samples:
        match = entry[0]['levels'][level]
        if match['status'] == 'matched':
            key = ('matched', match['interval_ids'][0])
        else:
            key = (match['status'], None)
        groups[key].append(entry)
    return dict(groups)
