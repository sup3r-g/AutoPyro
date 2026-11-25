"""Size thresholds for preparing project snapshots in a worker."""


def large_project(project):
    return len(project.rows) * len(project.columns) >= 100_000


def large_snapshot(project):
    return large_project(project) or len(project.science.measurements) >= 100_000
