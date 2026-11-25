"""Thickness and lithology measured along the scheme's declared depth axis."""

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Thickness:
    covered: float = 0
    gap: float | None = None
    overlap: float = 0
    rocks: dict = field(default_factory=lambda: defaultdict(float))


def merged_spans(intervals):
    """Disjoint spans of intervals ordered by top depth."""
    start, end = intervals[0]
    for top, bottom in intervals[1:]:
        if top > end:
            yield start, end
            start = top
        end = max(end, bottom)
    yield start, end


def summarize_intervals(intervals, rock_keys):
    """Partition each well and level into uncovered, unique and disputed spans.

    Identical geological claims count once. Composition includes only uniquely
    covered depth, even where competing claims name the same subdivision.
    """
    groups = defaultdict(set)
    for item in intervals:
        rock = rock_keys[item.get('lithology', '')]
        groups[item['wellbore'], item['level']].add((item['top'], item['bottom'], item['name'], rock))
    results = {}
    for (well, level), claims in sorted(groups.items()):
        items = sorted(claims, key=lambda claim: claim[:3])
        total = results[well, level, None] = Thickness(gap=0)
        sections = defaultdict(list)
        events = defaultdict(list)
        for index, (top, bottom, name, _) in enumerate(items):
            events[top].append(index)
            events[bottom].append(index)
            sections[name].append((top, bottom))
        subdivisions = {name: Thickness() for name in sorted(sections)}
        active = set()
        overlap_at = {}
        previous = items[0][0]
        for depth in sorted(events):
            length = depth - previous
            if length > 0 and active:
                total.covered += length
                if len(active) > 1:
                    total.overlap += length
                else:
                    _, _, name, rock = items[next(iter(active))]
                    total.rocks[rock] += length
                    subdivisions[name].rocks[rock] += length
            elif not active:
                total.gap += length
            overlap_at[depth] = total.overlap
            for index in events[depth]:
                # Each interval has exactly two events and positive thickness.
                if index in active:
                    active.remove(index)
                else:
                    active.add(index)
            previous = depth
        for name, subdivision in subdivisions.items():
            for top, bottom in merged_spans(sections[name]):
                subdivision.covered += bottom - top
                subdivision.overlap += overlap_at[bottom] - overlap_at[top]
            results[well, level, name] = subdivision
    return results
