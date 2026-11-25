from autopyro.stratigraphy_summary import summarize_intervals


def test_summary_counts_disputed_depth_once_and_separates_gaps_and_levels():
    def layer(top, bottom, name, rock='', level='FORMATION'):
        return dict(wellbore='A', level=level, top=top, bottom=bottom, name=name, lithology=rock)

    intervals = [
        layer(0, 10, 'A', 'sand'), layer(0, 10, 'A', 'sand'),
        layer(5, 8, 'B', 'clay'), layer(7, 12, 'A', 'sand'),
        layer(14, 20, 'C'),
        layer(0, 20, 'Parent', level='GROUP'),
    ]
    rows = summarize_intervals(intervals, {'sand': 'sand', 'clay': 'clay', '': None})
    total = rows['A', 'FORMATION', None]
    assert (total.covered, total.gap, total.overlap) == (18, 2, 5)
    assert total.rocks == {'sand': 7, None: 6}
    first = rows['A', 'FORMATION', 'A']
    assert (first.covered, first.gap, first.overlap) == (12, None, 5)
    assert rows['A', 'FORMATION', 'B'].overlap == 3
    assert not rows['A', 'FORMATION', 'B'].rocks
    parent = rows['A', 'GROUP', None]
    assert (parent.covered, parent.gap, parent.overlap) == (20, 0, 0)
