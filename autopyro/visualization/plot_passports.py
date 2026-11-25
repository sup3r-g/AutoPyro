"""Versioned axis presets for descriptive pyrolysis diagrams."""
PASSPORTS = {
    "HI–OI": ("pyro.hi-oi.v1", "OI", "HI"),
    "HI–Tmax": ("pyro.hi-tmax.v1", "Tmax", "HI"),
    "S2–TOC": ("pyro.s2-toc.v1", "TOC", "S2"),
    "S1–TOC": ("pyro.s1-toc.v1", "TOC", "S1"),
    "S1–S2": ("pyro.s1-s2.v1", "S1", "S2"),
    "PI–Tmax": ("pyro.pi-tmax.v1", "Tmax", "PI"),
    "HI–TOC": ("pyro.hi-toc.v1", "TOC", "HI"),
    "S1+S2–TOC": ("pyro.gp-toc.v1", "TOC", "GP"),
}

TEMPLATES = {title: (x, y) for title, (_, x, y) in PASSPORTS.items()}


PROFILE_PASSPORTS = {
    f'{role}–MD': (f'pyro.profile-{role.lower()}-md.v1', role, 'MD')
    for role in ('TOC', 'S1', 'S2', 'S3', 'Tmax', 'HI', 'OI', 'PI')
}
INTERVAL_PASSPORTS = {
    f'{role}–MD interval': (f'pyro.interval-{role.lower()}-md.v1', role, 'MD_TOP', 'MD_BOTTOM')
    for _, role, _ in PROFILE_PASSPORTS.values()
}
TABLET_TITLE = 'TOC / S1 / S2 / Tmax–MD'
PASSPORTS.update(PROFILE_PASSPORTS)
PASSPORTS.update(INTERVAL_PASSPORTS)
PASSPORTS[TABLET_TITLE] = ('pyro.tablet-basic-md.v1', 'TOC', 'S1', 'S2', 'Tmax', 'MD')


def template_kind(title):
    if title == TABLET_TITLE:
        return 'multi_depth'
    return 'depth' if title in PROFILE_PASSPORTS or title in INTERVAL_PASSPORTS else 'scatter'


def axis_requirements(title):
    from autopyro.schemes.interpretation_quality import CANONICAL_UNITS

    units = {**CANONICAL_UNITS, 'PI': '1', 'GP': 'mg HC/g rock', 'S3': 'mg CO2/g rock',
             'MD': 'm', 'MD_TOP': 'm', 'MD_BOTTOM': 'm'}
    return [(role, units[role.upper()]) for role in PASSPORTS[title][1:]]


def matches_axis(column, role, unit):
    from autopyro.schemes.interpretation import unit_key
    from autopyro.schemes.interpretation_quality import column_property_name, property_name

    actual = column_property_name(column)
    expected = property_name(role)
    if role in ('MD_TOP', 'MD_BOTTOM'):
        actual = actual.replace(' ', '_')
    return actual == expected and unit_key(column.unit) == unit_key(unit)


def plot_passport(spec, columns):
    title = next((title for title, definition in PASSPORTS.items()
                  if definition[0] == spec.passport_id), None)
    if title is None or spec.kind != template_kind(title) or spec.interpretation_panels:
        raise ValueError('Вид графика не соответствует паспорту')
    tablet = spec.kind == 'multi_depth'
    depth_plot = spec.kind in ('depth', 'multi_depth')
    interval = title in INTERVAL_PASSPORTS
    if interval:
        if not spec.depth_bottom_id or spec.depth_bottom_id == spec.y_id:
            raise ValueError('Для паспорта интервального профиля выберите отдельные колонки MD_TOP и MD_BOTTOM')
    elif depth_plot and spec.depth_bottom_id:
        raise ValueError('Паспорт графика по MD предназначен для точечных проб')
    requirements = axis_requirements(title)
    tracks = spec.track_ids if tablet else [spec.x_id]
    bindings = [*tracks, spec.y_id]
    axis_names = [f'X{index + 1}' for index in range(len(tracks))] if tablet else ['X']
    axis_names.append('Y')
    if interval:
        bindings.append(spec.depth_bottom_id)
        axis_names.append('MD_BOTTOM')
    if len(bindings) != len(requirements):
        raise ValueError(f'{title}: состав треков не соответствует паспорту')
    axes = []
    for axis, cid, (role, unit) in zip(axis_names, bindings, requirements, strict=True):
        column = columns.get(cid)
        if column is None or not matches_axis(column, role, unit):
            raise ValueError(f'{title}: для оси {axis} требуется {role} [{unit}]')
        axes.append({'property': role, 'unit': unit, 'column_id': column.id})
    applicability = {
        'scatter': 'descriptive_pyrolysis_xy',
        'depth': 'descriptive_pyrolysis_md_profile',
        'multi_depth': 'descriptive_pyrolysis_md_tablet',
    }
    passport = {'id': spec.passport_id, 'version': 1, 'name': title,
                'axes': {'y': axes[len(tracks)]}, 'applicability': applicability[spec.kind],
                'source': 'SCIENTIFIC_METHODS.md',
                'classification_layer': 'separate' if spec.interpretation else 'none'}
    if tablet:
        passport['tracks'] = axes[:-1]
    else:
        passport['axes']['x'] = axes[0]
    if depth_plot:
        passport['depth_reference'] = 'MD'
    if interval:
        passport['applicability'] = 'descriptive_pyrolysis_md_interval'
        passport['axes']['bottom'] = axes[-1]
    return passport
