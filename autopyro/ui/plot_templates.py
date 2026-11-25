"""Portable plot templates and their bindings to project columns."""

import json
from copy import deepcopy
from pathlib import Path

from autopyro.data_io.file_output import write_json
from autopyro.visualization.plot_controls import column_compatible
from autopyro.visualization.plotting import PlotSpec

from .i18n import tr


def write_plot_template(path, spec, columns):
    payload = {
        "format": "autopyro.plot",
        "version": 1,
        "spec": spec.to_dict(),
        "columns": [
            {"id": c.id, "name": c.name, "unit": c.unit,
             "kind": c.kind, "basis": c.metadata.get('basis', '')}
            for c in columns
        ],
    }
    write_json(path, payload)


def read_plot_template(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "autopyro.plot"
        or payload.get("version") != 1
    ):
        raise ValueError(tr('Неизвестный формат шаблона'))
    return payload


def bind_plot_template(payload, columns):
    spec = PlotSpec.from_dict(deepcopy(payload["spec"]))
    existing = {c.id: c for c in columns}
    by_name = {}
    for column in existing.values():
        by_name.setdefault(column.name, []).append(column)
    column_fields = (
        'x_id', 'y_id', 'z_id', 'depth_bottom_id',
        'color_by', 'shape_by', 'size_by', 'label_by', 'line_group_by',
    )
    referenced = {getattr(spec, key) for key in column_fields}
    from autopyro.model.filtering import leaf_rules

    referenced.update(rule['column'] for rule in leaf_rules(spec.filter_rules))
    referenced.update(spec.track_ids)
    referenced.update(item['track_id'] for item in spec.annotations if item.get('track_id'))
    referenced.update(spec.track_axes)
    referenced.update(spec.library_bindings.values())
    for maps in spec.style_maps.values():
        referenced.update(maps)
    for binding in [spec.interpretation, *(p.get('binding') for p in spec.interpretation_panels)]:
        if binding:
            referenced.update(binding.get(key) for key in ('x_id', 'y_id'))
    if spec.stratigraphy:
        referenced.update(spec.stratigraphy['binding'].get(key) for key in ('well_id', 'depth_id', 'bottom_id'))
    # Mapping by name is permitted only for a unique match including units.
    mapping = {}
    for source in payload.get("columns", []):
        if source['id'] not in referenced:
            continue
        matches = [
            c.id
            for c in by_name.get(source['name'], [])
            if column_compatible(c, source)
        ]
        if source["id"] in existing:
            if not column_compatible(existing[source['id']], source):
                raise ValueError(tr('Определение показателя изменилось: {0}. Проверьте тип, единицу и основание.', source['name']))
            mapping[source["id"]] = source["id"]
        elif len(matches) == 1:
            mapping[source["id"]] = matches[0]
    for key in column_fields:
        cid = getattr(spec, key)
        if cid:
            if cid not in mapping:
                raise ValueError(
                    tr('Неоднозначное сопоставление колонок: настройте их вручную в исходном проекте')
                )
            setattr(spec, key, mapping[cid])
    spec.library_bindings = {key: mapping[cid] for key, cid in spec.library_bindings.items() if cid}
    spec.track_ids = [mapping[cid] for cid in spec.track_ids]
    for rule in leaf_rules(spec.filter_rules):
        if rule['column'] not in mapping:
            raise ValueError(tr('Неоднозначное сопоставление колонок: настройте их вручную в исходном проекте'))
        rule['column'] = mapping[rule['column']]
    for item in spec.annotations:
        if item.get('track_id'):
            item['track_id'] = mapping[item['track_id']]
    spec.track_axes = {
        mapping[cid]: axis
        for cid, axis in spec.track_axes.items()
        if cid in mapping
    }
    spec.style_maps = {
        channel: {
            mapping[cid]: entries
            for cid, entries in maps.items()
            if cid in mapping
        }
        for channel, maps in spec.style_maps.items()
    }
    for binding in [spec.interpretation, *(p.get('binding') for p in spec.interpretation_panels)]:
        if binding:
            binding['x_id'] = mapping[binding['x_id']]
            binding['y_id'] = mapping[binding['y_id']]
    if spec.stratigraphy:
        for key in ("well_id", "depth_id", "bottom_id"):
            if spec.stratigraphy["binding"].get(key):
                spec.stratigraphy["binding"][key] = mapping[
                    spec.stratigraphy["binding"][key]
                ]
    return spec
