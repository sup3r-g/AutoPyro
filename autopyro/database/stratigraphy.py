"""Read versioned lithostratigraphic interpretations from PostgreSQL."""

import json
from contextlib import closing
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5

from autopyro.geology.stratigraphy import from_records, scheme_hash
from .connection import connect


def interpretation_catalog(parameters):
    with closing(connect(parameters)) as connection:
        return connection.execute("""
            SELECT v.interpretation_id, b.name AS wellbore_name, v.version_tag, v.status,
                   v.created_at::text AS created_at
            FROM geochem.lithostrat_interpretation v
            JOIN geochem.wellbore b USING (wellbore_id)
            WHERE EXISTS (
                SELECT 1 FROM geochem.lithostrat_interpretation_interval i
                WHERE i.interpretation_id = v.interpretation_id
            )
            ORDER BY b.name, v.created_at DESC, v.interpretation_id DESC
        """).fetchall()


def read_interpretations(parameters, ids, depth_reference, *, limit=100000):
    ids = sorted(set(ids))
    if not ids:
        raise ValueError('Выберите редакции для загрузки.')
    with closing(connect(parameters)) as connection:
        rows = connection.execute("""
            SELECT interpretation_id, interpretation_interval_id, interval_number,
                   wellbore_id, wellbore_name, version_tag, status, unit_name, unit_level,
                   top_depth_m, bottom_depth_m, depth_type, depth_datum, depth_reference,
                   source_interval_id, interval_note
            FROM geochem.v_lithostrat_interpretation
            WHERE interpretation_id = ANY(%s)
            ORDER BY wellbore_name, interpretation_id, interval_number
            LIMIT %s
        """, (ids, limit + 1)).fetchall()
        endpoint = {'host': connection.info.host, 'port': connection.info.port,
                    'database': connection.info.dbname}
    if len(rows) > limit:
        raise ValueError('Слишком много интервалов. Выберите меньше стволов.')
    if {row['interpretation_id'] for row in rows} != set(ids):
        raise ValueError('Некоторые редакции больше недоступны или не содержат интервалов. Обновите список.')
    return scheme_from_interpretations(rows, endpoint, depth_reference)


def scheme_from_interpretations(rows, endpoint, depth_reference):
    reference = depth_reference.strip()
    if not reference:
        raise ValueError('Укажите тип глубины и начало отсчёта интервалов.')
    if not rows:
        raise ValueError('Выберите редакции для загрузки.')
    revisions = {row['interpretation_id']: {key: row[key] for key in (
        'interpretation_id', 'wellbore_id', 'wellbore_name', 'version_tag', 'status')} for row in rows}
    if len({row['wellbore_name'] for row in revisions.values()}) != len(revisions):
        raise ValueError('Выберите только одну редакцию для каждого имени ствола.')

    headers = [key for key in rows[0] if key != 'depth_reference']
    records = []
    depth_types, datums = set(), set()
    pieces = reference.casefold().split(maxsplit=1)
    for row in rows:
        declared = (row['depth_reference'] or '').strip()
        depth_type = (row['depth_type'] or '').strip()
        datum = (row['depth_datum'] or '').strip()
        if depth_type:
            depth_types.add(depth_type.casefold())
        if datum:
            datums.add(datum.casefold())
        unspecified = not declared or declared.casefold() == 'not specified in source workbook'
        if unspecified:
            compatible = (not depth_type or pieces[0] == depth_type.casefold()) and (
                not datum or (len(pieces) == 2 and pieces[1] == datum.casefold()))
        else:
            compatible = declared == reference and (
                not depth_type or pieces[0] not in ('md', 'tvd', 'tvdss') or pieces[0] == depth_type.casefold())
        if not compatible or len(depth_types) > 1 or len(datums) > 1:
            raise ValueError('Система глубин выбранных интервалов БД не совпадает с объявленной.')
        records.append([str(row[key]) if isinstance(row[key], Decimal) else row[key] for key in headers])
    source = {'type': 'postgresql_lithostratigraphy', 'read_only': True, 'endpoint': endpoint,
              'view': 'geochem.v_lithostrat_interpretation', 'interpretations': list(revisions.values())}
    # The import parser checks a declared reference per row. The database values,
    # including unspecified legacy metadata, belong in the untouched raw cells.
    scheme = from_records(headers, records, [row['interval_number'] for row in rows],
                          name='PostgreSQL · Lithostratigraphy', depth_reference=reference, source=source)
    namespace = uuid5(NAMESPACE_URL, json.dumps(endpoint, sort_keys=True))
    for item, row in zip(scheme['intervals'], rows, strict=True):
        item['raw_cells']['depth_reference'] = row['depth_reference']
        item['id'] = str(uuid5(namespace, f"interval/{row['interpretation_interval_id']}"))
    wells = sorted(row['wellbore_id'] for row in revisions.values())
    scheme['id'] = str(uuid5(namespace, 'scheme/' + ','.join(map(str, wells))))
    tags = {row['version_tag'] for row in revisions.values()}
    tag = next(iter(tags)) if len(tags) == 1 else 'PostgreSQL'
    scheme['revision'] = f'{tag} · {scheme_hash(scheme)[:12]}'
    return scheme
