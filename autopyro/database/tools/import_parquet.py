"""Lossless, transactional import of the supplied two-level-column Parquet.

python -m autopyro.database.tools.import_parquet "Geochemical database Barents Sea.parquet"
Use --inspect for a full file preflight without connecting to PostgreSQL.
PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD override local Compose defaults.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[3] / "database"
EXTRA = dict(group_name='Metis', source_label='Sam Status', source_dtype='string',
             target_table='sample', target_column='metis_sam_status', scope='core', unit_label='')


def log(message):
    print(message, flush=True)


def prepare(path):
    with (ROOT / 'field_catalog.csv').open(encoding='utf-8-sig', newline='') as stream:
        fields = list(csv.DictReader(stream))
    fields.append(EXTRA)
    catalog = {(f['group_name'], f['source_label']): f for f in fields}
    parquet = pq.ParquetFile(path)
    definitions = []
    seen = set()
    for field in parquet.schema_arrow:
        pair = ast.literal_eval(field.name)
        if not isinstance(pair, tuple) or len(pair) != 2 or pair not in catalog or pair in seen:
            raise ValueError(f'Unknown or duplicate column: {field.name}')
        seen.add(pair)
        spec = catalog[pair]
        expected = {'string': pa.string(), 'Int64': pa.int64(), 'Float64': pa.float64()}[spec['source_dtype']]
        if field.type != expected:
            raise ValueError(f'Dtype mismatch: {pair}: {field.type} != {expected}')
        definitions.append(spec)
    missing = set(catalog) - seen - {('Metis', 'Sam Status')}
    if missing:
        raise ValueError(f'Missing expected columns: {missing}')
    return parquet, definitions


def preflight(parquet, definitions):
    counts = Counter()
    invalid = Counter()
    for batch in parquet.iter_batches(batch_size=512, use_threads=False):
        for col, spec in zip(batch.columns, definitions):
            if col.null_count == len(col):
                continue
            key = spec['group_name']
            counts[key] += len(col) - col.null_count
            if spec['source_dtype'] == 'Float64':
                n = pc.sum(pc.fill_null(pc.invert(pc.is_finite(col)), False)).as_py() or 0
                if n:
                    invalid[f'{key} / {spec["source_label"]}: nonfinite'] += n
            elif spec['source_dtype'] == 'string':
                n = pc.sum(pc.fill_null(pc.match_substring(col, '\x00'), False)).as_py() or 0
                if n:
                    invalid[f'{key} / {spec["source_label"]}: NUL'] += n
    if invalid:
        raise ValueError(f'Values incompatible with PostgreSQL/schema; nothing imported: {dict(invalid)}')
    return counts


def insert_core(conn, table, id_expression, filter_expression, definitions):
    columns = [d['target_column'] for d in definitions if d['target_table'] == table]
    id_name = table + '_id'
    prefix = []
    if table == 'sample':
        prefix = ['well_id']
    elif table == 'sample_geology':
        prefix = ['sample_id']
    identifiers = [id_name] + prefix + columns
    selections = [sql.SQL(id_expression)] + [sql.SQL('s.' + k) for k in prefix]
    selections += [sql.SQL('p.') + sql.Identifier(c) for c in columns]
    command = sql.SQL('''INSERT INTO geochem.{table} ({columns}) OVERRIDING SYSTEM VALUE
        SELECT {selections} FROM import_stage s
        CROSS JOIN LATERAL jsonb_populate_record(NULL::geochem.{table}, s.core -> {key}) p
        WHERE {condition}''').format(
            table=sql.Identifier(table), columns=sql.SQL(',').join(map(sql.Identifier, identifiers)),
            selections=sql.SQL(',').join(selections), key=sql.Literal(table), condition=sql.SQL(filter_expression))
    conn.execute(command)


def load(path, parquet, definitions, counts, digest):
    started = time.monotonic()
    expected_results = sum(n for group, n in counts.items()
                           if group not in {'Well', 'Sample', 'Geol', 'Ref', 'Metis'})
    connection = dict(host=os.getenv('PGHOST', '127.0.0.1'), port=os.getenv('PGPORT', '15432'),
                      dbname=os.getenv('PGDATABASE', 'geochemistry'), user=os.getenv('PGUSER', 'admin'),
                      password=os.getenv('PGPASSWORD', 'autopyro_local'), connect_timeout=15,
                      application_name='autopyro_parquet_import')
    with psycopg.connect(**connection) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended('geochem:parquet-import',0))")
        existing = conn.execute('SELECT import_batch_id FROM geochem.import_batch WHERE source_sha256 = %s', (digest,)).fetchall()
        if existing:
            log(f'File already imported: SHA-256 {digest}; batch IDs {[r[0] for r in existing]}. No changes.')
            return
        if any(d['group_name'] == 'Metis' for d in definitions):
            conn.execute((ROOT / 'migrations/002_metis_sample_status.sql').read_text(encoding='utf-8'))
        db_catalog = {(r[0], r[1]): r[2:] for r in conn.execute(
            'SELECT group_name,source_label,source_dtype,target_table,target_column FROM geochem.source_field')}
        for d in definitions:
            expected = (d['source_dtype'], d['target_table'], d['target_column'] or None)
            if db_catalog.get((d['group_name'], d['source_label'])) != expected:
                raise ValueError(f'Database catalog differs: {d}')
        batch_id = conn.execute('''INSERT INTO geochem.import_batch(source_name,source_sha256,comment)
            VALUES (%s,%s,%s) RETURNING import_batch_id''',
            (path.name, digest, 'One sample per source row; wells and references deduplicated by all attributes; raw non-null cells retained.')).fetchone()[0]
        conn.execute('''CREATE TEMP TABLE import_stage (
            row_number bigint PRIMARY KEY, well_key integer, ref_key integer,
            core jsonb NOT NULL, raw_cells jsonb NOT NULL,
            sample_id bigint DEFAULT nextval(pg_get_serial_sequence('geochem.sample','sample_id')),
            source_record_id bigint DEFAULT nextval(pg_get_serial_sequence('geochem.source_record','source_record_id')),
            well_id bigint, reference_document_id bigint, sample_geology_id bigint
        ) ON COMMIT DROP''')
        wells, refs = {}, {}
        row_number = 0
        with conn.cursor().copy('COPY import_stage(row_number,well_key,ref_key,core,raw_cells) FROM STDIN') as copy:
            for batch in parquet.iter_batches(batch_size=256, use_threads=False):
                raw = [[] for _ in range(batch.num_rows)]
                core = [{} for _ in range(batch.num_rows)]
                for col, d in zip(batch.columns, definitions):
                    if col.null_count == len(col):
                        continue
                    for i, value in enumerate(col.to_pylist()):
                        if value is None:
                            continue
                        if isinstance(value, float) and not math.isfinite(value):
                            raise ValueError('File changed after preflight: nonfinite value')
                        raw[i].append([d['group_name'], d['source_label'], value])
                        if d['scope'] == 'core':
                            core[i].setdefault(d['target_table'], {})[d['target_column']] = value
                for i in range(batch.num_rows):
                    well = tuple(sorted(core[i].get('well', {}).items()))
                    ref = tuple(sorted(core[i].get('reference_document', {}).items()))
                    well_key = wells.setdefault(well, len(wells) + 1) if well else None
                    ref_key = refs.setdefault(ref, len(refs) + 1) if ref else None
                    copy.write_row((row_number, well_key, ref_key, Jsonb(core[i]), Jsonb(raw[i])))
                    row_number += 1
                if row_number % 4096 == 0 or row_number == parquet.metadata.num_rows:
                    log(f'Staged source rows: {row_number}/{parquet.metadata.num_rows}')
        if row_number != parquet.metadata.num_rows:
            raise ValueError('Source row count changed')
        log('Creating wells, samples, geology and source records...')
        conn.execute('''CREATE TEMP TABLE well_map ON COMMIT DROP AS
            SELECT well_key, min(row_number) AS first_row,
                nextval(pg_get_serial_sequence('geochem.well','well_id')) AS well_id
            FROM import_stage WHERE well_key IS NOT NULL GROUP BY well_key''')
        conn.execute('UPDATE import_stage s SET well_id = w.well_id FROM well_map w WHERE s.well_key=w.well_key')
        insert_core(conn, 'well', 's.well_id', 's.row_number IN (SELECT first_row FROM well_map)', definitions)
        insert_core(conn, 'sample', 's.sample_id', 'true', definitions)
        conn.execute('''UPDATE import_stage SET sample_geology_id =
            nextval(pg_get_serial_sequence('geochem.sample_geology','sample_geology_id'))
            WHERE core ? 'sample_geology' ''')
        insert_core(conn, 'sample_geology', 's.sample_geology_id', 's.sample_geology_id IS NOT NULL', definitions)
        conn.execute('''CREATE TEMP TABLE ref_map ON COMMIT DROP AS
            SELECT ref_key, min(row_number) AS first_row,
                nextval(pg_get_serial_sequence('geochem.reference_document','reference_document_id')) AS reference_document_id
            FROM import_stage WHERE ref_key IS NOT NULL GROUP BY ref_key''')
        conn.execute('''UPDATE import_stage s SET reference_document_id=r.reference_document_id
            FROM ref_map r WHERE s.ref_key=r.ref_key''')
        insert_core(conn, 'reference_document', 's.reference_document_id', 's.row_number IN (SELECT first_row FROM ref_map)', definitions)
        conn.execute('''INSERT INTO geochem.source_record
            (source_record_id,import_batch_id,row_number,sample_id,sample_geology_id,raw_cells)
            OVERRIDING SYSTEM VALUE
            SELECT source_record_id,%s,row_number,sample_id,sample_geology_id,raw_cells FROM import_stage''', (batch_id,))
        conn.execute('''INSERT INTO geochem.source_record_reference
            SELECT source_record_id,reference_document_id FROM import_stage WHERE reference_document_id IS NOT NULL''')
        conn.execute('ANALYZE import_stage')
        log('Expanding analytical cells and creating analysis runs...')
        conn.execute('''CREATE TEMP TABLE result_stage ON COMMIT DROP AS
            SELECT s.source_record_id,s.sample_id,f.group_name,f.source_field_id,f.source_dtype,c.value -> 2 AS value
            FROM import_stage s CROSS JOIN LATERAL jsonb_array_elements(s.raw_cells) c(value)
            JOIN geochem.source_field f ON f.group_name=c.value ->> 0 AND f.source_label=c.value ->> 1
            WHERE f.target_table='analysis_result' ''')
        actual_results = conn.execute('SELECT count(*) FROM result_stage').fetchone()[0]
        if actual_results != expected_results:
            raise ValueError(f'Result count mismatch: {actual_results} != {expected_results}')
        conn.execute('''CREATE TEMP TABLE run_stage ON COMMIT DROP AS
            SELECT source_record_id,sample_id,group_name,
                nextval(pg_get_serial_sequence('geochem.analysis_run','analysis_run_id')) AS analysis_run_id
            FROM result_stage GROUP BY source_record_id,sample_id,group_name''')
        conn.execute('''INSERT INTO geochem.analysis_run(analysis_run_id,sample_id,source_record_id,group_name)
            OVERRIDING SYSTEM VALUE SELECT analysis_run_id,sample_id,source_record_id,group_name FROM run_stage''')
        conn.execute('ANALYZE result_stage')
        conn.execute('ANALYZE run_stage')
        conn.execute('''INSERT INTO geochem.analysis_result
            (analysis_run_id,source_field_id,group_name,source_dtype,value_float,value_int,value_text)
            SELECT a.analysis_run_id,r.source_field_id,r.group_name,r.source_dtype,
                CASE WHEN r.source_dtype='Float64' THEN (r.value #>> '{}')::double precision END,
                CASE WHEN r.source_dtype='Int64' THEN (r.value #>> '{}')::bigint END,
                CASE WHEN r.source_dtype='string' THEN r.value #>> '{}' END
            FROM result_stage r JOIN run_stage a USING (source_record_id,sample_id,group_name)''')
        log('Verifying every imported value against staged source cells...')
        mismatch = conn.execute('''SELECT count(*) FROM result_stage e
            JOIN run_stage a USING(source_record_id,sample_id,group_name)
            LEFT JOIN geochem.analysis_result r USING(analysis_run_id,source_field_id,group_name,source_dtype)
            WHERE r.analysis_run_id IS NULL OR CASE e.source_dtype
                WHEN 'Float64' THEN r.value_float IS DISTINCT FROM (e.value #>> '{}')::double precision
                WHEN 'Int64' THEN r.value_int IS DISTINCT FROM (e.value #>> '{}')::bigint
                ELSE r.value_text IS DISTINCT FROM (e.value #>> '{}') END''').fetchone()[0]
        if mismatch:
            raise ValueError(f'{mismatch} analytical values differ')
        for table in ['well', 'sample', 'sample_geology', 'reference_document']:
            bad = conn.execute(sql.SQL('''SELECT count(*) FROM import_stage s
                LEFT JOIN geochem.{table} t ON t.{id}=s.{id}
                WHERE s.{id} IS NOT NULL AND (t.{id} IS NULL OR EXISTS (
                    SELECT 1 FROM jsonb_each(s.core -> {key}) e
                    WHERE (to_jsonb(t) -> e.key) IS DISTINCT FROM e.value))''').format(
                        table=sql.Identifier(table), id=sql.Identifier(table + '_id'), key=sql.Literal(table))).fetchone()[0]
            if bad:
                raise ValueError(f'{bad} {table} records differ from source')
        raw_bad = conn.execute('''SELECT count(*) FROM import_stage s LEFT JOIN geochem.source_record t
            USING(source_record_id,sample_id,row_number) WHERE t.source_record_id IS NULL
            OR t.raw_cells IS DISTINCT FROM s.raw_cells OR t.import_batch_id <> %s''', (batch_id,)).fetchone()[0]
        if raw_bad:
            raise ValueError(f'{raw_bad} raw source snapshots differ')
        with path.open('rb') as stream:
            file_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if file_hash != digest:
            raise ValueError('Source file changed during import; transaction rolled back')
        report = dict(source_file=path.name, sha256=digest, batch_id=batch_id,
                      imported_at=datetime.now(timezone.utc).isoformat(), source_rows=row_number,
                      source_columns=len(definitions), wells=len(wells), samples=row_number,
                      geology_records=conn.execute('SELECT count(*) FROM import_stage WHERE sample_geology_id IS NOT NULL').fetchone()[0],
                      reference_documents=len(refs),
                      source_reference_links=conn.execute('SELECT count(*) FROM import_stage WHERE ref_key IS NOT NULL').fetchone()[0],
                      analysis_runs=conn.execute('SELECT count(*) FROM run_stage').fetchone()[0],
                      analysis_results=actual_results, non_null_source_cells=sum(counts.values()),
                      non_null_by_group=dict(counts), analytical_value_mismatches=mismatch,
                      core_value_mismatches=0, raw_snapshot_mismatches=raw_bad,
                      elapsed_seconds=round(time.monotonic()-started, 2))
        if conn.execute("SELECT to_regprocedure('geochem.refresh_analysis_catalogs()')").fetchone()[0]:
            conn.execute('SELECT geochem.refresh_analysis_catalogs()')
        if conn.execute("SELECT to_regprocedure('geochem.refresh_wellbore_catalog()')").fetchone()[0]:
            conn.execute('SELECT geochem.refresh_wellbore_catalog()')
        conn.commit()
        log(f'COMMITTED batch {batch_id}: {row_number} rows, {actual_results} analytical results.')
    # Refresh planner statistics only after the atomic import has committed.
    with psycopg.connect(**connection, autocommit=True) as conn:
        conn.execute('ANALYZE geochem.well, geochem.sample, geochem.sample_geology, geochem.source_record, geochem.reference_document, geochem.source_record_reference, geochem.analysis_run, geochem.analysis_result')
    output = ROOT / 'import_reports' / (path.stem + '.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    with args.file.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    parquet, definitions = prepare(args.file)
    log(f'Preflight: {parquet.metadata.num_rows} rows, {len(definitions)} columns; SHA-256 {digest}')
    counts = preflight(parquet, definitions)
    log(f'Validated {sum(counts.values())} non-null source cells; no nonfinite numbers or NUL text.')
    if args.inspect:
        log(json.dumps(dict(counts), ensure_ascii=False))
    else:
        load(args.file, parquet, definitions, counts, digest)


if __name__ == '__main__':
    main()
