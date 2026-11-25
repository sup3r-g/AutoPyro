"""Import the five-column lithostratigraphic Excel into separate geochem tables.

Usage: python -m autopyro.database.tools.import_lithostratigraphy path/to/strat_litho_wellbore.xlsx
PG* environment variables override the local Compose connection defaults.
The input workbook is opened read-only and never modified.
"""
import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import psycopg
import pandas as pd
from psycopg.types.json import Jsonb

from autopyro.data_io.io import read_excel_frame

ROOT = Path(__file__).resolve().parents[3] / "database"
HEADERS = ('Wellbore name','Top depth [m]','Bottom depth [m]','Lithostrat. unit','Level')
LEVELS = {'GROUP','SUBGROUP','FORMATION','MEMBER'}


def read_rows(path):
    rows = []
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    with pd.ExcelFile(path, engine="openpyxl", engine_kwargs={"data_only": False}) as workbook:
        for sheet_name in workbook.sheet_names:
            sheet = workbook.book[sheet_name]
            width = sheet.max_column
            frame = read_excel_frame(workbook, sheet_name)
            width = width if width is not None else len(frame.columns)
            frame = frame.reindex(columns=range(width))
            header = tuple(frame.iloc[0]) if not frame.empty else (None,) * width
            if header != HEADERS:
                raise ValueError(f'Unexpected headers in {sheet_name}: {header}')
            frame.columns = HEADERS
            data = frame.iloc[1:].dropna(how="all")
            # Retain physical Excel row numbers after dropping empty records.
            data.index = data.index + 1
            if data.empty:
                continue
            top, bottom, level = data[HEADERS[1]], data[HEADERS[2]], data[HEADERS[4]]
            valid_text = data[[HEADERS[0], HEADERS[3], HEADERS[4]]].map(
                lambda value: isinstance(value, str) and bool(value.strip()) and '\x00' not in value
            ).all(axis=1)
            valid_depth = data[[HEADERS[1], HEADERS[2]]].map(
                lambda value: type(value) in (int, float) and math.isfinite(value)
            ).all(axis=1)
            numeric_top = pd.to_numeric(top, errors="coerce")
            numeric_bottom = pd.to_numeric(bottom, errors="coerce")
            errors = pd.Series(None, index=data.index, dtype=object)
            errors.loc[(numeric_top < 0) | (numeric_bottom <= numeric_top)] = 'Nonpositive/reversed depth interval'
            errors.loc[~valid_depth] = 'Invalid depth'
            errors.loc[~level.isin(LEVELS)] = 'Unsupported level'
            errors.loc[~valid_text] = 'Invalid name/unit/level'
            # Inspect cell types only for formula/error candidates; literal strings
            # such as "=well" remain valid when Excel stores them as text.
            candidates = data.map(lambda value: isinstance(value, str) and value.startswith(('=', '#')))
            for r, c in zip(*candidates.to_numpy().nonzero()):
                source_row = int(data.index[r])
                if sheet.cell(source_row, int(c) + 1).data_type in ('f', 'e'):
                    errors.loc[source_row] = 'Formula/error'
            invalid = errors.dropna()
            if not invalid.empty:
                row = invalid.index[0]
                reason = invalid.iloc[0]
                if reason == 'Formula/error':
                    raise ValueError(f'Formula/error at {sheet_name}!{row}')
                values = tuple(data.loc[row])
                detail = values[4] if reason == 'Unsupported level' else values
                raise ValueError(f'{reason}: {detail}')
            rows.extend(
                (sheet_name, row, name, Decimal(str(top)), Decimal(str(bottom)), unit, level,
                 dict(zip(HEADERS, (name, top, bottom, unit, level))))
                for row, name, top, bottom, unit, level in data.itertuples(name=None)
            )
    if not rows:
        raise ValueError('No source rows')
    return digest,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file',type=Path)
    args=parser.parse_args()
    digest,rows=read_rows(args.file)
    print(f'Validated {len(rows)} intervals, {len({r[2] for r in rows})} wellbores.',flush=True)
    connection=dict(host=os.getenv('PGHOST','127.0.0.1'),port=os.getenv('PGPORT','15432'),
                    dbname=os.getenv('PGDATABASE','geochemistry'),user=os.getenv('PGUSER','admin'),
                    password=os.getenv('PGPASSWORD','autopyro_local'),connect_timeout=15,
                    application_name='autopyro_lithostrat_import')
    with psycopg.connect(**connection) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended('geochem:lithostrat-import',0))")
        existing=conn.execute('SELECT import_batch_id FROM geochem.import_batch WHERE source_sha256=%s',(digest,)).fetchall()
        if existing:
            print(f'Already imported: batch IDs {[r[0] for r in existing]}; no changes.')
            return
        conn.execute((ROOT/'migrations/003_wellbore_lithostratigraphy.sql').read_text(encoding='utf-8'))
        wells=defaultdict(list)
        for well_id,name in conn.execute('SELECT well_id,well_name FROM geochem.well'):
            wells[name].append(well_id)
        source_names=sorted({r[2] for r in rows})
        ambiguous=[name for name in source_names if len(wells[name])>1]
        if ambiguous:
            raise ValueError(f'Ambiguous exact well matches: {ambiguous}')
        batch_id=conn.execute('''INSERT INTO geochem.import_batch(source_name,source_sha256,comment)
            VALUES (%s,%s,%s) RETURNING import_batch_id''',
            (args.file.name,digest,'Lithostratigraphic Excel; exact wellbore-name matching; depths in metres, datum unspecified; source overlaps retained.')).fetchone()[0]
        wellbore_ids={}
        for name in source_names:
            well_id=wells[name][0] if wells[name] else None
            current=conn.execute('SELECT lithostrat_wellbore_id,well_id FROM geochem.lithostrat_wellbore WHERE wellbore_name=%s',(name,)).fetchone()
            if current:
                if current[1] != well_id:
                    raise ValueError(f'Existing wellbore mapping differs: {name}; review explicitly')
                wellbore_ids[name]=current[0]
            else:
                wellbore_ids[name]=conn.execute('''INSERT INTO geochem.lithostrat_wellbore(wellbore_name,well_id)
                    VALUES (%s,%s) RETURNING lithostrat_wellbore_id''',(name,well_id)).fetchone()[0]
        unit_ids={}
        for unit,level in sorted({(r[5],r[6]) for r in rows}):
            conn.execute('INSERT INTO geochem.lithostrat_unit(unit_name,unit_level) VALUES (%s,%s) ON CONFLICT DO NOTHING',(unit,level))
            unit_ids[unit,level]=conn.execute('SELECT lithostrat_unit_id FROM geochem.lithostrat_unit WHERE unit_name=%s AND unit_level=%s',(unit,level)).fetchone()[0]
        with conn.cursor().copy('''COPY geochem.wellbore_lithostrat_interval
            (import_batch_id,source_sheet,source_row,lithostrat_wellbore_id,lithostrat_unit_id,top_depth_m,bottom_depth_m,raw_cells)
            FROM STDIN''') as copy:
            for sheet,row,name,top,bottom,unit,level,raw in rows:
                copy.write_row((batch_id,sheet,row,wellbore_ids[name],unit_ids[unit,level],top,bottom,Jsonb(raw)))
        loaded=conn.execute('''SELECT source_sheet,source_row,wellbore_name,top_depth_m,bottom_depth_m,
            unit_name,unit_level,raw_cells,thickness_m,well_id,area
            FROM geochem.v_wellbore_lithostratigraphy WHERE import_batch_id=%s ORDER BY source_sheet,source_row''',(batch_id,)).fetchall()
        expected=sorted(rows,key=lambda r:(r[0],r[1]))
        if len(loaded)!=len(expected):
            raise ValueError('Row count mismatch')
        for source,target in zip(expected,loaded):
            if tuple(source)!=tuple(target[:8]) or target[8]!=source[4]-source[3]:
                raise ValueError(f'Source round-trip mismatch at {source[0]}!{source[1]}')
            expected_well=wells[source[2]][0] if wells[source[2]] else None
            if target[9]!=expected_well:
                raise ValueError(f'Well association mismatch: {source[2]}')
        with args.file.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:
                raise ValueError('Source workbook changed during import')
        overlaps=conn.execute('''SELECT wellbore_name,unit_level,source_row_a,unit_a,top_a,bottom_a,
            source_row_b,unit_b,top_b,bottom_b FROM geochem.v_lithostrat_interval_overlap
            WHERE import_batch_id=%s''',(batch_id,)).fetchall()
        report=dict(source_file=args.file.name,sha256=digest,batch_id=batch_id,
                    intervals=len(rows),wellbores=len(source_names),units=len(unit_ids),
                    intervals_by_level=dict(Counter(r[6] for r in rows)),
                    matched_wellbores=sum(bool(wells[n]) for n in source_names),
                    unmatched_wellbores=[n for n in source_names if not wells[n]],
                    same_level_overlaps=[list(r) for r in overlaps],value_mismatches=0,
                    area_source='geochem.well.area for exact matched wells; NULL otherwise',
                    depth_reference='not specified in source workbook')
        if conn.execute("SELECT to_regprocedure('geochem.refresh_wellbore_catalog()')").fetchone()[0]:
            conn.execute('SELECT geochem.refresh_wellbore_catalog()')
        if conn.execute("SELECT to_regprocedure('geochem.snapshot_lithostrat_sources()')").fetchone()[0]:
            conn.execute('SELECT geochem.snapshot_lithostrat_sources()')
        conn.commit()
        print(f'COMMITTED batch {batch_id}: {len(rows)} intervals; verified all five source columns and raw cells.',flush=True)
    with psycopg.connect(**connection,autocommit=True) as conn:
        conn.execute('ANALYZE geochem.lithostrat_wellbore,geochem.lithostrat_unit,geochem.wellbore_lithostrat_interval')
    output=ROOT/'import_reports'/'strat_litho_wellbore.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,default=str),flush=True)


if __name__=='__main__':
    main()
