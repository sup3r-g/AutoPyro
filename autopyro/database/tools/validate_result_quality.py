"""Check migration 008 with rollback by default; --apply commits it."""
import argparse
import os
from pathlib import Path

import psycopg
from psycopg import sql

from autopyro.database.connection import local_parameters


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('PGUSER', 'admin')
    path = Path(__file__).resolve().parents[3] / 'database/migrations/008_result_quality.sql'
    ddl = path.read_text(encoding='utf-8').replace('\nBEGIN;\n', '\n').removesuffix('COMMIT;\n')
    with psycopg.connect(**local_parameters()) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        tables = [r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname='geochem' ORDER BY tablename")]

        def fingerprint():
            return {name: conn.execute(sql.SQL('SELECT count(*),sum(hashtextextended(row_to_json(t)::text,0)::numeric) FROM geochem.{} t')
                    .format(sql.Identifier(name))).fetchone() for name in tables}

        before = fingerprint()
        conn.execute(ddl)
        conn.execute(ddl)
        assert before == fingerprint()
        total = conn.execute('SELECT count(*) FROM geochem.analysis_result').fetchone()[0]
        assert conn.execute('SELECT count(*) FROM geochem.v_geochemical_measurement_qc').fetchone()[0] == total
        assert conn.execute('SELECT sum(result_count) FROM geochem.v_geochemical_quality_summary').fetchone()[0] == total
        assert conn.execute('''SELECT count(*) FROM geochem.v_geochemical_measurement_qc
            WHERE NOT has_quality_metadata AND (qualifier<>'unreported' OR review_status<>'unreviewed'
            OR detection_limit IS NOT NULL OR absolute_uncertainty IS NOT NULL)''').fetchone()[0] == 0
        numeric_key = conn.execute("""SELECT r.analysis_run_id,r.source_field_id FROM geochem.analysis_result r
            LEFT JOIN geochem.analysis_result_quality q USING(analysis_run_id,source_field_id)
            WHERE r.source_dtype='Float64' AND q.analysis_run_id IS NULL LIMIT 1""").fetchone()
        text_key = conn.execute("""SELECT r.analysis_run_id,r.source_field_id FROM geochem.analysis_result r
            LEFT JOIN geochem.analysis_result_quality q USING(analysis_run_id,source_field_id)
            WHERE r.source_dtype='string' AND q.analysis_run_id IS NULL LIMIT 1""").fetchone()
        assert numeric_key and text_key, 'Validation requires unannotated numeric and text results'
        conn.execute('SAVEPOINT fixture')
        insert = '''INSERT INTO geochem.analysis_result_quality
            (analysis_run_id,source_field_id,evidence,qualifier,detection_limit,quantification_limit,
             absolute_uncertainty,uncertainty_kind,coverage_factor,confidence_level,
             review_status,reviewed_by,reviewed_at)
            VALUES (%s,%s,'Validation only',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
        base = ['unreported',None,None,None,None,None,None,'unreviewed',None,None]
        cases = [
            (numeric_key,['less_than',None,None,None,None,None,None,'unreviewed',None,None]),
            (numeric_key,['below_detection',None,None,None,None,None,None,'unreviewed',None,None]),
            (numeric_key,['below_quantification',None,None,None,None,None,None,'unreviewed',None,None]),
            (numeric_key,['measured',-1,None,None,None,None,None,'unreviewed',None,None]),
            (numeric_key,['measured',2,1,None,None,None,None,'unreviewed',None,None]),
            (numeric_key,['measured',None,None,1,None,None,None,'unreviewed',None,None]),
            (numeric_key,['measured',None,None,1,'standard',2,None,'unreviewed',None,None]),
            (numeric_key,['measured',None,None,None,None,None,0.95,'unreviewed',None,None]),
            (numeric_key,['measured',None,None,None,None,None,None,'accepted',None,None]),
            (numeric_key,['measured','NaN',None,None,None,None,None,'unreviewed',None,None]),
            (numeric_key,['measured','Infinity',None,None,None,None,None,'unreviewed',None,None]),
            (text_key,['below_detection',1,None,None,None,None,None,'unreviewed',None,None]),
            ((-1,-1),base),
        ]
        for key, values in cases:
            try:
                with conn.transaction():
                    conn.execute(insert,(*key,*values))
            except psycopg.IntegrityError:
                pass
            else:
                raise AssertionError(f'Invalid metadata accepted: {values}')
        conn.execute(insert,(*numeric_key,'below_detection',1,2,0.1,'expanded',2,0.95,'accepted','validator','2026-01-01T00:00:00Z'))
        row = conn.execute('''SELECT qualifier,detection_limit,review_status,has_quality_metadata
            FROM geochem.v_geochemical_measurement_qc WHERE analysis_run_id=%s AND source_field_id=%s''',numeric_key).fetchone()
        assert row == ('below_detection',1,'accepted',True)
        # Uniqueness prevents one result from becoming several rows in analytic views.
        try:
            with conn.transaction():
                conn.execute(insert,(*numeric_key,*base))
        except psycopg.errors.UniqueViolation:
            pass
        else:
            raise AssertionError('Duplicate metadata accepted')
        conn.execute('ROLLBACK TO SAVEPOINT fixture')
        assert before == fingerprint()
        conn.execute('SET LOCAL ROLE readonly')
        conn.execute('SELECT * FROM geochem.v_geochemical_measurement_qc LIMIT 1').fetchone()
        assert not conn.execute("SELECT has_table_privilege(current_user,'geochem.analysis_result_quality','INSERT')").fetchone()[0]
        conn.execute('RESET ROLE')
        print(f'Validated: {len(tables)} tables unchanged; {total} results preserved; valid annotation and {len(cases)+1} invalid cases; readonly access.')
        if args.apply:
            conn.commit()
            print('Migration committed; validation annotations rolled back.')
        else:
            conn.rollback()
            print('Validation rolled back.')


if __name__ == '__main__':
    main()
