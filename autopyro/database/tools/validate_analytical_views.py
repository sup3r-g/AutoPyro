"""Check migration 007 against live data; --apply commits after validation."""
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
    path = Path(__file__).resolve().parents[3] / 'database/migrations/007_analytical_views.sql'
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
        assert before == fingerprint(), 'Stored data changed'
        total = conn.execute('SELECT count(*) FROM geochem.analysis_result').fetchone()[0]
        assert conn.execute('SELECT count(*) FROM geochem.v_geochemical_measurement').fetchone()[0] == total
        assert conn.execute('SELECT count(*) FROM geochem.v_geochemical_run').fetchone() == conn.execute('SELECT count(*) FROM geochem.analysis_run').fetchone()
        assert conn.execute("SELECT count(*) FROM geochem.v_pyrolysis_report").fetchone() == conn.execute("SELECT count(*) FROM geochem.analysis_run WHERE group_name LIKE 'Pyrol%'").fetchone()
        # Every result retains its original identity and typed value exactly once.
        assert conn.execute('''SELECT count(*) FROM (
            (SELECT analysis_run_id,source_field_id,value_float,value_int,value_text FROM geochem.v_geochemical_measurement
             EXCEPT ALL SELECT analysis_run_id,source_field_id,value_float,value_int,value_text FROM geochem.analysis_result)
            UNION ALL
            (SELECT analysis_run_id,source_field_id,value_float,value_int,value_text FROM geochem.analysis_result
             EXCEPT ALL SELECT analysis_run_id,source_field_id,value_float,value_int,value_text FROM geochem.v_geochemical_measurement)
            ) d''').fetchone()[0] == 0
        assert conn.execute('SELECT sum(result_count) FROM geochem.v_geochemical_availability').fetchone()[0] == total
        assert conn.execute('''SELECT count(*) FROM geochem.v_geochemical_run v
            WHERE v.result_count<>(SELECT count(*) FROM jsonb_object_keys(v.values_by_field))''').fetchone()[0] == 0
        assert conn.execute('''SELECT count(*) FROM geochem.v_geochemical_run v
            JOIN geochem.analysis_result r USING(analysis_run_id)
            WHERE v.values_by_field->r.source_field_id::text IS DISTINCT FROM
                coalesce(to_jsonb(r.value_float),to_jsonb(r.value_int),to_jsonb(r.value_text))''').fetchone()[0] == 0
        labels = [('toc_wt_pct','TOC [%wt]'),('s1_mg_g','S1 [ppt (mg/g)]'),
                  ('s2_mg_g','S2 [ppt (mg/g)]'),('s3_mg_g','S3 [ppt (mg/g)]'),
                  ('tmax_c','T-max [°C]'),('hi_mg_g','HI [ppt (mg/g) (a/b)]'),('oi_mg_g','OI [ppt (mg/g) (a/b)]')]
        for column, label in labels:
            query = sql.SQL('''SELECT count(*) FROM geochem.v_pyrolysis_report v
                LEFT JOIN geochem.source_field f ON f.group_name=v.group_name AND f.source_label=%s
                LEFT JOIN geochem.analysis_result r ON r.analysis_run_id=v.analysis_run_id AND r.source_field_id=f.source_field_id
                WHERE v.{}::numeric IS DISTINCT FROM coalesce(r.value_float::numeric,r.value_int::numeric)''').format(sql.Identifier(column))
            assert conn.execute(query,(label,)).fetchone()[0] == 0, column
        conn.execute('SET LOCAL ROLE readonly')
        for view in ['v_geochemical_measurement','v_geochemical_run','v_pyrolysis_report','v_geochemical_availability']:
            conn.execute(sql.SQL('SELECT * FROM geochem.{} LIMIT 1').format(sql.Identifier(view))).fetchone()
            assert not conn.execute('SELECT has_table_privilege(current_user,%s,\'INSERT\')',('geochem.'+view,)).fetchone()[0]
        conn.execute('RESET ROLE')
        print(f'Validated: {len(tables)} tables unchanged; {total} results preserved; run JSON, seven pyrolysis columns, availability and readonly access checked.')
        if args.apply:
            conn.commit()
            print('Migration committed.')
        else:
            conn.rollback()
            print('Validation rolled back.')


if __name__ == '__main__':
    main()
