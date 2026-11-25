"""Compare optimized views with their old definitions exactly; always rollback."""
from pathlib import Path
import os

import psycopg
from psycopg import sql

from autopyro.database.connection import local_parameters


def main():
    os.environ.setdefault('PGUSER','admin')
    root = Path(__file__).resolve().parents[3] / 'database/migrations'
    with psycopg.connect(**local_parameters()) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        conn.execute("SET LOCAL statement_timeout='120s'")
        semantic = (root/'005_analysis_catalogs.sql').read_text(encoding='utf-8')
        semantic = 'CREATE OR REPLACE VIEW geochem.v_analysis_result_semantic AS'+semantic.split(
            'CREATE OR REPLACE VIEW geochem.v_analysis_result_semantic AS',1)[1].split('\nDO $$',1)[0]
        conn.execute(semantic)
        # Restore the original definitions inside this transaction only, making
        # this validator useful both before and after migration 011 is installed.
        for filename in ['007_analytical_views.sql','010_sample_lithostrat_matching.sql']:
            ddl = (root/filename).read_text(encoding='utf-8').replace('\nBEGIN;\n','\n').removesuffix('COMMIT;\n')
            conn.execute(ddl)
        targets = ['v_analysis_result_semantic','v_geochemical_measurement','v_pyrolysis_report','v_geochemical_availability','v_sample_lithostrat_status']
        for view in targets:
            conn.execute(sql.SQL('CREATE TEMP TABLE {} ON COMMIT DROP AS SELECT * FROM geochem.{}')
                         .format(sql.Identifier('before_'+view),sql.Identifier(view)))
            print('Captured baseline: '+view,flush=True)
        ddl = (root/'011_analytical_performance.sql').read_text(encoding='utf-8').replace('\nBEGIN;\n','\n').removesuffix('COMMIT;\n')
        conn.execute(ddl)
        conn.execute(ddl)
        semantic_optimized = (root/'012_semantic_join_performance.sql').read_text(encoding='utf-8')
        conn.execute(semantic_optimized.replace('\nBEGIN;\n','\n').removesuffix('COMMIT;\n'))
        selective = (root/'013_measurement_filter_performance.sql').read_text(encoding='utf-8')
        conn.execute(selective.replace('\nBEGIN;\n','\n').removesuffix('COMMIT;\n'))
        for view in targets:
            difference = conn.execute(sql.SQL('''SELECT count(*) FROM (
                (SELECT * FROM {} EXCEPT ALL SELECT * FROM geochem.{}) UNION ALL
                (SELECT * FROM geochem.{} EXCEPT ALL SELECT * FROM {})
                ) differences''').format(sql.Identifier('before_'+view),sql.Identifier(view),
                                         sql.Identifier(view),sql.Identifier('before_'+view))).fetchone()[0]
            assert difference==0,(view,difference)
            print('Exact multiset equality: '+view,flush=True)
            plan = conn.execute(sql.SQL('EXPLAIN (ANALYZE,BUFFERS,FORMAT JSON) SELECT * FROM geochem.{}')
                                .format(sql.Identifier(view))).fetchone()[0][0]
            print(f'Optimized execution: {view}: {plan["Execution Time"]} ms',flush=True)
        well = conn.execute("""SELECT s.well_id FROM geochem.sample s
            WHERE s.well_id IS NOT NULL AND s.top_md>=1000 AND s.top_md<3000
            AND EXISTS (SELECT FROM geochem.analysis_run a WHERE a.sample_id=s.sample_id AND a.group_name LIKE 'Pyrol%')
            GROUP BY s.well_id ORDER BY count(*) DESC,s.well_id LIMIT 1""").fetchone()[0]
        plan = conn.execute('''EXPLAIN (ANALYZE,BUFFERS,FORMAT JSON)
            SELECT * FROM geochem.v_geochemical_measurement WHERE well_id=%s AND top_md>=1000 AND top_md<3000''',(well,)).fetchone()[0][0]
        print(f'Filtered well/depth: {plan["Execution Time"]} ms, {plan["Plan"]["Actual Rows"]} rows',flush=True)
        conn.execute('SET LOCAL ROLE readonly')
        for view in targets:
            conn.execute(sql.SQL('SELECT * FROM geochem.{} LIMIT 1').format(sql.Identifier(view))).fetchone()
        conn.rollback()
        print('Validated column compatibility, NULLs, duplicates, all rows, idempotency and readonly access. Changes rolled back.')


if __name__ == '__main__':
    main()
