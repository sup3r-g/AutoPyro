"""Validate migration 005 transactionally; --apply commits after validation."""
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
    migration = Path(__file__).resolve().parents[3] / 'database/migrations/005_analysis_catalogs.sql'
    ddl = migration.read_text(encoding='utf-8').replace('\nBEGIN;\n', '\n').removesuffix('COMMIT;\n')
    with psycopg.connect(**local_parameters()) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        tables = [r[0] for r in conn.execute("""SELECT tablename FROM pg_tables
            WHERE schemaname='geochem' AND tablename NOT IN
            ('measurement_unit','analysis_method','indicator','source_field_semantics') ORDER BY tablename""")]

        def fingerprint():
            return {name: conn.execute(sql.SQL('SELECT count(*),sum(hashtextextended(row_to_json(t)::text,0)::numeric) FROM geochem.{} t')
                    .format(sql.Identifier(name))).fetchone() for name in tables}

        before = fingerprint()
        conn.execute(ddl)
        counts = conn.execute('SELECT count(*) FROM geochem.source_field_semantics').fetchone()
        conn.execute(ddl)
        assert conn.execute('SELECT count(*) FROM geochem.source_field_semantics').fetchone() == counts
        assert before == fingerprint(), 'Existing data changed'
        assert conn.execute("""SELECT count(*) FROM geochem.source_field f
            LEFT JOIN geochem.source_field_semantics s USING(source_field_id)
            WHERE f.target_table='analysis_result' AND s.source_field_id IS NULL""").fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM geochem.v_analysis_result_semantic').fetchone() == conn.execute(
            'SELECT count(*) FROM geochem.analysis_result').fetchone()
        # Re-running refresh preserves manual mappings and review decisions.
        conn.execute('SAVEPOINT curator_check')
        conn.execute('UPDATE geochem.source_field_semantics SET reviewed=true WHERE source_field_id=(SELECT min(source_field_id) FROM geochem.source_field_semantics)')
        conn.execute('SELECT geochem.refresh_analysis_catalogs()')
        assert conn.execute('SELECT count(*) FROM geochem.source_field_semantics WHERE reviewed').fetchone()[0] > 0
        conn.execute('ROLLBACK TO SAVEPOINT curator_check')
        conn.execute('SET LOCAL ROLE readonly')
        conn.execute('SELECT * FROM geochem.v_analysis_result_semantic LIMIT 1').fetchone()
        assert not conn.execute("SELECT has_table_privilege(current_user,'geochem.indicator','INSERT')").fetchone()[0]
        conn.execute('RESET ROLE')
        print(f'Validated: {len(tables)} existing tables unchanged; {counts[0]} mappings; repeatable migration; readonly SELECT.')
        if args.apply:
            conn.commit()
            print('Migration committed.')
        else:
            conn.rollback()
            print('Validation rolled back. Use --apply to commit.')


if __name__ == '__main__':
    main()
