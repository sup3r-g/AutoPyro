"""Validate migration 006 with rollback by default; --apply commits it."""
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
    path = Path(__file__).resolve().parents[3] / 'database/migrations/006_unified_wellbores.sql'
    ddl = path.read_text(encoding='utf-8').replace('\nBEGIN;\n', '\n').removesuffix('COMMIT;\n')
    with psycopg.connect(**local_parameters()) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        tables = [r[0] for r in conn.execute("""SELECT tablename FROM pg_tables
            WHERE schemaname='geochem' AND tablename NOT IN
            ('wellbore','well_wellbore_link','lithostrat_wellbore_link') ORDER BY tablename""")]

        def fingerprint():
            return {name: conn.execute(sql.SQL('SELECT count(*),sum(hashtextextended(row_to_json(t)::text,0)::numeric) FROM geochem.{} t')
                    .format(sql.Identifier(name))).fetchone() for name in tables}

        before = fingerprint()
        conn.execute(ddl)
        first = conn.execute('SELECT count(*) FROM geochem.wellbore').fetchone()
        conn.execute(ddl)
        assert first == conn.execute('SELECT count(*) FROM geochem.wellbore').fetchone()
        assert before == fingerprint(), 'Source data changed'
        for source, link, key in [('well','well_wellbore_link','well_id'),
                                  ('lithostrat_wellbore','lithostrat_wellbore_link','lithostrat_wellbore_id')]:
            assert conn.execute(sql.SQL('SELECT count(*) FROM geochem.{} s LEFT JOIN geochem.{} m USING({}) WHERE m.{} IS NULL')
                                .format(*map(sql.Identifier,[source,link,key,key]))).fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM geochem.v_sample_wellbore').fetchone() == conn.execute('SELECT count(*) FROM geochem.sample').fetchone()
        assert conn.execute('SELECT count(*) FROM geochem.v_lithostratigraphy_wellbore').fetchone() == conn.execute('SELECT count(*) FROM geochem.wellbore_lithostrat_interval').fetchone()
        assert conn.execute('SELECT count(*) FROM geochem.v_lithostratigraphy_wellbore WHERE depth_type IS NOT NULL OR depth_datum IS NOT NULL').fetchone()[0] == 0
        assert conn.execute("""SELECT count(*) FROM geochem.lithostrat_wellbore_link l
            JOIN geochem.lithostrat_wellbore s USING(lithostrat_wellbore_id)
            JOIN geochem.well_wellbore_link m ON m.well_id=s.well_id
            WHERE l.match_basis='existing_link' AND l.wellbore_id<>m.wellbore_id""").fetchone()[0] == 0
        conn.execute('SAVEPOINT manual_check')
        conn.execute("UPDATE geochem.lithostrat_wellbore_link SET reviewed=true,match_basis='manual'")
        conn.execute('SELECT geochem.refresh_wellbore_catalog()')
        assert conn.execute("SELECT count(*) FROM geochem.lithostrat_wellbore_link WHERE NOT reviewed OR match_basis<>'manual'").fetchone()[0] == 0
        conn.execute('ROLLBACK TO SAVEPOINT manual_check')
        conn.execute('SET LOCAL ROLE readonly')
        conn.execute('SELECT * FROM geochem.v_sample_wellbore LIMIT 1').fetchone()
        conn.execute('SELECT * FROM geochem.v_lithostratigraphy_wellbore LIMIT 1').fetchone()
        assert not conn.execute("SELECT has_table_privilege(current_user,'geochem.wellbore','INSERT')").fetchone()[0]
        conn.execute('RESET ROLE')
        summary = conn.execute('SELECT match_basis,count(*) FROM geochem.lithostrat_wellbore_link GROUP BY match_basis ORDER BY match_basis').fetchall()
        print(f'Validated: {len(tables)} tables unchanged; {first[0]} wellbores; links {summary}; old row counts preserved; readonly SELECT.')
        if args.apply:
            conn.commit()
            print('Migration committed.')
        else:
            conn.rollback()
            print('Validation rolled back.')


if __name__ == '__main__':
    main()
