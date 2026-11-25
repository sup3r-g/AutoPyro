"""Validate migration 009; --apply commits after tests, without test versions."""
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
    path = Path(__file__).resolve().parents[3] / 'database/migrations/009_lithostrat_versions.sql'
    ddl = path.read_text(encoding='utf-8').replace('\nBEGIN;\n', '\n').removesuffix('COMMIT;\n')
    with psycopg.connect(**local_parameters()) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        tables = [r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname='geochem' ORDER BY tablename")]

        def fingerprint():
            return {name: conn.execute(sql.SQL('SELECT count(*),sum(hashtextextended(row_to_json(t)::text,0)::numeric) FROM geochem.{} t')
                    .format(sql.Identifier(name))).fetchone() for name in tables}

        before = fingerprint()
        conn.execute(ddl)
        counts = conn.execute('SELECT count(*),(SELECT count(*) FROM geochem.lithostrat_interpretation_interval) FROM geochem.lithostrat_interpretation').fetchone()
        conn.execute(ddl)
        assert before == fingerprint(), 'Existing tables changed'
        assert counts == conn.execute('SELECT count(*),(SELECT count(*) FROM geochem.lithostrat_interpretation_interval) FROM geochem.lithostrat_interpretation').fetchone()
        assert conn.execute('''SELECT count(*) FROM geochem.wellbore_lithostrat_interval s
            LEFT JOIN geochem.lithostrat_interpretation_interval i ON i.source_interval_id=s.wellbore_lithostrat_interval_id
            LEFT JOIN geochem.lithostrat_interpretation v USING(interpretation_id)
            WHERE v.source_import_batch_id=s.import_batch_id AND
            (i.top_depth_m<>s.top_depth_m OR i.bottom_depth_m<>s.bottom_depth_m
             OR i.lithostrat_unit_id<>s.lithostrat_unit_id OR i.depth_type IS NOT NULL OR i.depth_datum IS NOT NULL)''').fetchone()[0] == 0
        assert conn.execute('''SELECT count(*) FROM geochem.wellbore_lithostrat_interval s
            WHERE NOT EXISTS (SELECT FROM geochem.lithostrat_interpretation_interval i
            JOIN geochem.lithostrat_interpretation v USING(interpretation_id)
            WHERE i.source_interval_id=s.wellbore_lithostrat_interval_id
            AND v.source_import_batch_id=s.import_batch_id)''').fetchone()[0] == 0
        source_id = conn.execute("SELECT min(interpretation_id) FROM geochem.lithostrat_interpretation WHERE status='published'").fetchone()[0]
        source_rows = conn.execute('SELECT count(*) FROM geochem.lithostrat_interpretation_interval WHERE interpretation_id=%s',(source_id,)).fetchone()[0]
        assert source_id and source_rows
        conn.execute('SAVEPOINT fixture')
        clone_id = conn.execute("SELECT geochem.clone_lithostrat_interpretation(%s,'validation-only','validator','Temporary validation fixture')",(source_id,)).fetchone()[0]
        assert conn.execute('SELECT count(*) FROM geochem.lithostrat_interpretation_interval WHERE interpretation_id=%s',(clone_id,)).fetchone()[0] == source_rows
        conn.execute("UPDATE geochem.lithostrat_interpretation_interval SET note='Edited draft' WHERE interpretation_id=%s",(clone_id,))
        assert conn.execute("SELECT count(*) FROM geochem.lithostrat_interpretation_interval WHERE interpretation_id=%s AND note='Edited draft'",(source_id,)).fetchone()[0] == 0

        def rejected(query, params):
            try:
                with conn.transaction():
                    conn.execute(query, params)
            except psycopg.Error:
                return
            raise AssertionError('Invalid operation accepted: '+query)

        rejected('UPDATE geochem.lithostrat_interpretation_interval SET bottom_depth_m=top_depth_m WHERE interpretation_id=%s',(clone_id,))
        rejected('UPDATE geochem.lithostrat_interpretation SET parent_interpretation_id=NULL WHERE interpretation_id=%s',(clone_id,))
        rejected("SELECT geochem.clone_lithostrat_interpretation(%s,'invalid-parent','validator','test')",(clone_id,))
        conn.execute("UPDATE geochem.lithostrat_interpretation SET status='published',published_at=current_timestamp WHERE interpretation_id=%s",(clone_id,))
        rejected("UPDATE geochem.lithostrat_interpretation SET description='changed' WHERE interpretation_id=%s",(clone_id,))
        rejected('DELETE FROM geochem.lithostrat_interpretation WHERE interpretation_id=%s',(clone_id,))
        rejected("UPDATE geochem.lithostrat_interpretation_interval SET note='changed' WHERE interpretation_id=%s",(clone_id,))
        rejected('DELETE FROM geochem.lithostrat_interpretation_interval WHERE interpretation_id=%s',(clone_id,))
        rejected('''INSERT INTO geochem.lithostrat_interpretation_interval
            (interpretation_id,interval_number,unit_name,unit_level,top_depth_m,bottom_depth_m)
            VALUES (%s,999999,'test','FORMATION',0,1)''',(clone_id,))
        empty_id = conn.execute("""INSERT INTO geochem.lithostrat_interpretation(wellbore_id,version_tag,author,description)
            SELECT wellbore_id,'empty-validation','validator','test' FROM geochem.lithostrat_interpretation
            WHERE interpretation_id=%s RETURNING interpretation_id""",(clone_id,)).fetchone()[0]
        rejected("UPDATE geochem.lithostrat_interpretation SET status='published',published_at=current_timestamp WHERE interpretation_id=%s",(empty_id,))
        conn.execute('ROLLBACK TO SAVEPOINT fixture')
        assert before == fingerprint()
        conn.execute('SET LOCAL ROLE readonly')
        conn.execute('SELECT * FROM geochem.v_lithostrat_interpretation LIMIT 1').fetchone()
        conn.execute('SELECT * FROM geochem.v_lithostrat_interpretation_overlap LIMIT 1').fetchone()
        assert not conn.execute("SELECT has_table_privilege(current_user,'geochem.lithostrat_interpretation','INSERT')").fetchone()[0]
        rejected("SELECT geochem.clone_lithostrat_interpretation(%s,'readonly-test','reader','test')",(source_id,))
        conn.execute('RESET ROLE')
        print(f'Validated: {len(tables)} existing tables unchanged; {counts[0]} versions, {counts[1]} intervals; cloning, draft edits, publication and immutability; readonly access.')
        if args.apply:
            conn.commit()
            print('Migration committed; test versions rolled back.')
        else:
            conn.rollback()
            print('Validation rolled back.')


if __name__ == '__main__':
    main()
