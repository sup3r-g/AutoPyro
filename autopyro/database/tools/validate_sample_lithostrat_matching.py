"""Validate migration 010 with rolled-back fixtures; --apply commits DDL only."""
import argparse
import os
from pathlib import Path

import psycopg
from psycopg import sql

from autopyro.database.connection import local_parameters


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--optimized', action='store_true', help='Test optimized status view from migration 011 with rollback')
    args = parser.parse_args()
    if args.apply and args.optimized:
        parser.error('--optimized is a rollback-only validation mode')
    os.environ.setdefault('PGUSER', 'admin')
    path = Path(__file__).resolve().parents[3] / 'database/migrations/010_sample_lithostrat_matching.sql'
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
        if args.optimized:
            optimized = path.with_name('011_analytical_performance.sql').read_text(encoding='utf-8')
            conn.execute(optimized.replace('\nBEGIN;\n','\n').removesuffix('COMMIT;\n'))
        assert before == fingerprint()
        assert conn.execute('SELECT count(*) FROM geochem.v_sample_lithostrat_status').fetchone() == conn.execute('SELECT count(*) FROM geochem.sample').fetchone()
        conn.execute('SAVEPOINT fixture')
        sample_id, well_id, wellbore_id, parent_id = conn.execute('''SELECT s.sample_id,s.well_id,s.wellbore_id,v.interpretation_id
            FROM geochem.v_sample_wellbore s JOIN geochem.lithostrat_interpretation v USING(wellbore_id)
            WHERE v.status='published' LIMIT 1''').fetchone()
        # Test data and manual confirmations exist only inside the rolled-back savepoint.
        conn.execute('UPDATE geochem.well_wellbore_link SET reviewed=true WHERE well_id=%s',(well_id,))
        version = conn.execute("SELECT geochem.clone_lithostrat_interpretation(%s,'matching-validation','validator','Temporary synthetic depth test')",(parent_id,)).fetchone()[0]
        conn.execute('DELETE FROM geochem.lithostrat_interpretation_interval WHERE interpretation_id=%s',(version,))
        for number,name,level,top,bottom in [(1,'A','FORMATION',0,10),(2,'B','FORMATION',10,20),(3,'G','GROUP',0,20)]:
            conn.execute('''INSERT INTO geochem.lithostrat_interpretation_interval
                (interpretation_id,interval_number,unit_name,unit_level,top_depth_m,bottom_depth_m,depth_type,depth_datum)
                VALUES (%s,%s,%s,%s,%s,%s,'MD','TEST_DATUM')''',(version,number,name,level,top,bottom))

        def rejected(query,params):
            try:
                with conn.transaction():
                    conn.execute(query,params)
            except psycopg.Error:
                return
            raise AssertionError('Invalid selection accepted')

        select = '''INSERT INTO geochem.wellbore_interpretation_selection(wellbore_id,interpretation_id,selected_by,evidence)
            VALUES (%s,%s,'validator','Temporary fixture') ON CONFLICT(wellbore_id) DO UPDATE
            SET interpretation_id=excluded.interpretation_id,selected_by=excluded.selected_by,evidence=excluded.evidence'''
        rejected(select,(wellbore_id,version))
        conn.execute("UPDATE geochem.lithostrat_interpretation SET status='published',published_at=current_timestamp WHERE interpretation_id=%s",(version,))
        other = conn.execute('SELECT wellbore_id FROM geochem.wellbore WHERE wellbore_id<>%s LIMIT 1',(wellbore_id,)).fetchone()[0]
        rejected(select,(other,version))
        conn.execute(select,(wellbore_id,version))

        def set_sample(top,bottom,datum='TEST_DATUM'):
            conn.execute('UPDATE geochem.sample SET top_md=%s,base_md=%s,md_datum=%s WHERE sample_id=%s',(top,bottom,datum,sample_id))

        def status():
            return conn.execute('SELECT match_status FROM geochem.v_sample_lithostrat_status WHERE sample_id=%s',(sample_id,)).fetchone()[0]

        set_sample(10,10)
        rows = conn.execute('SELECT unit_name,overlap_fraction,matches_at_level FROM geochem.v_sample_lithostrat_match WHERE sample_id=%s ORDER BY unit_name',(sample_id,)).fetchall()
        assert rows == [('B',None,1),('G',None,1)], rows
        assert status() == 'matched'
        set_sample(9,11)
        assert status() == 'ambiguous'
        assert conn.execute("SELECT count(*) FROM geochem.v_sample_lithostrat_match WHERE sample_id=%s AND unit_level='FORMATION' AND overlap_fraction=0.5 AND match_kind='partial'",(sample_id,)).fetchone()[0] == 2
        set_sample(0,10)
        assert status() == 'matched'
        matches = conn.execute('SELECT count(*) FROM geochem.v_sample_lithostrat_match WHERE sample_id=%s',(sample_id,)).fetchone()[0]
        results = conn.execute('SELECT count(*) FROM geochem.v_geochemical_measurement WHERE sample_id=%s',(sample_id,)).fetchone()[0]
        assert conn.execute('SELECT count(*) FROM geochem.v_geochemical_lithostratigraphy WHERE sample_id=%s',(sample_id,)).fetchone()[0] == matches*results
        for top,bottom,datum,expected in [(20,20,'TEST_DATUM','outside_intervals'),
                (None,10,'TEST_DATUM','invalid_or_missing_sample_depth'),
                (11,9,'TEST_DATUM','invalid_or_missing_sample_depth'),
                (0,10,'OTHER','no_compatible_depth_reference'),(0,10,'','unknown_sample_datum')]:
            set_sample(top,bottom,datum)
            assert status()==expected, (top,bottom,datum,status())
        set_sample(0,10)
        conn.execute('UPDATE geochem.well_wellbore_link SET reviewed=false WHERE well_id=%s',(well_id,))
        assert status()=='unreviewed_wellbore_link'
        conn.execute('DELETE FROM geochem.wellbore_interpretation_selection WHERE wellbore_id=%s',(wellbore_id,))
        assert status()=='no_selected_version'
        conn.execute('ROLLBACK TO SAVEPOINT fixture')
        assert before == fingerprint()
        conn.execute('SET LOCAL ROLE readonly')
        for view in ['v_sample_lithostrat_match','v_sample_lithostrat_status','v_geochemical_lithostratigraphy']:
            conn.execute(sql.SQL('SELECT * FROM geochem.{} LIMIT 1').format(sql.Identifier(view))).fetchone()
        assert not conn.execute("SELECT has_table_privilege(current_user,'geochem.wellbore_interpretation_selection','INSERT')").fetchone()[0]
        conn.execute('RESET ROLE')
        print(f'Validated: {len(tables)} existing tables unchanged; explicit published selection; boundaries, partial overlaps, ranks, ambiguity, datum checks and readonly access.')
        if args.apply:
            conn.commit()
            print('Migration committed; synthetic selections and edits rolled back.')
        else:
            conn.rollback()
            print('Validation rolled back.')


if __name__ == '__main__':
    main()
