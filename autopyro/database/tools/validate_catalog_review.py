"""Test review migration and atomic decisions with rollback; --apply installs 014 only."""
import argparse
from copy import deepcopy
import os
from pathlib import Path

import psycopg
from psycopg import sql

from autopyro.database.connection import local_parameters
from .catalog_review import apply_review,catalog_states,field_states,fingerprint


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    os.environ.setdefault('PGUSER','admin')
    ddl=(Path(__file__).resolve().parents[3]/'database/migrations/014_semantic_review_history.sql').read_text(encoding='utf-8')
    ddl=ddl.replace('\nBEGIN;\n','\n').removesuffix('COMMIT;\n')
    with psycopg.connect(**local_parameters(),connect_timeout=8) as conn:
        tables=[r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname='geochem' ORDER BY tablename")]

        def snapshot():
            return {name:conn.execute(sql.SQL('SELECT count(*),sum(hashtextextended(row_to_json(t)::text,0)::numeric) FROM geochem.{} t')
                .format(sql.Identifier(name))).fetchone() for name in tables}

        original=snapshot()
        conn.execute(ddl)
        conn.execute(ddl)
        assert original==snapshot()
        conn.execute('SAVEPOINT fixture')
        fields=field_states(conn)
        field=next(key for key,value in fields.items() if value['source_field']['source_dtype']=='Float64' and value['unit'] is not None)
        state=fields[field]
        mapping={'source_field_id':field,'expected':fingerprint(state),'indicator_code':'validation:canonical',
                 'unit_code':state['unit']['code'] if state['unit'] else None,
                 'method_code':state['method']['code'] if state['method'] else None,'note':'Synthetic review only'}
        doc={'schema_version':1,'reviewer':'validator','evidence':'Synthetic rollback-only test',
             'catalogs':[{'kind':'indicator','code':'validation:canonical','expected':None,
                          'label':'Validation only','description':'No scientific equivalence asserted'},
                         {'kind':'unit','code':state['unit']['code'],'expected':fingerprint(state['unit'])}],
             'mappings':[mapping]}

        def rejected(document):
            try:
                with conn.transaction():
                    apply_review(conn,document)
            except (ValueError,psycopg.Error):
                return
            raise AssertionError('Invalid review accepted')

        stale=deepcopy(doc)
        stale['mappings'][0]['expected']='0'*64
        rejected(stale)
        partial=deepcopy(doc)
        partial['mappings'][0]['method_code']='validation:missing'
        rejected(partial)
        assert 'validation:canonical' not in catalog_states(conn)['indicator']
        assert field_states(conn)[field]==state
        result=apply_review(conn,doc)
        assert result['already_applied'] is False
        assert apply_review(conn,doc)['already_applied'] is True
        updated=field_states(conn)[field]
        assert updated['indicator']['code']=='validation:canonical' and updated['mapping']['reviewed']
        assert updated['source_field']==state['source_field']
        assert conn.execute('SELECT count(*) FROM geochem.semantic_review_change WHERE semantic_review_id=%s',(result['review_id'],)).fetchone()[0]==3
        stale['evidence']='A second attempt'
        rejected(stale)
        # A same-plan catalog approval must not invalidate its mapping precondition.
        approved=deepcopy(doc)
        approved['evidence']='Second synthetic review'
        approved['catalogs']=[{'kind':'indicator','code':updated['indicator']['code'],'expected':fingerprint(updated['indicator'])}]
        approved['mappings'][0]['expected']=fingerprint(updated)
        second=apply_review(conn,approved)
        assert second['review_id']!=result['review_id']
        for table in ['semantic_review_batch','semantic_review_change']:
            try:
                with conn.transaction():
                    conn.execute(sql.SQL('DELETE FROM geochem.{} WHERE semantic_review_id=%s').format(sql.Identifier(table)),(result['review_id'],))
            except psycopg.errors.RaiseException:
                pass
            else:
                raise AssertionError('History deletion accepted')
        conn.execute('ROLLBACK TO SAVEPOINT fixture')
        assert original==snapshot()
        conn.execute('SET LOCAL ROLE readonly')
        conn.execute('SELECT * FROM geochem.v_semantic_review_history LIMIT 1').fetchall()
        assert not conn.execute("SELECT has_table_privilege(current_user,'geochem.semantic_review_batch','INSERT')").fetchone()[0]
        conn.execute('RESET ROLE')
        print(f'Validated: {len(tables)} existing tables unchanged; atomic review, stale-state checks, idempotency, history guards and readonly access. Synthetic reviews rolled back.')
        if args.apply:
            conn.commit()
        else:
            conn.rollback()


if __name__=='__main__':
    main()
