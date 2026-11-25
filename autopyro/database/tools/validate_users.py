"""Verify actual TCP logins, SELECT access, denied writes, and future object grants."""
import json
import secrets
from pathlib import Path

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[3]


def main():
    env = {}
    for line in (ROOT / '.env').read_text(encoding='utf-8').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            env[key.strip()] = value.strip().strip('\"\'')
    params = dict(host='127.0.0.1', port=env.get('POSTGRES_PORT', '15432'), dbname='geochemistry')
    passed = []

    def check(ok, label):
        if not ok:
            raise AssertionError(label)
        passed.append(label)

    with psycopg.connect(**params, user='admin', password=env['POSTGRES_PASSWORD'], autocommit=True) as admin:
        check(admin.execute("SELECT rolsuper AND rolcreatedb AND rolcreaterole FROM pg_roles WHERE rolname='admin'").fetchone()[0], 'admin retains full rights')
        check(admin.execute("SELECT pg_get_userbyid(datdba)='admin' FROM pg_database WHERE datname='geochemistry'").fetchone()[0], 'admin owns database')
        check(not admin.execute("SELECT 1 FROM pg_roles WHERE rolname='geochem' OR rolname LIKE '_autopyro_setup_%'").fetchall(), 'old login and bootstrap role absent')
        check(admin.execute("SELECT NOT rolsuper AND NOT rolcreatedb AND NOT rolcreaterole AND NOT rolreplication AND NOT rolbypassrls FROM pg_roles WHERE rolname='readonly'").fetchone()[0], 'reader has no administrative attributes')
        check(not admin.execute("SELECT 1 FROM pg_auth_members WHERE member='readonly'::regrole").fetchall(), 'reader has no inherited role memberships')
        with psycopg.connect(**params, user='readonly', password=env['READONLY_PASSWORD'], autocommit=True) as reader:
            check(reader.execute('SELECT current_user').fetchone()[0] == 'readonly', 'reader TCP password login')
            check(reader.execute('SHOW default_transaction_read_only').fetchone()[0] == 'on', 'read-only transactions by default')
            objects = admin.execute("SELECT tablename FROM pg_tables WHERE schemaname='geochem' UNION ALL SELECT viewname FROM pg_views WHERE schemaname='geochem'").fetchall()
            for (name,) in objects:
                reader.execute(sql.SQL('SELECT * FROM geochem.{} LIMIT 1').format(sql.Identifier(name))).fetchall()
            check(len(objects) == 17, 'SELECT works on all 14 tables and 3 views')
            sequences = admin.execute("SELECT sequencename FROM pg_sequences WHERE schemaname='geochem'").fetchall()
            for (name,) in sequences:
                reader.execute(sql.SQL('SELECT last_value FROM geochem.{}').format(sql.Identifier(name))).fetchone()
                check(not reader.execute('SELECT has_sequence_privilege(current_user,%s,\'USAGE\') OR has_sequence_privilege(current_user,%s,\'UPDATE\')', ('geochem.'+name, 'geochem.'+name)).fetchone()[0], 'no mutation rights on sequence ' + name)
            # Turn off the convenience read-only default: permission checks must
            # still block writes. Every attempted operation is rolled back.
            reader.execute('SET default_transaction_read_only=off')
            attempts = {
                'INSERT blocked': 'INSERT INTO geochem.sample(name) SELECT NULL WHERE false',
                'UPDATE blocked': 'UPDATE geochem.sample SET name=name WHERE false',
                'DELETE blocked': 'DELETE FROM geochem.sample WHERE false',
                'TRUNCATE blocked': 'TRUNCATE geochem.analysis_result',
                'ALTER TABLE blocked': 'ALTER TABLE geochem.sample ADD COLUMN _reader_probe integer',
                'DROP TABLE blocked': 'DROP TABLE geochem.analysis_result',
                'CREATE TABLE geochem blocked': 'CREATE TABLE geochem._reader_probe(id integer)',
                'CREATE TABLE public blocked': 'CREATE TABLE public._reader_probe(id integer)',
                'CREATE TEMP TABLE blocked': 'CREATE TEMP TABLE _reader_probe(id integer)',
                'CREATE SCHEMA blocked': 'CREATE SCHEMA _reader_probe',
                'SET ROLE admin blocked': 'SET LOCAL ROLE admin',
            }
            for label, statement in attempts.items():
                state = None
                try:
                    with reader.transaction():
                        reader.execute(statement)
                        raise psycopg.Rollback()
                except psycopg.Error as error:
                    state = error.sqlstate
                check(state == '42501', label)
        table = '_readonly_future_' + secrets.token_hex(6)
        with admin.transaction():
            admin.execute(sql.SQL('CREATE TABLE geochem.{} (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY)').format(sql.Identifier(table)))
            admin.execute('SET LOCAL ROLE readonly')
            admin.execute(sql.SQL('SELECT * FROM geochem.{}').format(sql.Identifier(table))).fetchall()
            check(True, 'future admin tables automatically readable')
            check(not admin.execute('SELECT has_table_privilege(current_user,%s,\'INSERT\')', ('geochem.'+table,)).fetchone()[0], 'future admin tables are not writable')
            check(admin.execute('SELECT has_sequence_privilege(current_user,%s,\'SELECT\')', ('geochem.'+table+'_id_seq',)).fetchone()[0], 'future admin sequences automatically readable')
            raise psycopg.Rollback()
        check(admin.execute('SELECT count(*) FROM geochem.sample').fetchone()[0] == 31665, 'sample rows preserved')
        check(admin.execute('SELECT count(*) FROM geochem.analysis_result').fetchone()[0] == 1056734, 'result rows preserved')
    print(json.dumps({'status': 'PASS', 'checks': len(passed), 'passed': passed}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
