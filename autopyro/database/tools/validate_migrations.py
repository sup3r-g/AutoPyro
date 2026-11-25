"""Exercise migration runner on local DB, always rolling back all changes."""
from dataclasses import replace
import os

import psycopg
from psycopg import sql

from autopyro.database.connection import local_parameters
from .migrate import Migration, apply_migrations, load_migrations, migration_plan


def main():
    os.environ.setdefault('PGUSER', 'admin')
    migrations = load_migrations()
    with psycopg.connect(**local_parameters()) as conn:
        tables = [r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname='geochem' ORDER BY tablename")]

        def fingerprint():
            return {name: conn.execute(sql.SQL('SELECT count(*),sum(hashtextextended(row_to_json(t)::text,0)::numeric) FROM geochem.{} t')
                    .format(sql.Identifier(name))).fetchone() for name in tables}

        before = fingerprint()
        bad = Migration('011_atomic_test.sql','0'*64,
                        'CREATE TABLE geochem.migration_atomic_test(id int); SELECT 1/0;')
        try:
            with conn.transaction():
                apply_migrations(conn,[*migrations,bad])
        except psycopg.errors.DivisionByZero:
            pass
        else:
            raise AssertionError('Failing migration was accepted')
        assert conn.execute("SELECT to_regclass('geochem.migration_atomic_test')").fetchone()[0] is None
        assert before == fingerprint()
        apply_migrations(conn,migrations)
        assert apply_migrations(conn,migrations) == []
        assert all(done for _, done in migration_plan(conn,migrations))
        assert before == fingerprint()
        for invalid in [[replace(migrations[0],checksum='f'*64),*migrations[1:]],migrations[:-1]]:
            try:
                migration_plan(conn,invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('Changed/missing migration not detected')
        with conn.transaction(force_rollback=True):
            conn.execute('DELETE FROM geochem.schema_migration WHERE filename=%s',(migrations[0].name,))
            try:
                migration_plan(conn,migrations)
            except ValueError as exc:
                assert 'gap' in str(exc)
            else:
                raise AssertionError('History gap not detected')
        conn.execute('SET LOCAL ROLE readonly')
        assert all(done for _, done in migration_plan(conn,migrations))
        assert not conn.execute("SELECT has_table_privilege(current_user,'geochem.schema_migration','INSERT')").fetchone()[0]
        conn.rollback()
        print(f'Validated: {len(tables)} tables unchanged; atomic rollback, idempotency, checksums, missing files, history gaps and readonly plan. All changes rolled back.')


if __name__ == '__main__':
    main()
