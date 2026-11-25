"""Reject source accounts that could write or administer the geochem schema."""
from collections.abc import Mapping


def values(row):
    return list(row.values()) if isinstance(row, Mapping) else list(row)


def verify_read_only(connection):
    mode = connection.execute("SHOW transaction_read_only").fetchone()
    if values(mode)[0] != "on":
        raise ValueError("Источник должен читаться в read-only транзакции")
    flags = connection.execute("""SELECT rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls
        FROM pg_roles WHERE rolname=current_user""").fetchone()
    if flags is None or any(values(flags)):
        raise ValueError("Для чтения источника нужна отдельная роль без административных прав (readonly)")
    writable = connection.execute("""WITH reachable AS (
        SELECT * FROM pg_roles WHERE pg_has_role(oid,'MEMBER')
        ) SELECT EXISTS (
        SELECT 1 FROM reachable r WHERE r.rolsuper OR r.rolcreatedb OR r.rolcreaterole
            OR r.rolreplication OR r.rolbypassrls
        UNION ALL
        SELECT 1 FROM reachable r CROSS JOIN pg_namespace n WHERE n.nspname='geochem'
        AND (has_schema_privilege(r.oid,n.oid,'CREATE') OR r.oid=n.nspowner)
        UNION ALL
        SELECT 1 FROM reachable r CROSS JOIN pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='geochem' AND c.relkind IN ('r','p','v','m','f','S')
        AND (r.oid=c.relowner OR CASE WHEN c.relkind='S'
            THEN has_sequence_privilege(r.oid,c.oid,'USAGE,UPDATE')
            ELSE has_table_privilege(r.oid,c.oid,'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')
                OR has_any_column_privilege(r.oid,c.oid,'INSERT,UPDATE,REFERENCES') END)
        UNION ALL SELECT 1 FROM reachable r WHERE has_database_privilege(r.oid,current_database(),'CREATE')
        ) AS can_write""").fetchone()
    if values(writable)[0]:
        raise ValueError("Учётная запись имеет права записи или владения источником. Используйте роль только с SELECT")
