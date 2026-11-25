-- Roles are created/renamed and passwords set by configure_users.py.
-- Database name is the project's local geochemistry database.
REVOKE ALL ON DATABASE geochemistry FROM readonly;
REVOKE TEMPORARY ON DATABASE geochemistry FROM PUBLIC;
GRANT CONNECT ON DATABASE geochemistry TO readonly;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA geochem FROM readonly;
GRANT USAGE ON SCHEMA geochem TO readonly;
REVOKE ALL ON ALL TABLES IN SCHEMA geochem FROM readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA geochem TO readonly;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA geochem FROM readonly;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA geochem TO readonly;
ALTER DEFAULT PRIVILEGES FOR ROLE admin IN SCHEMA geochem
    GRANT SELECT ON TABLES TO readonly;
ALTER DEFAULT PRIVILEGES FOR ROLE admin IN SCHEMA geochem
    GRANT SELECT ON SEQUENCES TO readonly;
ALTER ROLE readonly SET default_transaction_read_only = on;
