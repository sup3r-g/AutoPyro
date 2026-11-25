-- Additive semantic layer. Existing source fields, results and views stay intact.
BEGIN;
CREATE TABLE IF NOT EXISTS geochem.measurement_unit (
    unit_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code text NOT NULL UNIQUE CHECK (btrim(code) <> ''),
    symbol text NOT NULL CHECK (btrim(symbol) <> ''),
    description text,
    reviewed boolean NOT NULL DEFAULT false
);
CREATE TABLE IF NOT EXISTS geochem.analysis_method (
    method_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code text NOT NULL UNIQUE CHECK (btrim(code) <> ''),
    name text NOT NULL CHECK (btrim(name) <> ''),
    description text,
    reviewed boolean NOT NULL DEFAULT false
);
CREATE TABLE IF NOT EXISTS geochem.indicator (
    indicator_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code text NOT NULL UNIQUE CHECK (btrim(code) <> ''),
    name text NOT NULL CHECK (btrim(name) <> ''),
    description text,
    reviewed boolean NOT NULL DEFAULT false
);
CREATE TABLE IF NOT EXISTS geochem.source_field_semantics (
    source_field_id bigint PRIMARY KEY REFERENCES geochem.source_field(source_field_id),
    indicator_id bigint NOT NULL REFERENCES geochem.indicator(indicator_id),
    unit_id bigint REFERENCES geochem.measurement_unit(unit_id),
    method_id bigint REFERENCES geochem.analysis_method(method_id),
    reviewed boolean NOT NULL DEFAULT false,
    note text
);
CREATE INDEX IF NOT EXISTS semantics_indicator_idx ON geochem.source_field_semantics(indicator_id);
CREATE INDEX IF NOT EXISTS semantics_method_idx ON geochem.source_field_semantics(method_id);
CREATE INDEX IF NOT EXISTS semantics_unit_idx ON geochem.source_field_semantics(unit_id);

-- Run after extending source_field. Never overwrite curator decisions.
CREATE OR REPLACE FUNCTION geochem.refresh_analysis_catalogs() RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO geochem.measurement_unit(code,symbol,description)
    SELECT DISTINCT 'source:' || unit_label, unit_label, 'Exact source unit; conversion not verified'
    FROM geochem.source_field
    WHERE target_table='analysis_result' AND nullif(btrim(unit_label),'') IS NOT NULL
    ON CONFLICT (code) DO NOTHING;
    INSERT INTO geochem.analysis_method(code,name,description)
    SELECT 'source-group:' || group_name, group_name,
           'Provisional source group; analytical method requires review'
    FROM geochem.data_group WHERE scope='analysis'
    ON CONFLICT (code) DO NOTHING;
    INSERT INTO geochem.indicator(code,name,description)
    SELECT 'source-field:' || source_field_id, source_label,
           'Provisional source field in group ' || group_name
    FROM geochem.source_field WHERE target_table='analysis_result'
    ON CONFLICT (code) DO NOTHING;
    INSERT INTO geochem.source_field_semantics(source_field_id,indicator_id,unit_id,method_id,note)
    SELECT f.source_field_id,i.indicator_id,u.unit_id,m.method_id,
           'Automatically seeded; equivalence across source fields has not been established'
    FROM geochem.source_field f
    JOIN geochem.indicator i ON i.code='source-field:' || f.source_field_id
    LEFT JOIN geochem.measurement_unit u ON u.code='source:' || f.unit_label
    LEFT JOIN geochem.analysis_method m ON m.code='source-group:' || f.group_name
    WHERE f.target_table='analysis_result'
    ON CONFLICT (source_field_id) DO NOTHING;
END $$;
SELECT geochem.refresh_analysis_catalogs();

CREATE OR REPLACE VIEW geochem.v_analysis_result_semantic AS
SELECT r.*, f.source_field_id, s.indicator_id, i.code AS indicator_code, i.name AS indicator_name,
       s.unit_id, u.symbol AS unit_symbol, s.method_id, m.name AS method_name,
       s.reviewed AS mapping_reviewed, i.reviewed AS indicator_reviewed,
       u.reviewed AS unit_reviewed, m.reviewed AS method_reviewed
FROM geochem.v_analysis_result r
JOIN geochem.source_field f USING(group_name,source_label)
LEFT JOIN geochem.source_field_semantics s USING(source_field_id)
LEFT JOIN geochem.indicator i USING(indicator_id)
LEFT JOIN geochem.measurement_unit u USING(unit_id)
LEFT JOIN geochem.analysis_method m USING(method_id);

DO $$ BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname='readonly') THEN
        GRANT SELECT ON geochem.measurement_unit, geochem.analysis_method,
            geochem.indicator, geochem.source_field_semantics,
            geochem.v_analysis_result_semantic TO readonly;
    END IF;
END $$;
COMMIT;
