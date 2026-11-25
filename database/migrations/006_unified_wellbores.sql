-- Requires 003. Additive links preserve all source identities.
BEGIN;
CREATE TABLE IF NOT EXISTS geochem.wellbore (
    wellbore_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code text NOT NULL UNIQUE CHECK (btrim(code) <> ''),
    name text,
    reviewed boolean NOT NULL DEFAULT false,
    note text
);
CREATE TABLE IF NOT EXISTS geochem.well_wellbore_link (
    well_id bigint PRIMARY KEY REFERENCES geochem.well(well_id),
    wellbore_id bigint NOT NULL REFERENCES geochem.wellbore(wellbore_id),
    reviewed boolean NOT NULL DEFAULT false
);
CREATE TABLE IF NOT EXISTS geochem.lithostrat_wellbore_link (
    lithostrat_wellbore_id bigint PRIMARY KEY REFERENCES geochem.lithostrat_wellbore(lithostrat_wellbore_id),
    wellbore_id bigint NOT NULL REFERENCES geochem.wellbore(wellbore_id),
    match_basis text NOT NULL CHECK (match_basis IN ('existing_link','unique_exact_name','separate','manual')),
    reviewed boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS well_link_wellbore_idx ON geochem.well_wellbore_link(wellbore_id);
CREATE INDEX IF NOT EXISTS lithostrat_link_wellbore_idx ON geochem.lithostrat_wellbore_link(wellbore_id);

CREATE OR REPLACE FUNCTION geochem.refresh_wellbore_catalog() RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO geochem.wellbore(code,name,note)
    SELECT 'well:' || well_id,well_name,'Source well identity; not automatically merged by name'
    FROM geochem.well ON CONFLICT(code) DO NOTHING;
    INSERT INTO geochem.well_wellbore_link(well_id,wellbore_id)
    SELECT w.well_id,b.wellbore_id FROM geochem.well w
    JOIN geochem.wellbore b ON b.code='well:' || w.well_id
    ON CONFLICT(well_id) DO NOTHING;

    -- Explicit source relation takes priority. Name matching requires exactly one well.
    INSERT INTO geochem.lithostrat_wellbore_link(lithostrat_wellbore_id,wellbore_id,match_basis)
    SELECT l.lithostrat_wellbore_id,m.wellbore_id,
           CASE WHEN l.well_id IS NOT NULL THEN 'existing_link' ELSE 'unique_exact_name' END
    FROM geochem.lithostrat_wellbore l
    JOIN geochem.well_wellbore_link m ON m.well_id=coalesce(l.well_id,
        (SELECT min(w.well_id) FROM geochem.well w
         WHERE w.well_name=l.wellbore_name HAVING count(*)=1))
    ON CONFLICT(lithostrat_wellbore_id) DO NOTHING;

    INSERT INTO geochem.wellbore(code,name,note)
    SELECT 'lithostrat:' || l.lithostrat_wellbore_id,l.wellbore_name,
           'No unambiguous geochemical well match'
    FROM geochem.lithostrat_wellbore l
    LEFT JOIN geochem.lithostrat_wellbore_link m USING(lithostrat_wellbore_id)
    WHERE m.lithostrat_wellbore_id IS NULL ON CONFLICT(code) DO NOTHING;
    INSERT INTO geochem.lithostrat_wellbore_link(lithostrat_wellbore_id,wellbore_id,match_basis)
    SELECT l.lithostrat_wellbore_id,b.wellbore_id,'separate'
    FROM geochem.lithostrat_wellbore l
    JOIN geochem.wellbore b ON b.code='lithostrat:' || l.lithostrat_wellbore_id
    ON CONFLICT(lithostrat_wellbore_id) DO NOTHING;
END $$;
SELECT geochem.refresh_wellbore_catalog();

CREATE OR REPLACE VIEW geochem.v_sample_wellbore AS
SELECT s.*,m.wellbore_id,b.name AS wellbore_name,m.reviewed AS wellbore_link_reviewed,
       'MD'::text AS depth_type, 'm'::text AS depth_unit,
       s.md_datum AS depth_datum
FROM geochem.sample s
LEFT JOIN geochem.well_wellbore_link m USING(well_id)
LEFT JOIN geochem.wellbore b USING(wellbore_id);

CREATE OR REPLACE VIEW geochem.v_lithostratigraphy_wellbore AS
SELECT v.*,m.wellbore_id,m.match_basis,m.reviewed AS wellbore_link_reviewed,
       NULL::text AS depth_type, 'm'::text AS depth_unit,
       NULL::text AS depth_datum
FROM geochem.v_wellbore_lithostratigraphy v
LEFT JOIN geochem.lithostrat_wellbore_link m USING(lithostrat_wellbore_id);

DO $$ BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname='readonly') THEN
        GRANT SELECT ON geochem.wellbore,geochem.well_wellbore_link,
            geochem.lithostrat_wellbore_link,geochem.v_sample_wellbore,
            geochem.v_lithostratigraphy_wellbore TO readonly;
    END IF;
END $$;
COMMIT;
