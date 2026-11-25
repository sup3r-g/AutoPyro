-- Excel-sourced lithostratigraphic intervals. Applied atomically by import_lithostratigraphy.py.
CREATE TABLE IF NOT EXISTS geochem.lithostrat_wellbore (
    lithostrat_wellbore_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    wellbore_name text NOT NULL UNIQUE CHECK (length(btrim(wellbore_name)) > 0),
    -- Exact source-name match only; never collapse sidetracks or re-entries.
    well_id bigint REFERENCES geochem.well(well_id)
);
CREATE INDEX IF NOT EXISTS lithostrat_wellbore_well_idx ON geochem.lithostrat_wellbore(well_id);

CREATE TABLE IF NOT EXISTS geochem.lithostrat_unit (
    lithostrat_unit_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    unit_name text NOT NULL CHECK (length(btrim(unit_name)) > 0),
    unit_level text NOT NULL CHECK (unit_level IN ('GROUP','SUBGROUP','FORMATION','MEMBER')),
    UNIQUE(unit_name,unit_level)
);

CREATE TABLE IF NOT EXISTS geochem.wellbore_lithostrat_interval (
    wellbore_lithostrat_interval_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    import_batch_id bigint NOT NULL REFERENCES geochem.import_batch(import_batch_id),
    source_sheet text NOT NULL,
    source_row integer NOT NULL CHECK (source_row >= 2),
    lithostrat_wellbore_id bigint NOT NULL REFERENCES geochem.lithostrat_wellbore(lithostrat_wellbore_id),
    lithostrat_unit_id bigint NOT NULL REFERENCES geochem.lithostrat_unit(lithostrat_unit_id),
    top_depth_m numeric NOT NULL,
    bottom_depth_m numeric NOT NULL,
    thickness_m numeric GENERATED ALWAYS AS (bottom_depth_m - top_depth_m) STORED,
    -- Excel gives metres but no datum or MD/TVD designation. Leave unspecified.
    depth_reference text,
    raw_cells jsonb NOT NULL CHECK (jsonb_typeof(raw_cells) = 'object'),
    CHECK (top_depth_m >= 0 AND top_depth_m < 'Infinity'::numeric),
    CHECK (bottom_depth_m > top_depth_m AND bottom_depth_m < 'Infinity'::numeric),
    UNIQUE(import_batch_id,source_sheet,source_row)
);
CREATE INDEX IF NOT EXISTS lithostrat_interval_well_depth_idx
    ON geochem.wellbore_lithostrat_interval(lithostrat_wellbore_id,top_depth_m,bottom_depth_m);
CREATE INDEX IF NOT EXISTS lithostrat_interval_unit_idx
    ON geochem.wellbore_lithostrat_interval(lithostrat_unit_id);
COMMENT ON TABLE geochem.wellbore_lithostrat_interval IS
    'Source lithostratigraphic intervals in metres; datum and MD/TVD not specified by Excel. Nested levels and conflicting source rows are retained.';

CREATE OR REPLACE VIEW geochem.v_wellbore_lithostratigraphy AS
SELECT i.wellbore_lithostrat_interval_id, i.import_batch_id, b.source_name,
       i.source_sheet, i.source_row, wb.lithostrat_wellbore_id, wb.wellbore_name,
       wb.well_id, w.well_name, w.area, w.field, w.country, w.lat, w.long,
       u.unit_name, u.unit_level, i.top_depth_m, i.bottom_depth_m, i.thickness_m,
       i.depth_reference, i.raw_cells
FROM geochem.wellbore_lithostrat_interval i
JOIN geochem.lithostrat_wellbore wb USING(lithostrat_wellbore_id)
JOIN geochem.lithostrat_unit u USING(lithostrat_unit_id)
JOIN geochem.import_batch b USING(import_batch_id)
LEFT JOIN geochem.well w USING(well_id);

-- Same-level overlap is a source-quality observation, not an import rejection.
-- Different levels (e.g. GROUP and FORMATION) are expected to overlap.
CREATE OR REPLACE VIEW geochem.v_lithostrat_interval_overlap AS
SELECT a.import_batch_id, wb.wellbore_name, ua.unit_level,
       a.wellbore_lithostrat_interval_id AS interval_a_id, a.source_row AS source_row_a,
       ua.unit_name AS unit_a, a.top_depth_m AS top_a, a.bottom_depth_m AS bottom_a,
       b.wellbore_lithostrat_interval_id AS interval_b_id, b.source_row AS source_row_b,
       ub.unit_name AS unit_b, b.top_depth_m AS top_b, b.bottom_depth_m AS bottom_b
FROM geochem.wellbore_lithostrat_interval a
JOIN geochem.wellbore_lithostrat_interval b
    ON a.import_batch_id=b.import_batch_id AND a.lithostrat_wellbore_id=b.lithostrat_wellbore_id
    AND a.wellbore_lithostrat_interval_id < b.wellbore_lithostrat_interval_id
    AND a.top_depth_m < b.bottom_depth_m AND b.top_depth_m < a.bottom_depth_m
JOIN geochem.lithostrat_wellbore wb ON wb.lithostrat_wellbore_id=a.lithostrat_wellbore_id
JOIN geochem.lithostrat_unit ua ON ua.lithostrat_unit_id=a.lithostrat_unit_id
JOIN geochem.lithostrat_unit ub ON ub.lithostrat_unit_id=b.lithostrat_unit_id AND ub.unit_level=ua.unit_level;
