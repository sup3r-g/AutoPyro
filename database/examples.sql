-- Apply schema.sql first. These read-only queries also work on an empty database.

-- All pyrolysis measurements for the selected sample, keeping method and repeat.
SELECT sample_id, analysis_run_id, repeat_number, group_name,
       source_label, unit_label, value_float, value_int
FROM geochem.v_analysis_result
WHERE sample_id = 1
  AND group_name IN ('Pyrol2', 'Pyrol6', 'Pyrol6-Extr', 'PyrolHawk')
ORDER BY analysis_run_id, source_label;

-- A wide report per run (never merge independent runs by sample alone).
SELECT a.sample_id, a.analysis_run_id, a.repeat_number,
       max(r.value_float) FILTER (WHERE f.source_label = 'TOC [%wt]') AS toc_wt_pct,
       max(r.value_float) FILTER (WHERE f.source_label = 'S2 [ppt (mg/g)]') AS s2_mg_g,
       max(r.value_int) FILTER (WHERE f.source_label = 'T-max [°C]') AS tmax_c
FROM geochem.analysis_run a
JOIN geochem.analysis_result r USING (analysis_run_id, group_name)
JOIN geochem.source_field f USING (source_field_id, group_name, source_dtype)
WHERE a.group_name = 'Pyrol6'
GROUP BY a.sample_id, a.analysis_run_id, a.repeat_number;

-- Catalog completeness: expected 4261, 55, 4178.
SELECT count(*) AS fields, count(DISTINCT group_name) AS groups,
       count(*) FILTER (WHERE target_table = 'analysis_result') AS analysis_fields
FROM geochem.source_field;

-- Demonstration load. ROLLBACK leaves the database contents unchanged.
-- Identity sequences may advance even when the transaction is rolled back.
BEGIN;
DO $$
DECLARE
    batch_id bigint;
    well_key bigint;
    sample_key bigint;
    row_key bigint;
    run_key bigint;
BEGIN
    INSERT INTO geochem.import_batch(source_name) VALUES ('example only')
        RETURNING import_batch_id INTO batch_id;
    INSERT INTO geochem.well(well_name) VALUES ('W-1')
        RETURNING well_id INTO well_key;
    INSERT INTO geochem.sample(well_id, name, top_md, base_md)
        VALUES (well_key, 'S-1', 2500, 2501) RETURNING sample_id INTO sample_key;
    INSERT INTO geochem.source_record(import_batch_id, row_number, sample_id)
        VALUES (batch_id, 0, sample_key) RETURNING source_record_id INTO row_key;
    INSERT INTO geochem.analysis_run(sample_id, source_record_id, group_name)
        VALUES (sample_key, row_key, 'Pyrol6') RETURNING analysis_run_id INTO run_key;

    INSERT INTO geochem.analysis_result
        (analysis_run_id, source_field_id, group_name, source_dtype, value_float)
    SELECT run_key, source_field_id, group_name, source_dtype, 2.35
    FROM geochem.source_field
    WHERE group_name = 'Pyrol6' AND source_label = 'TOC [%wt]';

    INSERT INTO geochem.analysis_result
        (analysis_run_id, source_field_id, group_name, source_dtype, value_int)
    SELECT run_key, source_field_id, group_name, source_dtype, 435
    FROM geochem.source_field
    WHERE group_name = 'Pyrol6' AND source_label = 'T-max [°C]';
END $$;
SELECT * FROM geochem.v_analysis_result
WHERE source_record_id IN (
    SELECT s.source_record_id FROM geochem.source_record s
    JOIN geochem.import_batch b USING (import_batch_id)
    WHERE b.source_name = 'example only'
);
ROLLBACK;
