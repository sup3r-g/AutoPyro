\set ON_ERROR_STOP on
\pset pager off

-- All fixture data and helper functions are rolled back, including on failure.
BEGIN;
CREATE TEMP TABLE check_results (name text PRIMARY KEY, status text NOT NULL);
CREATE FUNCTION pg_temp.assert_true(ok boolean, label text) RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
    IF ok IS DISTINCT FROM true THEN
        RAISE EXCEPTION 'FAILED: %', label;
    END IF;
    INSERT INTO check_results VALUES (label, 'PASS');
END $$;
CREATE FUNCTION pg_temp.expect_error(statement text, expected_state text, label text)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE actual_state text;
BEGIN
    BEGIN
        EXECUTE statement;
    EXCEPTION WHEN OTHERS THEN
        GET STACKED DIAGNOSTICS actual_state = RETURNED_SQLSTATE;
    END;
    IF actual_state IS DISTINCT FROM expected_state THEN
        RAISE EXCEPTION 'FAILED: %; expected SQLSTATE %, got %', label, expected_state, actual_state;
    END IF;
    INSERT INTO check_results VALUES (label, 'PASS');
END $$;

-- Check the catalog against the entire CSV, not just counts.
CREATE TEMP TABLE expected_catalog (
    group_name text, source_label text, source_dtype text, unit_label text,
    target_table text, target_column text, scope text
);
\copy expected_catalog FROM '/checks/field_catalog.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
-- Migration 002 is optional in the base schema, applied when importing Metis metadata.
INSERT INTO expected_catalog
SELECT 'Metis', 'Sam Status', 'string', NULL, 'sample', 'metis_sam_status', 'core'
WHERE EXISTS (SELECT 1 FROM information_schema.columns
    WHERE table_schema='geochem' AND table_name='sample' AND column_name='metis_sam_status');
SELECT pg_temp.assert_true((SELECT count(*) FROM geochem.source_field) =
    (SELECT count(*) FROM expected_catalog), 'catalog field count matches supported schema');
SELECT pg_temp.assert_true((SELECT count(*) FROM geochem.data_group) =
    (SELECT count(DISTINCT group_name) FROM expected_catalog), 'group count matches supported schema');
SELECT pg_temp.assert_true((SELECT count(*) = 51 FROM geochem.data_group WHERE scope = 'analysis'), '51 analysis groups');
SELECT pg_temp.assert_true(NOT EXISTS (
    (SELECT group_name, source_label, source_dtype, unit_label, target_table, target_column, scope
     FROM geochem.source_field JOIN geochem.data_group USING (group_name)
     EXCEPT SELECT * FROM expected_catalog)
    UNION ALL
    (SELECT * FROM expected_catalog EXCEPT
     SELECT group_name, source_label, source_dtype, unit_label, target_table, target_column, scope
     FROM geochem.source_field JOIN geochem.data_group USING (group_name))
), 'catalog matches every CSV field exactly');
SELECT pg_temp.assert_true((SELECT count(*) FROM expected_catalog WHERE scope = 'core') =
    83 + CASE WHEN EXISTS (SELECT 1 FROM expected_catalog WHERE group_name='Metis') THEN 1 ELSE 0 END,
    'supported core attribute count');
SELECT pg_temp.assert_true(NOT EXISTS (
    SELECT 1 FROM expected_catalog c
    LEFT JOIN information_schema.columns p ON p.table_schema = 'geochem'
        AND p.table_name = c.target_table AND p.column_name = c.target_column
    WHERE c.scope = 'core' AND p.data_type IS DISTINCT FROM
        CASE c.source_dtype WHEN 'string' THEN 'text' WHEN 'Int64' THEN 'bigint' ELSE 'double precision' END
), 'all physical attributes have correct SQL types');

DO $$
DECLARE
    batch_key bigint;
    well_key bigint;
    sample_key bigint;
    other_sample bigint;
    geology_key bigint;
    ref_key bigint;
    row_key bigint;
    run_key bigint;
    repeat_key bigint;
    toc_key bigint;
    tmax_key bigint;
    lab_key bigint;
    other_toc_key bigint;
    item record;
    stmt text;
    matched boolean;
    target_id bigint;
    test_value text;
BEGIN
    INSERT INTO geochem.import_batch(source_name, source_sha256)
        VALUES ('validation fixture', repeat('a', 64)) RETURNING import_batch_id INTO batch_key;
    INSERT INTO geochem.well(well_name) VALUES ('Тестовая скважина') RETURNING well_id INTO well_key;
    INSERT INTO geochem.sample(well_id, name, top_md, base_md)
        VALUES (well_key, 'Проба 1', 2500.5, 2501.5) RETURNING sample_id INTO sample_key;
    INSERT INTO geochem.sample(name) VALUES ('Проба без скважины') RETURNING sample_id INTO other_sample;
    INSERT INTO geochem.sample_geology(sample_id) VALUES (sample_key) RETURNING sample_geology_id INTO geology_key;
    INSERT INTO geochem.reference_document(author) VALUES ('Автор') RETURNING reference_document_id INTO ref_key;
    INSERT INTO geochem.source_record(import_batch_id, row_number, sample_id, sample_geology_id, raw_cells)
        VALUES (batch_key, 0, sample_key, geology_key, '[["Sample","Name","Проба 1"]]')
        RETURNING source_record_id INTO row_key;
    INSERT INTO geochem.source_record_reference VALUES (row_key, ref_key);

    -- Round-trip every physical attribute, with Cyrillic and quotes in text.
    FOR item IN SELECT * FROM expected_catalog WHERE scope = 'core' LOOP
        target_id := CASE item.target_table WHEN 'well' THEN well_key WHEN 'sample' THEN sample_key
                     WHEN 'sample_geology' THEN geology_key ELSE ref_key END;
        test_value := CASE item.source_dtype WHEN 'Int64' THEN '9223372036854775807'
                      WHEN 'Float64' THEN '-1.25' ELSE 'Кириллица ''кавычки'' 22/10/2026' END;
        EXECUTE format('UPDATE geochem.%I SET %I = %L WHERE %I = %L',
            item.target_table, item.target_column, test_value, item.target_table || '_id', target_id);
        EXECUTE format('SELECT %I::text = %L FROM geochem.%I WHERE %I = %L',
            item.target_column, test_value, item.target_table, item.target_table || '_id', target_id) INTO matched;
        IF matched IS DISTINCT FROM true THEN RAISE EXCEPTION 'Core round-trip failed: %', item; END IF;
    END LOOP;
    PERFORM pg_temp.assert_true(true, 'round-trip all core fields without data loss');
    PERFORM pg_temp.assert_true((SELECT well_id IS NULL AND top_md IS NULL FROM geochem.sample WHERE sample_id = other_sample),
        'nullable core fields and sample without well');

    INSERT INTO geochem.analysis_run(sample_id, source_record_id, group_name)
    SELECT sample_key, row_key, group_name FROM geochem.data_group WHERE scope = 'analysis';
    INSERT INTO geochem.analysis_result
        (analysis_run_id, source_field_id, group_name, source_dtype, value_float, value_int, value_text)
    SELECT a.analysis_run_id, f.source_field_id, f.group_name, f.source_dtype,
           CASE WHEN f.source_dtype = 'Float64' THEN -1.25::double precision END,
           CASE WHEN f.source_dtype = 'Int64' THEN 9223372036854775807::bigint END,
           CASE WHEN f.source_dtype = 'string' THEN 'Кириллица ''кавычки'' 22/10/2026' END
    FROM geochem.source_field f JOIN geochem.analysis_run a USING (group_name)
    WHERE a.source_record_id = row_key;
    PERFORM pg_temp.assert_true((SELECT count(*) = 4178 FROM geochem.v_analysis_result WHERE source_record_id = row_key),
        'round-trip all 4178 analytical fields through view');
    PERFORM pg_temp.assert_true(NOT EXISTS (
        SELECT 1 FROM geochem.v_analysis_result WHERE source_record_id = row_key AND (
            (source_dtype = 'Float64' AND value_float IS DISTINCT FROM -1.25::double precision) OR
            (source_dtype = 'Int64' AND value_int IS DISTINCT FROM 9223372036854775807::bigint) OR
            (source_dtype = 'string' AND value_text IS DISTINCT FROM 'Кириллица ''кавычки'' 22/10/2026'))
    ), 'float64, int64 max, Unicode and unparsed dates preserved');
    PERFORM pg_temp.assert_true((SELECT unit_label = 'count' FROM geochem.v_analysis_result
        WHERE source_record_id = row_key AND group_name = 'Sat-GC' AND source_label = 'nC10[h] [count]'),
        'peak qualifier and unit preserved');

    SELECT analysis_run_id INTO STRICT run_key FROM geochem.analysis_run
        WHERE source_record_id = row_key AND group_name = 'Pyrol6';
    SELECT source_field_id INTO STRICT toc_key FROM geochem.source_field WHERE group_name = 'Pyrol6' AND source_label = 'TOC [%wt]';
    SELECT source_field_id INTO STRICT tmax_key FROM geochem.source_field WHERE group_name = 'Pyrol6' AND source_label = 'T-max [°C]';
    SELECT source_field_id INTO STRICT lab_key FROM geochem.source_field WHERE group_name = 'Pyrol6' AND source_label = 'Lab';
    SELECT source_field_id INTO STRICT other_toc_key FROM geochem.source_field WHERE group_name = 'Leco' AND source_label = 'TOC [%wt]';
    INSERT INTO geochem.analysis_run(sample_id, source_record_id, group_name, repeat_number, analyzed_on)
        VALUES (sample_key, row_key, 'Pyrol6', 2, DATE '2026-10-22') RETURNING analysis_run_id INTO repeat_key;
    INSERT INTO geochem.analysis_result(analysis_run_id, source_field_id, group_name, source_dtype, value_float)
        VALUES (repeat_key, toc_key, 'Pyrol6', 'Float64', 2.35);
    INSERT INTO geochem.analysis_result(analysis_run_id, source_field_id, group_name, source_dtype, value_int)
        VALUES (repeat_key, tmax_key, 'Pyrol6', 'Int64', -9223372036854775808);
    PERFORM pg_temp.assert_true((SELECT count(*) = 2 FROM geochem.analysis_run WHERE source_record_id = row_key AND group_name = 'Pyrol6'),
        'independent repeated analysis');
    PERFORM pg_temp.assert_true((SELECT value_int = -9223372036854775808 FROM geochem.v_analysis_result
        WHERE analysis_run_id = repeat_key AND source_label = 'T-max [°C]'), 'int64 minimum preserved');

    -- Each failing operation runs in a subtransaction and must produce its specific SQLSTATE.
    PERFORM pg_temp.expect_error(format('INSERT INTO geochem.source_record(import_batch_id,row_number,sample_id) VALUES (%s,0,%s)',batch_key,sample_key),
        '23505', 'reject duplicate imported row');
    PERFORM pg_temp.expect_error(format('INSERT INTO geochem.analysis_run(sample_id,source_record_id,group_name) VALUES (%s,%s,''Pyrol6'')', sample_key,row_key),
        '23505', 'reject duplicate analysis repeat');
    PERFORM pg_temp.expect_error(format('INSERT INTO geochem.analysis_run(sample_id,source_record_id,group_name,repeat_number) VALUES (%s,%s,''Leco-Extr'',2)',other_sample,row_key),
        '23503', 'reject run linked to wrong sample');
    PERFORM pg_temp.expect_error(format('INSERT INTO geochem.source_record(import_batch_id,row_number,sample_id,sample_geology_id) VALUES (%s,1,%s,%s)',batch_key,other_sample,geology_key),
        '23503', 'reject geology linked to wrong sample');
    PERFORM pg_temp.expect_error(format('INSERT INTO geochem.analysis_run(sample_id,source_record_id,group_name) VALUES (%s,%s,''Well'')',sample_key,row_key),
        '23503', 'reject core group used as analysis');
    PERFORM pg_temp.expect_error(format('INSERT INTO geochem.analysis_run(sample_id,source_record_id,group_name,repeat_number) VALUES (%s,%s,''Pyrol6'',0)',sample_key,row_key),
        '23514', 'reject nonpositive repeat number');
    PERFORM pg_temp.expect_error(format('INSERT INTO geochem.analysis_result(analysis_run_id,source_field_id,group_name,source_dtype,value_float) VALUES (%s,%s,''Leco'',''Float64'',1)',repeat_key,other_toc_key),
        '23503', 'reject field from wrong group');
    PERFORM pg_temp.expect_error(format('INSERT INTO geochem.analysis_result(analysis_run_id,source_field_id,group_name,source_dtype,value_float) VALUES (%s,%s,''Pyrol6'',''Float64'',1)',repeat_key,lab_key),
        '23503', 'reject dtype inconsistent with catalog');
    PERFORM pg_temp.expect_error(format('INSERT INTO geochem.analysis_result(analysis_run_id,source_field_id,group_name,source_dtype,value_text) VALUES (%s,%s,''Pyrol6'',''string'',NULL)',repeat_key,lab_key),
        '23514', 'reject result without value');
    PERFORM pg_temp.expect_error(format('INSERT INTO geochem.analysis_result(analysis_run_id,source_field_id,group_name,source_dtype,value_text,value_int) VALUES (%s,%s,''Pyrol6'',''string'',''lab'',1)',repeat_key,lab_key),
        '23514', 'reject multiple typed values');
    PERFORM pg_temp.expect_error(format('UPDATE geochem.analysis_result SET value_float = ''NaN'' WHERE analysis_run_id=%s AND source_field_id=%s',repeat_key,toc_key),
        '23514', 'reject NaN');
    PERFORM pg_temp.expect_error(format('UPDATE geochem.analysis_result SET value_float = ''Infinity'' WHERE analysis_run_id=%s AND source_field_id=%s',repeat_key,toc_key),
        '23514', 'reject positive infinity');
    PERFORM pg_temp.expect_error(format('UPDATE geochem.analysis_result SET value_float = ''-Infinity'' WHERE analysis_run_id=%s AND source_field_id=%s',repeat_key,toc_key),
        '23514', 'reject negative infinity');
    PERFORM pg_temp.expect_error(format('UPDATE geochem.analysis_result SET value_int = ''9223372036854775808'' WHERE analysis_run_id=%s AND source_field_id=%s',repeat_key,tmax_key),
        '22003', 'reject int64 overflow');
    PERFORM pg_temp.expect_error(format('INSERT INTO geochem.analysis_result(analysis_run_id,source_field_id,group_name,source_dtype,value_float) VALUES (%s,%s,''Pyrol6'',''Float64'',1)',repeat_key,toc_key),
        '23505', 'reject duplicate result');
    PERFORM pg_temp.expect_error(format('DELETE FROM geochem.sample WHERE sample_id=%s',sample_key),
        '23503', 'protect referenced sample from deletion');
    PERFORM pg_temp.expect_error(format('DELETE FROM geochem.source_field WHERE source_field_id=%s',toc_key),
        '23503', 'protect referenced field from deletion');
    PERFORM pg_temp.expect_error('INSERT INTO geochem.import_batch(source_name,source_sha256) VALUES (''bad hash'',''wrong'')',
        '23514', 'reject malformed source hash');
    PERFORM pg_temp.expect_error(format('UPDATE geochem.source_record SET raw_cells=''{}'' WHERE source_record_id=%s',row_key),
        '23514', 'reject non-array raw cells');
END $$;

TABLE check_results;
SELECT count(*) AS passed_checks FROM check_results;
ROLLBACK;
