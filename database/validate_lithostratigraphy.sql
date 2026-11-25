\set ON_ERROR_STOP on
\pset pager off
BEGIN;
CREATE TEMP TABLE litho_checks (name text PRIMARY KEY, status text NOT NULL);
CREATE FUNCTION pg_temp.assert_litho(ok boolean,label text) RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    IF ok IS DISTINCT FROM true THEN RAISE EXCEPTION 'FAILED: %',label; END IF;
    INSERT INTO litho_checks VALUES(label,'PASS');
END $$;
CREATE FUNCTION pg_temp.reject_litho(statement text,expected_state text,label text) RETURNS void LANGUAGE plpgsql AS $$
DECLARE actual_state text;
BEGIN
    BEGIN EXECUTE statement;
    EXCEPTION WHEN OTHERS THEN GET STACKED DIAGNOSTICS actual_state=RETURNED_SQLSTATE;
    END;
    IF actual_state IS DISTINCT FROM expected_state THEN
        RAISE EXCEPTION 'FAILED: %, expected %, got %',label,expected_state,actual_state;
    END IF;
    INSERT INTO litho_checks VALUES(label,'PASS');
END $$;

SELECT pg_temp.assert_litho((SELECT count(*)=2514 AND count(DISTINCT wellbore_name)=181
    FROM geochem.v_wellbore_lithostratigraphy), '2514 source intervals in 181 wellbores');
SELECT pg_temp.assert_litho((SELECT count(*)=51 FROM geochem.lithostrat_unit),'51 distinct named units');
SELECT pg_temp.assert_litho((SELECT count(*)=127 FROM geochem.lithostrat_wellbore WHERE well_id IS NOT NULL),'127 exact well links');
SELECT pg_temp.assert_litho((SELECT count(*)=54 FROM geochem.lithostrat_wellbore WHERE well_id IS NULL),'54 unmatched wellbores retained');
SELECT pg_temp.assert_litho(NOT EXISTS (SELECT 1 FROM geochem.v_wellbore_lithostratigraphy
    WHERE well_id IS NOT NULL AND wellbore_name IS DISTINCT FROM well_name),'all well links use exact names');
SELECT pg_temp.assert_litho(NOT EXISTS (SELECT 1 FROM geochem.v_wellbore_lithostratigraphy
    WHERE well_id IS NULL AND area IS NOT NULL),'unmatched area remains unknown');
SELECT pg_temp.assert_litho(NOT EXISTS (SELECT 1 FROM geochem.v_wellbore_lithostratigraphy
    WHERE wellbore_name IS DISTINCT FROM raw_cells->>'Wellbore name'
    OR unit_name IS DISTINCT FROM raw_cells->>'Lithostrat. unit'
    OR unit_level IS DISTINCT FROM raw_cells->>'Level'
    OR top_depth_m IS DISTINCT FROM (raw_cells->>'Top depth [m]')::numeric
    OR bottom_depth_m IS DISTINCT FROM (raw_cells->>'Bottom depth [m]')::numeric),
    'all five source values match retained raw cells');
SELECT pg_temp.assert_litho(NOT EXISTS (SELECT 1 FROM geochem.wellbore_lithostrat_interval
    WHERE thickness_m IS DISTINCT FROM bottom_depth_m-top_depth_m OR depth_reference IS NOT NULL),
    'thickness computed and unspecified datum preserved');
SELECT pg_temp.assert_litho((SELECT count(*)=1 FROM geochem.v_lithostrat_interval_overlap
    WHERE wellbore_name='7221/4-1' AND unit_level='FORMATION'),'source same-level conflict is visible');
SELECT pg_temp.assert_litho((SELECT count(*)=31665 FROM geochem.sample)
    AND (SELECT count(*)=1056734 FROM geochem.analysis_result),'original geochemical data unchanged');

DO $$
DECLARE i bigint; wb bigint; u bigint;
BEGIN
    SELECT wellbore_lithostrat_interval_id,lithostrat_wellbore_id,lithostrat_unit_id INTO STRICT i,wb,u
        FROM geochem.wellbore_lithostrat_interval ORDER BY wellbore_lithostrat_interval_id LIMIT 1;
    PERFORM pg_temp.reject_litho(format('UPDATE geochem.wellbore_lithostrat_interval SET top_depth_m=-1 WHERE wellbore_lithostrat_interval_id=%s',i),'23514','reject negative top depth');
    PERFORM pg_temp.reject_litho(format('UPDATE geochem.wellbore_lithostrat_interval SET bottom_depth_m=top_depth_m WHERE wellbore_lithostrat_interval_id=%s',i),'23514','reject zero thickness');
    PERFORM pg_temp.reject_litho(format('UPDATE geochem.wellbore_lithostrat_interval SET bottom_depth_m=top_depth_m-1 WHERE wellbore_lithostrat_interval_id=%s',i),'23514','reject reversed interval');
    PERFORM pg_temp.reject_litho(format('UPDATE geochem.wellbore_lithostrat_interval SET bottom_depth_m=''NaN'' WHERE wellbore_lithostrat_interval_id=%s',i),'23514','reject NaN depth');
    PERFORM pg_temp.reject_litho(format('UPDATE geochem.wellbore_lithostrat_interval SET bottom_depth_m=''Infinity'' WHERE wellbore_lithostrat_interval_id=%s',i),'23514','reject infinite depth');
    PERFORM pg_temp.reject_litho(format('UPDATE geochem.lithostrat_unit SET unit_level=''UNKNOWN'' WHERE lithostrat_unit_id=%s',u),'23514','reject unsupported rank');
    PERFORM pg_temp.reject_litho(format('UPDATE geochem.lithostrat_wellbore SET well_id=-1 WHERE lithostrat_wellbore_id=%s',wb),'23503','reject unknown well');
    PERFORM pg_temp.reject_litho(format('DELETE FROM geochem.lithostrat_wellbore WHERE lithostrat_wellbore_id=%s',wb),'23503','protect referenced wellbore');
    PERFORM pg_temp.reject_litho(format('DELETE FROM geochem.lithostrat_unit WHERE lithostrat_unit_id=%s',u),'23503','protect referenced unit');
    PERFORM pg_temp.reject_litho(format('INSERT INTO geochem.wellbore_lithostrat_interval
        (import_batch_id,source_sheet,source_row,lithostrat_wellbore_id,lithostrat_unit_id,top_depth_m,bottom_depth_m,raw_cells)
        SELECT import_batch_id,source_sheet,source_row,lithostrat_wellbore_id,lithostrat_unit_id,top_depth_m,bottom_depth_m,raw_cells
        FROM geochem.wellbore_lithostrat_interval WHERE wellbore_lithostrat_interval_id=%s',i),'23505','reject duplicate source row');
END $$;
TABLE litho_checks;
SELECT count(*) AS passed_checks FROM litho_checks;
ROLLBACK;
