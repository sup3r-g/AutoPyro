-- Requires 005 and 006. No stored data or existing interfaces are changed.
BEGIN;
CREATE OR REPLACE VIEW geochem.v_geochemical_measurement AS
SELECT r.*, s.name AS sample_name,s.well_id,s.wellbore_id,s.wellbore_name,
       s.wellbore_link_reviewed,w.well_name,w.area,w.field AS field_name,
       w.country,w.lat,w.long AS longitude,
       s.top_md,s.base_md,s.depth_type,s.depth_unit,s.depth_datum,
       (s.top_md IS NOT NULL AND s.base_md IS NOT NULL
        AND s.top_md NOT IN ('NaN'::float8,'Infinity'::float8,'-Infinity'::float8)
        AND s.base_md NOT IN ('NaN'::float8,'Infinity'::float8,'-Infinity'::float8)
        AND s.top_md>=0 AND s.base_md>=s.top_md) AS depth_interval_valid,
       CASE WHEN r.value_int IS NOT NULL THEN r.value_int::numeric
            WHEN r.value_float IS NOT NULL THEN r.value_float::numeric END AS value_numeric,
       sr.import_batch_id,sr.row_number,b.source_name,b.source_sha256
FROM geochem.v_analysis_result_semantic r
JOIN geochem.v_sample_wellbore s USING(sample_id)
LEFT JOIN geochem.well w ON w.well_id=s.well_id
JOIN geochem.source_record sr ON sr.source_record_id=r.source_record_id
JOIN geochem.import_batch b ON b.import_batch_id=sr.import_batch_id;

-- One row per run, including runs with no results. JSON keys are source_field_id,
-- not indicator names: several source fields may map to one canonical indicator.
CREATE OR REPLACE VIEW geochem.v_geochemical_run AS
SELECT a.analysis_run_id,a.sample_id,s.name AS sample_name,a.source_record_id,
       sr.import_batch_id,sr.row_number,a.group_name,a.repeat_number,a.analyzed_on,
       s.well_id,s.wellbore_id,s.wellbore_name,w.well_name,w.area,
       s.top_md,s.base_md,s.depth_type,s.depth_unit,s.depth_datum,
       coalesce(x.result_count,0) AS result_count,
       coalesce(x.values_by_field,'{}'::jsonb) AS values_by_field
FROM geochem.analysis_run a
JOIN geochem.v_sample_wellbore s USING(sample_id)
JOIN geochem.source_record sr ON sr.source_record_id=a.source_record_id
LEFT JOIN geochem.well w ON w.well_id=s.well_id
LEFT JOIN LATERAL (
    SELECT count(*) AS result_count,
           jsonb_object_agg(r.source_field_id::text,
               coalesce(to_jsonb(r.value_float),to_jsonb(r.value_int),to_jsonb(r.value_text))) AS values_by_field
    FROM geochem.analysis_result r WHERE r.analysis_run_id=a.analysis_run_id
) x ON true;

-- Only exact source labels in their original units; no inter-method conversion.
CREATE OR REPLACE VIEW geochem.v_pyrolysis_report AS
SELECT a.analysis_run_id,a.sample_id,s.name AS sample_name,a.source_record_id,
       sr.import_batch_id,sr.row_number,a.group_name,a.repeat_number,
       s.well_id,s.wellbore_id,s.wellbore_name,w.well_name,w.area,
       s.top_md,s.base_md,s.depth_type,s.depth_unit,s.depth_datum,
       x.toc_wt_pct,x.s1_mg_g,x.s2_mg_g,x.s3_mg_g,x.tmax_c,x.hi_mg_g,x.oi_mg_g
FROM geochem.analysis_run a
JOIN geochem.v_sample_wellbore s USING(sample_id)
JOIN geochem.source_record sr ON sr.source_record_id=a.source_record_id
LEFT JOIN geochem.well w ON w.well_id=s.well_id
LEFT JOIN LATERAL (
    SELECT max(r.value_float) FILTER (WHERE f.source_label='TOC [%wt]') AS toc_wt_pct,
           max(r.value_float) FILTER (WHERE f.source_label='S1 [ppt (mg/g)]') AS s1_mg_g,
           max(r.value_float) FILTER (WHERE f.source_label='S2 [ppt (mg/g)]') AS s2_mg_g,
           max(r.value_float) FILTER (WHERE f.source_label='S3 [ppt (mg/g)]') AS s3_mg_g,
           max(r.value_int) FILTER (WHERE f.source_label='T-max [°C]') AS tmax_c,
           max(r.value_float) FILTER (WHERE f.source_label='HI [ppt (mg/g) (a/b)]') AS hi_mg_g,
           max(r.value_float) FILTER (WHERE f.source_label='OI [ppt (mg/g) (a/b)]') AS oi_mg_g
    FROM geochem.analysis_result r
    JOIN geochem.source_field f USING(source_field_id,group_name,source_dtype)
    WHERE r.analysis_run_id=a.analysis_run_id
) x ON true
WHERE a.group_name LIKE 'Pyrol%';

-- Availability, not pooled scientific statistics. Source fields and groups stay separate.
CREATE OR REPLACE VIEW geochem.v_geochemical_availability AS
SELECT wellbore_id,wellbore_name,well_id,well_name,area,import_batch_id,
       group_name,source_field_id,source_label,source_dtype,unit_label,
       indicator_id,indicator_name,method_id,method_name,unit_id,unit_symbol,
       depth_type,depth_unit,depth_datum,
       count(*) AS result_count,count(value_numeric) AS numeric_result_count,
       count(DISTINCT sample_id) AS sample_count,
       count(DISTINCT analysis_run_id) AS run_count,
       min(top_md) FILTER (WHERE depth_interval_valid) AS min_top_md,
       max(base_md) FILTER (WHERE depth_interval_valid) AS max_base_md
FROM geochem.v_geochemical_measurement
GROUP BY wellbore_id,wellbore_name,well_id,well_name,area,import_batch_id,
         group_name,source_field_id,source_label,source_dtype,unit_label,
         indicator_id,indicator_name,method_id,method_name,unit_id,unit_symbol,
         depth_type,depth_unit,depth_datum;

DO $$ BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname='readonly') THEN
        GRANT SELECT ON geochem.v_geochemical_measurement,geochem.v_geochemical_run,
            geochem.v_pyrolysis_report,geochem.v_geochemical_availability TO readonly;
    END IF;
END $$;
COMMIT;
