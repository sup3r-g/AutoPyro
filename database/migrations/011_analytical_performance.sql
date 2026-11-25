-- Same columns and row semantics as 007/010; no materialized data or refresh step.
BEGIN;
-- Serves the correlated compatibility check and actual depth matching.
CREATE INDEX IF NOT EXISTS interpretation_reference_depth_idx
    ON geochem.lithostrat_interpretation_interval
    (interpretation_id,depth_type,depth_datum,top_depth_m,bottom_depth_m);
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
LEFT JOIN (
    SELECT group_name,
        max(source_field_id) FILTER (WHERE source_label='TOC [%wt]') AS toc_id,
        max(source_field_id) FILTER (WHERE source_label='S1 [ppt (mg/g)]') AS s1_id,
        max(source_field_id) FILTER (WHERE source_label='S2 [ppt (mg/g)]') AS s2_id,
        max(source_field_id) FILTER (WHERE source_label='S3 [ppt (mg/g)]') AS s3_id,
        max(source_field_id) FILTER (WHERE source_label='T-max [°C]') AS tmax_id,
        max(source_field_id) FILTER (WHERE source_label='HI [ppt (mg/g) (a/b)]') AS hi_id,
        max(source_field_id) FILTER (WHERE source_label='OI [ppt (mg/g) (a/b)]') AS oi_id
    FROM geochem.source_field WHERE group_name LIKE 'Pyrol%' GROUP BY group_name
) k ON k.group_name=a.group_name
LEFT JOIN LATERAL (
    SELECT max(r.value_float) FILTER (WHERE r.source_field_id=k.toc_id) AS toc_wt_pct,
           max(r.value_float) FILTER (WHERE r.source_field_id=k.s1_id) AS s1_mg_g,
           max(r.value_float) FILTER (WHERE r.source_field_id=k.s2_id) AS s2_mg_g,
           max(r.value_float) FILTER (WHERE r.source_field_id=k.s3_id) AS s3_mg_g,
           max(r.value_int) FILTER (WHERE r.source_field_id=k.tmax_id) AS tmax_c,
           max(r.value_float) FILTER (WHERE r.source_field_id=k.hi_id) AS hi_mg_g,
           max(r.value_float) FILTER (WHERE r.source_field_id=k.oi_id) AS oi_mg_g
    FROM geochem.analysis_result r WHERE r.analysis_run_id=a.analysis_run_id
      AND r.source_field_id=ANY(ARRAY[k.toc_id,k.s1_id,k.s2_id,k.s3_id,k.tmax_id,k.hi_id,k.oi_id])
) x ON true
WHERE a.group_name LIKE 'Pyrol%';

-- Aggregate using source identity before joining verbose dictionaries.
CREATE OR REPLACE VIEW geochem.v_geochemical_availability AS
SELECT m.wellbore_id,b.name AS wellbore_name,t.well_id,w.well_name,w.area,t.import_batch_id,
       f.group_name,t.source_field_id,f.source_label,f.source_dtype,f.unit_label,
       z.indicator_id,i.name AS indicator_name,z.method_id,d.name AS method_name,z.unit_id,u.symbol AS unit_symbol,
       'MD'::text AS depth_type,'m'::text AS depth_unit,t.md_datum AS depth_datum,
       t.result_count,t.numeric_result_count,t.sample_count,t.run_count,t.min_top_md,t.max_base_md
FROM (
    SELECT s.well_id,s.md_datum,sr.import_batch_id,r.source_field_id,
           count(*) AS result_count,
           count(*) FILTER (WHERE r.value_float IS NOT NULL OR r.value_int IS NOT NULL) AS numeric_result_count,
           count(DISTINCT a.sample_id) AS sample_count,count(DISTINCT a.analysis_run_id) AS run_count,
           min(s.top_md) FILTER (WHERE s.top_md>=0 AND s.base_md>=s.top_md
               AND s.top_md NOT IN ('NaN'::float8,'Infinity'::float8,'-Infinity'::float8)
               AND s.base_md NOT IN ('NaN'::float8,'Infinity'::float8,'-Infinity'::float8)) AS min_top_md,
           max(s.base_md) FILTER (WHERE s.top_md>=0 AND s.base_md>=s.top_md
               AND s.top_md NOT IN ('NaN'::float8,'Infinity'::float8,'-Infinity'::float8)
               AND s.base_md NOT IN ('NaN'::float8,'Infinity'::float8,'-Infinity'::float8)) AS max_base_md
    FROM geochem.analysis_result r
    JOIN geochem.analysis_run a USING(analysis_run_id)
    JOIN geochem.sample s USING(sample_id)
    JOIN geochem.source_record sr ON sr.source_record_id=a.source_record_id
    GROUP BY s.well_id,s.md_datum,sr.import_batch_id,r.source_field_id
) t
JOIN geochem.source_field f USING(source_field_id)
LEFT JOIN geochem.well w ON w.well_id=t.well_id
LEFT JOIN geochem.well_wellbore_link m ON m.well_id=t.well_id
LEFT JOIN geochem.wellbore b USING(wellbore_id)
LEFT JOIN geochem.source_field_semantics z USING(source_field_id)
LEFT JOIN geochem.indicator i USING(indicator_id)
LEFT JOIN geochem.analysis_method d USING(method_id)
LEFT JOIN geochem.measurement_unit u USING(unit_id);

CREATE OR REPLACE VIEW geochem.v_sample_lithostrat_status AS
SELECT s.sample_id,s.name AS sample_name,s.well_id,s.wellbore_id,s.wellbore_name,
       s.selected_interpretation_id,s.top_md,s.base_md,s.depth_type,s.depth_datum,
       coalesce(m.match_count,0) AS match_count,coalesce(m.ambiguous,false) AS ambiguous,
       CASE WHEN s.wellbore_id IS NULL THEN 'no_wellbore'
            WHEN s.selected_interpretation_id IS NULL THEN 'no_selected_version'
            WHEN s.wellbore_link_reviewed IS DISTINCT FROM true THEN 'unreviewed_wellbore_link'
            WHEN NOT s.sample_depth_valid THEN 'invalid_or_missing_sample_depth'
            WHEN nullif(btrim(s.depth_datum),'') IS NULL THEN 'unknown_sample_datum'
            WHEN NOT EXISTS (SELECT FROM geochem.lithostrat_interpretation_interval i
                WHERE i.interpretation_id=s.selected_interpretation_id
                AND i.depth_type=s.depth_type AND i.depth_datum=s.depth_datum)
                THEN 'no_compatible_depth_reference'
            WHEN coalesce(m.match_count,0)=0 THEN 'outside_intervals'
            WHEN m.ambiguous THEN 'ambiguous'
            ELSE 'matched' END AS match_status
FROM geochem.v_sample_lithostrat_context s
LEFT JOIN (
    SELECT sample_id,count(*) AS match_count,bool_or(matches_at_level>1) AS ambiguous
    FROM geochem.v_sample_lithostrat_match GROUP BY sample_id
) m USING(sample_id);
COMMIT;
