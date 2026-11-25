-- Put selective sample predicates in the first join group. With many dictionary
-- joins, PostgreSQL's join_collapse_limit can otherwise hide them until too late.
BEGIN;
CREATE OR REPLACE VIEW geochem.v_geochemical_measurement AS
SELECT a.sample_id,a.source_record_id,a.analysis_run_id,a.repeat_number,
       a.analyzed_on,a.group_name,f.source_label,f.unit_label,r.source_dtype,
       r.value_float,r.value_int,r.value_text,r.source_field_id,
       z.indicator_id,i.code AS indicator_code,i.name AS indicator_name,
       z.unit_id,u.symbol AS unit_symbol,z.method_id,d.name AS method_name,
       z.reviewed AS mapping_reviewed,i.reviewed AS indicator_reviewed,
       u.reviewed AS unit_reviewed,d.reviewed AS method_reviewed,
       s.name AS sample_name,s.well_id,l.wellbore_id,wb.name AS wellbore_name,
       l.reviewed AS wellbore_link_reviewed,w.well_name,w.area,w.field AS field_name,
       w.country,w.lat,w.long AS longitude,
       s.top_md,s.base_md,'MD'::text AS depth_type,'m'::text AS depth_unit,s.md_datum AS depth_datum,
       (s.top_md IS NOT NULL AND s.base_md IS NOT NULL
        AND s.top_md NOT IN ('NaN'::float8,'Infinity'::float8,'-Infinity'::float8)
        AND s.base_md NOT IN ('NaN'::float8,'Infinity'::float8,'-Infinity'::float8)
        AND s.top_md>=0 AND s.base_md>=s.top_md) AS depth_interval_valid,
       CASE WHEN r.value_int IS NOT NULL THEN r.value_int::numeric
            WHEN r.value_float IS NOT NULL THEN r.value_float::numeric END AS value_numeric,
       sr.import_batch_id,sr.row_number,b.source_name,b.source_sha256
FROM geochem.sample s
JOIN geochem.analysis_run a USING(sample_id)
JOIN geochem.analysis_result r USING(analysis_run_id,group_name)
JOIN geochem.source_field f USING(source_field_id,group_name,source_dtype)
JOIN geochem.source_record sr ON sr.source_record_id=a.source_record_id
JOIN geochem.import_batch b ON b.import_batch_id=sr.import_batch_id
LEFT JOIN geochem.well w ON w.well_id=s.well_id
LEFT JOIN geochem.well_wellbore_link l ON l.well_id=s.well_id
LEFT JOIN geochem.wellbore wb ON wb.wellbore_id=l.wellbore_id
LEFT JOIN geochem.source_field_semantics z USING(source_field_id)
LEFT JOIN geochem.indicator i USING(indicator_id)
LEFT JOIN geochem.measurement_unit u USING(unit_id)
LEFT JOIN geochem.analysis_method d USING(method_id);
COMMIT;
