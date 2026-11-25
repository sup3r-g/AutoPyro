-- Avoid joining source_field twice. FK constraints guarantee field/group/dtype
-- identity; exposing the field id directly also lets selective sample predicates
-- reach the result joins before the planner's default join-collapse limit.
BEGIN;
CREATE OR REPLACE VIEW geochem.v_analysis_result_semantic AS
SELECT a.sample_id,a.source_record_id,a.analysis_run_id,a.repeat_number,
       a.analyzed_on,a.group_name,f.source_label,f.unit_label,r.source_dtype,
       r.value_float,r.value_int,r.value_text,r.source_field_id,
       s.indicator_id,i.code AS indicator_code,i.name AS indicator_name,
       s.unit_id,u.symbol AS unit_symbol,s.method_id,m.name AS method_name,
       s.reviewed AS mapping_reviewed,i.reviewed AS indicator_reviewed,
       u.reviewed AS unit_reviewed,m.reviewed AS method_reviewed
FROM geochem.analysis_run a
JOIN geochem.analysis_result r USING(analysis_run_id,group_name)
JOIN geochem.source_field f USING(source_field_id,group_name,source_dtype)
LEFT JOIN geochem.source_field_semantics s USING(source_field_id)
LEFT JOIN geochem.indicator i USING(indicator_id)
LEFT JOIN geochem.measurement_unit u USING(unit_id)
LEFT JOIN geochem.analysis_method m USING(method_id);
COMMIT;
