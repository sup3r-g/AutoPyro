BEGIN;

CREATE OR REPLACE VIEW geochem.v_lithostrat_depth_review AS
SELECT v.interpretation_id,v.version_tag,v.wellbore_id,v.wellbore_name,
       v.interpretation_interval_id,v.interval_number,v.unit_name,v.unit_level,
       v.top_depth_m,v.bottom_depth_m,v.depth_type,v.depth_datum,v.depth_reference,
       v.depth_type IS NULL AS needs_depth_type,
       nullif(btrim(v.depth_datum),'') IS NULL AS needs_depth_datum,
       coalesce(p.interpretation_id=v.interpretation_id,false) AS is_selected,
       v.source_interval_id,s.import_batch_id,b.source_name,b.source_sha256,
       s.source_sheet,s.source_row
FROM geochem.v_lithostrat_interpretation v
LEFT JOIN geochem.wellbore_interpretation_selection p USING(wellbore_id)
LEFT JOIN geochem.wellbore_lithostrat_interval s
       ON s.wellbore_lithostrat_interval_id=v.source_interval_id
LEFT JOIN geochem.import_batch b ON b.import_batch_id=s.import_batch_id
WHERE v.status='published'
  AND (v.depth_type IS NULL OR nullif(btrim(v.depth_datum),'') IS NULL);

DO $$ BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname='readonly') THEN
        GRANT SELECT ON geochem.v_lithostrat_depth_review TO readonly;
    END IF;
END $$;
COMMIT;
