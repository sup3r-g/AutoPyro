-- Requires 007 and 009. Selection is explicit; nothing is selected automatically.
BEGIN;
CREATE TABLE IF NOT EXISTS geochem.wellbore_interpretation_selection (
    wellbore_id bigint PRIMARY KEY REFERENCES geochem.wellbore(wellbore_id),
    interpretation_id bigint NOT NULL,
    selected_by text NOT NULL CHECK (btrim(selected_by)<>''),
    evidence text NOT NULL CHECK (btrim(evidence)<>''),
    selected_at timestamptz NOT NULL DEFAULT current_timestamp,
    FOREIGN KEY(interpretation_id,wellbore_id)
        REFERENCES geochem.lithostrat_interpretation(interpretation_id,wellbore_id)
);
CREATE OR REPLACE FUNCTION geochem.check_selected_interpretation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (SELECT FROM geochem.lithostrat_interpretation
                   WHERE interpretation_id=NEW.interpretation_id AND status='published') THEN
        RAISE EXCEPTION 'Only a published interpretation can be selected';
    END IF;
    NEW.selected_at:=current_timestamp;
    RETURN NEW;
END $$;
CREATE OR REPLACE TRIGGER check_selected_interpretation
BEFORE INSERT OR UPDATE ON geochem.wellbore_interpretation_selection
FOR EACH ROW EXECUTE FUNCTION geochem.check_selected_interpretation();

CREATE OR REPLACE VIEW geochem.v_sample_lithostrat_context AS
SELECT s.*,p.interpretation_id AS selected_interpretation_id,
       p.selected_by,p.selected_at,p.evidence AS selection_evidence,
       (s.top_md IS NOT NULL AND s.base_md IS NOT NULL
        AND s.top_md NOT IN ('NaN'::float8,'Infinity'::float8,'-Infinity'::float8)
        AND s.base_md NOT IN ('NaN'::float8,'Infinity'::float8,'-Infinity'::float8)
        AND s.top_md>=0 AND s.base_md>=s.top_md) AS sample_depth_valid
FROM geochem.v_sample_wellbore s
LEFT JOIN geochem.wellbore_interpretation_selection p USING(wellbore_id);

CREATE OR REPLACE VIEW geochem.v_sample_lithostrat_match AS
SELECT s.sample_id,s.well_id,s.wellbore_id,s.wellbore_name,
       s.selected_interpretation_id AS interpretation_id,
       i.interpretation_interval_id,i.unit_name,i.unit_level,
       s.top_md AS sample_top_md,s.base_md AS sample_base_md,
       i.top_depth_m AS interval_top_md,i.bottom_depth_m AS interval_bottom_md,
       s.depth_type,s.depth_datum,
       CASE WHEN s.top_md::numeric>=i.top_depth_m AND s.base_md::numeric<=i.bottom_depth_m
            THEN 'contained' ELSE 'partial' END AS match_kind,
       CASE WHEN s.top_md=s.base_md THEN NULL::numeric ELSE
           (least(s.base_md::numeric,i.bottom_depth_m)-greatest(s.top_md::numeric,i.top_depth_m))
           /(s.base_md::numeric-s.top_md::numeric) END AS overlap_fraction,
       count(*) OVER (PARTITION BY s.sample_id,i.unit_level) AS matches_at_level
FROM geochem.v_sample_lithostrat_context s
JOIN geochem.lithostrat_interpretation_interval i ON i.interpretation_id=s.selected_interpretation_id
    AND i.depth_type=s.depth_type AND i.depth_datum=s.depth_datum
WHERE s.wellbore_link_reviewed AND s.sample_depth_valid
    AND nullif(btrim(s.depth_datum),'') IS NOT NULL
    AND ((s.top_md=s.base_md AND s.top_md::numeric>=i.top_depth_m AND s.top_md::numeric<i.bottom_depth_m)
      OR (s.top_md<s.base_md AND s.top_md::numeric<i.bottom_depth_m AND s.base_md::numeric>i.top_depth_m));

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
LEFT JOIN LATERAL (
    SELECT count(*) AS match_count,bool_or(matches_at_level>1) AS ambiguous
    FROM geochem.v_sample_lithostrat_match x WHERE x.sample_id=s.sample_id
) m ON true;

-- Multiple rows per result are intentional when levels or alternatives overlap.
CREATE OR REPLACE VIEW geochem.v_geochemical_lithostratigraphy AS
SELECT m.*,x.interpretation_id,x.interpretation_interval_id,x.unit_name,x.unit_level,
       x.interval_top_md,x.interval_bottom_md,x.match_kind,x.overlap_fraction,x.matches_at_level
FROM geochem.v_geochemical_measurement m
JOIN geochem.v_sample_lithostrat_match x USING(sample_id);
DO $$ BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname='readonly') THEN
        GRANT SELECT ON geochem.wellbore_interpretation_selection,
            geochem.v_sample_lithostrat_context,geochem.v_sample_lithostrat_match,
            geochem.v_sample_lithostrat_status,geochem.v_geochemical_lithostratigraphy TO readonly;
    END IF;
END $$;
COMMIT;
