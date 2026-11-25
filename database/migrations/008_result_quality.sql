-- Requires 007. Optional metadata, never inferred from source values.
BEGIN;
CREATE TABLE IF NOT EXISTS geochem.analysis_result_quality (
    analysis_run_id bigint NOT NULL,
    source_field_id bigint NOT NULL,
    qualifier text NOT NULL DEFAULT 'unreported'
        CHECK (qualifier IN ('unreported','measured','below_detection','below_quantification','less_than','greater_than','approximate')),
    source_literal text,
    reporting_bound numeric CHECK (reporting_bound>'-Infinity'::numeric AND reporting_bound<'Infinity'::numeric),
    detection_limit numeric CHECK (detection_limit>=0 AND detection_limit<'Infinity'::numeric),
    quantification_limit numeric CHECK (quantification_limit>=0 AND quantification_limit<'Infinity'::numeric),
    absolute_uncertainty numeric CHECK (absolute_uncertainty>=0 AND absolute_uncertainty<'Infinity'::numeric),
    uncertainty_kind text CHECK (uncertainty_kind IN ('standard','expanded','unspecified')),
    coverage_factor numeric CHECK (coverage_factor>0 AND coverage_factor<'Infinity'::numeric),
    confidence_level numeric CHECK (confidence_level>0 AND confidence_level<1),
    review_status text NOT NULL DEFAULT 'unreviewed'
        CHECK (review_status IN ('unreviewed','accepted','suspect','rejected')),
    reviewed_by text,
    reviewed_at timestamptz,
    evidence text NOT NULL CHECK (btrim(evidence)<>''),
    note text,
    PRIMARY KEY (analysis_run_id,source_field_id),
    FOREIGN KEY (analysis_run_id,source_field_id)
        REFERENCES geochem.analysis_result(analysis_run_id,source_field_id),
    CHECK (detection_limit IS NULL OR quantification_limit IS NULL OR quantification_limit>=detection_limit),
    CHECK ((absolute_uncertainty IS NULL) = (uncertainty_kind IS NULL)),
    CHECK (coverage_factor IS NULL OR (absolute_uncertainty IS NOT NULL AND uncertainty_kind='expanded')),
    CHECK (confidence_level IS NULL OR absolute_uncertainty IS NOT NULL),
    CHECK (review_status='unreviewed' OR (nullif(btrim(reviewed_by),'') IS NOT NULL AND reviewed_at IS NOT NULL)),
    CHECK (qualifier<>'below_detection' OR detection_limit IS NOT NULL),
    CHECK (qualifier<>'below_quantification' OR quantification_limit IS NOT NULL),
    CHECK (qualifier NOT IN ('less_than','greater_than') OR reporting_bound IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS result_quality_status_idx ON geochem.analysis_result_quality(review_status);

CREATE OR REPLACE FUNCTION geochem.check_numeric_result_quality() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (SELECT FROM geochem.analysis_result
               WHERE analysis_run_id=NEW.analysis_run_id AND source_field_id=NEW.source_field_id
                 AND source_dtype='string')
       AND (NEW.qualifier<>'unreported' OR NEW.reporting_bound IS NOT NULL OR NEW.detection_limit IS NOT NULL
            OR NEW.quantification_limit IS NOT NULL OR NEW.absolute_uncertainty IS NOT NULL) THEN
        RAISE EXCEPTION 'Numeric qualifiers, limits and uncertainty require a numeric result'
            USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END $$;
CREATE OR REPLACE TRIGGER numeric_result_quality_check
BEFORE INSERT OR UPDATE ON geochem.analysis_result_quality
FOR EACH ROW EXECUTE FUNCTION geochem.check_numeric_result_quality();

CREATE OR REPLACE VIEW geochem.v_geochemical_measurement_qc AS
SELECT m.*,q.analysis_run_id IS NOT NULL AS has_quality_metadata,
       coalesce(q.qualifier,'unreported') AS qualifier,q.source_literal,
       q.reporting_bound,q.detection_limit,q.quantification_limit,q.absolute_uncertainty,
       q.uncertainty_kind,q.coverage_factor,q.confidence_level,
       coalesce(q.review_status,'unreviewed') AS review_status,
       q.reviewed_by,q.reviewed_at,q.evidence AS quality_evidence,q.note AS quality_note
FROM geochem.v_geochemical_measurement m
LEFT JOIN geochem.analysis_result_quality q USING(analysis_run_id,source_field_id);

CREATE OR REPLACE VIEW geochem.v_geochemical_quality_summary AS
SELECT import_batch_id,group_name,source_field_id,source_label,unit_label,
       count(*) AS result_count,
       count(*) FILTER (WHERE has_quality_metadata) AS annotated_count,
       count(*) FILTER (WHERE review_status='unreviewed') AS unreviewed_count,
       count(*) FILTER (WHERE review_status='accepted') AS accepted_count,
       count(*) FILTER (WHERE review_status='suspect') AS suspect_count,
       count(*) FILTER (WHERE review_status='rejected') AS rejected_count,
       count(*) FILTER (WHERE qualifier IN ('below_detection','below_quantification','less_than','greater_than')) AS censored_count,
       count(*) FILTER (WHERE absolute_uncertainty IS NOT NULL) AS uncertainty_count
FROM geochem.v_geochemical_measurement_qc
GROUP BY import_batch_id,group_name,source_field_id,source_label,unit_label;

DO $$ BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname='readonly') THEN
        GRANT SELECT ON geochem.analysis_result_quality,geochem.v_geochemical_measurement_qc,
            geochem.v_geochemical_quality_summary TO readonly;
    END IF;
END $$;
COMMIT;
