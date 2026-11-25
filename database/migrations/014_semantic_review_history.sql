BEGIN;
CREATE TABLE IF NOT EXISTS geochem.semantic_review_batch (
    semantic_review_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_sha256 text NOT NULL UNIQUE CHECK (document_sha256 ~ '^[0-9a-f]{64}$'),
    reviewer text NOT NULL CHECK (btrim(reviewer)<>''),
    evidence text NOT NULL CHECK (btrim(evidence)<>''),
    reviewed_at timestamptz NOT NULL DEFAULT current_timestamp,
    applied_by text NOT NULL DEFAULT current_user,
    decision_document jsonb NOT NULL CHECK (jsonb_typeof(decision_document)='object')
);
CREATE TABLE IF NOT EXISTS geochem.semantic_review_change (
    semantic_review_id bigint NOT NULL REFERENCES geochem.semantic_review_batch(semantic_review_id),
    object_kind text NOT NULL CHECK (object_kind IN ('indicator','unit','method','mapping')),
    object_key text NOT NULL,
    before_state jsonb CHECK (before_state IS NULL OR jsonb_typeof(before_state)='object'),
    after_state jsonb NOT NULL CHECK (jsonb_typeof(after_state)='object'),
    PRIMARY KEY(semantic_review_id,object_kind,object_key)
);
CREATE INDEX IF NOT EXISTS semantic_review_object_idx
    ON geochem.semantic_review_change(object_kind,object_key,semantic_review_id);
CREATE OR REPLACE FUNCTION geochem.guard_semantic_review_history() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Review history cannot be updated or deleted; record a new review';
END $$;
CREATE OR REPLACE TRIGGER guard_semantic_review_batch
BEFORE UPDATE OR DELETE ON geochem.semantic_review_batch
FOR EACH ROW EXECUTE FUNCTION geochem.guard_semantic_review_history();
CREATE OR REPLACE TRIGGER guard_semantic_review_change
BEFORE UPDATE OR DELETE ON geochem.semantic_review_change
FOR EACH ROW EXECUTE FUNCTION geochem.guard_semantic_review_history();
CREATE OR REPLACE VIEW geochem.v_semantic_review_history AS
SELECT b.semantic_review_id,b.reviewer,b.evidence,b.reviewed_at,b.applied_by,
       b.document_sha256,c.object_kind,c.object_key,c.before_state,c.after_state
FROM geochem.semantic_review_batch b JOIN geochem.semantic_review_change c USING(semantic_review_id);
DO $$ BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname='readonly') THEN
        GRANT SELECT ON geochem.semantic_review_batch,geochem.semantic_review_change,
            geochem.v_semantic_review_history TO readonly;
    END IF;
END $$;
COMMIT;
