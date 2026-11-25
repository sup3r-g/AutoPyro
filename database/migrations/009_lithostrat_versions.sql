-- Requires 006. Published means frozen, not scientifically approved.
BEGIN;
CREATE TABLE IF NOT EXISTS geochem.lithostrat_interpretation (
    interpretation_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    wellbore_id bigint NOT NULL REFERENCES geochem.wellbore(wellbore_id),
    version_tag text NOT NULL CHECK (btrim(version_tag)<>''),
    parent_interpretation_id bigint,
    status text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','published')),
    author text NOT NULL CHECK (btrim(author)<>''),
    description text NOT NULL CHECK (btrim(description)<>''),
    created_at timestamptz NOT NULL DEFAULT current_timestamp,
    published_at timestamptz,
    source_import_batch_id bigint REFERENCES geochem.import_batch(import_batch_id),
    source_lithostrat_wellbore_id bigint REFERENCES geochem.lithostrat_wellbore(lithostrat_wellbore_id),
    UNIQUE(wellbore_id,version_tag),
    UNIQUE(interpretation_id,wellbore_id),
    UNIQUE(source_import_batch_id,source_lithostrat_wellbore_id),
    FOREIGN KEY(parent_interpretation_id,wellbore_id)
        REFERENCES geochem.lithostrat_interpretation(interpretation_id,wellbore_id),
    CHECK (parent_interpretation_id IS DISTINCT FROM interpretation_id),
    CHECK ((source_import_batch_id IS NULL)=(source_lithostrat_wellbore_id IS NULL)),
    CHECK ((status='published')=(published_at IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS geochem.lithostrat_interpretation_interval (
    interpretation_interval_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    interpretation_id bigint NOT NULL REFERENCES geochem.lithostrat_interpretation(interpretation_id),
    interval_number integer NOT NULL CHECK (interval_number>0),
    lithostrat_unit_id bigint REFERENCES geochem.lithostrat_unit(lithostrat_unit_id),
    -- Snapshot labels avoid changing old interpretations when a dictionary is edited.
    unit_name text NOT NULL CHECK (btrim(unit_name)<>''),
    unit_level text NOT NULL CHECK (unit_level IN ('GROUP','SUBGROUP','FORMATION','MEMBER')),
    top_depth_m numeric NOT NULL CHECK (top_depth_m>=0 AND top_depth_m<'Infinity'::numeric),
    bottom_depth_m numeric NOT NULL CHECK (bottom_depth_m>top_depth_m AND bottom_depth_m<'Infinity'::numeric),
    thickness_m numeric GENERATED ALWAYS AS (bottom_depth_m-top_depth_m) STORED,
    depth_type text CHECK (depth_type IN ('MD','TVD','TVDSS')),
    depth_datum text,
    depth_reference text,
    source_interval_id bigint REFERENCES geochem.wellbore_lithostrat_interval(wellbore_lithostrat_interval_id),
    note text,
    UNIQUE(interpretation_id,interval_number)
);
CREATE INDEX IF NOT EXISTS interpretation_depth_idx ON geochem.lithostrat_interpretation_interval(interpretation_id,top_depth_m,bottom_depth_m);
CREATE INDEX IF NOT EXISTS interpretation_source_interval_idx ON geochem.lithostrat_interpretation_interval(source_interval_id);

CREATE OR REPLACE FUNCTION geochem.guard_lithostrat_interpretation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP='INSERT' THEN
        IF NEW.status<>'draft' THEN RAISE EXCEPTION 'Create a draft before publication'; END IF;
    ELSE
        IF OLD.status='published' THEN RAISE EXCEPTION 'Published interpretations are immutable; clone a new version'; END IF;
        IF TG_OP='DELETE' THEN RETURN OLD; END IF;
        IF ROW(NEW.interpretation_id,NEW.wellbore_id,NEW.parent_interpretation_id,
               NEW.source_import_batch_id,NEW.source_lithostrat_wellbore_id)
           IS DISTINCT FROM ROW(OLD.interpretation_id,OLD.wellbore_id,OLD.parent_interpretation_id,
               OLD.source_import_batch_id,OLD.source_lithostrat_wellbore_id) THEN
            RAISE EXCEPTION 'Version identity, lineage and source cannot be changed';
        END IF;
    END IF;
    IF NEW.parent_interpretation_id IS NOT NULL AND NOT EXISTS (
        SELECT FROM geochem.lithostrat_interpretation WHERE interpretation_id=NEW.parent_interpretation_id AND status='published') THEN
        RAISE EXCEPTION 'Parent interpretation must be published';
    END IF;
    IF NEW.status='published' AND NOT EXISTS (
        SELECT FROM geochem.lithostrat_interpretation_interval WHERE interpretation_id=NEW.interpretation_id) THEN
        RAISE EXCEPTION 'Cannot publish an empty interpretation';
    END IF;
    RETURN NEW;
END $$;
CREATE OR REPLACE TRIGGER guard_interpretation
BEFORE INSERT OR UPDATE OR DELETE ON geochem.lithostrat_interpretation
FOR EACH ROW EXECUTE FUNCTION geochem.guard_lithostrat_interpretation();

CREATE OR REPLACE FUNCTION geochem.guard_lithostrat_interpretation_interval() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE target_id bigint; version_status text;
BEGIN
    IF TG_OP='UPDATE' AND NEW.interpretation_id<>OLD.interpretation_id THEN
        RAISE EXCEPTION 'An interval cannot be moved between versions';
    END IF;
    IF TG_OP='DELETE' THEN target_id:=OLD.interpretation_id; ELSE target_id:=NEW.interpretation_id; END IF;
    -- Serializes interval edits against publication of their header.
    SELECT status INTO version_status FROM geochem.lithostrat_interpretation
    WHERE interpretation_id=target_id FOR UPDATE;
    IF version_status IS DISTINCT FROM 'draft' THEN
        RAISE EXCEPTION 'Only draft intervals can be edited';
    END IF;
    IF TG_OP='DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END $$;
CREATE OR REPLACE TRIGGER guard_interpretation_interval
BEFORE INSERT OR UPDATE OR DELETE ON geochem.lithostrat_interpretation_interval
FOR EACH ROW EXECUTE FUNCTION geochem.guard_lithostrat_interpretation_interval();

CREATE OR REPLACE FUNCTION geochem.snapshot_lithostrat_sources() RETURNS void
LANGUAGE plpgsql AS $$
DECLARE v record; new_id bigint;
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended('geochem:lithostrat-snapshot',0));
    FOR v IN SELECT DISTINCT i.import_batch_id,i.lithostrat_wellbore_id,m.wellbore_id
        FROM geochem.wellbore_lithostrat_interval i
        JOIN geochem.lithostrat_wellbore_link m USING(lithostrat_wellbore_id)
    LOOP
        INSERT INTO geochem.lithostrat_interpretation(wellbore_id,version_tag,author,description,
            source_import_batch_id,source_lithostrat_wellbore_id)
        VALUES(v.wellbore_id,'source:' || v.import_batch_id || ':' || v.lithostrat_wellbore_id,
            'source import','Unmodified source snapshot; conflicts retained, depth system unknown',
            v.import_batch_id,v.lithostrat_wellbore_id)
        ON CONFLICT(source_import_batch_id,source_lithostrat_wellbore_id) DO NOTHING
        RETURNING interpretation_id INTO new_id;
        IF new_id IS NOT NULL THEN
            INSERT INTO geochem.lithostrat_interpretation_interval(interpretation_id,interval_number,
                lithostrat_unit_id,unit_name,unit_level,top_depth_m,bottom_depth_m,depth_reference,source_interval_id)
            SELECT new_id,row_number() OVER (ORDER BY i.source_sheet,i.source_row)::integer,
                i.lithostrat_unit_id,u.unit_name,u.unit_level,i.top_depth_m,i.bottom_depth_m,
                i.depth_reference,i.wellbore_lithostrat_interval_id
            FROM geochem.wellbore_lithostrat_interval i JOIN geochem.lithostrat_unit u USING(lithostrat_unit_id)
            WHERE i.import_batch_id=v.import_batch_id AND i.lithostrat_wellbore_id=v.lithostrat_wellbore_id;
            UPDATE geochem.lithostrat_interpretation SET status='published',published_at=current_timestamp
            WHERE interpretation_id=new_id;
        END IF;
    END LOOP;
END $$;

CREATE OR REPLACE FUNCTION geochem.clone_lithostrat_interpretation(
    source_id bigint,new_tag text,new_author text,new_description text) RETURNS bigint
LANGUAGE plpgsql AS $$
DECLARE new_id bigint; source_wellbore bigint;
BEGIN
    SELECT wellbore_id INTO source_wellbore FROM geochem.lithostrat_interpretation
    WHERE interpretation_id=source_id AND status='published' FOR SHARE;
    IF NOT FOUND THEN RAISE EXCEPTION 'Source interpretation must exist and be published'; END IF;
    INSERT INTO geochem.lithostrat_interpretation(wellbore_id,version_tag,parent_interpretation_id,author,description)
    VALUES(source_wellbore,new_tag,source_id,new_author,new_description) RETURNING interpretation_id INTO new_id;
    INSERT INTO geochem.lithostrat_interpretation_interval(interpretation_id,interval_number,
        lithostrat_unit_id,unit_name,unit_level,top_depth_m,bottom_depth_m,depth_type,depth_datum,
        depth_reference,source_interval_id,note)
    SELECT new_id,interval_number,lithostrat_unit_id,unit_name,unit_level,top_depth_m,bottom_depth_m,
        depth_type,depth_datum,depth_reference,source_interval_id,note
    FROM geochem.lithostrat_interpretation_interval WHERE interpretation_id=source_id;
    RETURN new_id;
END $$;
SELECT geochem.snapshot_lithostrat_sources();

CREATE OR REPLACE VIEW geochem.v_lithostrat_interpretation AS
SELECT v.*,b.name AS wellbore_name,i.interpretation_interval_id,i.interval_number,
       i.lithostrat_unit_id,i.unit_name,i.unit_level,i.top_depth_m,i.bottom_depth_m,
       i.thickness_m,i.depth_type,i.depth_datum,i.depth_reference,i.source_interval_id,i.note AS interval_note
FROM geochem.lithostrat_interpretation v
JOIN geochem.wellbore b USING(wellbore_id)
JOIN geochem.lithostrat_interpretation_interval i USING(interpretation_id);

CREATE OR REPLACE VIEW geochem.v_lithostrat_interpretation_overlap AS
SELECT a.interpretation_id,a.interpretation_interval_id AS interval_a_id,
       b.interpretation_interval_id AS interval_b_id,a.unit_level,
       a.unit_name AS unit_a,b.unit_name AS unit_b,
       greatest(a.top_depth_m,b.top_depth_m) AS overlap_top_m,
       least(a.bottom_depth_m,b.bottom_depth_m) AS overlap_bottom_m
FROM geochem.lithostrat_interpretation_interval a
JOIN geochem.lithostrat_interpretation_interval b ON a.interpretation_id=b.interpretation_id
    AND a.interpretation_interval_id<b.interpretation_interval_id AND a.unit_level=b.unit_level
    AND a.depth_type IS NOT DISTINCT FROM b.depth_type AND a.depth_datum IS NOT DISTINCT FROM b.depth_datum
    AND a.top_depth_m<b.bottom_depth_m AND b.top_depth_m<a.bottom_depth_m;
DO $$ BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname='readonly') THEN
        GRANT SELECT ON geochem.lithostrat_interpretation,geochem.lithostrat_interpretation_interval,
            geochem.v_lithostrat_interpretation,geochem.v_lithostrat_interpretation_overlap TO readonly;
    END IF;
END $$;
COMMIT;
