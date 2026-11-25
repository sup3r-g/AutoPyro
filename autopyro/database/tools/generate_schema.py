"""Generate PostgreSQL DDL and a lossless field catalog from tab-separated dtypes.

Usage: python -m autopyro.database.tools.generate_schema path/to/dtypes.txt
Standard library only. Outputs in the repository database/ directory.
"""
import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "database"
TYPES = {"string": "text", "Int64": "bigint", "Float64": "double precision"}
CORE = {"Well": "well", "Sample": "sample", "Geol": "sample_geology", "Ref": "reference_document"}


def literal(value):
    return "NULL" if value is None else "'" + str(value).replace("'", "''") + "'"


def generate(path):
    rows = []
    seen = set()
    used_columns = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3 or parts[2].strip() not in TYPES:
            raise ValueError(f"Unsupported input at line {line_no}: {line!r}")
        group, label, dtype = (part.strip() for part in parts)
        if (group, label) in seen:
            raise ValueError(f"Duplicate source column: {(group, label)}")
        seen.add((group, label))
        # Only the last, space-separated bracket is a unit. [h], [a], etc. stay in the name.
        match = re.search(r"\s+\[([^\[\]]+)\]$", label)
        unit = match.group(1) if match else None
        name = label[:match.start()] if match else label
        table = CORE.get(group)
        column = None
        if table:
            column = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            column = column or "field"
            if column[0].isdigit():
                column = "f_" + column
            column = column[:48]
            if (table, column) in used_columns:
                column += "_" + hashlib.sha256(label.encode()).hexdigest()[:8]
            used_columns[table, column] = True
        rows.append(dict(group_name=group, source_label=label, source_dtype=dtype,
                         unit_label=unit, target_table=table or "analysis_result",
                         target_column=column, scope="core" if table else "analysis"))

    sql = ["-- Generated from dataframe.dtypes. PostgreSQL 14+. Run once in a new schema.\nBEGIN;\nCREATE SCHEMA geochem;"]
    sql.append("""
CREATE TABLE geochem.import_batch (
    import_batch_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_name text NOT NULL,
    source_sha256 text,
    imported_at timestamptz NOT NULL DEFAULT current_timestamp,
    comment text,
    CHECK (source_sha256 IS NULL OR source_sha256 ~ '^[0-9a-fA-F]{64}$')
);
""")
    for group, table in CORE.items():
        lines = [f"CREATE TABLE geochem.{table} (",
                 f"    {table}_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,"]
        if group == "Sample":
            lines.append("    well_id bigint REFERENCES geochem.well(well_id),")
        if group == "Geol":
            lines.append("    sample_id bigint NOT NULL REFERENCES geochem.sample(sample_id),")
        lines.extend(f'    "{r["target_column"]}" {TYPES[r["source_dtype"]]},'
                     for r in rows if r["group_name"] == group)
        lines[-1] = lines[-1].rstrip(",")
        lines.append(");")
        sql.append("\n".join(lines))
        for r in rows:
            if r["group_name"] == group:
                sql.append(f'COMMENT ON COLUMN geochem.{table}."{r["target_column"]}" IS '
                           + literal(group + " / " + r["source_label"] + " / " + r["source_dtype"]) + ";")

    sql.append("""
-- A source row is provenance, not a claim that sample names are unique.
CREATE TABLE geochem.source_record (
    source_record_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    import_batch_id bigint NOT NULL REFERENCES geochem.import_batch(import_batch_id),
    row_number bigint NOT NULL CHECK (row_number >= 0),
    sample_id bigint NOT NULL REFERENCES geochem.sample(sample_id),
    sample_geology_id bigint,
    -- Optional lossless source cells as [[group, label, value], ...].
    -- Useful for rejected values and reconstruction; not the query store.
    raw_cells jsonb CHECK (raw_cells IS NULL OR jsonb_typeof(raw_cells) = 'array'),
    UNIQUE (import_batch_id, row_number),
    UNIQUE (source_record_id, sample_id)
);
ALTER TABLE geochem.sample_geology
    ADD CONSTRAINT sample_geology_identity UNIQUE (sample_geology_id, sample_id);
ALTER TABLE geochem.source_record
    ADD CONSTRAINT source_geology_sample_fk FOREIGN KEY (sample_geology_id, sample_id)
    REFERENCES geochem.sample_geology(sample_geology_id, sample_id);

CREATE TABLE geochem.source_record_reference (
    source_record_id bigint NOT NULL REFERENCES geochem.source_record(source_record_id),
    reference_document_id bigint NOT NULL REFERENCES geochem.reference_document(reference_document_id),
    PRIMARY KEY (source_record_id, reference_document_id)
);

CREATE TABLE geochem.data_group (
    group_name text PRIMARY KEY,
    scope text NOT NULL CHECK (scope IN ('core', 'analysis')),
    UNIQUE (group_name, scope)
);

-- Exact (group_name, source_label) identifies an imported DataFrame column.
CREATE TABLE geochem.source_field (
    source_field_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    group_name text NOT NULL REFERENCES geochem.data_group(group_name),
    source_label text NOT NULL,
    source_dtype text NOT NULL CHECK (source_dtype IN ('string', 'Int64', 'Float64')),
    unit_label text,
    target_table text NOT NULL,
    target_column text,
    UNIQUE (group_name, source_label),
    UNIQUE (source_field_id, group_name, source_dtype),
    CHECK ((target_table = 'analysis_result' AND target_column IS NULL)
        OR (target_table IN ('well', 'sample', 'sample_geology', 'reference_document')
            AND target_column IS NOT NULL))
);

CREATE TABLE geochem.analysis_run (
    analysis_run_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sample_id bigint NOT NULL REFERENCES geochem.sample(sample_id),
    source_record_id bigint NOT NULL,
    group_name text NOT NULL,
    scope text NOT NULL DEFAULT 'analysis' CHECK (scope = 'analysis'),
    repeat_number integer NOT NULL DEFAULT 1 CHECK (repeat_number > 0),
    -- Optional parsed date; the source 'Date' string remains an analysis_result.
    analyzed_on date,
    FOREIGN KEY (source_record_id, sample_id)
        REFERENCES geochem.source_record(source_record_id, sample_id),
    FOREIGN KEY (group_name, scope) REFERENCES geochem.data_group(group_name, scope),
    UNIQUE (source_record_id, group_name, repeat_number),
    UNIQUE (analysis_run_id, group_name)
);

-- Typed EAV: one value per source field per analysis run.
-- Composite foreign keys prevent mixing groups and mismatching declared dtypes.
CREATE TABLE geochem.analysis_result (
    analysis_run_id bigint NOT NULL,
    source_field_id bigint NOT NULL,
    group_name text NOT NULL,
    source_dtype text NOT NULL,
    value_float double precision,
    value_int bigint,
    value_text text,
    PRIMARY KEY (analysis_run_id, source_field_id),
    FOREIGN KEY (analysis_run_id, group_name)
        REFERENCES geochem.analysis_run(analysis_run_id, group_name),
    FOREIGN KEY (source_field_id, group_name, source_dtype)
        REFERENCES geochem.source_field(source_field_id, group_name, source_dtype),
    CHECK (
        (source_dtype = 'Float64' AND value_float IS NOT NULL AND value_int IS NULL AND value_text IS NULL)
        OR (source_dtype = 'Int64' AND value_int IS NOT NULL AND value_float IS NULL AND value_text IS NULL)
        OR (source_dtype = 'string' AND value_text IS NOT NULL AND value_float IS NULL AND value_int IS NULL)
    ),
    CHECK (value_float IS NULL OR value_float NOT IN
        ('NaN'::double precision, 'Infinity'::double precision, '-Infinity'::double precision))
);

CREATE INDEX sample_well_idx ON geochem.sample(well_id);
CREATE INDEX well_name_idx ON geochem.well(well_name);
CREATE INDEX sample_name_idx ON geochem.sample(name);
CREATE INDEX sample_depth_idx ON geochem.sample(top_md, base_md);
CREATE INDEX geology_sample_idx ON geochem.sample_geology(sample_id);
CREATE INDEX source_sample_idx ON geochem.source_record(sample_id);
CREATE INDEX source_geology_idx ON geochem.source_record(sample_geology_id, sample_id);
CREATE INDEX reference_document_idx ON geochem.source_record_reference(reference_document_id);
CREATE INDEX analysis_sample_group_idx ON geochem.analysis_run(sample_id, group_name);
CREATE INDEX result_field_float_idx ON geochem.analysis_result(source_field_id, value_float)
    WHERE value_float IS NOT NULL;
CREATE INDEX result_field_int_idx ON geochem.analysis_result(source_field_id, value_int)
    WHERE value_int IS NOT NULL;

CREATE VIEW geochem.v_analysis_result AS
SELECT a.sample_id, a.source_record_id, a.analysis_run_id, a.repeat_number,
       a.analyzed_on, a.group_name, f.source_label, f.unit_label, r.source_dtype,
       r.value_float, r.value_int, r.value_text
FROM geochem.analysis_run a
JOIN geochem.analysis_result r USING (analysis_run_id, group_name)
JOIN geochem.source_field f USING (source_field_id, group_name, source_dtype);
""")
    groups = dict((r["group_name"], r["scope"]) for r in rows)
    sql.append("INSERT INTO geochem.data_group (group_name, scope) VALUES\n" +
               ",\n".join("(" + literal(g) + ", " + literal(s) + ")" for g, s in groups.items()) + ";")
    sql.append("INSERT INTO geochem.source_field\n"
               "(group_name, source_label, source_dtype, unit_label, target_table, target_column) VALUES\n" +
               ",\n".join("(" + ", ".join(literal(r[k]) for k in
                    ("group_name", "source_label", "source_dtype", "unit_label", "target_table", "target_column")) + ")"
                    for r in rows) + ";\nCOMMIT;\n")
    (ROOT / "schema.sql").write_text("\n\n".join(sql), encoding="utf-8")
    with (ROOT / "field_catalog.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Fields: {len(rows)}; groups: {len(groups)}; dtypes: {dict(Counter(r['source_dtype'] for r in rows))}")
    print(f"Core fields: {sum(r['scope'] == 'core' for r in rows)}; analysis fields: {sum(r['scope'] == 'analysis' for r in rows)}")


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
