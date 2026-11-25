-- Additional source metadata found in the Barents Sea Parquet file.
-- No transaction statements: the importer applies this within its transaction.
ALTER TABLE geochem.sample ADD COLUMN IF NOT EXISTS metis_sam_status text;
COMMENT ON COLUMN geochem.sample.metis_sam_status IS 'Metis / Sam Status / string';
INSERT INTO geochem.data_group(group_name, scope) VALUES ('Metis', 'core')
ON CONFLICT (group_name) DO NOTHING;
INSERT INTO geochem.source_field
    (group_name, source_label, source_dtype, target_table, target_column)
VALUES ('Metis', 'Sam Status', 'string', 'sample', 'metis_sam_status')
ON CONFLICT (group_name, source_label) DO NOTHING;
