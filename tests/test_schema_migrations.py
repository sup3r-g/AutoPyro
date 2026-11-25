import pytest

from autopyro.database.tools.migrate import load_migrations


def test_checksum_is_portable_and_wrappers_are_removed(tmp_path):
    path = tmp_path / '011_example.sql'
    path.write_bytes('BEGIN;\r\nSELECT 1; -- тест\r\nCOMMIT;\r\n'.encode('utf-8'))
    first = load_migrations(tmp_path)[0]
    path.write_bytes('BEGIN;\nSELECT 1; -- тест\nCOMMIT;\n'.encode('utf-8'))
    second = load_migrations(tmp_path)[0]
    assert first.checksum == second.checksum
    assert second.statement == 'SELECT 1; -- тест\n'


def test_duplicate_version_is_rejected(tmp_path):
    for name in ['011_a.sql', '011_b.sql']:
        (tmp_path / name).write_text('SELECT 1;', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        load_migrations(tmp_path)


def test_incomplete_transaction_is_rejected(tmp_path):
    (tmp_path / '011_a.sql').write_text('BEGIN;\nSELECT 1;\n', encoding='utf-8')
    with pytest.raises(ValueError, match='transaction wrapper'):
        load_migrations(tmp_path)


def test_empty_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='No migrations'):
        load_migrations(tmp_path)


def test_security_migration_is_separate(tmp_path):
    (tmp_path / '004_readonly_access.sql').write_text('invalid security SQL', encoding='utf-8')
    (tmp_path / '005_a.sql').write_text('SELECT 1;', encoding='utf-8')
    assert [m.name for m in load_migrations(tmp_path)] == ['005_a.sql']


def test_repository_migrations_are_ordered():
    migrations = load_migrations()
    assert len(migrations) >= 8
    assert migrations[0].name == '002_metis_sample_status.sql'
    assert migrations[-1].name == max(m.name for m in migrations)
