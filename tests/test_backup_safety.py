import pytest

from autopyro.database.tools.backup_database import check_restore_name


@pytest.mark.parametrize('name',['geochemistry','postgres','template0','geochem_restorecheck_',
    'geochem_restorecheck_abcdefghijklmnop','geochem_restorecheck_0123456789abcdef;DROP DATABASE geochemistry'])
def test_cleanup_rejects_unowned_or_malformed_names(name):
    with pytest.raises(ValueError):
        check_restore_name(name,'geochemistry')


def test_cleanup_rejects_source_even_if_it_has_temporary_shape():
    name='geochem_restorecheck_0123456789abcdef'
    with pytest.raises(ValueError):
        check_restore_name(name,name)


def test_cleanup_accepts_only_expected_generated_namespace():
    check_restore_name('geochem_restorecheck_0123456789abcdef','geochemistry')
