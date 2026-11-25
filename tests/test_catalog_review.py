from copy import deepcopy

import pytest

from autopyro.database.tools.catalog_review import fingerprint,read_document,validate_document


def decision():
    return {'schema_version':1,'reviewer':'Reviewer','evidence':'Protocol, page 5',
            'catalogs':[], 'mappings':[{'source_field_id':5,'expected':'a'*64,
                'indicator_code':'toc','unit_code':None,'method_code':None,'note':'Checked'}]}


def test_fingerprint_ignores_json_key_order_but_tracks_changes():
    assert fingerprint({'a':1,'b':None})==fingerprint({'b':None,'a':1})
    assert fingerprint({'a':1})!=fingerprint({'a':2})


@pytest.mark.parametrize('key',['reviewer','evidence'])
def test_missing_expert_attribution_is_rejected(key):
    doc=decision()
    doc[key]=' '
    with pytest.raises(ValueError):
        validate_document(doc)


def test_unknown_keys_are_rejected_instead_of_silently_ignored():
    doc=decision()
    doc['mappings'][0]['approve_everything']=True
    with pytest.raises(ValueError):
        validate_document(doc)


def test_duplicate_mapping_is_rejected():
    doc=decision()
    doc['mappings']*=2
    with pytest.raises(ValueError,match='duplicate'):
        validate_document(doc)


def test_empty_exported_template_cannot_approve_anything():
    doc=decision()
    doc['mappings']=[]
    with pytest.raises(ValueError,match='explicit decision'):
        validate_document(doc)


def test_missing_unit_is_explicitly_allowed_without_inventing_a_unit():
    validate_document(decision())


def test_source_catalog_namespace_cannot_be_used_for_new_canonical_entry():
    doc=decision()
    doc['catalogs']=[{'kind':'indicator','code':'source-field:99','expected':None,'label':'X','description':'Evidence'}]
    with pytest.raises(ValueError,match='reserved'):
        validate_document(doc)


def test_creation_requires_definition():
    doc=decision()
    doc['catalogs']=[{'kind':'indicator','code':'canonical','expected':None}]
    with pytest.raises(ValueError,match='label and description'):
        validate_document(doc)


def test_boolean_is_not_a_source_field_identifier():
    doc=decision()
    doc['mappings'][0]['source_field_id']=True
    with pytest.raises(ValueError):
        validate_document(doc)


def test_repeated_catalog_decision_is_rejected():
    doc=decision()
    entry={'kind':'unit','code':'unit','expected':None,'label':'m','description':'Metres'}
    doc['catalogs']=[entry,deepcopy(entry)]
    with pytest.raises(ValueError,match='Duplicate catalog'):
        validate_document(doc)


def test_duplicate_json_keys_cannot_silently_override_a_decision(tmp_path):
    path=tmp_path/'review.json'
    path.write_text('{"reviewer":"A","reviewer":"B"}',encoding='utf-8')
    with pytest.raises(ValueError,match='Duplicate JSON key'):
        read_document(path)
