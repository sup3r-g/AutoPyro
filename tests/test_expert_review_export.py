from tools.export_interpretation_expert_review import retain_previous_evidence, update_decision_sheet


def test_reexport_preserves_expert_text_and_adds_new_revision_once(tmp_path):
    path = tmp_path / 'decisions.md'
    original = dict(name='Scheme', revision='1', sha256='a' * 64, cases=[{}])
    update_decision_sheet(path, [original])
    expert_text = path.read_text(encoding='utf-8').replace('Ожидает', 'Принято')
    revised = dict(original, revision='2', sha256='b' * 64)
    expert_text += f"\nКомментарий эксперта с упоминанием новой суммы {revised['sha256']}\n"
    path.write_text(expert_text, encoding='utf-8')
    before = path.read_bytes()

    update_decision_sheet(path, [original])
    assert path.read_bytes() == before
    update_decision_sheet(path, [original, revised])
    after = path.read_bytes()
    assert after.startswith(before)
    assert path.read_text(encoding='utf-8').count('| Ожидает |') == 1
    update_decision_sheet(path, [revised])
    assert path.read_bytes() == after


def test_reexport_retains_exact_evidence_for_previous_expert_decision(tmp_path):
    import json

    old = json.dumps({'schemes': [{'sha256': 'a' * 64, 'actual': ['classified']}]}).encode()
    new = json.dumps({'schemes': [{'sha256': 'b' * 64, 'actual': ['ambiguous']}]}).encode()
    evidence = tmp_path / '02-technical-evidence.json'
    evidence.write_bytes(old)
    (tmp_path / '01-blind-inputs.json').write_bytes(b'previous inputs')
    (tmp_path / '04-sources.md').write_bytes(b'previous sources')
    files = {'02-technical-evidence.json': old, '01-blind-inputs.json': b'previous inputs',
             '04-sources.md': b'previous sources'}
    retain_previous_evidence(tmp_path, files)
    assert not (tmp_path / 'history').exists()

    files[evidence.name] = new
    retain_previous_evidence(tmp_path, files)
    archived, = (tmp_path / 'history').iterdir()
    assert (archived / evidence.name).read_bytes() == old
    assert (archived / '01-blind-inputs.json').read_bytes() == b'previous inputs'
    assert (archived / '04-sources.md').read_bytes() == b'previous sources'
    evidence.write_bytes(new)
    retain_previous_evidence(tmp_path, files)
    assert list((tmp_path / 'history').iterdir()) == [archived]


def test_source_only_updates_preserve_each_previous_register(tmp_path):
    files = {'02-technical-evidence.json': b'unchanged results',
             '01-blind-inputs.json': b'unchanged inputs', '04-sources.md': b'original sources'}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    for sources in (b'first correction', b'second correction'):
        files['04-sources.md'] = sources
        retain_previous_evidence(tmp_path, files)
        (tmp_path / '04-sources.md').write_bytes(sources)
    snapshots = list((tmp_path / 'history').iterdir())
    assert len(snapshots) == 2
    assert {path.joinpath('04-sources.md').read_bytes() for path in snapshots} == {
        b'original sources', b'first correction',
    }
    assert all(path.joinpath('02-technical-evidence.json').read_bytes() == b'unchanged results'
               for path in snapshots)
    retain_previous_evidence(tmp_path, files)
    assert len(list((tmp_path / 'history').iterdir())) == 2


def test_failed_decision_sheet_replace_preserves_expert_conclusion(tmp_path, monkeypatch):
    import pytest
    from autopyro.data_io import file_output

    path = tmp_path / 'decisions.md'
    original = dict(name='Scheme', revision='1', sha256='a' * 64, cases=[{}])
    update_decision_sheet(path, [original])
    content = path.read_bytes().replace('Ожидает'.encode(), 'Принято экспертом'.encode())
    path.write_bytes(content)
    revised = dict(original, revision='2', sha256='b' * 64)
    replace = file_output.os.replace

    def fail_replace(*args):
        raise OSError('Disk error')

    monkeypatch.setattr(file_output.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='Disk error'):
        update_decision_sheet(path, [original, revised])
    assert path.read_bytes() == content
    assert list(tmp_path.iterdir()) == [path]
    monkeypatch.setattr(file_output.os, 'replace', replace)
    update_decision_sheet(path, [original, revised])
    assert path.read_bytes().startswith(content)
    assert path.read_text(encoding='utf-8').count('| Ожидает |') == 1


def test_saved_packet_replay_detects_modified_result_and_geometry(tmp_path, monkeypatch):
    import json
    import tools.export_interpretation_expert_review as exporter
    from autopyro.schemes.standard_interpretation import standard_schemes

    scheme = standard_schemes()['S2–TOC']
    monkeypatch.setattr(exporter, 'standard_schemes', lambda: {'S2–TOC': scheme})
    _, records = exporter.review_packets()
    path = tmp_path / 'evidence.json'

    def verify():
        path.write_text(json.dumps(dict(format_version=1, schemes=records)), encoding='utf-8')
        return exporter.verify_evidence(path)

    assert verify() == []
    # Replay uses the frozen scheme even when the installed catalog changes.
    monkeypatch.setattr(exporter, 'standard_schemes', lambda: {})
    assert verify() == []
    assert records[0]['actual'][0]['value_bounds'] is None
    del records[0]['actual'][0]['value_bounds']
    assert any('отсутствуют поля value_bounds' in failure for failure in verify())
    records[0]['actual'][0]['value_bounds'] = None
    records[0]['actual'][0]['unknown_field'] = None
    assert any('неизвестные поля unknown_field' in failure for failure in verify())
    del records[0]['actual'][0]['unknown_field']
    assert verify() == []
    records[0]['actual'][0]['status'] = 'outside'
    assert any('status' in failure for failure in verify())
    records[0]['scheme']['tolerance'] = .01
    assert any('контрольная сумма' in failure for failure in verify())


def test_replay_rejects_missing_duplicate_and_empty_scheme_sets(tmp_path, monkeypatch):
    import json
    from copy import deepcopy
    import tools.export_interpretation_expert_review as exporter
    from autopyro.schemes.standard_interpretation import standard_schemes

    schemes = standard_schemes()
    monkeypatch.setattr(exporter, 'standard_schemes',
                        lambda: {name: schemes[name] for name in ('S1–TOC', 'S2–TOC')})
    blind, records = exporter.review_packets()
    manifest = [dict(id=item['id'], revision=item['revision'], sha256=item['sha256'],
                     cases=len(item['cases'])) for item in blind]
    path = tmp_path / 'evidence.json'

    def verify(items):
        path.write_text(json.dumps(dict(format_version=2, manifest=manifest, schemes=items)), encoding='utf-8')
        return exporter.verify_evidence(path)

    assert verify(records) == []
    assert verify([])
    assert any('описи' in failure for failure in verify(records[:1]))
    assert any('повторяются' in failure for failure in verify([records[0], records[0]]))
    incomplete = deepcopy(records)
    incomplete[0]['scheme']['reference_cases'] = []
    incomplete[0]['sha256'] = exporter.scheme_hash(incomplete[0]['scheme'])
    incomplete[0]['actual'] = []
    assert any('отсутствуют эталоны' in failure for failure in verify(incomplete))
