import io
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
from docx import Document
from pypdf import PdfReader

from autopyro.io import build_project, infer_kind, read_file
from autopyro.analysis.qc import run_checks
from autopyro.model.core import Column, Project, Row
from autopyro.plotting import PlotSpec
from autopyro.reporting import ReportOptions, build_report, digest, export_report, render_html, render_pdf
from autopyro.reports.reporting import report_blocks
from autopyro.reports.history import compare_snapshots, comparison_text


@pytest.fixture
def data():
    return build_project(["Скважина", "TOC", "HI"],
        [["A", "1", "100"], ["A", "<0.1", "200"], ["B", "ND", "300"], ["B", "3", "400"]], [1, 2, 3, 4],
        [(0, "Скважина", "text", ""), (1, "TOC", "text", "%wt"), (2, "HI", "number", "mg HC/g TOC")],
        {"type": "test", "sha256": "abc"})


def test_report_carries_expert_context_and_rechecks_changed_evidence(data):
    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.interpretation_review import save_review
    from autopyro.schemes.standard_interpretation import standard_schemes

    data.sync_measurements()
    _, x, y = data.columns
    ids = [data.rows[0].id]
    scheme = standard_schemes()["HI–TOC"]
    settings = dict(x_id=x.id, y_id=y.id, x_unit=x.unit, y_unit=y.unit, text_numbers=True)
    data.view.update(interpretation_schemes={scheme["id"]: scheme},
                     interpretation_current=dict(settings, scheme_id=scheme["id"]),
                     interpretation_comparison=True)
    result = evaluate_project(data, ids, scheme=scheme, **settings)
    save_review(data, result, 0, "expert", "Prior conclusion", "First note", author="Reviewer")
    save_review(data, result, 0, "expert", "Expert conclusion on sample", "Rationale", author="Reviewer")
    data.science.samples[ids[0]].metadata["pyrogram_observation"] = "Signal observation"
    options = ReportOptions((y.id,), sections=("interpretation",), language="en")
    report = build_report(data, ids, options)
    interpretation = report["interpretation"]
    assert interpretation["comparison_panels"]
    review, = interpretation["expert_reviews"]
    assert review["state"] == "expert"
    assert any(item["conclusion"] == "Prior conclusion" for item in review["history"])
    html = render_html(report)
    assert "Expert conclusion on sample" in html and "Signal observation" in html

    data.rows[0].values[y.id] = 120
    data.sync_measurements()
    changed = build_report(data, ids, options)
    assert changed["interpretation"]["expert_reviews"][0]["state"] == "stale"
    assert "Previous decision. Review required" in render_html(changed)


def test_report_can_cancel_during_quality_checks(data):
    from autopyro.ui.tasks import TaskCancelled

    def progress(stage, percent):
        if stage == "Checking data quality…" and percent > 75:
            raise TaskCancelled()

    with pytest.raises(TaskCancelled):
        build_report(data, [row.id for row in data.rows],
                     ReportOptions((data.columns[2].id,), language="en"), progress=progress)


def test_long_qc_comment_keeps_identity_in_paged_exports(data, tmp_path):
    data.sync_measurements()
    cid = data.columns[2].id
    measurement = data.science.current(data.rows[0].id, cid)
    comment = " ".join(f"note{index:04d}" for index in range(350))
    data.view["qc_exclusions"] = {"statistics": {
        measurement.id: {"revision": measurement.revision, "comment": comment}
    }}
    report = build_report(data, [row.id for row in data.rows],
                          ReportOptions((cid,), sections=("quality",), language="en"))
    pages = [page.extract_text() for page in PdfReader(io.BytesIO(render_pdf(report))).pages]
    comment_pages = [text for text in pages if "note0" in text]
    assert len(comment_pages) > 1
    assert all("Sample: A" in text.partition("note0")[0]
               and "Indicator: HI" in text.partition("note0")[0] for text in comment_pages)
    assert all(sum(text.count(f"note{index:04d}") for text in pages) == 1 for index in range(350))

    path = tmp_path / "report.docx"
    export_report(path, report, "docx")
    rows = [row for table in Document(path).tables for row in table.rows
            if any(cell.text.startswith("note") for cell in row.cells)]
    assert len(rows) > 1
    assert all("Sample: A" in row.cells[0].text and "Indicator: HI" in row.cells[0].text for row in rows)
    parts = [cell.text for row in rows for cell in row.cells if cell.text.startswith("note")]
    assert "".join(parts) == comment


@pytest.mark.parametrize(("language", "limit"), [("ru", 0), ("en", 1), ("en", 100)])
def test_qc_report_discloses_row_limit_without_truncating_json(data, tmp_path, language, limit):
    data.sync_measurements()
    cid = data.columns[2].id
    measurements = [data.science.current(row.id, cid) for row in data.rows[:2]]
    data.view["qc_exclusions"] = {"statistics": {
        item.id: {"revision": item.revision, "comment": f"Exclusion {n}"}
        for n, item in enumerate(measurements, 1)
    }}
    report = build_report(data, [row.id for row in data.rows],
                          ReportOptions((cid,), data_limit=0, qc_limit=limit, language=language))
    total = len(report["qc"]["issues"])
    assert total > 1 and len(report["qc"]["active_exclusions"]) == 2
    assert any(check["code"] == "truncated_qc" for check in report["release_checks"]) == (limit < total)
    html = render_html(report)
    pdf = " ".join(" ".join(page.extract_text().split())
                   for page in PdfReader(io.BytesIO(render_pdf(report))).pages)
    if limit == 0:
        assert f"Замечания QC: показано 0 из {total}, пропущено {total}." in html
        assert "Действующие исключения QC: показано 0 из 2, пропущено 2." in pdf
        assert "Exclusion 1" not in html
    elif limit == 1:
        assert f"QC findings: 1 of {total} shown, {total - 1} omitted." in pdf
        assert "Active QC exclusions: 1 of 2 shown, 1 omitted." in html
        assert "Exclusion 1" in html and "Exclusion 2" not in html
    else:
        assert "QC row limit" not in html and "QC row limit" not in pdf
        assert "Exclusion 1" in html and "Exclusion 2" in html
    path = tmp_path / "qc.json"
    export_report(path, report, "json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["qc"]["issues"]) == total
    assert len(saved["qc"]["active_exclusions"]) == 2


def test_qc_report_filters_accepted_findings_before_the_row_limit(data, tmp_path):
    data.sync_measurements()
    issues = run_checks(data)
    accepted = issues[0]["id"]
    data.view["qc_decisions"] = {accepted: {"action": "acknowledged", "comment": "Reviewed marker"}}
    report = build_report(data, [row.id for row in data.rows],
                          ReportOptions((data.columns[2].id,), qc_limit=1, language="en", qc_open_only=True))
    pending = len(report["qc"]["issues"]) - 1
    assert pending > 1
    html = render_html(report)
    pdf = " ".join(" ".join(page.extract_text().split())
                   for page in PdfReader(io.BytesIO(render_pdf(report))).pages)
    for output in (html, pdf):
        assert f"To review: {pending}." in output
        assert f"QC findings: 1 of {pending} shown, {pending - 1} omitted." in output
        assert "Reviewed marker" not in output
    path = tmp_path / "qc.json"
    export_report(path, report, "json")
    assert any(item["id"] == accepted and item["comment"] == "Reviewed marker"
               for item in json.loads(path.read_text(encoding="utf-8"))["qc"]["issues"])


def test_qc_report_uses_sample_labels_for_findings_and_exclusions():
    columns = [Column("Well"), Column("Sample / Name"), Column("Depth", "number"),
               Column("TOC", "number", unit="%wt")]
    first, duplicate, anonymous, unique = [
        Row(dict(zip((column.id for column in columns), values)))
        for values in (("Well A", "001", 1200, 101), ("Well A", "001", 1200, 101),
                       (None, None, None, 101), ("Well B", "002", 0, 101))
    ]
    project = Project(columns=columns, rows=[first, duplicate, anonymous, unique])
    project.sync_measurements()
    cid = columns[-1].id
    measurement = project.science.current(unique.id, cid)
    project.view["qc_exclusions"] = {"statistics": {
        measurement.id: {"revision": measurement.revision, "comment": "QA exclusion"},
    }}
    report = build_report(project, [row.id for row in project.rows], ReportOptions((cid,), language="en"))
    first.values[columns[0].id] = "Changed well"
    blocks = {title: content for _, _, kind, title, content in report_blocks(report) if kind == "table"}
    findings = {row[1] for row in blocks["Замечания QC"][1]}
    assert {f"Well A · 001 · 1200 · {row.id}" for row in (first, duplicate)} <= findings
    assert anonymous.id in findings and "Well B · 002 · 0" in findings
    assert blocks["Действующие исключения QC"][1][0][0] == "Well B · 002 · 0"
    assert "Well B · 002 · 0" in render_html(report)
    pdf = " ".join(" ".join(page.extract_text().split())
                   for page in PdfReader(io.BytesIO(render_pdf(report))).pages)
    assert "Well B · 002 · 0" in pdf
    assert {item["sample_id"] for item in report["qc"]["issues"]} == {row.id for row in project.rows}


def test_qc_report_distinguishes_excluded_working_measurement_and_repeat(data):
    data.sync_measurements()
    sid, cid = data.rows[0].id, data.columns[2].id
    working = data.science.current(sid, cid)
    repeat = data.science.add_result(sid, cid, 150)
    data.view["qc_exclusions"] = {"statistics": {
        m.id: {"revision": m.revision, "comment": "Review"} for m in (working, repeat)
    }}
    stamp = "2026-10-01T12:34:56+03:00"
    data.view["qc_exclusions"]["statistics"][working.id]["at"] = stamp
    report = build_report(data, [sid], ReportOptions((cid,), language="en"))
    assert {item["measurement_id"]: item["is_working"] for item in report["qc"]["active_exclusions"]} == {
        working.id: True, repeat.id: False,
    }
    headers, rows = next(content for _, _, kind, title, content in report_blocks(report)
                         if kind == "table" and title == "Действующие исключения QC")
    assert headers[4] == "Working result"
    assert {row[3]: row[4] for row in rows} == {working.id: "Yes", repeat.id: "No"}
    assert headers[6] == "Set at"
    formatted = "2026-10-01 09:34:56 UTC"
    assert {row[3]: row[6] for row in rows} == {working.id: formatted, repeat.id: "—"}
    assert formatted in render_html(report)
    pdf = " ".join(" ".join(page.extract_text().split())
                   for page in PdfReader(io.BytesIO(render_pdf(report))).pages)
    assert formatted in pdf and "Set at" in pdf
    assert next(item for item in report["qc"]["active_exclusions"] if item["measurement_id"] == working.id)["at"] == stamp


def test_qc_report_displays_source_coordinates_without_technical_metadata(data):
    cid = data.columns[2].id
    report = build_report(data, [row.id for row in data.rows], ReportOptions((cid,), language="en"))
    finding = next(item for item in report["qc"]["issues"] if item["property_id"] == cid)
    report["qc"]["issues"] = [
        {**finding, "source": {"path": r"C:\internal\Pyro.xlsx", "sheet": "Batch A",
                               "row": 42, "sha256": "technical-marker"}},
        {**finding, "source": {"endpoint": {"database": "geochem"}, "relation": "raw_cells",
                               "row_number": 0}},
        {**finding, "source": {}},
    ]
    headers, rows = next(content for _, _, kind, title, content in report_blocks(report)
                         if kind == "table" and title == "Замечания QC")
    assert headers[2:4] == ["Indicator", "Source"]
    assert [row[3] for row in rows] == [
        "File: Pyro.xlsx · Sheet: Batch A · Row: 42",
        "Database: geochem · Table: raw_cells · Row: 0", "—",
    ]
    pdf = " ".join(" ".join(page.extract_text().split())
                   for page in PdfReader(io.BytesIO(render_pdf(report))).pages)
    assert "Indicator: HI Source: File: Pyro.xlsx · Sheet: Batch A · Row: 42" in pdf
    assert "Database: geochem · Table: raw_cells · Row: 0" in render_html(report)
    assert "technical-marker" not in pdf and "internal" not in pdf
    assert report["qc"]["issues"][0]["source"]["sha256"] == "technical-marker"


def test_qc_report_preserves_decision_timestamp(data, tmp_path):
    cid = data.columns[2].id
    data.sync_measurements()
    issue = next(item for item in run_checks(data) if item["property_id"] == cid)
    stamp = "2026-10-01T12:34:56+03:00"
    data.view["qc_decisions"] = {issue["id"]: {"action": "acknowledged", "comment": "Reviewed", "at": stamp}}
    report = build_report(data, [row.id for row in data.rows], ReportOptions((cid,), language="en"))
    found = next(item for item in report["qc"]["issues"] if item["id"] == issue["id"])
    assert found["decision_at"] == stamp
    for output in (render_html(report), " ".join(" ".join(page.extract_text().split())
                   for page in PdfReader(io.BytesIO(render_pdf(report))).pages)):
        assert "2026-10-01 09:34:56 UTC" in output
    path = tmp_path / "decision.json"
    export_report(path, report, "json")
    assert next(item for item in json.loads(path.read_text(encoding="utf-8"))["qc"]["issues"]
                if item["id"] == issue["id"])["decision_at"] == stamp
    data.view["qc_decisions"][issue["id"]].pop("at")
    legacy = build_report(data, [row.id for row in data.rows], ReportOptions((cid,), language="en"))
    _, rows = next(content for _, _, kind, title, content in report_blocks(legacy)
                   if kind == "table" and title == "Замечания QC")
    assert all(row[6] == "—" for row in rows)


def test_russian_qc_report_translates_severity_and_decision(data):
    cid = data.columns[2].id
    report = build_report(data, [row.id for row in data.rows], ReportOptions((cid,), language="ru"))
    template = report["qc"]["issues"][0]
    report["qc"]["issues"] = [
        {**template, "severity": severity, "decision": decision}
        for severity, decision in (("error", "open"), ("critical", "acknowledged"), ("warning", "open"))
    ]
    _, rows = next(content for _, _, kind, title, content in report_blocks(report)
                   if kind == "table" and title == "Замечания QC")
    assert [(row[0], row[-2]) for row in rows] == [
        ("Ошибка", "Не рассмотрено"), ("Критическое предупреждение", "Принято"), ("Предупреждение", "Не рассмотрено"),
    ]
    for output in (render_html(report), " ".join(" ".join(page.extract_text().split())
                   for page in PdfReader(io.BytesIO(render_pdf(report))).pages)):
        assert "Критическое предупреждение" in output and "Не рассмотрено" in output and "Принято" in output
    assert report["qc"]["issues"][1]["decision"] == "acknowledged"


def test_qc_report_displays_frozen_rule_names_and_versions(data):
    cid = data.columns[2].id
    report = build_report(data, [row.id for row in data.rows], ReportOptions((cid,), language="en"))
    finding = next(item for item in report["qc"]["issues"] if item["rule_id"] == "context")
    report["qc"]["issues"] = [
        finding,
        {**finding, "rule_id": "old_check", "rule_version": 7},
        {**finding, "rule_id": "missing_check", "rule_version": 2},
    ]
    report["qc"]["rules"]["old_check"] = {"description": "Historical check title"}
    for output in (render_html(report), " ".join(" ".join(page.extract_text().split())
                   for page in PdfReader(io.BytesIO(render_pdf(report))).pages)):
        assert "Missing measurement method or laboratory · v1" in output
        assert "Historical check title · v7" in output
        assert "missing_check · v2" in output
        assert "context v1" not in output
    assert finding["rule_id"] == "context"


def test_qc_report_identifies_the_indicator_even_for_a_generic_finding(data):
    cid = data.columns[2].id
    report = build_report(data, [row.id for row in data.rows],
                          ReportOptions((cid,), data_limit=1, language="en"))
    issue = next(item for item in report["qc"]["issues"] if item["property_id"] == cid)
    report["qc"]["issues"] = [{**issue, "message": "Generic QC message"}]
    next(column for column in report["project"]["columns"] if column["id"] == cid)["name"] = "Snapshot HI"
    data.columns[2].name = "Renamed after export"
    headers, rows = next(content for _, _, kind, title, content in report_blocks(report)
                         if kind == "table" and title == "Замечания QC")
    assert headers[2] == "Indicator"
    assert rows[0][2] == "Snapshot HI"
    assert "<th>Indicator</th>" in render_html(report)
    pdf = " ".join(" ".join(page.extract_text().split())
                   for page in PdfReader(io.BytesIO(render_pdf(report))).pages)
    assert "Indicator: Snapshot HI" in pdf
    assert "Generic QC message" in pdf
    assert "Renamed after export" not in pdf


def test_qc_report_prioritizes_severity_before_applying_the_row_limit(data):
    report = build_report(data, [row.id for row in data.rows],
                          ReportOptions((data.columns[2].id,), qc_limit=3, language="en"))
    template = report["qc"]["issues"][0]
    original = [
        {**template, "severity": severity, "message": label}
        for severity, label in (("warning", "Warning first in input"), ("error", "Error one"),
                                ("critical", "Critical one"), ("error", "Error two"))
    ]
    report["qc"]["issues"] = original.copy()
    html = render_html(report)
    pdf = " ".join(page.extract_text() for page in PdfReader(io.BytesIO(render_pdf(report))).pages)
    for output in (html, pdf):
        assert output.index("Error one") < output.index("Error two") < output.index("Critical one")
        assert "Warning first in input" not in output
    assert report["qc"]["issues"] == original


def test_joint_report_accepts_legacy_snapshot_and_reports_missing_selection(data):
    ids = [row.id for row in data.rows]
    columns = (data.columns[-1].id,)
    legacy = data.view["joint_curve_report"] = {
        "row_ids": ids, "saved_at": "2026-01-01", "axes": ["X", "Y"], "formula": "a*x+b",
        "settings": {"loss": "linear", "shared": ["a"]}, "metrics": {}, "parameters": [],
        "series": [], "counts": [], "coefficients": [], "rmse": [], "mae": [], "active_bounds": None,
        "bootstrap": None, "bootstrap_failed": False,
    }
    before = build_report(data, ids, ReportOptions(columns, joint_snapshot_id="legacy"))
    assert before["joint_curve_fit"]["source_status"] == "unverified"
    assert "source_status" not in legacy

    report = build_report(data, ids, ReportOptions(columns, joint_snapshot_id="unknown", language="en"))
    assert "joint_curve_fit" not in report
    assert report["joint_curve_omission"] == {"reason": "missing_snapshot"}
    assert "The selected joint fit snapshot is unavailable." in render_html(report)
    assert "unknown · Selected snapshot unavailable" in comparison_text(compare_snapshots(before, report), language="en")

    data.view["joint_curve_report"] = {**legacy, "id": "fit-2", "saved_at": "2026-01-02", "formula": "a*x+c"}
    after = build_report(data, ids, ReportOptions(columns))
    diff = compare_snapshots(before, after)
    label, old, new = next(item for item in diff["composition"] if item[0] == "Снимок совместной подгонки")
    assert old["formula"] == "a*x+b" and new["id"] == "fit-2"
    assert label in comparison_text(diff)
    assert "Joint fit snapshot" in comparison_text(diff, language="en")

    omitted = build_report(data, ids[:-1], ReportOptions(columns))
    assert omitted["joint_curve_omission"]["snapshot_id"] == "fit-2"
    assert "fit-2 · Omitted. Fitted rows outside the selection: 1" in comparison_text(
        compare_snapshots(after, omitted), language="en",
    )


def test_history_compares_report_language_and_defaults_legacy_snapshots_to_russian(data):
    ids = [row.id for row in data.rows]
    russian = build_report(data, ids, ReportOptions((data.columns[-1].id,)))
    legacy = deepcopy(russian)
    del legacy["options"]["language"]
    assert compare_snapshots(legacy, russian)["composition"] == []

    english = deepcopy(russian)
    english["options"]["language"] = "en"
    diff = compare_snapshots(legacy, english)
    assert diff["composition"] == [("Язык отчёта", "ru", "en")]
    assert "Язык отчёта · Русский → Английский" in comparison_text(diff)
    assert "Report language · Russian → English" in comparison_text(diff, language="en")


def test_docx_exports_editable_tables_figures_and_linked_contents(data, tmp_path):
    _, x, y = [column.id for column in data.columns]
    plot = PlotSpec(x_id=x, y_id=y, text_numbers=True, dpi=72)
    report = build_report(
        data, [row.id for row in data.rows], ReportOptions((x, y), text_numbers=True),
        [("Показатели пиролиза", plot.to_dict())],
    )
    report["options"]["logo_image"] = report["figures"][0]["png"]
    path = tmp_path / "report.docx"
    export_report(path, report, "docx")
    document = Document(path)
    assert document.paragraphs[1].style.name == "Title"
    assert any(p.text == "Оглавление" for p in document.paragraphs)
    assert document.element.xpath(".//w:hyperlink[@w:anchor='report_1']")
    assert document.element.xpath(".//w:bookmarkStart[@w:name='report_1']")
    assert any(p.text.startswith("Рисунок 1") for p in document.paragraphs)
    assert any(p.text.startswith("Таблица 1") for p in document.paragraphs)
    assert any("TOC" in cell.text for table in document.tables for row in table.rows for cell in row.cells)
    assert len(document.inline_shapes) == 2
    assert all(shape._inline.docPr.get("descr") for shape in document.inline_shapes)


def test_english_report_localizes_shared_labels_and_document_language(data, tmp_path):
    _, x, y = [column.id for column in data.columns]
    plot = PlotSpec(x_id=x, y_id=y, text_numbers=True, dpi=72, annotations=[
        {"text": "Note", "coordinates": "axes", "x": .2, "y": .2, "track_id": x},
    ])
    stages = []
    report = build_report(
        data, [row.id for row in data.rows],
        ReportOptions((x, y), title="Example report",
                      sections=("summary", "source", "method", "figures", "data"), language="en"),
        [("Author's chart", plot.to_dict())],
        progress=lambda stage, percent: stages.append((stage, percent)),
    )
    assert stages[:6] == [
        ("Preparing the data snapshot…", 0),
        ("Recalculating indicators…", 10),
        ("Calculating statistics…", 25),
        ("Building chart · Author's chart", 40),
        ("Preparing interpretation and stratigraphy…", 60),
        ("Checking data quality…", 75),
    ]
    assert stages[-1] == ("Preparing report sections…", 92)
    quality_progress = [percent for stage, percent in stages[6:-1] if stage == "Checking data quality…"]
    assert len(quality_progress) == len(stages[6:-1])
    assert quality_progress == sorted(quality_progress)
    assert 75 < max(quality_progress) < 92
    report["release_checks"].append({"code": "example", "message": "Example warning"})
    navigation = []
    html = render_html(report, navigation=navigation)
    assert '<html lang="en">' in html
    assert "Pre-release checks" in html and "Plotted:" in html
    assert "Warning · Example warning" in html and "Sample UUID" in html
    assert "Samples with QC findings:" in html
    assert "Filters and sorting at build time:" in html
    assert "Correlations were not calculated because the statistics section is disabled." in html
    assert "Joint fit charts use the saved snapshot." in html
    assert ("report-data", "Data appendix") in navigation
    original = {key: title for _, key, _, title, _ in report_blocks(report) if key is not None}
    localized = {key: title for _, key, _, title, _ in report_blocks(report, localized=True) if key is not None}
    assert localized.keys() == original.keys()
    assert "Selected columns" in localized.values()
    assert "Author's chart" in localized.values()
    notice = "Annotation 1: The selected track is missing from the chart."
    assert notice in html
    assert report["figures"][0]["notices"] == ["Аннотация №1: выбранный трек отсутствует на графике"]

    pdf = PdfReader(io.BytesIO(render_pdf(report)))
    assert pdf.trailer["/Root"]["/Lang"] == "en"
    assert "Pre-release checks" in pdf.pages[0].extract_text()
    assert notice in " ".join(" ".join(page.extract_text().split()) for page in pdf.pages)

    path = tmp_path / "english.docx"
    export_report(path, report, "docx")
    document = Document(path)
    assert document.core_properties.language == "en"
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    assert "Contents" in paragraphs and "Pre-release checks" in paragraphs
    assert any(text.startswith("Figure 1") for text in paragraphs)
    assert any(text.startswith("Release date (UTC)") for text in paragraphs)
    assert any(notice in text for text in paragraphs)


def test_english_report_localizes_labels_inside_plot_images(data, monkeypatch):
    from matplotlib.figure import Figure

    x = data.columns[-1].id
    savefig = Figure.savefig
    labels = []

    def capture(figure, *args, **kwargs):
        axis = figure.axes[0]
        labels.append((axis.get_xlabel(), axis.get_ylabel(), [text.get_text() for text in axis.get_yticklabels()]))
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    specs = [PlotSpec(kind=kind, x_id=x, dpi=72, show_result_type=True, density=density)
             for kind, density in (("hist", False), ("hist", True), ("box", False))]
    specs[-1].x_axis.label = "Подпись автора"
    build_report(data, [row.id for row in data.rows], ReportOptions((x,), sections=("figures",), language="en"),
                 [(str(index), spec.to_dict()) for index, spec in enumerate(specs)])
    assert labels[0][:2] == ("HI [mg HC/g TOC] · source data", "Sample count")
    assert labels[1][1] == "Density"
    assert labels[2][0] == "Подпись автора · source data"
    assert labels[2][2] == ["All samples"]


def test_english_quality_section_keeps_qc_identity_and_user_comments(data):
    ids = [row.id for row in data.rows]
    columns = tuple(column.id for column in data.columns)
    data.columns[-1].unit = "unknown"
    data.sync_measurements()
    issue = next(item for item in run_checks(data) if item["rule_id"] == "unit")
    data.view["qc_decisions"] = {issue["id"]: {"action": "acknowledged", "comment": "Проверено автором"}}
    data.view["qc_exclusions"] = {"statistics": {
        issue["measurement_id"]: {"revision": issue["revision"], "comment": "Причина автора"},
    }}
    report = build_report(data, ids, ReportOptions(columns, sections=("quality",), language="en"))
    localized = next(item for item in report["qc"]["issues"] if item["id"] == issue["id"])
    assert localized["message"] == "HI: check the unit (unknown)"
    assert localized["decision"] == "acknowledged"
    report["database_snapshot"] = {"records": [], "checked_at": "2026-01-01", "applied_at": "2026-01-02", "source": {}}
    html = render_html(report)
    for label in ("Database snapshot status", "Checked: 2026-01-01", "QC findings", "Acknowledged",
                  "Active QC exclusions", "Statistics", "Проверено автором", "Причина автора"):
        assert label in html


def test_english_nonpyro_section_localizes_status_without_changing_protocol():
    dia, regular = Column("Dia", "number", unit="count"), Column("Regular", "number", unit="count")
    row = Row({dia.id: None, regular.id: 2})
    valid = Row({dia.id: 4, regular.id: 2})
    project = Project(columns=[dia, regular], rows=[row, valid])
    project.view["nonpyro_configs"] = {"c27_diasterane_ratio": {
        "version": 1, "method": "c27_diasterane_ratio", "confirmed": True, "context": "Контекст автора",
        "inputs": {name: {"column_id": column.id, "unit": column.unit}
                   for name, column in (("dia", dia), ("regular", regular))},
    }}
    report = build_report(project, [row.id, valid.id], ReportOptions((dia.id,), sections=("nonpyro",), language="en"))
    protocol = deepcopy(report["nonpyro"])
    html = render_html(report)
    assert "Non-pyrolysis geochemistry · " in html and "Context: Контекст автора" in html
    assert "<th>Indicators</th><th>Interpretation</th>" in html
    assert "dia: Missing value" in html
    assert "C27 diasterane / regular sterane from peaks" in html
    assert "Ratio of two saturated-fraction peaks on a single analytical basis." in html
    assert "Ratio of two C27 peaks" in html
    assert report["nonpyro"] == protocol
    item = report["nonpyro"][0]["results"][0]
    assert item["status"] == "missing" and item["interpretation"] == "dia: missing"


def test_english_nonpyro_text_covers_builtin_methods_and_parameterized_findings():
    from autopyro.analysis.nonpyro import METHODS
    from autopyro.reports.nonpyro_text import ENGLISH_METHODS, english_finding

    assert ENGLISH_METHODS.keys() == METHODS.keys()
    assert english_finding("Максимумы совпадают: насыщенные, смолы") == "Fractions tied for the maximum: saturates, resins"
    assert english_finding("Сумма SARA=94%: баланс вне допуска; автоматическая нормировка не выполнялась") == (
        "SARA total = 94%. The balance is outside tolerance and was not automatically normalized."
    )


def test_english_statistics_and_group_comparison_preserve_source_values(data):
    well, toc, hi = (column.id for column in data.columns)
    report = build_report(data, [row.id for row in data.rows], ReportOptions(
        (toc, hi), group_ids=(well,), sections=("statistics", "comparison", "comparison_figures"),
        text_numbers=True, language="en", detailed_correlations=True,
    ))
    frozen = deepcopy(report["statistics"])
    html = render_html(report)
    for text in ("Descriptive statistics", "Data availability", "Correlations", "IQR outliers", "Incompatible context excluded",
                 "Group: Скважина", "Insufficient observation pairs", "Object comparison", "Grouping order",
                 "Median comparison: HI", "Points show medians.", "<td>A</td>", "<td>B</td>"):
        assert text in html
    assert report["statistics"] == frozen
    assert {row["status"] for row in frozen["correlations"]} == {"insufficient_pairs"}


@pytest.mark.parametrize(("language", "reason", "summary", "n_label"), [
    ("ru", "Недостаточно пар наблюдений", "Без коэффициентов - 33 из 36 пар показателей.", "N пар"),
    ("en", "Insufficient observation pairs", "No coefficients for 33 of 36 indicator pairs.", "Pairs"),
])
def test_demo_correlation_reports_preserve_full_json(language, reason, summary, n_label, tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "pyrolysis.csv"
    headers, rows, line_numbers = read_file(source)
    choices = [(i, name, infer_kind([row[i] for row in rows]), "") for i, name in enumerate(headers)]
    project = build_project(headers, rows, line_numbers, choices, {"type": "test"})
    ids = [row.id for row in project.rows[:4]]
    columns = tuple(column.id for column in project.columns)
    raw_pairs = None
    compact = None
    for detailed in (False, True):
        report = build_report(project, ids, ReportOptions(
            columns, sections=("statistics",), language=language, detailed_correlations=detailed,
        ))
        base = tmp_path / ("detailed" if detailed else "compact")
        for kind in ("pdf", "docx", "json"):
            export_report(base.with_suffix(f".{kind}"), report, kind)
        snapshot = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        pairs = snapshot["statistics"]["correlations"]
        if raw_pairs is not None:
            assert pairs == raw_pairs
        raw_pairs = pairs
        calculated = [pair for pair in pairs if pair["pearson"] is not None]
        assert len(pairs) == 36 and len(calculated) == 3
        assert sum(pair["status"] == "insufficient_pairs" for pair in pairs) == 33
        assert all(pair["n_pairs"] >= 3 and pair["spearman"] is not None for pair in calculated)

        docx = Document(base.with_suffix(".docx"))
        table, = [table for table in docx.tables if "Pearson r" in [c.text for c in table.rows[0].cells]]
        assert [cell.text for cell in table.rows[0].cells][2] == n_label
        cells = [[cell.text for cell in row.cells] for row in table.rows[1:]]
        assert len(cells) == (36 if detailed else 3)
        assert sum(row[-1] == reason for row in cells) == (33 if detailed else 0)
        assert {(x, y, int(n)) for x, y, n, r, _, _ in cells if r != "—"} == {
            (pair["x"], pair["y"], pair["n_pairs"]) for pair in calculated
        }
        text = " ".join(" ".join(page.extract_text().split()) for page in PdfReader(base.with_suffix(".pdf")).pages)
        assert "Pearson r" in text and n_label in text and summary in text
        assert reason in text and "insufficient_pairs" not in text
        assert all(f"{pair['pearson']:.6g}".replace("−", "-") in text for pair in calculated)
        assert "insufficient_pairs" not in "\n".join(paragraph.text for paragraph in docx.paragraphs)
        assert "insufficient_pairs" not in render_html(report)
        if detailed:
            assert ("Все пары корреляций", False, True) in compare_snapshots(compact, report)["composition"]
            del report["options"]["detailed_correlations"]
            assert ("Все пары корреляций", False, True) in compare_snapshots(compact, report)["composition"]
            correlation_table, = [content for _, _, kind, title, content in report_blocks(report)
                                  if kind == "table" and title == "Корреляции"]
            assert len(correlation_table[1]) == 36
        else:
            compact = report


def test_compact_correlations_include_available_coefficients_with_numerical_warning():
    from autopyro.reports.correlations import correlation_blocks

    def pair(status, pearson, spearman):
        return dict(group=["A"], x="X", y="Y", n_pairs=4, pearson=pearson, spearman=spearman, status=status)

    statistics = {"groups": ["Well"], "options": {}, "correlations": [
        pair("numerical_warning", None, 0.8), pair("constant", None, None),
    ]}
    table, omissions = list(correlation_blocks(statistics, "ru", detailed=False))
    assert table[2][1] == [["A", "X", "Y", 4, None, 0.8, "Численное предупреждение"]]
    assert omissions[2] == "Без коэффициентов — 1 из 2 пар показателей. Один из показателей постоянен — 1."


def test_comparison_identity_repeats_in_pdf_and_docx_with_custom_heading(data, tmp_path):
    well, toc, hi = (column.id for column in data.columns)
    report = build_report(
        data, [row.id for row in data.rows], ReportOptions(
            (toc, hi), group_ids=(well,), sections=("comparison",),
            section_titles={"comparison": "Пользовательский заголовок"}, text_numbers=True,
        ),
    )
    pdf = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(render_pdf(report))).pages)
    assert "Пользовательский заголовок" in pdf
    assert pdf.count("Объект: A") == 4  # Two indicators in each of two panels.
    assert pdf.count("Показатель: TOC") == 4
    path = tmp_path / "comparison.docx"
    export_report(path, report, "docx")
    document = Document(path)
    assert len(document.tables) == 2
    for table in document.tables:
        assert table.rows[0]._tr.xpath("./w:trPr/w:tblHeader")
        assert table.rows[1].cells[0].paragraphs[0].style.name == "Report cell"
        assert "Объект: A" in table.rows[1].cells[0].text
        assert "Показатель: TOC" in table.rows[1].cells[0].text


def test_snapshot_isolated_scope_and_counts(data):
    _, x, y = [c.id for c in data.columns]
    before = asdict(deepcopy(data))
    ids = [r.id for r in data.rows[:3]][::-1]
    report = build_report(data, ids, ReportOptions((x, y), text_numbers=True))
    assert asdict(data) == before
    assert [r["id"] for r in report["project"]["rows"]] == ids
    assert data.rows[-1].id not in json.dumps(report)
    assert report["statistics"]["summary"][0]["n"] == 1
    assert report["statistics"]["summary"][0]["censored"] == 1
    assert report["statistics"]["summary"][0]["non_detect"] == 1
    initial = digest(report)
    data.rows[0].values[x] = "900"
    assert digest(report) == initial
    assert report["project"]["rows"][-1]["values"][x] == "1"
    assert digest(json.loads(json.dumps(report))) == initial


def test_indexed_batch_snapshot_preserves_report_contents(data):
    from autopyro.reports.reporting import index_report_source

    data.sync_measurements()
    scope = [data.rows[2].id, data.rows[0].id]
    columns = tuple(column.id for column in data.columns[1:])
    options = ReportOptions(columns, text_numbers=True)
    index = index_report_source(data, [scope, [data.rows[1].id]])
    original = build_report(data, scope, options)
    indexed = build_report(data, scope, options, source_index=index)
    for report in (original, indexed):
        report.pop("created_at")
        report["statistics"].pop("created_at")
    assert indexed == original


def test_preview_freezes_only_selected_samples_and_preserves_batch_reports(data):
    from autopyro.reports.reporting import freeze_report_source

    data.sync_measurements()
    scope = [data.rows[2].id, data.rows[0].id]
    options = ReportOptions((data.columns[-1].id,), sections=("quality", "statistics"))
    source = freeze_report_source(data, scope, options, ())
    assert [row.id for row in source.rows] == scope
    assert {m.sample_id for m in source.science.measurements.values()} == set(scope)
    for group in (scope, scope[:1], scope[1:]):
        expected = build_report(data, group, options)
        actual = build_report(source, group, options)
        for report in (expected, actual):
            report.pop("created_at")
            report["statistics"].pop("created_at")
        assert actual == expected
    data.rows[2].values.clear()
    data.science.current(scope[0], data.columns[-1].id).value = 123456
    assert source.rows[0].values
    assert source.science.current(scope[0], data.columns[-1].id).value != 123456


def test_preview_retains_rows_needed_for_palettes_and_joint_provenance(data):
    from autopyro.reports.reporting import freeze_report_source

    options = ReportOptions((data.columns[-1].id,))
    scope = [data.rows[0].id]
    plots = [("Scatter", PlotSpec().to_dict())]
    assert len(freeze_report_source(data, scope, options, plots).rows) == len(data.rows)
    data.view["joint_curve_runs"] = {"saved": {"row_ids": [data.rows[-1].id]}}
    assert len(freeze_report_source(data, scope, options, ()).rows) == len(data.rows)


@pytest.mark.parametrize("language", ["ru", "en"])
def test_release_checks_detect_zero_point_plot_and_scope_warnings(data, language):
    _, toc, hi = data.columns
    hi.unit = "unknown"
    ids = [data.rows[1].id]
    plot = PlotSpec(x_id=toc.id, y_id=hi.id, text_numbers=True, dpi=72)
    report = build_report(
        data, ids, ReportOptions((toc.id, hi.id), text_numbers=True, data_limit=0, language=language),
        [("QC preview", plot.to_dict())],
    )
    assert {check["code"] for check in report["release_checks"]} == {
        "empty_figure", "no_numeric_data", "unit_check", "truncated_appendix",
    }
    unit_check = next(check for check in report["release_checks"] if check["code"] == "unit_check")
    assert unit_check["sample_count"] == 1
    html = render_html(report)
    if language == "en":
        assert "Pre-release checks" in html
        assert "The chart “QC preview” contains no data." in html
        assert "Increase the appendix row limit to show more samples." in html
        assert "No numeric values from “TOC” were included in the statistics." in html
        assert unit_check["message"] == "Check the units for “HI”. Affected samples: 1."
    else:
        assert "Проверка перед выпуском" in html


def test_english_release_blocker_is_localized_in_export(data, tmp_path):
    report = build_report(data, [], ReportOptions((data.columns[0].id,), sections=(), language="en"))
    checks = {check["code"]: check for check in report["release_checks"]}
    assert checks["empty_selection"]["message"] == "The selection is empty."
    assert checks["no_report_sections"]["hint"] == "Open Settings → Sections and enable a section with data."
    with pytest.raises(ValueError, match="Blocks release · The report has no sections with content"):
        export_report(tmp_path / "blocked.html", report, "html")


def test_report_interpolation_archive_respects_sample_scope(data):
    inside, outside = (row.id for row in data.rows[:2])
    source = data.columns[2]
    proposal = {"row_id": inside, "left_id": outside, "right_id": outside, "value": 1.0}
    protocol = {"rows": [{"id": outside, "value": 12.0}]}
    estimate = Column("Estimate", "number", metadata={
        "method": "linear", "y_id": source.id, "scope_row_ids": [inside, outside],
        "proposals": [proposal, {"row_id": outside, "value": 2.0}],
        "interpolation_protocol": protocol,
    })
    data.columns.append(estimate)
    for row in data.rows:
        row.values[estimate.id] = None

    def metadata_for(ids):
        report = build_report(data, ids, ReportOptions((source.id,)))
        return report["project"]["columns"][-1]["metadata"]

    metadata = metadata_for([inside])
    assert metadata["scope_row_ids"] == [inside]
    assert metadata["proposals"] == [proposal]
    assert metadata["interpolation_protocol_omitted"]
    assert "interpolation_protocol" not in metadata
    assert metadata_for([inside, outside])["interpolation_protocol"] == protocol
    assert estimate.metadata["interpolation_protocol"] == protocol


def test_hidden_graph_does_not_validate_or_render(data):
    ids = [row.id for row in data.rows]
    report = build_report(
        data, ids, ReportOptions((data.columns[-1].id,), sections=("summary",)),
        [("Удалённая колонка", PlotSpec(x_id="missing").to_dict())],
    )
    assert report["figures"] == []
    assert not report["statistics"]["options"]["correlations"]


def test_figures_are_embedded_and_rebuilt_on_scope(data):
    _, x, y = [c.id for c in data.columns]
    spec = PlotSpec(x_id=x, y_id=y, text_numbers=True, scope="all", dpi=72)
    report = build_report(data, [data.rows[0].id], ReportOptions((x, y)), [("TOC–HI", spec.to_dict())])
    figure = report["figures"][0]
    assert figure["row_ids"] == [data.rows[0].id]
    assert figure["original_scope"] == "all"
    assert figure["plotted"][x] == 1
    assert 'src="data:image/png;base64,' in render_html(report)
    assert spec.scope == "all" and not spec.style_maps


def test_report_preserves_archived_fit_provenance_and_stale_notice(data, tmp_path):
    _, x, y = data.columns
    archive = {
        "id": "saved-fit", "name": "Saved fit", "settings": {"x": x.id, "y": y.id, "method": "ols"},
        "axis_units": [x.unit, y.unit], "grid": [1, 3], "curve": [100, 400],
        "source_digest": "frozen-inputs", "source_digest_version": 2, "stale": True,
    }
    data.view["curve_fit_results"] = [archive]
    spec = PlotSpec(x_id=x.id, y_id=y.id, text_numbers=True, dpi=72,
                    curve_archive_id=archive["id"], legend=False)
    report = build_report(data, [data.rows[0].id],
                          ReportOptions((x.id, y.id), sections=("figures",), language="en"),
                          [("Archived fit", spec.to_dict())])

    expected_archive = deepcopy(archive)
    archive["curve"][0] = 999
    data.view["curve_fit_results"].clear()
    path = tmp_path / "archived-fit.json"
    export_report(path, report, "json")
    figure = json.loads(path.read_text(encoding="utf-8"))["figures"][0]
    assert figure["plotted"][x.id] == 1
    assert figure["curve_archive"] == expected_archive
    assert "Saved fit (OLS) · stale archive · archived curve, points from the current scope" in render_html(report)
    changed = deepcopy(report)
    changed["figures"][0]["curve_archive"]["stale"] = False
    assert any(key == "Графики и их настройки" for key, _, _ in compare_snapshots(report, changed)["composition"])
    changed = deepcopy(report)
    changed["figures"][0]["curve_archive"]["curve"][0] = 101
    assert any(key == "Графики и их настройки" for key, _, _ in compare_snapshots(report, changed)["composition"])
    changed = deepcopy(report)
    changed["figures"][0]["png"] = "different image encoding"
    assert compare_snapshots(report, changed)["composition"] == []


def test_small_report_does_not_copy_unselected_rows(data, monkeypatch):
    from autopyro.core import Row
    from autopyro.reports.reporting import freeze_report_source

    selected = data.rows[2]
    copied = []

    def copy_row(row, memo):
        assert row.id == selected.id
        copied.append(row.id)
        return Row(values=deepcopy(row.values, memo), id=row.id,
                   source=deepcopy(row.source, memo), original=deepcopy(row.original, memo))

    monkeypatch.setattr(Row, "__deepcopy__", copy_row, raising=False)
    column = data.columns[2]
    options = ReportOptions((column.id,))
    freeze_report_source(data, [selected.id], options, ())
    assert copied == [selected.id]
    copied.clear()
    report = build_report(data, [selected.id], options)
    assert copied == [selected.id]
    assert report["project"]["rows"][0]["values"][column.id] == 300
    assert set(report["project"]["science"]["samples"]) == {selected.id}


def test_report_palette_keeps_categories_outside_selection(data):
    from autopyro.plotting import ensure_style_maps

    group, x, y = [column.id for column in data.columns]
    spec = PlotSpec(x_id=x, y_id=y, color_by=group, text_numbers=True, dpi=72)
    reference = PlotSpec.from_dict(spec.to_dict())
    ensure_style_maps(data, reference)
    report = build_report(data, [data.rows[-1].id], ReportOptions((y,)),
                          [("Grouped", spec.to_dict())])
    palette = report["figures"][0]["spec"]["style_maps"]["color"][group]
    entries = list(palette.values())
    assert [entry["label"] for entry in entries] == ["A", "B"]
    assert palette == reference.style_maps["color"][group]
    assert not spec.style_maps


def test_grouped_report_copies_only_selected_rows_and_keeps_all_style_channels(data, monkeypatch):
    from autopyro.core import Row
    from autopyro.plotting import ensure_style_maps

    selected = data.rows[-1]
    group, x, y = [column.id for column in data.columns]
    spec = PlotSpec(x_id=x, y_id=y, color_by=group, shape_by=group,
                    size_by=group, size_mode="category", text_numbers=True, dpi=72)
    reference = PlotSpec.from_dict(spec.to_dict())
    ensure_style_maps(data, reference)
    copied = []

    def copy_row(row, memo):
        assert row.id == selected.id
        copied.append(row.id)
        return Row(values=deepcopy(row.values, memo), id=row.id,
                   source=deepcopy(row.source, memo), original=deepcopy(row.original, memo))

    monkeypatch.setattr(Row, "__deepcopy__", copy_row, raising=False)
    report = build_report(data, [selected.id], ReportOptions((y,)),
                          iter([("Groups", spec.to_dict())]))
    assert copied == [selected.id]
    assert report["figures"][0]["spec"]["style_maps"] == reference.style_maps
    assert not spec.style_maps


def test_derived_palette_still_uses_full_recalculated_project(data, monkeypatch):
    from autopyro.core import Column
    from autopyro import reporting

    category = Column("Derived family", metadata={"stratigraphy": {"placeholder": True}})
    data.columns.append(category)
    for row in data.rows:
        row.values[category.id] = "stale"

    def recalculate(project):
        for index, row in enumerate(project.rows):
            row.values[category.id] = "A" if index < 2 else "B"

    from autopyro import stratigraphy
    monkeypatch.setattr(stratigraphy, "recompute", recalculate)
    x, y = data.columns[1:3]
    spec = PlotSpec(x_id=x.id, y_id=y.id, color_by=category.id, text_numbers=True, dpi=72)
    report = reporting.build_report(data, [data.rows[-1].id], ReportOptions((y.id,)),
                                    [("Derived", spec.to_dict())])
    palette = report["figures"][0]["spec"]["style_maps"]["color"][category.id]
    assert [entry["label"] for entry in palette.values()] == ["A", "B"]
    assert all(row.values[category.id] == "stale" for row in data.rows)


@pytest.mark.parametrize("language", ["ru", "en"])
def test_interpretation_is_fresh_and_has_schema_version(data, language):
    _, x, y = [c.id for c in data.columns]
    scheme = json.loads(Path("examples/interpretation-demo.json").read_text(encoding="utf-8"))
    data.view["interpretation_schemes"] = {scheme["id"]: scheme}
    data.view["interpretation_current"] = {"scheme_id": scheme["id"], "x_id": x, "y_id": y,
        "x_unit": "%wt", "y_unit": "mg HC/g TOC", "text_numbers": True}
    data.view["interpretation_result"] = {"stale": True, "results": ["old"]}
    report = build_report(data, [data.rows[0].id, data.rows[1].id], ReportOptions((x, y), language=language))
    result = report["interpretation"]
    assert result["stale"] is False
    assert result["results"][0]["regions"] == ["A"]
    assert result["scheme"]["revision"] == "1"
    assert data.view["interpretation_result"]["results"] == ["old"]
    frozen = deepcopy(result)
    html = render_html(report)
    assert "demonstration" in html
    if language == "en":
        for label in ("Interpretation scheme", "Interpretation results", "Within a region", "Inside: A",
                      "Insufficient data", "No valid numeric X or Y value"):
            assert label in html
    assert result == frozen


def test_english_curve_interpretation_translates_nested_coordinate_error(data):
    _, x, y = (column.id for column in data.columns)
    scheme = json.loads(Path("examples/interpretation-demo.json").read_text(encoding="utf-8"))
    scheme.update(strategy="nearest_curve", distance_metric="analysis_euclidean", metric_note="Test axes",
                  regions=[], lines=[{"id": "curve", "label": "Curve",
                                      "geometry": {"type": "LineString", "coordinates": [[2, 0], [4, 500]]}}])
    scheme["x"]["minimum"] = 2
    data.view["interpretation_schemes"] = {scheme["id"]: scheme}
    data.view["interpretation_current"] = {"scheme_id": scheme["id"], "x_id": x, "y_id": y,
        "x_unit": "%wt", "y_unit": "mg HC/g TOC", "text_numbers": True}
    report = build_report(data, [data.rows[0].id], ReportOptions((x, y), language="en"))
    item, = report["interpretation"]["results"]
    reason = "Недопустимые координаты или интервал: Координата ниже области применимости"
    assert item["reason"] == reason
    html = render_html(report)
    assert "Invalid coordinates or interval: Coordinate is below the applicability range" in html
    assert reason not in html and item["reason"] == reason


def test_html_escapes_user_content_and_appendix_limit(data, tmp_path):
    column = data.columns[-1].id
    report = build_report(data, [r.id for r in data.rows], ReportOptions((column,),
        title='<script>alert("x")</script>', conclusions='<img src="https://example.org/x">', data_limit=1))
    html = render_html(report)
    assert "<script>" not in html and '<img src="https:' not in html
    assert "&lt;script&gt;" in html
    assert "Показано 1 из 4" in html
    export_report(tmp_path / "report.json", report, "json")
    restored = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert len(restored["project"]["rows"]) == 4
    assert digest(restored) == digest(report)


def test_empty_selection_does_not_expand(data):
    report = build_report(data, [], ReportOptions((data.columns[-1].id,)))
    assert report["row_ids"] == []
    assert report["statistics"]["summary"] == []


def test_pdf_unicode_and_pagination(data):
    report = build_report(data, [r.id for r in data.rows], ReportOptions(tuple(c.id for c in data.columns),
        (data.columns[0].id,), title="Пиролиз скважины", conclusions="Наблюдения автора.\n" * 120))
    pdf = render_pdf(report)
    reader = PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) > 2
    assert reader.outline
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Пиролиз скважины" in text
    assert "Наблюдения автора" in text
    assert "<0.1" in text and "ND" in text
    assert "SHA-256" in text


def test_export_failure_preserves_old_file(data, tmp_path, monkeypatch):
    import autopyro.reporting as module
    path = tmp_path / "report.html"
    path.write_text("old")
    report = build_report(data, [], ReportOptions((data.columns[-1].id,)))
    monkeypatch.setattr(module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("locked")))
    with pytest.raises(OSError):
        export_report(path, report, "html")
    assert path.read_text() == "old"
    assert not list(tmp_path.glob("*.tmp"))


def test_history_displays_qc_review_without_measurement_change(data):
    report = build_report(data, data.row_map(), ReportOptions((data.columns[-1].id,)))
    changed = deepcopy(report)
    issue = changed["qc"]["issues"][0]
    issue.update(decision="acknowledged", comment="Verified repeat")
    diff = compare_snapshots(report, changed)
    assert len(diff["qc_decisions"]) == 1
    assert issue["sample_id"] in comparison_text(diff)
    assert "Verified repeat" in comparison_text(diff)
    revised = deepcopy(changed)
    revised["qc"]["issues"][0]["comment"] = "Corrected note"
    assert "Corrected note" in comparison_text(compare_snapshots(changed, revised))
