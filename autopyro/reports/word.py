"""Render the shared report sections as an editable Word document."""

import base64
import io
import re
from collections import Counter
from datetime import datetime, timezone
from itertools import chain

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

from .reporting import display, figure_caption, report_language, sections, table_panels


def _text(value):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]", "�", display(value))


def _element(tag, **attributes):
    element = OxmlElement(f"w:{tag}")
    for key, value in attributes.items():
        element.set(qn(f"w:{key}"), str(value))
    return element


def _image(paragraph, encoded, max_width, max_height, *, description=""):
    shape = paragraph.add_run().add_picture(io.BytesIO(base64.b64decode(encoded)))
    if description:
        shape._inline.docPr.set("descr", _text(description))
    scale = min(max_width / shape.width, max_height / shape.height, 1)
    shape.width = round(shape.width * scale)
    shape.height = round(shape.height * scale)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _table(document, headers, rows, cell_style_id, *, wide_identity=False):
    table = document.add_table(rows=1, cols=len(headers))
    table.autofit = False
    if wide_identity:
        width = sum(column.width for column in table.columns)
        identity_width = round(width * 0.3)
        other_width = round((width - identity_width) / (len(headers) - 1))
        for index, column in enumerate(table.columns):
            column.width = identity_width if index == 0 else other_width
            column.cells[0].width = column.width
    borders = _element("tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(_element(edge, val="single", sz=4, color="D9D9D9"))
    table._tbl.tblPr.append(borders)
    header = table.rows[0]
    header._tr.get_or_add_trPr().append(_element("tblHeader"))
    for values, row in chain(((headers, header),), ((values, table.add_row()) for values in rows)):
        row._tr.get_or_add_trPr().insert(0, _element("cantSplit"))
        for cell, value in zip(row.cells, values):
            paragraph = cell.paragraphs[0]
            # The document created and validated this style once. Setting its ID
            # directly avoids scanning the style catalog for every table cell.
            paragraph._p.style = cell_style_id
            run = paragraph.add_run(_text(value))
            if row is header:
                run.bold = True
                cell._tc.get_or_add_tcPr().append(_element("shd", fill="E7EEF5"))
            elif isinstance(value, (float, int)):
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT


def render_docx(report):
    document = Document()
    options = report["options"]
    language = report_language(report)
    english = language == "en"
    document.core_properties.title = _text(options["title"])[:255]
    document.core_properties.author = _text(options["author"])[:255]
    document.core_properties.language = language
    for element in document.styles.element.xpath("w:docDefaults/w:rPrDefault/w:rPr/w:lang"):
        element.set(qn("w:val"), "en-US" if english else "ru-RU")
    page = document.sections[0]
    page.page_width, page.page_height = Mm(210), Mm(297)
    page.left_margin = page.right_margin = Mm(19)
    page.top_margin = page.bottom_margin = Mm(18)
    page.different_first_page_header_footer = True
    footer = page.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("AutoPyro  ·  ")
    footer._p.append(_element("fldSimple", instr="PAGE"))
    for name, size in (("Normal", 10), ("Title", 27), ("Heading 1", 14), ("Heading 2", 11), ("Caption", 9)):
        style = document.styles[name]
        style.font.name, style.font.size = "Arial", Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        properties = style.element.get_or_add_pPr()
        for border in properties.findall(qn("w:pBdr")):
            properties.remove(border)
    cell_style = document.styles.add_style("Report cell", WD_STYLE_TYPE.PARAGRAPH)
    cell_style.base_style = document.styles["Normal"]
    cell_style.font.size = Pt(9)
    cell_style.paragraph_format.space_before = Pt(3)
    cell_style.paragraph_format.space_after = Pt(3)
    cell_style_id = cell_style.style_id
    normal = document.styles["Normal"]
    normal.paragraph_format.space_after = Pt(6)

    logo = options.get("logo_image")
    if logo:
        _image(document.add_paragraph(), logo, Mm(55), Mm(25),
               description="Report logo" if english else "Логотип отчёта")
    outline = []
    numbers = Counter()
    for kind, label, content in sections(report, paged=True):
        if kind in ("table", "figure"):
            numbers[kind] += 1
            prefix = ("Table" if english else "Таблица") if kind == "table" else ("Figure" if english else "Рисунок")
            label = f"{prefix} {numbers[kind]} · {label}"
        outline.append((kind, _text(label), content))
    compact = len(outline) <= 4
    title = document.add_paragraph(_text(options["title"]), "Title")
    title.paragraph_format.space_before = Pt(0 if compact else 64 if not logo else 30)
    if options["author"]:
        document.add_paragraph(_text(options["author"]))
    date_label = "Release date (UTC)" if english else "Дата выпуска (UTC)"
    sample_label = "Samples in selection" if english else "Проб в выборке"
    created_at = _text(report['created_at'])
    try:
        release = datetime.fromisoformat(created_at)
        if release.tzinfo is not None:
            release = release.astimezone(timezone.utc)
        created_at = release.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    document.add_paragraph(f"{date_label}  {created_at}")
    document.add_paragraph(f"{sample_label}  {len(report['row_ids'])}")
    if not compact:
        document.add_page_break()
    if outline:
        document.add_paragraph("Contents" if english else "Оглавление", "Title")
        for index, (_, label, _) in enumerate(outline, 1):
            paragraph = document.add_paragraph()
            link = _element("hyperlink", anchor=f"report_{index}")
            run = _element("r")
            props = _element("rPr")
            props.append(_element("color", val="234B72"))
            run.append(props)
            text = _element("t")
            text.text = label
            run.append(text)
            link.append(run)
            paragraph._p.append(link)
        if not compact:
            document.add_page_break()
    for index, (kind, label, content) in enumerate(outline, 1):
        heading = document.add_heading(label, 1)
        heading._p.insert(1, _element("bookmarkStart", id=index, name=f"report_{index}"))
        heading._p.append(_element("bookmarkEnd", id=index))
        if kind == "text":
            for line in content.splitlines():
                document.add_paragraph(_text(line))
        elif kind == "figure":
            image = document.add_paragraph()
            image.paragraph_format.keep_with_next = True
            caption = figure_caption(content, language)
            _image(image, content["png"], Mm(172), Mm(148), description=f"{label}. {caption}")
            document.add_paragraph(_text(caption), "Caption")
        else:
            headers, rows, identifiers = content
            if not rows:
                document.add_paragraph("No data." if english else "Нет данных.")
                continue
            for index, (panel_headers, panel_rows) in enumerate(table_panels(headers, rows, identifiers, 6)):
                if index:
                    continuation = "continued" if english else "продолжение"
                    document.add_heading(f"{label} · {continuation}", 2)
                _table(document, panel_headers, panel_rows, cell_style_id,
                       wide_identity=identifiers > 1 and len(headers) > 6 and len(panel_headers) > 2)
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()
