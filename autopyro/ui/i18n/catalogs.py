"""Maintain Qt catalogs: python -m autopyro.ui.i18n.catalogs --update --compile."""

import argparse
import ast
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from string import Formatter

from ...data_io.file_output import atomic_output
from . import CONTEXT, SOURCE_LANGUAGE

DIRECTORY = Path(__file__).with_name("translations")


def write_ts(path, root):
    ET.indent(root, space="  ")
    with atomic_output(path) as temporary:
        temporary.write_bytes(b'<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE TS>\n'
                              + ET.tostring(root, encoding="utf-8") + b"\n")


def sources():
    """Keep domain messages from the source catalog and extract UI tr() calls."""
    root = ET.parse(DIRECTORY / "autopyro_en.ts").getroot()
    result = {entry.findtext("source"): entry.get("numerus") == "yes"
              for context in root.findall("context") if context.findtext("name") == CONTEXT
              for entry in context.findall("message")}
    ui = Path(__file__).resolve().parents[1]
    paths = [*ui.rglob("*.py"), *(ui.parent / "database" / "ui").rglob("*.py")]
    for path in paths:
        if "i18n" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name == "tr" and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                result[node.args[0].value] = any(keyword.arg == "n" for keyword in node.keywords)
    return result


def update(path, source_messages):
    root = ET.parse(path).getroot()
    context = next((c for c in root.findall("context") if c.findtext("name") == CONTEXT), None)
    if context is None:
        context = ET.SubElement(root, "context")
        ET.SubElement(context, "name").text = CONTEXT
    existing = {entry.findtext("source") for entry in context.findall("message")}
    for source, numerus in source_messages.items():
        if source not in existing:
            entry = ET.SubElement(context, "message", **({"numerus": "yes"} if numerus else {}))
            ET.SubElement(entry, "source").text = source
            if root.get("language") == SOURCE_LANGUAGE and not numerus:
                ET.SubElement(entry, "translation").text = source
            else:
                ET.SubElement(entry, "translation", type="unfinished")
    write_ts(path, root)


def create(path, code, native_name, source_messages):
    if path.exists():
        raise ValueError(f"Catalog already exists: {path}")
    root = ET.Element("TS", version="2.1", language=code, sourcelanguage="ru_RU")
    metadata = ET.SubElement(root, "context")
    ET.SubElement(metadata, "name").text = "AutoPyroLanguage"
    entry = ET.SubElement(metadata, "message")
    ET.SubElement(entry, "source").text = "native_name"
    ET.SubElement(entry, "translation").text = native_name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_ts(path, root)
    update(path, source_messages)


def validate(path):
    def placeholders(text):
        python = sorted((field, spec, conversion) for _, field, spec, conversion in Formatter().parse(text) if field is not None)
        qt = sorted(re.findall(r"%(?:L?n|L?\d+)", text))
        return python, qt

    seen = set()
    for context in ET.parse(path).findall("context"):
        for entry in context.findall("message"):
            source = entry.findtext("source", "")
            key = context.findtext("name"), source
            if key in seen:
                raise ValueError(f"Duplicate message in {path}: {source}")
            seen.add(key)
            translation = entry.find("translation")
            if translation is None or translation.get("type") in ("unfinished", "obsolete", "vanished"):
                continue
            targets = translation.findall("numerusform") if entry.get("numerus") == "yes" else [translation]
            for target in targets:
                if target.text and placeholders(source) != placeholders(target.text):
                    raise ValueError(f"Translation placeholders differ in {path}: {source}")


def lrelease():
    executable = shutil.which("pyside6-lrelease") or Path(sys.executable).with_name(
        "pyside6-lrelease.exe" if sys.platform == "win32" else "pyside6-lrelease"
    )
    return str(executable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--new", metavar="LOCALE")
    parser.add_argument("--name", help="Native language name shown in settings")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if not (args.new or args.update or args.compile):
        parser.error("Choose --new, --update or --compile")
    source_messages = sources()
    if args.new:
        if not re.fullmatch(r"[a-zA-Z]{2,3}(?:_[a-zA-Z0-9]{2,8})*", args.new):
            parser.error("Use a locale code such as de, pt_BR or zh_Hant_TW")
        create(args.directory / f"autopyro_{args.new}.ts", args.new, args.name or args.new, source_messages)
    paths = sorted(args.directory.glob("autopyro_*.ts"))
    for path in paths:
        if args.update:
            update(path, source_messages)
        validate(path)
        if args.compile:
            with atomic_output(path.with_suffix(".qm")) as temporary:
                subprocess.run([lrelease(), str(path), "-nounfinished", "-qm", str(temporary)], check=True)
    if args.directory.resolve() == DIRECTORY.resolve() and (args.update or args.new):
        patterns = [source for source in source_messages if "{0" in source]
        Path(__file__).with_name("source_patterns.py").write_text(
            '"""Source templates for recognizing formatted domain messages; no translations."""\n\nSOURCE_PATTERNS = (\n'
            + "".join(f"    {source!r},\n" for source in patterns) + ")\n", encoding="utf-8")


if __name__ == "__main__":
    main()
