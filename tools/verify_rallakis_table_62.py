"""Recalculate Table 6.2 Rc values using AutoPyro and the original PDF.

Run from the repository root with pypdf available:
    python -m tools.verify_rallakis_table_62 DimitriosRallakisMasterThesis.pdf
"""

import argparse
import json
import re
from collections import Counter

from pypdf import PdfReader

from autopyro.analysis.nonpyro import RALLAKIS_RC, rallakis_rc


def read_table(pdf):
    rows = {}
    for start, page in zip((1, 10, 19), PdfReader(pdf).pages[103:106]):
        seen = set()
        for line in page.extract_text().splitlines():
            fields = line.split()
            if len(fields) < 11 or not all(
                re.fullmatch(r"\d+\.\d{2}|b\.d\.l", field) for field in fields[-9:]
            ):
                continue
            sample, *formation = fields[:-9]
            if sample in seen:
                raise ValueError(f"Duplicate sample {sample} in columns starting at {start}")
            seen.add(sample)
            row = rows.setdefault(sample, {"formation": " ".join(formation), "published": {}})
            if row["formation"] != " ".join(formation):
                raise ValueError(f"Conflicting formation for {sample}")
            row["published"].update(
                (str(index), field) for index, field in enumerate(fields[-9:], start)
            )
    if len(rows) != 24 or any(len(row["published"]) != 27 for row in rows.values()):
        raise ValueError("Expected 24 complete rows of 27 parameters in Table 6.2")
    return rows


def compare_rc(published):
    for index, (label, _, _) in RALLAKIS_RC.items():
        parameter = label.split()[0]
        reference = float(published[parameter])
        raw_input = published[index]
        if raw_input == "b.d.l":
            yield parameter, {"input": raw_input, "published": reference,
                              "calculated": rallakis_rc(index, None), "status": "below_detection_limit"}
            continue
        value = float(raw_input)
        calculated = rallakis_rc(index, value)
        lower = rallakis_rc(index, value - 0.005)
        upper = rallakis_rc(index, value + 0.005)
        overlaps = lower <= reference + 0.005 and upper >= reference - 0.005
        yield parameter, {
            "input": value, "published": reference, "calculated": round(calculated, 8),
            "delta": round(calculated - reference, 8),
            "status": "matches_rounded_input" if abs(calculated - reference) <= 0.005 else
                      "compatible_with_rounding" if overlaps else "mismatch",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    args = parser.parse_args()
    rows = read_table(args.pdf)
    summary = Counter()
    for row in rows.values():
        row["rc_comparison"] = dict(compare_rc(row["published"]))
        summary.update(result["status"] for result in row["rc_comparison"].values())
    print(json.dumps({
        "source": "Rallakis (2015), Table 6.2, PDF pages 104–106",
        "scope": "Parameters 22–25 recalculated from published 18–21. Other parameters transcribed only.",
        "summary": dict(summary),
        "samples": rows,
    }, ensure_ascii=False, indent=2))
    return int(summary["mismatch"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
