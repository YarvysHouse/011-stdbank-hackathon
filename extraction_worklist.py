# -*- coding: utf-8 -*-
"""Build the list of figures still needed to complete the comparison.

Rather than a generic 600-row grid, this asks only for what the documents can
actually supply: the entity-years covered by a pack, crossed with the ten
benchmark line items, marked with what the OCR pass already produced and how
far it can be trusted.

Status per row:
    have      a value is in benchmarks/reported_financials.csv
    suspect   a value exists but the entity failed a sanity check - re-key it
    missing   nothing extracted; key it by hand

Priority runs 1 (unblocks everything) to 5 (drop if time is short).

Usage:
    python extraction_worklist.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_reported import (
    ENTITY_YEAR_END_MONTH,
    TARGET_YEARS,
    fiscal_year,
    years_in_filename,
)
from extract_statements import FILE_ENTITY, PDF_DIR, entity_for

BASE_DIR = Path(__file__).resolve().parent
REPORTED = BASE_DIR / "benchmarks" / "reported_financials.csv"
COMPARISON = BASE_DIR / "outputs" / "statements" / "comparison_to_reported.csv"
OUT_PATH = BASE_DIR / "benchmarks" / "extraction_worklist.csv"

# line_item -> (metric_type, priority, where to find it)
LINE_ITEMS = {
    "Total revenue": (
        "amount", 1,
        "Income statement, 'Revenue' - the group total, not a segment line",
    ),
    "Net cash from operating activities": (
        "amount", 2,
        "Cash flow statement subtotal",
    ),
    "Taxation paid": (
        "amount", 2,
        "Cash flow statement 'taxation paid' - NOT the income statement charge",
    ),
    "Cost of sales and supplier payments": (
        "amount", 3,
        "Cost of sales; if not disclosed, direct + other operating expenses",
    ),
    "Employee costs": (
        "amount", 3,
        "'Staff expenses' / employee benefits - often a note, not the face",
    ),
    "EBITDA proxy": (
        "amount", 3,
        "Operating profit + depreciation and amortisation",
    ),
    "Gross margin proxy": (
        "ratio", 4,
        "Computed: (revenue - cost of sales) / revenue",
    ),
    "EBITDA margin proxy": (
        "ratio", 4,
        "Computed: EBITDA / revenue",
    ),
    "Employee cost ratio": (
        "ratio", 4,
        "Computed: employee costs / revenue",
    ),
    "Export revenue share": (
        "ratio", 5,
        "Segment note, foreign revenue / total revenue - usually absent",
    ),
}

# A bank-side sample should imply broadly similar coverage across entities.
# Anything far outside this band means the reported figures are on the wrong
# scale rather than that the entity banks unusually little or much here.
PLAUSIBLE_COVERAGE = (20.0, 500.0)


def document_coverage() -> dict[tuple[str, str], list[str]]:
    """Which PDFs can supply which entity-year, from the filename year ranges."""
    coverage: dict[tuple[str, str], list[str]] = {}
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        entity_id, _ = entity_for(pdf)
        month = ENTITY_YEAR_END_MONTH.get(entity_id, 12)
        interim = "interim" in pdf.name.lower()
        for year in years_in_filename(pdf.name):
            fy = fiscal_year(year, month)
            if fy in TARGET_YEARS and not interim:
                coverage.setdefault((entity_id, fy), []).append(pdf.name)
    return coverage


def suspect_entities() -> dict[str, str]:
    """Entities whose extracted figures failed a sanity check."""
    flags: dict[str, str] = {}

    if REPORTED.exists():
        reported = pd.read_csv(REPORTED)
        wide = reported.pivot_table(
            index=["entity_id", "fiscal_year"], columns="line_item",
            values="reported_value", aggfunc="first",
        )
        for (entity_id, _), row in wide.iterrows():
            for ratio in ["Gross margin proxy", "EBITDA margin proxy", "Employee cost ratio"]:
                value = row.get(ratio)
                if pd.notna(value) and not 0 <= value <= 1:
                    flags[entity_id] = f"impossible {ratio} ({value:.2f})"

    if COMPARISON.exists():
        comparison = pd.read_csv(COMPARISON)
        for entity_id, group in comparison.groupby("entity_id"):
            coverage = group["revenue_coverage"].dropna()
            if coverage.empty:
                continue
            value = coverage.iloc[0]
            if not PLAUSIBLE_COVERAGE[0] <= value <= PLAUSIBLE_COVERAGE[1]:
                flags[entity_id] = f"implied coverage {value:,.0f}x outside plausible range"
    return flags


def main() -> None:
    coverage = document_coverage()
    suspects = suspect_entities()
    entities = {eid: name for eid, name in FILE_ENTITY.values()}

    have: set[tuple[str, str, str]] = set()
    values: dict[tuple[str, str, str], float] = {}
    if REPORTED.exists():
        reported = pd.read_csv(REPORTED)
        for _, row in reported.iterrows():
            key = (row["entity_id"], row["fiscal_year"], row["line_item"])
            have.add(key)
            values[key] = row["reported_value"]

    rows = []
    for entity_id in sorted(entities):
        for fy in TARGET_YEARS:
            docs = coverage.get((entity_id, fy), [])
            if not docs:
                continue  # no published pack covers this entity-year
            for line_item, (metric_type, priority, where) in LINE_ITEMS.items():
                key = (entity_id, fy, line_item)
                if key in have:
                    status = "suspect" if entity_id in suspects else "have"
                else:
                    status = "missing"
                rows.append({
                    "entity_id": entity_id,
                    "entity_name": entities[entity_id],
                    "fiscal_year": fy,
                    "line_item": line_item,
                    "metric_type": metric_type,
                    "priority": priority,
                    "status": status,
                    "current_value": values.get(key, ""),
                    "source_pdf": "; ".join(docs),
                    "where_to_look": where,
                    "note": suspects.get(entity_id, ""),
                })

    worklist = pd.DataFrame(rows).sort_values(
        ["priority", "entity_id", "fiscal_year", "line_item"]
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    worklist.to_csv(OUT_PATH, index=False)

    print(f"{len(worklist)} rows -> {OUT_PATH}")
    print("\n--- by status ---")
    print(worklist["status"].value_counts().to_string())
    print("\n--- by priority ---")
    print(
        worklist.pivot_table(
            index="priority", columns="status", values="line_item",
            aggfunc="count", fill_value=0,
        ).to_string()
    )
    needed = worklist[worklist["status"] != "have"]
    print(f"\nto key by hand: {len(needed)} ({len(needed[needed.priority <= 2])} at priority 1-2)")
    print("\n--- priority 1: revenue, the anchor for every other line ---")
    p1 = worklist[worklist["priority"] == 1]
    print(
        p1.pivot_table(index=["entity_id", "entity_name"], columns="fiscal_year",
                       values="status", aggfunc="first", fill_value="-").to_string()
    )
    if suspects:
        print("\n--- entities flagged suspect ---")
        for entity_id, reason in sorted(suspects.items()):
            print(f"  {entity_id} {entities.get(entity_id,''):<24} {reason}")


if __name__ == "__main__":
    main()
