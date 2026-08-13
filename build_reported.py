# -*- coding: utf-8 -*-
"""Turn the OCR'd statement lines into the benchmark file for comparison.

`extract_statements.py` produces one row per figure found on a page, labelled
with whatever wording the issuer used. `financial_statements.py` wants ten
named line items per entity-year, in rands. This bridges the two:

    match       issuer wording -> one of seven canonical concepts
    period      column header year + year-end month -> the proxy's FY label
    scale       statement units (usually millions) -> rands, FX where needed
    derive      EBITDA and the three ratios from the matched amounts
    write       benchmarks/reported_financials.csv

Every choice that could be wrong is recorded in the `note` column, so a row
can be traced back to the page it came from and overridden by hand.

Usage:
    python build_reported.py                 # build the benchmark CSV
    python build_reported.py --report        # + per-entity coverage summary
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
EXTRACTED = BASE_DIR / "outputs" / "extracted" / "extracted_lines.csv"
OUT_PATH = BASE_DIR / "benchmarks" / "reported_financials.csv"

FISCAL_YEAR_END_MONTH = 6  # the proxy labels every entity on a June year-end

# Issuer wording -> canonical concept. Anchored where a loose match would pull
# in subtotals ("total revenue" vs "revenue from associates").
CONCEPT_PATTERNS = {
    # Anchored hard: a loose "^revenue\b" pulls in segment lines such as
    # "Revenue from energy activity", which then win the first-occurrence pick
    # and stand in for group revenue.
    "revenue": r"^(total |group )?revenue$|^revenue from contracts with customers$|^turnover$",
    "cost_of_sales": r"^cost of (sales|goods sold)$|^direct expenses$|^cost of merchandise",
    "employee_costs": r"^staff (expenses|costs)$|^employee (benefit|cost)|^salaries and wages",
    "operating_profit": r"^operating profit|^trading profit$|^operating income$|^profit from operations",
    "depreciation": r"^depreciation and amortisation|^depreciation, amortisation|^depreciation$",
    "tax_paid": r"^tax(ation)? paid|^income tax paid",
    "net_cash_ops": r"net cash (generated|flows?|inflow).{0,30}operating activities|^cash generated (from|by) operations",
}

# Fallback only - the year-end month is read off the statement's own caption
# ("for the year ended 31 March 2025") wherever one was captured.
ENTITY_YEAR_END_MONTH = {
    "E01": 6, "E02": 12, "E03": 12, "E04": 12, "E05": 12, "E06": 12,
    "E07": 6, "E08": 12, "E09": 6, "E10": 6, "E11": 9, "E12": 8,
    "E13": 12, "E14": 3, "E15": 3, "E16": 12, "E17": 3, "E18": 6,
    "E19": 6, "E20": 12,
}

# Approximate average rates, ZAR per unit, by proxy fiscal year. Precision
# matters less than it looks: compare_to_reported() normalises every amount by
# the entity's own revenue coverage, so a constant FX error on all of an
# entity-year's lines cancels out of the deviation it flags on. It only moves
# the raw rand variance shown alongside.
FX_TO_ZAR = {
    "USD": {"FY2024": 18.6, "FY2025": 18.2, "FY2026": 17.9},
    "EUR": {"FY2024": 20.0, "FY2025": 19.7, "FY2026": 20.6},
    "GBP": {"FY2024": 23.4, "FY2025": 23.5, "FY2026": 24.1},
    "ZAR": {"FY2024": 1.0, "FY2025": 1.0, "FY2026": 1.0},
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

TARGET_YEARS = ["FY2024", "FY2025", "FY2026"]

# Band a JSE/LSE-listed group's annual revenue can occupy once converted to
# rands - R1bn to R20tn. Used only to catch 1000x unit errors, so it is
# deliberately wide enough to be uncontroversial.
PLAUSIBLE_REVENUE_ZAR = (1e9, 2e13)


def concept_for(label: str) -> str | None:
    """Match an issuer's row label to a canonical concept, or None."""
    text = re.sub(r"\s+", " ", str(label)).strip().lower()
    text = re.sub(r"[*†‡]|\(\d+\)|\bnote \d+\b", "", text).strip()
    for concept, pattern in CONCEPT_PATTERNS.items():
        if re.search(pattern, text):
            return concept
    return None


def year_in(text: str) -> int | None:
    """The four-digit reporting year in a column header, if there is one."""
    years = re.findall(r"\b(20[2-3]\d)\b", str(text))
    return int(years[0]) if years else None


def years_in_filename(name: str) -> list[int]:
    """The reporting years a pack covers, from its filename.

    "..._-_24&25.pdf" -> [2025, 2024]; "..._-_23-25.pdf" -> [2025, 2024, 2023].
    Returned latest-first, which is the order the columns appear in: of the 11
    tables where both years were read off the headers, the leftmost figure
    column was the later year in 11 and the earlier year in none.
    """
    stem = Path(name).stem
    pair = re.search(r"(\d{2})\s*&\s*(\d{2})", stem)
    if pair:
        return sorted({2000 + int(pair.group(1)), 2000 + int(pair.group(2))}, reverse=True)
    span = re.search(r"(\d{2})\s*-\s*(\d{2})\b", stem)
    if span:
        lo, hi = 2000 + int(span.group(1)), 2000 + int(span.group(2))
        return sorted(range(lo, hi + 1), reverse=True)
    return []


def assign_years(df: pd.DataFrame) -> pd.DataFrame:
    """Give every row a reporting year, by header where stated, else position.

    Most tables lose their header text to OCR and come back with positional
    column names, so the year is recovered from the filename's year range
    applied across the columns in order. Wider tables (segment sheets carrying
    several groups of the same two years) cycle through the range.
    """
    df = df.copy()
    df["header_year"] = df["column_header"].map(year_in)
    df["report_year"] = pd.NA
    df["year_source"] = ""

    for (source_file, page, table_index), group in df.groupby(
        ["source_file", "page", "table_index"], sort=False
    ):
        years = years_in_filename(source_file)
        order = list(dict.fromkeys(group["column_header"]))
        positional = {
            column: years[i % len(years)] for i, column in enumerate(order)
        } if years else {}

        for index in group.index:
            stated = df.at[index, "header_year"]
            if pd.notna(stated):
                df.at[index, "report_year"] = int(stated)
                df.at[index, "year_source"] = "header"
            elif positional:
                df.at[index, "report_year"] = positional[df.at[index, "column_header"]]
                df.at[index, "year_source"] = "position"
    return df


def year_end_month(titles: pd.Series) -> int | None:
    """Read the year-end month off the statement captions of one document."""
    for title in titles.dropna().astype(str):
        match = re.search(
            r"ended\s+(?:\d{1,2}\s+)?([A-Za-z]+)\s+20\d\d", title, re.I
        )
        if match:
            month = MONTHS.get(match.group(1).lower())
            if month:
                return month
    return None


def fiscal_year(report_year: int, month: int) -> str:
    """Apply the proxy's June-year-end rule to a statement's own year-end."""
    return f"FY{report_year + (1 if month > FISCAL_YEAR_END_MONTH else 0)}"


def build(extracted: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (benchmark rows, the matched source rows behind them)."""
    df = extracted.copy()
    df = df[df["row_label"].notna()]
    df["concept"] = df["row_label"].map(concept_for)
    df = df[df["concept"].notna()]

    # Interim packs are six-month figures; entering them as a year would read
    # as a ~50% coverage anomaly rather than a real discrepancy.
    df["interim"] = df["source_file"].str.contains("interim", case=False)
    df = df[~df["interim"]]

    month_by_doc = (
        df.groupby("source_file")["table_title"].apply(year_end_month).to_dict()
    )
    scale_by_doc = (
        df.groupby("source_file")["unit_scale"].apply(
            lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None
        ).to_dict()
    )

    # groupby.apply turns a None return into NaN, and NaN is truthy - so read
    # these back through pd.notna rather than `or`.
    def caption_month(source_file):
        month = month_by_doc.get(source_file)
        return int(month) if pd.notna(month) else None

    df = assign_years(df)
    df = df[df["report_year"].notna()]
    df["year_end_month"] = [
        caption_month(f) or ENTITY_YEAR_END_MONTH.get(e, 12)
        for f, e in zip(df["source_file"], df["entity_id"])
    ]
    df["month_source"] = [
        "caption" if caption_month(f) else "assumed"
        for f in df["source_file"]
    ]
    df["fiscal_year"] = [
        fiscal_year(int(y), int(m))
        for y, m in zip(df["report_year"], df["year_end_month"])
    ]
    df = df[df["fiscal_year"].isin(TARGET_YEARS)]

    # Scale to rands. Where no unit was detected on the page, fall back to the
    # document's own modal unit, then to millions - the near-universal
    # convention in these packs.
    df["scale"] = df["unit_scale"].fillna(
        df["source_file"].map(lambda f: scale_by_doc.get(f))
    ).fillna(1e6)
    df["scale_source"] = [
        "page" if pd.notna(u) else ("document" if scale_by_doc.get(f) else "assumed 1e6")
        for u, f in zip(df["unit_scale"], df["source_file"])
    ]
    df["fx"] = [
        FX_TO_ZAR.get(c, FX_TO_ZAR["ZAR"]).get(fy, 1.0)
        for c, fy in zip(df["currency"], df["fiscal_year"])
    ]
    df["rands"] = df["value"] * df["scale"] * df["fx"]

    # A dash means nil, which parse_amount rightly reads as 0 - but a zero
    # revenue is a missing reading, not a fact, and it would drive an entity's
    # whole coverage factor.
    df = df[~(df["concept"].eq("revenue") & df["value"].eq(0))]

    # Several packs report in R'000 without the unit surviving OCR, so the
    # millions fallback inflates them 1000x. Correct per document by requiring
    # group revenue to land in the range a listed corporate can occupy, and
    # apply the same factor to every line of that document so the internal
    # relationships (and the coverage factor) stay intact.
    df["scale_fix"] = 1.0
    for source_file, group in df.groupby("source_file"):
        revenue = group.loc[group["concept"].eq("revenue"), "rands"].abs()
        if revenue.empty or not revenue.max():
            continue
        factor, largest = 1.0, revenue.max()
        while largest * factor < PLAUSIBLE_REVENUE_ZAR[0]:
            factor *= 1000
        while largest * factor > PLAUSIBLE_REVENUE_ZAR[1]:
            factor /= 1000
        if factor != 1.0:
            df.loc[group.index, "scale_fix"] = factor
    df["rands"] = df["rands"] * df["scale_fix"]

    # One figure per entity-year-concept: keep the first occurrence in document
    # order, which is the primary statement rather than a note repeating it.
    df = df.sort_values(["entity_id", "fiscal_year", "concept", "page", "table_index"])
    picked = df.drop_duplicates(["entity_id", "fiscal_year", "concept"], keep="first")

    wide = picked.pivot_table(
        index=["entity_id", "entity_name", "fiscal_year"],
        columns="concept", values="rands", aggfunc="first",
    ).reset_index()

    for concept in CONCEPT_PATTERNS:
        if concept not in wide.columns:
            wide[concept] = pd.NA

    rows = []

    def emit(record, line_item, metric_type, value, note=""):
        if pd.isna(value):
            return
        rows.append({
            "entity_id": record["entity_id"],
            "entity_name": record["entity_name"],
            "fiscal_year": record["fiscal_year"],
            "line_item": line_item,
            "metric_type": metric_type,
            "reported_value": float(value),
            "source": "OCR extract of published statements",
            "note": note,
        })

    for _, r in wide.iterrows():
        revenue = r["revenue"]
        cos = r["cost_of_sales"]
        staff = r["employee_costs"]
        ebitda = (
            abs(r["operating_profit"]) + abs(r["depreciation"])
            if pd.notna(r["operating_profit"]) and pd.notna(r["depreciation"])
            else pd.NA
        )

        emit(r, "Total revenue", "amount", revenue)
        emit(r, "Cost of sales and supplier payments", "amount", cos)
        emit(r, "Employee costs", "amount", staff)
        emit(r, "Taxation paid", "amount", r["tax_paid"])
        emit(r, "Net cash from operating activities", "amount", r["net_cash_ops"])
        emit(r, "EBITDA proxy", "amount", ebitda, "operating profit + D&A")

        if pd.notna(revenue) and revenue:
            if pd.notna(cos):
                emit(r, "Gross margin proxy", "ratio",
                     (abs(revenue) - abs(cos)) / abs(revenue), "(revenue - cost of sales) / revenue")
            if pd.notna(ebitda):
                emit(r, "EBITDA margin proxy", "ratio", ebitda / abs(revenue), "EBITDA / revenue")
            if pd.notna(staff):
                emit(r, "Employee cost ratio", "ratio", abs(staff) / abs(revenue), "employee costs / revenue")
        # Export revenue share is left blank: it needs a segment note that
        # these condensed packs mostly do not carry.

    return pd.DataFrame(rows), picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="print a coverage summary")
    args = parser.parse_args()

    extracted = pd.read_csv(EXTRACTED)
    reported, picked = build(extracted)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reported.sort_values(["entity_id", "fiscal_year", "line_item"]).to_csv(OUT_PATH, index=False)
    print(f"{len(reported)} benchmark rows -> {OUT_PATH}")
    print(f"entities covered: {reported['entity_id'].nunique()} of 20")

    if args.report:
        print("\n--- Rows per entity-year ---")
        pivot = reported.pivot_table(
            index=["entity_id", "entity_name"], columns="fiscal_year",
            values="line_item", aggfunc="count", fill_value=0,
        )
        print(pivot.to_string())

        print("\n--- Revenue present? (drives every amount comparison) ---")
        rev = reported[reported["line_item"] == "Total revenue"]
        have = set(zip(rev["entity_id"], rev["fiscal_year"]))
        missing = [
            f"{e} {fy}" for e in sorted(reported["entity_id"].unique())
            for fy in TARGET_YEARS if (e, fy) not in have
        ]
        print(f"entity-years with revenue: {len(have)}")
        print("missing revenue:", ", ".join(missing) if missing else "none")

        print("\n--- Assumptions used ---")
        print(picked.groupby("month_source").size().to_string())
        print(picked.groupby("scale_source").size().to_string())


if __name__ == "__main__":
    main()
