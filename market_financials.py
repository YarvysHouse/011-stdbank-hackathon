# -*- coding: utf-8 -*-
"""Market-side financials: what the entities publish, in rands.

The dashboard compares two ledgers per entity:

    Syn Bank        what the bank can see - flows across its own rails
    Market          what the entity reports to the market - the published pack

The market side comes from `outputs/extracted/extracted_lines.csv`, the OCR
output of the 39 published statement PDFs. That file is raw: 10k label/value
rows with no line-item mapping, inconsistent period headers and units that are
only sometimes stated. This module turns it into the two numbers the dashboard
actually needs per entity-year - total income and total expenditure - and is
honest about how far it got:

    * income   revenue / turnover / total income / premium lines
    * spend    total expenses, cost of sales, direct + operating expenses
    * year     from the period column, else the table caption, else the
               filename; rows where none resolve are dropped
    * units    the stated scale when the table header gives one, otherwise
               millions - the house style of every pack in the set
    * currency converted at the static rates below, since these are annual
               figures being compared for shape, not settled

Hand-keyed figures always win over the OCR. Two files carry them, in order of
precedence: `benchmarks/extraction_worklist(extraction_worklist).csv` - the
worklist as checked back against the source PDFs, which also records how far
each line got (have / suspect / missing) - and then the older
`benchmarks/reported_financials_template.csv`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
EXTRACTED = BASE_DIR / "outputs" / "extracted" / "extracted_lines.csv"
TEMPLATE = BASE_DIR / "benchmarks" / "reported_financials_template.csv"
WORKLIST = BASE_DIR / "benchmarks" / "extraction_worklist(extraction_worklist).csv"

# The two summary metrics the reconciliation headlines on, and the worklist /
# template line item each is read from.
SUMMARY_LINES = {
    "Total revenue": "income",
    "Cost of sales and supplier payments": "expenditure",
}

# Static ZAR conversion. Period-average rates, deliberately not time-varying:
# the comparison is order-of-magnitude coverage, and a rate table per reporting
# date would imply a precision the OCR side does not have.
FX_TO_ZAR = {"ZAR": 1.0, "USD": 18.5, "EUR": 20.0, "GBP": 23.5}

# Absent a stated unit, these packs are in millions - every one in the set that
# does state a unit says so. Applied only to figures small enough that millions
# is the only reading that lands in a plausible range for a listed corporate.
DEFAULT_SCALE = 1e6
RAW_IF_ABOVE = 1e9  # already an absolute figure, don't scale it again

INCOME_PATTERNS = re.compile(
    r"(?i)^\s*(total\s+)?("
    r"revenue|turnover|total income|net operating income|"
    r"revenue from contracts?|income from contracts?|"
    r"gross written premium|net insurance premium|net earned premium"
    r")\b"
)

EXPENSE_PATTERNS = re.compile(
    r"(?i)("
    r"total (operating )?expen|cost of sales|cost of goods|"
    r"direct expenses|operating expenses|operating costs|"
    r"total costs and expenses"
    r")"
)

# Lines that read as income or expense but are neither: subtotals net of the
# thing we are measuring, or per-share and tax-effected variants.
EXCLUDE = re.compile(r"(?i)(per share|deferred|discontinued|other comprehensive)")

YEAR = re.compile(r"(20[12]\d)")
FILENAME_PAIR = re.compile(r"(\d{2})\s*&\s*(\d{2})")
FILENAME_RANGE = re.compile(r"(\d{2})\s*-\s*(\d{2})")

# A listed corporate's annual revenue or cost, in rands. Anything outside this
# is an OCR misread or a mis-scaled figure, not a real line.
PLAUSIBLE_ZAR = (1e8, 5e12)


def _year_from(*candidates: object) -> int | None:
    """First four-digit year found across the candidates, in priority order."""
    for candidate in candidates:
        found = YEAR.findall(str(candidate))
        if found:
            return int(found[0])
    return None


def _filename_years(source_file: str) -> list[int]:
    """Reporting years a filename covers, most recent first.

    `Vodacom Group - 24&25.pdf` covers 2025 and 2024; `BHB group - 23-25.pdf`
    covers 2025, 2024, 2023. Statement columns run most-recent-first, so the
    order here matches column order left to right.
    """
    pair = FILENAME_PAIR.search(source_file)
    if pair:
        years = [2000 + int(pair.group(1)), 2000 + int(pair.group(2))]
        return sorted(years, reverse=True)
    span = FILENAME_RANGE.search(source_file)
    if span:
        start, end = 2000 + int(span.group(1)), 2000 + int(span.group(2))
        return sorted(range(start, end + 1), reverse=True)
    single = re.search(r"\b(\d{2})\b", source_file)
    return [2000 + int(single.group(1))] if single else []


def _column_positions(frame: pd.DataFrame) -> pd.Series:
    """Position of each row's period column within its own table.

    `tidy()` melts column by column in the original left-to-right order, so
    the order in which a column header first appears within one table is its
    position on the page.
    """
    keys = ["source_file", "page", "table_index"]
    order = (
        frame.reset_index()
        .groupby(keys + ["column_header"], dropna=False)["index"]
        .min()
        .reset_index()
    )
    order["position"] = order.groupby(keys)["index"].rank(method="dense").astype(int) - 1
    merged = frame.merge(
        order[keys + ["column_header", "position"]], on=keys + ["column_header"], how="left"
    )
    return merged["position"].fillna(0).astype(int)


def _to_zar(value: float, currency: str, scale: float | None) -> float:
    """Absolute rands from a reported figure, its currency and stated scale."""
    magnitude = abs(value)
    if scale and scale > 0:
        value = value * scale
    elif magnitude < RAW_IF_ABOVE:
        value = value * DEFAULT_SCALE
    return value * FX_TO_ZAR.get(str(currency).upper(), 1.0)


def _classify(label: str) -> str | None:
    if EXCLUDE.search(label):
        return None
    if INCOME_PATTERNS.search(label):
        return "income"
    if EXPENSE_PATTERNS.search(label):
        return "expenditure"
    return None


def load_market_financials(path: Path = EXTRACTED) -> pd.DataFrame:
    """Tidy market income and expenditure per entity-year.

    Columns: entity_id, entity_name, fiscal_year, metric, value_zar, source,
    row_label, observations.
    """
    if not path.exists():
        return pd.DataFrame(
            columns=["entity_id", "entity_name", "fiscal_year", "metric",
                     "value_zar", "source", "row_label", "observations"]
        )

    raw = pd.read_csv(path)
    raw["row_label"] = raw["row_label"].fillna("").astype(str).str.strip()
    raw["column_header"] = raw["column_header"].fillna("?").astype(str)
    raw["metric"] = raw["row_label"].map(_classify)
    tagged = raw[raw["metric"].notna() & raw["value"].notna()].copy()
    if tagged.empty:
        return pd.DataFrame(
            columns=["entity_id", "entity_name", "fiscal_year", "metric",
                     "value_zar", "source", "row_label", "observations"]
        )

    # Three ways to date a row, best first. The last one reads the period off
    # the column's position on the page against the years in the filename,
    # which is a convention rather than a statement of fact - so it is tracked
    # separately and shown as lower confidence in the dashboard.
    positions = _column_positions(tagged).values
    years, bases = [], []
    for position, (_, row) in zip(positions, tagged.iterrows()):
        year = _year_from(row["column_header"])
        basis = "period column"
        if year is None:
            year = _year_from(row["table_title"])
            basis = "table caption"
        if year is None:
            candidates = _filename_years(str(row["source_file"]))
            year = candidates[position] if position < len(candidates) else None
            basis = "column order"
        years.append(year)
        bases.append(basis if year is not None else None)
    tagged["year"] = years
    tagged["year_basis"] = bases
    tagged = tagged[tagged["year"].notna()].copy()
    if tagged.empty:
        return pd.DataFrame(
            columns=["entity_id", "entity_name", "fiscal_year", "metric",
                     "value_zar", "source", "row_label", "observations"]
        )

    tagged["fiscal_year"] = "FY" + tagged["year"].astype(int).astype(str)
    tagged["value_zar"] = [
        abs(_to_zar(value, currency, scale))
        for value, currency, scale in zip(
            tagged["value"], tagged["currency"], tagged["unit_scale"]
        )
    ]
    low, high = PLAUSIBLE_ZAR
    tagged = tagged[tagged["value_zar"].between(low, high)]
    if tagged.empty:
        return pd.DataFrame(
            columns=["entity_id", "entity_name", "fiscal_year", "metric",
                     "value_zar", "source", "row_label", "observations"]
        )

    # The same figure is re-read from several tables in a pack (income
    # statement, five-year review, segment note). The median of the readings
    # for one entity-year-metric is the robust choice: it shrugs off a single
    # OCR misread without averaging in a subtotal that happens to be larger.
    tidy = (
        tagged.groupby(["entity_id", "entity_name", "fiscal_year", "metric"])
        .agg(
            value_zar=("value_zar", "median"),
            observations=("value_zar", "size"),
            row_label=("row_label", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            year_basis=("year_basis", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
        )
        .reset_index()
    )
    tidy["source"] = "published statements (OCR)"
    return tidy


KEYED_COLUMNS = [
    "entity_id", "entity_name", "fiscal_year", "metric",
    "value_zar", "source", "row_label", "observations",
]

# The worklist's own column for the figure read off the PDF. The older template
# calls the same thing `reported_value`; both shapes are accepted so an analyst
# can upload either.
VALUE_COLUMNS = ("current_value", "reported_value")


def _value_column(frame: pd.DataFrame) -> str | None:
    for column in VALUE_COLUMNS:
        if column in frame.columns:
            return column
    return None


def load_worklist(path: Path = WORKLIST, upload: bytes | None = None) -> pd.DataFrame:
    """The extraction worklist, tidied - every reported line, not just the two.

    One row per entity-year-line with the figure read off the source PDF, plus
    the bookkeeping the worklist carries: how far the extraction got (`status`),
    how much the line matters (`priority`), which PDF it came from, and any note
    an earlier plausibility check left behind. Rows with no figure are kept, so
    the dashboard can show what is still outstanding rather than silently
    narrowing to what happens to be filled.

    Columns: entity_id, entity_name, fiscal_year, line_item, metric_type,
    reported_value, status, priority, source_pdf, where_to_look, note.
    """
    columns = ["entity_id", "entity_name", "fiscal_year", "line_item", "metric_type",
               "reported_value", "status", "priority", "source_pdf", "where_to_look", "note"]
    if upload is not None:
        frame = pd.read_csv(pd.io.common.BytesIO(upload))
    elif path.exists():
        frame = pd.read_csv(path)
    else:
        return pd.DataFrame(columns=columns)

    value_column = _value_column(frame)
    if value_column is None:
        return pd.DataFrame(columns=columns)

    frame = frame.copy()
    frame["reported_value"] = pd.to_numeric(frame[value_column], errors="coerce")
    for optional in ("status", "priority", "source_pdf", "where_to_look", "note"):
        if optional not in frame.columns:
            frame[optional] = pd.NA
    # A template upload carries no status column; a filled figure there is as
    # good as a worklist "have".
    frame["status"] = frame["status"].fillna(
        frame["reported_value"].notna().map({True: "have", False: "missing"})
    )
    if "metric_type" not in frame.columns:
        frame["metric_type"] = "amount"
    return frame[columns]


def _keyed_summary(worklist: pd.DataFrame, source: str) -> pd.DataFrame:
    """The two summary metrics, in the same shape the OCR loader returns."""
    if worklist.empty:
        return pd.DataFrame(columns=KEYED_COLUMNS)
    frame = worklist.dropna(subset=["reported_value"])
    frame = frame[frame["line_item"].isin(SUMMARY_LINES)].copy()
    if frame.empty:
        return pd.DataFrame(columns=KEYED_COLUMNS)
    frame["metric"] = frame["line_item"].map(SUMMARY_LINES)
    frame["value_zar"] = frame["reported_value"].abs()
    # A "suspect" reading still beats an OCR median, but it is labelled so the
    # dashboard can show which figures are standing on soft ground.
    frame["source"] = source + frame["status"].map(
        lambda s: " (suspect)" if str(s).lower() == "suspect" else ""
    )
    frame["observations"] = 1
    frame["row_label"] = frame["line_item"]
    return frame[KEYED_COLUMNS]


def load_template_overrides(path: Path = TEMPLATE, upload: bytes | None = None) -> pd.DataFrame:
    """Hand-keyed figures from the older reported template."""
    return _keyed_summary(load_worklist(path=path, upload=upload), "reported template")


def load_worklist_overrides(path: Path = WORKLIST) -> pd.DataFrame:
    """Hand-checked figures from the extraction worklist."""
    return _keyed_summary(load_worklist(path=path), "extraction worklist")


def market_view(upload: bytes | None = None) -> pd.DataFrame:
    """Market financials, best available source per entity-year-metric.

    Precedence, highest first: an uploaded file, the extraction worklist, the
    older reported template, then the OCR median. Each layer only fills what
    the layers above it left empty.
    """
    layers = [
        _keyed_summary(load_worklist(upload=upload), "uploaded figures") if upload else
        pd.DataFrame(columns=KEYED_COLUMNS),
        load_worklist_overrides(),
        load_template_overrides(),
        load_market_financials(),
    ]

    keys = ["entity_id", "fiscal_year", "metric"]
    stacked = pd.DataFrame(columns=KEYED_COLUMNS)
    for layer in layers:
        if layer.empty:
            continue
        if stacked.empty:
            stacked = layer
            continue
        fresh = layer.merge(stacked[keys], on=keys, how="left", indicator=True)
        fresh = fresh[fresh["_merge"] == "left_only"].drop(columns="_merge")
        stacked = pd.concat([stacked, fresh], ignore_index=True)
    return stacked


# Beyond this the two ledgers are not telling the same story about the line,
# and it is worth an analyst's eye. Ratios are compared in points, amounts as a
# share of the reported figure.
MATERIAL_VARIANCE = 0.25
MATERIAL_RATIO_POINTS = 0.10


def worklist_comparison(pack: pd.DataFrame, worklist: pd.DataFrame) -> pd.DataFrame:
    """Every worklist line against the computed statement pack, side by side.

    The computed side is a bank-flow proxy: it only ever sees what crosses Syn
    Bank's rails, so on an amount line it is expected to sit *below* the
    published figure, and the ratio between them is the entity's implied
    coverage rather than an error. On a ratio line the two are directly
    comparable, since a margin is scale-free.

    Columns: the worklist's own bookkeeping, plus computed_value, difference
    (reported less computed), variance_pct, coverage and material.
    """
    columns = ["entity_id", "entity_name", "fiscal_year", "line_item", "metric_type",
               "status", "priority", "source_pdf", "note", "reported_value",
               "computed_value", "difference", "variance_pct", "coverage",
               "material", "comparison"]
    if worklist.empty:
        return pd.DataFrame(columns=columns)

    computed = (
        pack[["entity_id", "fiscal_year", "line_item", "value", "is_ratio"]]
        .rename(columns={"value": "computed_value"})
        .drop_duplicates(["entity_id", "fiscal_year", "line_item"])
    )
    merged = worklist.merge(
        computed, on=["entity_id", "fiscal_year", "line_item"], how="left"
    )

    reported = pd.to_numeric(merged["reported_value"], errors="coerce")
    proxy = pd.to_numeric(merged["computed_value"], errors="coerce")
    is_ratio = merged["metric_type"].eq("ratio")

    merged["difference"] = reported - proxy
    merged["variance_pct"] = merged["difference"] / reported.abs().replace(0, pd.NA)
    # Coverage only means something on an amount: how much of the published
    # figure the bank can see. On a ratio it would be a comparison of two
    # percentages, which the variance column already carries.
    merged["coverage"] = (proxy / reported.replace(0, pd.NA)).where(~is_ratio)

    merged["material"] = np.where(
        is_ratio,
        merged["difference"].abs() > MATERIAL_RATIO_POINTS,
        merged["variance_pct"].abs() > MATERIAL_VARIANCE,
    )
    merged["comparison"] = np.where(
        reported.isna(), "no reported figure",
        np.where(proxy.isna(), "no computed figure", "compared"),
    )
    return merged[columns]


def bank_view(panel: pd.DataFrame, channels: pd.DataFrame) -> pd.DataFrame:
    """Syn Bank income and expenditure per entity-year, from bank rails only.

    Income is money arriving for the entity (customer collections and
    cross-border inflows); expenditure is money leaving (suppliers, employees,
    tax, cross-border outflows). Intragroup sweeps are excluded from both -
    they are treasury movement, not trade.
    """
    income = (
        panel["receipts_from_customers"].fillna(0) + panel["xb_inflow_zar"].fillna(0)
    )
    spend = (
        panel["payments_to_suppliers"].fillna(0)
        + panel["payments_to_employees"].fillna(0)
        + panel["taxes_paid"].fillna(0)
        + panel["xb_outflow_zar"].fillna(0)
    )
    return pd.DataFrame({
        "entity_id": panel["entity_id"],
        "entity_name": panel["entity_name"],
        "sector": panel["sector"],
        "fiscal_year": panel["fiscal_year"],
        "bank_income": income.values,
        "bank_expenditure": spend.values,
        "bank_total_flow": panel["total_flow_zar"].fillna(0).values,
        "txn_count": panel["tb_txn_count"].fillna(0).values
        + panel["xb_txn_count"].fillna(0).values,
    })


def reconcile(bank: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Join the two ledgers into one entity-year row with the discrepancy.

    Discrepancy is market less bank: positive means the entity reports more
    activity than the bank can see, which is the wallet the bank does not hold.
    """
    wide = market.pivot_table(
        index=["entity_id", "fiscal_year"], columns="metric", values="value_zar"
    ).reset_index()
    for column in ("income", "expenditure"):
        if column not in wide.columns:
            wide[column] = pd.NA
    wide = wide.rename(
        columns={"income": "market_income", "expenditure": "market_expenditure"}
    )

    merged = bank.merge(wide, on=["entity_id", "fiscal_year"], how="left")
    # An all-missing market column merges in as object dtype, which turns the
    # arithmetic below into a zero-division rather than a NaN.
    for column in ("market_income", "market_expenditure"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    for metric in ("income", "expenditure"):
        market = merged[f"market_{metric}"]
        denominator = market.replace(0, pd.NA)
        merged[f"{metric}_gap"] = market - merged[f"bank_{metric}"]
        merged[f"{metric}_gap_pct"] = merged[f"{metric}_gap"] / denominator
        merged[f"{metric}_coverage"] = merged[f"bank_{metric}"] / denominator

    merged["has_market"] = merged[["market_income", "market_expenditure"]].notna().any(axis=1)
    return merged


if __name__ == "__main__":
    view = market_view()
    print(f"{len(view)} market observations, {view['entity_id'].nunique()} entities")
    print(view["source"].value_counts().to_string())
    print(view.head(20).to_string(index=False))

    work = load_worklist()
    print(f"\nworklist: {len(work)} lines, "
          f"{int(work['reported_value'].notna().sum())} with a figure")
    print(work["status"].value_counts().to_string())
