# -*- coding: utf-8 -*-
"""Pseudo consolidated financial statements per entity, FY2024-FY2026.

Reconstructs, from bank-side transaction flows only, a three-year statement pack
for each of the 20 entities:

    * income statement   - cash-basis proxy (revenue, cost of sales, employee
                           costs, EBITDA proxy, tax paid, net cash earnings)
    * cash flow          - direct method (operating receipts and payments,
                           intragroup/financing flows, net movement in cash)
    * contingent note    - trade finance instruments: notional, not cash

and then supports comparison against real published financials to surface
discrepancies.

What this can and cannot be:

  - It is a *cash* view. There are no accruals, no depreciation, no
    non-cash revenue, no working-capital movements, so "EBITDA proxy" is
    operating cash margin, not reported EBITDA.
  - Trade finance instruments are notional exposures. They are deliberately
    excluded from both revenue and cash flow and reported as a separate
    contingent note - adding them would double count the underlying trade.
  - The flows only cover what touches this bank, so absolute rands will be a
    fraction of reported figures. `compare_to_reported` handles this by
    deriving a per-entity coverage factor from revenue and flagging any line
    whose implied coverage deviates from it - that deviation, rather than the
    raw rand gap, is the real discrepancy signal.

Usage:
    python financial_statements.py                 # build + write outputs/statements/
    python financial_statements.py --entity E09    # print one entity's pack
    python financial_statements.py --compare benchmarks/reported_financials.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from augstdhackathon import BASE_DIR, build_entity_dimension, load_all

OUTPUT_DIR = BASE_DIR / "outputs" / "statements"
BENCHMARK_DIR = BASE_DIR / "benchmarks"
REPORTED_TEMPLATE = BENCHMARK_DIR / "reported_financials_template.csv"

PERIOD = "fiscal_year"

# Line items that are compared against published figures, and whether they are
# a rand amount (scaled by coverage) or a ratio (compared directly, in points).
COMPARABLE_LINES = {
    "Total revenue": "amount",
    "Cost of sales and supplier payments": "amount",
    "Employee costs": "amount",
    "EBITDA proxy": "amount",
    "Taxation paid": "amount",
    "Net cash from operating activities": "amount",
    "Gross margin proxy": "ratio",
    "EBITDA margin proxy": "ratio",
    "Employee cost ratio": "ratio",
    "Export revenue share": "ratio",
}

# Heuristic ranges for large JSE/LSE-listed corporates, used only to flag
# proxy ratios that cannot plausibly match any real annual report.
PLAUSIBLE_RATIO_RANGES = {
    "Gross margin proxy": (0.10, 0.70),
    "EBITDA margin proxy": (0.05, 0.45),
    "Employee cost ratio": (0.03, 0.35),
    "Effective cash tax rate": (0.10, 0.35),
}

MATERIALITY = 0.10  # 10% deviation from the entity's revenue coverage


# --------------------------------------------------------------------------
# Component aggregations
# --------------------------------------------------------------------------

def _sum_by(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """Total value_zar by entity x fiscal year for the rows matching `mask`."""
    return df.loc[mask].groupby(["entity_id", PERIOD])["value_zar"].sum()


def _components(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every raw building block of the statements, as entity x year columns."""
    tb = frames["transactional_banking"]
    xb = frames["cross_border_payments"]

    inbound, outbound = xb["direction"].eq("inbound"), xb["direction"].eq("outbound")
    trade = xb["corridor_type"].eq("trade")
    other = xb["corridor_type"].eq("other")
    interco = xb["corridor_type"].eq("intercompany")

    parts = {
        # Transactional banking: leg_type maps one-to-one onto direction.
        "collections": _sum_by(tb, tb["leg_type"].eq("collections")),
        "supplier_payments": _sum_by(tb, tb["leg_type"].eq("supplier_payments")),
        "payroll": _sum_by(tb, tb["leg_type"].eq("payroll")),
        "tax": _sum_by(tb, tb["leg_type"].eq("tax")),
        "sweeps_in": _sum_by(tb, tb["leg_type"].eq("intercompany_sweeps") & tb["direction"].eq("inbound")),
        "sweeps_out": _sum_by(tb, tb["leg_type"].eq("intercompany_sweeps") & tb["direction"].eq("outbound")),
        # Cross-border, split so intragroup funding never lands in revenue.
        "xb_trade_in": _sum_by(xb, inbound & trade),
        "xb_trade_out": _sum_by(xb, outbound & trade),
        "xb_other_in": _sum_by(xb, inbound & other),
        "xb_other_out": _sum_by(xb, outbound & other),
        "xb_interco_in": _sum_by(xb, inbound & interco),
        "xb_interco_out": _sum_by(xb, outbound & interco),
    }
    return pd.DataFrame(parts).fillna(0.0).sort_index()


def _contingent(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Trade finance exposures for the off-balance-sheet note."""
    tf = frames["trade_finance"]
    grouped = tf.groupby(["entity_id", PERIOD])

    note = pd.DataFrame({
        "lc_notional": _sum_by(tf, tf["instrument_type"].eq("letters_of_credit")),
        "guarantee_notional": _sum_by(tf, tf["instrument_type"].eq("guarantees")),
        "export_collection_notional": _sum_by(tf, tf["instrument_type"].eq("export_collections")),
        "export_notional": _sum_by(tf, tf["direction"].eq("export")),
        "import_notional": _sum_by(tf, tf["direction"].eq("import")),
        "open_exposure": _sum_by(tf, tf["status"].isin(["active", "issued"])),
        "settled_notional": _sum_by(tf, tf["status"].eq("settled")),
    }).fillna(0.0)

    weighted = tf.assign(_w=tf["tenor_days"] * tf["value_zar"]).groupby(["entity_id", PERIOD])
    note["weighted_avg_tenor_days"] = weighted["_w"].sum() / weighted["value_zar"].sum()
    note["instrument_count"] = grouped.size()
    note["counterparty_countries"] = grouped["counterparty_country"].nunique()
    return note.sort_index()


# --------------------------------------------------------------------------
# Statements
# --------------------------------------------------------------------------

def build_income_statement(components: pd.DataFrame) -> pd.DataFrame:
    """Cash-basis income statement proxy, one row per entity x fiscal year."""
    c = components
    out = pd.DataFrame(index=c.index)

    out["Revenue - domestic collections"] = c["collections"]
    out["Revenue - cross-border trade receipts"] = c["xb_trade_in"]
    out["Other operating income"] = c["xb_other_in"]
    out["Total revenue"] = (
        out["Revenue - domestic collections"]
        + out["Revenue - cross-border trade receipts"]
        + out["Other operating income"]
    )

    out["Cost of sales - domestic suppliers"] = -c["supplier_payments"]
    out["Cost of sales - cross-border trade payments"] = -c["xb_trade_out"]
    out["Cost of sales and supplier payments"] = (
        out["Cost of sales - domestic suppliers"] + out["Cost of sales - cross-border trade payments"]
    )
    out["Gross profit proxy"] = out["Total revenue"] + out["Cost of sales and supplier payments"]

    out["Employee costs"] = -c["payroll"]
    out["Other operating expenses"] = -c["xb_other_out"]
    out["EBITDA proxy"] = (
        out["Gross profit proxy"] + out["Employee costs"] + out["Other operating expenses"]
    )

    out["Taxation paid"] = -c["tax"]
    out["Net cash earnings proxy"] = out["EBITDA proxy"] + out["Taxation paid"]

    revenue = out["Total revenue"].replace(0, np.nan)
    out["Gross margin proxy"] = out["Gross profit proxy"] / revenue
    out["EBITDA margin proxy"] = out["EBITDA proxy"] / revenue
    out["Employee cost ratio"] = -out["Employee costs"] / revenue
    out["Export revenue share"] = out["Revenue - cross-border trade receipts"] / revenue
    out["Effective cash tax rate"] = np.where(
        out["EBITDA proxy"] > 0, -out["Taxation paid"] / out["EBITDA proxy"], np.nan
    )
    out["Revenue growth"] = out.groupby(level="entity_id")["Total revenue"].pct_change()
    return out


def build_cash_flow_statement(components: pd.DataFrame) -> pd.DataFrame:
    """Direct-method cash flow statement, one row per entity x fiscal year."""
    c = components
    out = pd.DataFrame(index=c.index)

    out["Receipts from customers"] = c["collections"] + c["xb_trade_in"] + c["xb_other_in"]
    out["Payments to suppliers"] = -(c["supplier_payments"] + c["xb_trade_out"] + c["xb_other_out"])
    out["Payments to employees"] = -c["payroll"]
    out["Taxation paid"] = -c["tax"]
    out["Net cash from operating activities"] = (
        out["Receipts from customers"]
        + out["Payments to suppliers"]
        + out["Payments to employees"]
        + out["Taxation paid"]
    )

    out["Intragroup sweeps received"] = c["sweeps_in"]
    out["Intragroup sweeps paid"] = -c["sweeps_out"]
    out["Cross-border intragroup received"] = c["xb_interco_in"]
    out["Cross-border intragroup paid"] = -c["xb_interco_out"]
    out["Net intragroup and financing flows"] = (
        out["Intragroup sweeps received"]
        + out["Intragroup sweeps paid"]
        + out["Cross-border intragroup received"]
        + out["Cross-border intragroup paid"]
    )

    out["Net movement in cash"] = (
        out["Net cash from operating activities"] + out["Net intragroup and financing flows"]
    )

    inflows = c[["collections", "xb_trade_in", "xb_other_in", "sweeps_in", "xb_interco_in"]].sum(axis=1)
    outflows = c[["supplier_payments", "xb_trade_out", "xb_other_out", "payroll", "tax",
                  "sweeps_out", "xb_interco_out"]].sum(axis=1)
    out["Memo - gross cash inflows"] = inflows
    out["Memo - gross cash outflows"] = -outflows
    out["Memo - foreign currency gross flows"] = c[
        ["xb_trade_in", "xb_trade_out", "xb_other_in", "xb_other_out",
         "xb_interco_in", "xb_interco_out"]
    ].sum(axis=1)
    out["Operating cash conversion"] = out["Net cash from operating activities"] / out[
        "Receipts from customers"
    ].replace(0, np.nan)
    return out


def build_contingent_note(components_note: pd.DataFrame) -> pd.DataFrame:
    """Trade finance note - notional exposure, excluded from cash and revenue."""
    n = components_note
    out = pd.DataFrame(index=n.index)
    out["Letters of credit - notional"] = n["lc_notional"]
    out["Guarantees - notional"] = n["guarantee_notional"]
    out["Export collections - notional"] = n["export_collection_notional"]
    out["Total trade finance notional"] = (
        out["Letters of credit - notional"]
        + out["Guarantees - notional"]
        + out["Export collections - notional"]
    )
    out["Of which export"] = n["export_notional"]
    out["Of which import"] = n["import_notional"]
    out["Open exposure at period end"] = n["open_exposure"]
    out["Settled during period"] = n["settled_notional"]
    out["Weighted average tenor (days)"] = n["weighted_avg_tenor_days"]
    out["Instrument count"] = n["instrument_count"]
    out["Counterparty countries"] = n["counterparty_countries"]
    return out


def build_statement_pack(frames: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Full three-statement pack as a tidy long frame.

    Columns: entity_id, entity_name, sector, fiscal_year, statement, line_item,
    line_order, value, is_ratio.
    """
    frames = frames or load_all()
    components = _components(frames)
    note_components = _contingent(frames)

    statements = {
        "income_statement": build_income_statement(components),
        "cash_flow": build_cash_flow_statement(components),
        "contingent_note": build_contingent_note(note_components),
    }

    tidy = []
    for statement, wide in statements.items():
        order = {line: i for i, line in enumerate(wide.columns)}
        long = (
            wide.reset_index()
            .melt(id_vars=["entity_id", PERIOD], var_name="line_item", value_name="value")
        )
        long["statement"] = statement
        long["line_order"] = long["line_item"].map(order)
        tidy.append(long)

    pack = pd.concat(tidy, ignore_index=True)
    pack["is_ratio"] = pack["line_item"].str.contains(
        "margin|ratio|share|rate|growth|conversion|tenor|count|countries", case=False
    )

    dim = build_entity_dimension(frames)
    pack = pack.join(dim, on="entity_id")
    pack = pack[["entity_id", "entity_name", "sector", PERIOD, "statement",
                 "line_item", "line_order", "value", "is_ratio"]]
    return pack.sort_values(["entity_id", "statement", "line_order", PERIOD]).reset_index(drop=True)


def entity_statement(pack: pd.DataFrame, entity_id: str, statement: str | None = None) -> pd.DataFrame:
    """One entity's statements as years-across-columns, the way a report reads."""
    sub = pack[pack["entity_id"] == entity_id]
    if statement:
        sub = sub[sub["statement"] == statement]
    wide = sub.pivot_table(
        index=["statement", "line_order", "line_item"], columns=PERIOD, values="value"
    ).sort_index(level=["statement", "line_order"])

    years = list(wide.columns)
    is_ratio = (
        sub.drop_duplicates("line_item").set_index("line_item")["is_ratio"]
        .reindex(wide.index.get_level_values("line_item")).values
    )
    # A three-year total only means something for flows, not for ratios.
    wide["3yr_total"] = np.where(is_ratio, np.nan, wide[years].sum(axis=1))
    return wide.droplevel("line_order")


def format_statement(pack: pd.DataFrame, entity_id: str, unit: float = 1e6) -> str:
    """Render an entity's pack as printable text, amounts in millions."""
    name = pack.loc[pack["entity_id"] == entity_id, "entity_name"].iloc[0]
    lines = [f"{entity_id}  {name}", f"Pseudo consolidated statements (ZAR m unless stated)", ""]

    ratio_rows = set(pack.loc[(pack["entity_id"] == entity_id) & pack["is_ratio"], "line_item"])

    for statement in ["income_statement", "cash_flow", "contingent_note"]:
        frame = entity_statement(pack, entity_id, statement).droplevel("statement")
        display = pd.DataFrame(
            [
                [
                    "-" if pd.isna(v) else (f"{v:,.2f}" if row in ratio_rows else f"{v / unit:,.1f}")
                    for v in frame.loc[row]
                ]
                for row in frame.index
            ],
            index=frame.index,
            columns=frame.columns,
        )
        lines.append(statement.replace("_", " ").upper())
        lines.append(display.to_string())
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Comparison against real published financials
# --------------------------------------------------------------------------

def write_reported_template(pack: pd.DataFrame, path: Path = REPORTED_TEMPLATE) -> Path:
    """Write the CSV to key real annual-report figures into.

    Fill in `reported_value` (rands for amounts, decimal fraction for ratios)
    and `source`; leave rows blank to skip them. Feed the result to
    `compare_to_reported`.
    """
    rows = pack[pack["line_item"].isin(COMPARABLE_LINES)][
        ["entity_id", "entity_name", PERIOD, "line_item"]
    ].drop_duplicates()
    rows["metric_type"] = rows["line_item"].map(COMPARABLE_LINES)
    rows["reported_value"] = np.nan
    rows["source"] = ""
    rows["note"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort_values(["entity_id", PERIOD, "line_item"]).to_csv(path, index=False)
    return path


def compare_to_reported(
    pack: pd.DataFrame, reported: pd.DataFrame | Path, materiality: float = MATERIALITY
) -> pd.DataFrame:
    """Compare the proxy statements to real reported figures.

    Because the transaction data is a bank-side sample, a raw rand gap is not
    a discrepancy on its own. So for each entity-year we derive a coverage
    factor from revenue::

        revenue_coverage = reported revenue / proxy revenue

    and express every other amount line as its own implied coverage. A line
    whose implied coverage sits within `materiality` of the revenue coverage is
    consistent with the reported accounts, just at a different scale. One that
    does not is a genuine structural discrepancy - the flows in that category
    are under- or over-represented relative to how the entity actually reports.

    Ratio lines carry no scale, so they are compared directly and the gap is
    reported in percentage points.
    """
    if isinstance(reported, (str, Path)):
        reported = pd.read_csv(reported)

    reported = reported.dropna(subset=["reported_value"])
    if reported.empty:
        raise ValueError("No reported_value rows populated - fill the template first")

    # Some lines (taxation paid) appear in both statements with the same value.
    proxy = (
        pack[pack["line_item"].isin(COMPARABLE_LINES)][
            ["entity_id", "entity_name", PERIOD, "line_item", "value", "is_ratio"]
        ]
        .drop_duplicates(subset=["entity_id", PERIOD, "line_item"])
        .rename(columns={"value": "proxy_value"})
    )

    merged = proxy.merge(
        reported[["entity_id", PERIOD, "line_item", "reported_value", "source"]],
        on=["entity_id", PERIOD, "line_item"],
        how="inner",
    )
    merged["metric_type"] = merged["line_item"].map(COMPARABLE_LINES)

    # Amounts are compared on sign-agnostic magnitude (expenses are negative here).
    amount = merged["metric_type"].eq("amount")
    merged["implied_coverage"] = np.where(
        amount & merged["proxy_value"].abs().gt(0),
        merged["reported_value"].abs() / merged["proxy_value"].abs(),
        np.nan,
    )

    revenue_coverage = (
        merged.loc[merged["line_item"].eq("Total revenue"), ["entity_id", PERIOD, "implied_coverage"]]
        .rename(columns={"implied_coverage": "revenue_coverage"})
    )
    merged = merged.merge(revenue_coverage, on=["entity_id", PERIOD], how="left")

    # Scale the proxy up to the entity's revenue coverage, then compare in rands.
    merged["scaled_proxy_value"] = np.where(
        amount, merged["proxy_value"] * merged["revenue_coverage"], merged["proxy_value"]
    )
    merged["variance"] = np.where(
        amount,
        merged["reported_value"].abs() - merged["scaled_proxy_value"].abs(),
        merged["reported_value"] - merged["proxy_value"],
    )
    denominator = np.where(amount, merged["scaled_proxy_value"].abs(), merged["proxy_value"].abs())
    merged["variance_pct"] = np.where(
        denominator > 0, merged["variance"] / denominator, np.nan
    )
    merged["variance_pp"] = np.where(~amount, merged["variance"] * 100, np.nan)
    merged["coverage_deviation"] = merged["implied_coverage"] / merged["revenue_coverage"] - 1

    deviation = np.where(amount, merged["coverage_deviation"], merged["variance_pct"])
    merged["material"] = np.abs(deviation) > materiality
    merged["direction"] = np.where(
        merged["material"],
        np.where(np.nan_to_num(deviation) > 0, "under-represented in data", "over-represented in data"),
        "consistent",
    )

    cols = ["entity_id", "entity_name", PERIOD, "statement_line", "metric_type", "proxy_value",
            "revenue_coverage", "scaled_proxy_value", "reported_value", "variance",
            "variance_pct", "variance_pp", "implied_coverage", "coverage_deviation",
            "material", "direction", "source"]
    return (
        merged.rename(columns={"line_item": "statement_line"})
        .reindex(columns=cols)
        .sort_values(["entity_id", PERIOD, "statement_line"])
        .reset_index(drop=True)
    )


def plausibility_report(pack: pd.DataFrame) -> pd.DataFrame:
    """Flag proxy ratios that no real annual report could match.

    Runs without any reported data - it checks the derived ratios against
    broad ranges typical of large listed corporates, so you know which lines
    will not reconcile before you go and key in the real numbers.
    """
    ratios = pack[pack["line_item"].isin(PLAUSIBLE_RATIO_RANGES)].copy()
    bounds = pd.DataFrame(PLAUSIBLE_RATIO_RANGES, index=["low", "high"]).T
    ratios = ratios.join(bounds, on="line_item")
    ratios["status"] = np.select(
        [ratios["value"] < ratios["low"], ratios["value"] > ratios["high"]],
        ["below plausible range", "above plausible range"],
        default="plausible",
    )
    return ratios[["entity_id", "entity_name", "sector", PERIOD, "line_item",
                   "value", "low", "high", "status"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", help="Print this entity's statement pack, e.g. E09")
    parser.add_argument("--compare", type=Path, help="CSV of reported financials to compare against")
    parser.add_argument("--materiality", type=float, default=MATERIALITY)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    pack = build_statement_pack()

    if args.entity:
        print(format_statement(pack, args.entity))
    else:
        print(format_statement(pack, pack["entity_id"].iloc[0]))

    checks = plausibility_report(pack)
    flagged = checks[checks["status"] != "plausible"]
    print(f"--- Plausibility: {len(flagged)} of {len(checks)} ratio observations outside range ---")
    if not flagged.empty:
        print(flagged.groupby(["line_item", "status"]).size().to_string())

    comparison = None
    if args.compare:
        comparison = compare_to_reported(pack, args.compare, args.materiality)
        print("\n--- Comparison to reported financials ---")
        print(comparison.to_string(index=False))
        print(f"\nMaterial discrepancies: {int(comparison['material'].sum())} of {len(comparison)}")

    if not args.no_write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pack.to_csv(OUTPUT_DIR / "statement_pack_long.csv", index=False)
        for statement in pack["statement"].unique():
            sub = pack[pack["statement"] == statement]
            sub.pivot_table(
                index=["entity_id", "entity_name", "line_order", "line_item"],
                columns=PERIOD, values="value",
            ).sort_index(level=["entity_id", "line_order"]).droplevel("line_order").to_csv(
                OUTPUT_DIR / f"{statement}.csv"
            )
        for entity_id in pack["entity_id"].unique():
            (OUTPUT_DIR / f"{entity_id}.txt").write_text(format_statement(pack, entity_id))
        checks.to_csv(OUTPUT_DIR / "plausibility_checks.csv", index=False)
        template = write_reported_template(pack)
        if comparison is not None:
            comparison.to_csv(OUTPUT_DIR / "comparison_to_reported.csv", index=False)
        print(f"\nWrote statements to {OUTPUT_DIR}")
        print(f"Key real financials into {template}, then rerun with --compare")


if __name__ == "__main__":
    main()
