# -*- coding: utf-8 -*-
"""Standard Bank hackathon: entity-level analytical base tables.

Builds three cleaned transaction-level DataFrames (transactional banking, trade
finance, cross-border payments), each sorted by ``entity_id`` then ``date``, and
rolls them up into an entity x fiscal-period panel whose columns line up with
line items you can read off a published annual report:

    receipts / supplier payments / payroll / tax paid / net operating cash flow
    export vs import trade exposure, contingent (guarantee) exposure
    FX inflow, FX outflow, net FX, exposure by currency pair

The data spans 2023-07-01 to 2026-06-30 — exactly three July-June fiscal years —
so periods are labelled on a June year-end (FY2024 = Jul 2023 to Jun 2024),
which is the convention most of these issuers report on.

Note on comparability: the transaction values are synthetic and only cover the
flows that touch the bank, so absolute ZAR will not tie to reported revenue or
opex. Compare *shape* instead — growth rates, period mix, sector mix, FX
dependence — for which the panel also carries indexed and share-of-total
columns.

Usage:
    python augstdhackathon.py                # build panel + write outputs/
    python augstdhackathon.py --eda          # + ABT profiling tables
    python augstdhackathon.py --plots        # + exploratory charts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "outputs"

FISCAL_YEAR_END_MONTH = 6  # June year-end -> FY2024 == Jul 2023 .. Jun 2024

DATASETS = {
    "transactional_banking": DATA_DIR / "transactional_banking.csv",
    "trade_finance": DATA_DIR / "trade_finance.csv",
    "cross_border_payments": DATA_DIR / "cross_border_payments.csv",
}

SORT_KEYS = ["entity_id", "date"]

# Maps the raw leg types onto the cash-flow-statement line they proxy.
LEG_TYPE_TO_STATEMENT_LINE = {
    "collections": "receipts_from_customers",
    "supplier_payments": "payments_to_suppliers",
    "payroll": "payments_to_employees",
    "tax": "taxes_paid",
    "intercompany_sweeps": "intercompany_transfers",
}


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------

def _add_fiscal_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Attach fiscal year / half / quarter labels off a June year-end."""
    month, year = df["date"].dt.month, df["date"].dt.year
    offset = (month > FISCAL_YEAR_END_MONTH).astype(int)

    fiscal_year = year + offset
    # Months since the start of the fiscal year (Jul = 0 ... Jun = 11).
    fiscal_month = (month - FISCAL_YEAR_END_MONTH - 1) % 12

    df["fiscal_year"] = "FY" + fiscal_year.astype(str)
    df["fiscal_half"] = df["fiscal_year"] + "H" + (fiscal_month // 6 + 1).astype(str)
    df["fiscal_quarter"] = df["fiscal_year"] + "Q" + (fiscal_month // 3 + 1).astype(str)
    df["calendar_month"] = df["date"].dt.to_period("M").astype(str)
    return df


def load_dataset(name: str, path: Path | None = None) -> pd.DataFrame:
    """Load one raw CSV, clean it, and sort it by entity then date.

    Cleaning applied (all three files carry the same defects):
      - exact duplicate rows dropped, duplicated ids flagged rather than dropped
      - ``date`` parsed to datetime, fiscal period labels attached
      - text columns stripped, currency/country casing normalised
      - the single value column aliased to ``value_zar`` so the three frames
        share a schema
    """
    path = path or DATASETS[name]
    df = pd.read_csv(path)

    df = df.drop_duplicates()

    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].str.strip()

    if "currency" in df:
        df["currency"] = df["currency"].str.upper()
    if "counterparty_country" in df:
        df["counterparty_country"] = df["counterparty_country"].fillna("Unknown")

    df["date"] = pd.to_datetime(df["date"])
    df = _add_fiscal_periods(df)

    # Unify the value column name across datasets (banking calls it amount_zar).
    if "amount_zar" in df:
        df = df.rename(columns={"amount_zar": "value_zar"})

    id_col = "instrument_id" if "instrument_id" in df else "transaction_id"
    df["is_duplicate_id"] = df[id_col].duplicated(keep=False)

    if "leg_type" in df:
        df["statement_line"] = df["leg_type"].map(LEG_TYPE_TO_STATEMENT_LINE)

    # Signed value: money in is positive, money out negative.
    outflow = {"outbound", "import"}
    df["signed_value_zar"] = np.where(
        df["direction"].isin(outflow), -df["value_zar"], df["value_zar"]
    )

    df["dataset"] = name
    return df.sort_values(SORT_KEYS).reset_index(drop=True)


def load_all() -> dict[str, pd.DataFrame]:
    """Load all three datasets, keyed by dataset name."""
    return {name: load_dataset(name) for name in DATASETS}


def build_entity_dimension(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The 20 entities the transactions are about, indexed by entity_id."""
    cols = ["entity_id", "entity_name", "sector"]
    dim = (
        pd.concat([df[cols] for df in frames.values()])
        .drop_duplicates()
        .sort_values("entity_id")
        .set_index("entity_id")
    )
    if dim.index.has_duplicates:
        raise ValueError("entity_id maps to more than one name/sector")
    return dim


# --------------------------------------------------------------------------
# Entity-level aggregation (one row per entity per period)
# --------------------------------------------------------------------------

def _pivot_sum(df: pd.DataFrame, period: str, column: str, prefix: str) -> pd.DataFrame:
    """Sum value_zar by entity x period, pivoted wide on `column`."""
    wide = (
        df.pivot_table(
            index=["entity_id", period],
            columns=column,
            values="value_zar",
            aggfunc="sum",
            fill_value=0.0,
        )
        .add_prefix(prefix)
    )
    wide.columns.name = None
    return wide


def banking_summary(df: pd.DataFrame, period: str = "fiscal_year") -> pd.DataFrame:
    """Cash-flow-statement shaped view of transactional banking."""
    lines = _pivot_sum(df, period, "statement_line", "")
    channels = _pivot_sum(df, period, "channel", "channel_")

    base = df.groupby(["entity_id", period]).agg(
        tb_txn_count=("value_zar", "size"),
        tb_total_value_zar=("value_zar", "sum"),
        tb_avg_txn_zar=("value_zar", "mean"),
        tb_median_txn_zar=("value_zar", "median"),
        tb_net_flow_zar=("signed_value_zar", "sum"),
    )

    out = base.join(lines).join(channels)
    receipts = out.get("receipts_from_customers", 0.0)
    opex = (
        out.get("payments_to_suppliers", 0.0)
        + out.get("payments_to_employees", 0.0)
        + out.get("taxes_paid", 0.0)
    )
    out["net_operating_cash_flow"] = receipts - opex
    # Cash conversion: how much of every rand received survives operating outflows.
    out["operating_margin_proxy"] = np.where(receipts > 0, out["net_operating_cash_flow"] / receipts, np.nan)
    return out


def trade_finance_summary(df: pd.DataFrame, period: str = "fiscal_year") -> pd.DataFrame:
    """Trade instruments as contingent exposure, split export vs import."""
    instruments = _pivot_sum(df, period, "instrument_type", "tf_")
    directions = _pivot_sum(df, period, "direction", "tf_")

    base = df.groupby(["entity_id", period]).agg(
        tf_instrument_count=("value_zar", "size"),
        tf_total_value_zar=("value_zar", "sum"),
        tf_avg_value_zar=("value_zar", "mean"),
        tf_net_trade_zar=("signed_value_zar", "sum"),
        tf_countries=("counterparty_country", "nunique"),
    )

    # Value-weighted tenor: the maturity profile a reader would want disclosed.
    weighted = df.assign(_w=df["tenor_days"] * df["value_zar"])
    base["tf_weighted_avg_tenor_days"] = (
        weighted.groupby(["entity_id", period])["_w"].sum()
        / weighted.groupby(["entity_id", period])["value_zar"].sum()
    )

    open_status = df[df["status"].isin(["active", "issued"])]
    base["tf_open_exposure_zar"] = (
        open_status.groupby(["entity_id", period])["value_zar"].sum()
    )
    base["tf_open_exposure_zar"] = base["tf_open_exposure_zar"].fillna(0.0)

    return base.join(instruments).join(directions)


def cross_border_summary(df: pd.DataFrame, period: str = "fiscal_year") -> pd.DataFrame:
    """FX flows by currency and corridor — the offshore-exposure note."""
    currencies = _pivot_sum(df, period, "currency_pair", "fx_")
    corridors = _pivot_sum(df, period, "corridor_type", "xb_corridor_")

    base = df.groupby(["entity_id", period]).agg(
        xb_txn_count=("value_zar", "size"),
        xb_total_value_zar=("value_zar", "sum"),
        xb_net_fx_zar=("signed_value_zar", "sum"),
        xb_countries=("counterparty_country", "nunique"),
    )

    flows = df.pivot_table(
        index=["entity_id", period],
        columns="direction",
        values="value_zar",
        aggfunc="sum",
        fill_value=0.0,
    ).rename(columns={"inbound": "xb_inflow_zar", "outbound": "xb_outflow_zar"})
    flows.columns.name = None

    return base.join(flows).join(currencies).join(corridors)


def build_entity_panel(
    frames: dict[str, pd.DataFrame], period: str = "fiscal_year"
) -> pd.DataFrame:
    """One row per entity per fiscal period, all three datasets side by side.

    This is the table to sit next to a published income statement / cash flow
    statement. Absolute rands will not match (the data is synthetic and
    bank-side only), so each total also carries:
      - ``*_index``: rebased to 100 in the entity's first period, for growth
      - ``*_share``: the entity's share of the all-entity total that period
    """
    panel = (
        banking_summary(frames["transactional_banking"], period)
        .join(trade_finance_summary(frames["trade_finance"], period), how="outer")
        .join(cross_border_summary(frames["cross_border_payments"], period), how="outer")
    )
    panel = panel.fillna(0.0)

    dim = build_entity_dimension(frames)
    panel = panel.join(dim, on="entity_id")

    panel["total_flow_zar"] = (
        panel["tb_total_value_zar"] + panel["tf_total_value_zar"] + panel["xb_total_value_zar"]
    )
    panel["fx_dependence"] = np.where(
        panel["total_flow_zar"] > 0, panel["xb_total_value_zar"] / panel["total_flow_zar"], np.nan
    )

    for col in ["tb_total_value_zar", "tf_total_value_zar", "xb_total_value_zar", "total_flow_zar"]:
        by_entity = panel.groupby(level="entity_id")[col]
        panel[f"{col}_index"] = 100 * panel[col] / by_entity.transform("first")
        panel[f"{col}_yoy"] = by_entity.pct_change()
        panel[f"{col}_share"] = panel[col] / panel.groupby(level=period)[col].transform("sum")

    front = ["entity_name", "sector"]
    return panel[front + [c for c in panel.columns if c not in front]].sort_index()


def split_by_entity(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    """Per-entity slice of every dataset: ``{entity_id: {dataset: DataFrame}}``."""
    return {
        entity_id: {name: df[df["entity_id"] == entity_id].copy() for name, df in frames.items()}
        for entity_id in sorted(frames["transactional_banking"]["entity_id"].unique())
    }


# --------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------

def data_quality_report(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Row counts, duplicate ids, missingness and coverage per dataset."""
    rows = []
    for name, df in frames.items():
        rows.append({
            "dataset": name,
            "rows": len(df),
            "entities": df["entity_id"].nunique(),
            "duplicate_ids": int(df["is_duplicate_id"].sum()),
            "date_from": df["date"].min().date(),
            "date_to": df["date"].max().date(),
            "total_value_zar": df["value_zar"].sum(),
            "memo_missing_pct": round(df["memo"].isna().mean() * 100, 2),
        })
    return pd.DataFrame(rows).set_index("dataset")


# --------------------------------------------------------------------------
# Optional EDA: profiling tables and charts
# --------------------------------------------------------------------------

def abt_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numeric and categorical analytical base tables for one DataFrame."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number, "datetime64[ns]"]).columns

    numeric_stats = pd.DataFrame({
        "variable": numeric_cols,
        "count": len(df),
        "cardinality": df[numeric_cols].nunique().values,
        "missingness": df[numeric_cols].isna().sum().values,
        "datatype": df[numeric_cols].dtypes.values,
        "mean": df[numeric_cols].mean().values,
        "median": df[numeric_cols].median().values,
        "1st_quartile": df[numeric_cols].quantile(0.25).values,
        "3rd_quartile": df[numeric_cols].quantile(0.75).values,
        "max": df[numeric_cols].max().values,
        "std_dev": df[numeric_cols].std().values,
    }).round(2)

    records = []
    for col in categorical_cols:
        counts = df[col].dropna().value_counts()
        total = counts.sum()
        top = counts.head(2)
        records.append({
            "variable": col,
            "count": len(df),
            "cardinality": df[col].nunique(),
            "missingness": df[col].isna().sum(),
            "datatype": df[col].dtype,
            "mode1": top.index[0] if len(top) > 0 else np.nan,
            "mode2": top.index[1] if len(top) > 1 else np.nan,
            "mode1_freq": top.iloc[0] if len(top) > 0 else np.nan,
            "mode2_freq": top.iloc[1] if len(top) > 1 else np.nan,
            "mode1_pct": top.iloc[0] / total if len(top) > 0 else np.nan,
            "mode2_pct": top.iloc[1] / total if len(top) > 1 else np.nan,
        })
    categorical_stats = pd.DataFrame(records).round(2)

    return numeric_stats, categorical_stats


def df_to_typst_table(df: pd.DataFrame, title: str = "Table", caption: str = "") -> str:
    """Render a DataFrame as a Typst ``titled-table`` figure."""
    headers = ", ".join(f"[*{col}*]" for col in df.columns)
    rows = ",\n    ".join(
        ", ".join(f"[{cell}]" for cell in row) for row in df.itertuples(index=False)
    )
    return (
        "#figure(\n"
        "  titled-table(\n"
        f'    title: "{title}",\n'
        f"    columns: {len(df.columns)},\n"
        f"    {headers},\n"
        f"    {rows}\n"
        "  ),\n"
        f"  caption: [{caption or title}]\n"
        ")"
    )


def run_eda(frames: dict[str, pd.DataFrame]) -> None:
    """Print ABT profiling tables as Typst figures."""
    for name, df in frames.items():
        label = name.replace("_", " ").title()
        numeric, categorical = abt_summary(df)
        print(df_to_typst_table(numeric, f"{label} - Numerical", f"{label} numerical ABT"))
        print(df_to_typst_table(categorical, f"{label} - Categorical", f"{label} categorical ABT"))


def _heatmap(ax, frame: pd.DataFrame, title: str, fmt: str | None = None) -> None:
    """Minimal matplotlib heatmap so the module needs no seaborn."""
    image = ax.imshow(frame.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(frame.columns)), frame.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(frame.index)), frame.index)
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax)
    if fmt:
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                ax.text(j, i, format(frame.values[i, j], fmt), ha="center", va="center",
                        fontsize=7, color="white")


def run_plots(frames: dict[str, pd.DataFrame], panel: pd.DataFrame) -> None:
    """Exploratory charts: entity totals, sector mix, volume trend, FX reliance."""
    import matplotlib.pyplot as plt

    period = panel.index.names[1]

    totals = panel.groupby(level="entity_id")["total_flow_zar"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(totals.index, totals.values / 1e9, color="#4c72b0")
    ax.set(title="Total flow by entity", xlabel="Entity ID", ylabel="Total value (ZAR bn)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    sector_mix = panel.reset_index().pivot_table(
        index="sector", columns="entity_id", values="total_flow_zar", aggfunc="sum", fill_value=0
    ) / 1e9
    fig, ax = plt.subplots(figsize=(14, 6))
    _heatmap(ax, sector_mix, "Total flow (ZAR bn) by sector and entity")
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(14, 6))
    for name, df in frames.items():
        monthly = df.groupby("calendar_month").size()
        ax.plot(monthly.index, monthly.values, label=name)
    ax.set(title="Monthly transaction volume by dataset", xlabel="Month", ylabel="Transactions")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()

    fx = panel.reset_index().pivot_table(
        index=period, columns="entity_id", values="fx_dependence"
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    _heatmap(ax, fx, f"Cross-border share of total flow, by entity and {period}", fmt=".2f")
    fig.tight_layout()

    plt.show()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="fiscal_year",
                        choices=["fiscal_year", "fiscal_half", "fiscal_quarter", "calendar_month"],
                        help="Aggregation period for the entity panel")
    parser.add_argument("--eda", action="store_true", help="Print ABT profiling tables")
    parser.add_argument("--plots", action="store_true", help="Show exploratory charts")
    parser.add_argument("--no-write", action="store_true", help="Skip writing outputs/")
    args = parser.parse_args()

    frames = load_all()
    entities = build_entity_dimension(frames)
    panel = build_entity_panel(frames, period=args.period)

    print("--- Entities ---")
    print(entities.to_string())
    print("\n--- Data quality ---")
    print(data_quality_report(frames).to_string())
    print(f"\n--- Entity panel ({args.period}) ---")
    print(f"shape: {panel.shape}")
    print(panel.head(10).to_string())

    if not args.no_write:
        OUTPUT_DIR.mkdir(exist_ok=True)
        entities.to_csv(OUTPUT_DIR / "entities.csv")
        panel.to_csv(OUTPUT_DIR / f"entity_panel_{args.period}.csv")
        data_quality_report(frames).to_csv(OUTPUT_DIR / "data_quality.csv")
        print(f"\nWrote outputs to {OUTPUT_DIR}")

    if args.eda:
        run_eda(frames)
    if args.plots:
        run_plots(frames, panel)


if __name__ == "__main__":
    main()
