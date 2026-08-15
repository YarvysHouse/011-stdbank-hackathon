""" The sizing maths, with no Streamlit and no globals.

`entity_app.py` renders these and `ai_analyst.py` calls them as tools, so the
dashboard and the chat panel cannot drift apart - a figure the analyst quotes is
the same figure the tab draws, out of the same function.

Pure functions over frames passed in. Nothing here reads a file except
`load_tables()`, which is the same artifacts-or-source fallback the app uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from build_artifacts import ARTIFACTS_, build

ID_COLS = ["entity_id", "entity_name", "sector"]
TABLES = ["summary", "reference", "reference_counts", "comparison", "geo", "projection"]


def load_tables(names: tuple[str, ...] = tuple(TABLES)) -> tuple[pd.DataFrame, ...]:
    """Precomputed artifacts where they exist, otherwise straight from the CSVs.

    Deployments ship only the 71 KB of parquet - the 409 MB of source data never
    leaves the machine that ran `build_artifacts.py`.
    """
    if all((ARTIFACTS_ / f"{name}.parquet").exists() for name in names):
        return tuple(pd.read_parquet(ARTIFACTS_ / f"{name}.parquet") for name in names)

    tables = build()
    return tuple(tables[name] for name in names)


def reliable_lines(comparison: pd.DataFrame) -> pd.DataFrame:
    """Comparison rows that reconcile.

    Scale errors from PDF extraction would swamp any sizing built on them, so they
    go on the same rule the comparison table paints amber.
    """
    return comparison[comparison["summation_value"].notna()
                      & (comparison["pct_of_reported"].abs() <= 50)].copy()


def missed_wallet(clean: pd.DataFrame, summary: pd.DataFrame,
                  fee_bps: float, fee_per_txn: float) -> pd.DataFrame:
    """Per-entity gap to reported financials, and the fee revenue closing it would carry."""
    clean = clean.copy()
    clean["missed"] = (clean["reported_value"] - clean["summation_value"]).clip(lower=0)

    missed = clean.groupby(ID_COLS, as_index=False).agg(
        missed_amount=("missed", "sum"), reported=("reported_value", "sum"),
        computed=("summation_value", "sum"))
    missed = missed.merge(summary[["entity_id", "incomes", "payments", "num_transactions"]],
                          on="entity_id", how="left")

    # what the bank already routes, per transaction, is the ticket it would carry
    missed["avg_ticket"] = ((missed["incomes"] + missed["payments"])
                            / missed["num_transactions"].replace(0, np.nan))
    missed["implied_txns"] = (missed["missed_amount"] / missed["avg_ticket"]).replace([np.inf, -np.inf], np.nan)
    missed["fee_revenue"] = (missed["missed_amount"] * fee_bps / 10_000
                             + missed["implied_txns"].fillna(0) * fee_per_txn)
    return missed


def project(projection: pd.DataFrame, horizon: int, bps: float) -> pd.DataFrame:
    """Published revenue compounded forward at each entity's CAGR, bank share held flat.

    Entities with no reported revenue line carry no base to compound and drop out.
    """
    p = projection.dropna(subset=["base_revenue", "cagr_pct"]).copy()

    p["projected_revenue"] = p["base_revenue"] * (1 + p["cagr_pct"] / 100) ** horizon
    p["revenue_growth"] = p["projected_revenue"] - p["base_revenue"]
    p["routed_now"] = p["base_revenue"] * p["wallet_share_pct"] / 100
    p["routed_future"] = p["projected_revenue"] * p["wallet_share_pct"] / 100
    p["bank_now"] = p["routed_now"] * bps / 10_000
    p["bank_future"] = p["routed_future"] * bps / 10_000
    p["bank_uplift"] = p["bank_future"] - p["bank_now"]
    return p
