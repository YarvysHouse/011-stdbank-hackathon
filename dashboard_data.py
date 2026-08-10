# -*- coding: utf-8 -*-
"""Pre-aggregates for the Streamlit dashboard.

The raw transactional file is 2.8m rows, so the app never touches it directly.
This module rolls it down to a handful of small frames written to
``outputs/dashboard/`` and reads those on every run. Rebuild with::

    uv run python dashboard_data.py --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from augstdhackathon import (
    BASE_DIR,
    build_entity_dimension,
    build_entity_panel,
    data_quality_report,
    load_all,
)
from financial_statements import build_statement_pack, plausibility_report

DASH_DIR = BASE_DIR / "outputs" / "dashboard"

ARTIFACTS = [
    "entities",
    "entity_panel",
    "entity_panel_quarterly",
    "statement_pack",
    "plausibility",
    "data_quality",
    "monthly_volume",
    "channel_mix",
    "corridor_mix",
    "country_flows",
    "trade_profile",
    "value_distribution",
]


def _monthly_volume(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name, df in frames.items():
        agg = (
            df.groupby(["entity_id", "calendar_month", "fiscal_year"])
            .agg(txn_count=("value_zar", "size"), value_zar=("value_zar", "sum"))
            .reset_index()
        )
        agg["dataset"] = name
        parts.append(agg)
    return pd.concat(parts, ignore_index=True)


def _channel_mix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tb = frames["transactional_banking"]
    return (
        tb.groupby(["entity_id", "fiscal_year", "leg_type", "channel", "direction"])
        .agg(txn_count=("value_zar", "size"), value_zar=("value_zar", "sum"))
        .reset_index()
    )


def _corridor_mix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    xb = frames["cross_border_payments"]
    return (
        xb.groupby(["entity_id", "fiscal_year", "corridor_type", "currency_pair", "direction"])
        .agg(txn_count=("value_zar", "size"), value_zar=("value_zar", "sum"))
        .reset_index()
    )


def _country_flows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name in ["cross_border_payments", "trade_finance"]:
        df = frames[name]
        agg = (
            df.groupby(["entity_id", "fiscal_year", "counterparty_country", "direction"])
            .agg(txn_count=("value_zar", "size"), value_zar=("value_zar", "sum"))
            .reset_index()
        )
        agg["dataset"] = name
        parts.append(agg)
    return pd.concat(parts, ignore_index=True)


def _trade_profile(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tf = frames["trade_finance"]
    return (
        tf.groupby([
            "entity_id", "fiscal_year", "instrument_type", "status", "direction",
            "tenor_days", "commodity_or_contract_type",
        ])
        .agg(instrument_count=("value_zar", "size"), value_zar=("value_zar", "sum"))
        .reset_index()
    )


def _value_distribution(frames: dict[str, pd.DataFrame], bins: int = 40) -> pd.DataFrame:
    """Log10 histogram of transaction size, per dataset and per entity."""
    edges = np.linspace(2, 9, bins + 1)  # R100 to R1bn
    rows = []
    for name, df in frames.items():
        for entity_id, values in df.groupby("entity_id")["value_zar"]:
            counts, _ = np.histogram(np.log10(values.clip(lower=100)), bins=edges)
            rows.append(pd.DataFrame({
                "dataset": name,
                "entity_id": entity_id,
                "log10_value": (edges[:-1] + edges[1:]) / 2,
                "txn_count": counts,
            }))
    return pd.concat(rows, ignore_index=True)


def build_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """Build every dashboard artifact, reusing what is already on disk."""
    DASH_DIR.mkdir(parents=True, exist_ok=True)
    paths = {name: DASH_DIR / f"{name}.csv" for name in ARTIFACTS}

    if not force and all(p.exists() for p in paths.values()):
        return {name: pd.read_csv(p) for name, p in paths.items()}

    frames = load_all()
    built = {
        "entities": build_entity_dimension(frames).reset_index(),
        "entity_panel": build_entity_panel(frames, "fiscal_year").reset_index(),
        "entity_panel_quarterly": build_entity_panel(frames, "fiscal_quarter").reset_index(),
        "statement_pack": build_statement_pack(frames),
        "data_quality": data_quality_report(frames).reset_index(),
        "monthly_volume": _monthly_volume(frames),
        "channel_mix": _channel_mix(frames),
        "corridor_mix": _corridor_mix(frames),
        "country_flows": _country_flows(frames),
        "trade_profile": _trade_profile(frames),
        "value_distribution": _value_distribution(frames),
    }
    built["plausibility"] = plausibility_report(built["statement_pack"])

    for name, df in built.items():
        df.to_csv(paths[name], index=False)
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild even if artifacts exist")
    args = parser.parse_args()

    built = build_all(force=args.force)
    for name, df in built.items():
        print(f"{name:24s} {str(df.shape):>14s}  -> {DASH_DIR / (name + '.csv')}")


if __name__ == "__main__":
    main()
