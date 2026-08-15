""" Precompute the dashboard tables so the deployed app carries no source data.

The five tables `entity_app.py` renders total about 67 KB as parquet, against
409 MB of source CSV - one of which exceeds GitHub's 100 MB file limit and can
never be committed. Run this locally whenever the source data or the benchmark
worklist changes, then commit `outputs/artifacts/`::

    uv run python build_artifacts.py

The app loads these when present and falls back to computing from `Data/` when
they are not, so local development is unaffected.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis_script import (load_data, reference_type_check, combined_df, entity_report,
                             compare_to_results, TRANSACTIONS_, CROSS_BORDER_, TRADE_FINANCE_)

ARTIFACTS_ = Path(__file__).resolve().parent / "outputs" / "artifacts"

GEO_COLS = ["entity_id", "entity_name", "sector", "counterparty_country", "signed_amount", "value_zar"]


def country_flows(cross_border, trade_finance):
    """Counterparty geography. Only these two sources carry a country column."""
    d = pd.concat([cross_border[GEO_COLS], trade_finance[GEO_COLS]], ignore_index=True)
    d["flow"] = np.where(d["signed_amount"] > 0, "Income", "Payment")
    return d.groupby(["entity_id", "entity_name", "sector", "counterparty_country", "flow"],
                     as_index=False).agg(value_zar=("value_zar", "sum"), txn_count=("value_zar", "size"))


def build() -> dict[str, pd.DataFrame]:
    """Every table the dashboard needs, computed from the source CSVs."""
    frames = [reference_type_check(load_data(p))
              for p in (TRANSACTIONS_, CROSS_BORDER_, TRADE_FINANCE_)]
    consolidated = combined_df(*frames)

    summary, reference, reference_counts = entity_report(consolidated)

    # the comparison carries no sector, so bring it across for the tab slicers
    sectors = consolidated[["entity_id", "sector"]].drop_duplicates()
    comparison = compare_to_results(consolidated).merge(sectors, on="entity_id", how="left")

    return {
        "summary": summary,
        "reference": reference,
        "reference_counts": reference_counts,
        "comparison": comparison,
        "geo": country_flows(frames[1], frames[2]),
    }


def main() -> None:
    ARTIFACTS_.mkdir(parents=True, exist_ok=True)

    total = 0
    for name, frame in build().items():
        path = ARTIFACTS_ / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        size = path.stat().st_size
        total += size
        print(f"  {name:<18} {len(frame):>5} rows x {frame.shape[1]:>2} cols -> {size / 1024:>6.1f} KB")

    print(f"\nWrote {total / 1024:.1f} KB to {ARTIFACTS_}")


if __name__ == "__main__":
    main()
