# Transaction Banking — Entity Analytics & Statement Reconciliation

Reconstructs a **pseudo consolidated financial statement pack** for 20 listed entities from
three years of bank-side transaction flows, then reconciles it against real published
financials to surface discrepancies — presented through a Streamlit board dashboard.

Standard Bank hackathon, 2026.

## The pipeline

```
Data/*.csv  ->  augstdhackathon.py     clean, sort by entity, fiscal periods, entity panel
            ->  financial_statements.py  income statement / cash flow / contingent note
            ->  dashboard_data.py       pre-aggregates for the UI
            ->  app.py                  Streamlit dashboard
```

| Module | Role |
| --- | --- |
| `augstdhackathon.py` | Loads and cleans the three sources, attaches June-year-end fiscal periods, builds the entity × period panel |
| `financial_statements.py` | Builds the three-year statement pack per entity; compares to reported financials and flags material discrepancies |
| `dashboard_data.py` | Rolls 2.8m transactions down to small artifacts in `outputs/dashboard/` |
| `app.py` | Five-tab dashboard, broad to granular |

## Quick start

```bash
uv sync
uv run python augstdhackathon.py        # clean + entity panel
uv run python financial_statements.py   # statement pack per entity
uv run python dashboard_data.py         # dashboard artifacts
uv run streamlit run app.py             # dashboard
```

`Data/` is not in the repo (see below) — drop the three source CSVs there first.

## Method

**Fiscal periods.** The data spans 2023-07-01 to 2026-06-30, exactly three July–June years,
so everything is labelled on a June year-end: `FY2024` = Jul 2023 – Jun 2024.

**Three classification rules drive the statements:**

1. **Intercompany is never revenue.** `intercompany_sweeps` and `corridor_type == intercompany`
   go to financing, not the income statement — including them would roughly double revenue.
2. **Trade finance is notional, never cash.** Letters of credit and guarantees are contingent
   exposures on trade already flowing through cross-border payments. They sit in a separate
   note to avoid double counting.
3. **Cash basis only.** No accruals, depreciation or working-capital movements, so "EBITDA
   proxy" is operating cash margin, not reported EBITDA.

**Reconciliation.** Absolute rands cannot match reported figures — these are bank-side flows
only. So `compare_to_reported()` derives a per-entity-year **coverage factor** from revenue
(`reported ÷ proxy`), scales every other line by it, and flags a line only when its own
implied coverage deviates materially (default 10%) from the revenue coverage. That deviation,
not the rand gap, is the discrepancy signal.

Fill `benchmarks/reported_financials_template.csv` (20 entities × 3 years × 10 metrics) with
real annual-report figures and upload it in the dashboard sidebar, or pass it to
`financial_statements.py --compare`.

## Known discrepancies

`plausibility_report()` runs without any reported data and flags 153 of 240 ratio observations
as outside plausible ranges for large listed corporates:

- **Employee costs are ~0.1% of revenue** across all 60 entity-years (real: 3–35%). Only
  16,524 payroll transactions in 2.8m rows — salaries are almost certainly paid via a bulk
  payroll file this data does not capture.
- **Effective cash tax rate median 1.6%** against a 27% SA corporate rate; 5,396 tax
  transactions in total.
- Revenue, cost of sales and export share should reconcile in shape; employee costs and tax
  will not. That is a data-coverage gap, not a modelling error.

## Data

The source CSVs are **not committed** — 409 MB total, and `transactional_banking.csv` alone is
375 MB, over GitHub's 100 MB per-file limit. The hackathon briefs and T&Cs are also excluded.

The 20 entities the transactions cover are listed in `entities.txt`.
