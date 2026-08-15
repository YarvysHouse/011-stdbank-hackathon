# Syn Bank — Wallet Share Analytics

Measures what share of 20 listed entities' published financials Syn Bank actually handles.

**Dashboard: https://yarvyshouse-011-stdbank-hackathon-entity-app-gzkftq.streamlit.app/**

Standard Bank hackathon, 2026.

---

## 1. Data extraction

- Three source CSVs in `Data/` — 409 MB, **not committed** (`transactional_banking.csv` alone is 392 MB, over GitHub's 100 MB limit).
- `transactional_banking.csv` — domestic payments, 20 reference types, channel and leg type.
- `cross_border_payments.csv` — FX payments, currency pair, counterparty country, corridor type.
- `trade_finance.csv` — LCs, guarantees, collections; tenor, commodity, counterparty country.
- Reported financials hand-keyed from 39 annual-report PDFs in `public_financial_statements/` into `benchmarks/extraction_worklist_15_aug.csv` — 247 rows, each carrying `source_pdf` and `where_to_look` for audit.
- Scope: 20 entities, 7 sectors, 3,044,428 transactions, FY2024–FY2026.

## 2. Processing — `analysis_script.py`

- `load_data()` — dedupe, parse dates, unify `amount_zar`/`value_zar` and `instrument_id`/`transaction_id`.
- **Direction sign:** `signed_amount = -value_zar` where direction is `outbound` or `import`, else `+value_zar`.
- **Fiscal year:** July–June. `FY = year + 1 if month >= 7`. Data spans 2023-07-01 to 2026-06-30 — exactly three years.
- **Reference type:** strip the trailing serial off `reference` (`INV-662227` → `INV`). 20 types.
- `combined_df()` — concatenates all three sources into one 3.05m-row frame, tagged by `source`.
- `entity_report()` — three tables: per-entity summary, reference-type amounts, reference-type counts.

## 3. Comparison — `line_comparison()` + `compare_to_results()`

Five reported line items mapped to reference types:

| Line item | Calculation |
| --- | --- |
| Total revenue | `sum(INV where signed_amount > 0)` |
| Taxation paid | `abs(sum(CIT + VAT201 + PROV-TAX))` |
| Cost of sales and supplier payments | `abs(sum(PO))` |
| Employee costs | `abs(sum(PAYROLL + PAYE))` |
| Net cash from operating activities | `sum(all refs except SWEEP, TERM-DEPOSIT, MM-PLACEMENT, CALL-ACCT, LOAN, LOAN-REPAY, FACILITY)` |

- Joined to the worklist on `(entity_id, fiscal_year, line_item)`.
- `difference = computed − reported`.
- **`pct_of_reported = computed ÷ reported × 100`** — this is the wallet share figure.
- Outflow lines compared on magnitude; the worklist is inconsistent on sign (Taxation paid is negative in 37 of 42 rows).
- **181 of 181 filled rows match.**

## 4. Output metrics

Median wallet share, by line — *dashboard page 1 · Reported vs Computed*:

| Line item | Rows | Median share |
| --- | --- | --- |
| Net cash from operating activities | 43 | 1.100% |
| Total revenue | 47 | 0.657% |
| Cost of sales and supplier payments | 44 | 0.098% |
| Taxation paid | 42 | 0.030% |
| Employee costs | 5 | 0.008% |

- Overall median: **0.148%**.
- Book flow: R318.9bn in, R256.5bn out across 3.04m transactions — *page 3 · Entity Analysis*.
- Sector split and reported-vs-calculated per sector — *page 2 · Sector*.
- Cross-border: 34 counterparty countries, R169.0bn over 256,443 transactions, net importer — *page 4 · Geography*.

## 5. Analysis and recommendation outputs

- Every dashboard page opens with a generated read of its own figures — key insight plus one suggested move.
- Computed live from whatever that page's slicers currently resolve to, not a fixed snapshot.
- **Rule-based, not a model call** — branching logic over real statistics, so figures are always correct and it runs offline.
- Sizing on *page 5 · Opportunity*:
  - `missed = max(reported − computed, 0)` per entity-year.
  - `avg_ticket = (incomes + payments) ÷ transactions`.
  - `implied_txns = missed ÷ avg_ticket`.
  - `fee_revenue = missed × bps/10,000 + implied_txns × fee_per_txn` (both sliders).
- Headline: R58.05tn reported, **0.235% carried**, R57.92tn addressable, ~R88bn modelled fee revenue at 15 bps + R5/txn.
- Rows where computed exceeds 50% of reported are flagged amber and excluded from sizing — extraction scale errors, not real matches.

## Run it

```bash
uv sync
uv run python build_artifacts.py        # needs Data/ — writes outputs/artifacts/*.parquet
uv run streamlit run entity_app.py
```

- `build_artifacts.py` precomputes the five dashboard tables to **65 KB of parquet**.
- The app reads those when present, falls back to the raw CSVs when absent.
- Deployment ships only the parquet — load time under a second instead of ~40s.
- Re-run `build_artifacts.py` and push after changing the worklist or source data.

## Layout

| File | Role |
| --- | --- |
| `analysis_script.py` | Load, sign, fiscal-year, aggregate, compare |
| `build_artifacts.py` | Precompute the five tables to parquet |
| `entity_app.py` | Five-tab Streamlit dashboard |
| `benchmarks/` | Extraction worklist — the reported-side denominator |
| `public_financial_statements/` | Source PDFs behind every reported figure |
| `outputs/artifacts/` | What the deployed app reads |

## Known limits

- **Employee costs: 5 rows only** (Vodacom, MTN). Treat 0.008% as indicative.
- Payroll is likely paid via bulk files this data does not capture — a coverage gap, not a modelling error.
- Some reported figures carry scale errors from PDF extraction; a plausibility ceiling on `reported_value` is not yet implemented.
- Cash basis only — no accruals, so no true EBITDA.
- Trade finance counterparties are all external; no entity finances another.
