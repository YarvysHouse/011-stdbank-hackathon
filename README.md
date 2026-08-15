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
- **176 of 176 filled rows match.**

## 4. Output metrics

Median wallet share, by line — *dashboard page 1 · Portfolio Summary*:

| Line item | Rows | Median share |
| --- | --- | --- |
| Net cash from operating activities | 43 | 1.100% |
| Total revenue | 47 | 0.657% |
| Cost of sales and supplier payments | 44 | 0.098% |
| Taxation paid | 42 | 0.030% |

- Overall median: **0.174%**.
- Book flow: R318.9bn in, R256.5bn out across 3.04m transactions — *page 3 · Entity Analysis*.
- Sector split and reported-vs-calculated per sector — *page 2 · Sector*.
- Cross-border: 34 counterparty countries, R169.0bn over 256,443 transactions, net importer — *page 4 · Geography*.
- Product concentration: top 3 references carry 90.7% of all transactions — *page 3 · Entity Analysis*.

**Portfolio summary and opportunity heatmap** — *page 1*:

- Page 1 opens book-wide before any filter: entities, transactions, book flow, cross-border, reported financials, share carried, addressable gap, and the 5-year growth uplift. Quoted at a fixed 15 bps + R5/txn over 5 years, so the headline never moves under someone else's slider; page 5 makes those adjustable.
- **Opportunity heatmap** — 19 clients × 5 measures: wallet gap, volume potential, share headroom, client growth, projected uplift.
- Cells are **percentile rank within the book**, not absolute value — the five measures are in different units (ZAR, transactions, %), so ranking is what makes them comparable across a row. Hover gives the underlying figure.
- Every column is oriented so brighter = more opportunity; share headroom inverts wallet share so thinly banked clients read hot.
- The ramp runs dark→light because the configured theme is dark: the low end sits near the surface and the high end lifts off it, which is the reverse of a light build.
- Rows sorted by mean percentile. Bright across all five is a coverage failure; bright in one column is a product sale.
- NEPI Rockcastle leads at 75.8, Glencore and Clicks Group both 74.7; Aspen Pharmacare last at 22.1.
- Sizing maths is shared with page 5 and the AI tools (`sizing.py`) so the summary cannot drift from what it summarises.

## 5. Analysis and recommendation outputs

**Growth capacity** — *page 5, top section*:

- Current and future opportunity sit on one page: the same client is a target for one and a defence for the other, and splitting them across tabs made that easy to miss.
- The top section ranks clients by what they could be worth in fee income at a chosen year, with a **year slider (0–5)** that reorders the board.
- Two components on different clocks — `organic = bank_now × (1 + cagr)^year`, compounding at the client's own growth rate, plus `captured = fee_on_gap × capture% × year/5`, the wallet gap phased in linearly.
- Stacked bars keep the split visible, so a client leading on captured gap never reads as one leading on growth.
- **Capture rate is a slider (default 1%)** because the two components are orders of magnitude apart: the gap is ~R58tn against R7tn of client revenue, so an unweighted sum would drown the growth signal entirely. At 0% the ranking is pure organic growth.
- The reordering is real: at 0% capture six clients change rank over the horizon; at 1% NEPI Rockcastle climbs 17 places to lead.

- Every dashboard page opens with a generated read of its own figures — key insight plus one suggested move.
- Computed live from whatever that page's slicers currently resolve to, not a fixed snapshot.
- These are **rule-based, not a model call** — branching logic over real statistics, so figures are always correct and they run offline. The generative layer is separate, on page 6 (section 7 below).
- Sizing on *page 5 · Opportunity → Current opportunity*:
  - `missed = max(reported − computed, 0)` per entity-year.
  - `avg_ticket = (incomes + payments) ÷ transactions`.
  - `implied_txns = missed ÷ avg_ticket`.
  - `fee_revenue = missed × bps/10,000 + implied_txns × fee_per_txn` (both sliders).
- Headline: R57.99tn reported, **0.235% carried**, R57.85tn addressable, ~R88bn modelled fee revenue at 15 bps + R5/txn.
- Rows where computed exceeds 50% of reported are flagged amber and excluded from sizing — extraction scale errors, not real matches.

**Product bundling** — *page 3 · Entity Analysis*:

- Top 3 reference types per entity, ranked by **transaction count, not value** — a bundle is billed per instruction, so volume concentration determines whether a discount recovers its margin.
- Table gives each reference, its share of the entity's transactions, and the combined total.
- Group package recommended when the three clear **60%** of transactions.
- Every entity clears it (AngloGold 80.1%, Bidvest 88.7%, book-wide 90.7%) — INV, SWEEP and PO dominate throughout.

## 6. Future projection — *page 5 · Future opportunity*

- Growth rates hand-entered per entity in `benchmarks/entity_cagr.csv`.
- Base is the **latest reported revenue**, not bank flow: CAGR is published on the company top line, so compounding bank flow by it would assume share grows with the client — the thing under test.
- `projected_revenue = base_revenue × (1 + cagr)^years`.
- `routed = projected_revenue × current_wallet_share`.
- `bank_revenue = routed × bps/10,000`. Horizon and bps are sliders.
- **Current share held flat** — this section sizes growth already committed to; winning share is the Current opportunity section's job.
- At 5 years / 15 bps: client revenue R7.20tn → **R9.33tn**, bank fee income R57.3m → **R77.7m**, uplift **R20.4m** with no new mandates.
- Fastest growers: Gold Fields 13.1%, Pepkor 12.3%, OUTsurance 12.2%. Two entities contracting (Anglo American −0.98%, Bidvest −4.70%).
- Sanlam is excluded — no reported revenue line, so no base to compound. 19 of 20 project.

## 7. Generative AI — the analyst on page 6

A chat panel over the book. Gemini answers by **calling the same aggregations the tabs render**, so every figure it quotes is one the dashboard already computes and the analysis was validated against.

### Why tool-calling and not a pandas agent

- A pandas agent executes model-written code against the frame — `df.query(model_string)` reaches any column, any row, and whatever `eval` exposes.
- Here the model gets **six named tools with typed arguments** and nothing else. It never sees the rows and cannot run code.
- Consequence: a bad answer is a wrong *sentence* about right figures, never an invented figure. The failure mode is bounded.
- Both the dashboard and the tools import `sizing.py` — one implementation of the maths, so the chat and the tabs cannot disagree.

### The tools

| Tool | Returns |
| --- | --- |
| `portfolio_overview` | Book-wide: entities, transactions, flow, reported, carried, gap, fee revenue, 5y uplift |
| `entity_detail` | One client: flow, share, missed wallet, top references, CAGR, projected uplift |
| `sector_rollup` | Every sector with entity count, flow, share and gap |
| `opportunity_ranking` | Clients ranked by missed wallet / volume / fee revenue, bps configurable |
| `projection` | Revenue compounded at CAGR, share flat, horizon and bps configurable |
| `cross_border` | Counterparty countries, income against payment, optionally per client |

- `MAX_TOOL_TURNS = 8`; the loop stops rather than spiralling.
- Every answer ships an expander listing the calls made and their raw JSON — the audit trail is the point.
- Six preset prompts, three of which are **client briefing notes** (Glencore, Shoprite, NEPI Rockcastle).
- No key configured → the tab explains the wiring and the other six tabs are unaffected.
- Temperature 0.2. The system prompt is `SYSTEM` in `ai_analyst.py` — committed, so the prompt is reviewable evidence rather than a screenshot.
- **The model is not pinned.** A hardcoded `gemini-2.5-flash` began returning `404 NOT_FOUND — no longer available to new users` in production with nothing in this repo having changed. `resolve_model()` now lists what the key can actually call and picks the best of them: newest generation, then flash over pro (this is a high-frequency lookup panel, so latency and cost beat depth), then a bare alias over a dated snapshot, since aliases outlive pins. A 404 mid-session clears the cache and re-discovers rather than breaking the tab.
- `GEMINI_MODEL` in secrets or the environment overrides the choice. `--models` prints everything the key sees, so a future 404 is diagnosable rather than guesswork.

### Wiring the Gemini key

**1 — Generate.** Sign in at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → **Create API key** → pick or create a Google Cloud project → copy it. Starts `AIza`, about 39 characters. Free tier is enough for demo traffic. This is a Google AI Studio key, *not* a Vertex AI service account.

**2 — Local.** Create `.streamlit/secrets.toml` (already gitignored — check `git status` shows nothing before committing):

```toml
GEMINI_API_KEY = "AIza..."
```

Or `export GEMINI_API_KEY=AIza...`. The app reads `st.secrets` first, then the environment.

**3 — Deployed.** Streamlit Cloud → the app → **⋮ → Settings → Secrets** → paste the same line → **Save**. The app reboots itself; no redeploy needed. Cloud secrets are per-app and are not readable from the repo.

**4 — Validate.**

```bash
uv run python ai_analyst.py --check
uv run python ai_analyst.py --models                              # what the key can call
uv run python ai_analyst.py --ask "Where is our largest wallet gap?"
```

`--check` proves four things in order: the key resolves and from which source, it authenticates against Google, the parquet artifacts load, and a real question completes a full tool round-trip. It prints the tool names called — if that line is empty the model answered without touching the data, which is a failure even when the prose looks right.

Failure modes it names: `API_KEY_INVALID` (wrong value, or an IP/referrer restriction on the key), `403` (Generative Language API not enabled on the key's project), missing artifacts (run `build_artifacts.py`).

**Key hygiene:** rotate in AI Studio if it ever lands in a commit, a screenshot, or a shared terminal — Cloud picks up the new value from the Secrets box with no code change.

## Run it

```bash
uv sync
uv run python build_artifacts.py        # needs Data/ — writes outputs/artifacts/*.parquet
uv run streamlit run entity_app.py
```

- `build_artifacts.py` precomputes the six dashboard tables to **71 KB of parquet**.
- The app reads those when present, falls back to the raw CSVs when absent.
- Deployment ships only the parquet — load time under a second instead of ~40s.
- Re-run `build_artifacts.py` and push after changing the worklist or source data.

## Layout

| File | Role |
| --- | --- |
| `analysis_script.py` | Load, sign, fiscal-year, aggregate, compare |
| `build_artifacts.py` | Precompute the six tables to parquet |
| `sizing.py` | The sizing maths, pure — shared by the dashboard and the analyst |
| `ai_analyst.py` | Gemini tools, the tool loop, and the `--check` wiring test |
| `entity_app.py` | Six-tab Streamlit dashboard |
| `benchmarks/` | Extraction worklist (reported-side denominator) and `entity_cagr.csv` |
| `public_financial_statements/` | Source PDFs behind every reported figure |
| `outputs/artifacts/` | What the deployed app reads |

## Known limits

- **Employee costs has no reported rows** — computed from `PAYROLL + PAYE` but nothing to compare against, so it does not appear in the metrics.
- Payroll is likely paid via bulk files this data does not capture — a coverage gap, not a modelling error.
- Projections assume wallet share holds flat and CAGR is uniform across the horizon; neither is guaranteed.
- Some reported figures carry scale errors from PDF extraction; a plausibility ceiling on `reported_value` is not yet implemented.
- Cash basis only — no accruals, so no true EBITDA.
- Trade finance counterparties are all external; no entity finances another.
