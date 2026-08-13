# -*- coding: utf-8 -*-
"""Syn Bank wallet-share dashboard.

Four sections, navigated from the left rail:

    0  Total Summary       Syn Bank against market financials, and the gap
    1  Sector Analysis     where the book sits and where the gap sits
    2  Per Entity          one entity's financials grouping, line by line
    3  Opportunity         what the gap is worth to the bank
    4  Reported Recon      the extraction worklist against the computed pack
    5  Trade              net import position, GDP impact and the trade channels

Run with::

    uv run streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_data import build_all
from market_financials import (
    bank_view, load_worklist, market_view, reconcile, worklist_comparison,
)

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------

BG = "#0b0b0f"
SURFACE = "#16161c"
BORDER = "#24242e"
ACCENT = "#7c5cff"
ACCENT_SOFT = "#a78bfa"
TEXT = "#ececf1"
MUTED = "#8b8b99"
POS = "#34d399"
NEG = "#f87171"
WARN = "#fbbf24"
BANK = "#7c5cff"
MARKET = "#22d3ee"

SEQUENCE = [ACCENT, "#22d3ee", "#f472b6", "#facc15", "#34d399", "#fb923c", "#818cf8", "#e879f9"]

SECTIONS = [
    "0 · Total Summary",
    "1 · Sector Analysis",
    "2 · Per Entity Analysis",
    "3 · Opportunity for Growth",
    "4 · Reported vs Computed",
    "5 · Trade & Cross-Border",
]

st.set_page_config(
    page_title="Syn Bank - Wallet Share Analytics",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .stApp {{ background: {BG}; }}
      section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
      div.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1600px; }}

      h1, h2, h3, h4 {{ color: {TEXT}; font-weight: 600; letter-spacing: -0.01em; }}
      .muted {{ color: {MUTED}; font-size: 0.86rem; line-height: 1.5; }}

      .card {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 1.1rem 1.25rem; height: 100%;
      }}
      .card-label {{ color: {MUTED}; font-size: 0.78rem; letter-spacing: 0.02em; margin-bottom: 0.35rem; }}
      .card-value {{ color: {TEXT}; font-size: 1.75rem; font-weight: 650; line-height: 1.15; }}
      .card-delta {{ font-size: 0.8rem; margin-top: 0.4rem; }}
      .up {{ color: {POS}; }} .down {{ color: {NEG}; }} .flat {{ color: {MUTED}; }}

      .brand {{ display: flex; align-items: center; gap: 0.7rem; padding: 0.3rem 0 1.1rem 0; }}
      .brand-mark {{
        width: 42px; height: 42px; border-radius: 50%; background: {ACCENT};
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 700; font-size: 1rem;
      }}
      .brand-title {{ color: {TEXT}; font-weight: 600; font-size: 0.95rem; }}
      .brand-sub {{ color: {MUTED}; font-size: 0.76rem; }}

      .pill {{
        display: inline-block; padding: 0.18rem 0.6rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600;
      }}
      .pill-bad {{ background: rgba(248,113,113,0.15); color: {NEG}; }}
      .pill-ok {{ background: rgba(52,211,153,0.15); color: {POS}; }}
      .pill-warn {{ background: rgba(251,191,36,0.15); color: {WARN}; }}

      /* Left rail navigation: the radio group rendered as stacked tabs. */
      div[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 0.25rem; }}
      div[data-testid="stSidebar"] div[role="radiogroup"] > label {{
        width: 100%; padding: 0.6rem 0.85rem; border-radius: 10px;
        border: 1px solid transparent; cursor: pointer;
        transition: background 120ms ease, border-color 120ms ease;
      }}
      div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
        background: rgba(124,92,255,0.10);
      }}
      div[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
        background: rgba(124,92,255,0.18); border-color: {ACCENT};
      }}
      div[data-testid="stSidebar"] div[role="radiogroup"] input {{ display: none; }}
      div[data-testid="stSidebar"] div[role="radiogroup"] label p {{
        font-size: 0.9rem; font-weight: 550; color: {TEXT};
      }}

      .stTabs [data-baseweb="tab-list"] {{ gap: 0.35rem; border-bottom: 1px solid {BORDER}; }}
      .stTabs [data-baseweb="tab"] {{
        background: transparent; border-radius: 10px 10px 0 0; padding: 0.55rem 1rem;
        color: {MUTED}; font-size: 0.86rem;
      }}
      .stTabs [aria-selected="true"] {{ background: {SURFACE}; color: {TEXT}; }}
      .stTabs [data-baseweb="tab-highlight"] {{ background: {ACCENT}; }}

      div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; }}
      hr {{ border-color: {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------

def zar(value: float, decimals: int = 1) -> str:
    """Compact rand formatting: R 12.3bn, R 456.0m, R 12.3k."""
    if value is None or pd.isna(value):
        return "-"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    for cutoff, suffix in ((1e12, "tn"), (1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if magnitude >= cutoff:
            return f"{sign}R {magnitude / cutoff:,.{decimals}f}{suffix}"
    return f"{sign}R {magnitude:,.0f}"


def pct(value: float, decimals: int = 1) -> str:
    return "-" if value is None or pd.isna(value) else f"{value:.{decimals}%}"


def card(label: str, value: str, delta: str | None = None, tone: str = "flat") -> str:
    arrow = {"up": "↗", "down": "↘", "flat": "•"}[tone]
    delta_html = f'<div class="card-delta {tone}">{arrow} {delta}</div>' if delta else ""
    return (
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>{delta_html}</div>'
    )


def card_row(cards: list[str]) -> None:
    for column, html in zip(st.columns(len(cards)), cards):
        column.markdown(html, unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        colorway=SEQUENCE,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font_size=12),
        title=dict(font=dict(size=14, color=TEXT), x=0, xanchor="left"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


def section(title: str, caption: str = "") -> None:
    st.markdown(f"### {title}")
    if caption:
        st.markdown(f'<div class="muted">{caption}</div>', unsafe_allow_html=True)
    st.write("")


def table(df: pd.DataFrame, height: int | str = "content") -> None:
    st.dataframe(df, width="stretch", height=height, hide_index=True)


def donut(labels, values, centre_label: str, colors=None, height: int = 340) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=list(labels), values=list(values), hole=0.66,
        marker=dict(colors=colors or SEQUENCE, line=dict(color=BG, width=2)),
        textinfo="percent", textfont=dict(size=11),
        hovertemplate="%{label}<br>%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(annotations=[dict(
        text=f"<span style='font-size:12px;color:{MUTED}'>{centre_label}</span>",
        showarrow=False,
    )])
    return style(fig, height)


# Rough centroids, enough to place a bubble on a world map. Only the countries
# that appear in the cross-border and trade finance files are listed.
COUNTRY_COORDS = {
    "Angola": (-11.2, 17.9), "Botswana": (-22.3, 24.7), "Brazil": (-14.2, -51.9),
    "Cameroon": (7.4, 12.4), "China": (35.9, 104.2), "DRC": (-4.0, 21.8),
    "Egypt": (26.8, 30.8), "Eswatini": (-26.5, 31.5), "Ethiopia": (9.1, 40.5),
    "Germany": (51.2, 10.5), "Ghana": (7.9, -1.0), "Guinea": (9.9, -9.7),
    "India": (20.6, 79.0), "Ivory Coast": (7.5, -5.5), "Japan": (36.2, 138.3),
    "Kenya": (-0.02, 37.9), "Lesotho": (-29.6, 28.2), "Malawi": (-13.3, 34.3),
    "Mauritius": (-20.3, 57.6), "Morocco": (31.8, -7.1), "Mozambique": (-18.7, 35.5),
    "Namibia": (-22.96, 18.5), "Netherlands": (52.1, 5.3), "Nigeria": (9.1, 8.7),
    "South Sudan": (6.9, 31.3), "Sudan": (12.9, 30.2), "Switzerland": (46.8, 8.2),
    "Tanzania": (-6.4, 34.9), "Uganda": (1.4, 32.3), "United Arab Emirates": (23.4, 53.8),
    "United Kingdom": (55.4, -3.4), "United States": (37.1, -95.7), "Zambia": (-13.1, 27.8),
    "Zimbabwe": (-19.0, 29.2),
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Building analytics from source transactions...")
def load_artifacts() -> dict[str, pd.DataFrame]:
    return build_all()


@st.cache_data(show_spinner=False)
def load_market(upload: bytes | None) -> pd.DataFrame:
    return market_view(upload)


@st.cache_data(show_spinner=False)
def load_reported(upload: bytes | None) -> pd.DataFrame:
    """The extraction worklist - every reported line, filled or outstanding.

    An uploaded file overlays it line by line, so a keyed file covering three
    entities does not discard the worklist's reading of the other seventeen.
    """
    base = load_worklist()
    if upload is None:
        return base
    keyed = load_worklist(upload=upload)
    if keyed.empty:
        return base
    keys = ["entity_id", "fiscal_year", "line_item"]
    kept = base.merge(keyed[keys], on=keys, how="left", indicator=True)
    kept = kept[kept["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([kept, keyed], ignore_index=True)


data = load_artifacts()
entities = data["entities"]
panel = data["entity_panel"]
pack = data["statement_pack"]
channels_all = data["channel_mix"]
countries_all = data["country_flows"]
trade_all = data["trade_profile"]
corridors_all = data["corridor_mix"]

ENTITY_LABEL = dict(zip(entities["entity_id"], entities["entity_name"]))
ENTITY_SECTOR = dict(zip(entities["entity_id"], entities["sector"]))
FISCAL_YEARS = sorted(panel["fiscal_year"].unique())
ALL_SECTORS = sorted(entities["sector"].unique())

# Which bank-side transaction groups land in a financials line at all. The
# intragroup sweeps do not: they are the same money moving between the
# entity's own accounts, so counting them as income would double-count.
LEG_APPLICABILITY = {
    "collections": ("Revenue - domestic collections", True),
    "supplier_payments": ("Cost of sales - domestic suppliers", True),
    "payroll": ("Employee costs", True),
    "tax": ("Taxation paid", True),
    "intercompany_sweeps": ("Not a financials line - treasury movement", False),
}


# --------------------------------------------------------------------------
# Sidebar - left rail navigation and filters
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">SB</div>'
        '<div><div class="brand-title">Syn Bank</div>'
        '<div class="brand-sub">Wallet share analytics</div></div></div>',
        unsafe_allow_html=True,
    )

    active = st.radio("Section", SECTIONS, label_visibility="collapsed")

    st.divider()
    st.markdown("**Filters**")
    years = st.multiselect("Fiscal year", FISCAL_YEARS, default=FISCAL_YEARS)
    sectors = st.multiselect("Sector", ALL_SECTORS, default=ALL_SECTORS)

    st.divider()
    st.markdown("**Market financials**")
    upload = st.file_uploader(
        "Override with keyed figures (CSV)", type="csv", label_visibility="collapsed"
    )
    st.markdown(
        '<div class="muted">Read from the extraction worklist where a figure has been '
        'checked back against the source PDF, otherwise from the OCR of the published '
        'statements. Upload a filled worklist or template to override any line.</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Bank revenue assumptions**")
    fee_bps = st.slider("Fee on flow (bps)", 1, 60, 15, 1,
                        help="Blended transaction-banking margin on value routed")
    fee_per_txn = st.slider("Fee per transaction (R)", 0, 60, 12, 1)

    st.divider()
    if st.button("Rebuild from source", width="stretch"):
        st.cache_data.clear()
        build_all(force=True)
        st.rerun()

years = years or FISCAL_YEARS
sectors = sectors or ALL_SECTORS
scope = entities[entities["sector"].isin(sectors)]["entity_id"].tolist()

panel_f = panel[panel["entity_id"].isin(scope) & panel["fiscal_year"].isin(years)]
channels = channels_all[
    channels_all["entity_id"].isin(scope) & channels_all["fiscal_year"].isin(years)
]
countries = countries_all[
    countries_all["entity_id"].isin(scope) & countries_all["fiscal_year"].isin(years)
]
trade = trade_all[
    trade_all["entity_id"].isin(scope) & trade_all["fiscal_year"].isin(years)
]
corridors = corridors_all[
    corridors_all["entity_id"].isin(scope) & corridors_all["fiscal_year"].isin(years)
]

bank = bank_view(panel_f, channels)
market = load_market(upload.getvalue() if upload else None)
market_f = market[market["entity_id"].isin(scope) & market["fiscal_year"].isin(years)]
recon = reconcile(bank, market_f)

# Line-level view of the same question: every figure the extraction worklist
# carries, against the computed statement pack line it corresponds to.
worklist = load_reported(upload.getvalue() if upload else None)
lines_all = worklist_comparison(pack, worklist)
lines_all["sector"] = lines_all["entity_id"].map(ENTITY_SECTOR)
lines = lines_all[
    lines_all["entity_id"].isin(scope) & lines_all["fiscal_year"].isin(years)
]
compared = lines[lines["comparison"] == "compared"]

# One row per entity across the selected years, which is the grain every
# section headlines on.
by_entity = (
    recon.groupby(["entity_id", "entity_name", "sector"], as_index=False)
    .agg(
        bank_income=("bank_income", "sum"),
        bank_expenditure=("bank_expenditure", "sum"),
        bank_total_flow=("bank_total_flow", "sum"),
        txn_count=("txn_count", "sum"),
        # min_count keeps an entity with no published figures at NaN rather
        # than folding it to zero, which would read as a 100% discrepancy.
        market_income=("market_income", lambda s: s.sum(min_count=1)),
        market_expenditure=("market_expenditure", lambda s: s.sum(min_count=1)),
        years_with_market=("has_market", "sum"),
    )
)
for metric in ("income", "expenditure"):
    market_column = pd.to_numeric(by_entity[f"market_{metric}"], errors="coerce")
    by_entity[f"market_{metric}"] = market_column
    by_entity[f"{metric}_gap"] = market_column - by_entity[f"bank_{metric}"]
    by_entity[f"{metric}_gap_pct"] = (
        by_entity[f"{metric}_gap"] / market_column.replace(0, np.nan)
    )
by_entity["covered"] = by_entity["years_with_market"] > 0

covered = by_entity[by_entity["covered"]]


st.markdown(f"## {active.split(' · ')[1]}")
st.markdown(
    f'<div class="muted">{len(scope)} entities · {years[0]} to {years[-1]} · '
    f'market financials available for {len(covered)} of {len(by_entity)}</div>',
    unsafe_allow_html=True,
)
st.write("")


# ==========================================================================
# 0 - Total Summary
# ==========================================================================

if active == SECTIONS[0]:
    bank_income = covered["bank_income"].sum()
    bank_spend = covered["bank_expenditure"].sum()
    market_income = covered["market_income"].sum()
    market_spend = covered["market_expenditure"].sum()
    income_gap = market_income - bank_income
    spend_gap = market_spend - bank_spend

    card_row([
        card("Recorded income · Syn Bank", zar(bank_income),
             f"{len(covered)} entities with market data"),
        card("Recorded income · Market", zar(market_income), "published statements"),
        card("Income discrepancy", zar(income_gap),
             f"{pct(income_gap / market_income) if market_income else '-'} of market",
             "down" if income_gap > 0 else "up"),
    ])
    st.write("")
    card_row([
        card("Recorded expenditure · Syn Bank", zar(bank_spend)),
        card("Recorded expenditure · Market", zar(market_spend), "published statements"),
        card("Expenditure discrepancy", zar(spend_gap),
             f"{pct(spend_gap / market_spend) if market_spend else '-'} of market",
             "down" if spend_gap > 0 else "up"),
    ])
    st.markdown(
        '<div class="muted">The Syn Bank ledger is a sample of the transaction book, so the '
        'discrepancy runs wide by construction. Read the ordering and the sector mix rather '
        'than the absolute rand gap.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if covered.empty:
        st.warning(
            "No market financials resolved for the current filters. Widen the sector "
            "or year selection, or upload keyed figures in the sidebar."
        )
    else:
        left, right = st.columns([2, 3])

        with left:
            section("Syn Bank against market", "Totals across entities with both ledgers.")
            fig = go.Figure()
            fig.add_bar(x=["Income", "Expenditure"], y=[bank_income, bank_spend],
                        name="Syn Bank", marker_color=BANK)
            fig.add_bar(x=["Income", "Expenditure"], y=[market_income, market_spend],
                        name="Market", marker_color=MARKET)
            fig.update_layout(barmode="group", yaxis_title="ZAR")
            st.plotly_chart(style(fig, 380), width="stretch")

        with right:
            section("Share of market activity Syn Bank sees",
                    "Coverage is bank flow over reported flow; the remainder is the gap.")
            coverage = pd.DataFrame({
                "Ledger": ["Income", "Expenditure"],
                "Seen": [bank_income, bank_spend],
                "Unseen": [max(income_gap, 0), max(spend_gap, 0)],
            })
            fig = go.Figure()
            fig.add_bar(y=coverage["Ledger"], x=coverage["Seen"], orientation="h",
                        name="Seen by Syn Bank", marker_color=BANK)
            fig.add_bar(y=coverage["Ledger"], x=coverage["Unseen"], orientation="h",
                        name="Not on the bank's rails", marker_color=NEG)
            fig.update_layout(barmode="stack", xaxis_title="ZAR")
            st.plotly_chart(style(fig, 380), width="stretch")

        st.divider()
        section("Ordered discrepancy by entity",
                "Income gap left, expenditure gap right. A wide bar is an entity whose "
                "reported activity Syn Bank barely touches.")

        ranked = covered.assign(
            total_gap=lambda d: d["income_gap"].fillna(0) + d["expenditure_gap"].fillna(0)
        ).sort_values("total_gap")

        fig = go.Figure()
        fig.add_bar(
            y=ranked["entity_name"], x=-ranked["income_gap"].fillna(0), orientation="h",
            name="Income gap", marker_color=ACCENT,
            customdata=ranked["income_gap"],
            hovertemplate="%{y}<br>Income gap %{customdata:,.0f}<extra></extra>",
        )
        fig.add_bar(
            y=ranked["entity_name"], x=ranked["expenditure_gap"].fillna(0), orientation="h",
            name="Expenditure gap", marker_color=MARKET,
            hovertemplate="%{y}<br>Expenditure gap %{x:,.0f}<extra></extra>",
        )
        fig.update_layout(barmode="relative", xaxis_title="ZAR  (left: income · right: expenditure)")
        st.plotly_chart(style(fig, max(320, 46 * len(ranked))), width="stretch")

        show = covered[[
            "entity_id", "entity_name", "sector", "bank_income", "market_income",
            "income_gap", "income_gap_pct", "bank_expenditure", "market_expenditure",
            "expenditure_gap", "expenditure_gap_pct",
        ]].sort_values("income_gap", ascending=False).copy()
        for column in show.columns:
            if column.endswith("_pct"):
                show[column] = show[column].map(pct)
            elif show[column].dtype.kind == "f":
                show[column] = show[column].map(zar)
        table(show)


# ==========================================================================
# 1 - Sector Analysis
# ==========================================================================

elif active == SECTIONS[1]:
    sector_totals = (
        by_entity.groupby("sector", as_index=False)
        .agg(
            bank_flow=("bank_total_flow", "sum"),
            income_gap=("income_gap", "sum"),
            expenditure_gap=("expenditure_gap", "sum"),
            entities=("entity_id", "nunique"),
        )
    )
    sector_totals["total_gap"] = (
        sector_totals["income_gap"].fillna(0) + sector_totals["expenditure_gap"].fillna(0)
    )

    left, right = st.columns(2)
    with left:
        section("All sector split", "Share of Syn Bank flow by sector.")
        st.plotly_chart(
            donut(sector_totals["sector"], sector_totals["bank_flow"], "Syn Bank flow", height=400),
            width="stretch",
        )
    with right:
        section("Per sector discrepancies", "Share of the total gap to market that each sector carries.")
        gaps = sector_totals[sector_totals["total_gap"] > 0]
        if gaps.empty:
            st.info("No positive discrepancy resolved for the current filters.")
        else:
            st.plotly_chart(
                donut(gaps["sector"], gaps["total_gap"], "Gap to market", height=400),
                width="stretch",
            )

    st.divider()
    section("Per sector detail", "Syn Bank transactions: income against expenses, entities and geography.")

    present = [s for s in sectors if s in set(by_entity["sector"])]
    if not present:
        st.info("No sectors in scope.")
    else:
        total_bank_flow = by_entity["bank_total_flow"].sum()
        for tab, sector_name in zip(st.tabs([s.replace("_", " ").title() for s in present]), present):
            with tab:
                members = by_entity[by_entity["sector"] == sector_name]
                sector_flow = members["bank_total_flow"].sum()

                card_row([
                    card("Entities", f"{len(members)}", f"of {len(by_entity)} in scope"),
                    card("Share of Syn Bank", pct(sector_flow / total_bank_flow) if total_bank_flow else "-",
                         "by total flow"),
                    card("Income · Syn Bank", zar(members["bank_income"].sum())),
                    card("Expenses · Syn Bank", zar(members["bank_expenditure"].sum())),
                    card("Gap to market",
                         zar(members["income_gap"].fillna(0).sum() + members["expenditure_gap"].fillna(0).sum()),
                         f"{int(members['covered'].sum())} entities with market data"),
                ])
                st.write("")

                left, right = st.columns([2, 3])
                panel_height = max(300, 60 * len(members))

                with left:
                    section("Income against expenses", "Syn Bank transactions per entity in this sector.")
                    ordered = members.sort_values("bank_income")
                    fig = go.Figure()
                    fig.add_bar(y=ordered["entity_name"], x=ordered["bank_income"],
                                orientation="h", name="Income", marker_color=POS)
                    fig.add_bar(y=ordered["entity_name"], x=-ordered["bank_expenditure"],
                                orientation="h", name="Expenses", marker_color=NEG,
                                customdata=ordered["bank_expenditure"],
                                hovertemplate="%{y}<br>Expenses %{customdata:,.0f}<extra></extra>")
                    fig.update_layout(barmode="relative", xaxis_title="ZAR")
                    st.plotly_chart(style(fig, panel_height), width="stretch")

                    show = members[["entity_id", "entity_name", "txn_count", "bank_total_flow"]].copy()
                    show["% of Syn Bank"] = (
                        members["bank_total_flow"] / total_bank_flow
                    ).map(pct) if total_bank_flow else "-"
                    show["bank_total_flow"] = show["bank_total_flow"].map(zar)
                    show["txn_count"] = show["txn_count"].map(lambda v: f"{int(v):,}")
                    table(show)

                with right:
                    section("Geographical location of transactions",
                            "Counterparty countries, bubble size by value routed.")
                    geo = (
                        countries[countries["entity_id"].isin(members["entity_id"])]
                        .groupby("counterparty_country", as_index=False)
                        .agg(value_zar=("value_zar", "sum"), txn_count=("txn_count", "sum"))
                    )
                    geo = geo[geo["counterparty_country"].isin(COUNTRY_COORDS)]
                    if geo.empty:
                        st.info("No cross-border activity recorded for this sector.")
                    else:
                        geo["lat"] = geo["counterparty_country"].map(lambda c: COUNTRY_COORDS[c][0])
                        geo["lon"] = geo["counterparty_country"].map(lambda c: COUNTRY_COORDS[c][1])
                        fig = go.Figure(go.Scattergeo(
                            lat=geo["lat"], lon=geo["lon"], text=geo["counterparty_country"],
                            customdata=np.stack([geo["value_zar"], geo["txn_count"]], axis=-1),
                            hovertemplate="%{text}<br>R %{customdata[0]:,.0f}"
                                          "<br>%{customdata[1]:,} transactions<extra></extra>",
                            marker=dict(
                                size=geo["value_zar"],
                                sizemode="area",
                                sizeref=2.0 * geo["value_zar"].max() / (55.0 ** 2),
                                sizemin=4,
                                color=ACCENT_SOFT,
                                line=dict(width=0.5, color=BG),
                                opacity=0.8,
                            ),
                        ))
                        fig.update_geos(
                            bgcolor="rgba(0,0,0,0)", showland=True, landcolor="#1c1c24",
                            showcountries=True, countrycolor=BORDER, showcoastlines=False,
                            showframe=False, showocean=False, projection_type="natural earth",
                        )
                        st.plotly_chart(style(fig, panel_height, legend=False), width="stretch")


# ==========================================================================
# 2 - Per Entity Analysis
# ==========================================================================

elif active == SECTIONS[2]:
    focus = st.selectbox(
        "Entity", scope or entities["entity_id"].tolist(),
        format_func=lambda e: f"{e} · {ENTITY_LABEL[e]}",
    )
    row = by_entity[by_entity["entity_id"] == focus]
    row = row.iloc[0] if not row.empty else None
    entity_channels = channels[channels["entity_id"] == focus]
    entity_lines = lines[lines["entity_id"] == focus]

    if row is None:
        st.info("No transactions in scope for this entity.")
    else:
        card_row([
            card("Total transaction amount", zar(row["bank_total_flow"]),
                 f"{int(row['txn_count']):,} transactions"),
            card("Field of business", ENTITY_SECTOR[focus].replace("_", " ").title()),
            card("Income · Syn Bank", zar(row["bank_income"])),
            card("Expenditure · Syn Bank", zar(row["bank_expenditure"])),
            card("Market documentation",
                 f"{int(row['years_with_market'])} of {len(years)} years",
                 "published statements matched",
                 "up" if row["years_with_market"] else "down"),
        ])
        st.write("")

        section("Transaction groups as collected into the financials grouping",
                "Each Syn Bank transaction group and the financials line it feeds. "
                "Groups that feed no line are excluded from income and expenditure.")

        groups = (
            entity_channels.groupby("leg_type", as_index=False)
            .agg(txn_count=("txn_count", "sum"), value_zar=("value_zar", "sum"))
        )
        groups["financials_line"] = groups["leg_type"].map(
            lambda leg: LEG_APPLICABILITY.get(leg, ("Unmapped", False))[0]
        )
        groups["applies"] = groups["leg_type"].map(
            lambda leg: LEG_APPLICABILITY.get(leg, ("Unmapped", False))[1]
        )
        groups["share"] = groups["value_zar"] / groups["value_zar"].sum()

        left, right = st.columns([3, 2])
        with left:
            ordered = groups.sort_values("value_zar")
            fig = go.Figure(go.Bar(
                y=ordered["leg_type"].str.replace("_", " "), x=ordered["value_zar"],
                orientation="h",
                marker_color=[POS if applies else MUTED for applies in ordered["applies"]],
                customdata=ordered["financials_line"],
                hovertemplate="%{y}<br>R %{x:,.0f}<br>%{customdata}<extra></extra>",
            ))
            fig.update_layout(xaxis_title="ZAR")
            st.plotly_chart(style(fig, 320, legend=False), width="stretch")
            st.markdown(
                f'<div class="muted"><span class="pill pill-ok">green</span> feeds a financials '
                f'line · <span class="pill">grey</span> does not apply</div>',
                unsafe_allow_html=True,
            )
        with right:
            show = groups[["leg_type", "financials_line", "txn_count", "value_zar", "share"]].copy()
            show["leg_type"] = show["leg_type"].str.replace("_", " ")
            show["txn_count"] = show["txn_count"].map(lambda v: f"{int(v):,}")
            show["value_zar"] = show["value_zar"].map(zar)
            show["share"] = show["share"].map(pct)
            show["applies"] = groups["applies"].map({True: "applies", False: "excluded"})
            table(show.rename(columns={
                "leg_type": "Transaction group", "financials_line": "Financials line",
                "txn_count": "Transactions", "value_zar": "Value", "share": "Share",
                "applies": "Status",
            }))

        st.divider()
        section("Per year, per financials line",
                "The Syn Bank proxy against the entity's own documentation, for every line "
                "the extraction worklist tracks - not only revenue and cost of sales. Lines "
                "the worklist has not yet resolved are marked outstanding.")

        detail = entity_lines.sort_values(["line_item", "fiscal_year"]).rename(columns={
            "fiscal_year": "Fiscal year",
            "line_item": "Financials line",
            "metric_type": "Type",
            "computed_value": "Syn Bank",
            "reported_value": "Documented",
            "difference": "Difference",
            "coverage": "Coverage",
            "status": "Worklist status",
            "note": "Note",
        })
        if detail.empty:
            st.info("No statement lines built for this entity in the selected years.")
        else:
            matched = detail[detail["comparison"] == "compared"]
            amounts = matched[matched["Type"] == "amount"]
            card_row([
                card("Lines with documentation", f"{len(matched)}",
                     f"of {len(detail)} line-years tracked",
                     "up" if len(matched) else "down"),
                card("Median coverage",
                     pct(amounts["Coverage"].median()) if len(amounts) else "-",
                     "Syn Bank ÷ documented, amounts only"),
                card("Material variances", f"{int(matched['material'].sum())}",
                     "beyond 25% of the reported figure",
                     "down" if matched["material"].any() else "up"),
            ])
            st.write("")

            if not amounts.empty:
                fig = go.Figure()
                labels = amounts["Financials line"] + " · " + amounts["Fiscal year"]
                fig.add_bar(x=labels, y=amounts["Syn Bank"], name="Syn Bank", marker_color=BANK)
                fig.add_bar(x=labels, y=amounts["Documented"], name="Documented",
                            marker_color=MARKET)
                # The two ledgers differ by up to three orders of magnitude, so a
                # linear axis renders the bank bar as a flat line against the
                # published figure.
                fig.update_layout(barmode="group", yaxis_title="ZAR (log)",
                                  yaxis_type="log")
                st.plotly_chart(style(fig, 380), width="stretch")

            display = detail[[
                "Fiscal year", "Financials line", "Type", "Syn Bank", "Documented",
                "Difference", "Coverage", "Worklist status", "Note",
            ]].copy()
            ratio_rows = display["Type"] == "ratio"
            for column in ("Syn Bank", "Documented", "Difference"):
                # A ratio line is a margin, not a rand figure, so it is shown as
                # a percentage in the same column rather than formatted as money.
                display[column] = np.where(
                    ratio_rows, display[column].map(pct), display[column].map(zar)
                )
            display["Coverage"] = display["Coverage"].map(pct)
            display["Note"] = display["Note"].fillna("")
            table(display, height=440)


# ==========================================================================
# 3 - Total Opportunity for Growth
# ==========================================================================

elif active == SECTIONS[3]:
    st.markdown(
        f'<div class="muted">Bank revenue modelled at {fee_bps} bps on value routed plus '
        f'R{fee_per_txn} per transaction. Adjust both in the sidebar.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    opportunity = covered.copy()
    if opportunity.empty:
        st.warning(
            "Opportunity sizing needs market financials on at least one entity. "
            "Widen the filters or upload keyed figures in the sidebar."
        )
    else:
        opportunity["missed_amount"] = (
            opportunity["income_gap"].clip(lower=0).fillna(0)
            + opportunity["expenditure_gap"].clip(lower=0).fillna(0)
        )
        market_total = (
            opportunity["market_income"].fillna(0) + opportunity["market_expenditure"].fillna(0)
        )
        opportunity["missed_pct"] = opportunity["missed_amount"] / market_total.replace(0, np.nan)

        # Transactions the bank would carry if it held the missing wallet, at the
        # entity's own observed average ticket.
        opportunity["avg_ticket"] = (
            opportunity["bank_total_flow"] / opportunity["txn_count"].replace(0, np.nan)
        )
        opportunity["implied_txns"] = (
            opportunity["missed_amount"] / opportunity["avg_ticket"]
        ).replace([np.inf, -np.inf], np.nan)
        opportunity["fee_revenue"] = (
            opportunity["missed_amount"] * fee_bps / 10_000
            + opportunity["implied_txns"].fillna(0) * fee_per_txn
        )
        opportunity["avg_fee_per_txn"] = (
            opportunity["fee_revenue"] / opportunity["implied_txns"].replace(0, np.nan)
        )
        # Applicability: how much of what the entity already routes through Syn
        # Bank lands in a financials line - the share the bank can evidence.
        applicable = (
            channels[channels["leg_type"].isin(
                [leg for leg, (_, applies) in LEG_APPLICABILITY.items() if applies]
            )]
            .groupby("entity_id")["value_zar"].sum()
        )
        routed = channels.groupby("entity_id")["value_zar"].sum()
        opportunity["applicability"] = (
            opportunity["entity_id"].map(applicable) / opportunity["entity_id"].map(routed)
        )

        card_row([
            card("Total missed wallet", zar(opportunity["missed_amount"].sum()),
                 f"{pct(opportunity['missed_amount'].sum() / market_total.sum()) if market_total.sum() else '-'}"
                 " of reported activity",
                 "down"),
            card("Implied transactions", f"{opportunity['implied_txns'].sum():,.0f}",
                 "at each entity's average ticket"),
            card("Modelled fee revenue", zar(opportunity["fee_revenue"].sum()),
                 f"{fee_bps} bps + R{fee_per_txn}/txn"),
            card("Avg fee per transaction",
                 f"R {opportunity['fee_revenue'].sum() / max(opportunity['implied_txns'].sum(), 1):,.2f}"),
            card("Median applicability", pct(opportunity["applicability"].median()),
                 "of routed value feeds a financials line"),
        ])
        st.write("")

        left, right = st.columns(2)

        with left:
            section("Greatest missed segments", "Wallet reported to the market that Syn Bank does not carry.")
            top_missed = opportunity.nlargest(10, "missed_amount").sort_values("missed_amount")
            fig = go.Figure(go.Bar(
                y=top_missed["entity_name"], x=top_missed["missed_amount"], orientation="h",
                marker_color=NEG, customdata=top_missed["missed_pct"],
                hovertemplate="%{y}<br>R %{x:,.0f}<br>%{customdata:.1%} of reported<extra></extra>",
            ))
            fig.update_layout(xaxis_title="ZAR")
            st.plotly_chart(style(fig, 400, legend=False), width="stretch")

        with right:
            section("Greatest potential for revenue",
                    "Modelled fee income, sized by the transactions the bank would carry.")
            top_revenue = opportunity.nlargest(10, "fee_revenue")
            fig = go.Figure(go.Scatter(
                x=top_revenue["implied_txns"], y=top_revenue["fee_revenue"],
                mode="markers+text", text=top_revenue["entity_name"], textposition="top center",
                textfont=dict(size=10, color=MUTED),
                marker=dict(
                    size=top_revenue["missed_amount"],
                    sizemode="area",
                    sizeref=2.0 * top_revenue["missed_amount"].max() / (45.0 ** 2),
                    sizemin=6, color=ACCENT, opacity=0.8,
                ),
                customdata=top_revenue["avg_fee_per_txn"],
                hovertemplate="%{text}<br>%{x:,.0f} transactions<br>R %{y:,.0f} fees"
                              "<br>R %{customdata:,.2f} per transaction<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Implied transactions", yaxis_title="Modelled fee revenue (ZAR)")
            st.plotly_chart(style(fig, 400, legend=False), width="stretch")

        section("Greatest applicability",
                "Share of each entity's routed value that already feeds a financials line - "
                "where the bank can evidence the relationship, not just move money.")
        applicability = (
            opportunity.dropna(subset=["applicability"])
            .sort_values("applicability")
        )
        fig = go.Figure(go.Bar(
            y=applicability["entity_name"], x=applicability["applicability"], orientation="h",
            marker_color=[POS if v >= 0.6 else WARN if v >= 0.4 else NEG
                          for v in applicability["applicability"]],
            hovertemplate="%{y}<br>%{x:.1%} of routed value<extra></extra>",
        ))
        fig.update_layout(xaxis_tickformat=".0%", xaxis_title="Share of routed value applying to a financials line")
        st.plotly_chart(style(fig, max(320, 42 * len(applicability)), legend=False), width="stretch")

        show = opportunity[[
            "entity_id", "entity_name", "sector", "missed_amount", "missed_pct",
            "implied_txns", "avg_fee_per_txn", "fee_revenue", "applicability",
        ]].sort_values("missed_amount", ascending=False).copy()
        show["missed_amount"] = show["missed_amount"].map(zar)
        show["fee_revenue"] = show["fee_revenue"].map(zar)
        show["implied_txns"] = show["implied_txns"].map(
            lambda v: "-" if pd.isna(v) else f"{v:,.0f}")
        show["avg_fee_per_txn"] = show["avg_fee_per_txn"].map(
            lambda v: "-" if pd.isna(v) else f"R {v:,.2f}")
        for column in ("missed_pct", "applicability"):
            show[column] = show[column].map(pct)
        table(show.rename(columns={
            "missed_amount": "Missed wallet", "missed_pct": "% of reported",
            "implied_txns": "Implied transactions", "avg_fee_per_txn": "Avg fee / txn",
            "fee_revenue": "Modelled fee revenue", "applicability": "Applicability",
        }))


# ==========================================================================
# 4 - Reported vs Computed
# ==========================================================================

elif active == SECTIONS[4]:
    st.markdown(
        '<div class="muted">Figures checked back against the source PDFs in '
        '<code>benchmarks/extraction_worklist</code>, against the same line computed from '
        'Syn Bank transactions. The computed side only ever sees the bank\'s own rails, so '
        'on an amount line it sits below the published figure by construction - the ratio '
        'between them is the entity\'s implied coverage, and it is the consistency of that '
        'ratio, not its size, that says whether an extraction is sound.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if lines.empty:
        st.warning("No worklist lines in scope. Widen the sector or year filters.")
    else:
        amounts = compared[compared["metric_type"] == "amount"].copy()
        # Implied coverage is read the other way up here - reported over
        # computed - because it lands in the tens-to-hundreds range an analyst
        # can hold in their head, rather than four leading zeros.
        amounts["implied_multiple"] = (
            amounts["reported_value"] / amounts["computed_value"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        median_multiple = amounts["implied_multiple"].median()
        # An extraction whose implied coverage sits an order of magnitude off
        # the book's own median is far more likely a unit or scale misread than
        # a genuinely unusual banking relationship.
        outliers = amounts[
            (amounts["implied_multiple"] > median_multiple * 10)
            | (amounts["implied_multiple"] < median_multiple / 10)
        ]

        filled = lines["reported_value"].notna().sum()
        card_row([
            card("Worklist lines in scope", f"{len(lines):,}",
                 f"{lines['entity_id'].nunique()} entities · {len(years)} years"),
            card("Lines with a figure", f"{int(filled):,}",
                 f"{pct(filled / len(lines))} of the worklist extracted",
                 "up" if filled else "down"),
            card("Median implied coverage",
                 f"{median_multiple:,.0f}x" if pd.notna(median_multiple) else "-",
                 "reported ÷ computed, amounts"),
            card("Scale outliers", f"{len(outliers)}",
                 "an order of magnitude off the median", "down" if len(outliers) else "up"),
            card("Material variances", f"{int(compared['material'].sum())}",
                 f"of {len(compared)} compared lines",
                 "down" if compared["material"].any() else "up"),
        ])
        st.write("")

        st.divider()
        section("Revenue: reported against computed",
                "Total revenue per entity, both ledgers, summed across the years in scope. "
                "Log axis - the two sides differ by up to three orders of magnitude.")

        revenue = compared[compared["line_item"] == "Total revenue"]
        if revenue.empty:
            st.info("No revenue lines resolved for the current filters.")
        else:
            by_name = (
                revenue.groupby("entity_name", as_index=False)
                .agg(reported=("reported_value", "sum"),
                     computed=("computed_value", "sum"),
                     entity_years=("fiscal_year", "count"))
                .sort_values("reported", ascending=False)
            )
            by_name["multiple"] = by_name["reported"] / by_name["computed"].replace(0, np.nan)

            fig = go.Figure()
            fig.add_bar(x=by_name["entity_name"], y=by_name["reported"],
                        name="Reported (worklist)", marker_color=MARKET)
            fig.add_bar(x=by_name["entity_name"], y=by_name["computed"],
                        name="Computed (Syn Bank)", marker_color=BANK)
            fig.update_layout(barmode="group", yaxis_type="log", yaxis_title="ZAR (log)")
            st.plotly_chart(style(fig, 400), width="stretch")

            section("Implied coverage per entity",
                    "Reported revenue over computed revenue. A consistent book would cluster; "
                    "bars far from the median are the extractions worth re-reading.")
            ordered = by_name.dropna(subset=["multiple"]).sort_values("multiple")
            fig = go.Figure(go.Bar(
                y=ordered["entity_name"], x=ordered["multiple"], orientation="h",
                marker_color=[
                    NEG if v > median_multiple * 10 or v < median_multiple / 10 else ACCENT
                    for v in ordered["multiple"]
                ],
                hovertemplate="%{y}<br>%{x:,.0f}x implied coverage<extra></extra>",
            ))
            fig.add_vline(x=median_multiple, line_dash="dot", line_color=MUTED,
                          annotation_text=f"median {median_multiple:,.0f}x",
                          annotation_font_color=MUTED)
            fig.update_layout(xaxis_type="log", xaxis_title="Reported ÷ computed (log)")
            st.plotly_chart(style(fig, max(320, 40 * len(ordered)), legend=False),
                            width="stretch")

            show = by_name.copy()
            show["difference"] = show["reported"] - show["computed"]
            show["reported"] = show["reported"].map(zar)
            show["computed"] = show["computed"].map(zar)
            show["difference"] = show["difference"].map(zar)
            show["multiple"] = show["multiple"].map(
                lambda v: "-" if pd.isna(v) else f"{v:,.0f}x")
            table(show.rename(columns={
                "entity_name": "Entity", "entity_years": "Years",
                "reported": "Reported", "computed": "Computed",
                "difference": "Difference", "multiple": "Implied coverage",
            }))

        st.divider()
        section("Extraction completeness by line",
                "How far the worklist got on each financials line. Outstanding lines are the "
                "next figures to key in; suspect ones carry a plausibility warning.")

        completeness = (
            lines.groupby(["line_item", "status"]).size().unstack(fill_value=0)
        )
        for status in ("have", "suspect", "missing"):
            if status not in completeness.columns:
                completeness[status] = 0
        completeness = completeness.sort_values("have")
        fig = go.Figure()
        for status, colour, label in (
            ("have", POS, "extracted"),
            ("suspect", WARN, "suspect"),
            ("missing", MUTED, "outstanding"),
        ):
            fig.add_bar(y=completeness.index, x=completeness[status], orientation="h",
                        name=label, marker_color=colour)
        fig.update_layout(barmode="stack", xaxis_title="Entity-years")
        st.plotly_chart(style(fig, max(320, 44 * len(completeness))), width="stretch")

        st.divider()
        section("Every compared line",
                "Reported less computed, per entity-year. Filter to a single line to work "
                "through one figure at a time.")

        choices = sorted(lines["line_item"].unique())
        left, right = st.columns([3, 2])
        with left:
            chosen = st.multiselect("Financials line", choices, default=choices)
        with right:
            only_material = st.checkbox("Material variances only", value=False)

        detail = lines[lines["line_item"].isin(chosen or choices)]
        if only_material:
            detail = detail[detail["material"]]

        display = detail.sort_values(
            ["entity_id", "line_item", "fiscal_year"]
        )[[
            "entity_id", "entity_name", "fiscal_year", "line_item", "metric_type",
            "reported_value", "computed_value", "difference", "coverage",
            "status", "source_pdf", "note",
        ]].copy()
        ratio_rows = display["metric_type"] == "ratio"
        for column in ("reported_value", "computed_value", "difference"):
            display[column] = np.where(
                ratio_rows, display[column].map(pct), display[column].map(zar)
            )
        display["coverage"] = display["coverage"].map(pct)
        display[["source_pdf", "note"]] = display[["source_pdf", "note"]].fillna("")
        table(display.rename(columns={
            "entity_id": "Entity", "entity_name": "Name", "fiscal_year": "Year",
            "line_item": "Financials line", "metric_type": "Type",
            "reported_value": "Reported", "computed_value": "Computed",
            "difference": "Difference", "coverage": "Coverage",
            "status": "Status", "source_pdf": "Source PDF", "note": "Note",
        }), height=520)

        st.download_button(
            "Download comparison (CSV)",
            detail.to_csv(index=False).encode("utf-8"),
            file_name="reported_vs_computed.csv",
            mime="text/csv",
        )


# ==========================================================================
# 5 - Trade & Cross-Border
# ==========================================================================

else:
    st.markdown(
        '<div class="muted">Net import position is <b>imports less exports</b>, so a '
        'positive figure is a net importer and reads as a drag on GDP; a negative figure '
        'is a net exporter contributing to it. Note this is the reverse of the '
        'national-accounts sign, where net exports enter GDP positively - the magnitudes '
        'are the same, the sign is flipped.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if panel_f.empty:
        st.warning("No entities in scope. Widen the sector or year filters.")
    else:
        # Trade finance instruments carry the import and export legs; the trade
        # corridor of the cross-border book is the settlement that goes with
        # them. Both are trade, so both count toward trade intensity, but only
        # the instrument legs have a direction to net off.
        trade_flow = (
            panel_f.groupby(["entity_id", "entity_name", "sector"], as_index=False)
            .agg(
                imports=("tf_import", "sum"),
                exports=("tf_export", "sum"),
                tf_value=("tf_total_value_zar", "sum"),
                xb_trade=("xb_corridor_trade", "sum"),
                total_flow=("total_flow_zar", "sum"),
                open_exposure=("tf_open_exposure_zar", "sum"),
                tenor=("tf_weighted_avg_tenor_days", "mean"),
                countries=("tf_countries", "max"),
            )
        )
        trade_flow["net_import"] = trade_flow["imports"] - trade_flow["exports"]
        trade_flow["trade_value"] = trade_flow["tf_value"] + trade_flow["xb_trade"]
        trade_flow["intensity"] = (
            trade_flow["trade_value"] / trade_flow["total_flow"].replace(0, np.nan)
        )
        trade_flow["position"] = np.where(
            trade_flow["net_import"] > 0, "net importer", "net exporter"
        )

        book_net = trade_flow["net_import"].sum()
        card_row([
            card("Gross trade value", zar(trade_flow["trade_value"].sum()),
                 "instruments plus trade corridor"),
            card("Imports", zar(trade_flow["imports"].sum()), "import leg of the instruments"),
            card("Exports", zar(trade_flow["exports"].sum()), "export leg of the instruments"),
            card("Net import position", zar(book_net),
                 "drag on GDP" if book_net > 0 else "contribution to GDP",
                 "down" if book_net > 0 else "up"),
            card("Median trade intensity", pct(trade_flow["intensity"].median()),
                 "trade ÷ total routed value"),
        ])
        st.write("")

        # One height for both columns so the two panels line up.
        trade_height = max(340, 40 * len(trade_flow))
        left, right = st.columns([3, 2])

        with left:
            section("Net import position by entity",
                    "Imports less exports. Bars to the right are net importers - value "
                    "leaving the country - and bars to the left are net exporters.")
            ordered = trade_flow.sort_values("net_import")
            fig = go.Figure(go.Bar(
                y=ordered["entity_name"], x=ordered["net_import"], orientation="h",
                marker_color=[NEG if v > 0 else POS for v in ordered["net_import"]],
                customdata=np.stack([ordered["imports"], ordered["exports"]], axis=-1),
                hovertemplate="%{y}<br>Net R %{x:,.0f}<br>Imports R %{customdata[0]:,.0f}"
                              "<br>Exports R %{customdata[1]:,.0f}<extra></extra>",
            ))
            fig.add_vline(x=0, line_color=BORDER)
            fig.update_layout(xaxis_title="ZAR  (right: net importer · left: net exporter)")
            st.plotly_chart(style(fig, trade_height, legend=False), width="stretch")

        with right:
            section("Imports against exports, by year",
                    "The book's trade balance over the years in scope.")
            by_year = (
                panel_f.groupby("fiscal_year", as_index=False)
                .agg(imports=("tf_import", "sum"), exports=("tf_export", "sum"))
            )
            by_year["net_import"] = by_year["imports"] - by_year["exports"]
            fig = go.Figure()
            fig.add_bar(x=by_year["fiscal_year"], y=by_year["imports"],
                        name="Imports", marker_color=NEG)
            fig.add_bar(x=by_year["fiscal_year"], y=by_year["exports"],
                        name="Exports", marker_color=POS)
            fig.add_trace(go.Scatter(
                x=by_year["fiscal_year"], y=by_year["net_import"], name="Net import position",
                mode="lines+markers", line=dict(color=ACCENT, width=2),
            ))
            fig.update_layout(barmode="group", yaxis_title="ZAR")
            st.plotly_chart(style(fig, trade_height), width="stretch")

        st.divider()
        section("Trade intensity",
                "Trade value as a share of everything the entity routes through Syn Bank. "
                "A high share is a relationship the bank holds through trade rather than "
                "through domestic collections.")

        intensity = trade_flow.dropna(subset=["intensity"]).sort_values("intensity")
        fig = go.Figure(go.Bar(
            y=intensity["entity_name"], x=intensity["intensity"], orientation="h",
            marker_color=[POS if v >= 0.3 else WARN if v >= 0.15 else MUTED
                          for v in intensity["intensity"]],
            customdata=intensity["trade_value"],
            hovertemplate="%{y}<br>%{x:.1%} of routed value"
                          "<br>R %{customdata:,.0f} trade<extra></extra>",
        ))
        fig.update_layout(xaxis_tickformat=".0%",
                          xaxis_title="Trade value ÷ total routed value")
        st.plotly_chart(style(fig, max(320, 40 * len(intensity)), legend=False),
                        width="stretch")

        st.divider()
        section("Trade channel mix",
                "How the trade value moves: the three trade finance instruments, and the "
                "cross-border trade corridor that settles alongside them. Shares are of "
                "each entity's own trade value, so the rows are comparable across "
                "entities of very different size.")

        instruments = (
            trade.groupby(["entity_id", "instrument_type"], as_index=False)["value_zar"].sum()
            .rename(columns={"instrument_type": "channel"})
        )
        corridor_trade = (
            corridors[corridors["corridor_type"] == "trade"]
            .groupby("entity_id", as_index=False)["value_zar"].sum()
            .assign(channel="cross-border trade corridor")
        )
        mix = pd.concat([instruments, corridor_trade], ignore_index=True)
        mix["entity_name"] = mix["entity_id"].map(ENTITY_LABEL)
        mix["share"] = mix["value_zar"] / mix.groupby("entity_id")["value_zar"].transform("sum")

        order = trade_flow.sort_values("trade_value")["entity_name"].tolist()
        fig = go.Figure()
        for colour, channel in zip(SEQUENCE, sorted(mix["channel"].unique())):
            part = mix[mix["channel"] == channel]
            fig.add_bar(
                y=part["entity_name"], x=part["share"], orientation="h",
                name=channel.replace("_", " "), marker_color=colour,
                customdata=part["value_zar"],
                hovertemplate="%{y}<br>%{x:.1%}<br>R %{customdata:,.0f}<extra></extra>",
            )
        fig.update_layout(barmode="stack", xaxis_tickformat=".0%",
                          xaxis_title="Share of the entity's trade value",
                          yaxis=dict(categoryorder="array", categoryarray=order))
        st.plotly_chart(style(fig, max(340, 42 * len(order))), width="stretch")

        st.divider()
        section("Trade book by entity",
                "Open exposure is the value of instruments still live at year end; tenor is "
                "the value-weighted average across them.")

        show = trade_flow[[
            "entity_id", "entity_name", "sector", "imports", "exports", "net_import",
            "position", "trade_value", "intensity", "open_exposure", "tenor", "countries",
        ]].sort_values("net_import", ascending=False).copy()
        for column in ("imports", "exports", "net_import", "trade_value", "open_exposure"):
            show[column] = show[column].map(zar)
        show["intensity"] = show["intensity"].map(pct)
        show["tenor"] = show["tenor"].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f} days")
        show["countries"] = show["countries"].map(
            lambda v: "-" if pd.isna(v) else f"{int(v)}")
        table(show.rename(columns={
            "entity_id": "Entity", "entity_name": "Name", "sector": "Sector",
            "imports": "Imports", "exports": "Exports", "net_import": "Net import position",
            "position": "Position", "trade_value": "Trade value",
            "intensity": "Trade intensity", "open_exposure": "Open exposure",
            "tenor": "Avg tenor", "countries": "Countries",
        }))
