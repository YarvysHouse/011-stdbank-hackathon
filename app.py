# -*- coding: utf-8 -*-
"""Board dashboard for the transaction-banking pipeline.

Five tabs, broad to granular:

    1  Reconciliation   - where the pipeline disagrees with reported financials
    2  Discrepancies    - which lines break, by entity and by year
    3  Statements       - the pseudo consolidated pack per entity
    4  Flows            - corridors, currencies, trade finance exposure
    5  Transactions     - transaction grain and data quality

Run with::

    uv run streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_data import build_all
from financial_statements import (
    COMPARABLE_LINES,
    PLAUSIBLE_RATIO_RANGES,
    REPORTED_TEMPLATE,
    compare_to_reported,
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

SEQUENCE = [ACCENT, "#22d3ee", "#f472b6", "#facc15", "#34d399", "#fb923c", "#818cf8", "#e879f9"]

st.set_page_config(
    page_title="Transaction Banking - Board Analytics",
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
      .card-value {{ color: {TEXT}; font-size: 1.85rem; font-weight: 650; line-height: 1.15; }}
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

      .stTabs [data-baseweb="tab-list"] {{ gap: 0.35rem; border-bottom: 1px solid {BORDER}; }}
      .stTabs [data-baseweb="tab"] {{
        background: transparent; border-radius: 10px 10px 0 0; padding: 0.6rem 1.1rem;
        color: {MUTED}; font-size: 0.88rem;
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
    if pd.isna(value):
        return "-"
    sign = "-" if value < 0 else ""
    v = abs(value)
    for cutoff, suffix in ((1e12, "tn"), (1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if v >= cutoff:
            return f"{sign}R {v / cutoff:,.{decimals}f}{suffix}"
    return f"{sign}R {v:,.0f}"


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


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Building analytics from source transactions...")
def load_artifacts() -> dict[str, pd.DataFrame]:
    return build_all()


@st.cache_data(show_spinner=False)
def load_reported(upload: bytes | None) -> pd.DataFrame | None:
    """Reported financials, from an upload or the on-disk template."""
    if upload is not None:
        reported = pd.read_csv(pd.io.common.BytesIO(upload))
    elif REPORTED_TEMPLATE.exists():
        reported = pd.read_csv(REPORTED_TEMPLATE)
    else:
        return None
    reported = reported.dropna(subset=["reported_value"])
    return reported if not reported.empty else None


@st.cache_data(show_spinner=False)
def run_comparison(pack: pd.DataFrame, reported: pd.DataFrame, materiality: float) -> pd.DataFrame:
    return compare_to_reported(pack, reported, materiality)


data = load_artifacts()
entities = data["entities"]
panel = data["entity_panel"]
panel_q = data["entity_panel_quarterly"]
pack = data["statement_pack"]
plaus = data["plausibility"]
quality = data["data_quality"]

ENTITY_LABEL = dict(zip(entities["entity_id"], entities["entity_name"]))
FISCAL_YEARS = sorted(panel["fiscal_year"].unique())


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">SB</div>'
        '<div><div class="brand-title">Transaction Banking</div>'
        '<div class="brand-sub">Board analytics pipeline</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Filters**")
    years = st.multiselect("Fiscal year", FISCAL_YEARS, default=FISCAL_YEARS)
    sectors = st.multiselect(
        "Sector", sorted(entities["sector"].unique()), default=sorted(entities["sector"].unique())
    )

    in_scope = entities[entities["sector"].isin(sectors)]["entity_id"].tolist()
    focus = st.selectbox(
        "Focus entity",
        in_scope or entities["entity_id"].tolist(),
        format_func=lambda e: f"{e} - {ENTITY_LABEL[e]}",
    )

    st.divider()
    st.markdown("**Reported financials**")
    upload = st.file_uploader("Upload filled template (CSV)", type="csv", label_visibility="collapsed")
    materiality = st.slider("Materiality threshold", 0.02, 0.50, 0.10, 0.01,
                            help="Deviation from the entity's revenue coverage before a line is flagged")

    st.divider()
    if st.button("Rebuild from source", width="stretch"):
        st.cache_data.clear()
        build_all(force=True)
        st.rerun()
    st.markdown(
        '<div class="muted">Cash-basis proxy built from bank-side flows only. '
        'Absolute rands are a sample of reported figures; compare shape and coverage.</div>',
        unsafe_allow_html=True,
    )

years = years or FISCAL_YEARS
sectors = sectors or sorted(entities["sector"].unique())
scope = entities[entities["sector"].isin(sectors)]["entity_id"].tolist()

panel_f = panel[panel["entity_id"].isin(scope) & panel["fiscal_year"].isin(years)]
pack_f = pack[pack["entity_id"].isin(scope) & pack["fiscal_year"].isin(years)]
plaus_f = plaus[plaus["entity_id"].isin(scope) & plaus["fiscal_year"].isin(years)]

reported = load_reported(upload.getvalue() if upload else None)
comparison = None
if reported is not None:
    try:
        comparison = run_comparison(pack, reported, materiality)
        comparison = comparison[
            comparison["entity_id"].isin(scope) & comparison["fiscal_year"].isin(years)
        ]
    except ValueError:
        comparison = None


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown("## Pipeline reconciliation and financial analytics")
st.markdown(
    f'<div class="muted">{len(scope)} entities · {years[0]} to {years[-1]} · '
    f'three data sources reconciled to a pseudo consolidated statement pack</div>',
    unsafe_allow_html=True,
)
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1 · Reconciliation",
    "2 · Discrepancies",
    "3 · Statements",
    "4 · Flows & corridors",
    "5 · Transaction grain",
])


# ==========================================================================
# Tab 1 - Reconciliation: the headline mismatch position
# ==========================================================================

with tab1:
    failing = plaus_f[plaus_f["status"] != "plausible"]
    fail_rate = len(failing) / len(plaus_f) if len(plaus_f) else 0
    total_flow = panel_f["total_flow_zar"].sum()
    revenue = pack_f.loc[pack_f["line_item"] == "Total revenue", "value"].sum()

    if comparison is not None:
        material = int(comparison["material"].sum())
        coverage = comparison["revenue_coverage"].median()
        cards = [
            card("Lines compared", f"{len(comparison):,}", f"{comparison['entity_id'].nunique()} entities"),
            card("Material discrepancies", f"{material:,}",
                 f"{material / len(comparison):.0%} of compared lines",
                 "down" if material / len(comparison) > 0.3 else "up"),
            card("Median revenue coverage", f"{coverage:,.0f}x", "reported ÷ proxy"),
            card("Proxy revenue (period)", zar(revenue)),
            card("Ratio checks failed", f"{len(failing)} / {len(plaus_f)}",
                 f"{fail_rate:.0%} outside plausible range", "down" if fail_rate > 0.25 else "up"),
        ]
    else:
        cards = [
            card("Entities in scope", f"{len(scope)}", f"{len(sectors)} sectors"),
            card("Structural breaks", f"{len(failing)}",
                 f"{fail_rate:.0%} of ratio checks fail", "down" if fail_rate > 0.25 else "up"),
            card("Proxy revenue (period)", zar(revenue)),
            card("Total flow (period)", zar(total_flow)),
            card("Reported financials", "Not loaded", "upload to reconcile", "flat"),
        ]
    card_row(cards)
    st.write("")

    if comparison is None:
        st.info(
            "No reported financials loaded, so the reconciliation runs in **structural mode**: "
            "it checks whether the proxy statements could ever match a real annual report. "
            f"Fill `{REPORTED_TEMPLATE.relative_to(Path.cwd()) if REPORTED_TEMPLATE.is_relative_to(Path.cwd()) else REPORTED_TEMPLATE.name}` "
            "and upload it in the sidebar to switch to full line-by-line variance."
        )

    left, right = st.columns([3, 2])

    with left:
        if comparison is not None:
            section("Material discrepancies by statement line",
                    "Lines whose implied coverage deviates from the entity's revenue coverage.")
            by_line = (
                comparison.groupby("statement_line")["material"]
                .agg(["sum", "count"]).reset_index()
                .sort_values("sum", ascending=True)
            )
            by_line["consistent"] = by_line["count"] - by_line["sum"]
            fig = go.Figure()
            fig.add_bar(y=by_line["statement_line"], x=by_line["sum"], orientation="h",
                        name="Material", marker_color=NEG)
            fig.add_bar(y=by_line["statement_line"], x=by_line["consistent"], orientation="h",
                        name="Consistent", marker_color=ACCENT)
            fig.update_layout(barmode="stack", xaxis_title="Entity-year observations")
            st.plotly_chart(style(fig, 420), width="stretch")
        else:
            section("Where the proxy cannot match a real annual report",
                    "Derived ratios against plausible ranges for large listed corporates.")
            by_metric = (
                plaus_f.assign(fails=plaus_f["status"] != "plausible")
                .groupby("line_item")["fails"].agg(["sum", "count"]).reset_index()
                .sort_values("sum")
            )
            by_metric["ok"] = by_metric["count"] - by_metric["sum"]
            fig = go.Figure()
            fig.add_bar(y=by_metric["line_item"], x=by_metric["sum"], orientation="h",
                        name="Outside range", marker_color=NEG)
            fig.add_bar(y=by_metric["line_item"], x=by_metric["ok"], orientation="h",
                        name="Plausible", marker_color=ACCENT)
            fig.update_layout(barmode="stack", xaxis_title="Entity-year observations")
            st.plotly_chart(style(fig, 420), width="stretch")

    with right:
        section("Reconciliation status", "Share of checks that would survive an audit.")
        if comparison is not None:
            ok = len(comparison) - int(comparison["material"].sum())
            values, labels = [ok, int(comparison["material"].sum())], ["Consistent", "Material"]
        else:
            values = [len(plaus_f) - len(failing), len(failing)]
            labels = ["Plausible", "Structural break"]
        fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.68,
                               marker=dict(colors=[ACCENT, NEG]), textinfo="none"))
        share = values[1] / max(sum(values), 1)
        fig.update_layout(annotations=[dict(
            text=f"<b>{share:.0%}</b><br><span style='font-size:11px;color:{MUTED}'>break</span>",
            showarrow=False, font=dict(size=26, color=TEXT))])
        st.plotly_chart(style(fig, 420), width="stretch")

    st.divider()
    section("Reconciliation by entity",
            "Every entity-year, ranked by how much of the statement pack fails to reconcile.")

    if comparison is not None:
        grid = comparison.pivot_table(index="entity_name", columns="statement_line",
                                      values="material", aggfunc="mean")
        title, colorbar = "Share of years flagged material", "Material"
    else:
        grid = (
            plaus_f.assign(fails=(plaus_f["status"] != "plausible").astype(float))
            .pivot_table(index="entity_name", columns="line_item", values="fails", aggfunc="mean")
        )
        title, colorbar = "Share of years outside plausible range", "Break rate"

    fig = go.Figure(go.Heatmap(
        z=grid.values, x=grid.columns, y=grid.index,
        colorscale=[[0, "#1c1c24"], [0.5, ACCENT], [1, NEG]],
        colorbar=dict(title=colorbar, outlinewidth=0), zmin=0, zmax=1,
    ))
    fig.update_layout(title=title)
    st.plotly_chart(style(fig, 520, legend=False), width="stretch")


# ==========================================================================
# Tab 2 - Discrepancies: which line, which entity, which year
# ==========================================================================

with tab2:
    if comparison is not None:
        worst = comparison[comparison["material"]].copy()
        card_row([
            card("Material lines", f"{len(worst):,}", f"of {len(comparison):,} compared",
                 "down" if len(worst) else "up"),
            card("Under-represented", f"{(worst['direction'] == 'under-represented in data').sum():,}",
                 "data understates the accounts"),
            card("Over-represented", f"{(worst['direction'] == 'over-represented in data').sum():,}",
                 "data overstates the accounts"),
            card("Widest deviation",
                 f"{worst['coverage_deviation'].abs().max():,.1f}x" if len(worst) else "-",
                 "vs revenue coverage"),
        ])
        st.write("")

        section("Coverage factor by entity",
                "How much larger the reported accounts are than the flows we can see. "
                "A flat profile across lines means the pipeline is internally consistent.")
        cov = (
            comparison.dropna(subset=["implied_coverage"])
            .groupby(["entity_name", "statement_line"])["implied_coverage"].median().reset_index()
        )
        fig = go.Figure()
        for i, (line, grp) in enumerate(cov.groupby("statement_line")):
            fig.add_trace(go.Scatter(
                x=grp["entity_name"], y=grp["implied_coverage"], mode="markers",
                name=line, marker=dict(size=11, color=SEQUENCE[i % len(SEQUENCE)], opacity=0.85),
            ))
        fig.update_yaxes(type="log", title="Implied coverage (reported ÷ proxy, log)")
        st.plotly_chart(style(fig, 420), width="stretch")

        section("Variance detail", "Proxy scaled to the entity's revenue coverage, then compared.")
        show = comparison[[
            "entity_id", "entity_name", "fiscal_year", "statement_line", "proxy_value",
            "scaled_proxy_value", "reported_value", "variance", "variance_pct",
            "coverage_deviation", "direction",
        ]].copy()
        for col in ["proxy_value", "scaled_proxy_value", "reported_value", "variance"]:
            show[col] = show[col].map(zar)
        for col in ["variance_pct", "coverage_deviation"]:
            show[col] = show[col].map(lambda v: "-" if pd.isna(v) else f"{v:+.1%}")
        table(show, height=420)
        st.download_button("Download comparison CSV", comparison.to_csv(index=False),
                           "comparison_to_reported.csv", "text/csv")

    else:
        section("Ratio checks against plausible reporting ranges",
                "Each dot is one entity-year. Bands are typical ranges for large listed "
                "corporates - anything outside cannot reconcile to a real annual report.")

        metric = st.selectbox("Metric", list(PLAUSIBLE_RATIO_RANGES))
        sub = plaus_f[plaus_f["line_item"] == metric]
        low, high = PLAUSIBLE_RATIO_RANGES[metric]

        fig = go.Figure()
        fig.add_hrect(y0=low, y1=high, fillcolor=ACCENT, opacity=0.13, line_width=0,
                      annotation_text="plausible range", annotation_position="top left")
        for year, grp in sub.groupby("fiscal_year"):
            fig.add_trace(go.Scatter(
                x=grp["entity_name"], y=grp["value"], mode="markers", name=year,
                marker=dict(size=12, opacity=0.9),
            ))
        fig.update_layout(yaxis_title=metric)
        st.plotly_chart(style(fig, 430), width="stretch")

        breaks = plaus_f[plaus_f["status"] != "plausible"]
        section("Structural breaks", f"{len(breaks)} of {len(plaus_f)} observations outside range.")
        summary = (
            breaks.groupby(["line_item", "status"]).size().reset_index(name="observations")
            .sort_values("observations", ascending=False)
        )
        col_a, col_b = st.columns([1, 2])
        with col_a:
            table(summary)
        with col_b:
            show = breaks[["entity_id", "entity_name", "sector", "fiscal_year",
                           "line_item", "value", "low", "high", "status"]].copy()
            show["value"] = show["value"].map(lambda v: f"{v:.3f}")
            table(show, height=320)

        st.markdown(
            f'<div class="muted">To move from structural checks to line-by-line variance, key the '
            f'real figures into the template ({len(COMPARABLE_LINES)} metrics per entity-year) '
            f'and upload it in the sidebar.</div>',
            unsafe_allow_html=True,
        )


# ==========================================================================
# Tab 3 - Statements: the pseudo consolidated pack
# ==========================================================================

with tab3:
    name = ENTITY_LABEL[focus]
    entity_pack = pack[(pack["entity_id"] == focus) & pack["fiscal_year"].isin(years)]

    def line(item: str, year: str | None = None) -> float:
        sub = entity_pack[entity_pack["line_item"] == item]
        if year:
            sub = sub[sub["fiscal_year"] == year]
        return sub["value"].sum() if not sub.empty else np.nan

    latest = years[-1]
    prior = years[-2] if len(years) > 1 else None
    rev_now, rev_prior = line("Total revenue", latest), line("Total revenue", prior) if prior else np.nan
    growth = (rev_now / rev_prior - 1) if prior and rev_prior else np.nan

    ebitda_margin = entity_pack.loc[
        (entity_pack["line_item"] == "EBITDA margin proxy") & (entity_pack["fiscal_year"] == latest),
        "value",
    ].mean()

    card_row([
        card(f"{name} · revenue {latest}", zar(rev_now),
             f"{growth:+.1%} vs {prior}" if prior else None,
             "up" if growth and growth > 0 else "down"),
        card("EBITDA proxy", zar(line("EBITDA proxy", latest)),
             f"{ebitda_margin:.1%} margin" if pd.notna(ebitda_margin) else None),
        card("Operating cash flow", zar(line("Net cash from operating activities", latest))),
        card("Trade finance notional", zar(line("Total trade finance notional", latest))),
        card("Open exposure", zar(line("Open exposure at period end", latest))),
    ])
    st.write("")

    left, right = st.columns([2, 3])

    with left:
        section("Revenue composition", "Domestic collections versus cross-border receipts.")
        comp_lines = ["Revenue - domestic collections", "Revenue - cross-border trade receipts",
                      "Other operating income"]
        comp = entity_pack[entity_pack["line_item"].isin(comp_lines)]
        fig = go.Figure()
        for i, item in enumerate(comp_lines):
            grp = comp[comp["line_item"] == item].sort_values("fiscal_year")
            fig.add_bar(x=grp["fiscal_year"], y=grp["value"], name=item.replace("Revenue - ", ""),
                        marker_color=SEQUENCE[i])
        fig.update_layout(barmode="stack", yaxis_title="ZAR")
        st.plotly_chart(style(fig, 340), width="stretch")

    with right:
        section("Margin and conversion trend", "Cash-basis, so read as operating cash margin.")
        trend_lines = ["Gross margin proxy", "EBITDA margin proxy", "Operating cash conversion"]
        fig = go.Figure()
        for i, item in enumerate(trend_lines):
            grp = entity_pack[entity_pack["line_item"] == item].sort_values("fiscal_year")
            fig.add_trace(go.Scatter(x=grp["fiscal_year"], y=grp["value"], mode="lines+markers",
                                     name=item, line=dict(width=3, color=SEQUENCE[i]),
                                     marker=dict(size=9)))
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="")
        st.plotly_chart(style(fig, 340), width="stretch")

    st.divider()
    statement_names = {
        "income_statement": "Income statement (cash-basis proxy)",
        "cash_flow": "Cash flow statement (direct method)",
        "contingent_note": "Contingent note - trade finance notional",
    }
    choice = st.radio("Statement", list(statement_names), horizontal=True,
                      format_func=statement_names.get, label_visibility="collapsed")

    sub = entity_pack[entity_pack["statement"] == choice].sort_values("line_order")
    wide = sub.pivot_table(index=["line_order", "line_item"], columns="fiscal_year",
                           values="value").sort_index().droplevel("line_order")
    ratio_rows = set(sub.loc[sub["is_ratio"], "line_item"])
    display = pd.DataFrame(
        {
            year: [
                "-" if pd.isna(v) else (f"{v:,.2f}" if row in ratio_rows else f"{v / 1e6:,.1f}")
                for row, v in zip(wide.index, wide[year])
            ]
            for year in wide.columns
        },
        index=wide.index,
    )
    display.insert(0, "Line item", display.index)
    section(statement_names[choice], f"{name} · ZAR millions unless stated")
    table(display.reset_index(drop=True), height=min(60 + 35 * len(display), 780))


# ==========================================================================
# Tab 4 - Flows and corridors
# ==========================================================================

with tab4:
    corridor = data["corridor_mix"]
    corridor = corridor[corridor["entity_id"].isin(scope) & corridor["fiscal_year"].isin(years)]
    countries = data["country_flows"]
    countries = countries[countries["entity_id"].isin(scope) & countries["fiscal_year"].isin(years)]
    trade = data["trade_profile"]
    trade = trade[trade["entity_id"].isin(scope) & trade["fiscal_year"].isin(years)]

    fx_total = corridor["value_zar"].sum()
    interco = corridor.loc[corridor["corridor_type"] == "intercompany", "value_zar"].sum()
    card_row([
        card("Cross-border value", zar(fx_total), f"{corridor['txn_count'].sum():,} payments"),
        card("Intragroup share", f"{interco / fx_total:.0%}" if fx_total else "-",
             "excluded from revenue"),
        card("Trade finance notional", zar(trade["value_zar"].sum()),
             f"{trade['instrument_count'].sum():,} instruments"),
        card("Open exposure", zar(trade.loc[trade["status"].isin(["active", "issued"]), "value_zar"].sum())),
        card("Counterparty countries", f"{countries['counterparty_country'].nunique()}"),
    ])
    st.write("")

    left, right = st.columns(2)

    with left:
        section("Corridor mix by year", "Trade flows are revenue; intragroup is treasury.")
        mix = corridor.groupby(["fiscal_year", "corridor_type"])["value_zar"].sum().reset_index()
        fig = go.Figure()
        for i, (corr, grp) in enumerate(mix.groupby("corridor_type")):
            fig.add_bar(x=grp["fiscal_year"], y=grp["value_zar"], name=corr, marker_color=SEQUENCE[i])
        fig.update_layout(barmode="stack", yaxis_title="ZAR")
        st.plotly_chart(style(fig, 340), width="stretch")

    with right:
        section("Currency exposure", "Gross flow by pair, both directions.")
        ccy = corridor.groupby("currency_pair")["value_zar"].sum().sort_values()
        fig = go.Figure(go.Bar(x=ccy.values, y=ccy.index, orientation="h",
                               marker_color=ACCENT))
        fig.update_layout(xaxis_title="ZAR")
        st.plotly_chart(style(fig, 340, legend=False), width="stretch")

    left, right = st.columns(2)

    with left:
        section("Top counterparty countries", "Cross-border payments and trade instruments.")
        top = (
            countries.groupby("counterparty_country")["value_zar"].sum()
            .sort_values(ascending=False).head(15).sort_values()
        )
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=ACCENT_SOFT))
        fig.update_layout(xaxis_title="ZAR")
        st.plotly_chart(style(fig, 420, legend=False), width="stretch")

    with right:
        section("Trade finance exposure", "Notional by instrument and status.")
        grid = trade.pivot_table(index="instrument_type", columns="status",
                                 values="value_zar", aggfunc="sum", fill_value=0)
        fig = go.Figure()
        for i, status in enumerate(grid.columns):
            fig.add_bar(x=grid.index, y=grid[status], name=status, marker_color=SEQUENCE[i])
        fig.update_layout(barmode="stack", yaxis_title="ZAR")
        st.plotly_chart(style(fig, 420), width="stretch")

    section("Net foreign currency position by entity",
            "Inflow less outflow. Persistent negatives mean structural import funding.")
    net = (
        panel_f.groupby("entity_name")[["xb_inflow_zar", "xb_outflow_zar"]].sum()
        .assign(net=lambda d: d["xb_inflow_zar"] - d["xb_outflow_zar"])
        .sort_values("net")
    )
    fig = go.Figure(go.Bar(
        x=net["net"], y=net.index, orientation="h",
        marker_color=[NEG if v < 0 else POS for v in net["net"]],
    ))
    fig.update_layout(xaxis_title="Net FX flow (ZAR)")
    st.plotly_chart(style(fig, 520, legend=False), width="stretch")


# ==========================================================================
# Tab 5 - Transaction grain and data quality
# ==========================================================================

with tab5:
    monthly = data["monthly_volume"]
    monthly = monthly[monthly["entity_id"].isin(scope) & monthly["fiscal_year"].isin(years)]
    channels = data["channel_mix"]
    channels = channels[channels["entity_id"].isin(scope) & channels["fiscal_year"].isin(years)]
    dist = data["value_distribution"]
    dist = dist[dist["entity_id"].isin(scope)]

    card_row([
        card("Transactions in scope", f"{monthly['txn_count'].sum():,}"),
        card("Duplicate ids removed", f"{quality['duplicate_ids'].sum():,}",
             "flagged, not dropped", "flat"),
        card("Memo completeness",
             f"{100 - quality['memo_missing_pct'].mean():.1f}%", "unusable as a feature", "down"),
        card("Payroll transactions",
             f"{int(channels.loc[channels['leg_type'] == 'payroll', 'txn_count'].sum()):,}",
             "far too few for real salary runs", "down"),
        card("Tax transactions",
             f"{int(channels.loc[channels['leg_type'] == 'tax', 'txn_count'].sum()):,}",
             "explains the low effective rate", "down"),
    ])
    st.write("")

    section("Monthly volume by source", "Seasonality and any collection gaps in the pipeline.")
    vol = monthly.groupby(["calendar_month", "dataset"])["txn_count"].sum().reset_index()
    fig = go.Figure()
    for i, (name_, grp) in enumerate(vol.groupby("dataset")):
        grp = grp.sort_values("calendar_month")
        fig.add_trace(go.Scatter(x=grp["calendar_month"], y=grp["txn_count"], mode="lines",
                                 name=name_.replace("_", " "), line=dict(width=2.5, color=SEQUENCE[i])))
    fig.update_layout(yaxis_title="Transactions", yaxis_type="log")
    st.plotly_chart(style(fig, 360), width="stretch")

    left, right = st.columns(2)

    with left:
        section("Leg type by channel", "Value routed through each payment rail.")
        grid = channels.pivot_table(index="leg_type", columns="channel",
                                    values="value_zar", aggfunc="sum", fill_value=0)
        fig = go.Figure(go.Heatmap(z=grid.values, x=grid.columns, y=grid.index,
                                   colorscale=[[0, "#1c1c24"], [1, ACCENT]],
                                   colorbar=dict(outlinewidth=0)))
        st.plotly_chart(style(fig, 360, legend=False), width="stretch")

    with right:
        section("Transaction size distribution", "Log10 rand value, by source.")
        fig = go.Figure()
        for i, (name_, grp) in enumerate(dist.groupby("dataset")):
            agg = grp.groupby("log10_value")["txn_count"].sum().reset_index()
            fig.add_trace(go.Scatter(x=agg["log10_value"], y=agg["txn_count"], mode="lines",
                                     name=name_.replace("_", " "), fill="tozeroy",
                                     line=dict(width=2, color=SEQUENCE[i])))
        fig.update_layout(xaxis_title="log10 value (ZAR)", yaxis_title="Transactions")
        st.plotly_chart(style(fig, 360), width="stretch")

    st.divider()
    section("Source data quality", "Cleaning applied before any aggregation.")
    dq = quality.copy()
    dq["total_value_zar"] = dq["total_value_zar"].map(zar)
    dq["rows"] = dq["rows"].map(lambda v: f"{v:,}")
    dq["duplicate_ids"] = dq["duplicate_ids"].map(lambda v: f"{v:,}")
    table(dq)

    st.markdown(
        '<div class="muted">Exact duplicate rows dropped; duplicate ids flagged rather than removed. '
        'Currency casing normalised, missing counterparty countries filled as Unknown. '
        'Trade finance is notional and never enters cash or revenue.</div>',
        unsafe_allow_html=True,
    )
