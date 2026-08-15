""" Syn Bank entity analytics. Run with:

    uv run streamlit run entity_app.py

Tabs, left to right:

    1 Reported vs Computed   the financials line comparison
    2 Sector                 sector split, and reported against computed per sector
    3 Entity Analysis        incomes, payments and reference types per entity
    4 Geography              counterparty countries, income against payment
    5 Opportunity            missed wallet and the revenue it would carry

Each tab opens with a generated read of its own figures - see `insight()`.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from build_artifacts import ARTIFACTS_, build

TABLES = ["summary", "reference", "reference_counts", "comparison", "geo", "projection"]

st.set_page_config(page_title="Syn Bank Entity Analysis", layout="wide")

ID_COLS = ["entity_id", "entity_name", "sector"]
TOTAL = "All Entities (total)"

# validated categorical palette - worst adjacent CVD dE 9.1 on a white surface
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, MUTED, GRID = "#ffffff", "#898781", "#e1e0d9"
# status palette - income green against payment red
POS, NEG = "#0ca30c", "#d03b3b"
GREY, AMBER = "rgba(137,135,129,0.16)", "rgba(250,178,25,0.22)"

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

st.markdown("""
<style>
.insight { border-left: 3px solid #2a78d6; background: rgba(42,120,214,0.07);
           padding: 0.85rem 1.1rem; border-radius: 6px; margin: 0.2rem 0 1.2rem 0; }
.insight-label { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
                 opacity: 0.6; margin-bottom: 0.35rem; }
.insight-body { line-height: 1.5; }
.insight-action { margin-top: 0.6rem; font-size: 0.93em; opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_data
def build_tables(names: tuple[str, ...]):
    """Precomputed artifacts where they exist, otherwise straight from the CSVs.

    Deployments ship only the 71 KB of parquet - the 409 MB of source data never
    leaves the machine that ran `build_artifacts.py`.

    `names` is passed in rather than read off the module so it lands in the cache
    key: st.cache_data hashes the arguments and the function body, not the globals
    the body closes over. Adding a table while it was a global left the previous
    deploy's shorter tuple cached against unchanged code.
    """
    if all((ARTIFACTS_ / f"{name}.parquet").exists() for name in names):
        return tuple(pd.read_parquet(ARTIFACTS_ / f"{name}.parquet") for name in names)

    tables = build()
    return tuple(tables[name] for name in names)

summary_df, reference_df, reference_count_df, comparison_df, geo_df, projection_df = build_tables(tuple(TABLES))

REFERENCE_TYPES = [c for c in reference_df.columns if c not in ID_COLS]
SECTORS = sorted(summary_df["sector"].dropna().unique())

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def zar(value):
    if abs(value) >= 1e12:
        return f"R {value / 1e12:,.2f}tn"
    if abs(value) >= 1e9:
        return f"R {value / 1e9:,.2f}bn"
    if abs(value) >= 1e6:
        return f"R {value / 1e6:,.1f}m"
    return f"R {value:,.0f}"

def tidy(name):
    return str(name).replace("_", " ").title()

def entity_labels(frame):
    return (frame["entity_id"] + " - " + frame["entity_name"]).drop_duplicates().sort_values()

def insight(body, action):
    """The generated read at the top of each tab.

    Rule-based, computed from whatever the tab's slicers currently resolve to -
    so it moves with the filters rather than describing a fixed snapshot.
    """
    st.markdown(
        f'<div class="insight"><div class="insight-label">Analysis</div>'
        f'<div class="insight-body">{body}</div>'
        f'<div class="insight-action"><b>Suggested move:</b> {action}</div></div>',
        unsafe_allow_html=True,
    )

def frame_style(fig, height=420, y_title=None):
    fig.update_layout(height=height, yaxis_title=y_title,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#c3c2b7", color=MUTED)
    return fig

def fold_to_eight(df, dim, value_col):
    """Categorical colour caps at 8 slots - keep the 7 largest, fold the rest into Other."""
    d = df.groupby(dim, as_index=False)[value_col].sum()
    d = d.sort_values(value_col, ascending=False, key=abs).reset_index(drop=True)
    if len(d) > 8:
        other = pd.DataFrame({dim: ["Other"], value_col: [d[value_col][7:].sum()]})
        d = pd.concat([d.head(7), other], ignore_index=True)
    return d

def donut(labels, values, height=400):
    fig = go.Figure(go.Pie(labels=labels, values=values, sort=False, hole=0.55,
                           marker=dict(colors=CATEGORICAL, line=dict(color=SURFACE, width=2)),
                           textinfo="label+percent", textposition="outside"))
    fig.update_layout(height=height, showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
    return fig

# --------------------------------------------------------------------------
# Sidebar - drives the Entity Analysis tab only
# --------------------------------------------------------------------------

st.sidebar.markdown("**Entity Analysis filters**")
labels = entity_labels(summary_df).tolist()
choice = st.sidebar.selectbox("Entity", [TOTAL] + labels)
selected_id = None if choice == TOTAL else choice.split(" - ", 1)[0]

split = st.sidebar.radio("Split by", ["Sector", "Entity"], horizontal=True)
measure = st.sidebar.radio("Measure", ["Amount", "Count"], horizontal=True)

dim = "sector" if split == "Sector" else "entity_name"
in_col, out_col = ("incomes", "payments") if measure == "Amount" else ("num_incomes", "num_payments")

def view(df):
    """One entity's row, or every entity with a summed TOTAL row appended."""
    if selected_id is not None:
        return df[df["entity_id"] == selected_id]

    totals = df.drop(columns=ID_COLS).sum().to_frame().T
    totals["entity_id"] = ""
    totals["entity_name"] = "TOTAL"
    totals["sector"] = ""

    return pd.concat([df, totals[df.columns]], ignore_index=True)

summary_view = view(summary_df)
reference_view = view(reference_df)
reference_count_view = view(reference_count_df)

def share_pie(df_rows, dim, value_col, title):
    d = fold_to_eight(df_rows[df_rows["entity_name"] != "TOTAL"], dim, value_col)
    fig = donut(d[dim], d[value_col], 380)
    fig.update_layout(title=title)
    return fig

def reference_bar(df_wide, dim, y_title):
    rows = df_wide[df_wide["entity_name"] != "TOTAL"]
    long = rows.melt(id_vars=ID_COLS, var_name="reference_type", value_name="value")

    # collapse to the chosen split, then fold past the 8-slot colour cap
    long = long.groupby(["reference_type", dim], as_index=False)["value"].sum()
    ranked = (long.assign(mag=long["value"].abs())
                  .groupby(dim)["mag"].sum().nlargest(7).index)
    long[dim] = long[dim].where(long[dim].isin(ranked), "Other")
    long = long.groupby(["reference_type", dim], as_index=False)["value"].sum()
    order = list(ranked) + (["Other"] if (long[dim] == "Other").any() else [])

    fig = px.bar(long, x="reference_type", y="value", color=dim,
                 category_orders={dim: order},
                 color_discrete_sequence=CATEGORICAL, barmode="relative")
    fig.update_traces(marker_line_width=2, marker_line_color=SURFACE)
    fig.update_layout(bargap=0.25, xaxis_title=None, legend_title_text=split)
    return frame_style(fig, 460, y_title)

def highlight(row):
    """Grey where nothing was computed, amber where the ratio implies a reporting scale error."""
    if pd.isna(row["summation_value"]):
        return [f"background-color: {GREY}"] * len(row)
    if abs(row["pct_of_reported"]) > 50:
        return [f"background-color: {AMBER}"] * len(row)
    return [""] * len(row)

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tabs = st.tabs(["1 · Reported vs Computed", "2 · Sector", "3 · Entity Analysis",
                "4 · Geography", "5 · Opportunity", "6 · Future Projection"])

# 1 - REPORTED VS COMPUTED --------------------------------------------------
with tabs[0]:
    st.title("Reported vs Computed")

    scope = st.radio("View", ["All", "By sector", "By entity"], horizontal=True)

    if scope == "By sector":
        cmp_sector = st.selectbox("Sector", SECTORS, key="cmp_sector")
        table = comparison_df[comparison_df["sector"] == cmp_sector]
        scope_label = tidy(cmp_sector)
    elif scope == "By entity":
        cmp_pick = st.selectbox("Entity", entity_labels(comparison_df), key="cmp_entity")
        table = comparison_df[comparison_df["entity_id"] == cmp_pick.split(" - ", 1)[0]]
        scope_label = cmp_pick
    else:
        table = comparison_df
        scope_label = "the full book"

    matched = table["summation_value"].notna()
    reliable = table[matched & (table["pct_of_reported"].abs() <= 50)]
    by_line = table.groupby("line_item", as_index=False).agg(
        reported_lines=("reported_value", "size"),
        matched=("summation_value", "count"),
        median_pct=("pct_of_reported", "median"))

    if reliable.empty:
        insight(f"No reliable comparison lines resolve for {scope_label}.",
                "Widen the selection, or extend extraction coverage to these entity-years.")
    else:
        med = reliable["pct_of_reported"].median()
        ranked = by_line.dropna(subset=["median_pct"]).sort_values("median_pct")
        thinnest, thickest = ranked.iloc[0], ranked.iloc[-1]
        flagged = int((table["pct_of_reported"].abs() > 50).sum())
        insight(
            f"Across {len(table)} reported lines for {scope_label}, Syn Bank can evidence a median "
            f"<b>{med:.2f}%</b> of what these clients publish — {int(matched.sum())} of {len(table)} lines "
            f"reconcile to transaction data. Coverage is deepest on <b>{thickest['line_item']}</b> "
            f"({thickest['median_pct']:.2f}%) and shallowest on <b>{thinnest['line_item']}</b> "
            f"({thinnest['median_pct']:.2f}%)"
            + (f", with {flagged} line{'s' if flagged > 1 else ''} flagged as an extraction scale error "
               f"rather than a genuine match." if flagged else "."),
            f"Lead with <b>{thinnest['line_item']}</b> in client conversations — it is the flow these "
            f"companies report but bank almost entirely elsewhere, so it is the least contested to win.")

    b1, b2, b3 = st.columns(3)
    b1.metric("Reported lines", len(table))
    b2.metric("Matched", f"{int(matched.sum())} of {len(table)}")
    b3.metric("Median wallet share",
              f"{reliable['pct_of_reported'].median():.2f}%" if not reliable.empty else "-")

    styled = (table.drop(columns=["sector"])
                   .sort_values(["entity_id", "line_item", "fiscal_year"])
                   .style.apply(highlight, axis=1)
                   .format({c: "{:,.0f}" for c in ("reported_value", "summation_value", "difference")}
                           | {"pct_of_reported": "{:,.2f}%"}))

    st.dataframe(styled, hide_index=True, width="stretch", height=520)
    st.caption("Grey - no computed equivalent. Amber - computed exceeds 50% of reported, "
               "which in this data means a scale error in the extracted figure rather than a real match.")

    st.subheader("By line item")
    st.dataframe(by_line.style.format({"median_pct": "{:,.3f}%"}), hide_index=True, width="stretch")

# 2 - SECTOR ----------------------------------------------------------------
with tabs[1]:
    st.title("Per Sector Split & Discrepancy")

    sector_totals = summary_df.groupby("sector", as_index=False).agg(
        incomes=("incomes", "sum"), payments=("payments", "sum"),
        entities=("entity_id", "nunique"), transactions=("num_transactions", "sum"))
    sector_totals["bank_flow"] = sector_totals["incomes"] + sector_totals["payments"]

    matched_only = comparison_df[comparison_df["summation_value"].notna()
                                 & (comparison_df["pct_of_reported"].abs() <= 50)]
    sector_gap = matched_only.groupby("sector", as_index=False).agg(
        reported=("reported_value", "sum"), computed=("summation_value", "sum"))
    sector_gap["gap"] = sector_gap["reported"] - sector_gap["computed"]
    sector_gap["wallet_share"] = sector_gap["computed"] / sector_gap["reported"] * 100

    if sector_gap.empty:
        insight("No sector resolves a reliable reported-against-computed comparison.",
                "Extend extraction coverage before sizing sector opportunity.")
    else:
        best = sector_gap.loc[sector_gap["wallet_share"].idxmax()]
        widest = sector_gap.loc[sector_gap["gap"].idxmax()]
        biggest_flow = sector_totals.loc[sector_totals["bank_flow"].idxmax()]
        insight(
            f"<b>{tidy(best['sector'])}</b> is where Syn Bank holds the deepest position, evidencing "
            f"{best['wallet_share']:.2f}% of reported value, while <b>{tidy(widest['sector'])}</b> carries the "
            f"largest absolute shortfall at {zar(widest['gap'])} unbanked against only "
            f"{widest['wallet_share']:.2f}% currently held. By transaction flow the book leans on "
            f"<b>{tidy(biggest_flow['sector'])}</b> ({zar(biggest_flow['bank_flow'])} across "
            f"{int(biggest_flow['entities'])} entities).",
            f"Push origination into <b>{tidy(widest['sector'])}</b> — the gap there is the largest single "
            f"pool of flow the bank already has relationships to reach but does not yet carry.")

    s1, s2 = st.columns(2)
    with s1:
        st.subheader("All sector split")
        st.caption("Share of Syn Bank flow by sector.")
        d = sector_totals.sort_values("bank_flow", ascending=False)
        st.plotly_chart(donut(d["sector"], d["bank_flow"]), width="stretch")
    with s2:
        st.subheader("Per sector discrepancy")
        st.caption("Share of the total gap to reported financials that each sector carries.")
        d = sector_gap[sector_gap["gap"] > 0].sort_values("gap", ascending=False)
        st.plotly_chart(donut(d["sector"], d["gap"]), width="stretch")

    st.divider()
    st.subheader("Reported against calculated")
    st.caption("Log scale - the bank sees around 1% of reported financials, so the two "
               "series are orders of magnitude apart and will not read side by side on a linear axis.")

    d = sector_gap.sort_values("reported", ascending=False)
    fig = go.Figure()
    fig.add_bar(x=d["sector"], y=d["reported"], name="Reported", marker_color=CATEGORICAL[0],
                marker_line=dict(width=2, color=SURFACE))
    fig.add_bar(x=d["sector"], y=d["computed"], name="Calculated", marker_color=CATEGORICAL[1],
                marker_line=dict(width=2, color=SURFACE))
    fig.update_layout(barmode="group", bargap=0.3, xaxis_title=None)
    fig.update_yaxes(type="log")
    st.plotly_chart(frame_style(fig, 440, "ZAR (log)"), width="stretch")

    st.subheader("Wallet share by sector")
    d = sector_gap.sort_values("wallet_share", ascending=False)
    fig = go.Figure(go.Bar(x=d["sector"], y=d["wallet_share"], marker_color=CATEGORICAL[2],
                           marker_line=dict(width=2, color=SURFACE),
                           hovertemplate="%{x}<br>%{y:.2f}% of reported<extra></extra>"))
    fig.update_layout(bargap=0.3, xaxis_title=None)
    st.plotly_chart(frame_style(fig, 380, "% of reported"), width="stretch")

    st.dataframe(sector_gap.style.format({"reported": "{:,.0f}", "computed": "{:,.0f}",
                                          "gap": "{:,.0f}", "wallet_share": "{:,.2f}%"}),
                 hide_index=True, width="stretch")

# 3 - ENTITY ANALYSIS -------------------------------------------------------
with tabs[2]:
    st.title(choice)

    live = summary_view[summary_view["entity_name"] != "TOTAL"]
    inc, pay = summary_view["incomes"].iloc[-1], summary_view["payments"].iloc[-1]
    net = inc - pay

    ref_row = reference_view[reference_view["entity_name"] != "TOTAL"][REFERENCE_TYPES].sum()
    top_in = ref_row.idxmax()
    top_out = ref_row.idxmin()
    lead = live.loc[live["incomes"].idxmax()] if len(live) > 1 else None

    insight(
        f"{choice} moves {zar(inc)} in and {zar(pay)} out, a net position of <b>{zar(net)}</b> "
        f"across {int(summary_view['num_transactions'].iloc[-1]):,} transactions. The heaviest inbound "
        f"reference type is <b>{top_in}</b> and the heaviest outbound is <b>{top_out}</b>"
        + (f", with <b>{lead['entity_name']}</b> contributing the most income of any single entity in view."
           if lead is not None else "."),
        f"<b>{top_out}</b> is the largest outbound flow on this view — payment-side mandates there are "
        f"the fastest route to lifting share without competing for the client's collections business.")

    p1, p2 = st.columns(2)
    p1.plotly_chart(share_pie(summary_view, dim, in_col, f"Incomes by {split.lower()}"), width="stretch")
    p2.plotly_chart(share_pie(summary_view, dim, out_col, f"Payments by {split.lower()}"), width="stretch")

    # the total row is last, so iloc[-1] reads the total for TOTAL and the only row otherwise
    c1, c2 = st.columns(2)
    c1.metric("Incomes", zar(inc))
    c2.metric("Payments", zar(pay))

    st.subheader("Summary")
    st.dataframe(summary_view, hide_index=True, width="stretch")

    st.subheader("Reference Types")
    st.dataframe(reference_view, hide_index=True, width="stretch")

    bar_view, bar_title = ((reference_view, "Net ZAR") if measure == "Amount"
                           else (reference_count_view, "Transactions"))
    st.plotly_chart(reference_bar(bar_view, dim, bar_title), width="stretch")

    # -- top products and the case for bundling them ------------------------
    st.divider()
    st.subheader("Top 3 products & group discount")

    counts = reference_count_view[reference_count_view["entity_name"] != "TOTAL"][REFERENCE_TYPES].sum()
    values = reference_view[reference_view["entity_name"] != "TOTAL"][REFERENCE_TYPES].sum()
    total_txns = counts.sum()

    top3 = counts.nlargest(3)
    bundle = pd.DataFrame({
        "reference_type": top3.index,
        "transactions": top3.to_numpy(dtype=float),
        "pct_of_transactions": top3.to_numpy(dtype=float) / total_txns * 100,
        "net_zar": [values[t] for t in top3.index],
    })
    bundle.loc[len(bundle)] = ["COMBINED", top3.sum(), top3.sum() / total_txns * 100,
                               sum(values[t] for t in top3.index)]

    combined_pct = top3.sum() / total_txns * 100
    # a bundle only prices sensibly when the three carry most of the traffic
    qualifies = combined_pct >= 60

    st.dataframe(bundle.style.format({"transactions": "{:,.0f}", "pct_of_transactions": "{:,.1f}%",
                                      "net_zar": "{:,.0f}"}),
                 hide_index=True, width="stretch")

    d1, d2 = st.columns(2)
    d1.metric("Combined share of transactions", f"{combined_pct:.1f}%")
    d2.metric("Group package", "Recommended" if qualifies else "Not yet")

    names = ", ".join(f"<b>{t}</b>" for t in top3.index)
    insight(
        f"{choice} runs {int(total_txns):,} transactions across {len(REFERENCE_TYPES)} reference types. "
        f"The three heaviest — {names} — carry <b>{combined_pct:.1f}%</b> of them "
        f"({int(top3.sum()):,} transactions). Selection is on transaction count rather than value: a "
        f"pricing bundle is billed per instruction, so the volume concentration is what determines "
        f"whether a discount recovers its margin. "
        + (f"At {combined_pct:.0f}% the three cover most of the traffic, so a single negotiated rate "
           f"across them is priceable against predictable volume."
           if qualifies else
           f"At {combined_pct:.0f}% the traffic is too spread for these three alone to anchor a package — "
           f"a bundle here discounts a minority of instructions while leaving the rest at rack rate."),
        (f"Offer a bundled per-instruction rate on {names} in exchange for a volume commitment — "
         f"the concentration makes the discount self-funding, and it locks the flows that are "
         f"cheapest to keep."
         if qualifies else
         f"Widen the bundle beyond three references, or price {top3.index[0]} alone — on this mix a "
         f"three-product package gives away margin without securing enough of the relationship."))

# 4 - GEOGRAPHY -------------------------------------------------------------
with tabs[3]:
    st.title("Geographical Location of Transactions")
    st.caption("Counterparty countries from cross-border payments and trade finance. "
               "Domestic transactional banking carries no country and is not shown.")

    g1, g2 = st.columns([2, 3])
    flow_pick = g1.radio("Flow", ["Both", "Income", "Payment"], horizontal=True)
    geo_sector = g2.selectbox("Sector", ["All sectors"] + SECTORS, key="geo_sector")

    geo = geo_df if geo_sector == "All sectors" else geo_df[geo_df["sector"] == geo_sector]
    if flow_pick != "Both":
        geo = geo[geo["flow"] == flow_pick]

    rolled = (geo.groupby(["counterparty_country", "flow"], as_index=False)
                 .agg(value_zar=("value_zar", "sum"), txn_count=("txn_count", "sum")))
    rolled = rolled[rolled["counterparty_country"].isin(COUNTRY_COORDS)]

    if rolled.empty:
        insight("No cross-border activity resolves for this selection.",
                "Clear the sector filter, or focus on entities with trade finance exposure.")
        st.info("No cross-border activity for this selection.")
    else:
        by_country = rolled.groupby("counterparty_country")["value_zar"].sum().sort_values(ascending=False)
        income = rolled.loc[rolled["flow"] == "Income", "value_zar"].sum()
        payment = rolled.loc[rolled["flow"] == "Payment", "value_zar"].sum()
        stance = "net importer" if payment > income else "net exporter"
        top_country = by_country.index[0]
        concentration = by_country.head(3).sum() / by_country.sum() * 100

        spread = (f"the top three corridors hold {concentration:.0f}% of cross-border value"
                  if concentration >= 50 else
                  f"no corridor dominates, with the top three together holding only {concentration:.0f}%")
        action = (f"Defend <b>{top_country}</b> with corridor-specific FX and trade product — it carries the "
                  f"single largest block of cross-border flow, and losing it would move the whole book."
                  if concentration >= 50 else
                  f"Flow this evenly spread across {len(by_country)} countries is thin everywhere — build depth "
                  f"in <b>{top_country}</b> first, where the bank already has the largest corridor to grow from.")
        insight(
            f"{len(by_country)} counterparty countries carry {zar(rolled['value_zar'].sum())} across "
            f"{int(rolled['txn_count'].sum()):,} transactions, with <b>{top_country}</b> the single largest "
            f"corridor at {zar(by_country.iloc[0])}. The book runs {zar(income)} in against {zar(payment)} out — "
            f"a <b>{stance}</b> position — and {spread}.",
            action)

        m1, m2, m3 = st.columns(3)
        m1.metric("Countries", int(rolled["counterparty_country"].nunique()))
        m2.metric("Value routed", zar(rolled["value_zar"].sum()))
        m3.metric("Transactions", f"{int(rolled['txn_count'].sum()):,}")

        biggest = rolled["value_zar"].max()
        fig = go.Figure()
        for flow_name, colour in (("Income", POS), ("Payment", NEG)):
            part = rolled[rolled["flow"] == flow_name]
            if part.empty:
                continue
            fig.add_trace(go.Scattergeo(
                lat=part["counterparty_country"].map(lambda c: COUNTRY_COORDS[c][0]),
                lon=part["counterparty_country"].map(lambda c: COUNTRY_COORDS[c][1]),
                text=part["counterparty_country"], name=flow_name,
                customdata=np.stack([part["value_zar"], part["txn_count"]], axis=-1),
                hovertemplate="%{text}<br>" + flow_name + " R %{customdata[0]:,.0f}"
                              "<br>%{customdata[1]:,} transactions<extra></extra>",
                marker=dict(size=part["value_zar"], sizemode="area",
                            sizeref=2.0 * biggest / (55.0 ** 2), sizemin=4,
                            color=colour, opacity=0.75,
                            line=dict(width=0.5, color=SURFACE)),
            ))
        fig.update_geos(bgcolor="rgba(0,0,0,0)", showland=True, landcolor="#f0efec",
                        showcountries=True, countrycolor=GRID, showcoastlines=False,
                        showframe=False, showocean=False, projection_type="natural earth")
        fig.update_layout(height=560, paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, width="stretch")

        st.subheader("By country")
        wide = (rolled.pivot_table(index="counterparty_country", columns="flow",
                                   values="value_zar", aggfunc="sum", fill_value=0)
                      .reset_index())
        wide.columns.name = None
        for col in ("Income", "Payment"):
            if col not in wide:
                wide[col] = 0.0
        wide["Net"] = wide["Income"] - wide["Payment"]
        st.dataframe(wide.sort_values("Net", ascending=False)
                         .style.format({c: "{:,.0f}" for c in ("Income", "Payment", "Net")}),
                     hide_index=True, width="stretch")

# 5 - OPPORTUNITY -----------------------------------------------------------
with tabs[4]:
    st.title("Opportunities for Growth")

    o1, o2 = st.columns(2)
    fee_bps = o1.slider("Fee on value routed (bps)", 1, 100, 15)
    fee_per_txn = o2.slider("Fee per transaction (R)", 0, 50, 5)

    # scale errors would swamp the sizing, so drop them on the same rule the
    # comparison tab highlights in amber
    clean = comparison_df[comparison_df["summation_value"].notna()
                          & (comparison_df["pct_of_reported"].abs() <= 50)].copy()
    clean["missed"] = (clean["reported_value"] - clean["summation_value"]).clip(lower=0)

    missed = clean.groupby(ID_COLS, as_index=False).agg(
        missed_amount=("missed", "sum"), reported=("reported_value", "sum"),
        computed=("summation_value", "sum"))
    missed = missed.merge(summary_df[["entity_id", "incomes", "payments", "num_transactions"]],
                          on="entity_id", how="left")

    # what the bank already routes, per transaction, is the ticket it would carry
    missed["avg_ticket"] = ((missed["incomes"] + missed["payments"])
                            / missed["num_transactions"].replace(0, np.nan))
    missed["implied_txns"] = (missed["missed_amount"] / missed["avg_ticket"]).replace([np.inf, -np.inf], np.nan)
    missed["fee_revenue"] = (missed["missed_amount"] * fee_bps / 10_000
                             + missed["implied_txns"].fillna(0) * fee_per_txn)

    seg = missed.groupby("sector", as_index=False).agg(
        missed_amount=("missed_amount", "sum"), fee_revenue=("fee_revenue", "sum"),
        entities=("entity_id", "nunique")).sort_values("missed_amount", ascending=False)

    if missed.empty:
        insight("No entity resolves a reliable gap to size.",
                "Extend extraction coverage before modelling revenue.")
    else:
        top_entity = missed.loc[missed["missed_amount"].idxmax()]
        top_seg = seg.iloc[0]
        share = missed["computed"].sum() / missed["reported"].sum() * 100
        insight(
            f"Against {zar(missed['reported'].sum())} of reported financials Syn Bank currently carries "
            f"{share:.2f}%, leaving <b>{zar(missed['missed_amount'].sum())}</b> of addressable wallet. "
            f"At {fee_bps} bps plus R{fee_per_txn} per transaction that models to "
            f"<b>{zar(missed['fee_revenue'].sum())}</b> in fee revenue, concentrated in "
            f"<b>{tidy(top_seg['sector'])}</b> ({zar(top_seg['missed_amount'])} across "
            f"{int(top_seg['entities'])} entities), with <b>{top_entity['entity_name']}</b> the single "
            f"largest gap at {zar(top_entity['missed_amount'])}.",
            f"Open with <b>{top_entity['entity_name']}</b> — one existing relationship carries "
            f"{top_entity['missed_amount'] / missed['missed_amount'].sum() * 100:.0f}% of the total gap, "
            f"so a single mandate win moves the book further than broad origination.")

    k1, k2, k3 = st.columns(3)
    k1.metric("Total missed wallet", zar(missed["missed_amount"].sum()))
    k2.metric("Implied transactions", f"{missed['implied_txns'].sum():,.0f}")
    k3.metric("Modelled fee revenue", zar(missed["fee_revenue"].sum()))

    st.subheader("Greatest missed segments")
    st.caption("Reported financials the bank does not currently see, by sector.")

    fig = go.Figure(go.Bar(x=seg["sector"], y=seg["missed_amount"], marker_color=CATEGORICAL[1],
                           marker_line=dict(width=2, color=SURFACE),
                           hovertemplate="%{x}<br>R %{y:,.0f} missed<extra></extra>"))
    fig.update_layout(bargap=0.3, xaxis_title=None)
    st.plotly_chart(frame_style(fig, 400, "Missed wallet (ZAR)"), width="stretch")

    st.subheader("Greatest potential for revenue")
    rank_by = st.radio("Rank by", ["Transaction amount", "Transaction volume"], horizontal=True)
    rank_col = "missed_amount" if rank_by == "Transaction amount" else "implied_txns"

    top = missed.sort_values(rank_col, ascending=False).head(10)
    fig = go.Figure(go.Bar(y=top["entity_name"], x=top[rank_col], orientation="h",
                           marker_color=CATEGORICAL[0], marker_line=dict(width=2, color=SURFACE),
                           hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
    fig.update_layout(bargap=0.25, yaxis=dict(autorange="reversed"), xaxis_title=rank_by)
    st.plotly_chart(frame_style(fig, 460), width="stretch")

    st.dataframe(
        missed.sort_values(rank_col, ascending=False)[
            ID_COLS + ["missed_amount", "avg_ticket", "implied_txns", "fee_revenue"]]
        .style.format({"missed_amount": "{:,.0f}", "avg_ticket": "{:,.0f}",
                       "implied_txns": "{:,.0f}", "fee_revenue": "{:,.0f}"}),
        hide_index=True, width="stretch")

# 6 - FUTURE PROJECTION -----------------------------------------------------
with tabs[5]:
    st.title("Future Projection")
    st.caption("Published revenue CAGR compounded forward, with the bank's current share of "
               "each client held flat — so the projection sizes growth already committed to, "
               "not share the bank has yet to win.")

    f1, f2 = st.columns(2)
    horizon = f1.slider("Projection horizon (years)", 1, 10, 5)
    proj_bps = f2.slider("Fee on value routed (bps)", 1, 100, 15, key="proj_bps")

    proj = projection_df.dropna(subset=["base_revenue", "cagr_pct"]).copy()
    dropped = projection_df[projection_df["base_revenue"].isna()]["entity_name"].tolist()

    growth = (1 + proj["cagr_pct"] / 100) ** horizon
    proj["projected_revenue"] = proj["base_revenue"] * growth
    proj["revenue_growth"] = proj["projected_revenue"] - proj["base_revenue"]
    proj["routed_now"] = proj["base_revenue"] * proj["wallet_share_pct"] / 100
    proj["routed_future"] = proj["projected_revenue"] * proj["wallet_share_pct"] / 100
    proj["bank_now"] = proj["routed_now"] * proj_bps / 10_000
    proj["bank_future"] = proj["routed_future"] * proj_bps / 10_000
    proj["bank_uplift"] = proj["bank_future"] - proj["bank_now"]

    fastest = proj.nlargest(3, "cagr_pct")
    top_uplift = proj.loc[proj["bank_uplift"].idxmax()]
    shrinking = proj[proj["cagr_pct"] < 0]

    insight(
        f"Compounded over {horizon} years, the book's clients grow from "
        f"{zar(proj['base_revenue'].sum())} to <b>{zar(proj['projected_revenue'].sum())}</b> of "
        f"published revenue. Holding today's share flat, Syn Bank's modelled fee income rises from "
        f"{zar(proj['bank_now'].sum())} to <b>{zar(proj['bank_future'].sum())}</b> — an uplift of "
        f"{zar(proj['bank_uplift'].sum())} earned without winning a single new mandate. Fastest growers "
        f"are <b>{'</b>, <b>'.join(fastest['entity_name'])}</b> "
        f"({', '.join(f'{c:.1f}%' for c in fastest['cagr_pct'])} CAGR)"
        + (f", while {len(shrinking)} entities are contracting." if len(shrinking) else "."),
        f"<b>{top_uplift['entity_name']}</b> carries the largest uplift at {zar(top_uplift['bank_uplift'])} "
        f"on {top_uplift['cagr_pct']:.1f}% growth — defend that mandate first, since retaining existing "
        f"share there returns more than chasing a comparable gap elsewhere.")

    k1, k2, k3 = st.columns(3)
    k1.metric(f"Client revenue in {horizon}y", zar(proj["projected_revenue"].sum()))
    k2.metric("Bank fee income", zar(proj["bank_future"].sum()))
    k3.metric("Uplift at flat share", zar(proj["bank_uplift"].sum()))

    st.subheader("Fastest growing entities")
    d = proj.sort_values("cagr_pct", ascending=False)
    fig = go.Figure(go.Bar(y=d["entity_name"], x=d["cagr_pct"], orientation="h",
                           marker_color=np.where(d["cagr_pct"] >= 0, POS, NEG),
                           marker_line=dict(width=2, color=SURFACE),
                           hovertemplate="%{y}<br>%{x:.2f}% CAGR<extra></extra>"))
    fig.update_layout(bargap=0.25, yaxis=dict(autorange="reversed"), xaxis_title="Revenue CAGR (%)")
    st.plotly_chart(frame_style(fig, 620), width="stretch")

    st.subheader("Projected bank fee income")
    st.caption("Top eight entities by uplift, compounded year by year at current share.")

    years = list(range(horizon + 1))
    lead = proj.nlargest(8, "bank_uplift")
    fig = go.Figure()
    for colour, (_, row) in zip(CATEGORICAL, lead.iterrows()):
        path = [row["bank_now"] * (1 + row["cagr_pct"] / 100) ** y for y in years]
        fig.add_trace(go.Scatter(x=years, y=path, name=row["entity_name"], mode="lines",
                                 line=dict(color=colour, width=2),
                                 hovertemplate=f"{row['entity_name']}<br>year %{{x}}<br>R %{{y:,.0f}}<extra></extra>"))
    fig.update_layout(xaxis_title="Years from now", legend_title_text="Entity")
    st.plotly_chart(frame_style(fig, 460, "Fee income (ZAR)"), width="stretch")

    st.dataframe(
        proj.sort_values("bank_uplift", ascending=False)[
            ["entity_id", "entity_name", "sector", "cagr_pct", "base_year", "base_revenue",
             "projected_revenue", "wallet_share_pct", "bank_now", "bank_future", "bank_uplift"]]
        .style.format({"cagr_pct": "{:,.2f}%", "wallet_share_pct": "{:,.3f}%",
                       "base_revenue": "{:,.0f}", "projected_revenue": "{:,.0f}",
                       "bank_now": "{:,.0f}", "bank_future": "{:,.0f}", "bank_uplift": "{:,.0f}"}),
        hide_index=True, width="stretch")

    if dropped:
        st.caption(f"Excluded for want of a reported revenue line: {', '.join(dropped)}.")
