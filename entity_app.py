""" Entity Summary Table. Run with:

    uv run streamlit run entity_app.py

"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis_script import (load_data, reference_type_check, combined_df, entity_report,
                             compare_to_results, TRANSACTIONS_, CROSS_BORDER_, TRADE_FINANCE_)

st.set_page_config(page_title="Syn Bank Entity Analysis", layout="wide")

@st.cache_data
def build_tables():
    frames = [load_data(p) for p in (TRANSACTIONS_, CROSS_BORDER_, TRADE_FINANCE_)]
    frames = [reference_type_check(f) for f in frames]
    consolidated = combined_df(*frames)

    # the comparison carries no sector, so bring it across for its own slicer
    sectors = consolidated[["entity_id", "sector"]].drop_duplicates()
    comparison = compare_to_results(consolidated).merge(sectors, on="entity_id", how="left")

    return (*entity_report(consolidated), comparison)

summary_df, reference_df, reference_count_df, comparison_df = build_tables()

ID_COLS = ["entity_id", "entity_name", "sector"]
TOTAL = "All Entities (total)"

# validated categorical palette - worst adjacent CVD dE 9.1 on a white surface
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, MUTED, GRID = "#ffffff", "#898781", "#e1e0d9"

labels = (summary_df["entity_id"] + " - " + summary_df["entity_name"]).sort_values().tolist()
choice = st.sidebar.selectbox("Entity", [TOTAL] + labels)
selected_id = None if choice == TOTAL else choice.split(" - ", 1)[0]

split = st.sidebar.radio("Split by", ["Sector", "Entity"], horizontal=True)
measure = st.sidebar.radio("Measure", ["Amount", "Count"], horizontal=True)

dim = "sector" if split == "Sector" else "entity_name"
in_col, out_col = ("incomes", "payments") if measure == "Amount" else ("num_incomes", "num_payments")

# VIEWING
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

# CHARTS
def fold_to_eight(df, dim, value_col):
    """Categorical colour caps at 8 slots - keep the 7 largest, fold the rest into Other."""
    d = df.groupby(dim, as_index=False)[value_col].sum()
    d = d.sort_values(value_col, ascending=False, key=abs).reset_index(drop=True)
    if len(d) > 8:
        other = pd.DataFrame({dim: ["Other"], value_col: [d[value_col][7:].sum()]})
        d = pd.concat([d.head(7), other], ignore_index=True)
    return d

def share_pie(df_rows, dim, value_col, title):
    d = fold_to_eight(df_rows[df_rows["entity_name"] != "TOTAL"], dim, value_col)
    fig = go.Figure(go.Pie(labels=d[dim], values=d[value_col], sort=False, hole=0.55,
                           marker=dict(colors=CATEGORICAL, line=dict(color=SURFACE, width=2)),
                           textinfo="label+percent", textposition="outside"))
    fig.update_layout(title=title, height=380, showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
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
    fig.update_layout(height=460, bargap=0.25, xaxis_title=None, yaxis_title=y_title,
                      legend_title_text=split,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#c3c2b7", color=MUTED)
    return fig

tab_entity, tab_benchmark = st.tabs(["Entity Analysis", "Reported vs Computed"])

with tab_entity:
    st.title(choice)

    p1, p2 = st.columns(2)
    p1.plotly_chart(share_pie(summary_view, dim, in_col, f"Incomes by {split.lower()}"), width="stretch")
    p2.plotly_chart(share_pie(summary_view, dim, out_col, f"Payments by {split.lower()}"), width="stretch")

    # the total row is last, so iloc[-1] reads the total for TOTAL and the only row otherwise
    c1, c2 = st.columns(2)
    c1.metric("Incomes", f"R {summary_view['incomes'].iloc[-1]:,.0f}")
    c2.metric("Payments", f"R {summary_view['payments'].iloc[-1]:,.0f}")

    st.subheader("Summary")
    st.dataframe(summary_view, hide_index=True, width="stretch")

    st.subheader("Reference Types")
    st.dataframe(reference_view, hide_index=True, width="stretch")

    bar_view, bar_title = ((reference_view, "Net ZAR") if measure == "Amount"
                           else (reference_count_view, "Transactions"))
    st.plotly_chart(reference_bar(bar_view, dim, bar_title), width="stretch")

# BENCHMARK COMPARISON
GREY, AMBER = "rgba(137,135,129,0.16)", "rgba(250,178,25,0.22)"
VALUE_COLS = ["reported_value", "summation_value", "difference"]

def highlight(row):
    """Grey where nothing was computed, amber where the ratio implies a reporting scale error."""
    if pd.isna(row["summation_value"]):
        return [f"background-color: {GREY}"] * len(row)
    if abs(row["pct_of_reported"]) > 50:
        return [f"background-color: {AMBER}"] * len(row)
    return [""] * len(row)

with tab_benchmark:
    st.title("Reported vs Computed")

    scope = st.radio("View", ["All", "By sector", "By entity"], horizontal=True)

    if scope == "By sector":
        sector = st.selectbox("Sector", sorted(comparison_df["sector"].dropna().unique()))
        table = comparison_df[comparison_df["sector"] == sector]
        st.caption(sector.replace("_", " ").title())
    elif scope == "By entity":
        picks = (comparison_df["entity_id"] + " - " + comparison_df["entity_name"]).drop_duplicates().sort_values()
        pick = st.selectbox("Entity", picks)
        table = comparison_df[comparison_df["entity_id"] == pick.split(" - ", 1)[0]]
        st.caption(pick)
    else:
        table = comparison_df
        st.caption("All entities")

    matched = table["summation_value"].notna()

    b1, b2, b3 = st.columns(3)
    b1.metric("Reported lines", len(table))
    b2.metric("Matched", f"{matched.sum()} of {len(table)}")
    b3.metric("Median wallet share",
              f"{table.loc[matched, 'pct_of_reported'].median():.2f}%" if matched.any() else "-")

    styled = (table.drop(columns=["sector"])
                   .sort_values(["entity_id", "line_item", "fiscal_year"])
                   .style.apply(highlight, axis=1)
                   .format({c: "{:,.0f}" for c in VALUE_COLS} | {"pct_of_reported": "{:,.2f}%"}))

    st.dataframe(styled, hide_index=True, width="stretch", height=520)
    st.caption("Grey - no computed equivalent. Amber - computed exceeds 50% of reported, "
               "which in this data means a scale error in the extracted figure rather than a real match.")

    st.subheader("By line item")
    by_line = table.groupby("line_item", as_index=False).agg(
        reported_lines=("reported_value", "size"),
        matched=("summation_value", "count"),
        median_pct=("pct_of_reported", "median"),
    )
    st.dataframe(by_line.style.format({"median_pct": "{:,.3f}%"}), hide_index=True, width="stretch")
