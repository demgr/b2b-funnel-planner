# ============================================================
#  B2B Revenue Reverse Funnel Planner – Streamlit App
#  Methodology: PERT Estimation + Monte Carlo Simulation
#  Based on: B2B_Reverse_Funnel_Planner v49
#
#  Run:  streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="B2B Reverse Funnel Planner · Marko Gross",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stSidebar"] { background-color: #f0f2f6; }
    h1 { color: #1a1a2e; }
    h2 { color: #1a1a2e; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS & DEFAULT DATA
# ============================================================

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Funnel archetypes with conversion rates in % (integer, e.g. 25 = 25%)
FUNNEL_ARCHETYPES = {
    "Classic B2B": {
        "labels": ["Win Rate (Opp→Deal)", "SQL→Opp", "MQL→SQL", "Lead→MQL", "Touchpoint→Lead"],
        "stage_names": ["Revenue", "Deals", "Opportunities", "SQLs", "MQLs", "Leads", "Touchpoints"],
        "base":  [25, 40, 30, 15, 2],
        "worst": [15, 25, 20, 10, 1],
        "best":  [35, 55, 40, 20, 3],
    },
    "Enterprise": {
        "labels": ["Win Rate (Opp→Deal)", "SQL→Opp", "MQL→SQL", "Lead→MQL", "Touchpoint→Lead"],
        "stage_names": ["Revenue", "Deals", "Opportunities", "SQLs", "MQLs", "Leads", "Touchpoints"],
        "base":  [18, 35, 60, 25, 1],
        "worst": [12, 20, 45, 15, 1],
        "best":  [25, 50, 75, 35, 2],
    },
    "SaaS / PLG": {
        "labels": ["Win Rate (Opp→Deal)", "SQL→Opp", "PQL→SQL", "Signup→PQL", "Visitor→Signup"],
        "stage_names": ["Revenue", "Deals", "Opportunities", "PQLs", "Signups", "Visitors", "Touchpoints"],
        "base":  [20, 45, 40, 15, 5],
        "worst": [12, 30, 25,  8, 2],
        "best":  [30, 60, 55, 25, 10],
    },
    "Channel / Partner": {
        "labels": ["Win Rate (Opp→Deal)", "Partner SQL→Opp", "Partner Lead→SQL", "Contact→Lead", "Touchpoint→Contact"],
        "stage_names": ["Revenue", "Deals", "Opportunities", "SQLs", "MQLs", "Contacts", "Touchpoints"],
        "base":  [30, 50, 35, 20, 4],
        "worst": [20, 35, 20, 10, 2],
        "best":  [40, 70, 50, 30, 7],
    },
}

SEASONALITY_PROFILES = {
    "Flat":            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "Q4 heavy":        [0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 0.8, 0.8, 1.1, 1.3, 1.4, 1.2],
    "Events & Autumn": [1.1, 1.1, 1.2, 1.0, 0.9, 0.8, 0.7, 0.8, 1.2, 1.3, 1.2, 0.7],
    "Q1 Budget Push":  [1.3, 1.2, 1.1, 0.9, 0.8, 0.8, 0.7, 0.7, 0.9, 1.0, 1.0, 1.0],
    "Summer Slowdown": [1.0, 1.0, 1.0, 1.1, 1.1, 0.9, 0.6, 0.6, 1.0, 1.2, 1.3, 1.2],
    "Mid-year Ramp":   [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.8],
    "Spring Launch":   [1.1, 1.2, 1.3, 1.2, 1.0, 0.8, 0.7, 0.7, 0.8, 0.9, 0.9, 0.8],
    "Autumn Launch":   [0.8, 0.8, 0.9, 1.0, 1.0, 0.9, 0.9, 1.0, 1.2, 1.3, 1.3, 0.9],
}

DEFAULT_CHANNELS = [
    {"group": "Paid Search",          "activity": "Brand Campaigns",        "cost_per_mql": 40,  "share": 14.0},
    {"group": "Paid Search",          "activity": "Non-Brand Campaigns",    "cost_per_mql": 70,  "share": 12.0},
    {"group": "Paid Social",          "activity": "Prospecting",            "cost_per_mql": 60,  "share": 11.0},
    {"group": "Paid Social",          "activity": "Retargeting",            "cost_per_mql": 40,  "share":  8.0},
    {"group": "Website",              "activity": "Organic Onsite",         "cost_per_mql": 15,  "share":  6.0},
    {"group": "Website",              "activity": "Direct & Referral",      "cost_per_mql": 15,  "share":  6.0},
    {"group": "Events",               "activity": "Trade Shows",            "cost_per_mql": 400, "share":  3.0},
    {"group": "Events",               "activity": "Roundtables & Sponsoring","cost_per_mql": 500, "share":  4.0},
    {"group": "Content",              "activity": "Blogs & Articles",       "cost_per_mql": 20,  "share":  5.0},
    {"group": "Content",              "activity": "Whitepapers",            "cost_per_mql": 60,  "share":  4.0},
    {"group": "Content",              "activity": "Case Studies",           "cost_per_mql": 100, "share":  3.0},
    {"group": "SEO & AI Visibility",  "activity": "Technical SEO",          "cost_per_mql": 15,  "share":  5.0},
    {"group": "SEO & AI Visibility",  "activity": "Content Cluster SEO",    "cost_per_mql": 25,  "share":  5.0},
    {"group": "ABM",                  "activity": "1:1 Enterprise ABM",     "cost_per_mql": 600, "share":  2.0},
    {"group": "ABM",                  "activity": "1:Few ABM",              "cost_per_mql": 200, "share":  3.0},
    {"group": "Marketing Automation", "activity": "Nurture Programs",       "cost_per_mql": 10,  "share":  5.0},
    {"group": "Marketing Automation", "activity": "Reactivation Flows",     "cost_per_mql": 15,  "share":  4.0},
]

# ============================================================
# CALCULATION ENGINE
# ============================================================

def run_funnel(revenue_target: float, deal_size: float, cr_pct: list) -> tuple:
    """
    Reverse funnel calculation.
    cr_pct: list of 5 conversion rates in % [0-100], e.g. [25, 40, 30, 15, 2]
    Returns: (deals, opps, sqls/stage3, mqls/stage4, leads/stage5, touchpoints)
    """
    crs = [c / 100 for c in cr_pct]
    deals       = revenue_target / deal_size
    opps        = deals  / crs[0]
    stage3      = opps   / crs[1]
    stage4      = stage3 / crs[2]
    stage5      = stage4 / crs[3]
    touchpoints = stage5 / crs[4]
    return deals, opps, stage3, stage4, stage5, touchpoints


def calc_channel_budget(mqls: float, ch_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate required spend per channel based on MQL target and channel mix."""
    df = ch_df.copy()
    total_share = df["share"].sum()
    if total_share == 0:
        total_share = 1
    df["share_norm"]    = df["share"] / total_share
    df["planned_mqls"]  = df["share_norm"] * mqls
    df["required_spend"] = df["planned_mqls"] * df["cost_per_mql"]
    return df


@st.cache_data(show_spinner=False)
def run_monte_carlo(
    revenue_target: float, deal_size: float,
    cr_means_pct: tuple, worst_pct: tuple, best_pct: tuple,
    avg_cpm: float, n_sims: int, seed: int = 42
) -> pd.DataFrame:
    """
    Monte Carlo simulation using PERT-based normal distributions.
    cr_means_pct / worst_pct / best_pct: tuples of int percentages
    avg_cpm: weighted average cost per MQL
    Returns DataFrame with columns: deals, stage3, stage4, stage5, mqls, leads, touchpoints, budget_req
    """
    cr_means = [c / 100 for c in cr_means_pct]
    # PERT standard deviation: sigma ≈ (Best - Worst) / 6
    cr_stds  = [(b - w) / 600 for w, b in zip(worst_pct, best_pct)]
    cpm_std  = avg_cpm * 0.25

    rng = np.random.default_rng(seed)
    results = []

    for _ in range(n_sims):
        crs = [
            float(np.clip(rng.normal(m, s), 0.005, 0.99))
            for m, s in zip(cr_means, cr_stds)
        ]
        cpm = max(1.0, float(rng.normal(avg_cpm, cpm_std)))

        deals       = revenue_target / deal_size
        opps        = deals  / crs[0]
        stage3      = opps   / crs[1]
        stage4      = stage3 / crs[2]
        stage5      = stage4 / crs[3]
        touchpoints = stage5 / crs[4]
        budget_req  = stage4 * cpm   # stage4 = MQLs for Classic/Enterprise/Partner

        results.append({
            "deals": deals, "opps": opps,
            "stage3": stage3, "stage4": stage4, "stage5": stage5,
            "touchpoints": touchpoints, "budget_req": budget_req,
        })

    return pd.DataFrame(results)


def get_season_weights(profile: str) -> list:
    """Normalize seasonality profile to sum = 1."""
    idx = SEASONALITY_PROFILES[profile]
    total = sum(idx)
    return [v / total for v in idx]


# ============================================================
# SESSION STATE INIT
# ============================================================

if "channels_df" not in st.session_state:
    st.session_state.channels_df = pd.DataFrame(DEFAULT_CHANNELS)

if "actual_df" not in st.session_state:
    st.session_state.actual_df = pd.DataFrame({
        "Month":              MONTHS,
        "Actual MQLs":        [0.0] * 12,
        "Actual Deals":       [0.0] * 12,
        "Actual Revenue (€)": [0.0] * 12,
        "Actual Budget (€)":  [0.0] * 12,
    })

# ============================================================
# SIDEBAR — INPUTS
# ============================================================

with st.sidebar:
    st.markdown("## 📊 Reverse Funnel Planner")
    st.caption("B2B Revenue Planning · PERT + Monte Carlo")
    st.divider()

    # ── Business Targets ──────────────────────────────────────
    st.markdown("### Business Targets")
    revenue_target = st.number_input(
        "Revenue Target (€)", 10_000, 100_000_000, 1_000_000, 50_000, format="%d"
    )
    deal_size = st.number_input(
        "Avg. Deal Size (€)", 500, 500_000, 5_000, 500, format="%d"
    )
    available_budget = st.number_input(
        "Available Marketing Budget (€)", 0, 20_000_000, 300_000, 10_000, format="%d"
    )

    st.divider()

    # ── Funnel Archetype & Scenario ────────────────────────────
    st.markdown("### Funnel Setup")
    archetype = st.selectbox("Funnel Archetype", list(FUNNEL_ARCHETYPES.keys()))
    scenario  = st.radio("Scenario", ["Worst", "Base", "Best"], index=1, horizontal=True)

    arch       = FUNNEL_ARCHETYPES[archetype]
    cr_labels  = arch["labels"]
    scen_defaults = {"Worst": arch["worst"], "Base": arch["base"], "Best": arch["best"]}[scenario]

    st.divider()

    # ── Conversion Rates ───────────────────────────────────────
    st.markdown("### Conversion Rates")
    st.caption(f"Default: {scenario} case · {archetype}")

    cr_pct = []
    for label, default in zip(cr_labels, scen_defaults):
        val = st.slider(label, 1, 99, int(default), 1, format="%d%%")
        cr_pct.append(val)

    st.divider()

    # ── Seasonality ────────────────────────────────────────────
    st.markdown("### Seasonality")
    season_profile = st.selectbox("Profile", list(SEASONALITY_PROFILES.keys()))

    st.divider()

    # ── Monte Carlo ────────────────────────────────────────────
    st.markdown("### Monte Carlo")
    n_sims = st.select_slider("Simulations", [200, 500, 1_000, 2_000], value=500)

# ============================================================
# MAIN CALCULATIONS
# ============================================================

try:
    deals, opps, stage3, stage4, stage5, touchpoints = run_funnel(
        revenue_target, deal_size, cr_pct
    )
except (ZeroDivisionError, ValueError):
    st.error("⚠️ Division by zero — please raise one or more conversion rates above 0%.")
    st.stop()

stage_names  = arch["stage_names"]
funnel_vals  = [revenue_target, deals, opps, stage3, stage4, stage5, touchpoints]

# Channel budget
ch_calc       = calc_channel_budget(stage4, st.session_state.channels_df)
total_required = ch_calc["required_spend"].sum()
avg_cpm        = (total_required / stage4) if stage4 > 0 else 50.0
coverage       = (available_budget / total_required) if total_required > 0 else 0.0
budget_gap     = available_budget - total_required

# Seasonality
season_weights = get_season_weights(season_profile)

# Monte Carlo
mc_df = run_monte_carlo(
    float(revenue_target), float(deal_size),
    tuple(cr_pct), tuple(arch["worst"]), tuple(arch["best"]),
    float(avg_cpm), int(n_sims),
)

# ============================================================
# HEADER & KPI STRIP
# ============================================================

st.title("📊 B2B Revenue Reverse Funnel Planner")
st.caption(
    f"Archetype: **{archetype}** · Scenario: **{scenario}** · "
    f"Season: **{season_profile}** · {n_sims} Monte Carlo Simulations"
)

cov_icon  = "🟢" if coverage >= 0.9 else ("🟡" if coverage >= 0.6 else "🔴")
gap_label = f"+€{budget_gap:,.0f}" if budget_gap >= 0 else f"-€{abs(budget_gap):,.0f}"
gap_color = "normal" if budget_gap >= 0 else "inverse"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Revenue Target",      f"€{revenue_target:,.0f}")
k2.metric("Required Budget",     f"€{total_required:,.0f}", delta=gap_label, delta_color=gap_color)
k3.metric(f"{cov_icon} Coverage", f"{coverage:.0%}")
k4.metric("MQLs needed",         f"{stage4:,.0f}")
k5.metric("Deals needed",        f"{deals:,.0f}")

st.divider()

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔻 Funnel & Budget",
    "🎲 Risk / Monte Carlo",
    "📅 Monthly Plan",
    "📡 Channels",
    "📈 Plan vs. Actual",
])

# ──────────────────────────────────────────────────────────────
# TAB 1 – FUNNEL & BUDGET
# ──────────────────────────────────────────────────────────────
with tab1:
    col_funnel, col_budget = st.columns(2)

    with col_funnel:
        st.subheader("Reverse Funnel")
        # Funnel chart: Touchpoints (top/widest) → Revenue (bottom/narrowest)
        f_labels = list(reversed(stage_names[1:]))   # skip Revenue for cleaner chart
        f_values = list(reversed(funnel_vals[1:]))

        blue_gradient = ["#003580", "#004ea6", "#0066cc", "#1a8cff", "#4da6ff", "#80bfff"]

        fig_funnel = go.Figure(go.Funnel(
            y=f_labels,
            x=f_values,
            textinfo="value+percent initial",
            textfont=dict(size=12),
            marker=dict(color=blue_gradient),
            connector=dict(line=dict(color="#dee2e6", width=1, dash="dot")),
        ))
        fig_funnel.update_layout(
            height=430,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_budget:
        st.subheader("Budget Analysis")

        bar_color = "#dc3545" if total_required > available_budget else "#28a745"
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            x=["Available Budget", "Required Budget"],
            y=[available_budget, total_required],
            marker_color=["#0066cc", bar_color],
            text=[f"€{available_budget:,.0f}", f"€{total_required:,.0f}"],
            textposition="outside",
            width=0.45,
        ))
        fig_bars.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickformat=",.0f", title="€"),
            showlegend=False,
        )
        st.plotly_chart(fig_bars, use_container_width=True)

        # Coverage gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=min(coverage * 100, 150),
            delta={"reference": 100, "suffix": "%", "relative": False},
            number={"suffix": "%", "valueformat": ".0f", "font": {"size": 34}},
            title={"text": "Budget Coverage", "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 150], "ticksuffix": "%"},
                "bar": {"color": "#0066cc", "thickness": 0.3},
                "steps": [
                    {"range": [0,  60], "color": "#f8d7da"},
                    {"range": [60, 90], "color": "#fff3cd"},
                    {"range": [90, 150], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "#333", "width": 2},
                    "thickness": 0.75,
                    "value": 100,
                },
            },
        ))
        fig_gauge.update_layout(
            height=190,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Funnel table
    st.subheader("Funnel Numbers")
    rows = []
    for name, val in zip(stage_names, funnel_vals):
        prefix = "€" if name == "Revenue" else ""
        rows.append({
            "Stage":      name,
            "Annual":     f"{prefix}{val:,.0f}",
            "Per Month":  f"{prefix}{val/12:,.1f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# TAB 2 – MONTE CARLO
# ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Risk Analysis — Monte Carlo")
    st.caption(
        f"**{n_sims} simulations** · PERT distributions derived from Worst/Base/Best CRs · "
        f"Cost/MQL variance: ±25%"
    )

    p10_b  = float(np.percentile(mc_df["budget_req"], 10))
    p50_b  = float(np.percentile(mc_df["budget_req"], 50))
    p90_b  = float(np.percentile(mc_df["budget_req"], 90))
    p10_m  = float(np.percentile(mc_df["stage4"], 10))
    p50_m  = float(np.percentile(mc_df["stage4"], 50))
    p90_m  = float(np.percentile(mc_df["stage4"], 90))
    prob_ok = float((mc_df["budget_req"] <= available_budget).mean())

    ca, cb, cc = st.columns(3)

    with ca:
        st.markdown("**Required Budget (€)**")
        st.metric("P10 – optimistic",    f"€{p10_b:,.0f}")
        st.metric("Median",              f"€{p50_b:,.0f}")
        st.metric("P90 – conservative",  f"€{p90_b:,.0f}")

    with cb:
        st.markdown(f"**{stage_names[4]} needed**")
        st.metric("P10 – optimistic",    f"{p10_m:,.0f}")
        st.metric("Median",              f"{p50_m:,.0f}")
        st.metric("P90 – conservative",  f"{np.percentile(mc_df['stage4'], 90):,.0f}")

    with cc:
        st.markdown("**Budget sufficiency**")
        st.metric("Probability budget covers need", f"{prob_ok:.0%}")
        if prob_ok >= 0.70:
            st.success(f"✅ In {prob_ok:.0%} of scenarios your budget is sufficient.")
        elif prob_ok >= 0.30:
            st.warning(f"⚠️ Budget covers the need in only {prob_ok:.0%} of scenarios.")
        else:
            st.error(f"🔴 Budget is insufficient in {1 - prob_ok:.0%} of scenarios.")

    # Histograms
    fig_hist = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Required Budget Distribution", f"{stage_names[4]} Distribution"],
        horizontal_spacing=0.08,
    )
    fig_hist.add_trace(
        go.Histogram(x=mc_df["budget_req"], nbinsx=40,
                     marker_color="#0066cc", opacity=0.75, name="Budget",
                     hovertemplate="€%{x:,.0f}<br>Count: %{y}"),
        row=1, col=1,
    )
    fig_hist.add_vline(x=available_budget, line_dash="dash", line_color="#dc3545",
                       annotation_text=f"Available €{available_budget:,.0f}",
                       annotation_font_color="#dc3545", row=1, col=1)
    fig_hist.add_vline(x=p50_b, line_dash="dot", line_color="#28a745",
                       annotation_text=f"Median €{p50_b:,.0f}",
                       annotation_font_color="#28a745", row=1, col=1)

    fig_hist.add_trace(
        go.Histogram(x=mc_df["stage4"], nbinsx=40,
                     marker_color="#28a745", opacity=0.75, name="MQLs",
                     hovertemplate="%{x:,.0f}<br>Count: %{y}"),
        row=1, col=2,
    )
    fig_hist.update_layout(
        height=360, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig_hist.update_xaxes(title_text="Required Budget (€)", row=1, col=1)
    fig_hist.update_xaxes(title_text=stage_names[4], row=1, col=2)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Risk Corridor
    st.subheader("Risk Corridor")
    corridor_items = [
        ("Required Budget (€)", "budget_req"),
        (f"{stage_names[4]} (MQLs)", "stage4"),
        (f"{stage_names[5]} (Leads)", "stage5"),
        ("Deals", "deals"),
    ]
    fig_corridor = go.Figure()
    for label, col_name in corridor_items:
        p10c = float(np.percentile(mc_df[col_name], 10))
        p50c = float(np.percentile(mc_df[col_name], 50))
        p90c = float(np.percentile(mc_df[col_name], 90))
        fig_corridor.add_trace(go.Scatter(
            x=[p10c, p50c, p90c],
            y=[label, label, label],
            mode="lines+markers+text",
            line=dict(color="#0066cc", width=3),
            marker=dict(size=[8, 14, 8], color=["#adb5bd", "#0066cc", "#adb5bd"]),
            text=[f"P10: {p10c:,.0f}", f"P50: {p50c:,.0f}", f"P90: {p90c:,.0f}"],
            textposition=["bottom center", "top center", "bottom center"],
            textfont=dict(size=10),
            showlegend=False,
        ))
    fig_corridor.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Value",
    )
    st.plotly_chart(fig_corridor, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 3 – MONTHLY PLAN
# ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader(f"Monthly Plan — {season_profile} Seasonality")

    monthly_data = {
        stage_names[4]:    [stage4 * w         for w in season_weights],   # MQLs
        stage_names[1]:    [deals * w           for w in season_weights],   # Deals
        "Revenue (€)":     [revenue_target * w  for w in season_weights],
        "Budget (€)":      [total_required * w  for w in season_weights],
    }
    metric_colors = {
        stage_names[4]: "#0066cc",
        stage_names[1]: "#28a745",
        "Revenue (€)":  "#fd7e14",
        "Budget (€)":   "#dc3545",
    }

    fig_monthly = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(monthly_data.keys()),
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )
    for i, (metric, values) in enumerate(monthly_data.items()):
        row, col = i // 2 + 1, i % 2 + 1
        avg = sum(values) / 12
        fig_monthly.add_trace(
            go.Bar(
                x=MONTHS, y=values,
                marker_color=metric_colors[metric],
                opacity=0.85, showlegend=False,
                hovertemplate=f"%{{x}}: %{{y:,.0f}}",
            ),
            row=row, col=col,
        )
        fig_monthly.add_hline(
            y=avg, line_dash="dot", line_color="#6c757d",
            annotation_text=f"⌀ {avg:,.0f}",
            annotation_position="top right",
            row=row, col=col,
        )
    fig_monthly.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=60, t=40, b=0),
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Monthly table
    table_rows = {"Month": MONTHS}
    for metric, values in monthly_data.items():
        if "€" in metric:
            table_rows[metric] = [f"€{v:,.0f}" for v in values]
        elif metric == stage_names[1]:   # Deals — often fractional
            table_rows[metric] = [f"{v:.1f}" for v in values]
        else:
            table_rows[metric] = [f"{v:,.0f}" for v in values]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# TAB 4 – CHANNELS
# ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Channel Mix & Budget Breakdown")
    st.caption(
        "Edit **Cost per MQL** and **Share** to customize your mix. "
        "Shares are auto-normalized — the total doesn't need to equal 100."
    )

    display_df = st.session_state.channels_df[
        ["group", "activity", "cost_per_mql", "share"]
    ].copy()
    display_df.columns = ["Channel Group", "Activity", "Cost per MQL (€)", "Share (%)"]

    edited = st.data_editor(
        display_df,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Channel Group": st.column_config.TextColumn("Channel Group", disabled=True),
            "Activity":      st.column_config.TextColumn("Activity",      disabled=True),
            "Cost per MQL (€)": st.column_config.NumberColumn(
                "Cost per MQL (€)", min_value=0, max_value=5_000, step=5,
                help="Average cost to generate one MQL via this channel",
            ),
            "Share (%)": st.column_config.NumberColumn(
                "Share (%)", min_value=0.0, max_value=100.0, step=0.5,
                help="Relative MQL share. Auto-normalized — totals don't need to be 100.",
            ),
        },
        key="ch_editor",
    )

    # Persist to session state
    updated_ch = st.session_state.channels_df.copy()
    updated_ch["cost_per_mql"] = edited["Cost per MQL (€)"].values
    updated_ch["share"]        = edited["Share (%)"].values
    st.session_state.channels_df = updated_ch

    # Recalculate with latest edits
    ch_result = calc_channel_budget(stage4, updated_ch)

    st.divider()

    col_pie, col_bar = st.columns(2)

    grp = (
        ch_result
        .groupby("group", as_index=False)
        .agg(planned_mqls=("planned_mqls", "sum"), required_spend=("required_spend", "sum"))
        .sort_values("required_spend", ascending=False)
    )

    with col_pie:
        st.markdown("**Budget Share by Channel Group**")
        fig_pie = px.pie(
            grp, values="required_spend", names="group",
            hole=0.42,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(
            height=370,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.markdown("**Required Spend per Channel Group**")
        fig_bar = go.Figure(go.Bar(
            y=grp["group"],
            x=grp["required_spend"],
            orientation="h",
            marker_color="#0066cc",
            text=[f"€{v:,.0f}" for v in grp["required_spend"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=370,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=90, t=10, b=0),
            xaxis=dict(title="Required Spend (€)", tickformat=",.0f"),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Summary table
    grp_display = grp.rename(columns={
        "group":          "Channel Group",
        "planned_mqls":   "Planned MQLs",
        "required_spend": "Required Spend (€)",
    })
    grp_display["Planned MQLs"]      = grp_display["Planned MQLs"].apply(lambda x: f"{x:,.0f}")
    grp_display["Required Spend (€)"] = grp_display["Required Spend (€)"].apply(lambda x: f"€{x:,.0f}")
    st.dataframe(grp_display, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# TAB 5 – PLAN VS. ACTUAL
# ──────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Plan vs. Actual")
    st.caption(
        "Trage deine monatlichen Ist-Werte ein. "
        "Oder lade Musterdaten zum Ausprobieren."
    )

    # ── Monatliche Planwerte berechnen ────────────────────────
    plan_mqls    = [stage4          * w for w in season_weights]
    plan_deals   = [deals           * w for w in season_weights]
    plan_revenue = [revenue_target  * w for w in season_weights]
    plan_budget  = [total_required  * w for w in season_weights]

    # ── Controls ──────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        reporting_month = st.selectbox(
            "Berichtsmonat (YTD bis einschließlich)",
            MONTHS,
            index=3,   # April als Default
        )
    with ctrl2:
        load_sample = st.button("📥 Musterdaten laden", use_container_width=True)
    with ctrl3:
        clear_data  = st.button("🗑️ Daten löschen",    use_container_width=True)

    report_idx = MONTHS.index(reporting_month)   # 0-based, e.g. April = 3

    # ── Musterdaten ──────────────────────────────────────────
    # Multiplier je Monat (Jan–Apr leicht variierend, Rest leer)
    SAMPLE_MULT = [0.90, 1.07, 1.14, 0.86, 0, 0, 0, 0, 0, 0, 0, 0]

    if load_sample:
        st.session_state.actual_df = pd.DataFrame({
            "Month":              MONTHS,
            "Actual MQLs":        [round(plan_mqls[i]    * SAMPLE_MULT[i], 0) for i in range(12)],
            "Actual Deals":       [round(plan_deals[i]   * SAMPLE_MULT[i], 1) for i in range(12)],
            "Actual Revenue (€)": [round(plan_revenue[i] * SAMPLE_MULT[i], 0) for i in range(12)],
            "Actual Budget (€)":  [round(plan_budget[i]  * SAMPLE_MULT[i], 0) for i in range(12)],
        })
        st.rerun()

    if clear_data:
        st.session_state.actual_df = pd.DataFrame({
            "Month":              MONTHS,
            "Actual MQLs":        [0.0] * 12,
            "Actual Deals":       [0.0] * 12,
            "Actual Revenue (€)": [0.0] * 12,
            "Actual Budget (€)":  [0.0] * 12,
        })
        st.rerun()

    # ── Editierbare Tabelle ───────────────────────────────────
    st.markdown("**Monatliche Ist-Werte eingeben**")
    edited_actual = st.data_editor(
        st.session_state.actual_df,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Month":              st.column_config.TextColumn("Monat", disabled=True),
            "Actual MQLs":        st.column_config.NumberColumn("Ist MQLs",       min_value=0, step=1,    format="%.0f"),
            "Actual Deals":       st.column_config.NumberColumn("Ist Deals",      min_value=0, step=0.5,  format="%.1f"),
            "Actual Revenue (€)": st.column_config.NumberColumn("Ist Revenue (€)", min_value=0, step=1000, format="€%.0f"),
            "Actual Budget (€)":  st.column_config.NumberColumn("Ist Budget (€)",  min_value=0, step=500,  format="€%.0f"),
        },
        key="actual_editor",
    )
    st.session_state.actual_df = edited_actual

    act_mqls    = edited_actual["Actual MQLs"].tolist()
    act_deals   = edited_actual["Actual Deals"].tolist()
    act_revenue = edited_actual["Actual Revenue (€)"].tolist()
    act_budget  = edited_actual["Actual Budget (€)"].tolist()

    st.divider()

    # ── YTD KPI Strip ─────────────────────────────────────────
    def ytd_rate(actuals, plans, end_idx):
        """Achievement rate YTD up to end_idx (inclusive)."""
        a = sum(actuals[:end_idx + 1])
        p = sum(plans[:end_idx + 1])
        return (a / p) if p > 0 else 0.0

    def ytd_icon(rate):
        return "🟢" if rate >= 0.95 else ("🟡" if rate >= 0.75 else "🔴")

    r_mql  = ytd_rate(act_mqls,    plan_mqls,    report_idx)
    r_deal = ytd_rate(act_deals,   plan_deals,   report_idx)
    r_rev  = ytd_rate(act_revenue, plan_revenue, report_idx)
    r_bud  = ytd_rate(act_budget,  plan_budget,  report_idx)

    st.markdown(f"**YTD-Zielerreichung bis {reporting_month}**")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{ytd_icon(r_mql)}  MQLs",        f"{r_mql:.0%}",
              delta=f"{sum(act_mqls[:report_idx+1]):,.0f} / {sum(plan_mqls[:report_idx+1]):,.0f}")
    k2.metric(f"{ytd_icon(r_deal)} Deals",        f"{r_deal:.0%}",
              delta=f"{sum(act_deals[:report_idx+1]):.1f} / {sum(plan_deals[:report_idx+1]):.1f}")
    k3.metric(f"{ytd_icon(r_rev)}  Revenue",      f"{r_rev:.0%}",
              delta=f"€{sum(act_revenue[:report_idx+1]):,.0f} / €{sum(plan_revenue[:report_idx+1]):,.0f}")
    k4.metric(f"{ytd_icon(r_bud)}  Budget Spend", f"{r_bud:.0%}",
              delta=f"€{sum(act_budget[:report_idx+1]):,.0f} / €{sum(plan_budget[:report_idx+1]):,.0f}")

    st.divider()

    # ── Plan vs. Actual Charts (2×2) ─────────────────────────
    metrics_pva = [
        ("MQLs",        plan_mqls,    act_mqls,    "#0066cc", "#80bfff"),
        ("Deals",       plan_deals,   act_deals,   "#28a745", "#a3d9b1"),
        ("Revenue (€)", plan_revenue, act_revenue, "#fd7e14", "#ffd0a0"),
        ("Budget (€)",  plan_budget,  act_budget,  "#6f42c1", "#c9b8f0"),
    ]

    fig_pva = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m[0] for m in metrics_pva],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    for i, (label, plan_vals, act_vals, color_plan, color_act) in enumerate(metrics_pva):
        row, col = i // 2 + 1, i % 2 + 1

        # Plan bars (light)
        fig_pva.add_trace(go.Bar(
            x=MONTHS, y=plan_vals,
            name=f"Plan {label}", marker_color=color_act,
            opacity=0.5, showlegend=(i == 0),
            legendgroup="plan",
            hovertemplate=f"Plan %{{x}}: %{{y:,.0f}}",
        ), row=row, col=col)

        # Actual bars (solid) — only up to reporting month
        act_visible = [v if j <= report_idx else None for j, v in enumerate(act_vals)]
        fig_pva.add_trace(go.Bar(
            x=MONTHS, y=act_visible,
            name=f"Ist {label}", marker_color=color_plan,
            opacity=0.9, showlegend=(i == 0),
            legendgroup="actual",
            hovertemplate=f"Ist %{{x}}: %{{y:,.0f}}",
        ), row=row, col=col)

        # Vertical line at reporting month
        fig_pva.add_vline(
            x=reporting_month, line_dash="dot", line_color="#6c757d",
            row=row, col=col,
        )

    fig_pva.update_layout(
        barmode="overlay",
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_pva, use_container_width=True)

    # ── Kumulativer YTD-Verlauf ───────────────────────────────
    st.subheader("Kumulativer Verlauf (Revenue)")

    cum_plan_rev = [sum(plan_revenue[:i+1]) for i in range(12)]
    cum_act_rev  = [sum(act_revenue[:i+1]) if i <= report_idx else None for i in range(12)]

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=MONTHS, y=cum_plan_rev,
        name="Plan kumuliert",
        line=dict(color="#adb5bd", width=2, dash="dot"),
        mode="lines+markers", marker=dict(size=6),
    ))
    fig_cum.add_trace(go.Scatter(
        x=MONTHS[:report_idx+1],
        y=cum_act_rev[:report_idx+1],
        name="Ist kumuliert",
        line=dict(color="#fd7e14", width=3),
        mode="lines+markers", marker=dict(size=8),
        fill="tonexty", fillcolor="rgba(253,126,20,0.08)",
    ))
    fig_cum.add_hline(
        y=revenue_target, line_dash="dash", line_color="#0066cc",
        annotation_text=f"Jahresziel €{revenue_target:,.0f}",
        annotation_font_color="#0066cc",
    )
    fig_cum.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(tickformat=",.0f", title="Revenue (€)"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_cum, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "**B2B Revenue Reverse Funnel Planner** · "
    "Methodology: PERT Estimation + Monte Carlo Simulation · "
    "Built with Streamlit & Plotly · "
    "© Marko Gross"
)
