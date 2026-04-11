# ============================================================
#  B2B Revenue Reverse Funnel Planner
#  Methodology: PERT Estimation + Monte Carlo Simulation
#  © Marko Gross
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
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stSidebar"] { background-color: #0d1b2a; }
    div[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    h1 { color: #1a1a2e; }
    .section-header {
        font-size: 1rem; font-weight: 700;
        color: #0066cc; margin-bottom: 4px;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

FUNNEL_ARCHETYPES = {
    "Classic B2B": {
        "labels":      ["Win Rate (Opp→Deal)", "SQL→Opp", "MQL→SQL", "Lead→MQL", "Touchpoint→Lead"],
        "stage_names": ["Revenue", "Deals", "Opportunities", "SQLs", "MQLs", "Leads", "Touchpoints"],
        "base":  [25, 40, 30, 15, 2],
        "worst": [15, 25, 20, 10, 1],
        "best":  [35, 55, 40, 20, 3],
    },
    "Enterprise": {
        "labels":      ["Win Rate (Opp→Deal)", "SQL→Opp", "MQL→SQL", "Lead→MQL", "Touchpoint→Lead"],
        "stage_names": ["Revenue", "Deals", "Opportunities", "SQLs", "MQLs", "Leads", "Touchpoints"],
        "base":  [18, 35, 60, 25, 1],
        "worst": [12, 20, 45, 15, 1],
        "best":  [25, 50, 75, 35, 2],
    },
    "SaaS / PLG": {
        "labels":      ["Win Rate (Opp→Deal)", "SQL→Opp", "PQL→SQL", "Signup→PQL", "Visitor→Signup"],
        "stage_names": ["Revenue", "Deals", "Opportunities", "PQLs", "Signups", "Visitors", "Touchpoints"],
        "base":  [20, 45, 40, 15, 5],
        "worst": [12, 30, 25,  8, 2],
        "best":  [30, 60, 55, 25, 10],
    },
    "Channel / Partner": {
        "labels":      ["Win Rate (Opp→Deal)", "Partner SQL→Opp", "Partner Lead→SQL", "Contact→Lead", "Touchpoint→Contact"],
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
    "Eigenes Profil":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

DEFAULT_CHANNELS = [
    {"group": "Paid Search",          "activity": "Brand Campaigns",         "cost_per_mql": 40,  "share": 14.0},
    {"group": "Paid Search",          "activity": "Non-Brand Campaigns",     "cost_per_mql": 70,  "share": 12.0},
    {"group": "Paid Social",          "activity": "Prospecting",             "cost_per_mql": 60,  "share": 11.0},
    {"group": "Paid Social",          "activity": "Retargeting",             "cost_per_mql": 40,  "share":  8.0},
    {"group": "Website",              "activity": "Organic Onsite",          "cost_per_mql": 15,  "share":  6.0},
    {"group": "Website",              "activity": "Direct & Referral",       "cost_per_mql": 15,  "share":  6.0},
    {"group": "Events",               "activity": "Trade Shows",             "cost_per_mql": 400, "share":  3.0},
    {"group": "Events",               "activity": "Roundtables & Sponsoring","cost_per_mql": 500, "share":  4.0},
    {"group": "Content",              "activity": "Blogs & Articles",        "cost_per_mql": 20,  "share":  5.0},
    {"group": "Content",              "activity": "Whitepapers",             "cost_per_mql": 60,  "share":  4.0},
    {"group": "Content",              "activity": "Case Studies",            "cost_per_mql": 100, "share":  3.0},
    {"group": "SEO & AI Visibility",  "activity": "Technical SEO",           "cost_per_mql": 15,  "share":  5.0},
    {"group": "SEO & AI Visibility",  "activity": "Content Cluster SEO",     "cost_per_mql": 25,  "share":  5.0},
    {"group": "ABM",                  "activity": "1:1 Enterprise ABM",      "cost_per_mql": 600, "share":  2.0},
    {"group": "ABM",                  "activity": "1:Few ABM",               "cost_per_mql": 200, "share":  3.0},
    {"group": "Marketing Automation", "activity": "Nurture Programs",        "cost_per_mql": 10,  "share":  5.0},
    {"group": "Marketing Automation", "activity": "Reactivation Flows",      "cost_per_mql": 15,  "share":  4.0},
]

# ============================================================
# HELPERS
# ============================================================

def make_cr_df(archetype: str) -> pd.DataFrame:
    a = FUNNEL_ARCHETYPES[archetype]
    return pd.DataFrame({
        "Stage":      a["labels"],
        "Worst (%)":  a["worst"],
        "Base (%)":   a["base"],
        "Best (%)":   a["best"],
    })

def run_funnel(revenue_target, deal_size, cr_pct):
    crs = [c / 100 for c in cr_pct]
    deals       = revenue_target / deal_size
    opps        = deals  / crs[0]
    stage3      = opps   / crs[1]
    stage4      = stage3 / crs[2]
    stage5      = stage4 / crs[3]
    touchpoints = stage5 / crs[4]
    return deals, opps, stage3, stage4, stage5, touchpoints

def calc_channel_budget(mqls, ch_df):
    df = ch_df.copy()
    total = df["share"].sum() or 1
    df["share_norm"]     = df["share"] / total
    df["planned_mqls"]   = df["share_norm"] * mqls
    df["required_spend"] = df["planned_mqls"] * df["cost_per_mql"]
    return df

@st.cache_data(show_spinner=False)
def run_monte_carlo(revenue_target, deal_size, cr_means_pct, worst_pct, best_pct,
                    avg_cpm, n_sims, seed=42):
    cr_means = [c / 100 for c in cr_means_pct]
    cr_stds  = [(b - w) / 600 for w, b in zip(worst_pct, best_pct)]
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_sims):
        crs = [float(np.clip(rng.normal(m, s), 0.005, 0.99))
               for m, s in zip(cr_means, cr_stds)]
        cpm = max(1.0, float(rng.normal(avg_cpm, avg_cpm * 0.25)))
        deals  = revenue_target / deal_size
        opps   = deals  / crs[0]
        s3     = opps   / crs[1]
        s4     = s3     / crs[2]
        s5     = s4     / crs[3]
        tp     = s5     / crs[4]
        results.append({"deals": deals, "opps": opps,
                        "stage3": s3, "stage4": s4, "stage5": s5,
                        "touchpoints": tp, "budget_req": s4 * cpm})
    return pd.DataFrame(results)

def get_season_weights(profile):
    idx   = SEASONALITY_PROFILES[profile]
    total = sum(idx)
    return [v / total for v in idx]

# ============================================================
# SESSION STATE — DEFAULTS
# ============================================================

_DEFAULTS = {
    "inp_revenue":       1_000_000,
    "inp_deal_size":     5_000,
    "inp_budget":        300_000,
    "inp_archetype":     "Classic B2B",
    "inp_scenario":      "Base",
    "inp_season":        "Flat",
    "inp_custom_season": [1.0] * 12,
    "inp_n_sims":        500,
    "inp_report_month":  "Apr",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "inp_cr_df" not in st.session_state:
    st.session_state.inp_cr_df = make_cr_df("Classic B2B")

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
# READ ACTIVE INPUTS FROM SESSION STATE
# ============================================================

revenue_target  = st.session_state.inp_revenue
deal_size       = st.session_state.inp_deal_size
available_budget= st.session_state.inp_budget
archetype       = st.session_state.inp_archetype
scenario        = st.session_state.inp_scenario
season_profile  = st.session_state.inp_season
n_sims          = st.session_state.inp_n_sims
arch            = FUNNEL_ARCHETYPES[archetype]

# Active CR column
scen_col = {"Worst": "Worst (%)", "Base": "Base (%)", "Best": "Best (%)"}[scenario]
cr_pct   = st.session_state.inp_cr_df[scen_col].tolist()

# Seasonality weights
if season_profile == "Eigenes Profil":
    raw_weights = st.session_state.inp_custom_season
else:
    raw_weights = SEASONALITY_PROFILES[season_profile]
total_w        = sum(raw_weights) or 1
season_weights = [v / total_w for v in raw_weights]

# ============================================================
# MAIN CALCULATIONS
# ============================================================

try:
    deals, opps, stage3, stage4, stage5, touchpoints = run_funnel(
        revenue_target, deal_size, cr_pct
    )
except (ZeroDivisionError, ValueError):
    deals = opps = stage3 = stage4 = stage5 = touchpoints = 0.0

stage_names = arch["stage_names"]
funnel_vals = [revenue_target, deals, opps, stage3, stage4, stage5, touchpoints]

ch_calc        = calc_channel_budget(stage4, st.session_state.channels_df)
total_required = ch_calc["required_spend"].sum()
avg_cpm        = (total_required / stage4) if stage4 > 0 else 50.0
coverage       = (available_budget / total_required) if total_required > 0 else 0.0
budget_gap     = available_budget - total_required

mc_df = run_monte_carlo(
    float(revenue_target), float(deal_size),
    tuple(cr_pct), tuple(arch["worst"]), tuple(arch["best"]),
    float(avg_cpm), int(n_sims),
)

plan_mqls    = [stage4         * w for w in season_weights]
plan_deals   = [deals          * w for w in season_weights]
plan_revenue = [revenue_target * w for w in season_weights]
plan_budget  = [total_required * w for w in season_weights]

# ============================================================
# SIDEBAR — compact KPI panel
# ============================================================

with st.sidebar:
    st.markdown("### 📊 B2B Funnel Planner")
    st.caption("© Marko Gross")
    st.divider()
    cov_icon = "🟢" if coverage >= 0.9 else ("🟡" if coverage >= 0.6 else "🔴")
    st.metric("Revenue Target",   f"€{revenue_target:,.0f}")
    st.metric("Required Budget",  f"€{total_required:,.0f}")
    st.metric(f"{cov_icon} Coverage", f"{coverage:.0%}")
    st.metric("MQLs needed",      f"{stage4:,.0f}")
    st.metric("Deals needed",     f"{deals:,.0f}")
    st.divider()
    st.caption(f"Archetype: **{archetype}**")
    st.caption(f"Scenario: **{scenario}**")
    st.caption(f"Seasonality: **{season_profile}**")

# ============================================================
# HEADER
# ============================================================

st.title("📊 B2B Revenue Reverse Funnel Planner")
gap_label = f"+€{budget_gap:,.0f}" if budget_gap >= 0 else f"-€{abs(budget_gap):,.0f}"
gap_color = "normal" if budget_gap >= 0 else "inverse"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Revenue Target",       f"€{revenue_target:,.0f}")
k2.metric("Required Budget",      f"€{total_required:,.0f}", delta=gap_label, delta_color=gap_color)
k3.metric(f"{cov_icon} Coverage", f"{coverage:.0%}")
k4.metric("MQLs needed",          f"{stage4:,.0f}")
k5.metric("Deals needed",         f"{deals:,.0f}")
st.divider()

# ============================================================
# TABS
# ============================================================

tab_inp, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚙️ Inputs",
    "🔻 Funnel & Budget",
    "🎲 Risk / Monte Carlo",
    "📅 Monthly Plan",
    "📈 Plan vs. Actual",
    "📡 Channels",
])

# ══════════════════════════════════════════════════════════════
# TAB: INPUTS
# ══════════════════════════════════════════════════════════════
with tab_inp:

    # ── A: Business Targets ───────────────────────────────────
    st.markdown('<p class="section-header">A · Business Targets</p>', unsafe_allow_html=True)
    ca1, ca2, ca3 = st.columns(3)
    with ca1:
        st.session_state.inp_revenue = st.number_input(
            "Revenue Target (€)", 10_000, 100_000_000,
            st.session_state.inp_revenue, 50_000, format="%d",
            help="Umsatzziel für die Planungsperiode"
        )
    with ca2:
        st.session_state.inp_deal_size = st.number_input(
            "Avg. Deal Size (€)", 500, 500_000,
            st.session_state.inp_deal_size, 500, format="%d",
            help="Durchschnittlicher Auftragswert"
        )
    with ca3:
        st.session_state.inp_budget = st.number_input(
            "Available Marketing Budget (€)", 0, 20_000_000,
            st.session_state.inp_budget, 10_000, format="%d",
            help="Verfügbares Marketingbudget gesamt"
        )

    st.divider()

    # ── B: Funnel Setup ───────────────────────────────────────
    st.markdown('<p class="section-header">B · Funnel Archetype & Szenario</p>', unsafe_allow_html=True)
    cb1, cb2, cb3 = st.columns([2, 2, 1])

    with cb1:
        prev_arch = st.session_state.inp_archetype
        st.session_state.inp_archetype = st.selectbox(
            "Funnel Archetype",
            list(FUNNEL_ARCHETYPES.keys()),
            index=list(FUNNEL_ARCHETYPES.keys()).index(st.session_state.inp_archetype),
        )
        # Auto-reset CR table when archetype changes
        if st.session_state.inp_archetype != prev_arch:
            st.session_state.inp_cr_df = make_cr_df(st.session_state.inp_archetype)

    with cb2:
        st.session_state.inp_scenario = st.radio(
            "Aktives Szenario",
            ["Worst", "Base", "Best"],
            index=["Worst", "Base", "Best"].index(st.session_state.inp_scenario),
            horizontal=True,
            help="Bestimmt welche Conversion-Rate-Spalte aktiv ist"
        )

    with cb3:
        if st.button("↺ CRs zurücksetzen", use_container_width=True,
                     help="Conversion Rates auf Archetype-Defaults zurücksetzen"):
            st.session_state.inp_cr_df = make_cr_df(st.session_state.inp_archetype)
            st.rerun()

    st.divider()

    # ── C: Conversion Rates ───────────────────────────────────
    st.markdown('<p class="section-header">C · Conversion Rates</p>', unsafe_allow_html=True)
    active_arch = FUNNEL_ARCHETYPES[st.session_state.inp_archetype]
    scen_col_now = {"Worst": "Worst (%)", "Base": "Base (%)", "Best": "Best (%)"}[st.session_state.inp_scenario]
    st.caption(
        f"Aktiv: **{st.session_state.inp_scenario}**-Spalte · "
        f"Archetype: {st.session_state.inp_archetype} · "
        f"Alle Werte in Prozent (z.B. 25 = 25 %)"
    )

    edited_cr = st.data_editor(
        st.session_state.inp_cr_df,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Stage":      st.column_config.TextColumn("Funnel-Stufe", disabled=True, width="large"),
            "Worst (%)":  st.column_config.NumberColumn("Worst (%)",  min_value=1, max_value=99, step=1,
                           help="Pessimistischer, aber realistischer Wert"),
            "Base (%)":   st.column_config.NumberColumn("Base (%)",   min_value=1, max_value=99, step=1,
                           help="Realistisch erwarteter Wert"),
            "Best (%)":   st.column_config.NumberColumn("Best (%)",   min_value=1, max_value=99, step=1,
                           help="Optimistischer, aber realistischer Wert"),
        },
        key="cr_editor",
    )
    st.session_state.inp_cr_df = edited_cr

    # Visual: aktive CRs als kleine Balken
    active_crs = edited_cr[scen_col_now].tolist()
    fig_cr = go.Figure(go.Bar(
        x=edited_cr["Stage"].tolist(),
        y=active_crs,
        marker_color="#0066cc",
        text=[f"{v}%" for v in active_crs],
        textposition="outside",
    ))
    fig_cr.update_layout(
        height=200, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(range=[0, 105], title="Rate (%)"),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    st.plotly_chart(fig_cr, use_container_width=True)

    st.divider()

    # ── D: Seasonality ────────────────────────────────────────
    st.markdown('<p class="section-header">D · Saisonalität</p>', unsafe_allow_html=True)
    cd1, cd2 = st.columns([1, 2])

    with cd1:
        prev_season = st.session_state.inp_season
        st.session_state.inp_season = st.selectbox(
            "Seasonality-Profil",
            list(SEASONALITY_PROFILES.keys()),
            index=list(SEASONALITY_PROFILES.keys()).index(st.session_state.inp_season),
        )

        if st.session_state.inp_season == "Eigenes Profil":
            st.caption("Indexwerte pro Monat (werden automatisch normiert):")
            custom_vals = []
            cols_m = st.columns(4)
            for i, month in enumerate(MONTHS):
                default_v = st.session_state.inp_custom_season[i]
                v = cols_m[i % 4].number_input(
                    month, min_value=0.1, max_value=5.0,
                    value=float(default_v), step=0.1,
                    key=f"custom_season_{i}"
                )
                custom_vals.append(v)
            st.session_state.inp_custom_season = custom_vals
            display_weights = custom_vals
        else:
            display_weights = SEASONALITY_PROFILES[st.session_state.inp_season]

    with cd2:
        norm_total = sum(display_weights) or 1
        norm_w = [v / norm_total * 12 for v in display_weights]   # normalized around 1.0
        colors_season = ["#0066cc" if v >= 1.0 else "#80bfff" for v in norm_w]

        fig_season = go.Figure(go.Bar(
            x=MONTHS, y=norm_w,
            marker_color=colors_season,
            text=[f"{v:.2f}x" for v in norm_w],
            textposition="outside",
        ))
        fig_season.add_hline(y=1.0, line_dash="dot", line_color="#6c757d",
                             annotation_text="Ø")
        fig_season.update_layout(
            height=230, margin=dict(l=0, r=0, t=10, b=20),
            yaxis=dict(title="Faktor (1.0 = Durchschnitt)", range=[0, max(norm_w) * 1.3]),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        )
        st.plotly_chart(fig_season, use_container_width=True)

    st.divider()

    # ── E: Channel Mix & Kosten ───────────────────────────────
    st.markdown('<p class="section-header">E · Channel Mix & Kosten</p>', unsafe_allow_html=True)
    st.caption(
        "**Cost per MQL**: durchschnittliche Kosten, um einen MQL über diesen Kanal zu erzeugen. "
        "**Share**: relativer Anteil — wird automatisch normiert."
    )

    ch_display = st.session_state.channels_df[
        ["group", "activity", "cost_per_mql", "share"]
    ].copy()
    ch_display.columns = ["Kanal-Gruppe", "Aktivität", "Cost per MQL (€)", "Share (%)"]

    edited_ch = st.data_editor(
        ch_display,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Kanal-Gruppe": st.column_config.TextColumn(disabled=True),
            "Aktivität":    st.column_config.TextColumn(disabled=True),
            "Cost per MQL (€)": st.column_config.NumberColumn(
                min_value=0, max_value=5_000, step=5,
                help="Kosten pro MQL für diesen Kanal"),
            "Share (%)": st.column_config.NumberColumn(
                min_value=0.0, max_value=100.0, step=0.5,
                help="Relativer MQL-Anteil (wird normiert)"),
        },
        key="inp_ch_editor",
    )
    updated_ch = st.session_state.channels_df.copy()
    updated_ch["cost_per_mql"] = edited_ch["Cost per MQL (€)"].values
    updated_ch["share"]        = edited_ch["Share (%)"].values
    st.session_state.channels_df = updated_ch

    st.divider()

    # ── F: Simulation & Berichtsmonat ─────────────────────────
    st.markdown('<p class="section-header">F · Simulation & Berichtsmonat</p>', unsafe_allow_html=True)
    cf1, cf2 = st.columns(2)
    with cf1:
        st.session_state.inp_n_sims = st.select_slider(
            "Monte Carlo Simulationen",
            [200, 500, 1_000, 2_000],
            value=st.session_state.inp_n_sims,
            help="Mehr Simulationen = präzisere Risikokorridore, aber etwas langsamer"
        )
    with cf2:
        st.session_state.inp_report_month = st.selectbox(
            "Berichtsmonat (für Plan vs. Actual)",
            MONTHS,
            index=MONTHS.index(st.session_state.inp_report_month),
            help="Bis zu welchem Monat wird YTD ausgewertet"
        )

    st.info(
        "💡 Alle Änderungen hier werden sofort in allen anderen Tabs übernommen.",
        icon="ℹ️"
    )


# ══════════════════════════════════════════════════════════════
# TAB: FUNNEL & BUDGET
# ══════════════════════════════════════════════════════════════
with tab1:
    col_funnel, col_budget = st.columns(2)

    with col_funnel:
        st.subheader("Reverse Funnel")
        f_labels = list(reversed(stage_names[1:]))
        f_values = list(reversed(funnel_vals[1:]))
        blue_gradient = ["#003580", "#004ea6", "#0066cc", "#1a8cff", "#4da6ff", "#80bfff"]
        fig_funnel = go.Figure(go.Funnel(
            y=f_labels, x=f_values,
            textinfo="value+percent initial",
            textfont=dict(size=12),
            marker=dict(color=blue_gradient),
            connector=dict(line=dict(color="#dee2e6", width=1, dash="dot")),
        ))
        fig_funnel.update_layout(
            height=430, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
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
            textposition="outside", width=0.45,
        ))
        fig_bars.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickformat=",.0f", title="€"), showlegend=False,
        )
        st.plotly_chart(fig_bars, use_container_width=True)

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
                "threshold": {"line": {"color": "#333", "width": 2},
                              "thickness": 0.75, "value": 100},
            },
        ))
        fig_gauge.update_layout(
            height=190, margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("Funnel Numbers")
    rows = []
    for name, val in zip(stage_names, funnel_vals):
        prefix = "€" if name == "Revenue" else ""
        rows.append({"Stage": name, "Annual": f"{prefix}{val:,.0f}",
                     "Per Month": f"{prefix}{val/12:,.1f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB: MONTE CARLO
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Risk Analysis — Monte Carlo")
    st.caption(f"**{n_sims} Simulationen** · PERT-Verteilungen aus Worst/Base/Best · Cost/MQL ±25 %")

    p10_b  = float(np.percentile(mc_df["budget_req"], 10))
    p50_b  = float(np.percentile(mc_df["budget_req"], 50))
    p90_b  = float(np.percentile(mc_df["budget_req"], 90))
    prob_ok = float((mc_df["budget_req"] <= available_budget).mean())

    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown("**Required Budget (€)**")
        st.metric("P10 – optimistisch",   f"€{p10_b:,.0f}")
        st.metric("Median",               f"€{p50_b:,.0f}")
        st.metric("P90 – konservativ",    f"€{float(np.percentile(mc_df['budget_req'], 90)):,.0f}")
    with cb:
        st.markdown(f"**{stage_names[4]} (MQLs)**")
        st.metric("P10", f"{float(np.percentile(mc_df['stage4'], 10)):,.0f}")
        st.metric("P50", f"{float(np.percentile(mc_df['stage4'], 50)):,.0f}")
        st.metric("P90", f"{float(np.percentile(mc_df['stage4'], 90)):,.0f}")
    with cc:
        st.markdown("**Budget-Ausreichend-Wahrscheinlichkeit**")
        st.metric("Wahrscheinlichkeit", f"{prob_ok:.0%}")
        if prob_ok >= 0.70:
            st.success(f"✅ In {prob_ok:.0%} der Szenarien reicht das Budget.")
        elif prob_ok >= 0.30:
            st.warning(f"⚠️ Budget reicht nur in {prob_ok:.0%} der Szenarien.")
        else:
            st.error(f"🔴 Budget zu knapp in {1-prob_ok:.0%} der Szenarien.")

    fig_hist = make_subplots(rows=1, cols=2,
        subplot_titles=["Required Budget", f"{stage_names[4]}"],
        horizontal_spacing=0.08)
    fig_hist.add_trace(go.Histogram(x=mc_df["budget_req"], nbinsx=40,
        marker_color="#0066cc", opacity=0.75, name="Budget"), row=1, col=1)
    fig_hist.add_vline(x=available_budget, line_dash="dash", line_color="#dc3545",
        annotation_text=f"Verfügbar €{available_budget:,.0f}", row=1, col=1)
    fig_hist.add_vline(x=p50_b, line_dash="dot", line_color="#28a745",
        annotation_text=f"Median €{p50_b:,.0f}", row=1, col=1)
    fig_hist.add_trace(go.Histogram(x=mc_df["stage4"], nbinsx=40,
        marker_color="#28a745", opacity=0.75, name="MQLs"), row=1, col=2)
    fig_hist.update_layout(height=340, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Risiko-Korridor")
    corridor_items = [("Required Budget (€)", "budget_req"),
                      (f"{stage_names[4]}", "stage4"),
                      (f"{stage_names[5]}", "stage5"), ("Deals", "deals")]
    fig_corridor = go.Figure()
    for label, col_name in corridor_items:
        p10c = float(np.percentile(mc_df[col_name], 10))
        p50c = float(np.percentile(mc_df[col_name], 50))
        p90c = float(np.percentile(mc_df[col_name], 90))
        fig_corridor.add_trace(go.Scatter(
            x=[p10c, p50c, p90c], y=[label, label, label],
            mode="lines+markers+text",
            line=dict(color="#0066cc", width=3),
            marker=dict(size=[8, 14, 8], color=["#adb5bd", "#0066cc", "#adb5bd"]),
            text=[f"P10: {p10c:,.0f}", f"P50: {p50c:,.0f}", f"P90: {p90c:,.0f}"],
            textposition=["bottom center", "top center", "bottom center"],
            textfont=dict(size=10), showlegend=False,
        ))
    fig_corridor.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_corridor, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB: MONTHLY PLAN
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Monthly Plan — {season_profile}")

    monthly_data = {
        stage_names[4]:   plan_mqls,
        stage_names[1]:   plan_deals,
        "Revenue (€)":    plan_revenue,
        "Budget (€)":     plan_budget,
    }
    metric_colors = {stage_names[4]: "#0066cc", stage_names[1]: "#28a745",
                     "Revenue (€)": "#fd7e14", "Budget (€)": "#dc3545"}

    fig_monthly = make_subplots(rows=2, cols=2,
        subplot_titles=list(monthly_data.keys()),
        vertical_spacing=0.18, horizontal_spacing=0.08)
    for i, (metric, values) in enumerate(monthly_data.items()):
        row, col = i // 2 + 1, i % 2 + 1
        avg = sum(values) / 12
        fig_monthly.add_trace(go.Bar(x=MONTHS, y=values,
            marker_color=metric_colors[metric], opacity=0.85, showlegend=False,
            hovertemplate=f"%{{x}}: %{{y:,.0f}}"), row=row, col=col)
        fig_monthly.add_hline(y=avg, line_dash="dot", line_color="#6c757d",
            annotation_text=f"⌀ {avg:,.0f}", annotation_position="top right",
            row=row, col=col)
    fig_monthly.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=60, t=40, b=0))
    st.plotly_chart(fig_monthly, use_container_width=True)

    table_rows = {"Monat": MONTHS}
    for metric, values in monthly_data.items():
        if "€" in metric:
            table_rows[metric] = [f"€{v:,.0f}" for v in values]
        elif metric == stage_names[1]:
            table_rows[metric] = [f"{v:.1f}" for v in values]
        else:
            table_rows[metric] = [f"{v:,.0f}" for v in values]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB: PLAN VS. ACTUAL
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Plan vs. Actual")
    st.caption("Trage deine monatlichen Ist-Werte ein. Oder lade Musterdaten zum Ausprobieren.")

    report_idx = MONTHS.index(st.session_state.inp_report_month)
    # Musterdaten: realistisches B2B-Jahr
    # Jan–Mär: langsamer Start, Apr–Jun: Normalisierung, Jul–Aug: Sommerloch,
    # Sep–Nov: starker Herbst, Dez: Jahresendspurt (Ø ≈ 100 % des Plans)
    SAMPLE_MULT = [0.88, 1.05, 1.16, 0.91, 0.97, 1.06,
                   0.80, 0.76, 1.14, 1.20, 1.10, 0.97]

    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        st.info(f"Berichtsmonat: **{st.session_state.inp_report_month}** (änderbar im Inputs-Tab → Abschnitt F)")
    with ctrl2:
        if st.button("📥 Musterdaten laden", use_container_width=True):
            st.session_state.actual_df = pd.DataFrame({
                "Month":              MONTHS,
                "Actual MQLs":        [round(plan_mqls[i]    * SAMPLE_MULT[i], 0) for i in range(12)],
                "Actual Deals":       [round(plan_deals[i]   * SAMPLE_MULT[i], 1) for i in range(12)],
                "Actual Revenue (€)": [round(plan_revenue[i] * SAMPLE_MULT[i], 0) for i in range(12)],
                "Actual Budget (€)":  [round(plan_budget[i]  * SAMPLE_MULT[i], 0) for i in range(12)],
            })
            st.session_state.inp_report_month = "Dec"
            st.rerun()
    with ctrl3:
        if st.button("🗑️ Daten löschen", use_container_width=True):
            st.session_state.actual_df = pd.DataFrame({
                "Month":              MONTHS,
                "Actual MQLs":        [0.0] * 12,
                "Actual Deals":       [0.0] * 12,
                "Actual Revenue (€)": [0.0] * 12,
                "Actual Budget (€)":  [0.0] * 12,
            })
            st.rerun()

    edited_actual = st.data_editor(
        st.session_state.actual_df, use_container_width=True,
        num_rows="fixed", hide_index=True,
        column_config={
            "Month":              st.column_config.TextColumn("Monat", disabled=True),
            "Actual MQLs":        st.column_config.NumberColumn("Ist MQLs",        min_value=0, step=1,    format="%.0f"),
            "Actual Deals":       st.column_config.NumberColumn("Ist Deals",       min_value=0, step=0.5,  format="%.1f"),
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

    def ytd_rate(actuals, plans, end_idx):
        a = sum(actuals[:end_idx + 1])
        p = sum(plans[:end_idx + 1])
        return (a / p) if p > 0 else 0.0

    def ytd_icon(rate):
        return "🟢" if rate >= 0.95 else ("🟡" if rate >= 0.75 else "🔴")

    r_mql  = ytd_rate(act_mqls,    plan_mqls,    report_idx)
    r_deal = ytd_rate(act_deals,   plan_deals,   report_idx)
    r_rev  = ytd_rate(act_revenue, plan_revenue, report_idx)
    r_bud  = ytd_rate(act_budget,  plan_budget,  report_idx)

    st.markdown(f"**YTD-Zielerreichung bis {st.session_state.inp_report_month}**")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{ytd_icon(r_mql)}  MQLs",         f"{r_mql:.0%}",
              delta=f"{sum(act_mqls[:report_idx+1]):,.0f} / {sum(plan_mqls[:report_idx+1]):,.0f}")
    k2.metric(f"{ytd_icon(r_deal)} Deals",         f"{r_deal:.0%}",
              delta=f"{sum(act_deals[:report_idx+1]):.1f} / {sum(plan_deals[:report_idx+1]):.1f}")
    k3.metric(f"{ytd_icon(r_rev)}  Revenue",       f"{r_rev:.0%}",
              delta=f"€{sum(act_revenue[:report_idx+1]):,.0f} / €{sum(plan_revenue[:report_idx+1]):,.0f}")
    k4.metric(f"{ytd_icon(r_bud)}  Budget Spend",  f"{r_bud:.0%}",
              delta=f"€{sum(act_budget[:report_idx+1]):,.0f} / €{sum(plan_budget[:report_idx+1]):,.0f}")

    st.divider()

    metrics_pva = [
        ("MQLs",        plan_mqls,    act_mqls,    "#0066cc", "#80bfff"),
        ("Deals",       plan_deals,   act_deals,   "#28a745", "#a3d9b1"),
        ("Revenue (€)", plan_revenue, act_revenue, "#fd7e14", "#ffd0a0"),
        ("Budget (€)",  plan_budget,  act_budget,  "#6f42c1", "#c9b8f0"),
    ]
    fig_pva = make_subplots(rows=2, cols=2,
        subplot_titles=[m[0] for m in metrics_pva],
        vertical_spacing=0.18, horizontal_spacing=0.08)
    for i, (label, plan_vals, act_vals, color_plan, color_act) in enumerate(metrics_pva):
        row, col = i // 2 + 1, i % 2 + 1
        act_visible = [v if j <= report_idx else None for j, v in enumerate(act_vals)]
        fig_pva.add_trace(go.Bar(x=MONTHS, y=plan_vals, name="Plan",
            marker_color=color_act, opacity=0.5, showlegend=(i == 0),
            legendgroup="plan"), row=row, col=col)
        fig_pva.add_trace(go.Bar(x=MONTHS, y=act_visible, name="Ist",
            marker_color=color_plan, opacity=0.9, showlegend=(i == 0),
            legendgroup="actual"), row=row, col=col)
        fig_pva.add_vline(x=st.session_state.inp_report_month,
            line_dash="dot", line_color="#6c757d", row=row, col=col)
    fig_pva.update_layout(barmode="overlay", height=540,
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_pva, use_container_width=True)

    st.subheader("Kumulativer Revenue-Verlauf")
    cum_plan = [sum(plan_revenue[:i+1]) for i in range(12)]
    cum_act  = [sum(act_revenue[:i+1]) if i <= report_idx else None for i in range(12)]
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=MONTHS, y=cum_plan, name="Plan kumuliert",
        line=dict(color="#adb5bd", width=2, dash="dot"), mode="lines+markers",
        marker=dict(size=6)))
    fig_cum.add_trace(go.Scatter(x=MONTHS[:report_idx+1], y=cum_act[:report_idx+1],
        name="Ist kumuliert", line=dict(color="#fd7e14", width=3),
        mode="lines+markers", marker=dict(size=8),
        fill="tonexty", fillcolor="rgba(253,126,20,0.08)"))
    fig_cum.add_hline(y=revenue_target, line_dash="dash", line_color="#0066cc",
        annotation_text=f"Jahresziel €{revenue_target:,.0f}",
        annotation_font_color="#0066cc")
    fig_cum.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(tickformat=",.0f", title="Revenue (€)"),
        legend=dict(orientation="h"))
    st.plotly_chart(fig_cum, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB: CHANNELS (Ergebnis-Ansicht)
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Channel-Ergebnisse")
    st.caption("Inputs (Cost per MQL, Share) änderst du im **⚙️ Inputs**-Tab → Abschnitt E.")

    ch_result = calc_channel_budget(stage4, st.session_state.channels_df)
    grp = (ch_result.groupby("group", as_index=False)
           .agg(planned_mqls=("planned_mqls","sum"), required_spend=("required_spend","sum"))
           .sort_values("required_spend", ascending=False))

    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig_pie = px.pie(grp, values="required_spend", names="group", hole=0.42,
            color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(height=370, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar = go.Figure(go.Bar(
            y=grp["group"], x=grp["required_spend"], orientation="h",
            marker_color="#0066cc",
            text=[f"€{v:,.0f}" for v in grp["required_spend"]],
            textposition="outside"))
        fig_bar.update_layout(height=370, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=90, t=10, b=0),
            xaxis=dict(title="Required Spend (€)", tickformat=",.0f"),
            showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    grp_d = grp.rename(columns={"group":"Kanal-Gruppe",
                                  "planned_mqls":"Geplante MQLs",
                                  "required_spend":"Required Spend (€)"})
    grp_d["Geplante MQLs"]      = grp_d["Geplante MQLs"].apply(lambda x: f"{x:,.0f}")
    grp_d["Required Spend (€)"] = grp_d["Required Spend (€)"].apply(lambda x: f"€{x:,.0f}")
    st.dataframe(grp_d, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Details pro Kanal")
    ch_detail = ch_result[["group","activity","planned_mqls","required_spend"]].copy()
    ch_detail.columns = ["Gruppe","Aktivität","Geplante MQLs","Spend (€)"]
    ch_detail["Geplante MQLs"] = ch_detail["Geplante MQLs"].apply(lambda x: f"{x:,.0f}")
    ch_detail["Spend (€)"]     = ch_detail["Spend (€)"].apply(lambda x: f"€{x:,.0f}")
    st.dataframe(ch_detail, use_container_width=True, hide_index=True)


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
