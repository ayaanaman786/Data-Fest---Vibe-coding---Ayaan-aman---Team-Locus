"""
Future Wallet — Streamlit Dashboard
Premium dark-themed financial projection visualization.
Enhanced with full multi-currency support (Spec 2.1).
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from engine import (
    run_scenarios, SimulationEngine, PetState, PET_EMOJI,
    CURRENCY_CONFIG, CurrencyEngine, AssetClass, LockType,
)

# ──────────────────────────────────────────────────────────────────────
# Page Config & Custom CSS
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Future Wallet | Financial Projection Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .hero-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0; letter-spacing: -0.02em;
    }
    .hero-subtitle { font-size: 1.1rem; color: #94a3b8; margin-top: 0; font-weight: 300; }

    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(100, 116, 139, 0.2); border-radius: 16px;
        padding: 24px; text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.15);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e2e8f0; margin: 8px 0 4px 0; }
    .metric-label {
        font-size: 0.85rem; color: #64748b; text-transform: uppercase;
        letter-spacing: 0.08em; font-weight: 500;
    }
    .metric-icon { font-size: 1.8rem; }

    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #e2e8f0;
        margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }

    .pet-badge {
        display: inline-block; padding: 8px 20px; border-radius: 50px;
        font-size: 1.2rem; font-weight: 600; letter-spacing: 0.02em;
    }
    .pet-chill { background: linear-gradient(135deg, #10b981, #059669); color: white; }
    .pet-content { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; }
    .pet-worried { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
    .pet-panic { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
    .pet-dead { background: linear-gradient(135deg, #6b7280, #374151); color: white; }

    .sidebar-header { font-size: 1.1rem; font-weight: 600; color: #cbd5e1; margin: 1rem 0 0.5rem 0; }

    .dag-box {
        background: linear-gradient(145deg, #1e1b4b 0%, #312e81 100%);
        border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 12px;
        padding: 16px; font-size: 0.85rem; color: #c4b5fd;
    }

    .precision-box {
        background: linear-gradient(145deg, #052e16 0%, #064e3b 100%);
        border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px;
        padding: 16px; font-size: 0.85rem; color: #6ee7b7;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 16px; padding: 16px 20px;
    }
</style>
""", unsafe_allow_html=True)

CURRENCY_OPTIONS = list(CURRENCY_CONFIG.keys())

# Helper to map pet state string to CSS class
def _pet_css(pet_str: str) -> str:
    for state in PetState:
        if state.value in pet_str:
            return {
                PetState.CHILL: "pet-chill", PetState.CONTENT: "pet-content",
                PetState.WORRIED: "pet-worried", PetState.PANIC: "pet-panic",
                PetState.DEAD: "pet-dead",
            }.get(state, "pet-content")
    return "pet-content"


# ──────────────────────────────────────────────────────────────────────
# Sidebar Inputs
# ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Simulation Parameters")
    st.markdown("---")

    st.markdown('<p class="sidebar-header">💵 Income & Savings</p>', unsafe_allow_html=True)
    initial_savings = st.number_input("Initial Savings", 0, 1_000_000, 20_000, step=1000)
    monthly_salary = st.number_input("Monthly Salary", 0, 500_000, 5_000, step=500)

    st.markdown('<p class="sidebar-header">💱 Currency Configuration</p>', unsafe_allow_html=True)
    home_currency = st.selectbox("Home Currency", CURRENCY_OPTIONS, index=0,
                                  help="Primary denomination for balances & reporting")
    salary_currency = st.selectbox("Salary Currency", CURRENCY_OPTIONS, index=0,
                                    help="Currency your salary is paid in")
    expense_currency = st.selectbox("Expense Currency", CURRENCY_OPTIONS, index=0,
                                     help="Currency your expenses are denominated in")

    st.markdown('<p class="sidebar-header">💸 Expenses & Debt</p>', unsafe_allow_html=True)
    monthly_expenses = st.number_input("Monthly Expenses", 0, 200_000, 3_000, step=500)
    initial_debt = st.number_input("Total Debt", 0, 1_000_000, 15_000, step=1000)
    monthly_debt_payment = st.number_input("Monthly Debt Payment", 0, 50_000, 500, step=100)

    st.markdown('<p class="sidebar-header">⚡ Risk & Timeline</p>', unsafe_allow_html=True)
    shock_probability = st.slider("Shock Probability", 0.0, 0.20, 0.02, 0.005,
                                   format="%.3f",
                                   help="Daily probability of a financial shock event")
    simulation_years = st.slider("Simulation Horizon (Years)", 1, 30, 5)
    seed = st.number_input("Random Seed (Determinism)", 0, 999999, 42, step=1)

    st.markdown("---")
    run_button = st.button("🚀 **Run Simulation**", use_container_width=True, type="primary")


# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="hero-title">Future Wallet</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">High-Fidelity Financial Projection & Simulation Engine — DataFest\'26</p>',
            unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Main Logic
# ──────────────────────────────────────────────────────────────────────

SCENARIO_COLORS = {
    "Optimistic": "#10b981", "Base Case": "#3b82f6", "Pessimistic": "#ef4444",
}
SCENARIO_FILLS = {
    "Optimistic": "rgba(16,185,129,0.08)", "Base Case": "rgba(59,130,246,0.08)",
    "Pessimistic": "rgba(239,68,68,0.08)",
}
SCENARIO_ICONS = {"Optimistic": "🟢", "Base Case": "🔵", "Pessimistic": "🔴"}

if run_button or "results" in st.session_state:
    if run_button:
        with st.spinner("⏳ Running deterministic simulation across 3 scenarios..."):
            results = run_scenarios(
                seed=seed,
                initial_savings=initial_savings,
                monthly_salary=monthly_salary,
                monthly_expenses=monthly_expenses,
                initial_debt=initial_debt,
                monthly_debt_payment=monthly_debt_payment,
                shock_probability=shock_probability,
                simulation_years=simulation_years,
                salary_currency=salary_currency,
                expense_currency=expense_currency,
                home_currency=home_currency,
            )
            st.session_state["results"] = results
            st.session_state["params"] = {
                "seed": seed, "home_currency": home_currency,
                "salary_currency": salary_currency, "expense_currency": expense_currency,
                "simulation_years": simulation_years,
            }
    else:
        results = st.session_state["results"]

    base = results["Base Case"]
    params = st.session_state.get("params", {})
    hc = params.get("home_currency", "USD")

    # ── DAG Execution Order ──
    with st.expander("🔗 DAG Dependency Resolution Order", expanded=False):
        engine_tmp = SimulationEngine(seed=seed)
        order = engine_tmp.execution_order
        st.markdown('<div class="dag-box">', unsafe_allow_html=True)
        st.markdown(f"**Topological Order:** {' → '.join(order)}")
        st.markdown("Each simulated day processes financial nodes in this exact order, "
                    "resolving dependencies before dependents.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Top Metric Cards ──
    st.markdown('<p class="section-header">📊 Base Case Summary</p>', unsafe_allow_html=True)

    pet_class = _pet_css(base["pet_state"])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Final Balance", f"{base['final_balance']:,.0f} {hc}")
    with c2:
        st.metric("Credit Score", f"{base['credit_score']:.0f}")
    with c3:
        st.metric("Collapse Prob", f"{base['collapse_probability']:.1%}")
    with c4:
        st.metric("NAV", f"{base['nav']:,.0f} {hc}")
    with c5:
        st.metric("Resilience (RSI)", f"{base['rsi']:.1f}")
    with c6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PET STATE</div>
            <div class="metric-value"><span class="pet-badge {pet_class}">{base['pet_state']}</span></div>
            <div class="metric-label">Vibe: {base['vibe']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    # Charts
    # ──────────────────────────────────────────────────────────────

    st.markdown('<p class="section-header">📈 Projected Wealth Trajectory</p>', unsafe_allow_html=True)

    fig_wealth = go.Figure()
    for name, res in results.items():
        days = np.arange(len(res["daily_balances"]))
        fig_wealth.add_trace(go.Scatter(
            x=days, y=res["daily_balances"],
            name=f"{SCENARIO_ICONS.get(name, '')} {name}",
            line=dict(color=SCENARIO_COLORS.get(name, "#8b5cf6"), width=2),
            fill="tozeroy",
            fillcolor=SCENARIO_FILLS.get(name, "rgba(139,92,246,0.05)"),
        ))
    fig_wealth.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=500, margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Day", yaxis_title=f"Cash Balance ({hc})",
        yaxis=dict(gridcolor="rgba(100,116,139,0.15)"),
        xaxis=dict(gridcolor="rgba(100,116,139,0.15)"),
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

    # ── Credit Score & NAV side by side ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<p class="section-header">🏦 Credit Score Evolution</p>', unsafe_allow_html=True)
        fig_credit = go.Figure()
        for name, res in results.items():
            days = np.arange(len(res["daily_credit_scores"]))
            fig_credit.add_trace(go.Scatter(
                x=days, y=res["daily_credit_scores"], name=name,
                line=dict(color=SCENARIO_COLORS.get(name), width=2),
            ))
        fig_credit.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=350, margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
            xaxis_title="Day", yaxis_title="Credit Score",
            yaxis=dict(range=[300, 850], gridcolor="rgba(100,116,139,0.15)"),
            xaxis=dict(gridcolor="rgba(100,116,139,0.15)"),
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_credit, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">💎 Net Asset Value (NAV)</p>', unsafe_allow_html=True)
        fig_nav = go.Figure()
        for name, res in results.items():
            days = np.arange(len(res["daily_nav"]))
            fig_nav.add_trace(go.Scatter(
                x=days, y=res["daily_nav"], name=name,
                line=dict(color=SCENARIO_COLORS.get(name), width=2),
            ))
        fig_nav.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=350, margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
            xaxis_title="Day", yaxis_title=f"NAV ({hc})",
            yaxis=dict(gridcolor="rgba(100,116,139,0.15)"),
            xaxis=dict(gridcolor="rgba(100,116,139,0.15)"),
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_nav, use_container_width=True)

    # ── Exchange Rate Chart (NEW: multi-currency visualization) ──
    st.markdown('<p class="section-header">💱 Exchange Rate Dynamics</p>', unsafe_allow_html=True)

    engine_fx = SimulationEngine(seed=params.get("seed", seed),
                                  simulation_years=params.get("simulation_years", simulation_years))
    rate_df = engine_fx.exchange_df
    fx_currencies = [c for c in rate_df.columns if c != "USD"]

    if fx_currencies:
        fx_tab1, fx_tab2 = st.tabs(["📈 Rate Trajectories", "📊 Volatility Heatmap"])

        with fx_tab1:
            fig_fx = go.Figure()
            fx_colors = ["#f59e0b", "#3b82f6", "#ef4444", "#10b981", "#8b5cf6"]
            for i, cur in enumerate(fx_currencies):
                # Normalize to base=100 for comparability
                series = rate_df[cur].values
                normalized = (series / series[0]) * 100
                fig_fx.add_trace(go.Scatter(
                    x=np.arange(len(series)), y=normalized,
                    name=f"USD/{cur}",
                    line=dict(color=fx_colors[i % len(fx_colors)], width=2),
                ))
            fig_fx.add_hline(y=100, line_dash="dash", line_color="rgba(148,163,184,0.3)")
            fig_fx.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=380, margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="Day", yaxis_title="Indexed Rate (Base=100)",
                yaxis=dict(gridcolor="rgba(100,116,139,0.15)"),
                xaxis=dict(gridcolor="rgba(100,116,139,0.15)"),
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_fx, use_container_width=True)

        with fx_tab2:
            # Rolling 30-day volatility heatmap
            vol_data = {}
            for cur in fx_currencies:
                returns = np.diff(np.log(rate_df[cur].values))
                rolling_vol = pd.Series(returns).rolling(30).std() * np.sqrt(252) * 100
                vol_data[cur] = rolling_vol.values
            vol_df = pd.DataFrame(vol_data)
            vol_df = vol_df.dropna()

            fig_heat = go.Figure(data=go.Heatmap(
                z=vol_df.values.T,
                x=np.arange(len(vol_df)),
                y=fx_currencies,
                colorscale="Inferno",
                colorbar=dict(title="Ann. Vol %"),
            ))
            fig_heat.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=250, margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Day", yaxis_title="Currency",
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # ── Precision Integrity Report (Spec 2.1 compliance) ──
    with st.expander("🔬 Currency Conversion Precision Report", expanded=False):
        st.markdown('<div class="precision-box">', unsafe_allow_html=True)
        st.markdown("**Floating-Point Integrity across High-Frequency Conversions**")
        st.markdown(f"- Total Conversions (Base Case): **{base.get('currency_conversions', 0):,}**")
        st.markdown(f"- Max Precision Drift: **{base.get('max_precision_drift', '0')}**")
        st.markdown(f"- Avg Precision Drift: **{base.get('avg_precision_drift', '0')}**")
        if base.get("wallet_balances"):
            st.markdown("**Final Wallet Balances:**")
            for cur, bal in base["wallet_balances"].items():
                st.markdown(f"  - {cur}: **{bal:,.4f}**")
        if base.get("final_exchange_rates"):
            st.markdown("**Final Day Exchange Rates (vs USD):**")
            for cur, rate in base["final_exchange_rates"].items():
                if cur != "USD":
                    st.markdown(f"  - USD/{cur}: **{rate:.6f}**")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Asset Portfolio Dashboard (Spec 2.2) ──
    st.markdown('<p class="section-header">💼 Asset Portfolio & Liquidity Engine</p>', unsafe_allow_html=True)

    pb = base.get("portfolio_breakdown", {})
    if pb:
        port_tab1, port_tab2, port_tab3 = st.tabs(
            ["📊 Portfolio Breakdown", "📋 Asset Details", "🔻 Liquidation Log"]
        )

        with port_tab1:
            # Sunburst / Treemap of portfolio by class
            treemap_labels, treemap_values, treemap_parents, treemap_colors = [], [], [], []
            class_colors = {
                "LIQUID": "#10b981", "YIELD": "#3b82f6",
                "VOLATILE": "#f59e0b", "ILLIQUID": "#8b5cf6",
            }
            for cls_name, info in pb.items():
                if info["count"] > 0:
                    treemap_labels.append(cls_name)
                    treemap_values.append(info["total_value"])
                    treemap_parents.append("")
                    treemap_colors.append(class_colors.get(cls_name, "#64748b"))
                    for a in info["assets"]:
                        treemap_labels.append(a["name"])
                        treemap_values.append(a["value"])
                        treemap_parents.append(cls_name)
                        treemap_colors.append(class_colors.get(cls_name, "#64748b"))

            fig_tree = go.Figure(go.Treemap(
                labels=treemap_labels, values=treemap_values,
                parents=treemap_parents,
                marker=dict(colors=treemap_colors, line=dict(color="#0f172a", width=2)),
                textinfo="label+value+percent parent",
                textfont=dict(family="Inter, sans-serif"),
            ))
            fig_tree.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=400, margin=dict(l=10, r=10, t=10, b=10),
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_tree, use_container_width=True)

            # Summary metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                total_pv = sum(info["total_value"] for info in pb.values())
                st.metric("Total Portfolio", f"{total_pv:,.0f} {hc}")
            with m2:
                st.metric("Yield Earned", f"{base.get('total_yield_earned', 0):,.0f} {hc}")
            with m3:
                liq = base.get("liquidation_summary", {})
                st.metric("Liquidation Events", f"{liq.get('total_events', 0)}")
            with m4:
                st.metric("Total Liquidated", f"{base.get('total_liquidated_value', 0):,.0f} {hc}")

        with port_tab2:
            # Detailed asset table
            asset_rows = []
            for cls_name, info in pb.items():
                for a in info["assets"]:
                    asset_rows.append({
                        "Asset": a["name"],
                        "Class": cls_name,
                        f"Value ({a['currency']})": f"{a['value']:,.2f}",
                        f"Cost Basis": f"{a['cost_basis']:,.2f}",
                        "Unrealized G/L": f"{a['unrealized_gain']:,.2f}",
                        "Yield Accrued": f"{a['accrued_yield']:,.2f}",
                        "Lock": a["lock_type"],
                        "Liquidations": a["liquidation_events"],
                    })
            if asset_rows:
                st.dataframe(pd.DataFrame(asset_rows), use_container_width=True, hide_index=True)

        with port_tab3:
            liq = base.get("liquidation_summary", {})
            if liq.get("total_events", 0) > 0:
                st.markdown(f"**Total Events:** {liq['total_events']} | "
                            f"**Proceeds:** {liq['total_proceeds']:,.0f} {hc} | "
                            f"**Penalties:** {liq['total_penalties']:,.0f} {hc} | "
                            f"**Lock Overrides:** {liq['lock_overrides']}")
                for cls, cinfo in liq.get("by_class", {}).items():
                    st.markdown(f"- **{cls}**: {cinfo['events']} events, "
                                f"{cinfo['proceeds']:,.0f} {hc} proceeds")
            else:
                st.success("No liquidation events triggered during this simulation. "
                           "Portfolio maintained positive liquidity throughout.")

    # ── Credit Evolution (Spec 2.3) ──
    st.markdown('<p class="section-header">📊 Credit Evolution & Scoring</p>', unsafe_allow_html=True)

    credit_factors = base.get("credit_factors", [])
    if credit_factors:
        cred_tab1, cred_tab2 = st.tabs(["📈 Score Trajectory", "🔍 Credit Factors"])

        with cred_tab1:
            fig_credit = go.Figure()
            cf_days = [cf["day"] for cf in credit_factors]
            cf_scores = [cf["score"] for cf in credit_factors]
            fig_credit.add_trace(go.Scatter(
                x=cf_days, y=cf_scores, mode="lines+markers",
                fill="tozeroy",
                line=dict(color="#10b981", width=2),
                marker=dict(size=4),
                name="Credit Score",
            ))
            # Reference lines
            fig_credit.add_hline(y=750, line_dash="dot", line_color="#3b82f6",
                                 annotation_text="Excellent (750+)")
            fig_credit.add_hline(y=650, line_dash="dot", line_color="#f59e0b",
                                 annotation_text="Good (650+)")
            fig_credit.add_hline(y=500, line_dash="dot", line_color="#ef4444",
                                 annotation_text="Fair (500+)")
            fig_credit.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Day", yaxis_title="Credit Score",
                yaxis=dict(range=[300, 850]),
                height=350, margin=dict(l=50, r=20, t=30, b=40),
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_credit, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Final Score", f"{base.get('credit_score', 650):.0f}")
            with c2:
                st.metric("On-Time Streak", f"{base.get('consecutive_on_time', 0)}")
            with c3:
                st.metric("Credit Utilization",
                          f"{base.get('credit_utilization', 0):.1%}")
            with c4:
                st.metric("Restructuring Events",
                          f"{base.get('restructuring_events', 0)}")

        with cred_tab2:
            factor_rows = []
            for cf in credit_factors:
                factor_rows.append({
                    "Day": cf["day"],
                    "Score": cf["score"],
                    "Punctuality": f"{cf['punctuality']:.2%}",
                    "Debt Ratio": f"{cf['debt_ratio']:.4f}",
                    "Utilization": f"{cf['utilization']:.2%}",
                    "On-Time Streak": cf["streak"],
                    "f(x)": f"{cf['f_val']:.6f}",
                })
            if factor_rows:
                st.dataframe(pd.DataFrame(factor_rows),
                             use_container_width=True, hide_index=True)
    else:
        st.info("No credit factor history available for this simulation.")

    # ── Taxation Breakdown (Spec 2.3) ──
    st.markdown('<p class="section-header">🏦 Taxation Breakdown</p>', unsafe_allow_html=True)

    tax_records = base.get("tax_records", [])
    if tax_records:
        tax_tab1, tax_tab2 = st.tabs(["📊 Annual Tax Summary", "📋 Detailed Breakdown"])

        with tax_tab1:
            years = [f"Year {tr['year']}" for tr in tax_records]
            fig_tax = go.Figure()
            fig_tax.add_trace(go.Bar(
                x=years,
                y=[tr["income_tax"] for tr in tax_records],
                name="Income Tax", marker_color="#3b82f6",
            ))
            fig_tax.add_trace(go.Bar(
                x=years,
                y=[tr["short_term_tax"] for tr in tax_records],
                name="Short-Term CG Tax", marker_color="#f59e0b",
            ))
            fig_tax.add_trace(go.Bar(
                x=years,
                y=[tr["long_term_tax"] for tr in tax_records],
                name="Long-Term CG Tax", marker_color="#10b981",
            ))
            fig_tax.add_trace(go.Bar(
                x=years,
                y=[tr["fx_tax"] for tr in tax_records],
                name="FX Tax", marker_color="#8b5cf6",
            ))
            fig_tax.update_layout(
                barmode="stack",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Tax Year", yaxis_title=f"Tax ({hc})",
                height=350, margin=dict(l=50, r=20, t=30, b=40),
                font=dict(family="Inter, sans-serif"),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_tax, use_container_width=True)

            t1, t2, t3 = st.columns(3)
            with t1:
                st.metric("Total Tax Paid", f"{base.get('tax_paid', 0):,.0f} {hc}")
            with t2:
                st.metric("Unrealized Gains",
                          f"{base.get('total_unrealized_gains', 0):,.0f} {hc}")
            with t3:
                eff_rate = base.get("tax_paid", 0) / max(
                    sum(tr["ordinary_income"] for tr in tax_records), 1)
                st.metric("Effective Rate", f"{eff_rate:.1%}")

        with tax_tab2:
            tax_rows = []
            for tr in tax_records:
                tax_rows.append({
                    "Year": tr["year"],
                    f"Income ({hc})": f"{tr['ordinary_income']:,.0f}",
                    f"ST Gains": f"{tr['short_term_gains']:,.0f}",
                    f"LT Gains": f"{tr['long_term_gains']:,.0f}",
                    f"FX Gains": f"{tr['fx_gains']:,.0f}",
                    f"Unrealized": f"{tr['unrealized_gains']:,.0f}",
                    f"Income Tax": f"{tr['income_tax']:,.0f}",
                    f"ST Tax": f"{tr['short_term_tax']:,.0f}",
                    f"LT Tax": f"{tr['long_term_tax']:,.0f}",
                    f"FX Tax": f"{tr['fx_tax']:,.0f}",
                    f"Total Tax": f"{tr['total_tax']:,.0f}",
                })
            st.dataframe(pd.DataFrame(tax_rows),
                         use_container_width=True, hide_index=True)
    else:
        st.info("No tax records generated for this simulation period.")

    # ── DAG Structure & Rolling Metrics (Spec 3.1 / 3.2) ──
    st.markdown('<p class="section-header">🔗 Dependency Graph & Behavioral Metrics</p>',
                unsafe_allow_html=True)

    dag_col, metrics_col = st.columns([1, 1])

    with dag_col:
        st.markdown("##### 🧩 Component DAG")
        dag_info = base.get("dag_structure", {})
        if dag_info:
            nodes = dag_info.get("nodes", [])
            edges = dag_info.get("edges", [])
            order = dag_info.get("order", [])

            node_icons = {
                "EXCHANGE": "💱", "INCOME": "💰", "EXPENSE": "💸",
                "TAX": "🏦", "ASSET_LIQUID": "📊", "ASSET_ILLIQUID": "🏠",
                "CREDIT": "📋",
            }

            for i, name in enumerate(order):
                node_info = next((n for n in nodes if n["name"] == name), None)
                if node_info:
                    icon = node_icons.get(node_info["type"], "⬡")
                    deps = node_info.get("deps", [])
                    dep_str = f" ← {', '.join(deps)}" if deps else ""
                    st.markdown(f"{'│ ' * i}├─ {icon} **{name}** `{node_info['type']}`{dep_str}")

            st.caption(f"v{dag_info.get('version', 0)} · "
                       f"{len(nodes)} nodes · {len(edges)} edges · "
                       f"{dag_info.get('changes', 0)} structural changes")

    with metrics_col:
        st.markdown("##### 📡 Rolling Behavioral Metrics")

        m1, m2 = st.columns(2)
        with m1:
            scd = base.get("rolling_scd", 0)
            scd_color = "🟢" if scd < 0.3 else "🟡" if scd < 0.6 else "🔴"
            st.metric(f"{scd_color} Shock Clustering", f"{scd:.3f}")

            recovery = base.get("rolling_recovery", 1.0)
            rec_color = "🟢" if recovery > 0.5 else "🟡" if recovery > 0 else "🔴"
            st.metric(f"{rec_color} Recovery Slope", f"{recovery:.4f}")

        with m2:
            vibe = base.get("rolling_vibe", "Neutral")
            pet = base.get("rolling_pet", "Content")
            vibe_icons = {"Chill": "😎", "Stable": "😃", "Uneasy": "😟",
                          "Stressed": "😰", "Critical": "💀"}
            st.metric(f"{vibe_icons.get(vibe, '🤖')} Vibe", vibe)
            st.metric(f"🐾 Pet State", pet)

        st.caption("Metrics update every 7 days and influence shock sensitivity + expense behavior")

    # ── What-If Branch Analysis (Spec 3.3) ──
    st.markdown('<p class="section-header">🌿 What-If Branch Analysis</p>',
                unsafe_allow_html=True)

    with st.expander("🔮 Configure & Run Branch Scenarios", expanded=False):
        st.markdown("Run hypothetical scenarios from the **current base case** simulation.")

        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            br_salary = st.number_input("Optimistic Salary", value=7000, step=500,
                                         key="br_salary")
        with bcol2:
            br_expenses = st.number_input("Pessimistic Expenses", value=5000, step=500,
                                           key="br_expenses")
        with bcol3:
            br_shock = st.slider("Pessimistic Shock %", 0.0, 0.15, 0.05, 0.01,
                                  key="br_shock")

        if st.button("🚀 Run Branches", key="run_branches"):
            with st.spinner("Branching simulations..."):
                from engine import SimulationEngine as SE
                branch_eng = SE(
                    seed=sidebar_seed,
                    initial_savings=sidebar_savings,
                    monthly_salary=sidebar_salary,
                    monthly_expenses=sidebar_expenses,
                    initial_debt=sidebar_debt,
                    monthly_debt_payment=sidebar_debt_payment,
                    shock_probability=sidebar_shock,
                    simulation_years=sidebar_years,
                    salary_currency=sidebar_salary_cur,
                    expense_currency=sidebar_expense_cur,
                    home_currency=sidebar_home_cur,
                )
                # Run to halfway
                midpoint = branch_eng.n_days // 2
                for d in range(midpoint):
                    branch_eng._step(d)

                branches = branch_eng.branch_scenarios({
                    "Optimistic": {"monthly_salary": br_salary, "shock_probability": 0.01},
                    "Pessimistic": {"monthly_expenses": br_expenses,
                                    "shock_probability": br_shock},
                    "Baseline Continue": {},
                })

                merged = SE.merge_branches(branches)

                # Trajectory chart
                fig_branch = go.Figure()
                branch_colors = {"Optimistic": "#10b981", "Pessimistic": "#ef4444",
                                 "Baseline Continue": "#3b82f6"}
                for bname, bres in branches.items():
                    fig_branch.add_trace(go.Scatter(
                        y=bres["daily_balances"], mode="lines",
                        name=bname,
                        line=dict(color=branch_colors.get(bname, "#888"), width=2),
                    ))
                fig_branch.add_vline(x=midpoint, line_dash="dash", line_color="#888",
                                     annotation_text="Branch Point")
                fig_branch.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Day", yaxis_title=f"Balance ({hc})",
                    height=350, margin=dict(l=50, r=20, t=30, b=40),
                    font=dict(family="Inter, sans-serif"),
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig_branch, use_container_width=True)

                # Comparison table
                comp_rows = []
                for metric, mdata in merged["comparison"].items():
                    row = {"Metric": metric.replace("_", " ").title()}
                    for bname in merged["branch_names"]:
                        val = mdata["per_branch"].get(bname, 0)
                        row[bname] = f"{val:,.2f}" if isinstance(val, float) else str(val)
                    row["Best"] = mdata.get("best_branch", "")
                    comp_rows.append(row)
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True,
                             hide_index=True)

                # Recommendation
                rec = merged.get("recommendation", "")
                if rec:
                    st.success(f"🏆 **Recommended scenario: {rec}** "
                               f"(highest projected NAV)")


    st.markdown('<p class="section-header">📉 Recovery Slope Analysis</p>', unsafe_allow_html=True)

    fig_recovery = go.Figure()
    for name, res in results.items():
        balances = np.array(res["daily_balances"])
        window = 30
        if len(balances) > window:
            slopes = []
            for i in range(window, len(balances)):
                segment = balances[i - window:i]
                x = np.arange(window)
                slope = np.polyfit(x, segment, 1)[0]
                slopes.append(slope)
            days = np.arange(window, len(balances))
            fig_recovery.add_trace(go.Scatter(
                x=days, y=slopes, name=f"{SCENARIO_ICONS.get(name, '')} {name}",
                line=dict(color=SCENARIO_COLORS.get(name), width=2),
            ))
    fig_recovery.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,0.4)")
    fig_recovery.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=350, margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Day", yaxis_title=f"30-Day Rolling Slope ({hc}/day)",
        yaxis=dict(gridcolor="rgba(100,116,139,0.15)"),
        xaxis=dict(gridcolor="rgba(100,116,139,0.15)"),
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig_recovery, use_container_width=True)

    # ──────────────────────────────────────────────────────────────
    # Comparison Table
    # ──────────────────────────────────────────────────────────────

    st.markdown('<p class="section-header">📋 Scenario Comparison</p>', unsafe_allow_html=True)

    table_data = []
    for name, res in results.items():
        table_data.append({
            "Scenario": f"{SCENARIO_ICONS.get(name, '')} {name}",
            f"Final Balance ({hc})": f"{res['final_balance']:,.0f}",
            f"Balance 5th %": f"{res['balance_5th']:,.0f}",
            f"Balance 95th %": f"{res['balance_95th']:,.0f}",
            "Collapse Prob": f"{res['collapse_probability']:.1%}",
            "Credit Score": f"{res['credit_score']:.0f}",
            f"NAV ({hc})": f"{res['nav']:,.0f}",
            "Liquidity Ratio": f"{res['liquidity_ratio']:.2%}",
            "Shock Resilience": f"{res['rsi']:.1f}",
            "Shock Clustering": f"{res['shock_clustering_density']:.3f}",
            "Pet State": res["pet_state"],
            "Vibe": res["vibe"],
            f"Tax Paid ({hc})": f"{res['tax_paid']:,.0f}",
            f"Debt Remain ({hc})": f"{res['total_debt_remaining']:,.0f}",
            "Recovery Slope": f"{res['recovery_slope']:.2f}",
            "FX Conversions": f"{res.get('currency_conversions', 0):,}",
        })

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True,
                 column_config={"Scenario": st.column_config.TextColumn(width="medium")})

    # ── Exchange Rate Table (sample) ──
    with st.expander("💱 Exchange Rate Sample (First 30 Days)", expanded=False):
        st.dataframe(rate_df.head(30).round(6), use_container_width=True)

    # ── What-If Branching Demo ──
    st.markdown('<p class="section-header">🔀 What-If Branching</p>', unsafe_allow_html=True)
    st.markdown("Snapshot the simulation at a specific day and explore alternative futures.")

    branch_day = st.slider("Branch at Day", 30, max(base["n_days"] - 30, 31),
                           min(365, base["n_days"] // 2))

    if st.button("🌿 Run Branch Simulation", use_container_width=True):
        with st.spinner("Branching simulation..."):
            engine_base = SimulationEngine(
                seed=seed, initial_savings=initial_savings,
                monthly_salary=monthly_salary, monthly_expenses=monthly_expenses,
                initial_debt=initial_debt, monthly_debt_payment=monthly_debt_payment,
                shock_probability=shock_probability, simulation_years=simulation_years,
                salary_currency=salary_currency, expense_currency=expense_currency,
                home_currency=home_currency,
            )
            for d in range(branch_day):
                engine_base._step(d)
            snapshot = engine_base.get_snapshot()

            engine_a = SimulationEngine(
                seed=seed + 100, initial_savings=initial_savings,
                monthly_salary=monthly_salary, monthly_expenses=monthly_expenses,
                initial_debt=initial_debt, shock_probability=shock_probability * 3,
                simulation_years=simulation_years,
                salary_currency=salary_currency, expense_currency=expense_currency,
                home_currency=home_currency,
            )
            result_a = engine_a.run_from_snapshot(snapshot.snapshot(), base["n_days"] - branch_day)

            engine_b = SimulationEngine(
                seed=seed + 200, initial_savings=initial_savings,
                monthly_salary=monthly_salary * 1.5, monthly_expenses=monthly_expenses,
                initial_debt=initial_debt, shock_probability=shock_probability * 0.5,
                simulation_years=simulation_years,
                salary_currency=salary_currency, expense_currency=expense_currency,
                home_currency=home_currency,
            )
            result_b = engine_b.run_from_snapshot(snapshot.snapshot(), base["n_days"] - branch_day)

        fig_branch = go.Figure()
        fig_branch.add_trace(go.Scatter(
            x=list(range(base["n_days"])), y=base["daily_balances"],
            name="Original", line=dict(color="#3b82f6", width=2),
        ))
        branch_x = list(range(branch_day, branch_day + len(result_a["daily_balances"])))
        fig_branch.add_trace(go.Scatter(
            x=branch_x, y=result_a["daily_balances"],
            name="Branch A: High Shock", line=dict(color="#ef4444", width=2, dash="dash"),
        ))
        branch_x_b = list(range(branch_day, branch_day + len(result_b["daily_balances"])))
        fig_branch.add_trace(go.Scatter(
            x=branch_x_b, y=result_b["daily_balances"],
            name="Branch B: Salary Boost", line=dict(color="#10b981", width=2, dash="dash"),
        ))
        fig_branch.add_vline(x=branch_day, line_dash="dot", line_color="#f59e0b",
                             annotation_text="Branch Point")
        fig_branch.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=450, margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Day", yaxis_title=f"Cash Balance ({hc})",
            yaxis=dict(gridcolor="rgba(100,116,139,0.15)"),
            xaxis=dict(gridcolor="rgba(100,116,139,0.15)"),
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_branch, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Branch A: High Shock** 🔴")
            st.metric("Final Balance", f"{result_a['final_balance']:,.0f} {hc}")
            st.metric("Credit Score", f"{result_a['credit_score']:.0f}")
            st.markdown(f"Pet: {result_a['pet_state']}")
        with col2:
            st.markdown("**Branch B: Salary Boost** 🟢")
            st.metric("Final Balance", f"{result_b['final_balance']:,.0f} {hc}")
            st.metric("Credit Score", f"{result_b['credit_score']:.0f}")
            st.markdown(f"Pet: {result_b['pet_state']}")

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#64748b; font-size:0.8rem;">'
        'Future Wallet v2.0 — DataFest\'26 — Deterministic Financial Projection Engine — Multi-Currency Enabled'
        '</p>', unsafe_allow_html=True,
    )

else:
    # Landing state
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">DAG</div>
            <div class="metric-label">Dependency Resolution</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🔢</div>
            <div class="metric-value">Daily</div>
            <div class="metric-label">Granularity</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">💱</div>
            <div class="metric-value">6 FX</div>
            <div class="metric-label">Multi-Currency</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🔀</div>
            <div class="metric-value">Branch</div>
            <div class="metric-label">What-If Scenarios</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info("👈 **Configure your parameters in the sidebar and click 'Run Simulation'** to start the deterministic projection engine.")

    with st.expander("📐 Architecture Overview"):
        st.markdown("""
        **Future Wallet** processes financial components as nodes in a **Directed Acyclic Graph (DAG)**:

        ```
        Exchange Rates → Income → Debt Service ──┐
                       → Expenses                │→ Credit Score Update
                       → Asset Valuation → Tax ──┘→ Liquidation Check
        ```

        **Multi-Currency Engine (Spec 2.1):**
        - **6 currencies** (USD, EUR, GBP, JPY, PKR, CHF) with correlated exchange rates
        - Ornstein-Uhlenbeck mean-reversion + Geometric Brownian Motion
        - Cholesky-decomposed correlation matrix for realistic co-movement
        - **Decimal-precision** conversions with round-trip integrity verification
        - Full conversion audit trail for compliance validation
        - Transaction-time realization: conversion at exact daily rate
        """)
