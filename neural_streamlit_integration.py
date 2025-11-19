# streamlit_app.py
# Frontend UI for the Market-Learning Intuition Engine

import streamlit as st
from datetime import date

from market_learning_engine import (
    FundamentalSnapshot,
    TechnicalSnapshot,
    MarketContext,
    IntuitionInputs,
    score_stock,
    MarketOutcomeLearner,
)


# ================================
# Session init
# ================================
def get_learner() -> MarketOutcomeLearner:
    if "learner" not in st.session_state:
        # You can change the history path if you want
        st.session_state["learner"] = MarketOutcomeLearner(
            history_path="prediction_history.csv"
        )
    return st.session_state["learner"]


# ================================
# UI Helper Functions
# ================================
def render_fundamental_inputs() -> FundamentalSnapshot:
    st.subheader("Fundamentals")

    col1, col2, col3 = st.columns(3)

    with col1:
        revenue_cagr_3y = st.number_input(
            "Revenue CAGR (3y, %)",
            value=15.0,
            help="Annualized revenue growth over the last 3 years (percent).",
        ) / 100.0

        gross_margin_trend_pp = st.number_input(
            "Gross Margin Trend (pp over 3y)",
            value=3.0,
            help="Change in gross margin in percentage points over ~3 years.",
        )

        ebit_margin_trend_pp = st.number_input(
            "EBIT Margin Trend (pp over 3y)",
            value=1.5,
        )

        fcf_margin = st.number_input(
            "FCF Margin (%)",
            value=14.0,
            help="Free cash flow margin as % of revenue.",
        ) / 100.0

    with col2:
        roe = st.number_input(
            "ROE (%)",
            value=18.0,
            help="Return on equity, percent.",
        ) / 100.0

        net_debt_to_ebitda = st.number_input(
            "Net Debt / EBITDA",
            value=1.2,
            help="Leverage multiple.",
        )

        share_count_cagr = st.number_input(
            "Share Count CAGR (%, negative = buybacks)",
            value=-1.5,
        ) / 100.0

        dso_trend_days = st.number_input(
            "DSO Trend (days over 3y)",
            value=0.5,
            help="Positive = collections getting slower.",
        )

    with col3:
        inventory_trend_days = st.number_input(
            "Inventory Trend (days over 3y)",
            value=-1.0,
        )

        insider_ownership_change_pct = st.number_input(
            "Insider Ownership Change (%, last 12–24m)",
            value=2.0,
        ) / 100.0

        founder_led = st.checkbox("Founder-led?", value=True)

        accounting_red_flags = st.number_input(
            "Number of Accounting Red Flags",
            value=0,
            min_value=0,
            step=1,
        )

    fundamentals = FundamentalSnapshot(
        revenue_cagr_3y=revenue_cagr_3y,
        gross_margin_trend_pp=gross_margin_trend_pp,
        ebit_margin_trend_pp=ebit_margin_trend_pp,
        fcf_margin=fcf_margin,
        roe=roe,
        net_debt_to_ebitda=net_debt_to_ebitda,
        share_count_cagr=share_count_cagr,
        dso_trend_days=dso_trend_days,
        inventory_trend_days=inventory_trend_days,
        insider_ownership_change_pct=insider_ownership_change_pct,
        founder_led=founder_led,
        accounting_red_flags=accounting_red_flags,
    )
    return fundamentals


def render_technical_inputs() -> TechnicalSnapshot:
    st.subheader("Technicals & Flow")

    col1, col2, col3 = st.columns(3)

    with col1:
        rel_strength_3m = (
            st.number_input(
                "Relative Strength 3M (vs benchmark, %)",
                value=10.0,
                help="Example: 10 means stock outperformed benchmark by 10% over 3 months.",
            )
            / 100.0
            + 1.0
        )

        rel_strength_12m = (
            st.number_input(
                "Relative Strength 12M (vs benchmark, %)",
                value=25.0,
            )
            / 100.0
            + 1.0
        )

    with col2:
        volatility_3m = st.number_input(
            "Volatility 3M (annualized, %)",
            value=24.0,
            help="Approx annualized volatility based on recent history.",
        ) / 100.0

        avg_daily_dollar_volume = st.number_input(
            "Avg Daily Dollar Volume ($)",
            value=8_000_000,
            step=250_000,
        )

    with col3:
        drawdown_from_high_pct = st.number_input(
            "Drawdown from High (%)",
            value=-18.0,
            help="Negative number means below peak; -18 = 18% off high.",
        ) / 100.0

        price_above_200d_ma_pct = st.number_input(
            "Price vs 200d MA (%)",
            value=6.0,
            help="Positive means above 200d MA.",
        ) / 100.0

    technicals = TechnicalSnapshot(
        rel_strength_3m=rel_strength_3m,
        rel_strength_12m=rel_strength_12m,
        volatility_3m=volatility_3m,
        avg_daily_dollar_volume=avg_daily_dollar_volume,
        drawdown_from_high_pct=drawdown_from_high_pct,
        price_above_200d_ma_pct=price_above_200d_ma_pct,
    )
    return technicals


def render_market_context_inputs() -> MarketContext:
    st.subheader("Market Context")

    col1, col2, col3 = st.columns(3)

    with col1:
        market_regime = st.selectbox(
            "Market Regime",
            ["NEUTRAL", "RISK_ON", "RISK_OFF", "VOLATILE"],
        )

    with col2:
        sector_tailwind_score = st.slider(
            "Sector Tailwind (0–1)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
        )

    with col3:
        macro_risk_score = st.slider(
            "Macro Risk (0–1, higher = more risk)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )

    ctx = MarketContext(
        market_regime=market_regime,
        sector_tailwind_score=sector_tailwind_score,
        macro_risk_score=macro_risk_score,
    )
    return ctx


def render_intuition_inputs() -> IntuitionInputs:
    st.subheader("Human Intuition Layer")

    col1, col2 = st.columns(2)

    with col1:
        analyst_conviction = st.slider(
            "Analyst Conviction (0–10)",
            0,
            10,
            8,
        )
        narrative_clarity = st.slider(
            "Narrative Clarity (0–10)",
            0,
            10,
            9,
        )
        management_quality = st.slider(
            "Management Quality (0–10)",
            0,
            10,
            8,
        )

    with col2:
        business_understandability = st.slider(
            "Business Understandability (0–10)",
            0,
            10,
            9,
        )

        green_flags_str = st.text_input(
            "Green Flags (comma-separated)",
            value="network_effects, recurring_revenue",
            help="Example: 'network_effects, recurring_revenue'",
        )
        red_flags_str = st.text_input(
            "Red Flags (comma-separated)",
            value="customer_concentration",
            help="Example: 'customer_concentration, regulatory_overhang'",
        )

        analyst_override = st.slider(
            "Analyst Override (-5 to +5)",
            -5.0,
            5.0,
            0.0,
            step=0.5,
            help="Manual adjustment on top of intuition score.",
        )

    green_flags = [s.strip() for s in green_flags_str.split(",") if s.strip()]
    red_flags = [s.strip() for s in red_flags_str.split(",") if s.strip()]

    intuition = IntuitionInputs(
        analyst_conviction=analyst_conviction,
        narrative_clarity=narrative_clarity,
        management_quality=management_quality,
        business_understandability=business_understandability,
        green_flags=green_flags,
        red_flags=red_flags,
        analyst_override=analyst_override,
    )
    return intuition


# ================================
# Main app
# ================================
def main():
    st.set_page_config(
        page_title="Market-Learning Intuition Engine",
        layout="wide",
    )

    st.title("📈 Market-Learning Intuition Engine")
    st.caption(
        "Combines fundamentals, technicals, macro context, and human intuition — "
        "then learns from realized outcomes over time."
    )

    learner = get_learner()

    tab_score, tab_train = st.tabs(["🔍 Score Stock", "🧠 Train on Outcomes"])

    # -------------- SCORE TAB --------------
    with tab_score:
        st.subheader("Stock Setup")

        col_top = st.columns(3)
        with col_top[0]:
            ticker = st.text_input(
                "Ticker",
                value="TEST",
            ).upper()
        with col_top[1]:
            as_of_date = st.date_input(
                "As-of Date",
                value=date.today(),
            )
        with col_top[2]:
            horizon_days = st.number_input(
                "Target Horizon (days) – informational",
                value=90,
                min_value=1,
                step=1,
                help="Used only as a reference here; training horizon is set on the Train tab.",
            )

        st.markdown("---")

        with st.expander("Fundamentals", expanded=True):
            fundamentals = render_fundamental_inputs()

        with st.expander("Technicals & Flow", expanded=False):
            technicals = render_technical_inputs()

        with st.expander("Market Context", expanded=False):
            market_context = render_market_context_inputs()

        with st.expander("Human Intuition", expanded=False):
            intuition = render_intuition_inputs()

        st.markdown("---")
        col_btn1, col_btn2 = st.columns([1, 1])

        with col_btn1:
            run_base = st.button("Run Base Rule-Based Score", use_container_width=True)
        with col_btn2:
            run_learned = st.button(
                "Run Score with Learning (if trained)",
                use_container_width=True,
            )

        if run_base or run_learned:
            st.markdown("### Results")

            base_result = score_stock(
                fundamentals=fundamentals,
                technicals=technicals,
                market_context=market_context,
                intuition=intuition,
                ticker=ticker,
            )

            # Display base result
            base_col1, base_col2, base_col3 = st.columns(3)
            with base_col1:
                st.metric("Base Final Score", f"{base_result.final_score:.2f}")
            with base_col2:
                st.metric("Base Rating", base_result.rating)
            with base_col3:
                st.metric(
                    "Fundamental Subscore",
                    f"{base_result.subscores.fundamentals:.2f}",
                )

            st.markdown("**Subscore Breakdown (Base Model)**")
            st.json(base_result.subscores.details)

            # If learning run is requested, call learner
            if run_learned:
                learned_result = learner.predict_with_learning(
                    fundamentals=fundamentals,
                    technicals=technicals,
                    market_context=market_context,
                    intuition=intuition,
                    ticker=ticker,
                )

                learning_info = learned_result.subscores.details.get(
                    "learning_layer", {}
                )

                st.markdown("### Learning-Adjusted Result")

                learned_col1, learned_col2, learned_col3 = st.columns(3)
                with learned_col1:
                    st.metric(
                        "Learned Final Score",
                        f"{learned_result.final_score:.2f}",
                        delta=f"{learned_result.final_score - base_result.final_score:.2f}",
                    )
                with learned_col2:
                    st.metric("Learned Rating", learned_result.rating)
                with learned_col3:
                    if learning_info.get("active"):
                        st.metric(
                            "Predicted Alpha",
                            f"{learning_info.get('predicted_alpha', 0.0)*100:.2f} %",
                        )
                    else:
                        st.metric("Predicted Alpha", "N/A")

                st.markdown("**Learning Layer Details**")
                st.json(learning_info)

                if not learning_info.get("active"):
                    st.info(
                        "Learning layer is not active yet. "
                        "You need to log training examples and train the model on the 'Train on Outcomes' tab."
                    )

    # -------------- TRAIN TAB --------------
    with tab_train:
        st.subheader("Train on Realized Outcomes")

        st.markdown(
            "Use this tab **after** a holding period has passed to tell the model "
            "what actually happened over a given horizon."
        )

        st.markdown("#### Inputs (should match the setup when you made the decision)")

        ticker_train = st.text_input(
            "Ticker (for training)",
            value="TEST",
            key="ticker_train",
        ).upper()

        as_of_date_train = st.date_input(
            "As-of Date at Decision Time",
            value=date.today(),
            key="as_of_date_train",
        )

        horizon_days_train = st.number_input(
            "Horizon (days) used for this outcome",
            value=90,
            min_value=1,
            step=1,
            key="horizon_days_train",
        )

        st.markdown("---")

        with st.expander("Fundamentals at Decision Time", expanded=True):
            fundamentals_train = render_fundamental_inputs()
        with st.expander("Technicals & Flow at Decision Time", expanded=False):
            technicals_train = render_technical_inputs()
        with st.expander("Market Context at Decision Time", expanded=False):
            market_context_train = render_market_context_inputs()
        with st.expander("Intuition at Decision Time", expanded=False):
            intuition_train = render_intuition_inputs()

        st.markdown("---")
        st.markdown("#### Realized Outcome")

        col_res = st.columns(2)
        with col_res[0]:
            realized_return = st.number_input(
                "Realized Stock Return over Horizon (%)",
                value=12.0,
                help="Example: 12 means +12% over the horizon.",
                key="realized_return",
            ) / 100.0
        with col_res[1]:
            benchmark_return = st.number_input(
                "Benchmark Return over Horizon (%)",
                value=3.0,
                help="Example: 3 means +3% for SPY or relevant index.",
                key="benchmark_return",
            ) / 100.0

        st.markdown("---")
        col_train1, col_train2 = st.columns([1, 1])

        with col_train1:
            if st.button("➕ Add Training Example", use_container_width=True):
                learner.add_training_example(
                    ticker=ticker_train,
                    as_of_date=str(as_of_date_train),
                    horizon_days=int(horizon_days_train),
                    fundamentals=fundamentals_train,
                    technicals=technicals_train,
                    market_context=market_context_train,
                    intuition=intuition_train,
                    realized_return=realized_return,
                    benchmark_return=benchmark_return,
                )
                st.success(
                    f"Training example for {ticker_train} added to prediction_history.csv"
                )

        with col_train2:
            if st.button("🧠 Train / Retrain Model", use_container_width=True):
                success = learner.train_model(min_samples=10)
                if success:
                    st.success(
                        "Model trained successfully. "
                        "Future 'Score with Learning' calls will use the updated model."
                    )
                else:
                    st.warning(
                        "Not enough valid samples yet (or no history file). "
                        "Add more training examples before training."
                    )

        st.markdown("---")
        st.markdown("#### Debug: History File Preview")

        import os
        import pandas as pd

        history_path = learner.history_path
        if os.path.exists(history_path):
            try:
                df_hist = pd.read_csv(history_path)
                st.dataframe(df_hist.tail(20))
            except Exception as e:
                st.error(f"Could not read history file: {e}")
        else:
            st.info("No history file found yet. Add a training example to create it.")


if __name__ == "__main__":
    main()
