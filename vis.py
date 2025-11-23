# vis.py
"""
Gulf Bank Datathon 2025 - ATM Cash Demand Dashboard

Features:
 - Overview: historical withdrawals & counts (train data)
 - Predictions: baseline vs ML model forecasts
 - Scenarios: simple what-if simulation (salary/weekend multipliers)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path
from datetime import datetime

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="ATM Cash Demand Dashboard",
    layout="wide",
)

st.title("🏦 ATM Cash Demand Dashboard")


# -------------------------
# File paths (adjust if needed)
# -------------------------
BASE_DIR = Path(".")  # change if you keep CSVs under "data/"
TRAIN_PATH = BASE_DIR / "atm_transactions_train_clean (1).csv"
CALENDAR_PATH = BASE_DIR / "calendar_cleaned.csv"
META_PATH = BASE_DIR / "atm_metadata_cleaned.csv"
REGION_LOOKUP_PATH = BASE_DIR / "atm_region_lookup.csv"
PRED_BASELINE_PATH = BASE_DIR / "predictions (4).csv"            # Task 1
PRED_ML_PATH = BASE_DIR / "predictions2 (1).csv"            # Task 2 ML


# ------------------------- 
# Data loaders (cached)
# -------------------------
@st.cache_data
def load_train(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.warning(f"Training file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
        df = df.dropna(subset=["dt"])
    return df


@st.cache_data
def load_calendar(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    cal = pd.read_csv(path)
    if "dt" in cal.columns:
        cal["dt"] = pd.to_datetime(cal["dt"], errors="coerce")
        cal = cal.dropna(subset=["dt"])
    return cal


@st.cache_data
def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    meta = pd.read_csv(path)
    return meta


@st.cache_data
def load_region_lookup(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    lookup = pd.read_csv(path)
    return lookup


@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
        df = df.dropna(subset=["dt"])
    return df


# -------------------------
# Helper: merge region into transactions / predictions
# -------------------------
def add_region(df: pd.DataFrame,
               meta: pd.DataFrame,
               lookup: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "atm_id" not in out.columns:
        return out

    # Try region from metadata
    if not meta.empty and "atm_id" in meta.columns and "region" in meta.columns:
        out = out.merge(meta[["atm_id", "region"]], on="atm_id", how="left")

    # If still missing, try region lookup
    if not lookup.empty and "atm_id" in lookup.columns and "region" in lookup.columns:
        if "region" in out.columns:
            out = out.merge(
                lookup[["atm_id", "region"]].rename(columns={"region": "region_lookup"}),
                on="atm_id",
                how="left",
            )
            # Use region if present, else region_lookup
            out["region"] = out["region"].fillna(out["region_lookup"])
            out = out.drop(columns=["region_lookup"])
        else:
            out = out.merge(lookup[["atm_id", "region"]], on="atm_id", how="left")

    if "region" not in out.columns:
        out["region"] = "Unknown"

    out["region"] = out["region"].fillna("Unknown")
    return out


# -------------------------
# Load all base data
# -------------------------
train_df = load_train(TRAIN_PATH)
calendar_df = load_calendar(CALENDAR_PATH)
meta_df = load_metadata(META_PATH)
region_lookup_df = load_region_lookup(REGION_LOOKUP_PATH)

pred_baseline = load_predictions(PRED_BASELINE_PATH)
pred_ml = load_predictions(PRED_ML_PATH)

# Attach region where possible
train_df = add_region(train_df, meta_df, region_lookup_df)
if not pred_baseline.empty:
    pred_baseline = add_region(pred_baseline, meta_df, region_lookup_df)
if not pred_ml.empty:
    pred_ml = add_region(pred_ml, meta_df, region_lookup_df)


# -------------------------
# Sidebar filters (global)
# -------------------------
st.sidebar.header("🔍 Global Filters")

# ATM filter
atm_options = ["All"]
if not train_df.empty and "atm_id" in train_df.columns:
    atm_options += sorted(train_df["atm_id"].unique().tolist())
selected_atm = st.sidebar.selectbox("ATM ID", atm_options, key="sb_atm")

# Region filter
region_options = ["All"]
if "region" in train_df.columns:
    region_options += sorted(train_df["region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Region", region_options, key="sb_region")

# Date range filter based on train data
if not train_df.empty:
    min_date = train_df["dt"].min().date()
    max_date = train_df["dt"].max().date()
else:
    today = datetime.today().date()
    min_date, max_date = today, today

date_range = st.sidebar.date_input(
    "Date range (train data)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="sb_daterange",
)

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔮 Predictions", "🎯 Scenarios"])


# =========================
# TAB 1: OVERVIEW
# =========================
with tab1:
    st.subheader("📈 Historical ATM Activity Overview")

    if train_df.empty:
        st.info("Training data not available. Please check atm_transactions_train_clean.csv.")
    else:
        df = train_df.copy()

        # Apply filters
        if selected_atm != "All":
            df = df[df["atm_id"] == selected_atm]

        if selected_region != "All":
            if "region" in df.columns:
                df = df[df["region"] == selected_region]

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df["dt"] >= pd.to_datetime(start_date)) &
                    (df["dt"] <= pd.to_datetime(end_date))]

        if df.empty:
            st.warning("No data for selected filters.")
        else:
            # KPIs
            col1, col2, col3 = st.columns(3)
            with col1:
                total_kwd = df["total_withdrawn_amount_kwd"].sum()
                st.metric("Total Withdrawn (KWD)", f"{total_kwd:,.0f}")
            with col2:
                total_txn = df["total_withdraw_txn_count"].sum()
                st.metric("Total Withdrawals (Count)", f"{total_txn:,}")
            with col3:
                avg_kwd = df["total_withdrawn_amount_kwd"].mean()
                st.metric("Average Daily Withdrawn (KWD)", f"{avg_kwd:,.0f}")

            # Time series - amount
            st.markdown("#### Daily Withdrawn Amount (KWD)")
            ts_amt = df.sort_values("dt")
            fig_amt = px.line(
                ts_amt,
                x="dt",
                y="total_withdrawn_amount_kwd",
                color="atm_id" if selected_atm == "All" else None,
            )
            st.plotly_chart(fig_amt, use_container_width=True)

            # Time series - count
            st.markdown("#### Daily Withdrawal Count")
            ts_cnt = df.sort_values("dt")
            fig_cnt = px.line(
                ts_cnt,
                x="dt",
                y="total_withdraw_txn_count",
                color="atm_id" if selected_atm == "All" else None,
            )
            st.plotly_chart(fig_cnt, use_container_width=True)

            # Table
            st.markdown("#### Sample Transactions")
            st.dataframe(
                df.sort_values(["dt", "atm_id"]).head(200),
                use_container_width=True,
                height=300,
            )


# =========================
# TAB 2: PREDICTIONS
# =========================
with tab2:
    st.subheader("🔮 ATM Forecasts (Baseline vs ML)")

    # Select prediction source
    pred_sources = []
    if not pred_baseline.empty:
        pred_sources.append("Baseline (Task 1)")
    if not pred_ml.empty:
        pred_sources.append("ML Model (Task 2)")

    if not pred_sources:
        st.info("No predictions found. Please generate predictions.csv and/or predictions_task2.csv.")
    else:
        source = st.radio("Prediction Source", pred_sources, key="pred_source_radio")

        if source == "Baseline (Task 1)":
            preds = pred_baseline.copy()
        else:
            preds = pred_ml.copy()

        # Filter controls inside tab
        colA, colB, colC = st.columns(3)

        with colA:
            atm_list = ["All"] + sorted(preds["atm_id"].unique().tolist())
            sel_atm_pred = st.selectbox("ATM (Predictions)", atm_list, key="pred_atm")

        with colB:
            if "region" in preds.columns:
                region_list_pred = ["All"] + sorted(preds["region"].dropna().unique().tolist())
            else:
                region_list_pred = ["All"]
            sel_region_pred = st.selectbox("Region (Predictions)", region_list_pred, key="pred_region")

        with colC:
            min_dt_pred = preds["dt"].min().date()
            max_dt_pred = preds["dt"].max().date()
            date_range_pred = st.date_input(
                "Date range (Predictions)",
                value=(min_dt_pred, max_dt_pred),
                min_value=min_dt_pred,
                max_value=max_dt_pred,
                key="pred_dates",
            )

        # Apply filters
        dfp = preds.copy()
        if sel_atm_pred != "All":
            dfp = dfp[dfp["atm_id"] == sel_atm_pred]

        if sel_region_pred != "All" and "region" in dfp.columns:
            dfp = dfp[dfp["region"] == sel_region_pred]

        if isinstance(date_range_pred, tuple) and len(date_range_pred) == 2:
            start_p, end_p = date_range_pred
            dfp = dfp[(dfp["dt"] >= pd.to_datetime(start_p)) &
                      (dfp["dt"] <= pd.to_datetime(end_p))]

        # Round predictions
        for col in ["predicted_withdrawn_kwd", "predicted_withdraw_count"]:
            if col in dfp.columns:
                dfp[col] = dfp[col].astype(float).round(2)

        if dfp.empty:
            st.warning("No predictions for selected filters.")
        else:
            # Show table
            st.markdown("#### Filtered Predictions")
            st.dataframe(
                dfp.sort_values(["dt", "atm_id"]),
                use_container_width=True,
                height=350,
            )

            # Charts
            st.markdown("#### Predicted Withdrawn Amount (KWD)")
            fig_p_amt = px.line(
                dfp.sort_values("dt"),
                x="dt",
                y="predicted_withdrawn_kwd",
                color="atm_id" if sel_atm_pred == "All" else None,
            )
            st.plotly_chart(fig_p_amt, use_container_width=True)

            st.markdown("#### Predicted Withdrawal Count")
            fig_p_cnt = px.line(
                dfp.sort_values("dt"),
                x="dt",
                y="predicted_withdraw_count",
                color="atm_id" if sel_atm_pred == "All" else None,
            )
            st.plotly_chart(fig_p_cnt, use_container_width=True)

            # Download
            csv_data = dfp.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Filtered Predictions",
                data=csv_data,
                file_name="filtered_predictions.csv",
                mime="text/csv",
                key="download_preds",
            )


# =========================
# TAB 3: SCENARIOS
# =========================
with tab3:
    st.subheader("🎯 Scenario Simulation (What-if Analysis)")

    st.markdown(
        "This is a simple what-if tool that scales predicted demand based on "
        "**salary-day** and **weekend** multipliers. It does not retrain the model, "
        "but helps visualize impact assumptions."
    )

    # Use ML predictions if available, else baseline, else exit
    if not pred_ml.empty:
        base_scenario = pred_ml.copy()
        st.caption("Using ML predictions (Task 2) as baseline for scenarios.")
    elif not pred_baseline.empty:
        base_scenario = pred_baseline.copy()
        st.caption("Using Baseline predictions (Task 1) as fallback for scenarios.")
    else:
        st.info("No predictions available to run scenarios.")
        st.stop()

    # Merge calendar flags for salary + weekend
    if not calendar_df.empty:
        cal_small = calendar_df[["dt", "is_weekend", "is_salary_disbursement"]].copy()
        base_scenario = base_scenario.merge(cal_small, on="dt", how="left")
    else:
        base_scenario["is_weekend"] = False
        base_scenario["is_salary_disbursement"] = False

    base_scenario["is_weekend"] = base_scenario["is_weekend"].fillna(False).astype(int)
    base_scenario["is_salary_disbursement"] = (
        base_scenario["is_salary_disbursement"].fillna(False).astype(int)
    )

    # Filters
    colS1, colS2, colS3 = st.columns(3)
    with colS1:
        atm_list_s = ["All"] + sorted(base_scenario["atm_id"].unique().tolist())
        sel_atm_s = st.selectbox("ATM (Scenario)", atm_list_s, key="sc_atm")

    with colS2:
        if "region" in base_scenario.columns:
            region_s_list = ["All"] + sorted(base_scenario["region"].dropna().unique().tolist())
        else:
            region_s_list = ["All"]
        sel_region_s = st.selectbox("Region (Scenario)", region_s_list, key="sc_region")

    with colS3:
        min_dt_s = base_scenario["dt"].min().date()
        max_dt_s = base_scenario["dt"].max().date()
        date_range_s = st.date_input(
            "Scenario Date Range",
            value=(min_dt_s, max_dt_s),
            min_value=min_dt_s,
            max_value=max_dt_s,
            key="sc_dates",
        )

    colM1, colM2 = st.columns(2)
    with colM1:
        salary_mult = st.slider(
            "Salary-day multiplier",
            min_value=0.5,
            max_value=2.0,
            value=1.3,
            step=0.1,
            key="sc_salary_mult",
        )
    with colM2:
        weekend_mult = st.slider(
            "Weekend multiplier",
            min_value=0.5,
            max_value=2.0,
            value=1.1,
            step=0.1,
            key="sc_weekend_mult",
        )

    df_s = base_scenario.copy()

    # Apply filters
    if sel_atm_s != "All":
        df_s = df_s[df_s["atm_id"] == sel_atm_s]

    if sel_region_s != "All" and "region" in df_s.columns:
        df_s = df_s[df_s["region"] == sel_region_s]

    if isinstance(date_range_s, tuple) and len(date_range_s) == 2:
        start_s, end_s = date_range_s
        df_s = df_s[(df_s["dt"] >= pd.to_datetime(start_s)) &
                    (df_s["dt"] <= pd.to_datetime(end_s))]

    if df_s.empty:
        st.warning("No predictions for selected scenario filters.")
    else:
        # Base values
        base_amt = df_s["predicted_withdrawn_kwd"].astype(float)

        # Scenario multiplier per row
        mult = 1.0
        # Salary effect
        mult_salary = 1 + (salary_mult - 1) * df_s["is_salary_disbursement"]
        # Weekend effect
        mult_weekend = 1 + (weekend_mult - 1) * df_s["is_weekend"]

        total_mult = mult_salary * mult_weekend
        df_s["scenario_withdrawn_kwd"] = (base_amt * total_mult).round(2)

        st.markdown("#### Scenario vs Baseline (Withdrawn KWD)")

        compare_df = df_s[["dt", "atm_id", "region",
                           "predicted_withdrawn_kwd", "scenario_withdrawn_kwd"]].copy()

        st.dataframe(
            compare_df.sort_values(["dt", "atm_id"]),
            use_container_width=True,
            height=320,
        )

        # Plot baseline vs scenario
        plot_df = compare_df.melt(
            id_vars=["dt", "atm_id"],
            value_vars=["predicted_withdrawn_kwd", "scenario_withdrawn_kwd"],
            var_name="type",
            value_name="withdrawn_kwd",
        )

        fig_sc = px.line(
            plot_df.sort_values("dt"),
            x="dt",
            y="withdrawn_kwd",
            color="type",
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Aggregate impact
        total_base = compare_df["predicted_withdrawn_kwd"].sum()
        total_scenario = compare_df["scenario_withdrawn_kwd"].sum()
        diff = total_scenario - total_base
        pct = (diff / total_base * 100) if total_base != 0 else 0

        col_imp1, col_imp2, col_imp3 = st.columns(3)
        with col_imp1:
            st.metric("Baseline Total (KWD)", f"{total_base:,.0f}")
        with col_imp2:
            st.metric("Scenario Total (KWD)", f"{total_scenario:,.0f}")
        with col_imp3:
            st.metric("Change (KWD / %)", f"{diff:,.0f} ({pct:+.1f}%)")

        # Download scenario results
        scen_out = compare_df.copy()
        scen_out["change_kwd"] = scen_out["scenario_withdrawn_kwd"] - scen_out["predicted_withdrawn_kwd"]
        scen_csv = scen_out.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Scenario Results",
            data=scen_csv,
            file_name="scenario_results.csv",
            mime="text/csv",
            key="download_scenario",
        )
