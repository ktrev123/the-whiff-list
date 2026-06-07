"""The Whiff List — Streamlit app skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.eda_dashboard import render_eda_dashboard
from src.model_viz import (
    build_batter_pred_figure,
    build_calibration_figure,
    build_engineered_importance_figure,
    build_pred_grid_figure,
    build_regression_scatter_figure,
    build_roc_figure,
)
from src.pitch_models import load_pitch_lab_models
from src.pitch_simulator import render_whiff_lab
from src.whiff_features import SEASON_END, SEASON_START

load_pitch_lab_models()

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "data" / "model"
INSIGHTS_FILE = MODEL_DIR / "model_insights.json"
BATTER_PRED_FILE = MODEL_DIR / "batter_predictions.csv"
GRID_PRED_FILE = MODEL_DIR / "league_whiff_grid.parquet"
SWING_GRID_FILE = MODEL_DIR / "league_swing_grid.parquet"
EDA_REPORT_FILE = ROOT / "data" / "reports" / "eda_report.html"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="The Whiff List",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- STYLING ---
st.markdown(
    """
<style>
:root {
    --whiff-navy: #0f172a;
    --whiff-navy-2: #162033;
    --whiff-cream: #f5efe3;
    --whiff-cream-muted: #cbbfa8;
    --whiff-gold: #d4a937;
    --whiff-border: rgba(245, 239, 227, 0.10);
}
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3, h4 { letter-spacing: -0.02em; color: var(--whiff-cream); }
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--whiff-navy-2) 0%, var(--whiff-navy) 100%);
    border: 1px solid var(--whiff-border); border-radius: 16px;
    padding: 10px 14px;
}
.whiff-subtle { color: var(--whiff-cream-muted); font-size: 0.95rem; margin-bottom: 1.25rem; }
.whiff-section-label {
    color: var(--whiff-gold); font-size: 0.86rem; font-weight: 700;
    text-transform: uppercase; margin-bottom: 0.25rem;
}
.eda-metric-split {
    height: 72px;
    border-left: 1px solid var(--whiff-border);
    margin: 12px auto;
}
.methodology-box {
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--whiff-border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
}
.whiff-hitter-card {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 14px;
    border: 1px solid var(--whiff-border);
    border-radius: 16px;
    background: linear-gradient(180deg, var(--whiff-navy-2) 0%, var(--whiff-navy) 100%);
}
.whiff-hitter-photo {
    width: 96px;
    height: 96px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid var(--whiff-gold);
}
.whiff-hitter-name {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--whiff-cream);
    line-height: 1.2;
}
.whiff-hitter-bats {
    color: var(--whiff-cream-muted);
    font-size: 0.92rem;
    margin-top: 4px;
}
.whiff-zone-panel {
    text-align: center;
    padding: 10px 12px;
    border: 1px solid var(--whiff-border);
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.02);
}
.whiff-zone-label {
    color: var(--whiff-cream-muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
}
.whiff-zone-name {
    font-size: 1.75rem;
    font-weight: 800;
    margin: 2px 0 6px;
}
.whiff-zone-desc {
    color: var(--whiff-cream-muted);
    font-size: 0.9rem;
    line-height: 1.35;
}
.whiff-zone-coords {
    color: var(--whiff-gold);
    font-size: 0.85rem;
    margin-top: 8px;
}
.whiff-hitter-switch {
    color: var(--whiff-gold);
    font-size: 0.82rem;
    margin-top: 6px;
    line-height: 1.35;
}
.whiff-hitter-stat {
    color: var(--whiff-cream-muted);
    font-size: 0.82rem;
    margin: 8px 0 0;
    line-height: 1.35;
}
.whiff-hitter-stat-league {
    color: var(--whiff-cream-muted);
    font-size: 0.74rem;
    margin: 0 0 6px;
    opacity: 0.85;
}
.whiff-hitter-stat-muted {
    color: var(--whiff-cream-muted);
    font-size: 0.82rem;
    margin-top: 8px;
}
.whiff-guess-muted {
    color: var(--whiff-cream-muted);
    font-size: 0.95rem;
    opacity: 0.45;
    margin: 0.25rem 0 0.5rem;
}
.whiff-input-section {
    margin: 1.25rem 0 0.5rem;
}
.whiff-input-section h4 {
    color: var(--whiff-gold);
    font-size: 0.86rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0 0 0.75rem;
}
.whiff-location-panel {
    border: 1px dashed var(--whiff-border);
    border-radius: 16px;
    padding: 14px 16px 8px;
    margin-top: 0.5rem;
    background: rgba(255, 255, 255, 0.02);
}
.whiff-location-panel-label {
    color: var(--whiff-cream-muted);
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
}
.whiff-count-display {
    border: 1px solid var(--whiff-border);
    border-radius: 12px;
    padding: 10px 14px;
    margin-top: 0.5rem;
    color: var(--whiff-cream);
    font-size: 1.05rem;
    font-weight: 600;
}
.whiff-location-mode {
    border: 1px solid var(--whiff-border);
    border-radius: 14px;
    padding: 12px 16px;
    margin: 0.75rem 0;
    background: rgba(255, 255, 255, 0.02);
}
.whiff-location-mode-active {
    border-color: var(--whiff-gold);
    background: rgba(212, 169, 55, 0.08);
}
.whiff-match-note {
    color: var(--whiff-cream-muted);
    font-size: 0.88rem;
    font-weight: 400;
}
.whiff-prescription-banner {
    border: 1px solid var(--whiff-gold);
    border-radius: 16px;
    padding: 18px 20px;
    margin: 12px 0 16px;
    background: linear-gradient(135deg, rgba(212, 169, 55, 0.12) 0%, rgba(15, 23, 42, 0.6) 100%);
}
.whiff-prescription-label {
    color: var(--whiff-gold);
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.whiff-prescription-headline {
    color: var(--whiff-cream);
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1.35;
    margin-bottom: 8px;
}
.whiff-prescription-meta {
    color: var(--whiff-cream-muted);
    font-size: 0.86rem;
    margin-bottom: 8px;
}
.whiff-prescription-rationale {
    color: var(--whiff-cream-muted);
    font-size: 0.92rem;
    line-height: 1.45;
}
.whiff-xwoba-box {
    display: inline-block;
    min-width: 120px;
    padding: 14px 22px;
    border-radius: 12px;
    font-size: 2.1rem;
    font-weight: 800;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    letter-spacing: 0.04em;
    color: #f8fafc;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
}
.whiff-xwoba-cold { background: #2563eb; }
.whiff-xwoba-average { background: #64748b; color: #f8fafc; }
.whiff-xwoba-hot { background: #dc2626; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_batter_name_lookup():
    lb_path = ROOT / "data" / "whiff_leaderboard_2025.csv"
    if not lb_path.exists():
        return pd.DataFrame(columns=["batter", "player_name"])
    return pd.read_csv(lb_path)[["batter", "player_name"]].drop_duplicates()


@st.cache_data
def load_model_insights():
    if INSIGHTS_FILE.exists():
        return json.loads(INSIGHTS_FILE.read_text(encoding="utf-8"))
    return None


@st.cache_data
def load_batter_predictions():
    if BATTER_PRED_FILE.exists():
        return pd.read_csv(BATTER_PRED_FILE)
    return None


@st.cache_data
def load_pred_grid():
    if GRID_PRED_FILE.exists():
        return pd.read_parquet(GRID_PRED_FILE)
    return None


@st.cache_data
def load_swing_grid():
    if SWING_GRID_FILE.exists():
        return pd.read_parquet(SWING_GRID_FILE)
    return None


def render_regression_panel(block: dict):
    st.markdown(block["layman"]["what_it_outputs"])
    if block.get("training_note"):
        st.caption(block["training_note"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{block['mae']:.3f}")
    c2.metric("RMSE", f"{block['rmse']:.3f}")
    c3.metric("R²", f"{block['r2']:.2f}")
    c4.metric("Algorithm", block["selected_model"].replace("_", " ").title())

    st.markdown(block["layman"]["mae"])
    st.markdown(block["layman"]["r2"])

    st.plotly_chart(build_regression_scatter_figure(block), use_container_width=True)
    st.plotly_chart(build_engineered_importance_figure("xwoba"), use_container_width=True)
    st.caption(
        "Grouped by engineered categories. Target: Statcast estimated_woba_using_speedangle on hit_into_play rows."
    )


def render_model_panel(block, rate_label: str, model_key: str = "swing"):
    """Read-only diagnostics for a pre-trained model block."""
    st.markdown(block["layman"]["what_it_outputs"])
    if block.get("training_note"):
        st.caption(block["training_note"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{block['roc_auc']:.2f}")
    c2.metric("Log Loss", f"{block['log_loss']:.3f}")
    c3.metric(f"September {rate_label}", f"{block['test_positive_rate'] * 100:.1f}%")
    c4.metric("Algorithm", block["selected_model"].replace("_", " ").title())

    st.markdown(block["layman"]["roc_auc"])
    st.markdown(block["layman"]["log_loss"])

    v1, v2 = st.columns(2)
    with v1:
        st.plotly_chart(build_roc_figure(block), use_container_width=True)
        st.caption("Yellow line above the dashed diagonal = better than random guessing.")
    with v2:
        st.plotly_chart(build_calibration_figure(block), use_container_width=True)
        st.caption("Points on the dashed line = predicted % matches what actually happened.")

    st.plotly_chart(build_engineered_importance_figure(model_key), use_container_width=True)
    st.caption(
        "Grouped by engineered categories (count leverage, pitch category, location, hitter zone swing rates). "
        "Legacy training artifacts may still encode individual pitch-type one-hot columns (FF, SL, …)."
    )


def tab_whiff_lab():
    st.markdown('<div class="whiff-section-label">Interactive Simulator</div>', unsafe_allow_html=True)
    st.header("The Whiff Lab")
    render_whiff_lab()


def tab_real_world_use():
    st.markdown('<div class="whiff-section-label">Application</div>', unsafe_allow_html=True)
    st.header("Real World Use")
    st.markdown(
        """
        Placeholder for production-facing views that answer *"so what?"* for fans, analysts, and front-office users.

        **Planned features**
        - **Embarrassment Index (EI)** leaderboard — context-aware chase-whiff severity
        - **League whiff heatmaps** and seasonal chase trends
        - **Player breakdown** — individual whiff maps and top flails
        - **Scouting angles** — platoon splits, pitch-type chase profiles

        This tab will consume the same Statcast pipeline and models documented in EDA and Model Refinement,
        but presentation-first rather than methodology-first.
        """
    )
    st.info("Heavy data pulls and chart rendering for this tab are intentionally deferred in the skeleton build.")


def tab_exploratory_data_analysis():
    st.markdown('<div class="whiff-section-label">Methodology</div>', unsafe_allow_html=True)
    st.header("Exploratory Data Analysis")
    render_eda_dashboard()


def tab_model_refinement():
    st.markdown('<div class="whiff-section-label">Predictive Models</div>', unsafe_allow_html=True)
    st.header("Model Refinement")
    st.markdown(
        """
        Read-only view of **pre-trained** swing, whiff, and xwOBA-on-contact models. This tab loads saved artifacts only —
        no training runs in the app.
        """
    )

    insights = load_model_insights()
    batter_preds = load_batter_predictions()
    pred_grid = load_pred_grid()
    swing_grid = load_swing_grid()
    name_lookup = load_batter_name_lookup()

    if insights is None:
        st.warning(
            "No model insights found. Train offline, then refresh:\n\n"
            "`python notebooks/train_whiff_model.py`"
        )
        st.markdown(
            """
            **What training produces**
            - `data/model/swing_model.joblib` / `whiff_model.joblib` / `xwoba_model.joblib`
            - `data/model/model_insights.json` — metrics, feature importance, example scenarios
            - Prediction grids and September holdout batter summaries
            """
        )
        return

    if "swing" not in insights:
        st.warning("Saved insights are outdated. Re-run `python notebooks/train_whiff_model.py`.")
        return

    st.markdown(
        f"""
        <div class="methodology-box">
            <h4>Three-model pipeline + hitter personalization</h4>
            <p>{insights['layman']['validation']}</p>
            <p>{insights['layman'].get('pipeline', insights['layman']['combined'])}</p>
            <p><b>Hitter inputs:</b> Each batter's <b>in-zone swing %</b> and <b>O-zone swing %</b>
            (2025 Statcast) personalize swing decisions in the Whiff Lab. High in-zone swing %
            favors called-strike paths; high O-zone swing % opens chase-whiff paths.</p>
            <p><b>Features:</b> Models use engineered categories — count leverage, pitch category
            (Fastball / Breaking / Offspeed), location tiers — not individual pitch-type codes (FF, SL, …).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_swing, tab_whiff, tab_xwoba, tab_combo, tab_notes = st.tabs(
        ["Model A — Swing", "Model B — Whiff", "Model C — xwOBAcon", "Combined scenarios", "Refinement notes"]
    )

    with tab_swing:
        st.markdown("#### Will the hitter swing?")
        render_model_panel(insights["swing"], "swing rate", model_key="swing")
        if swing_grid is not None and "pred_swing_prob" in swing_grid.columns:
            st.plotly_chart(
                build_pred_grid_figure(
                    swing_grid,
                    "pred_swing_prob",
                    "Swing probability map (2-2 count, league-average zone)",
                    "P(swing)",
                ),
                use_container_width=True,
            )
        if batter_preds is not None and "mean_pred_swing" in batter_preds.columns:
            st.plotly_chart(
                build_batter_pred_figure(
                    batter_preds,
                    name_lookup,
                    "actual_swing_rate",
                    "mean_pred_swing",
                    "Actual swing % (September)",
                    "Predicted swing %",
                    "September: predicted vs. actual swing rate",
                ),
                use_container_width=True,
            )

    with tab_whiff:
        st.markdown("#### If he swings, will he miss?")
        render_model_panel(insights["whiff"], "whiff rate (swings only)", model_key="whiff")
        if pred_grid is not None and "pred_whiff_prob" in pred_grid.columns:
            st.plotly_chart(
                build_pred_grid_figure(
                    pred_grid,
                    "pred_whiff_prob",
                    "Whiff-if-swing map (2-2 count, league-average zone)",
                    "P(whiff|swing)",
                ),
                use_container_width=True,
            )
        if batter_preds is not None and "mean_pred_whiff" in batter_preds.columns:
            st.plotly_chart(
                build_batter_pred_figure(
                    batter_preds,
                    name_lookup,
                    "actual_whiff_rate",
                    "mean_pred_whiff",
                    "Actual whiff % (September)",
                    "Predicted whiff-if-swing %",
                    "September: predicted vs. actual whiff rate",
                ),
                use_container_width=True,
            )

    with tab_xwoba:
        st.markdown("#### If he puts it in play, how much damage?")
        if "xwoba" in insights:
            render_regression_panel(insights["xwoba"])
        else:
            st.warning(
                "No xwOBA model in saved insights. Re-pull Statcast with "
                "`estimated_woba_using_speedangle`, then run `python notebooks/train_whiff_model.py`."
            )

    with tab_combo:
        st.markdown("#### Example pitches — swing, whiff, and contact quality")
        examples = pd.DataFrame(insights.get("example_pitches", []))
        if examples.empty:
            st.caption("No example scenarios in insights file.")
        else:
            display_cols = [
                c
                for c in [
                    "label",
                    "pitch_type",
                    "count",
                    "runners_on",
                    "swing_prob_pct",
                    "whiff_if_swing_pct",
                    "swing_whiff_pct",
                    "contact_prob_pct",
                    "xwoba_con",
                    "swing_takeaway",
                    "whiff_takeaway",
                    "xwoba_takeaway",
                ]
                if c in examples.columns
            ]
            st.dataframe(
                examples[display_cols].rename(
                    columns={
                        "label": "Scenario",
                        "pitch_type": "Pitch type",
                        "count": "Count",
                        "runners_on": "Runners on",
                        "swing_prob_pct": "P(swing) %",
                        "whiff_if_swing_pct": "P(whiff|swing) %",
                        "swing_whiff_pct": "P(swing & whiff) %",
                        "contact_prob_pct": "P(contact) %",
                        "xwoba_con": "xwOBAcon",
                        "swing_takeaway": "Swing takeaway",
                        "whiff_takeaway": "Whiff takeaway",
                        "xwoba_takeaway": "Damage takeaway",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("P(swing & whiff) = P(swing) × P(whiff | swing). xwOBAcon applies when P(contact) > 0.")

    with tab_notes:
        st.markdown(
            """
            **Refinement checklist (offline)**
            - Re-pull Statcast if pitch-type filters change: `python notebooks/statcast_pull.py`
            - Regenerate EDA after feature changes: `python notebooks/exploratory_analysis.py`
            - Retrain models: `python notebooks/train_whiff_model.py`
            - Refresh Pitch Lab deploy bundles (included in train script)

            **Three-model Whiff Lab**
            - **Take %** = 1 − P(swing)
            - **Whiff %** = P(swing) × P(whiff | swing)
            - **Contact %** = P(swing) × (1 − P(whiff | swing))
            - **xStrike %** = Take + Whiff in-zone; Whiff only out-of-zone
            - **xwOBAcon** from Model C when contact path is open

            **Hitter personalization (Whiff Lab)**
            - **In-zone swing %** — share of swings on pitches inside the strike zone (Statcast `miss_dist_in ≤ 0`)
            - **O-zone swing %** — share of swings on pitches outside the zone (chase tendency)
            - Both rates are computed per batter from 2025 Statcast and compared to league norms in Section 1 of the Whiff Lab

            **Feature engineering (model inputs)**
            - `count_state` (Hitter Ahead / Pitcher Ahead / Even / Full) + `is_two_strike`
            - `pitch_category` (Fastball / Breaking / Offspeed) instead of raw pitch-type one-hots (FF, SI, SL, …)
            - `speed_diff` instead of raw effective velocity (reduces multicollinearity)
            - Attack zones aligned to FanGraphs Heart / Shadow / Chase / Waste definitions
            - Hitter **in-zone** and **O-zone swing %** as personalization features
            """
        )


# --- HEADER ---
st.title("The Whiff List 💨")
st.markdown(
    '<div class="whiff-subtle">MLB swing & whiff modeling — Statcast 2025 · hybrid league + personalization strategy</div>',
    unsafe_allow_html=True,
)

tab_lab, tab_use, tab_eda, tab_model = st.tabs(
    ["The Whiff Lab", "Real World Use", "Exploratory Data Analysis", "Model Refinement"]
)

with tab_lab:
    tab_whiff_lab()
with tab_use:
    tab_real_world_use()
with tab_eda:
    tab_exploratory_data_analysis()
with tab_model:
    tab_model_refinement()
