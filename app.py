"""The Whiff List — Streamlit app skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model_viz import (
    build_batter_pred_figure,
    build_calibration_figure,
    build_importance_figure,
    build_pred_grid_figure,
    build_roc_figure,
)
from src.pitch_simulator import render_whiff_lab
from src.whiff_features import SEASON_END, SEASON_START

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


def render_model_panel(block, rate_label: str):
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

    st.plotly_chart(build_importance_figure(block), use_container_width=True)


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

    st.markdown(
        f"""
        <div class="methodology-box">
        <h4>What this covers</h4>
        <p>2025 MLB Statcast pitch-level data ({SEASON_START} – {SEASON_END}), qualified hitters (502+ AB),
        competitive pitch types only. Train split: <b>Mar–Aug</b>; September holdout for model evaluation.</p>
        <p>The EDA justifies a <b>hybrid modeling strategy</b>: league-wide swing/whiff models as the default,
        with hitter-specific personalization when sample size and residuals support it (~600-pitch crossover).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Report sections")
    sections = [
        ("Data Quality", "Velocity (lower-fence only) and spin (two-sided IQR) outlier removal per pitch type."),
        ("Count & Pitch Type", "All 12 counts; swing/whiff by Fastballs / Breaking Balls / Off-Speed."),
        ("Hitter Heterogeneity", "Distribution of swing, whiff-if-swing, and chase rates across qualified hitters."),
        ("Pitch Physics", "Velocity, spin, and movement summaries by pitch category."),
        ("Plate Location", "Swing and whiff-if-swing heatmaps by horizontal / vertical location."),
        ("Correlations", "Contextual variables → swing; pitch physics → whiff (no redundant speed_diff in matrix)."),
        ("Attack Zones", "FanGraphs Heart / Shadow / Chase / Waste; personalization gains by zone."),
        ("League vs Personalized", "Learning curve, residuals, and hitter-specific swing model comparison."),
        ("Feature Importance & Conclusions", "Random-forest drivers and production recommendations."),
    ]
    for title, desc in sections:
        st.markdown(f"**{title}** — {desc}")

    st.subheader("Interactive report")
    if EDA_REPORT_FILE.exists():
        st.success(f"Found local report: `{EDA_REPORT_FILE.relative_to(ROOT)}`")
        st.markdown(
            f"[Open full HTML report]({EDA_REPORT_FILE.resolve().as_uri()}) "
            "(best in a new browser tab — Plotly charts are interactive)."
        )
        with st.expander("Preview report (embedded)"):
            report_html = EDA_REPORT_FILE.read_text(encoding="utf-8")
            st.components.v1.html(report_html, height=720, scrolling=True)
    else:
        st.warning("Report not generated yet. From the project folder run:")
        st.code("python notebooks/exploratory_analysis.py", language="bash")
        st.caption("Output path: `data/reports/eda_report.html` (gitignored locally).")

    st.subheader("Key findings (summary)")
    st.markdown(
        """
        - **Attack zones:** Heart ~72% swing; Waste ~21% swing (after fixing zone sign logic).
        - **Personalization:** Hitter models beat league for most qualified hitters past ~600 training pitches.
        - **Zone gains:** Largest log-loss improvements in **Chase** and **Shadow**; Waste gains are positive but noisier.
        - **Features:** Location and count drive swing; pitch physics matter more for whiff. Models use `count_state`
          + `is_two_strike` and `speed_diff` (not collinear effective velocity).
        """
    )


def tab_model_refinement():
    st.markdown('<div class="whiff-section-label">Predictive Models</div>', unsafe_allow_html=True)
    st.header("Model Refinement")
    st.markdown(
        """
        Read-only view of **pre-trained** swing and whiff models. This tab loads saved artifacts only —
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
            - `data/model/swing_model.joblib` / `whiff_model.joblib`
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
            <h4>Two-step pipeline</h4>
            <p>{insights['layman']['validation']}</p>
            <p>{insights['layman']['combined']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_swing, tab_whiff, tab_combo, tab_notes = st.tabs(
        ["Model A — Swing", "Model B — Whiff", "Combined scenarios", "Refinement notes"]
    )

    with tab_swing:
        st.markdown("#### Will the hitter swing?")
        render_model_panel(insights["swing"], "swing rate")
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
        render_model_panel(insights["whiff"], "whiff rate (swings only)")
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

    with tab_combo:
        st.markdown("#### Example pitches — both models together")
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
                    "swing_takeaway",
                    "whiff_takeaway",
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
                        "swing_takeaway": "Swing takeaway",
                        "whiff_takeaway": "Whiff takeaway",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("P(swing & whiff) = P(swing) × P(whiff | swing).")

    with tab_notes:
        st.markdown(
            """
            **Refinement checklist (offline)**
            - Re-pull Statcast if pitch-type filters change: `python notebooks/statcast_pull.py`
            - Regenerate EDA after feature changes: `python notebooks/exploratory_analysis.py`
            - Retrain models: `python notebooks/train_whiff_model.py`
            - Refresh Pitch Lab deploy bundles (included in train script)

            **Recent feature engineering**
            - `count_state` (Hitter Ahead / Pitcher Ahead / Even / Full) + `is_two_strike`
            - `speed_diff` instead of raw effective velocity (reduces multicollinearity)
            - Attack zones aligned to FanGraphs Heart / Shadow / Chase / Waste definitions
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
