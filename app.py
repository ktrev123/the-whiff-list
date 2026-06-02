import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.model_viz import (
    build_batter_pred_figure,
    build_calibration_figure,
    build_importance_figure,
    build_pred_grid_figure,
    build_roc_figure,
)
from src.pitch_simulator import render_pitch_lab
from src.whiff_features import SEASON_END, SEASON_START

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="The Whiff List",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- STYLING ---
st.markdown("""
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
div[data-testid="stDataFrame"] { border: 1px solid var(--whiff-border); border-radius: 14px; overflow: hidden; }
.whiff-section-label { color: var(--whiff-gold); font-size: 0.86rem; font-weight: 700; text-transform: uppercase; margin-bottom: 0.25rem; }
.whiff-divider { height: 1px; width: 100%; background: linear-gradient(90deg, rgba(212,169,55,0) 0%, rgba(212,169,55,0.55) 50%, rgba(212,169,55,0) 100%); margin: 2rem 0 1.5rem 0; }
.methodology-box {
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--whiff-border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 25px;
}
.leaderboard-sub { color: var(--whiff-cream-muted); font-size: 0.88rem; margin-top: -0.5rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- DATA HELPERS ---
@st.cache_data
def load_leaderboard_data():
    df = pd.read_csv("data/whiff_leaderboard_2025.csv")
    df["player_name"] = df["player_name"].str.title()
    return df

@st.cache_data
def load_pitch_data():
    return pd.read_parquet("data/statcast_2025.parquet")

def last_first_to_first_last(name):
    if isinstance(name, str) and "," in name:
        parts = [part.strip() for part in name.split(",", 1)]
        return f"{parts[1]} {parts[0]}"
    return name

def calculate_miss_distance(row):
    x, z = row["plate_x"], row["plate_z"]
    left, right = -0.708, 0.708
    bot, top = row["sz_bot"], row["sz_top"]
    x_out = max(0, left - x) if x < left else max(0, x - right)
    z_out = max(0, bot - z) if z < bot else max(0, z - top)
    return np.sqrt((x_out ** 2) + (z_out ** 2))

EI_THERMO_MIN = 20
EI_THERMO_MAX = 60
MAIN_PITCHES = ["4-Seam Fastball", "Slider", "Changeup", "Curveball", "Sinker", "Cutter", "Sweeper"]


def ei_marker_color(ei):
    if ei >= 47:
        return "#e63946"
    if ei <= 35:
        return "#22c55e"
    return "#f5efe3"


def ei_marker_line(ei):
    if 35 < ei < 47:
        return "#9ca3af"
    return "#0f172a"


def build_ei_thermometer(pitch_ei_df, handedness, title):
    hand_df = pitch_ei_df[pitch_ei_df["Handedness"] == handedness]
    ei_by_pitch = hand_df.set_index("pitch_name")["ei"].to_dict()

    fig = go.Figure()
    y_positions = list(range(len(MAIN_PITCHES)))

    for x0, x1, color in [
        (EI_THERMO_MIN, 35, "rgba(34, 197, 94, 0.18)"),
        (35, 47, "rgba(245, 239, 227, 0.10)"),
        (47, EI_THERMO_MAX, "rgba(230, 57, 70, 0.18)"),
    ]:
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=-0.55,
            y1=len(MAIN_PITCHES) - 0.45,
            fillcolor=color,
            line_width=0,
            layer="below",
        )

    for i in y_positions:
        fig.add_shape(
            type="rect",
            x0=EI_THERMO_MIN,
            x1=EI_THERMO_MAX,
            y0=i - 0.18,
            y1=i + 0.18,
            fillcolor="rgba(255, 255, 255, 0.03)",
            line=dict(color="rgba(245, 239, 227, 0.22)", width=1.5),
            layer="below",
        )
        for tick in range(EI_THERMO_MIN, EI_THERMO_MAX + 1, 10):
            fig.add_shape(
                type="line",
                x0=tick,
                x1=tick,
                y0=i - 0.08,
                y1=i + 0.08,
                line=dict(color="rgba(245, 239, 227, 0.12)", width=1),
                layer="below",
            )

    marker_x, marker_y, marker_colors, marker_lines, hover_labels = [], [], [], [], []
    for i, pitch in enumerate(MAIN_PITCHES):
        ei = ei_by_pitch.get(pitch)
        if pd.isna(ei):
            continue
        ei = float(np.clip(ei, EI_THERMO_MIN, EI_THERMO_MAX))
        marker_x.append(ei)
        marker_y.append(i)
        marker_colors.append(ei_marker_color(ei))
        marker_lines.append(ei_marker_line(ei))
        hover_labels.append(pitch)

    fig.add_trace(
        go.Scatter(
            x=marker_x,
            y=marker_y,
            mode="markers",
            marker=dict(size=24, color=marker_colors, line=dict(color=marker_lines, width=2), symbol="circle"),
            customdata=np.array(hover_labels).reshape(-1, 1),
            hovertemplate="<b>%{customdata[0]}</b><br>Avg EI: %{x:.1f}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#f5efe3")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=10, r=20, t=45, b=20),
        xaxis=dict(
            range=[EI_THERMO_MIN - 1, EI_THERMO_MAX + 1],
            title="Embarrassment Index",
            dtick=10,
            gridcolor="rgba(245, 239, 227, 0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_positions,
            ticktext=MAIN_PITCHES,
            autorange="reversed",
            showgrid=False,
        ),
    )
    return fig


MODEL_DIR = Path("data/model")
INSIGHTS_FILE = MODEL_DIR / "model_insights.json"
BATTER_PRED_FILE = MODEL_DIR / "batter_predictions.csv"
GRID_PRED_FILE = MODEL_DIR / "league_whiff_grid.parquet"
SWING_GRID_FILE = MODEL_DIR / "league_swing_grid.parquet"


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
    if GRID_PRED_FILE.exists():
        return pd.read_parquet(GRID_PRED_FILE)
    return None


def render_model_panel(block, rate_label):
    """One model's metrics, plain-English copy, and diagnostic charts."""
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


# --- DATA PROCESSING ---
df_base = load_leaderboard_data().copy()
pitch_data = load_pitch_data().copy()
pitch_data["game_date"] = pd.to_datetime(pitch_data["game_date"])

# Global Qualification Filter (502 ABs)
df_base = df_base[df_base["ab"] >= 502].copy()

# Hard Date Cutoff
pitch_data = pitch_data[(pitch_data["game_date"] >= SEASON_START) & (pitch_data["game_date"] <= SEASON_END)].copy()

whiff_desc = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
pitch_data = pitch_data[pitch_data["description"].isin(whiff_desc)].copy()
pitch_data = pitch_data.dropna(subset=["batter", "plate_x", "plate_z", "sz_top", "sz_bot"])

name_lookup = df_base[["batter", "player_name"]].drop_duplicates().rename(columns={"player_name": "batter_name"})
pitch_data = pitch_data.merge(name_lookup, on="batter", how="inner")

pitch_data["batter_name"] = pitch_data["batter_name"].str.title()
pitch_data["player_name"] = pitch_data["player_name"].str.title().apply(last_first_to_first_last)
pitch_data["miss_dist_in"] = (pitch_data.apply(calculate_miss_distance, axis=1) * 12).round(1)
pitch_data["zone_split"] = np.where(pitch_data["miss_dist_in"] == 0, "In Zone", "Out of Zone")
pitch_data["runners_on"] = pitch_data[["on_1b", "on_2b", "on_3b"]].notna().sum(axis=1)
pitch_data["count"] = pitch_data["balls"].fillna(0).astype(int).astype(str) + "-" + pitch_data["strikes"].fillna(0).astype(int).astype(str)

# Formula scaled up out of 100
pitch_data["ei"] = ((100 / 0.85) * (0.45 * np.minimum(pitch_data["miss_dist_in"]/18, 1.0) + 
                                    0.20 * (pitch_data["zone_split"] == "Out of Zone").astype(float) + 
                                    0.20 * (pitch_data["runners_on"]/3.0))).round(1)

# --- HEADER ---
st.title("The Whiff List 💨")
st.markdown('<div class="whiff-subtle">Tracking the ugliest chase whiffs and repeat flails from the 2025 MLB season.</div>', unsafe_allow_html=True)

# --- METHODOLOGY SECTION ---
st.markdown('<div class="whiff-section-label">Valuation Framework</div>', unsafe_allow_html=True)
st.markdown("### Metrics Architecture: The Embarrassment Index")

with st.container():
    st.markdown("""
    <div class="methodology-box">
        <h4>Why evaluate Whiff Quality?</h4>
        <p>Standard box-score metrics treat every swing-and-miss identically. But let's be real: protecting the plate on a borderline sinking 
        fastball is just a professional hazard. Swinging and missing on a pitch 17 inches out of the zone when the count is 3-2 with 
        your teammates desperately needing you on base? That's... <b>embarrassing</b>.</p>
        <p>The <b>Embarrassment Index (EI)</b> is a context-aware tracking metric designed to isolate non-competitive, high-leverage plate 
        discipline failures from structural swing-and-miss tendencies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.latex(r"EI = \frac{100}{0.85} \cdot \left(0.45 \cdot D + 0.20 \cdot Z + 0.20 \cdot R\right)")
    with col_m2:
        st.markdown("""
        * **$D$ (Distance Penalty):** Linear scaling of the raw miss distance from the strike zone boundary, capped at 18 inches.
        * **$Z$ (Zone Split):** A binary penalty applied instantly if the pitch is tracked completely outside the strike zone.
        * **$R$ (Leverage Factor):** Scales the severity of the whiff based on base-runner occupancy (punishing high-leverage flails).
        """)

st.markdown('<div class="whiff-divider"></div>', unsafe_allow_html=True)

# --- LEAGUE HEATMAP ---
st.markdown('<div class="whiff-section-label">League Profile</div>', unsafe_allow_html=True)
st.markdown("### Full-Season Whiff Density Heatmap")

fig_heat = go.Figure()
fig_heat.add_trace(go.Histogram2dContour(
    x=pitch_data["plate_x"],
    y=pitch_data["plate_z"],
    colorscale=[
        [0.0, 'rgba(15, 23, 42, 0)'],
        [0.05, 'rgba(72, 40, 120, 0.2)'],
        [0.2, 'rgba(60, 80, 140, 0.6)'],
        [0.4, '#20908d'],
        [0.7, '#5ec962'],
        [1.0, '#fde725']
    ],
    reversescale=False,
    ncontours=45,
    line=dict(width=0),
    showscale=True,
    colorbar=dict(title="Whiff Density"),
    hoverinfo="skip" # REMOVED TOOLTIP COMPLETELY
))
fig_heat.add_shape(type="rect", x0=-0.708, x1=0.708, y0=1.6, y1=3.4, line=dict(color="#f5efe3", width=3))
fig_heat.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(range=[-3, 3], title="Horizontal Plate Location (ft)", scaleanchor="y", scaleratio=1), # Exact axis boundaries
    yaxis=dict(range=[0, 4.5], title="Vertical Plate Location (ft)"), # Exact axis boundaries
    height=600
)
st.plotly_chart(fig_heat, use_container_width=True)

# --- TRENDS ---
st.markdown('<div class="whiff-section-label">Seasonal Trends</div>', unsafe_allow_html=True)
trend = pitch_data[pitch_data["zone_split"] == "Out of Zone"].groupby("game_date").agg(vol=("description", "count"), ei=("ei", "mean")).reset_index()
trend["vol_roll"] = trend["vol"].rolling(7).mean()
trend["ei_roll"] = trend["ei"].rolling(7).mean()

fig_t = go.Figure()
fig_t.add_trace(go.Scatter(x=trend["game_date"], y=trend["vol_roll"], name="7-Day Vol", line=dict(color="#20908d")))
fig_t.add_trace(go.Scatter(x=trend["game_date"], y=trend["ei_roll"], name="7-Day Avg EI", line=dict(color="#fde725"), yaxis="y2"))
fig_t.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(title="Volume"), yaxis2=dict(title="Avg EI", overlaying="y", side="right"), height=300)
st.plotly_chart(fig_t, use_container_width=True)

# --- PLATOON SPLITS THERMOMETER ---
st.markdown('<div class="whiff-section-label">Platoon Splits</div>', unsafe_allow_html=True)
st.markdown("### Chase Thermometer by Pitch Type")
st.caption(
    "Out-of-zone whiff severity on a 20–60 EI scale. Each row is a pitch-type thermometer; "
    "the baseball marks where that handedness lands. Green = tamer chase, white = moderate, red = ugly flail."
)

splits_df = pitch_data[pitch_data["zone_split"] == "Out of Zone"].copy()
splits_df["Handedness"] = splits_df["stand"].map({"L": "Left", "R": "Right"})
pitch_splits = (
    splits_df[splits_df["pitch_name"].isin(MAIN_PITCHES)]
    .groupby(["pitch_name", "Handedness"], as_index=False)["ei"]
    .mean()
)

therm_left, therm_right = st.columns(2)
with therm_left:
    st.plotly_chart(
        build_ei_thermometer(pitch_splits, "Left", "Left-Handed Hitters"),
        use_container_width=True,
    )
with therm_right:
    st.plotly_chart(
        build_ei_thermometer(pitch_splits, "Right", "Right-Handed Hitters"),
        use_container_width=True,
    )

# --- PREDICTIVE MODEL (PLAIN ENGLISH) ---
st.markdown('<div class="whiff-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="whiff-section-label">Predictive Model</div>', unsafe_allow_html=True)
st.markdown("### Swing & Whiff Models — Plain English")
st.caption(
    "Scroll to this section after training. Visuals appear below in tabs: "
    "**Model A (Swing)**, **Model B (Whiff)**, and **Combined scenarios**."
)

insights = load_model_insights()
batter_preds = load_batter_predictions()
pred_grid = load_pred_grid()
swing_grid = load_swing_grid()

if insights is None:
    st.warning(
        "No model insights found. From the project folder run:\n\n"
        "`python notebooks/train_whiff_model.py`\n\n"
        "Then refresh this page (R in Streamlit)."
    )
elif "swing" not in insights:
    st.warning(
        "Your saved insights are from the old single-model run. Re-run "
        "`python notebooks/train_whiff_model.py` to train **both** swing and whiff models."
    )
else:
    st.markdown(
        f"""
        <div class="methodology-box">
            <h4>Two-step question</h4>
            <p>{insights['layman']['validation']}</p>
            <p>{insights['layman']['combined']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_swing, tab_whiff, tab_combo = st.tabs(
        ["Model A — Swing rate", "Model B — Whiff rate (if he swings)", "Combined scenarios"]
    )

    with tab_swing:
        st.markdown("#### (A) Will the hitter swing?")
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
            name_lookup_pred = df_base[["batter", "player_name"]].drop_duplicates()
            st.plotly_chart(
                build_batter_pred_figure(
                    batter_preds,
                    name_lookup_pred,
                    "actual_swing_rate",
                    "mean_pred_swing",
                    "Actual swing % (September)",
                    "Predicted swing %",
                    "September: predicted vs. actual swing rate",
                ),
                use_container_width=True,
            )

    with tab_whiff:
        st.markdown("#### (B) If he swings, will he miss?")
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
            name_lookup_pred = df_base[["batter", "player_name"]].drop_duplicates()
            st.plotly_chart(
                build_batter_pred_figure(
                    batter_preds,
                    name_lookup_pred,
                    "actual_whiff_rate",
                    "mean_pred_whiff",
                    "Actual whiff % (September, all pitches)",
                    "Predicted whiff-if-swing %",
                    "September: predicted vs. actual whiff rate",
                ),
                use_container_width=True,
            )

    with tab_combo:
        st.markdown("#### Example pitches — both models together")
        examples = pd.DataFrame(insights["example_pitches"])
        example_cols = [
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
        example_cols = [c for c in example_cols if c in examples.columns]
        st.dataframe(
            examples[example_cols].rename(
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
        st.caption(
            "P(swing & whiff) = P(swing) × P(whiff | swing). "
            "That is the estimated chance of a swinging strike on this pitch profile."
        )

# --- PITCH LAB ---
st.markdown('<div class="whiff-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="whiff-section-label">Pitch Lab</div>', unsafe_allow_html=True)
st.markdown("### Interactive Swing & Whiff Simulator")
render_pitch_lab(df_base)

# --- LEADERBOARD ---
st.markdown('<div class="whiff-section-label">League View</div>', unsafe_allow_html=True)
st.markdown("### Chase Leaderboard", help="Minimum 502 At-Bats required to qualify.")

avg_ei_ozone = pitch_data[pitch_data["zone_split"] == "Out of Zone"].groupby("batter")["ei"].mean().reset_index().rename(columns={"ei": "avg_ei"})
df_lb = df_base.merge(avg_ei_ozone, on="batter", how="left").sort_values("avg_ei", ascending=False).reset_index(drop=True)
df_lb["Rank"] = df_lb.index + 1
df_lb["Whiff%"] = (df_lb["whiff_rate"] * 100).round(1)

# FIXED: Removed specific metric objects, embedded qualified hitter details seamlessly as a low-profile line
st.markdown(f'<div class="leaderboard-sub"><b>Qualified Hitters:</b> {len(df_lb)} (Dataset strictly restricted to active batting title contenders)</div>', unsafe_allow_html=True)

lb_display = df_lb[["Rank", "player_name", "ab", "swings", "whiffs", "Whiff%", "avg_ei"]].rename(columns={"player_name": "Batter", "ab": "ABs", "avg_ei": "Avg EI"})
st.dataframe(lb_display, use_container_width=True, hide_index=True)

# --- WORST WHIFFERS ---
st.markdown('<div class="whiff-section-label">Worst Swings</div>', unsafe_allow_html=True)
st.markdown("### Worst Whiffers")
worst_df = pitch_data[pitch_data["zone_split"] == "Out of Zone"].sort_values("ei", ascending=False).head(25)
st.dataframe(worst_df[["batter_name", "player_name", "pitch_name", "count", "runners_on", "miss_dist_in", "ei"]].rename(
    columns={"batter_name": "Batter", "player_name": "Pitcher", "pitch_name": "Pitch Type", "count": "Count", "runners_on": "Runners On", "miss_dist_in": "Miss Dist (in)", "ei": "EI"}
), use_container_width=True, hide_index=True)

# --- PLAYER BREAKDOWN & SELECTION OPTIMIZATION ---
st.markdown('<div class="whiff-divider"></div>', unsafe_allow_html=True)
p_list = sorted(pitch_data["batter_name"].dropna().unique())

# FIXED: Restructured on-page selector grid to compact wide, empty rows into clean, balanced data blocks
col_control, col_headshot = st.columns([3, 1])

with col_control:
    selected_player_box = st.selectbox("Select Hitter for Performance Breakdown", p_list, index=p_list.index("Shohei Ohtani") if "Shohei Ohtani" in p_list else 0)
    p_whiffs = pitch_data[pitch_data["batter_name"] == selected_player_box].copy()
    p_whiffs["date_str"] = p_whiffs["game_date"].dt.strftime("%m-%d") # Format date metadata without year
    
    st.markdown(f"### Player Breakdown: {selected_player_box}")
    
    # Inline structural metrics inside control column block to balance visual asymmetry
    sub_m1, sub_m2, sub_m3 = st.columns(3)
    sub_m1.metric("Total Whiffs Detected", len(p_whiffs))
    sub_m2.metric("Out-of-Zone Chases", len(p_whiffs[p_whiffs["zone_split"] == "Out of Zone"]))
    sub_m3.metric("Player Peak EI", f"{p_whiffs['ei'].max():.0f}" if not p_whiffs.empty else "0")

with col_headshot:
    pid = int(p_whiffs["batter"].iloc[0]) if not p_whiffs.empty else None
    if pid: 
        st.image(f"https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{pid}/headshot/67/current", width=145)

# FIXED: Individual Strike Zone visual positioned BEFORE player data-table grid rows
if not p_whiffs.empty:
    fig_sz = go.Figure()
    fig_sz.add_trace(go.Scatter(
        x=p_whiffs["plate_x"], y=p_whiffs["plate_z"], mode="markers", 
        marker=dict(size=11, color=p_whiffs["ei"], colorscale="Viridis", showscale=True, colorbar=dict(title="EI")),
        customdata=p_whiffs[["player_name", "pitch_name", "runners_on", "count", "miss_dist_in", "ei", "date_str"]],
        hovertemplate=(
            "<b>%{customdata[0]}'s %{customdata[1]}</b><br>"
            "Date: %{customdata[6]}<br>" # No year format inside tooltips
            "Count: %{customdata[3]}<br>"
            "Runners On: %{customdata[2]}<br>"
            "Miss Distance: %{customdata[4]:.0f} in<br>" # Clean formatted integers
            "Embarrassment Index: %{customdata[5]:.0f}<br>" # Clean formatted integers
            "<extra></extra>"
        )
    ))
    avg_bot, avg_top = p_whiffs["sz_bot"].mean(), p_whiffs["sz_top"].mean()
    fig_sz.add_shape(type="rect", x0=-0.708, x1=0.708, y0=avg_bot, y1=avg_top, line=dict(color="#f5efe3", width=3))
    
    # Custom compressed margins to eliminate large empty vertical spacing
    fig_sz.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
        xaxis=dict(range=[-3, 3], title="Horizontal (ft)", scaleanchor="y", scaleratio=1), 
        yaxis=dict(range=[0, 4.5], title="Vertical (ft)"), 
        height=720,
        margin=dict(t=5, b=5, l=0, r=0)
    )
    st.plotly_chart(fig_sz, use_container_width=True)

# FIXED: Player View Table rendered cleanly beneath plot visual
st.markdown('<div class="whiff-section-label">Player View</div>', unsafe_allow_html=True)
st.markdown(f"### {selected_player_box}'s Top Whiffs")
# Confirmed: Count is positioned directly before Runners On
st.dataframe(p_whiffs[["player_name", "pitch_name", "count", "runners_on", "miss_dist_in", "ei"]].sort_values("ei", ascending=False).head(10).rename(
    columns={"player_name": "Pitcher", "pitch_name": "Pitch Type", "count": "Count", "runners_on": "Runners On", "miss_dist_in": "Miss Dist (in)", "ei": "EI"}
), use_container_width=True, hide_index=True)