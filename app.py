import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="The Whiff List",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
<style>
:root {
    --whiff-navy: #0f172a;
    --whiff-navy-2: #162033;
    --whiff-cream: #f5efe3;
    --whiff-cream-muted: #cbbfa8;
    --whiff-red: #c24141;
    --whiff-red-soft: rgba(194, 65, 65, 0.16);
    --whiff-gold: #d4a937;
    --whiff-border: rgba(245, 239, 227, 0.10);
    --whiff-grid: rgba(245, 239, 227, 0.08);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}

h1 {
    font-weight: 800;
    color: var(--whiff-cream);
}

h3 {
    margin-top: 0.35rem;
    margin-bottom: 0.75rem;
    color: var(--whiff-cream);
}

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--whiff-navy-2) 0%, var(--whiff-navy) 100%);
    border: 1px solid var(--whiff-border);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.22);
}

div[data-testid="stMetricLabel"] {
    color: var(--whiff-cream-muted);
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: var(--whiff-cream);
    font-weight: 800;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--whiff-border);
    border-radius: 14px;
    overflow: hidden;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(245, 239, 227, 0.08);
}

.whiff-subtle {
    color: var(--whiff-cream-muted);
    font-size: 0.98rem;
    margin-top: -0.25rem;
    margin-bottom: 0.75rem;
}

.whiff-divider {
    height: 1px;
    width: 100%;
    background: linear-gradient(
        90deg,
        rgba(212,169,55,0) 0%,
        rgba(212,169,55,0.55) 50%,
        rgba(212,169,55,0) 100%
    );
    margin: 2rem 0 1.5rem 0;
}

.whiff-section-label {
    color: var(--whiff-gold);
    font-size: 0.86rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
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
    x = row["plate_x"]
    z = row["plate_z"]
    left_edge, right_edge = -0.708, 0.708
    bottom_edge, top_edge = row["sz_bot"], row["sz_top"]

    x_out = max(0, left_edge - x) if x < left_edge else max(0, x - right_edge)
    z_out = max(0, bottom_edge - z) if z < bottom_edge else max(0, z - top_edge)
    
    return np.sqrt((x_out ** 2) + (z_out ** 2))

# --- HEADER ---
st.title("The Whiff List 💨")
st.markdown(
    '<div class="whiff-subtle">Tracking the ugliest chase whiffs and worst misses from the 2025 MLB season.</div>',
    unsafe_allow_html=True
)

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")
min_swings = st.sidebar.slider("Minimum swings", 0, 60, 10, 5)

# --- DATA PROCESSING ---
df_base = load_leaderboard_data().copy()
pitch_data = load_pitch_data().copy()
pitch_data["game_date"] = pd.to_datetime(pitch_data["game_date"])

# Filter for regular season (Post-Spring Training)
pitch_data = pitch_data[pitch_data["game_date"] >= "2025-03-23"].copy()

whiff_descriptions = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
pitch_data = pitch_data[pitch_data["description"].isin(whiff_descriptions)].copy()
pitch_data = pitch_data.dropna(subset=["batter", "plate_x", "plate_z", "sz_top", "sz_bot"])

# Global Embarrassment Index Calculation (Calculated for everyone together)
pitch_data["miss_distance"] = pitch_data.apply(calculate_miss_distance, axis=1)
pitch_data["miss_distance_inches"] = (pitch_data["miss_distance"] * 12).round(1)
pitch_data["zone_split"] = np.where(pitch_data["miss_distance"] == 0, "In Zone", "Out of Zone")

# Handle base runners and counts
for col in ["on_1b", "on_2b", "on_3b"]:
    if col not in pitch_data.columns: pitch_data[col] = np.nan
pitch_data["runners_on"] = pitch_data[["on_1b", "on_2b", "on_3b"]].notna().sum(axis=1)

# Sort for previous pitch logic
if all(col in pitch_data.columns for col in ["game_pk", "at_bat_number", "pitch_number"]):
    pitch_data = pitch_data.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    group = pitch_data.groupby(["game_pk", "at_bat_number"])
    pitch_data["prev_desc"] = group["description"].shift(1)
    pitch_data["prev_x"] = group["plate_x"].shift(1)
    pitch_data["prev_z"] = group["plate_z"].shift(1)
    # Simple O-Zone check for prev pitch
    pitch_data["prev_whiff_ozone"] = np.where(
        pitch_data["prev_desc"].isin(whiff_descriptions) & 
        ((pitch_data["prev_x"].abs() > 0.708) | (pitch_data["prev_z"] < 1.5) | (pitch_data["prev_z"] > 3.5)), 
        1, 0
    )
else:
    pitch_data["prev_whiff_ozone"] = 0

# Scoring components
pitch_data["distance_score"] = np.minimum(pitch_data["miss_distance_inches"] / 18, 1.0)
pitch_data["zone_score"] = np.where(pitch_data["zone_split"] == "Out of Zone", 1.0, 0.0)
pitch_data["embarrassment_index"] = (
    100 * (0.45 * pitch_data["distance_score"] + 0.20 * pitch_data["zone_score"] + 
           0.15 * pitch_data["prev_whiff_ozone"] + 0.20 * (pitch_data["runners_on"] / 3.0))
).round(1)

# --- SEASONAL TRENDS ---
st.markdown('<div class="whiff-section-label">Seasonal Trends</div>', unsafe_allow_html=True)
st.markdown("### League Eagerness & Whiff Quality Over Time")

seasonal_stats = pitch_data[pitch_data["zone_split"] == "Out of Zone"].groupby("game_date").agg(
    whiff_count=("description", "count"),
    avg_ei=("embarrassment_index", "mean")
).reset_index()

fig_trends = go.Figure()
fig_trends.add_trace(go.Scatter(
    x=seasonal_stats["game_date"], y=seasonal_stats["whiff_count"],
    name="O-Zone Whiff Volume", line=dict(color="#c24141", width=2)
))
fig_trends.add_trace(go.Scatter(
    x=seasonal_stats["game_date"], y=seasonal_stats["avg_ei"],
    name="Avg Embarrassment Index", line=dict(color="#d4a937", width=2),
    yaxis="y2"
))
fig_trends.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(title="Whiff Volume"),
    yaxis2=dict(title="Avg EI", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400
)
st.plotly_chart(fig_trends, use_container_width=True)

# --- LEADERBOARD ---
st.markdown('<div class="whiff-section-label">League View</div>', unsafe_allow_html=True)
st.markdown("### Chase Leaderboard")

avg_ei_df = pitch_data[pitch_data["zone_split"] == "Out of Zone"].groupby("batter")["embarrassment_index"].mean().reset_index()
avg_ei_df.columns = ["batter", "avg_ei"]

df_filtered = df_base[df_base["swings"] >= min_swings].merge(avg_ei_df, on="batter", how="left")
df_filtered = df_filtered.sort_values("avg_ei", ascending=False).reset_index(drop=True)
df_filtered["rank"] = df_filtered.index + 1

st.dataframe(
    df_filtered[["rank", "player_name", "ab", "swings", "whiffs", "whiff_rate", "avg_ei"]].rename(
        columns={"rank": "Rank", "player_name": "Batter", "ab": "AB", "avg_ei": "Avg O-Zone EI"}
    ), use_container_width=True, hide_index=True
)

# --- WORST WHIFFERS ---
st.markdown('<div class="whiff-section-label">Worst Swings</div>', unsafe_allow_html=True)
st.markdown("### Worst Whiffers")

worst_swings = pitch_data[pitch_data["zone_split"] == "Out of Zone"].sort_values("embarrassment_index", ascending=False).head(25)
worst_swings["batter_name"] = worst_swings["player_name_x"] if "player_name_x" in worst_swings.columns else "Batter" # Example fix if names merged
# Use simplified column selection
st.dataframe(
    worst_swings[[
        "player_name", "pitch_name", "runners_on", "miss_distance_inches", "embarrassment_index"
    ]].rename(columns={"player_name": "Pitcher", "miss_distance_inches": "Miss Distance (in)"}),
    use_container_width=True, hide_index=True
)

# --- PLAYER BREAKDOWN ---
st.markdown('<div class="whiff-divider"></div>', unsafe_allow_html=True)
player_options = sorted(pitch_data["player_name"].unique()) # Note: In Statcast, player_name is often the pitcher. Adjust if you merged batter names.
selected_player = st.sidebar.selectbox("Select Hitter for Breakdown", player_options)

st.markdown(f"### Player Breakdown: {selected_player}")
player_whiffs = pitch_data[pitch_data["player_name"] == selected_player].copy() # Filter logic depends on how you map names to IDs

# Pruned Player View Table
st.dataframe(
    player_whiffs[[
        "pitch_name", "zone_split", "runners_on", "miss_distance_inches", "embarrassment_index"
    ]].sort_values("embarrassment_index", ascending=False).head(10),
    use_container_width=True, hide_index=True
)

# Plotly Visual
if not player_whiffs.empty:
    fig = go.Figure(go.Scatter(
        x=player_whiffs["plate_x"], y=player_whiffs["plate_z"],
        mode="markers", marker=dict(size=12, color=player_whiffs["embarrassment_index"], colorscale="Cividis", showscale=True)
    ))
    fig.add_shape(type="rect", x0=-0.708, x1=0.708, y0=1.6, y1=3.4, line=dict(color="white"))
    fig.update_layout(template="plotly_dark", xaxis=dict(range=[-2, 2]), yaxis=dict(range=[0, 5]))
    st.plotly_chart(fig, use_container_width=True)