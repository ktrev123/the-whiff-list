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

# --- STYLING (Preserving your original CSS) ---
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
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; color: var(--whiff-cream); }
h1 { font-weight: 800; }
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--whiff-navy-2) 0%, var(--whiff-navy) 100%);
    border: 1px solid var(--whiff-border);
    border-radius: 16px;
    padding: 16px 18px;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--whiff-border); border-radius: 14px; overflow: hidden; }
.whiff-subtle { color: var(--whiff-cream-muted); font-size: 0.98rem; margin-top: -0.25rem; margin-bottom: 0.75rem; }
.whiff-divider { height: 1px; width: 100%; background: linear-gradient(90deg, rgba(212,169,55,0) 0%, rgba(212,169,55,0.55) 50%, rgba(212,169,55,0) 100%); margin: 2rem 0 1.5rem 0; }
.whiff-section-label { color: var(--whiff-gold); font-size: 0.86rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.25rem; }
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
    left_edge, right_edge = -0.708, 0.708
    bot, top = row["sz_bot"], row["sz_top"]
    x_out = max(0, left_edge - x) if x < left_edge else max(0, x - right_edge)
    z_out = max(0, bot - z) if z < bot else max(0, z - top)
    return np.sqrt((x_out ** 2) + (z_out ** 2))

# --- DATA PROCESSING ---
df_leaderboard = load_leaderboard_data().copy()
pitch_data = load_pitch_data().copy()
pitch_data["game_date"] = pd.to_datetime(pitch_data["game_date"])

# HARD DATE CUTOFF (March 23 - Sept 27)
pitch_data = pitch_data[(pitch_data["game_date"] >= "2025-03-23") & (pitch_data["game_date"] <= "2025-09-27")].copy()

whiff_descriptions = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
pitch_data = pitch_data[pitch_data["description"].isin(whiff_descriptions)].copy()
pitch_data = pitch_data.dropna(subset=["batter", "plate_x", "plate_z", "sz_top", "sz_bot"])

# Global Calculations
pitch_data["miss_distance"] = pitch_data.apply(calculate_miss_distance, axis=1)
pitch_data["miss_distance_inches"] = (pitch_data["miss_distance"] * 12).round(1)
pitch_data["zone_split"] = np.where(pitch_data["miss_distance"] == 0, "In Zone", "Out of Zone")

# Handle Batter Names and Runners
name_lookup = df_leaderboard[["batter", "player_name"]].drop_duplicates().rename(columns={"player_name": "batter_name"})
pitch_data = pitch_data.merge(name_lookup, on="batter", how="left")
pitch_data["batter_name"] = pitch_data["batter_name"].str.title()
pitch_data["player_name"] = pitch_data["player_name"].str.title().apply(last_first_to_first_last)

pitch_data["runners_on"] = pitch_data[["on_1b", "on_2b", "on_3b"]].notna().sum(axis=1)

# Count Logic
pitch_data["count"] = pitch_data["balls"].fillna(0).astype(int).astype(str) + "-" + pitch_data["strikes"].fillna(0).astype(int).astype(str)

# Prev Pitch Logic
if all(col in pitch_data.columns for col in ["game_pk", "at_bat_number", "pitch_number"]):
    pitch_data = pitch_data.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    group = pitch_data.groupby(["game_pk", "at_bat_number"])
    pitch_data["prev_desc"] = group["description"].shift(1)
    pitch_data["prev_x"] = group["plate_x"].shift(1)
    pitch_data["prev_z"] = group["plate_z"].shift(1)
    pitch_data["prev_whiff_ozone"] = np.where(
        pitch_data["prev_desc"].isin(whiff_descriptions) & 
        ((pitch_data["prev_x"].abs() > 0.708) | (pitch_data["prev_z"] < 1.5) | (pitch_data["prev_z"] > 3.5)), 1, 0
    )
else:
    pitch_data["prev_whiff_ozone"] = 0

# Embarrassment Index
pitch_data["distance_score"] = np.minimum(pitch_data["miss_distance_inches"] / 18, 1.0)
pitch_data["zone_score"] = np.where(pitch_data["zone_split"] == "Out of Zone", 1.0, 0.0)
pitch_data["embarrassment_index"] = (
    100 * (0.45 * pitch_data["distance_score"] + 0.20 * pitch_data["zone_score"] + 
           0.15 * pitch_data["prev_whiff_ozone"] + 0.20 * (pitch_data["runners_on"] / 3.0))
).round(1)

# --- HEADER ---
st.title("The Whiff List 💨")
st.markdown('<div class="whiff-subtle">Tracking the ugliest chase whiffs and repeat flails from the 2025 season.</div>', unsafe_allow_html=True)

# --- TREND PLOT (7-Day Rolling Average) ---
st.markdown('<div class="whiff-section-label">Seasonal Trends</div>', unsafe_allow_html=True)
st.markdown("### League Eagerness vs. Whiff Quality (7-Day Rolling)")

trend_data = pitch_data[pitch_data["zone_split"] == "Out of Zone"].groupby("game_date").agg(
    daily_vol=("description", "count"),
    daily_ei=("embarrassment_index", "mean")
).reset_index()

trend_data["vol_rolling"] = trend_data["daily_vol"].rolling(7).mean()
trend_data["ei_rolling"] = trend_data["daily_ei"].rolling(7).mean()

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=trend_data["game_date"], y=trend_data["vol_rolling"], name="Rolling Vol", line=dict(color="#c24141")))
fig_trend.add_trace(go.Scatter(x=trend_data["game_date"], y=trend_data["ei_rolling"], name="Rolling Avg EI", line=dict(color="#d4a937"), yaxis="y2"))
fig_trend.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(title="Whiff Volume"), yaxis2=dict(title="Avg EI", overlaying="y", side="right"),
    xaxis=dict(range=["2025-03-23", "2025-09-27"]), height=350
)
st.plotly_chart(fig_trend, use_container_width=True)

# --- LEADERBOARD ---
st.sidebar.header("Filters")
min_swings = st.sidebar.slider("Minimum swings", 0, 60, 10, 5)

avg_ei = pitch_data[pitch_data["zone_split"] == "Out of Zone"].groupby("batter")["embarrassment_index"].mean().reset_index()
df_leaderboard = df_leaderboard.merge(avg_ei, on="batter", how="left").rename(columns={"embarrassment_index": "avg_ei"})
df_filt = df_leaderboard[df_leaderboard["swings"] >= min_swings].sort_values("avg_ei", ascending=False).reset_index(drop=True)
df_filt["Rank"] = df_filt.index + 1

st.markdown('<div class="whiff-section-label">League View</div>', unsafe_allow_html=True)
st.markdown("### Chase Leaderboard")
st.dataframe(df_filt[["Rank", "player_name", "ab", "swings", "whiffs", "avg_ei"]].rename(columns={"player_name": "Batter", "avg_ei": "Avg O-Zone EI"}), use_container_width=True, hide_index=True)

# --- WORST WHIFFERS (Restoring Batter Name) ---
st.markdown('<div class="whiff-section-label">Worst Swings</div>', unsafe_allow_html=True)
st.markdown("### Worst Whiffers")
worst_df = pitch_data[pitch_data["zone_split"] == "Out of Zone"].sort_values("embarrassment_index", ascending=False).head(25)
st.dataframe(
    worst_df[["batter_name", "player_name", "pitch_name", "count", "runners_on", "miss_distance_inches", "embarrassment_index"]].rename(
        columns={"batter_name": "Batter", "player_name": "Pitcher", "pitch_name": "Pitch Type", "miss_distance_inches": "Miss Dist (in)"}
    ), use_container_width=True, hide_index=True
)

# --- PLAYER BREAKDOWN ---
st.markdown('<div class="whiff-divider"></div>', unsafe_allow_html=True)
player_options = sorted(pitch_data["batter_name"].dropna().unique())
selected_player = st.sidebar.selectbox("Select Hitter", player_options, index=player_options.index("Shohei Ohtani") if "Shohei Ohtani" in player_options else 0)

player_whiffs = pitch_data[pitch_data["batter_name"] == selected_player].copy()
player_id = int(player_whiffs["batter"].iloc[0]) if not player_whiffs.empty else None

col_txt, col_img = st.columns([4, 1])
with col_txt:
    st.markdown(f"### Player Breakdown: {selected_player}")
with col_img:
    if player_id:
        st.image(f"https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{player_id}/headshot/67/current", width=150)

# Pruned Player Table
st.markdown(f"### {selected_player}'s Top Whiffs")
st.dataframe(
    player_whiffs[["player_name", "pitch_name", "zone_split", "runners_on", "miss_distance_inches", "embarrassment_index"]].sort_values("embarrassment_index", ascending=False).head(10).rename(
        columns={"player_name": "Pitcher", "miss_distance_inches": "Miss Dist (in)"}
    ), use_container_width=True, hide_index=True
)

# Strike Zone Visual (Restoring your original logic)
if not player_whiffs.empty:
    fig_sz = go.Figure(go.Scatter(
        x=player_whiffs["plate_x"], y=player_whiffs["plate_z"], mode="markers",
        marker=dict(size=12, color=player_whiffs["embarrassment_index"], colorscale="Cividis", showscale=True)
    ))
    fig_sz.add_shape(type="rect", x0=-0.708, x1=0.708, y0=player_whiffs["sz_bot"].mean(), y1=player_whiffs["sz_top"].mean(), line=dict(color="white", width=2))
    fig_sz.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(range=[-2.5, 2.5]), yaxis=dict(range=[0, 5]))
    st.plotly_chart(fig_sz, use_container_width=True)