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
    --whiff-gold: #d4a937;
    --whiff-border: rgba(245, 239, 227, 0.10);
}
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; color: var(--whiff-cream); }
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--whiff-navy-2) 0%, var(--whiff-navy) 100%);
    border: 1px solid var(--whiff-border); border-radius: 16px;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--whiff-border); border-radius: 14px; overflow: hidden; }
.whiff-section-label { color: var(--whiff-gold); font-size: 0.86rem; font-weight: 700; text-transform: uppercase; margin-bottom: 0.25rem; }
.whiff-divider { height: 1px; width: 100%; background: linear-gradient(90deg, rgba(212,169,55,0) 0%, rgba(212,169,55,0.55) 50%, rgba(212,169,55,0) 100%); margin: 2rem 0 1.5rem 0; }
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

# --- DATA PROCESSING ---
df_base = load_leaderboard_data().copy()
pitch_data = load_pitch_data().copy()
pitch_data["game_date"] = pd.to_datetime(pitch_data["game_date"])

# Hard Date Cutoff (Mar 23 - Sept 27)
pitch_data = pitch_data[(pitch_data["game_date"] >= "2025-03-23") & (pitch_data["game_date"] <= "2025-09-27")].copy()

whiff_desc = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
pitch_data = pitch_data[pitch_data["description"].isin(whiff_desc)].copy()
pitch_data = pitch_data.dropna(subset=["batter", "plate_x", "plate_z", "sz_top", "sz_bot"])

# Global Logic
name_lookup = df_base[["batter", "player_name"]].drop_duplicates().rename(columns={"player_name": "batter_name"})
pitch_data = pitch_data.merge(name_lookup, on="batter", how="left")
pitch_data["batter_name"] = pitch_data["batter_name"].str.title()
pitch_data["player_name"] = pitch_data["player_name"].str.title().apply(last_first_to_first_last)
pitch_data["miss_dist_in"] = (pitch_data.apply(calculate_miss_distance, axis=1) * 12).round(1)
pitch_data["zone_split"] = np.where(pitch_data["miss_dist_in"] == 0, "In Zone", "Out of Zone")
pitch_data["runners_on"] = pitch_data[["on_1b", "on_2b", "on_3b"]].notna().sum(axis=1)
pitch_data["count"] = pitch_data["balls"].fillna(0).astype(int).astype(str) + "-" + pitch_data["strikes"].fillna(0).astype(int).astype(str)

# EI Score (100 * (0.45D + 0.20Z + 0.20R))
pitch_data["ei"] = (100 * (0.45 * np.minimum(pitch_data["miss_dist_in"]/18, 1.0) + 
                          0.20 * (pitch_data["zone_split"] == "Out of Zone").astype(float) + 
                          0.20 * (pitch_data["runners_on"]/3.0))).round(1)

# --- HEADER ---
st.title("The Whiff List 💨")

# --- TRENDS (7-Day Rolling) ---
st.markdown('<div class="whiff-section-label">Seasonal Trends</div>', unsafe_allow_html=True)
trend = pitch_data[pitch_data["zone_split"] == "Out of Zone"].groupby("game_date").agg(vol=("description", "count"), ei=("ei", "mean")).reset_index()
trend["vol_roll"] = trend["vol"].rolling(7).mean()
trend["ei_roll"] = trend["ei"].rolling(7).mean()

fig_t = go.Figure()
fig_t.add_trace(go.Scatter(x=trend["game_date"], y=trend["vol_roll"], name="7-Day Vol", line=dict(color="#c24141")))
fig_t.add_trace(go.Scatter(x=trend["game_date"], y=trend["ei_roll"], name="7-Day Avg EI", line=dict(color="#d4a937"), yaxis="y2"))
fig_t.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(title="Volume"), yaxis2=dict(title="Avg EI", overlaying="y", side="right"), height=300)
st.plotly_chart(fig_t, use_container_width=True)

# --- LEFT VS. RIGHT COMPARISON ---
st.markdown('<div class="whiff-section-label">Platoon Splits</div>', unsafe_allow_html=True)
st.markdown("### Avg Embarrassment Index by Pitch Type")

splits_df = pitch_data[pitch_data["zone_split"] == "Out of Zone"].copy()
splits_df["Handedness"] = splits_df["stand"].map({"L": "Left", "R": "Right"})

# Filter for primary pitch classifications to keep the x-axis clean
main_pitches = ["4-Seam Fastball", "Slider", "Changeup", "Curveball", "Sinker", "Cutter", "Sweeper"]
pitch_splits = splits_df[splits_df["pitch_name"].isin(main_pitches)].groupby(["pitch_name", "Handedness"])["ei"].mean().reset_index()

fig_splits = go.Figure()

for hand in ["Left", "Right"]:
    hand_df = pitch_splits[pitch_splits["Handedness"] == hand]
    fig_splits.add_trace(go.Bar(
        x=hand_df["pitch_name"],
        y=hand_df["ei"],
        name=f"{hand}-Handed Hitters",
        marker_color="#d4a937" if hand == "Left" else "#c24141"
    ))

fig_splits.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    barmode="group",
    yaxis_title="Avg Embarrassment Index",
    xaxis_title="Pitch Type",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=20, b=20)
)

st.plotly_chart(fig_splits, use_container_width=True)

# --- LEADERBOARD ---
st.sidebar.header("Filters")
min_swings = st.sidebar.slider("Min Swings", 0, 100, 10, 5)

avg_ei_ozone = pitch_data[pitch_data["zone_split"] == "Out of Zone"].groupby("batter")["ei"].mean().reset_index().rename(columns={"ei": "avg_ei"})
df_lb = df_base[df_base["swings"] >= min_swings].merge(avg_ei_ozone, on="batter", how="left").sort_values("avg_ei", ascending=False).reset_index(drop=True)
df_lb["Rank"] = df_lb.index + 1
df_lb["Whiff%"] = (df_lb["whiff_rate"] * 100).round(1)

st.markdown('<div class="whiff-section-label">League View</div>', unsafe_allow_html=True)
st.markdown("### Chase Leaderboard")
# Headers: Rank, Batter, ABs, Swings, Whiffs, Whiff%, Avg EI
lb_display = df_lb[["Rank", "player_name", "ab", "swings", "whiffs", "Whiff%", "avg_ei"]].rename(columns={"player_name": "Batter", "ab": "ABs", "avg_ei": "Avg EI"})
st.dataframe(lb_display, use_container_width=True, hide_index=True)

# --- WORST WHIFFERS ---
st.markdown('<div class="whiff-section-label">Worst Swings</div>', unsafe_allow_html=True)
st.markdown("### Worst Whiffers")
# Headers: Batter, Pitcher, Pitch Type, Count, Runners On, Miss Dist (in), EI
worst_df = pitch_data[pitch_data["zone_split"] == "Out of Zone"].sort_values("ei", ascending=False).head(25)
st.dataframe(worst_df[["batter_name", "player_name", "pitch_name", "count", "runners_on", "miss_dist_in", "ei"]].rename(
    columns={"batter_name": "Batter", "player_name": "Pitcher", "pitch_name": "Pitch Type", "count": "Count", "runners_on": "Runners On", "miss_dist_in": "Miss Dist (in)", "ei": "EI"}
), use_container_width=True, hide_index=True)

# --- PLAYER BREAKDOWN ---
st.markdown('<div class="whiff-divider"></div>', unsafe_allow_html=True)
p_list = sorted(pitch_data["batter_name"].dropna().unique())
sel_hitter = st.sidebar.selectbox("Select Hitter", p_list, index=p_list.index("Shohei Ohtani") if "Shohei Ohtani" in p_list else 0)
p_whiffs = pitch_data[pitch_data["batter_name"] == sel_hitter].copy()

c_t, c_i = st.columns([4, 1])
with c_t: 
    st.markdown(f"### Player Breakdown: {sel_hitter}")
with c_i:
    pid = int(p_whiffs["batter"].iloc[0]) if not p_whiffs.empty else None
    if pid: 
        st.image(f"https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{pid}/headshot/67/current", width=150)

# Headers: Pitcher, Pitch Type, Runners On, Miss Dist (in), EI
st.markdown('<div class="whiff-section-label">Player View</div>', unsafe_allow_html=True)
st.markdown(f"### {sel_hitter}'s Top Whiffs")
st.dataframe(p_whiffs[["player_name", "pitch_name", "runners_on", "miss_dist_in", "ei"]].sort_values("ei", ascending=False).head(10).rename(
    columns={"player_name": "Pitcher", "pitch_name": "Pitch Type", "runners_on": "Runners On", "miss_dist_in": "Miss Dist (in)", "ei": "EI"}
), use_container_width=True, hide_index=True)

# --- STRIKE ZONE VISUAL + TOOLTIPS ---
if not p_whiffs.empty:
    fig_sz = go.Figure()
    
# Custom Tooltips: Added EI to the final slot
    fig_sz.add_trace(go.Scatter(
        x=p_whiffs["plate_x"], 
        y=p_whiffs["plate_z"], 
        mode="markers", 
        marker=dict(size=11, color=p_whiffs["ei"], colorscale="Cividis", showscale=True, colorbar=dict(title="EI")),
        customdata=p_whiffs[["player_name", "pitch_name", "runners_on", "count", "miss_dist_in", "ei"]],
        hovertemplate=(
            "<b>%{customdata[0]}'s %{customdata[1]}</b><br>"
            "Runners On: %{customdata[2]}<br>"
            "Count: %{customdata[3]}<br>"
            "Miss Distance: %{customdata[4]} in<br>"
            "Embarrassment Index: %{customdata[5]}<br>"
            "<extra></extra>"
        )
    ))
    
    avg_bot, avg_top = p_whiffs["sz_bot"].mean(), p_whiffs["sz_top"].mean()
    # Main Rectangle: 1:1 Scaleratio for accurate proportions
    fig_sz.add_shape(type="rect", x0=-0.708, x1=0.708, y0=avg_bot, y1=avg_top, line=dict(color="#f5efe3", width=3))
    
    fig_sz.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-2.5, 2.5], title="Horizontal (ft)", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, 6], title="Vertical (ft)"),
        height=800 
    )
    st.plotly_chart(fig_sz, use_container_width=True)