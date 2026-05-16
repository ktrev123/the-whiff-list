import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
h1, h2, h3 { letter-spacing: -0.02em; color: var(--whiff-cream); }
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--whiff-navy-2) 0%, var(--whiff-navy) 100%);
    border: 1px solid var(--whiff-border); border-radius: 16px;
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

# GLOBAL AB FILTER: Batting Title Qualifiers Only
df_base = df_base[df_base["ab"] >= 502].copy()

# Hard Date Cutoff (Mar 23 - Sept 27)
pitch_data = pitch_data[(pitch_data["game_date"] >= "2025-03-23") & (pitch_data["game_date"] <= "2025-09-27")].copy()

whiff_desc = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
pitch_data = pitch_data[pitch_data["description"].isin(whiff_desc)].copy()
pitch_data = pitch_data.dropna(subset=["batter", "plate_x", "plate_z", "sz_top", "sz_bot"])

# Inner join filters pitch records down to validated 502 AB hitters
name_lookup = df_base[["batter", "player_name"]].drop_duplicates().rename(columns={"player_name": "batter_name"})
pitch_data = pitch_data.merge(name_lookup, on="batter", how="inner")

pitch_data["batter_name"] = pitch_data["batter_name"].str.title()
pitch_data["player_name"] = pitch_data["player_name"].str.title().apply(last_first_to_first_last)
pitch_data["miss_dist_in"] = (pitch_data.apply(calculate_miss_distance, axis=1) * 12).round(1)
pitch_data["zone_split"] = np.where(pitch_data["miss_dist_in"] == 0, "In Zone", "Out of Zone")
pitch_data["runners_on"] = pitch_data[["on_1b", "on_2b", "on_3b"]].notna().sum(axis=1)
pitch_data["count"] = pitch_data["balls"].fillna(0).astype(int).astype(str) + "-" + pitch_data["strikes"].fillna(0).astype(int).astype(str)

# Scaled formula out of 100 max capacity
pitch_data["ei"] = ((100 / 0.85) * (0.45 * np.minimum(pitch_data["miss_dist_in"]/18, 1.0) + 
                                    0.20 * (pitch_data["zone_split"] == "Out of Zone").astype(float) + 
                                    0.20 * (pitch_data["runners_on"]/3.0))).round(1)

# --- HEADER ---
st.title("The Whiff List 💨")
st.markdown('<div class="whiff-subtle">Tracking the ugliest chase whiffs and repeat flails from the 2025 MLB season.</div>', unsafe_allow_html=True)

st.info("📋 **Qualification Note:** The entire dataset is globally filtered to include only hitters with at least **502 At-Bats**, matching Major League Baseball's structural requirement for the season batting title.")

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
    colorbar=dict(title="Whiff Density")
))
fig_heat.add_shape(type="rect", x0=-0.708, x1=0.708, y0=1.6, y1=3.4, line=dict(color="#f5efe3", width=3))
fig_heat.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(range=[-3, 3], title="Horizontal Plate Location (ft)", scaleanchor="y", scaleratio=1),
    yaxis=dict(range=[0, 4.5], title="Vertical Plate Location (ft)"),
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

# --- PLATOON SPLITS BAR CHART ---
st.markdown('<div class="whiff-section-label">Platoon Splits</div>', unsafe_allow_html=True)
st.markdown("### Avg Embarrassment Index by Pitch Type")
splits_df = pitch_data[pitch_data["zone_split"] == "Out of Zone"].copy()
splits_df["Handedness"] = splits_df["stand"].map({"L": "Left", "R": "Right"})
main_pitches = ["4-Seam Fastball", "Slider", "Changeup", "Curveball", "Sinker", "Cutter", "Sweeper"]
pitch_splits = splits_df[splits_df["pitch_name"].isin(main_pitches)].groupby(["pitch_name", "Handedness"])["ei"].mean().reset_index()

fig_splits = go.Figure()
for hand in ["Left", "Right"]:
    hand_df = pitch_splits[pitch_splits["Handedness"] == hand]
    fig_splits.add_trace(go.Bar(x=hand_df["pitch_name"], y=hand_df["ei"], name=f"{hand}-Handed Hitters", marker_color="#20908d" if hand == "Left" else "#440154"))
fig_splits.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", barmode="group", yaxis_title="Avg EI", height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_splits, use_container_width=True)

# --- LEADERBOARD & METRICS ---
avg_ei_ozone = pitch_data[pitch_data["zone_split"] == "Out of Zone"].groupby("batter")["ei"].mean().reset_index().rename(columns={"ei": "avg_ei"})
df_lb = df_base.merge(avg_ei_ozone, on="batter", how="left").sort_values("avg_ei", ascending=False).reset_index(drop=True) # Removed min_swings logic
df_lb["Rank"] = df_lb.index + 1
df_lb["Whiff%"] = (df_lb["whiff_rate"] * 100).round(1)

st.markdown('<div class="whiff-section-label">League View</div>', unsafe_allow_html=True)
st.markdown("### Chase Leaderboard")

# Pruned down to two clean layout tracking metrics
m_col1, m_col2 = st.columns(2)
m_col1.metric("Qualified Hitters", len(df_lb))
m_col2.metric("Season Window", "Mar 23 - Sep 27")

lb_display = df_lb[["Rank", "player_name", "ab", "swings", "whiffs", "Whiff%", "avg_ei"]].rename(columns={"player_name": "Batter", "ab": "ABs", "avg_ei": "Avg EI"})
st.dataframe(lb_display, use_container_width=True, hide_index=True)

# --- WORST WHIFFERS ---
st.markdown('<div class="whiff-section-label">Worst Swings</div>', unsafe_allow_html=True)
st.markdown("### Worst Whiffers")
worst_df = pitch_data[pitch_data["zone_split"] == "Out of Zone"].sort_values("ei", ascending=False).head(25)
st.dataframe(worst_df[["batter_name", "player_name", "pitch_name", "count", "runners_on", "miss_dist_in", "ei"]].rename(
    columns={"batter_name": "Batter", "player_name": "Pitcher", "pitch_name": "Pitch Type", "count": "Count", "runners_on": "Runners On", "miss_dist_in": "Miss Dist (in)", "ei": "EI"}
), use_container_width=True, hide_index=True)

# --- PLAYER BREAKDOWN ---
st.markdown('<div class="whiff-divider"></div>', unsafe_allow_html=True)
p_list = sorted(pitch_data["batter_name"].dropna().unique())

# On-page filter configuration selector
selected_player_box = st.selectbox("Select Hitter for Performance Breakdown", p_list, index=p_list.index("Shohei Ohtani") if "Shohei Ohtani" in p_list else 0)
p_whiffs = pitch_data[pitch_data["batter_name"] == selected_player_box].copy()

c_t, c_i = st.columns([4, 1])
with c_t: st.markdown(f"### Player Breakdown: {selected_player_box}")
with c_i:
    pid = int(p_whiffs["batter"].iloc[0]) if not p_whiffs.empty else None
    if pid: st.image(f"https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{pid}/headshot/67/current", width=150)

# FIXED: Individual Strike Zone visual rendered BEFORE the raw dataframe
if not p_whiffs.empty:
    fig_sz = go.Figure()
    fig_sz.add_trace(go.Scatter(
        x=p_whiffs["plate_x"], y=p_whiffs["plate_z"], mode="markers", 
        marker=dict(size=11, color=p_whiffs["ei"], colorscale="Viridis", showscale=True, colorbar=dict(title="EI")),
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
    fig_sz.add_shape(type="rect", x0=-0.708, x1=0.708, y0=avg_bot, y1=avg_top, line=dict(color="#f5efe3", width=3))
    
    # FIXED: Added explicit margin adjustments to pull chart up closer to headshots
    fig_sz.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
        xaxis=dict(range=[-3, 3], title="Horizontal (ft)", scaleanchor="y", scaleratio=1), 
        yaxis=dict(range=[0, 4.5], title="Vertical (ft)"), 
        height=750,
        margin=dict(t=10, b=10, l=0, r=0)
    )
    st.plotly_chart(fig_sz, use_container_width=True)

# FIXED: Player View DataFrame tables display at the absolute bottom of the script layout
st.markdown('<div class="whiff-section-label">Player View</div>', unsafe_allow_html=True)
st.markdown(f"### {selected_player_box}'s Top Whiffs")
st.dataframe(p_whiffs[["player_name", "pitch_name", "count", "runners_on", "miss_dist_in", "ei"]].sort_values("ei", ascending=False).head(10).rename(
    columns={"player_name": "Pitcher", "pitch_name": "Pitch Type", "count": "Count", "runners_on": "Runners On", "miss_dist_in": "Miss Dist (in)", "ei": "EI"}
), use_container_width=True, hide_index=True)