import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIG & STYLING ---
st.set_page_config(page_title="The Whiff List", page_icon="💨", layout="wide")

st.markdown("""
<style>
    :root { --whiff-navy: #0f172a; --whiff-cream: #f5efe3; --whiff-gold: #d4a937; --whiff-red: #c24141; }
    .stMetric { background: #162033; border: 1px solid rgba(245,239,227,0.1); border-radius: 12px; padding: 15px; }
    .whiff-section-label { color: var(--whiff-gold); font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_parquet("data/statcast_2025.parquet")
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['month'] = df['game_date'].dt.month_name()
    # Normalize name case
    df['player_name'] = df['player_name'].str.title()
    return df

try:
    raw_data = load_data()
except FileNotFoundError:
    st.error("Data file 'data/statcast_2025.parquet' not found. Please run your ingestion script first.")
    st.stop()

# --- CONSTANTS & LOGIC ---
SWING_DESCS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", 
               "hit_into_play", "hit_into_play_no_out", "hit_into_play_score", "foul_bunt", "missed_bunt"}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
AB_EVENTS = {"single", "double", "triple", "home_run", "field_out", "field_error", 
             "grounded_into_double_play", "force_out", "double_play", "fielders_choice", "strikeout", "strikeout_double_play"}

def calculate_ei(df):
    """Calculates the Embarrassment Index (0-100) for Whiffs."""
    # 1. Miss Distance
    lx, rx = -0.708, 0.708
    x_dist = np.maximum(0, np.maximum(lx - df['plate_x'], df['plate_x'] - rx))
    z_dist = np.maximum(0, np.maximum(df['sz_bot'] - df['plate_z'], df['plate_z'] - df['sz_top']))
    df['miss_dist_in'] = (np.sqrt(x_dist**2 + z_dist**2) * 12).round(2)
    
    # 2. O-Zone Flag
    df['is_ozone'] = (df['miss_dist_in'] > 0).astype(int)
    
    # 3. Runners On
    df['runners_count'] = df[['on_1b', 'on_2b', 'on_3b']].notna().sum(axis=1)
    
    # 4. Consecutive Whiff (P-Flag)
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
    df['prev_desc'] = df.groupby(['game_pk', 'at_bat_number'])['description'].shift(1)
    df['prev_ozone'] = df.groupby(['game_pk', 'at_bat_number'])['is_ozone'].shift(1)
    df['p_flag'] = ((df['prev_desc'].isin(WHIFF_DESCS)) & (df['prev_ozone'] == 1)).astype(int)
    
    # EI Components
    d_score = np.minimum(df['miss_dist_in'] / 18, 1.0)
    z_score = df['is_ozone']
    p_score = df['p_flag']
    r_score = df['runners_count'] / 3.0
    
    df['ei'] = (100 * (0.45*d_score + 0.20*z_score + 0.15*p_score + 0.20*r_score)).round(1)
    return df

# --- SIDEBAR FILTERS ---
st.sidebar.header("Navigation & Filters")
view = st.sidebar.radio("View", ["Hall of Shame", "Player Breakdown"])

months = ["March", "April", "May", "June", "July", "August", "September", "October"]
selected_months = st.sidebar.multiselect("Months", months, default=["April", "May"])

min_swings = st.sidebar.slider("Min Swings", 0, 100, 20)
min_abs = st.sidebar.slider("Min ABs", 0, 50, 10)

# --- DATA PROCESSING ---
filtered_df = raw_data[raw_data['month'].isin(selected_months)].copy()
filtered_df['is_swing'] = filtered_df['description'].isin(SWING_DESCS)
filtered_df['is_whiff'] = filtered_df['description'].isin(WHIFF_DESCS)
filtered_df['is_ab'] = filtered_df['events'].isin(AB_EVENTS)

# Calculate EI only for whiffs to save compute
whiffs_only = calculate_ei(filtered_df[filtered_df['is_whiff']].copy())

# Build Dynamic Leaderboard
leaderboard = filtered_df.groupby(['batter', 'player_name']).agg(
    AB=('is_ab', 'sum'),
    Swings=('is_swing', 'sum'),
    Whiffs=('is_whiff', 'sum')
).reset_index()

# Merge Avg EI (O-Zone Only) from the whiffs_only df
avg_ei = whiffs_only[whiffs_only['is_ozone'] == 1].groupby('batter')['ei'].mean().reset_index(name='Avg O-Zone EI')
leaderboard = leaderboard.merge(avg_ei, on='batter', how='left').fillna(0)
leaderboard['Whiff Rate (%)'] = (leaderboard['Whiffs'] / leaderboard['Swings'] * 100).round(1)
leaderboard['Avg O-Zone EI'] = leaderboard['Avg O-Zone EI'].round(1)

# Apply volume filters
leaderboard = leaderboard[(leaderboard['Swings'] >= min_swings) & (leaderboard['AB'] >= min_abs)]

# --- PLAYER SELECTION LOGIC ---
player_list = sorted(leaderboard['player_name'].unique())
if "selected_player_name" not in st.session_state:
    st.session_state.selected_player_name = player_list[0] if player_list else ""

selected_player = st.sidebar.selectbox("Select Player", player_list, index=player_list.index(st.session_state.selected_player_name) if st.session_state.selected_player_name in player_list else 0)
st.session_state.selected_player_name = selected_player

# --- VIEW: HALL OF SHAME ---
if view == "Hall of Shame":
    st.title("The Whiff List: Hall of Shame 💨")
    st.markdown("### League Chase Leaderboard")
    
    sort_col = st.selectbox("Sort By", ["Avg O-Zone EI", "Whiff Rate (%)", "Whiffs"], index=0)
    st.dataframe(leaderboard.sort_values(sort_col, ascending=False), use_container_width=True, hide_index=True)

    st.markdown("### Top 25 Most Embarrassing Individual Whiffs")
    hos_table = whiffs_only.sort_values('ei', ascending=False).head(25)
    st.dataframe(hos_table[['player_name', 'pitch_name', 'game_date', 'ei', 'miss_dist_in', 'runners_count']]
                 .rename(columns={'player_name': 'Batter', 'ei': 'EI', 'miss_dist_in': 'Miss (in)'}), 
                 use_container_width=True, hide_index=True)

# --- VIEW: PLAYER BREAKDOWN ---
else:
    st.title(f"Player Breakdown: {selected_player}")
    
    p_whiffs = whiffs_only[whiffs_only['player_name'] == selected_player].copy()
    p_pitch_types = st.sidebar.multiselect("Pitch Types", sorted(p_whiffs['pitch_name'].unique()), default=sorted(p_whiffs['pitch_name'].unique()))
    p_whiffs = p_whiffs[p_whiffs['pitch_name'].isin(p_pitch_types)]

    col1, col2 = st.columns([3, 1])
    
    with col1:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Whiffs", len(p_whiffs))
        m2.metric("In-Zone", len(p_whiffs[p_whiffs['is_ozone'] == 0]))
        m3.metric("Out-of-Zone", len(p_whiffs[p_whiffs['is_ozone'] == 1]))
        m4.metric("Avg O-Zone EI", leaderboard[leaderboard['player_name'] == selected_player]['Avg O-Zone EI'].values[0] if not p_whiffs.empty else 0)

        st.markdown("#### Recent Whiff Log")
        p_whiffs['Count'] = p_whiffs['balls'].astype(str) + "-" + p_whiffs['strikes'].astype(str)
        st.dataframe(p_whiffs[['game_date', 'pitch_name', 'Count', 'miss_dist_in', 'ei']]
                     .sort_values('game_date', ascending=False).head(10), use_container_width=True, hide_index=True)

    with col2:
        if not p_whiffs.empty:
            bid = int(p_whiffs['batter'].iloc[0])
            st.image(f"https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{bid}/headshot/67/current")

    st.markdown("---")
    st.markdown("### Whiff Location & Severity")
    if not p_whiffs.empty:
        fig = go.Figure()
        # Strike Zone
        z_top, z_bot = p_whiffs['sz_top'].median(), p_whiffs['sz_bot'].median()
        fig.add_shape(type="rect", x0=-0.708, y0=z_bot, x1=0.708, y1=z_top, line=dict(color="White", width=2))
        
        # Whiffs
        fig.add_trace(go.Scatter(
            x=p_whiffs['plate_x'], y=p_whiffs['plate_z'], mode='markers',
            marker=dict(size=12, color=p_whiffs['ei'], colorscale='Reds', showscale=True, colorbar=dict(title="EI")),
            text=p_whiffs['pitch_name'] + " | EI: " + p_whiffs['ei'].astype(str)
        ))
        
        fig.update_layout(template="plotly_dark", height=600, xaxis_range=[-2.5, 2.5], yaxis_range=[0, 5],
                          xaxis_title="Horizontal (ft)", yaxis_title="Vertical (ft)")
        st.plotly_chart(fig, use_container_width=True)