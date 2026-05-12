import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="The Whiff List",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("The Whiff List 💨")
st.subheader("2025 MLB Swing-and-Miss Offenders")
st.write(
    """
    An interactive Statcast dashboard built with Streamlit
    to uncover the season's biggest swing-and-miss offenders.
    """
)

st.sidebar.header("Filters")

season = st.sidebar.selectbox(
    "Season",
    options=[2025],
    index=0
)

min_swings = st.sidebar.slider(
    "Minimum swings",
    min_value=0,
    max_value=60,
    value=10,
    step=5
)

view_type = st.sidebar.selectbox(
    "View",
    options=["Leaderboard"],
    index=0
)

df = pd.read_csv("data/whiff_leaderboard_test.csv")
df["player_name"] = df["player_name"].str.title()


df_filtered = df[df["swings"] >= min_swings].copy()
df_filtered["whiff_rate_pct"] = (df_filtered["whiff_rate"] * 100).round(1)

df_filtered = df_filtered.rename(
    columns={
        "rank": "Rank",
        "player_name": "Batter",
        "swings": "Swings",
        "whiffs": "Whiffs",
        "whiff_rate_pct": "Whiff Rate (%)"
    }
)

col1, col2, col3 = st.columns(3)
col1.metric("Players shown", len(df_filtered))
col2.metric("Season", season)
col3.metric("Minimum swings", min_swings)

st.markdown("### Hall of Shame leaderboard")

st.dataframe(
    df_filtered[["Rank", "Batter", "Swings", "Whiffs", "Whiff Rate (%)"]],
    use_container_width=True,
    hide_index=True
)

st.markdown("### Notes")
st.write(
    "Swings include fouls, balls in play, and swinging strikes; "
    "whiffs include swinging strikes and missed bunts. "
    "Batter ID is the MLBAM identifier used in Statcast and pybaseball."
)