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
    An interactive Statcast dashboard built with pybaseball and Streamlit
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
    min_value=25,
    max_value=400,
    value=100,
    step=25
)

view_type = st.sidebar.selectbox(
    "View",
    options=["Leaderboard", "Player breakdown"],
    index=0
)

sample_data = pd.DataFrame(
    {
        "Rank": [1, 2, 3, 4, 5],
        "Player": ["Player A", "Player B", "Player C", "Player D", "Player E"],
        "Team": ["NYY", "OAK", "CWS", "COL", "MIA"],
        "Swings": [212, 198, 245, 176, 201],
        "Whiffs": [89, 80, 95, 67, 75],
        "Whiff Rate": [0.420, 0.404, 0.388, 0.381, 0.373]
    }
)

filtered_data = sample_data[sample_data["Swings"] >= min_swings].copy()
filtered_data["Whiff Rate"] = (filtered_data["Whiff Rate"] * 100).round(1).astype(str) + "%"

col1, col2, col3 = st.columns(3)
col1.metric("Players shown", len(filtered_data))
col2.metric("Season", season)
col3.metric("Minimum swings", min_swings)

st.markdown("### Hall of Shame leaderboard")

st.dataframe(
    filtered_data,
    use_container_width=True,
    hide_index=True
)

st.markdown("### Notes")
st.write("- This is currently sample data for layout testing.")
st.write("- Next step: replace placeholders with real 2025 Statcast swing-and-miss data.")
st.write("- Future filters will include pitch type, handedness, and count.")