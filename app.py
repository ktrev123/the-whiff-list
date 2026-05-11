import streamlit as st

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

st.markdown("### Current settings")
st.write(f"**Season:** {season}")
st.write(f"**Minimum swings:** {min_swings}")
st.write(f"**View:** {view_type}")

st.markdown("### Coming next")
st.write("- Pull 2025 Statcast pitch-level data with pybaseball")
st.write("- Identify swings and whiffs")
st.write("- Build the leaderboard of top offenders")
st.write("- Add player-level charts and pitch breakdowns")