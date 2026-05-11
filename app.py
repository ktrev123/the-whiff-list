import streamlit as st

st.set_page_config(
    page_title="The Whiff List",
    page_icon="💨 Test",
    layout="wide"
)

st.title("The Whiff List 💨")
st.subheader("2025 MLB Swing-and-Miss Offenders")
st.write(
    """
    An interactive Statcast dashboard built with pybaseball and Streamlit
    to uncover the season's biggest swing-and-miss offenders.
    """
)

st.markdown("### Planned features")
st.write("- Season leaderboard for worst whiff offenders")
st.write("- Filters for minimum swings, pitch type, handedness, and count")
st.write("- Player-level whiff breakdowns")
st.write("- A tongue-in-cheek 'Hall of Shame' style presentation")