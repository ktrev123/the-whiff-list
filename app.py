import streamlit as st

st.set_page_config(
    page_title="The Whiff List",
    page_icon="💨",
    layout="wide"
)

st.title("The Whiff List 💨")
st.subheader("2025 MLB Swing-and-Miss Offenders")
st.write(
    """
    Welcome to The Whiff List — a tongue-in-cheek Statcast dashboard
    built to find the biggest swing-and-miss offenders of the 2025 MLB season.
    """
)

st.markdown("### Coming soon")
st.write("- Pull 2025 Statcast pitch data")
st.write("- Build a whiff leaderboard")
st.write("- Add filters for pitch type, handedness, and minimum swings")
st.write("- Add player detail pages")
