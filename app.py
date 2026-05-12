import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

min_ab = st.sidebar.slider(
    "Minimum AB",
    min_value=0,
    max_value=50,
    value=10,
    step=5
)

# -----------------------------
# Leaderboard data
# -----------------------------
df = pd.read_csv("data/whiff_leaderboard_test.csv")
st.write("Leaderboard columns:", df.columns.tolist())
st.write(df.head())

if "player_name" in df.columns:
    df["player_name"] = df["player_name"].str.title()
else:
    st.write("player_name column not found")

df_filtered = df[
    (df["swings"] >= min_swings) &
    (df["ab"] >= min_ab)
].copy()

df_filtered["whiff_rate_pct"] = (df_filtered["whiff_rate"] * 100).round(1)

df_filtered = df_filtered.sort_values(
    ["whiff_rate", "whiffs"],
    ascending=[False, False]
).reset_index(drop=True)

df_filtered["rank_display"] = df_filtered.index + 1

df_filtered = df_filtered.rename(
    columns={
        "rank_display": "Rank",
        "player_name": "Batter",
        "ab": "AB",
        "swings": "Swings",
        "whiffs": "Whiffs",
        "whiff_rate_pct": "Whiff Rate (%)"
    }
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Players shown", len(df_filtered))
col2.metric("Season", season)
col3.metric("Minimum swings", min_swings)
col4.metric("Minimum AB", min_ab)

st.markdown("### Leaderboard")

st.dataframe(
    df_filtered[["Rank", "Batter", "AB", "Swings", "Whiffs", "Whiff Rate (%)"]],
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Player Breakdown
# -----------------------------
pitch_data = pd.read_csv("data/statcast_test_2025_04_01_to_04_07.csv")

whiff_descriptions = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt"
}

pitch_data = pitch_data[pitch_data["description"].isin(whiff_descriptions)].copy()
pitch_data = pitch_data.dropna(subset=["batter", "plate_x", "plate_z", "sz_top", "sz_bot"])

name_lookup = (
    df[["batter", "player_name"]]
    .drop_duplicates()
    .rename(columns={"player_name": "batter_name"})
)

pitch_data = pitch_data.merge(name_lookup, on="batter", how="left")

def calculate_miss_distance(row):
    x = row["plate_x"]
    z = row["plate_z"]
    left_edge = -0.708
    right_edge = 0.708
    bottom_edge = row["sz_bot"]
    top_edge = row["sz_top"]

    if x < left_edge:
        x_out = left_edge - x
    elif x > right_edge:
        x_out = x - right_edge
    else:
        x_out = 0

    if z < bottom_edge:
        z_out = bottom_edge - z
    elif z > top_edge:
        z_out = z - top_edge
    else:
        z_out = 0

    return np.sqrt((x_out ** 2) + (z_out ** 2))

pitch_data["miss_distance"] = pitch_data.apply(calculate_miss_distance, axis=1)
pitch_data["miss_distance_inches"] = (pitch_data["miss_distance"] * 12).round(1)
pitch_data["batter_name"] = pitch_data["batter_name"].str.title()

player_options = sorted(pitch_data["batter_name"].dropna().unique())

selected_player = st.sidebar.selectbox(
    "Player Breakdown",
    player_options
)

player_whiffs = pitch_data[pitch_data["batter_name"] == selected_player].copy()
player_whiffs = player_whiffs.sort_values("miss_distance", ascending=False)

st.markdown(f"### Player Breakdown: {selected_player}")
st.write("These are the worst swing-and-miss pitches by distance from the strike zone.")

st.dataframe(
    player_whiffs[
        [
            "game_date",
            "pitch_name",
            "plate_x",
            "plate_z",
            "sz_top",
            "sz_bot",
            "miss_distance_inches"
        ]
    ]
    .rename(
        columns={
            "game_date": "Date",
            "pitch_name": "Pitch Type",
            "plate_x": "Plate X",
            "plate_z": "Plate Z",
            "sz_top": "SZ Top",
            "sz_bot": "SZ Bot",
            "miss_distance_inches": "Miss Distance (in)"
        }
    )
    .head(10),
    use_container_width=True,
    hide_index=True
)

avg_top = player_whiffs["sz_top"].mean()
avg_bot = player_whiffs["sz_bot"].mean()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=player_whiffs["plate_x"],
        y=player_whiffs["plate_z"],
        mode="markers",
        marker=dict(
            size=10,
            color=player_whiffs["miss_distance_inches"],
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Miss (in)")
        ),
        text=player_whiffs["pitch_name"],
        customdata=player_whiffs[["game_date", "miss_distance_inches"]],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Date: %{customdata[0]}<br>"
            "Miss Distance: %{customdata[1]} in<br>"
            "plate_x: %{x:.2f}<br>"
            "plate_z: %{y:.2f}<extra></extra>"
        )
    )
)

fig.add_shape(
    type="rect",
    x0=-0.708,
    x1=0.708,
    y0=avg_bot,
    y1=avg_top,
    line=dict(color="white", width=2)
)

fig.update_layout(
    title=f"{selected_player} Whiff Locations",
    xaxis_title="Horizontal Location (plate_x)",
    yaxis_title="Vertical Location (plate_z)",
    xaxis=dict(range=[-2.5, 2.5]),
    yaxis=dict(range=[0, 5]),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Notes")
st.write(
    "Swings include fouls, balls in play, and swinging strikes; "
    "whiffs include swinging strikes and missed bunts. "
    "Player Breakdown ranks whiffs by estimated distance from the strike zone."
)