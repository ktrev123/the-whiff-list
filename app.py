import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load leaderboard
leaderboard = pd.read_csv("data/whiff_leaderboard_test.csv")
leaderboard["player_name"] = leaderboard["player_name"].str.title()

# Load pitch-level data
pitches = pd.read_csv("data/statcast_test_2025_04_01_to_04_07.csv")

whiff_descriptions = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt"
}

pitches = pitches[pitches["description"].isin(whiff_descriptions)].copy()

# Keep only rows with needed coordinates
pitches = pitches.dropna(subset=["batter", "plate_x", "plate_z", "sz_top", "sz_bot"])

# Add player names from leaderboard lookup
name_lookup = leaderboard[["batter", "player_name"]].drop_duplicates()
pitches = pitches.merge(name_lookup, on="batter", how="left")

# Strike zone distance function
def miss_distance(row):
    x = row["plate_x"]
    z = row["plate_z"]
    left_edge = -0.708
    right_edge = 0.708
    bot_edge = row["sz_bot"]
    top_edge = row["sz_top"]

    if x < left_edge:
        x_out = left_edge - x
    elif x > right_edge:
        x_out = x - right_edge
    else:
        x_out = 0

    if z < bot_edge:
        z_out = bot_edge - z
    elif z > top_edge:
        z_out = z - top_edge
    else:
        z_out = 0

    return np.sqrt(x_out**2 + z_out**2)

pitches["miss_distance"] = pitches.apply(miss_distance, axis=1)
pitches["miss_distance_inches"] = (pitches["miss_distance"] * 12).round(1)

player_options = sorted(pitches["player_name"].dropna().unique())
selected_player = st.sidebar.selectbox("Player breakdown", player_options)

player_whiffs = pitches[pitches["player_name"] == selected_player].copy()
player_whiffs = player_whiffs.sort_values("miss_distance", ascending=False)

st.markdown(f"### Player breakdown: {selected_player}")
st.write("Worst swing-and-miss pitches by distance from the strike zone.")

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
    ].rename(
        columns={
            "game_date": "Date",
            "pitch_name": "Pitch",
            "plate_x": "Plate X",
            "plate_z": "Plate Z",
            "sz_top": "SZ Top",
            "sz_bot": "SZ Bot",
            "miss_distance_inches": "Miss Distance (in)"
        }
    ).head(10),
    use_container_width=True,
    hide_index=True
)

# Plot whiffs
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
            colorbar=dict(title="Miss in")
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

# Strike zone rectangle
fig.add_shape(
    type="rect",
    x0=-0.708, x1=0.708,
    y0=avg_bot, y1=avg_top,
    line=dict(color="white", width=2)
)

fig.update_layout(
    title=f"{selected_player} Whiff Locations",
    xaxis_title="plate_x",
    yaxis_title="plate_z",
    xaxis=dict(range=[-2.5, 2.5]),
    yaxis=dict(range=[0, 5]),
    height=600
)

st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})