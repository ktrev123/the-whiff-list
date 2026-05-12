import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


@st.cache_data
def load_leaderboard_data():
    df = pd.read_csv("data/whiff_leaderboard_test.csv")
    df["player_name"] = df["player_name"].str.title()
    return df


@st.cache_data
def load_pitch_data():
    return pd.read_csv("data/statcast_test_2025_04_01_to_04_07.csv", low_memory=False)


def last_first_to_first_last(name):
    if isinstance(name, str) and "," in name:
        parts = [part.strip() for part in name.split(",", 1)]
        return f"{parts[1]} {parts[0]}"
    return name


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

view = st.radio(
    "View",
    ["Leaderboard", "Player Breakdown"],
    horizontal=True
)

if view == "Leaderboard":
    df = load_leaderboard_data().copy()

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

elif view == "Player Breakdown":
    df = load_leaderboard_data().copy()
    pitch_data = load_pitch_data().copy()

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
    pitch_data["player_name"] = pitch_data["player_name"].str.title()
    pitch_data["player_name"] = pitch_data["player_name"].apply(last_first_to_first_last)

    player_options = sorted(pitch_data["batter_name"].dropna().unique())

    selected_player = st.sidebar.selectbox(
    "Player Breakdown",
    player_options,
    key="selected_player_breakdown")

    player_whiffs = pitch_data[pitch_data["batter_name"] == selected_player].copy()
    player_whiffs = player_whiffs.sort_values("miss_distance", ascending=False)

    st.markdown(f"### Player Breakdown: {selected_player}")
    st.write("These are the worst swing-and-miss pitches by distance from the strike zone.")

    selected_player_id = (
        int(player_whiffs["batter"].dropna().iloc[0])
        if not player_whiffs.empty else None
    )

    if selected_player_id:
        headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{selected_player_id}/headshot/67/current"

    left_spacer, image_col, right_col = st.columns([1.2, 1, 4])

    with image_col:
        st.image(headshot_url, width=140)

    st.dataframe(
        player_whiffs[
            [
                "game_date",
                "player_name",
                "pitch_name",
                "miss_distance_inches"
            ]
        ]
        .rename(
            columns={
                "game_date": "Date",
                "player_name": "Pitcher",
                "pitch_name": "Pitch Type",
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
                colorscale="Cividis",
                showscale=True,
                colorbar=dict(title="Miss (in)")
            ),
            customdata=player_whiffs[
                ["player_name", "pitch_name", "game_date", "miss_distance_inches"]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}'s %{customdata[1]}</b><br>"
                "Date: %{customdata[2]}<br>"
                "Miss Distance: %{customdata[3]} in"
                "<extra></extra>"
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