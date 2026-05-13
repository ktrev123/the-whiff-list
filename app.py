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

view = st.radio(
    "View",
    ["Leaderboard", "Player Breakdown"],
    horizontal=True
)

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
pitch_data["zone_split"] = np.where(
    pitch_data["miss_distance"] == 0,
    "In Zone",
    "Out of Zone"
)

pitch_data["batter_name"] = pitch_data["batter_name"].str.title()
pitch_data["player_name"] = pitch_data["player_name"].str.title()
pitch_data["player_name"] = pitch_data["player_name"].apply(last_first_to_first_last)

for base_col in ["on_1b", "on_2b", "on_3b"]:
    if base_col not in pitch_data.columns:
        pitch_data[base_col] = np.nan

pitch_data["runners_on"] = (
    pitch_data["on_1b"].notna().astype(int) +
    pitch_data["on_2b"].notna().astype(int) +
    pitch_data["on_3b"].notna().astype(int)
)

if all(col in pitch_data.columns for col in ["game_pk", "at_bat_number", "pitch_number"]):
    pitch_data = pitch_data.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    ).copy()

    pitch_data["prev_whiff"] = (
        pitch_data.groupby(["game_pk", "at_bat_number"])["description"]
        .shift(1)
        .isin(whiff_descriptions)
        .astype(int)
    )
else:
    pitch_data["prev_whiff"] = 0

pitch_data["embarrassment_index"] = (
    10
    + np.where(pitch_data["zone_split"] == "Out of Zone", 10, 0)
    + (pitch_data["runners_on"] * 6)
    + np.where(pitch_data["prev_whiff"] == 1, 10, 0)
    + (pitch_data["miss_distance_inches"] * 1.25)
).round(1)

avg_embarrassment = (
    pitch_data.groupby("batter", as_index=False)["embarrassment_index"]
    .mean()
    .rename(columns={"embarrassment_index": "avg_embarrassment_index"})
)

df = df.merge(avg_embarrassment, on="batter", how="left")
df["avg_embarrassment_index"] = df["avg_embarrassment_index"].round(1)

if view == "Leaderboard":
    df_filtered = df[
        df["swings"] >= min_swings
    ].copy()

    df_filtered["whiff_rate_pct"] = (df_filtered["whiff_rate"] * 100).round(1)

    df_filtered = df_filtered.sort_values(
        ["avg_embarrassment_index", "whiff_rate", "whiffs"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    df_filtered["rank_display"] = df_filtered.index + 1

    df_filtered = df_filtered.rename(
        columns={
            "rank_display": "Rank",
            "player_name": "Batter",
            "ab": "AB",
            "swings": "Swings",
            "whiffs": "Whiffs",
            "whiff_rate_pct": "Whiff Rate (%)",
            "avg_embarrassment_index": "Avg Embarrassment Index"
        }
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Players shown", len(df_filtered))
    col2.metric("Season", season)
    col3.metric("Minimum swings", min_swings)

    st.markdown("### Leaderboard")

    st.dataframe(
        df_filtered[
            ["Rank", "Batter", "AB", "Swings", "Whiffs", "Whiff Rate (%)", "Avg Embarrassment Index"]
        ],
        use_container_width=True,
        hide_index=True
    )

elif view == "Player Breakdown":
    player_options = sorted(pitch_data["batter_name"].dropna().unique())

    if "selected_player_name" not in st.session_state:
        st.session_state.selected_player_name = player_options[0]

    if st.session_state.selected_player_name not in player_options:
        st.session_state.selected_player_name = player_options[0]

    selected_player = st.sidebar.selectbox(
        "Player Breakdown",
        player_options,
        key="selected_player_name"
    )

    player_whiffs = pitch_data[pitch_data["batter_name"] == selected_player].copy()

    pitch_type_options = sorted(player_whiffs["pitch_name"].dropna().unique())

    if "selected_pitch_types" not in st.session_state:
        st.session_state.selected_pitch_types = pitch_type_options

    st.session_state.selected_pitch_types = [
        p for p in st.session_state.selected_pitch_types if p in pitch_type_options
    ]

    if not st.session_state.selected_pitch_types:
        st.session_state.selected_pitch_types = pitch_type_options

    selected_pitch_types = st.sidebar.multiselect(
        "Pitch Type",
        options=pitch_type_options,
        key="selected_pitch_types"
    )

    if selected_pitch_types:
        player_whiffs = player_whiffs[player_whiffs["pitch_name"].isin(selected_pitch_types)].copy()

    player_whiffs = player_whiffs.sort_values(
        ["embarrassment_index", "miss_distance"],
        ascending=[False, False]
    ).copy()

    in_zone_whiffs = player_whiffs[player_whiffs["zone_split"] == "In Zone"]
    out_zone_whiffs = player_whiffs[player_whiffs["zone_split"] == "Out of Zone"]

    selected_player_id = (
        int(player_whiffs["batter"].dropna().iloc[0])
        if not player_whiffs.empty else None
    )

    text_col, image_col = st.columns([4, 1])

    with text_col:
        st.markdown(f"### Player Breakdown: {selected_player}")
        st.write("These are the worst swing-and-miss pitches ranked by Embarrassment Index.")

    with image_col:
        if selected_player_id:
            headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{selected_player_id}/headshot/67/current"
            st.image(headshot_url, width=170)

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Total Whiffs", len(player_whiffs))
    metric2.metric("In-Zone Whiffs", len(in_zone_whiffs))
    metric3.metric("Out-of-Zone Whiffs", len(out_zone_whiffs))
    metric4.metric(
        "Avg Embarrassment",
        f"{player_whiffs['embarrassment_index'].mean():.1f}" if len(player_whiffs) > 0 else "0.0"
    )

    st.dataframe(
        player_whiffs[
            [
                "game_date",
                "player_name",
                "pitch_name",
                "zone_split",
                "runners_on",
                "prev_whiff",
                "miss_distance_inches",
                "embarrassment_index"
            ]
        ]
        .rename(
            columns={
                "game_date": "Date",
                "player_name": "Pitcher",
                "pitch_name": "Pitch Type",
                "zone_split": "Zone Split",
                "runners_on": "Runners On",
                "prev_whiff": "Prev Pitch Whiff",
                "miss_distance_inches": "Miss Distance (in)",
                "embarrassment_index": "Embarrassment Index"
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
                color=player_whiffs["embarrassment_index"],
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Embarrassment Index")
            ),
            customdata=player_whiffs[
                [
                    "player_name",
                    "pitch_name",
                    "game_date",
                    "miss_distance_inches",
                    "zone_split",
                    "runners_on",
                    "prev_whiff",
                    "embarrassment_index"
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}'s %{customdata[1]}</b><br>"
                "Date: %{customdata[2]}<br>"
                "Miss Distance: %{customdata[3]} in<br>"
                "Zone Split: %{customdata[4]}<br>"
                "Runners On: %{customdata[5]}<br>"
                "Prev Pitch Whiff: %{customdata[6]}<br>"
                "Embarrassment Index: %{customdata[7]}"
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
        "Embarrassment Index is a custom score that increases for out-of-zone whiffs, "
        "more runners on base, consecutive whiffs within the same at-bat, "
        "and larger miss distance from the strike zone."
    )