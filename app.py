import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="The Whiff List", layout="wide")

st.title("The Whiff List")

@st.cache_data
def load_data():
    return pd.read_csv("all_ranked_whiffs.csv")

pitch_data = load_data()

required_columns = [
    "batter_name",
    "batter",
    "player_name",
    "pitch_name",
    "game_date",
    "zone_split",
    "runners_on",
    "prev_whiff",
    "miss_distance_inches",
    "embarrassment_index",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "miss_distance"
]

missing_cols = [col for col in required_columns if col not in pitch_data.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

pitch_data = pitch_data.dropna(subset=["batter_name"]).copy()

player_options = sorted(pitch_data["batter_name"].unique())

if not player_options:
    st.error("No players found in the dataset.")
    st.stop()

if "selected_player_name" not in st.session_state:
    st.session_state.selected_player_name = player_options[0]

if st.session_state.selected_player_name not in player_options:
    st.session_state.selected_player_name = player_options[0]

view = st.sidebar.radio(
    "Choose View",
    ["Leaderboard", "Player Breakdown"]
)

selected_player = st.sidebar.selectbox(
    "Player Breakdown",
    player_options,
    key="selected_player_name"
)

if view == "Leaderboard":
    st.subheader("Leaderboard")

    leaderboard_df = (
        pitch_data.groupby("batter_name", as_index=False)
        .agg(
            total_whiffs=("batter_name", "size"),
            avg_embarrassment=("embarrassment_index", "mean"),
            max_embarrassment=("embarrassment_index", "max")
        )
        .sort_values(["avg_embarrassment", "total_whiffs"], ascending=[False, False])
        .reset_index(drop=True)
    )

    st.dataframe(
        leaderboard_df.rename(
            columns={
                "batter_name": "Player",
                "total_whiffs": "Total Whiffs",
                "avg_embarrassment": "Avg Embarrassment",
                "max_embarrassment": "Max Embarrassment"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

elif view == "Player Breakdown":
    player_whiffs = pitch_data[pitch_data["batter_name"] == selected_player].copy()

    pitch_type_options = sorted(player_whiffs["pitch_name"].dropna().unique())

    current_pitch_key = f"selected_pitch_types::{selected_player}"

    if "pitch_type_state_by_player" not in st.session_state:
        st.session_state.pitch_type_state_by_player = {}

    if selected_player not in st.session_state.pitch_type_state_by_player:
        st.session_state.pitch_type_state_by_player[selected_player] = pitch_type_options

    saved_pitch_types = st.session_state.pitch_type_state_by_player[selected_player]
    saved_pitch_types = [p for p in saved_pitch_types if p in pitch_type_options]

    if not saved_pitch_types:
        saved_pitch_types = pitch_type_options

    if current_pitch_key not in st.session_state:
        st.session_state[current_pitch_key] = saved_pitch_types
    else:
        st.session_state[current_pitch_key] = [
            p for p in st.session_state[current_pitch_key] if p in pitch_type_options
        ]
        if not st.session_state[current_pitch_key]:
            st.session_state[current_pitch_key] = pitch_type_options

    selected_pitch_types = st.sidebar.multiselect(
        "Pitch Type",
        options=pitch_type_options,
        key=current_pitch_key
    )

    st.session_state.pitch_type_state_by_player[selected_player] = selected_pitch_types

    if selected_pitch_types:
        player_whiffs = player_whiffs[
            player_whiffs["pitch_name"].isin(selected_pitch_types)
        ].copy()

    player_whiffs = player_whiffs.sort_values(
        ["embarrassment_index", "miss_distance"],
        ascending=[False, False]
    ).copy()

    in_zone_whiffs = player_whiffs[player_whiffs["zone_split"] == "In Zone"]
    out_zone_whiffs = player_whiffs[player_whiffs["zone_split"] == "Out of Zone"]

    selected_player_id = (
        int(player_whiffs["batter"].dropna().iloc[0])
        if not player_whiffs.empty
        else None
    )

    text_col, image_col = st.columns([4, 1])

    with text_col:
        st.markdown(f"### Player Breakdown: {selected_player}")
        st.write("These are the worst swing-and-miss pitches ranked by Embarrassment Index.")

    with image_col:
        if selected_player_id:
            headshot_url = (
                f"https://img.mlbstatic.com/mlb-photos/image/upload/"
                f"w_180,q_auto:best/v1/people/{selected_player_id}/headshot/67/current"
            )
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
        ].rename(
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
        ).head(10),
        use_container_width=True,
        hide_index=True
    )

    if not player_whiffs.empty:
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
    else:
        st.info("No whiffs found for the selected filters.")

    st.markdown("### Notes")
    st.write(
        "Embarrassment Index is a custom score that increases for out-of-zone whiffs, "
        "more runners on base, consecutive whiffs within the same at-bat, "
        "and larger miss distance from the strike zone."
    )