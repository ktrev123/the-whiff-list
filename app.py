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


st.title("The Whiff List 💨")
st.subheader("2025 MLB Swing-and-Miss Offenders")
st.write(
    """
    An interactive Statcast dashboard built with Streamlit
    to uncover the season's biggest swing-and-miss offenders.
    """
)

with st.expander("How the Embarrassment Index works", expanded=False):
    st.markdown(
        """
        The **Embarrassment Index** is a custom 0-100 pitch-level score where higher values represent more embarrassing swing-and-miss events.

        The score combines four ideas:
        - How far the whiff finished outside the strike zone
        - Whether the whiff was out of the zone at all
        - Whether the previous pitch in the same at-bat was also a swing-and-miss out of the zone
        - How many runners were on base

        Every pitch is scored on a bounded scale so a pitch that is only barely off the plate is not punished too harshly.
        """
    )

    st.markdown("**Formula**")
    st.latex(r"""
    EI = 100 \cdot \left(0.45D + 0.20Z + 0.15P + 0.20R\right)
    """)

    st.markdown("**Components**")

    st.latex(r"""
    D = \min\left(\frac{\text{miss distance in inches}}{18}, 1\right)
    """)

    st.latex(r"""
    Z =
    \begin{cases}
    1, & \text{if the whiff was out of zone} \\
    0, & \text{if the whiff was in zone}
    \end{cases}
    """)

    st.latex(r"""
    P =
    \begin{cases}
    1, & \text{if the previous pitch was also a swing-and-miss out of zone} \\
    0, & \text{otherwise}
    \end{cases}
    """)

    st.latex(r"""
    R = \frac{\text{runners on base}}{3}
    """)

    st.markdown(
        """
        **Strike-zone logic**
        - The pitch is in zone when `plate_x` is between -0.708 and 0.708 feet, and `plate_z` is between `sz_bot` and `sz_top`.
        - If the pitch is inside those bounds, miss distance is 0.
        - If the pitch is outside those bounds, the app measures the straight-line distance from the nearest edge of the strike zone.
        - That distance is converted from feet to inches and then capped at 18 inches for scoring.

        **Example**
        - Out-of-zone whiff
        - 2 runners on base
        - Previous pitch was also a swing-and-miss out of zone
        - Miss distance = 14.4 inches
        """
    )

    st.latex(r"""
    D = 14.4 / 18 = 0.8,\quad Z = 1,\quad P = 1,\quad R = 2/3
    """)

    st.latex(r"""
    EI = 100 \cdot \left(0.45(0.8) + 0.20(1) + 0.15(1) + 0.20(2/3)\right)
    """)

    st.latex(r"""
    EI = 100 \cdot (0.36 + 0.20 + 0.15 + 0.1333) = 84.3
    """)

    st.markdown(
        """
        **How to read it**
        - Lower scores usually come from in-zone misses or low-leverage situations.
        - Mid-range scores usually reflect either a real chase or extra pressure.
        - The highest scores usually combine a clear chase, runners on base, a prior out-of-zone whiff, and a large miss distance.
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

if "balls" not in pitch_data.columns:
    pitch_data["balls"] = np.nan

if "strikes" not in pitch_data.columns:
    pitch_data["strikes"] = np.nan

pitch_data["count"] = np.where(
    pitch_data["balls"].notna() & pitch_data["strikes"].notna(),
    pitch_data["balls"].astype("Int64").astype(str) + "-" + pitch_data["strikes"].astype("Int64").astype(str),
    "Unknown"
)

pitch_data["two_strikes"] = np.where(
    pitch_data["strikes"].fillna(-1) == 2,
    "Yes",
    "No"
)

if all(col in pitch_data.columns for col in ["game_pk", "at_bat_number", "pitch_number"]):
    pitch_data = pitch_data.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    ).copy()

    pitch_data["prev_description"] = (
        pitch_data.groupby(["game_pk", "at_bat_number"])["description"].shift(1)
    )
    pitch_data["prev_plate_x"] = (
        pitch_data.groupby(["game_pk", "at_bat_number"])["plate_x"].shift(1)
    )
    pitch_data["prev_plate_z"] = (
        pitch_data.groupby(["game_pk", "at_bat_number"])["plate_z"].shift(1)
    )
    pitch_data["prev_sz_top"] = (
        pitch_data.groupby(["game_pk", "at_bat_number"])["sz_top"].shift(1)
    )
    pitch_data["prev_sz_bot"] = (
        pitch_data.groupby(["game_pk", "at_bat_number"])["sz_bot"].shift(1)
    )

    prev_left_edge = -0.708
    prev_right_edge = 0.708

    prev_x_out = np.where(
        pitch_data["prev_plate_x"] < prev_left_edge,
        prev_left_edge - pitch_data["prev_plate_x"],
        np.where(
            pitch_data["prev_plate_x"] > prev_right_edge,
            pitch_data["prev_plate_x"] - prev_right_edge,
            0
        )
    )

    prev_z_out = np.where(
        pitch_data["prev_plate_z"] < pitch_data["prev_sz_bot"],
        pitch_data["prev_sz_bot"] - pitch_data["prev_plate_z"],
        np.where(
            pitch_data["prev_plate_z"] > pitch_data["prev_sz_top"],
            pitch_data["prev_plate_z"] - pitch_data["prev_sz_top"],
            0
        )
    )

    pitch_data["prev_miss_distance"] = np.sqrt((prev_x_out ** 2) + (prev_z_out ** 2))
    pitch_data["prev_zone_split"] = np.where(
        pitch_data["prev_miss_distance"] == 0,
        "In Zone",
        "Out of Zone"
    )
    pitch_data["prev_whiff_ozone"] = np.where(
        pitch_data["prev_description"].isin(whiff_descriptions) &
        (pitch_data["prev_zone_split"] == "Out of Zone"),
        1,
        0
    )
else:
    pitch_data["prev_whiff_ozone"] = 0

pitch_data["distance_score"] = np.minimum(pitch_data["miss_distance_inches"] / 18, 1.0)
pitch_data["zone_score"] = np.where(pitch_data["zone_split"] == "Out of Zone", 1.0, 0.0)
pitch_data["prev_whiff_score"] = pitch_data["prev_whiff_ozone"].astype(float)
pitch_data["runners_score"] = pitch_data["runners_on"] / 3.0

pitch_data["embarrassment_index"] = (
    100 * (
        0.45 * pitch_data["distance_score"] +
        0.20 * pitch_data["zone_score"] +
        0.15 * pitch_data["prev_whiff_score"] +
        0.20 * pitch_data["runners_score"]
    )
).round(1)

pitch_data["embarrassment_index_ozone_only"] = np.where(
    pitch_data["zone_split"] == "Out of Zone",
    pitch_data["embarrassment_index"],
    np.nan
)

avg_embarrassment = (
    pitch_data.groupby("batter", as_index=False)["embarrassment_index_ozone_only"]
    .mean()
    .rename(columns={"embarrassment_index_ozone_only": "avg_embarrassment_index"})
)

df = df.merge(avg_embarrassment, on="batter", how="left")
df["avg_embarrassment_index"] = df["avg_embarrassment_index"].round(1)

player_options = sorted(pitch_data["batter_name"].dropna().unique())

default_player_name = "Shohei Ohtani"
default_selected_player = default_player_name if default_player_name in player_options else player_options[0]

if "selected_player_name" not in st.session_state:
    st.session_state.selected_player_name = default_selected_player

if st.session_state.selected_player_name not in player_options:
    st.session_state.selected_player_name = default_selected_player

if "player_breakdown_selectbox" not in st.session_state:
    st.session_state.player_breakdown_selectbox = st.session_state.selected_player_name

if st.session_state.player_breakdown_selectbox not in player_options:
    st.session_state.player_breakdown_selectbox = st.session_state.selected_player_name

if "leaderboard_target_player" in st.session_state:
    target_player = st.session_state.leaderboard_target_player
    if target_player in player_options:
        st.session_state.player_breakdown_selectbox = target_player
        st.session_state.selected_player_name = target_player
    del st.session_state["leaderboard_target_player"]

if "selected_pitch_types_by_player" not in st.session_state:
    st.session_state.selected_pitch_types_by_player = {}


def sync_selected_player():
    st.session_state.selected_player_name = st.session_state.player_breakdown_selectbox


selected_player = st.sidebar.selectbox(
    "Player Breakdown",
    player_options,
    key="player_breakdown_selectbox",
    on_change=sync_selected_player
)

selected_player = st.session_state.selected_player_name

df_filtered = df[
    df["swings"] >= min_swings
].copy()

df_filtered["whiff_rate_pct"] = (df_filtered["whiff_rate"] * 100).round(1)

df_filtered = df_filtered.sort_values(
    ["avg_embarrassment_index", "whiff_rate", "whiffs"],
    ascending=[False, False, False]
).reset_index(drop=True)

df_filtered["rank_display"] = df_filtered.index + 1

leaderboard_display = df_filtered.rename(
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
col1.metric("Players shown", len(leaderboard_display))
col2.metric("Season", season)
col3.metric("Minimum swings", min_swings)

st.markdown("### Leaderboard")

selection_event = st.dataframe(
    leaderboard_display[
        ["Rank", "Batter", "AB", "Swings", "Whiffs", "Whiff Rate (%)", "Avg Embarrassment Index"]
    ],
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="leaderboard_table"
)

selected_rows = selection_event.selection.get("rows", [])
if selected_rows:
    selected_row_idx = selected_rows[0]
    selected_batter_name = leaderboard_display.iloc[selected_row_idx]["Batter"]

    if selected_batter_name in player_options and selected_batter_name != st.session_state.selected_player_name:
        st.session_state.leaderboard_target_player = selected_batter_name
        st.rerun()

selected_player = st.session_state.selected_player_name

st.markdown("### Most Embarrassing Swings in the Dataset")

most_embarrassing_swings = (
    pitch_data[pitch_data["zone_split"] == "Out of Zone"]
    .sort_values(
        ["embarrassment_index", "miss_distance_inches", "runners_on"],
        ascending=[False, False, False]
    )
    .reset_index(drop=True)
    .copy()
)

most_embarrassing_swings["Rank"] = most_embarrassing_swings.index + 1

st.dataframe(
    most_embarrassing_swings[
        [
            "Rank",
            "game_date",
            "batter_name",
            "player_name",
            "pitch_name",
            "count",
            "runners_on",
            "miss_distance_inches",
            "embarrassment_index"
        ]
    ]
    .rename(
        columns={
            "game_date": "Date",
            "batter_name": "Batter",
            "player_name": "Pitcher",
            "pitch_name": "Pitch Type",
            "count": "Count",
            "runners_on": "Runners On",
            "miss_distance_inches": "Miss Distance (in)",
            "embarrassment_index": "Embarrassment Index"
        }
    )
    .head(25),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

player_whiffs = pitch_data[pitch_data["batter_name"] == selected_player].copy()

pitch_type_options = sorted(player_whiffs["pitch_name"].dropna().unique())

saved_pitch_types = st.session_state.selected_pitch_types_by_player.get(selected_player)
valid_saved_pitch_types = (
    [p for p in saved_pitch_types if p in pitch_type_options]
    if saved_pitch_types is not None
    else pitch_type_options.copy()
)

if not valid_saved_pitch_types and pitch_type_options:
    valid_saved_pitch_types = pitch_type_options.copy()

selected_pitch_types = st.sidebar.multiselect(
    "Pitch Type",
    options=pitch_type_options,
    default=valid_saved_pitch_types,
    key=f"pitch_type_multiselect_{selected_player}"
)

st.session_state.selected_pitch_types_by_player[selected_player] = selected_pitch_types

if selected_pitch_types:
    player_whiffs = player_whiffs[player_whiffs["pitch_name"].isin(selected_pitch_types)].copy()
else:
    player_whiffs = player_whiffs.iloc[0:0].copy()

player_whiffs = player_whiffs.sort_values(
    ["embarrassment_index", "miss_distance"],
    ascending=[False, False]
).copy()

in_zone_whiffs = player_whiffs[player_whiffs["zone_split"] == "In Zone"]
out_zone_whiffs = player_whiffs[player_whiffs["zone_split"] == "Out of Zone"]

selected_player_id = (
    int(player_whiffs["batter"].dropna().iloc[0])
    if not player_whiffs.empty else (
        int(pitch_data.loc[pitch_data["batter_name"] == selected_player, "batter"].dropna().iloc[0])
        if not pitch_data.loc[pitch_data["batter_name"] == selected_player, "batter"].dropna().empty
        else None
    )
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

st.markdown("### Pitch Type Summary")
if not player_whiffs.empty:
    pitch_summary = (
        player_whiffs.groupby("pitch_name", as_index=False)
        .agg(
            Whiffs=("pitch_name", "size"),
            Avg_Embarrassment=("embarrassment_index", "mean"),
            Avg_Miss_Distance_In=("miss_distance_inches", "mean"),
            In_Zone_Whiffs=("zone_split", lambda x: (x == "In Zone").sum()),
            Out_of_Zone_Whiffs=("zone_split", lambda x: (x == "Out of Zone").sum())
        )
        .rename(columns={"pitch_name": "Pitch Type"})
    )

    pitch_summary["Avg Embarrassment"] = pitch_summary["Avg_Embarrassment"].round(1)
    pitch_summary["Avg Miss Distance (in)"] = pitch_summary["Avg_Miss_Distance_In"].round(1)
    pitch_summary = pitch_summary.drop(columns=["Avg_Embarrassment", "Avg_Miss_Distance_In"])
    pitch_summary = pitch_summary.sort_values(
        ["Whiffs", "Avg Embarrassment"],
        ascending=[False, False]
    ).reset_index(drop=True)

    st.dataframe(
        pitch_summary[
            [
                "Pitch Type",
                "Whiffs",
                "Avg Embarrassment",
                "Avg Miss Distance (in)",
                "In_Zone_Whiffs",
                "Out_of_Zone_Whiffs"
            ]
        ].rename(
            columns={
                "In_Zone_Whiffs": "In-Zone Whiffs",
                "Out_of_Zone_Whiffs": "Out-of-Zone Whiffs"
            }
        ),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No pitch summary available for the selected pitch types.")

st.markdown(f"### {selected_player}'s Top 10 Whiffs")
st.dataframe(
    player_whiffs[
        [
            "game_date",
            "player_name",
            "pitch_name",
            "count",
            "two_strikes",
            "zone_split",
            "runners_on",
            "prev_whiff_ozone",
            "miss_distance_inches",
            "embarrassment_index"
        ]
    ]
    .rename(
        columns={
            "game_date": "Date",
            "player_name": "Pitcher",
            "pitch_name": "Pitch Type",
            "count": "Count",
            "two_strikes": "Two Strikes",
            "zone_split": "Zone Split",
            "runners_on": "Runners On",
            "prev_whiff_ozone": "Prev Pitch O-Zone Whiff",
            "miss_distance_inches": "Miss Distance (in)",
            "embarrassment_index": "Embarrassment Index"
        }
    )
    .head(10),
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
            colorscale="Cividis",
            cmin=0,
            cmax=100,
            showscale=True,
            colorbar=dict(title="Embarrassment Index")
        ),
        customdata=player_whiffs[
            [
                "player_name",
                "pitch_name",
                "count",
                "miss_distance_inches",
                "runners_on",
                "embarrassment_index"
            ]
        ],
        hovertemplate=(
            "<b>%{customdata[0]}'s %{customdata[1]}</b><br>"
            "Count: %{customdata[2]}<br>"
            "Miss Distance: %{customdata[3]} in<br>"
            "Runners On: %{customdata[4]}<br>"
            "Embarrassment Index: %{customdata[5]}"
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
        title=f"{selected_player}'s Whiff Locations",
        xaxis_title="Horizontal Location (plate_x)",
        yaxis_title="Vertical Location (plate_z)",
        xaxis=dict(range=[-2.5, 2.5]),
        yaxis=dict(range=[0, 5]),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No whiffs found for the selected pitch types.")