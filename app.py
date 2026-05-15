import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="The Whiff List", layout="wide")


@st.cache_data
def load_pitch_data():
    df = pd.read_parquet("data/statcast_2025.parquet")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    numeric_cols = [
        "plate_x", "plate_z", "sz_top", "sz_bot",
        "balls", "strikes",
        "on_1b", "on_2b", "on_3b",
        "game_pk", "at_bat_number", "pitch_number",
        "batter", "pitcher"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_headshot_url(player_id):
    if pd.isna(player_id):
        return None
    try:
        return f"https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{int(player_id)}/headshot/67/current"
    except:
        return None


def calculate_miss_distance(row):
    if pd.isna(row["plate_x"]) or pd.isna(row["plate_z"]) or pd.isna(row["sz_top"]) or pd.isna(row["sz_bot"]):
        return np.nan

    left_edge = -0.7083
    right_edge = 0.7083
    px = row["plate_x"]
    pz = row["plate_z"]
    sz_top = row["sz_top"]
    sz_bot = row["sz_bot"]

    dx = 0
    dy = 0

    if px < left_edge:
        dx = left_edge - px
    elif px > right_edge:
        dx = px - right_edge

    if pz < sz_bot:
        dy = sz_bot - pz
    elif pz > sz_top:
        dy = pz - sz_top

    return np.sqrt(dx ** 2 + dy ** 2)


def calculate_embarrassment_index(df):
    df = df.copy()

    if "miss_distance" not in df.columns:
        df["miss_distance"] = df.apply(calculate_miss_distance, axis=1)

    df["in_zone"] = np.where(df["miss_distance"] == 0, 1, 0)

    df["runner_count"] = (
        df["on_1b"].notna().astype(int) +
        df["on_2b"].notna().astype(int) +
        df["on_3b"].notna().astype(int)
    )

    df = df.sort_values(
        by=["player_name", "game_date", "game_pk", "at_bat_number", "pitch_number"],
        kind="stable"
    )

    df["prev_pitch_oz_whiff"] = 0
    same_ab = (
        (df["player_name"] == df["player_name"].shift(1)) &
        (df["game_pk"] == df["game_pk"].shift(1)) &
        (df["at_bat_number"] == df["at_bat_number"].shift(1))
    )

    prev_was_oz_whiff = (
        same_ab &
        df["description"].shift(1).isin(["swinging_strike", "swinging_strike_blocked", "missed_bunt"]) &
        (df["in_zone"].shift(1) == 0)
    )

    df.loc[prev_was_oz_whiff, "prev_pitch_oz_whiff"] = 1

    base = np.where(
        df["in_zone"] == 0,
        20 + (df["miss_distance"].fillna(0) * 18),
        0
    )

    runner_penalty = df["runner_count"] * 6
    prev_penalty = df["prev_pitch_oz_whiff"] * 10

    raw_score = base + runner_penalty + prev_penalty

    df["embarrassment_index"] = np.where(
        df["description"].isin(["swinging_strike", "swinging_strike_blocked", "missed_bunt"]),
        np.clip(raw_score, 0, 100),
        np.nan
    )

    return df


pitch_data = load_pitch_data()
pitch_data["miss_distance"] = pitch_data.apply(calculate_miss_distance, axis=1)
pitch_data = calculate_embarrassment_index(pitch_data)

whiff_descriptions = ["swinging_strike", "swinging_strike_blocked", "missed_bunt"]
swing_descriptions = [
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
    "foul", "foul_tip", "foul_bunt",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score"
]
ab_events = [
    "single", "double", "triple", "home_run", "field_out",
    "grounded_into_double_play", "force_out", "double_play",
    "fielders_choice", "field_error", "strikeout", "strikeout_double_play"
]

pitch_data["is_whiff"] = pitch_data["description"].isin(whiff_descriptions).astype(int)
pitch_data["is_swing"] = pitch_data["description"].isin(swing_descriptions).astype(int)
pitch_data["ab_flag"] = pitch_data["events"].isin(ab_events).astype(int)

pitch_data["oz_embarrassment"] = np.where(
    (pitch_data["is_whiff"] == 1) & (pitch_data["in_zone"] == 0),
    pitch_data["embarrassment_index"],
    np.nan
)

st.sidebar.title("The Whiff List")

view = st.sidebar.radio(
    "View",
    ["Hall of Shame", "Player Breakdown"]
)

available_months = sorted(
    pitch_data["game_date"].dropna().dt.month.unique().tolist()
)

month_map = {
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October"
}

default_months = [month_map[m] for m in available_months if m in month_map]

selected_month_names = st.sidebar.multiselect(
    "Month",
    options=default_months,
    default=default_months
)

selected_month_nums = [
    month_num for month_num, month_name in month_map.items()
    if month_name in selected_month_names
]

if selected_month_nums:
    filtered_pitch_data = pitch_data[
        pitch_data["game_date"].dt.month.isin(selected_month_nums)
    ].copy()
else:
    filtered_pitch_data = pitch_data.iloc[0:0].copy()

min_ab = st.sidebar.slider("Minimum AB", min_value=0, max_value=100, value=10)
min_swings = st.sidebar.slider("Minimum Swings", min_value=0, max_value=200, value=20)

leaderboard_metric = st.sidebar.selectbox(
    "Leaderboard Metric",
    ["Whiff Rate (%)", "Avg O-Zone Embarrassment", "Total Whiffs"]
)

leaderboard = (
    filtered_pitch_data.groupby("player_name", as_index=False)
    .agg(
        AB=("ab_flag", "sum"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
        Avg_O_Zone_Embarrassment=("oz_embarrassment", "mean"),
        batter_id=("batter", "first")
    )
)

leaderboard["Whiff Rate (%)"] = np.where(
    leaderboard["Swings"] > 0,
    (leaderboard["Whiffs"] / leaderboard["Swings"]) * 100,
    np.nan
)

leaderboard["Avg O-Zone Embarrassment"] = leaderboard["Avg_O_Zone_Embarrassment"].round(1)
leaderboard["Whiff Rate (%)"] = leaderboard["Whiff Rate (%)"].round(1)

leaderboard = leaderboard[
    (leaderboard["AB"] >= min_ab) &
    (leaderboard["Swings"] >= min_swings)
].copy()

sort_map = {
    "Whiff Rate (%)": "Whiff Rate (%)",
    "Avg O-Zone Embarrassment": "Avg O-Zone Embarrassment",
    "Total Whiffs": "Whiffs"
}

sort_col = sort_map[leaderboard_metric]

leaderboard = leaderboard.sort_values(
    by=sort_col,
    ascending=False,
    na_position="last"
).reset_index(drop=True)

leaderboard["Rank"] = leaderboard.index + 1

display_df = leaderboard[
    ["Rank", "player_name", "AB", "Swings", "Whiffs", "Whiff Rate (%)", "Avg O-Zone Embarrassment"]
].rename(columns={"player_name": "Batter"})

player_options = sorted(filtered_pitch_data["player_name"].dropna().unique().tolist())

if "selected_player_name" not in st.session_state:
    st.session_state.selected_player_name = player_options[0] if player_options else None

if player_options:
    if st.session_state.selected_player_name not in player_options:
        st.session_state.selected_player_name = player_options[0]

    selected_player = st.sidebar.selectbox(
        "Select Player",
        player_options,
        key="selected_player_name"
    )
else:
    selected_player = None
    st.session_state.selected_player_name = None

if view == "Hall of Shame":
    st.title("Hall of Shame")
    st.caption("The ugliest whiff rates and out-of-zone misses in baseball.")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

elif view == "Player Breakdown":
    if selected_player is None:
        st.warning("No players available for the current filter selection.")
    else:
        player_whiffs = filtered_pitch_data[
            (filtered_pitch_data["player_name"] == selected_player) &
            (filtered_pitch_data["is_whiff"] == 1)
        ].copy()

        available_pitch_types = sorted(player_whiffs["pitch_name"].dropna().unique().tolist())

        selected_pitch_types = st.sidebar.multiselect(
            "Pitch Type",
            options=available_pitch_types,
            default=available_pitch_types
        )

        if selected_pitch_types:
            player_whiffs = player_whiffs[
                player_whiffs["pitch_name"].isin(selected_pitch_types)
            ].copy()
        else:
            player_whiffs = player_whiffs.iloc[0:0].copy()

        player_whiffs = player_whiffs.sort_values(
            by="embarrassment_index",
            ascending=False,
            na_position="last"
        )

        player_id_series = filtered_pitch_data.loc[
            filtered_pitch_data["player_name"] == selected_player, "batter"
        ].dropna()

        player_id = player_id_series.iloc[0] if not player_id_series.empty else None
        headshot_url = get_headshot_url(player_id)

        title_col, image_col = st.columns([5, 1.4])

        with title_col:
            st.title(f"Player Breakdown: {selected_player}")
            st.write("The ugliest whiffs for this hitter, ranked by Embarrassment Index.")

        with image_col:
            if headshot_url:
                st.image(headshot_url, use_container_width=True)

        total_whiffs = len(player_whiffs)
        in_zone_whiffs = int((player_whiffs["in_zone"] == 1).sum()) if "in_zone" in player_whiffs.columns else 0
        out_zone_whiffs = int((player_whiffs["in_zone"] == 0).sum()) if "in_zone" in player_whiffs.columns else 0
        avg_embarrassment = round(
            player_whiffs.loc[player_whiffs["in_zone"] == 0, "embarrassment_index"].mean(), 1
        ) if out_zone_whiffs > 0 else np.nan

        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Total Whiffs", total_whiffs)
        metric2.metric("In-Zone Whiffs", in_zone_whiffs)
        metric3.metric("Out-of-Zone Whiffs", out_zone_whiffs)
        metric4.metric("Avg Embarrassment", avg_embarrassment if not pd.isna(avg_embarrassment) else "-")

        display_player_whiffs = player_whiffs.copy()

        if "game_date" in display_player_whiffs.columns:
            display_player_whiffs["game_date"] = pd.to_datetime(
                display_player_whiffs["game_date"], errors="coerce"
            ).dt.strftime("%m-%d")

        display_player_whiffs["count"] = (
            display_player_whiffs["balls"].fillna(0).astype(int).astype(str)
            + "-"
            + display_player_whiffs["strikes"].fillna(0).astype(int).astype(str)
        )

        table_cols = [
            col for col in [
                "game_date", "pitch_name", "count",
                "miss_distance", "embarrassment_index"
            ] if col in display_player_whiffs.columns
        ]

        rename_map = {
            "game_date": "Date",
            "pitch_name": "Pitch Type",
            "count": "Count",
            "miss_distance": "Miss Distance",
            "embarrassment_index": "Embarrassment Index"
        }

        st.markdown("### Player View")
        st.dataframe(
            display_player_whiffs[table_cols].rename(columns=rename_map),
            use_container_width=True,
            hide_index=True
        )

        chart_df = player_whiffs.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot"]).copy()

        fig = go.Figure()

        if not chart_df.empty:
            fig.add_trace(go.Scatter(
                x=chart_df["plate_x"],
                y=chart_df["plate_z"],
                mode="markers",
                marker=dict(
                    size=12,
                    color=chart_df["embarrassment_index"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Embarrassment")
                ),
                text=chart_df["pitch_name"],
                customdata=np.stack([
                    pd.to_datetime(chart_df["game_date"], errors="coerce").dt.strftime("%m-%d"),
                    chart_df["pitch_name"].fillna("Unknown"),
                    chart_df["embarrassment_index"].round(1)
                ], axis=-1),
                hovertemplate=(
                    "Date: %{customdata[0]}<br>"
                    "Pitch: %{customdata[1]}<br>"
                    "Embarrassment: %{customdata[2]}<br>"
                    "X: %{x:.2f}<br>"
                    "Z: %{y:.2f}<extra></extra>"
                )
            ))

            sz_top = chart_df["sz_top"].median()
            sz_bot = chart_df["sz_bot"].median()

            fig.add_shape(
                type="rect",
                x0=-0.7083, x1=0.7083,
                y0=sz_bot, y1=sz_top,
                line=dict(color="white", width=2)
            )

        fig.update_layout(
            title=f"Whiff Locations: {selected_player}",
            xaxis_title="Horizontal Location (plate_x)",
            yaxis_title="Vertical Location (plate_z)",
            xaxis=dict(range=[-2.5, 2.5]),
            yaxis=dict(range=[0, 5]),
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

st.markdown("### Notes")
st.write(
    "Whiffs include swinging strikes and missed bunts. "
    "Swings include whiffs, fouls, foul bunts, and balls put into play. "
    "Avg Embarrassment uses out-of-zone whiffs only."
)