import pandas as pd
from pybaseball import playerid_reverse_lookup

input_file = "data/statcast_2025.parquet"
output_file = "data/whiff_leaderboard_2025.csv"

df = pd.read_parquet(input_file)

swing_descriptions = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
    "foul_bunt",
    "missed_bunt"
}

whiff_descriptions = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt"
}

ab_events = {
    "single",
    "double",
    "triple",
    "home_run",
    "field_out",
    "grounded_into_double_play",
    "force_out",
    "double_play",
    "fielders_choice",
    "field_error",
    "strikeout",
    "strikeout_double_play"
}

df["is_swing"] = df["description"].isin(swing_descriptions)
df["is_whiff"] = df["description"].isin(whiff_descriptions)
df["is_ab"] = df["events"].isin(ab_events)

leaderboard = (
    df.groupby("batter", dropna=False)
    .agg(
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        ab=("is_ab", "sum")
    )
    .reset_index()
)

leaderboard = leaderboard[leaderboard["swings"] > 0].copy()
leaderboard["whiff_rate"] = leaderboard["whiffs"] / leaderboard["swings"]

player_ids = leaderboard["batter"].dropna().astype(int).tolist()
name_lookup = playerid_reverse_lookup(player_ids, key_type="mlbam")

print("Reverse lookup columns:", name_lookup.columns.tolist())

name_lookup = name_lookup.rename(
    columns={
        "key_mlbam": "batter"
    }
)

if "name_first" in name_lookup.columns and "name_last" in name_lookup.columns:
    name_lookup["player_name"] = (
        name_lookup["name_first"].astype(str).str.title() + " " +
        name_lookup["name_last"].astype(str).str.title()
    )
elif "name_first" in name_lookup.columns and "name_last_x" in name_lookup.columns:
    name_lookup["player_name"] = (
        name_lookup["name_first"].astype(str).str.title() + " " +
        name_lookup["name_last_x"].astype(str).str.title()
    )
elif "name_first" in name_lookup.columns and "last_name" in name_lookup.columns:
    name_lookup["player_name"] = (
        name_lookup["name_first"].astype(str).str.title() + " " +
        name_lookup["last_name"].astype(str).str.title()
    )
elif "name_last" in name_lookup.columns and "name_first" not in name_lookup.columns:
    name_lookup["player_name"] = name_lookup["name_last"].astype(str).str.title()
else:
    raise ValueError(f"Unexpected columns returned from playerid_reverse_lookup: {name_lookup.columns.tolist()}")

leaderboard = leaderboard.merge(
    name_lookup[["batter", "player_name"]],
    on="batter",
    how="left"
)

leaderboard["player_name"] = leaderboard["player_name"].fillna("Unknown")

leaderboard = leaderboard.sort_values(
    ["whiff_rate", "whiffs"],
    ascending=[False, False]
).reset_index(drop=True)

leaderboard["rank"] = leaderboard.index + 1

leaderboard = leaderboard[
    ["rank", "batter", "player_name", "ab", "swings", "whiffs", "whiff_rate"]
]

leaderboard.to_csv(output_file, index=False)

print(leaderboard.head(20))
print(f"Saved leaderboard to {output_file}")