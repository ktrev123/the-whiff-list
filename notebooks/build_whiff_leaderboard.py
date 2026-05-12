import pandas as pd
from pybaseball import playerid_reverse_lookup

input_file = "data/statcast_test_2025_04_01_to_04_07.csv"
output_file = "data/whiff_leaderboard_test.csv"

df = pd.read_csv(input_file)

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

df["is_swing"] = df["description"].isin(swing_descriptions)
df["is_whiff"] = df["description"].isin(whiff_descriptions)

leaderboard = (
    df.groupby("batter", dropna=False)
      .agg(
          swings=("is_swing", "sum"),
          whiffs=("is_whiff", "sum")
      )
      .reset_index()
)

leaderboard = leaderboard[leaderboard["swings"] > 0].copy()
leaderboard["whiff_rate"] = leaderboard["whiffs"] / leaderboard["swings"]

player_ids = leaderboard["batter"].dropna().astype(int).tolist()
name_lookup = playerid_reverse_lookup(player_ids, key_type="mlbam")

name_lookup = name_lookup.rename(
    columns={
        "key_mlbam": "batter",
        "name_first": "first_name",
        "name_last": "last_name"
    }
)

name_lookup["player_name"] = (
    name_lookup["first_name"].str.title() + " " + name_lookup["last_name"].str.title()
)

leaderboard = leaderboard.merge(
    name_lookup[["batter", "player_name"]],
    on="batter",
    how="left"
)

leaderboard = leaderboard.sort_values(
    ["whiff_rate", "whiffs"],
    ascending=[False, False]
).reset_index(drop=True)

leaderboard["rank"] = leaderboard.index + 1

leaderboard = leaderboard[
    ["rank", "batter", "player_name", "swings", "whiffs", "whiff_rate"]
]

leaderboard.to_csv(output_file, index=False)

print(leaderboard.head(20))
print(f"Saved leaderboard to {output_file}")