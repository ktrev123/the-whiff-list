import os
import pandas as pd
from pybaseball import statcast, cache


cache.enable()


os.makedirs("data/raw_parquet", exist_ok=True)


date_ranges = [
    ("2025-03-01", "2025-03-31"),
    ("2025-04-01", "2025-04-30"),
    ("2025-05-01", "2025-05-31"),
    ("2025-06-01", "2025-06-30"),
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-30"),
    ("2025-10-01", "2025-10-31"),
]


keep_cols = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "player_name",
    "description",
    "events",
    "pitch_name",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "balls",
    "strikes",
    "on_1b",
    "on_2b",
    "on_3b",
    "stand",
    "p_throws",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "home_team",
    "away_team",
]


all_parts = []


for start_date, end_date in date_ranges:
    out_file = f"data/raw_parquet/statcast_{start_date}_to_{end_date}.parquet"

    if os.path.exists(out_file):
        print(f"Skipping existing file: {out_file}")
        month_df = pd.read_parquet(out_file)
    else:
        print(f"Pulling {start_date} to {end_date} ...")
        df = statcast(start_date, end_date)

        available_cols = [c for c in keep_cols if c in df.columns]
        month_df = df[available_cols].copy()

        if "game_date" in month_df.columns:
            month_df["game_date"] = pd.to_datetime(month_df["game_date"])
            # HARD CUTOFF: drop anything before 2025-03-25
            month_df = month_df[month_df["game_date"] >= pd.Timestamp("2025-03-25")].copy()

        int_cols = [
            "game_pk", "at_bat_number", "pitch_number", "batter", "pitcher",
            "balls", "strikes", "inning", "outs_when_up"
        ]
        for col in int_cols:
            if col in month_df.columns:
                month_df[col] = pd.to_numeric(month_df[col], errors="coerce").astype("Int32")

        float_cols = ["plate_x", "plate_z", "sz_top", "sz_bot"]
        for col in float_cols:
            if col in month_df.columns:
                month_df[col] = pd.to_numeric(month_df[col], errors="coerce").astype("float32")

        base_cols = ["on_1b", "on_2b", "on_3b"]
        for col in base_cols:
            if col in month_df.columns:
                month_df[col] = pd.to_numeric(month_df[col], errors="coerce").astype("Int32")

        string_cols = [
            "player_name", "description", "events", "pitch_name",
            "stand", "p_throws", "inning_topbot", "home_team", "away_team"
        ]
        for col in string_cols:
            if col in month_df.columns:
                month_df[col] = month_df[col].astype("string")

        month_df.to_parquet(out_file, index=False)
        print(f"Saved {len(month_df):,} rows to {out_file}")

    all_parts.append(month_df)


season_df = pd.concat(all_parts, ignore_index=True).drop_duplicates()
season_df = season_df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])

season_file = "data/statcast_2025.parquet"
season_df.to_parquet(season_file, index=False)

print(f"Saved full season parquet: {season_file}")
print(f"Final rows: {len(season_df):,}")
print(f"Columns kept: {season_df.columns.tolist()}")