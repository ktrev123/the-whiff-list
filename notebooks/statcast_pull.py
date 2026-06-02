"""Pull and cache 2025 MLB Statcast pitch data for The Whiff List."""

from pathlib import Path

import pandas as pd
from pybaseball import cache, statcast

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_parquet"
OUTPUT_FILE = DATA_DIR / "statcast_2025.parquet"

SEASON_START = "2025-03-23"
SEASON_END = "2025-09-27"

DATE_RANGES = [
    ("2025-03-23", "2025-03-31"),
    ("2025-04-01", "2025-04-30"),
    ("2025-05-01", "2025-05-31"),
    ("2025-06-01", "2025-06-30"),
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-27"),
]

KEEP_COLS = [
    # keys
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    # players
    "batter",
    "pitcher",
    "player_name",
    # outcomes
    "description",
    "events",
    # pitch identity
    "pitch_type",
    "pitch_name",
    # location
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    # context
    "balls",
    "strikes",
    "on_1b",
    "on_2b",
    "on_3b",
    "stand",
    "p_throws",
    # pitch characteristics
    "release_speed",
    "effective_speed",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "arm_angle",
]

INT_COLS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "balls",
    "strikes",
    "on_1b",
    "on_2b",
    "on_3b",
]

FLOAT_COLS = [
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "release_speed",
    "effective_speed",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "arm_angle",
]

STRING_COLS = [
    "player_name",
    "description",
    "events",
    "pitch_type",
    "pitch_name",
    "stand",
    "p_throws",
]


def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"])

    for col in INT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int32")

    for col in FLOAT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    for col in STRING_COLS:
        if col in out.columns:
            out[col] = out[col].astype("string")

    return out


def filter_season(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["game_date"])
    start = pd.Timestamp(SEASON_START)
    end = pd.Timestamp(SEASON_END)
    return df[(dates >= start) & (dates <= end)].copy()


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in KEEP_COLS if col in df.columns]
    missing = sorted(set(KEEP_COLS) - set(available))
    if missing:
        print(f"  Warning: missing columns in source pull: {missing}")
    return df[available].copy()


def cache_is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    cached_cols = set(pd.read_parquet(path).columns)
    return not set(KEEP_COLS).issubset(cached_cols)


def pull_month(start_date: str, end_date: str, out_file: Path) -> pd.DataFrame:
    if not cache_is_stale(out_file):
        print(f"Using cache: {out_file.name}")
        month_df = pd.read_parquet(out_file)
        return normalize_dtypes(filter_season(month_df))

    print(f"Pulling {start_date} to {end_date} ...")
    raw = statcast(start_date, end_date)
    month_df = select_columns(raw)
    month_df = normalize_dtypes(filter_season(month_df))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    month_df.to_parquet(out_file, index=False)
    print(f"Saved {len(month_df):,} rows to {out_file.name}")
    return month_df


def build_season_file() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parts = []
    for start_date, end_date in DATE_RANGES:
        out_file = RAW_DIR / f"statcast_{start_date}_to_{end_date}.parquet"
        parts.append(pull_month(start_date, end_date, out_file))

    season_df = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
        .sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
        .reset_index(drop=True)
    )
    season_df.to_parquet(OUTPUT_FILE, index=False)
    return season_df


def main():
    cache.enable()
    season_df = build_season_file()

    print(f"\nSaved season file: {OUTPUT_FILE}")
    print(f"Rows: {len(season_df):,}")
    print(f"Date range: {season_df['game_date'].min()} to {season_df['game_date'].max()}")
    print(f"Columns ({len(season_df.columns)}): {season_df.columns.tolist()}")


if __name__ == "__main__":
    main()
