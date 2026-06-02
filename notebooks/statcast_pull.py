"""Pull and cache 2025 MLB regular-season Statcast data for The Whiff List."""

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pybaseball import cache, playerid_reverse_lookup, statcast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.statcast_schema import (
    FLOAT_COLS,
    INT_COLS,
    SEASON_END,
    SEASON_START,
    STATCAST_KEEP_COLS,
    STRING_COLS,
    filter_regular_season,
    filter_season_dates,
    regular_season_month_ranges,
)

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_parquet"
OUTPUT_FILE = DATA_DIR / "statcast_2025.parquet"
CACHE_SCHEMA_VERSION = 3  # bump when columns or season filters change


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


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in STATCAST_KEEP_COLS if col in df.columns]
    missing = sorted(set(STATCAST_KEEP_COLS) - set(available))
    if missing:
        print(f"  Warning: missing columns in source pull: {missing}")
    return df[available].copy()


def cache_is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    meta_path = path.with_suffix(path.suffix + ".meta")
    if not meta_path.exists() or meta_path.read_text(encoding="utf-8").strip() != str(CACHE_SCHEMA_VERSION):
        return True
    cached_cols = set(pq.read_schema(path).names)
    return cached_cols != set(STATCAST_KEEP_COLS)


def write_cache_meta(path: Path) -> None:
    path.with_suffix(path.suffix + ".meta").write_text(str(CACHE_SCHEMA_VERSION), encoding="utf-8")


def filter_known_players(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing IDs or batters that do not resolve in Chadwick lookup."""
    out = df.dropna(subset=["batter", "pitcher"]).copy()
    out = out[out["player_name"].notna() & (out["player_name"].str.strip() != "")]

    batter_ids = out["batter"].dropna().astype(int).unique().tolist()
    if not batter_ids:
        return out

    lookup = playerid_reverse_lookup(batter_ids, key_type="mlbam")
    valid_batters = set(lookup["key_mlbam"].astype(int))
    before = len(out)
    out = out[out["batter"].astype(int).isin(valid_batters)].copy()
    dropped = before - len(out)
    if dropped:
        print(f"  Dropped {dropped:,} rows with unresolvable batter IDs")
    return out


def pull_month(start_date: str, end_date: str, out_file: Path) -> pd.DataFrame:
    if not cache_is_stale(out_file):
        print(f"Using cache: {out_file.name}")
        month_df = select_columns(pd.read_parquet(out_file))
        return normalize_dtypes(filter_season_dates(month_df))

    print(f"Pulling {start_date} to {end_date} ...")
    raw = statcast(start_date, end_date)
    raw = filter_regular_season(raw)
    month_df = select_columns(raw)
    month_df = normalize_dtypes(month_df)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    month_df.to_parquet(out_file, index=False)
    write_cache_meta(out_file)
    print(f"Saved {len(month_df):,} regular-season rows to {out_file.name}")
    return month_df


def build_season_file() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parts = []
    for start_date, end_date in regular_season_month_ranges():
        out_file = RAW_DIR / f"statcast_{start_date}_to_{end_date}.parquet"
        parts.append(pull_month(start_date, end_date, out_file))

    season_df = (
        pd.concat(parts, ignore_index=True)
        .pipe(select_columns)
        .drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
        .pipe(filter_known_players)
        .sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
        .reset_index(drop=True)
    )
    season_df.to_parquet(OUTPUT_FILE, index=False)
    return season_df


def main():
    cache.enable()
    season_df = build_season_file()

    print(f"\nSaved season file: {OUTPUT_FILE}")
    print(f"Regular season window: {SEASON_START} to {SEASON_END}")
    print(f"Rows: {len(season_df):,}")
    print(f"Unique batters: {season_df['batter'].nunique():,}")
    print(f"Date range in file: {season_df['game_date'].min()} to {season_df['game_date'].max()}")
    print(f"Columns ({len(season_df.columns)}): {season_df.columns.tolist()}")


if __name__ == "__main__":
    main()
