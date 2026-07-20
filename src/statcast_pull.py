"""Pull 2025 regular-season Statcast into parquet (raw columns only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.statcast_schema import (
    DEDUP_KEYS,
    FLOAT_COLS,
    INT_COLS,
    REQUIRED_STATCAST_COLS,
    SCHEMA_VERSION,
    SEASON_END,
    SEASON_START,
    STATCAST_KEEP_COLS,
    STRING_COLS,
    filter_regular_season,
    filter_season_dates,
    regular_season_month_ranges,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_parquet"
OUTPUT_FILE = DATA_DIR / "statcast_2025.parquet"
PLAYERS_FILE = DATA_DIR / "players.parquet"


def validate_required_columns(df: pd.DataFrame, *, context: str) -> None:
    """Raise if Savant response is missing pipeline-critical columns."""
    missing = sorted(REQUIRED_STATCAST_COLS - set(df.columns))
    if missing:
        raise ValueError(
            f"{context}: Statcast pull is missing required columns: {missing}. "
            "Update pybaseball / check Savant schema, then re-run the pull."
        )


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
    validate_required_columns(df, context="Before column selection")
    available = [col for col in STATCAST_KEEP_COLS if col in df.columns]
    extra_missing = sorted(set(STATCAST_KEEP_COLS) - set(available))
    if extra_missing:
        print(f"  Warning: optional keep columns missing from source: {extra_missing}")
    return df[available].copy()


def season_file_is_current(path: Path) -> bool:
    """True when merged season parquet matches current schema and column contract."""
    return not cache_is_stale(path, force=False)


def cache_is_stale(path: Path, *, force: bool = False) -> bool:
    if force:
        return True
    if not path.exists():
        return True
    meta_path = path.with_suffix(path.suffix + ".meta")
    if not meta_path.exists() or meta_path.read_text(encoding="utf-8").strip() != str(SCHEMA_VERSION):
        return True
    cached_cols = set(pq.read_schema(path).names)
    return cached_cols != set(STATCAST_KEEP_COLS)


def write_cache_meta(path: Path) -> None:
    path.with_suffix(path.suffix + ".meta").write_text(str(SCHEMA_VERSION), encoding="utf-8")


def clear_monthly_cache() -> int:
    """Delete cached monthly parquet + meta files. Returns files removed."""
    removed = 0
    if not RAW_DIR.exists():
        return removed
    for path in RAW_DIR.glob("statcast_*"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def pull_month(
    start_date: str,
    end_date: str,
    out_file: Path,
    *,
    force: bool = False,
) -> pd.DataFrame:
    if not cache_is_stale(out_file, force=force):
        print(f"Using cache: {out_file.name}", flush=True)
        month_df = select_columns(pd.read_parquet(out_file))
        return normalize_dtypes(filter_season_dates(month_df))

    from pybaseball import statcast

    print(f"Pulling {start_date} to {end_date} ...", flush=True)
    raw = statcast(start_date, end_date)
    if raw is None or raw.empty:
        print(f"  No rows returned for {start_date} to {end_date}")
        return pd.DataFrame(columns=STATCAST_KEEP_COLS)

    validate_required_columns(raw, context=f"Savant {start_date} to {end_date}")
    raw = filter_regular_season(raw)
    month_df = select_columns(raw)
    month_df = normalize_dtypes(month_df)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    month_df.to_parquet(out_file, index=False)
    write_cache_meta(out_file)
    print(f"Saved {len(month_df):,} regular-season rows to {out_file.name}")
    return month_df


def build_season_file(*, build_players: bool = True, force: bool = False) -> pd.DataFrame:
    from src.player_lookup import build_players_table

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if force:
        n_removed = clear_monthly_cache()
        if n_removed:
            print(f"Cleared {n_removed} cached monthly file(s) (--force).")

    parts: list[pd.DataFrame] = []
    for start_date, end_date in regular_season_month_ranges():
        out_file = RAW_DIR / f"statcast_{start_date}_to_{end_date}.parquet"
        parts.append(pull_month(start_date, end_date, out_file, force=force))

    non_empty = [p for p in parts if not p.empty]
    if not non_empty:
        raise RuntimeError("No Statcast rows pulled. Check dates and network access.")

    season_df = (
        pd.concat(non_empty, ignore_index=True)
        .pipe(select_columns)
        .drop_duplicates(subset=DEDUP_KEYS)
        .sort_values(["game_date", *DEDUP_KEYS])
        .reset_index(drop=True)
    )
    validate_required_columns(season_df, context="Merged season file")
    season_df.to_parquet(OUTPUT_FILE, index=False)
    write_cache_meta(OUTPUT_FILE)

    if build_players:
        players = build_players_table(season_df, PLAYERS_FILE)
        print(f"Saved player lookup: {PLAYERS_FILE} ({len(players):,} IDs)")

    return season_df


def pull_summary(season_df: pd.DataFrame) -> dict:
    null_pct = (season_df.isna().mean() * 100).round(1).to_dict()
    release_cols = [
        c
        for c in (
            "release_pos_x",
            "release_pos_y",
            "release_pos_z",
            "release_extension",
            "release_spin_rate",
            "p_throws",
            "stand",
        )
        if c in season_df.columns
    ]
    return {
        "rows": len(season_df),
        "columns": season_df.columns.tolist(),
        "schema_version": SCHEMA_VERSION,
        "date_min": str(season_df["game_date"].min()),
        "date_max": str(season_df["game_date"].max()),
        "unique_batters": int(season_df["batter"].nunique()),
        "unique_pitchers": int(season_df["pitcher"].nunique()),
        "release_null_pct": {col: null_pct[col] for col in release_cols},
        "null_pct": null_pct,
    }


def main(force: bool = False) -> pd.DataFrame:
    from pybaseball import cache

    cache.enable()
    season_df = build_season_file(build_players=True, force=force)
    summary = pull_summary(season_df)

    print(f"\nSaved season file: {OUTPUT_FILE}")
    print(f"Schema version: {summary['schema_version']}")
    print(f"Regular season window: {SEASON_START} to {SEASON_END}")
    print(f"Rows: {summary['rows']:,}")
    print(f"Unique batters: {summary['unique_batters']:,}")
    print(f"Unique pitchers: {summary['unique_pitchers']:,}")
    print(f"Date range: {summary['date_min']} to {summary['date_max']}")
    print(f"Columns ({len(summary['columns'])}): {summary['columns']}")
    print("Release / platoon null %:", summary["release_null_pct"])
    return season_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull 2025 Statcast regular-season pitches.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore monthly cache and re-download all months (schema v3+).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(force=args.force)
