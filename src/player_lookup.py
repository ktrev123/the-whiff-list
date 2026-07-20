"""Build a helper table mapping MLBAM player IDs to names (optional, does not filter pitches)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLAYERS_FILE = ROOT / "data" / "players.parquet"

BATCH_SIZE = 500
MIN_QUALIFIED_AB = 502

AB_EVENTS = frozenset(
    {
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
        "strikeout_double_play",
    }
)


def _format_player_name(row: pd.Series) -> str:
    if "name_first" in row.index and "name_last" in row.index:
        return f"{str(row['name_first']).title()} {str(row['name_last']).title()}".strip()
    if "name_last" in row.index:
        return str(row["name_last"]).title()
    return "Unknown"


def lookup_player_ids(player_ids: list[int]) -> pd.DataFrame:
    from pybaseball import playerid_reverse_lookup

    if not player_ids:
        return pd.DataFrame(columns=["mlbam_id", "player_name"])

    frames: list[pd.DataFrame] = []
    n_batches = (len(player_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(player_ids), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = player_ids[i : i + BATCH_SIZE]
        print(f"  Player lookup batch {batch_num}/{n_batches} ({len(batch)} IDs)...", flush=True)
        looked_up = playerid_reverse_lookup(batch, key_type="mlbam")
        if looked_up is None or looked_up.empty:
            continue
        looked_up = looked_up.copy()
        looked_up["mlbam_id"] = looked_up["key_mlbam"].astype(int)
        looked_up["player_name"] = looked_up.apply(_format_player_name, axis=1)
        frames.append(looked_up[["mlbam_id", "player_name"]])

    if not frames:
        return pd.DataFrame(columns=["mlbam_id", "player_name"])

    out = pd.concat(frames, ignore_index=True).drop_duplicates("mlbam_id")
    return out.sort_values("player_name").reset_index(drop=True)


def batter_ab_counts(frame: pd.DataFrame) -> pd.Series:
    """Plate-appearance terminal events counted as at-bats, grouped by batter."""
    if "events" not in frame.columns:
        raise ValueError("events column required to compute AB counts")
    ab_mask = frame["events"].isin(AB_EVENTS)
    return frame.loc[ab_mask].groupby("batter").size().rename("ab")


def qualified_batter_ids(frame: pd.DataFrame, min_ab: int = MIN_QUALIFIED_AB) -> set[int]:
    """Batters meeting the MLB qualified-hitter AB threshold (app / leaderboard use)."""
    counts = batter_ab_counts(frame)
    return set(counts[counts >= min_ab].index.astype(int))


def filter_qualified_batters(
    frame: pd.DataFrame,
    *,
    min_ab: int = MIN_QUALIFIED_AB,
    season_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Subset a pitch frame to qualified hitters only (for app display, not modeling)."""
    reference = season_frame if season_frame is not None else frame
    qualified = qualified_batter_ids(reference, min_ab=min_ab)
    return frame[frame["batter"].isin(qualified)].copy()


def build_players_table(season_df: pd.DataFrame, out_path: Path = PLAYERS_FILE) -> pd.DataFrame:
    batter_ids = season_df["batter"].dropna().astype(int).unique().tolist()
    pitcher_ids = season_df["pitcher"].dropna().astype(int).unique().tolist()
    all_ids = sorted(set(batter_ids) | set(pitcher_ids))
    print(f"Looking up {len(all_ids):,} unique batter/pitcher IDs...", flush=True)

    players = lookup_player_ids(all_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    players.to_parquet(out_path, index=False)
    return players
