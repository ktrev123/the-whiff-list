"""Build batter Statcast-zone xwOBAcon cache for the Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.batter_roster import qualified_batter_mlb_ids
from src.batter_zone_profiles import build_batter_zone_cache


if __name__ == "__main__":
    batter_ids = qualified_batter_mlb_ids()
    lookup = build_batter_zone_cache(batter_ids=batter_ids)
    print(f"Saved zone profiles for {lookup['batter'].nunique()} batters")
    print(lookup.groupby("batter")["zone"].count().head(10).to_string())
