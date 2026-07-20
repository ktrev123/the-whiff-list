"""Build qualified batter roster cache for the Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.batter_roster import build_batter_roster_table, save_batter_roster


if __name__ == "__main__":
    roster = build_batter_roster_table()
    out = save_batter_roster(roster)
    print(f"Saved {len(roster):,} qualified batters to {out}")
