"""Run exploratory analysis and open the HTML report in your browser.

Usage:
    python notebooks/exploratory_analysis.py
    python notebooks/exploratory_analysis.py --no-browser

Requires data/statcast_2025.parquet (see notebooks/statcast_pull.py).
Writes data/reports/eda_report.html
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exploratory_analysis import REPORT_FILE, run_eda


def main():
    parser = argparse.ArgumentParser(description="Exploratory analysis → HTML report")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Save report without opening a browser tab.",
    )
    args = parser.parse_args()

    print("Loading Statcast and building EDA report (this may take a few minutes)...")
    path = run_eda(open_browser=not args.no_browser)
    print(f"Saved report: {path}")
    print(f"Open manually: {REPORT_FILE.resolve().as_uri()}")


if __name__ == "__main__":
    main()
