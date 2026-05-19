"""
Download KBO game schedules using kbodata.

Usage:
    python scripts/download_game_logs.py --year 2024
    python scripts/download_game_logs.py --start-year 2019 --end-year 2024
    python scripts/download_game_logs.py --year 2024 --output-dir data/kbo_games

Requires:
    pip install kbodata
    ChromeDriver installed and on PATH (or specify --driver-path)

Output: data/kbo_games/<year>/game_log.csv
Schema: game_id, date, home_team, away_team, venue, home_score, away_score
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "kbo_games"
_DEFAULT_DRIVER = "chromedriver"

# KBO regular season: April–October
_SEASON_MONTHS = list(range(4, 11))


def _import_kbodata():
    try:
        import kbodata
        return kbodata
    except ImportError:
        raise ImportError("kbodata is required. pip install kbodata")


def download_year(year: int, output_dir: Path, driver_path: str, delay: float = 1.0) -> Path:
    kbodata = _import_kbodata()

    all_schedules = []
    for month in _SEASON_MONTHS:
        try:
            logger.info("Fetching schedule %d-%02d ...", year, month)
            sched = kbodata.get_yearly_schedule(year, driver_path)
            all_schedules.append(sched)
            break  # get_yearly_schedule covers the full year in one call
        except Exception:
            pass

    # Fallback: fetch month by month if yearly call failed
    if not all_schedules:
        for month in _SEASON_MONTHS:
            try:
                logger.info("Fetching schedule %d-%02d ...", year, month)
                # Try each day — kbodata has no get_monthly_schedule in 0.2.x
                days_in_month = 31
                for day in range(1, days_in_month + 1):
                    try:
                        sched = kbodata.get_daily_schedule(year, month, day, driver_path)
                        if sched is not None and not sched.empty:
                            all_schedules.append(sched)
                    except Exception:
                        pass
                    time.sleep(delay)
            except Exception as e:
                logger.warning("Month %d-%02d failed: %s", year, month, e)

    if not all_schedules:
        logger.error("No schedule data retrieved for %d", year)
        return None

    sched_df = pd.concat(all_schedules, ignore_index=True)
    # Only keep finished games
    if "status" in sched_df.columns:
        sched_df = sched_df[sched_df["status"] == "finished"].copy()

    # Build game_id: date + gameid  e.g. "20240501_LGOB0"
    sched_df["game_id"] = sched_df["date"].astype(str) + "_" + sched_df["gameid"].astype(str)

    out_path = output_dir / str(year) / "game_log.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = sched_df[["game_id", "date", "home", "away", "gameid"]].rename(
        columns={"home": "home_team", "away": "away_team"}
    )
    result.to_csv(out_path, index=False)
    logger.info("Saved %d games → %s", len(result), out_path)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download KBO game schedules via kbodata")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--year", type=int)
    group.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int, default=date.today().year - 1)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--driver-path", type=str, default=_DEFAULT_DRIVER,
                        help="Path to ChromeDriver binary")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between requests")
    args = parser.parse_args(argv)

    if args.year:
        years = [args.year]
    elif args.start_year:
        years = list(range(args.start_year, args.end_year + 1))
    else:
        years = [date.today().year - 1]

    for year in years:
        download_year(year, args.output_dir, args.driver_path, args.delay)


if __name__ == "__main__":
    main()
