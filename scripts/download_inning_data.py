"""
Download KBO inning-by-inning data using kbodata.

Fetches game scoreboard data and parses the i_1..i_18 per-inning columns
into the (game_id, inning, half, team, runs) format expected by KBOInningEnv.

Output: data/kbo_innings/<year>/<game_id>.parquet

Usage:
    # Single year
    python scripts/download_inning_data.py --year 2024

    # Multiple years in parallel (one browser per year)
    python scripts/download_inning_data.py --start-year 2019 --end-year 2024

    # Headed mode (visible browser windows)
    python scripts/download_inning_data.py --year 2024 --headed

Requires:
    pip install kbodata webdriver-manager
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "kbo_innings"
_SEASON_MONTHS = list(range(4, 11))
_INNING_COLS = [f"i_{n}" for n in range(1, 19)]


def _import_kbodata():
    try:
        import kbodata
        return kbodata
    except ImportError:
        raise ImportError("kbodata is required. pip install kbodata")


def _get_driver_path() -> str:
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        return ChromeDriverManager().install()
    except ImportError:
        return "chromedriver"


def _make_driver(driver_path: str, headed: bool):
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    options = webdriver.ChromeOptions()
    if not headed:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    return webdriver.Chrome(service=Service(driver_path), options=options)


def scoreboard_records_to_innings(scoreboard_records: list, game_id: str) -> pd.DataFrame:
    """Convert kbodata scoreboard list (2 dicts, i_1..i_18) to half-inning rows."""
    if not scoreboard_records:
        return pd.DataFrame(columns=["game_id", "inning", "half", "team", "runs"])

    rows = []
    for row_idx, record in enumerate(scoreboard_records[:2]):
        half = "top" if row_idx == 0 else "bot"
        team = str(record.get("team", f"TEAM_{row_idx}"))

        for col in _INNING_COLS:
            val = record.get(col)
            if val is None or str(val).strip() in ("", "X", "-", "x"):
                continue
            try:
                runs = int(float(val))
            except (ValueError, TypeError):
                continue
            rows.append({
                "game_id": game_id,
                "inning": int(col.split("_")[1]),
                "half": half,
                "team": team,
                "runs": runs,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["game_id", "inning", "half", "team", "runs"]
    )


def fetch_day_innings(
    year: int, month: int, day: int,
    driver_path: str, output_dir: Path, headed: bool,
) -> int:
    """Fetch and save inning data for all games on a given day."""
    kbodata = _import_kbodata()

    try:
        schedule = kbodata.get_daily_schedule(year, month, day, driver_path)
    except Exception as e:
        logger.debug("No schedule %d-%02d-%02d: %s", year, month, day, e)
        return 0

    if schedule is None or schedule.empty:
        return 0

    finished = schedule[schedule["status"] == "finished"] if "status" in schedule.columns else schedule
    if finished.empty:
        return 0

    # Skip days where all games are already on disk
    needed = finished[finished.apply(
        lambda r: not (output_dir / f"{r['date']}_{r['gameid']}.parquet").exists(), axis=1
    )]
    if needed.empty:
        logger.debug("All games for %d-%02d-%02d already downloaded", year, month, day)
        return 0

    try:
        game_data = kbodata.get_game_data(needed, driver_path)
    except Exception as e:
        logger.warning("get_game_data failed %d-%02d-%02d: %s", year, month, day, e)
        return 0

    saved = 0
    for sched_row, gdata in zip(needed.itertuples(), game_data):
        game_id = f"{sched_row.date}_{sched_row.gameid}"
        out_path = output_dir / f"{game_id}.parquet"
        if out_path.exists():
            continue
        try:
            sb_records = gdata["contents"]["scoreboard"]
            innings_df = scoreboard_records_to_innings(sb_records, game_id)
            if innings_df.empty:
                logger.warning("Empty scoreboard for %s", game_id)
                continue
            innings_df.to_parquet(out_path, index=False)
            saved += 1
            logger.info("Saved %s (%d half-innings)", game_id, len(innings_df))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", game_id, e)

    return saved


def download_year(
    year: int,
    output_dir: Path,
    driver_path: str,
    headed: bool,
    delay: float,
    months: list[int] | None = None,
) -> int:
    """Download all inning data for a single year. Runs in its own thread."""
    out_dir = output_dir / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_months = months or _SEASON_MONTHS

    total = 0
    for month in target_months:
        for day in range(1, 32):
            n = fetch_day_innings(year, month, day, driver_path, out_dir, headed)
            total += n
            if n > 0:
                time.sleep(delay)

    logger.info("Year %d complete — %d games saved.", year, total)
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download KBO inning data via kbodata")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--year", type=int)
    group.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int, default=date.today().year - 1)
    parser.add_argument("--month", type=int, help="Restrict to a single month")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--driver-path", type=str, default=None,
                        help="Path to ChromeDriver (auto-detected if omitted)")
    parser.add_argument("--headed", action="store_true",
                        help="Run browsers in headed (visible) mode")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel browser workers (default: one per year)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between daily requests per worker")
    args = parser.parse_args(argv)

    driver_path = args.driver_path or _get_driver_path()
    logger.info("Using ChromeDriver: %s", driver_path)

    if args.year:
        years = [args.year]
    elif args.start_year:
        years = list(range(args.start_year, args.end_year + 1))
    else:
        years = [date.today().year - 1]

    months = [args.month] if args.month else None
    workers = args.workers or len(years)

    if len(years) == 1:
        total = download_year(years[0], args.output_dir, driver_path,
                              args.headed, args.delay, months)
        logger.info("Done. Total games saved: %d", total)
        return

    logger.info("Downloading %d years with %d parallel workers: %s",
                len(years), workers, years)

    grand_total = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kbo") as pool:
        futures = {
            pool.submit(download_year, yr, args.output_dir, driver_path,
                        args.headed, args.delay, months): yr
            for yr in years
        }
        for future in as_completed(futures):
            yr = futures[future]
            try:
                n = future.result()
                grand_total += n
                logger.info("Year %d finished: %d games", yr, n)
            except Exception as e:
                logger.error("Year %d failed: %s", yr, e)

    logger.info("All done. Grand total: %d games saved.", grand_total)


if __name__ == "__main__":
    main()
