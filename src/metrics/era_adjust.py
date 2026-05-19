"""
Pre/post-2019 KBO run environment normalization.

In 2019 KBO introduced a dejuiced ball, dramatically reducing scoring.
Any cross-era comparison of rate stats must account for this structural break.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate league-average runs per game by era (derived from historical data)
# These can be overridden by compute_run_env_from_data() once game logs are available.
_ERA_RUNS_PER_GAME: dict[int, float] = {
    # Pre-2019 juiced ball (higher scoring)
    2006: 5.05, 2007: 4.85, 2008: 5.10, 2009: 5.20, 2010: 4.95,
    2011: 4.80, 2012: 4.90, 2013: 5.00, 2014: 5.10, 2015: 5.30,
    2016: 5.50, 2017: 5.65, 2018: 5.60,
    # Post-2019 dejuiced ball (lower scoring)
    2019: 4.35, 2020: 4.10, 2021: 4.25, 2022: 4.30, 2023: 4.40,
    2024: 4.45, 2025: 4.45,
}

BALL_ERA_BOUNDARY = 2019
_BASELINE_YEAR = 2022  # normalisation reference year


def run_env_factor(year: int) -> float:
    """Return scaling factor relative to _BASELINE_YEAR.

    factor > 1  → that year was a higher-scoring environment than baseline
    factor < 1  → lower-scoring environment
    Multiply a rate stat from `year` by `1 / factor` to normalize to baseline.
    """
    baseline = _ERA_RUNS_PER_GAME.get(_BASELINE_YEAR, 4.30)
    year_rpg = _ERA_RUNS_PER_GAME.get(year, baseline)
    return year_rpg / baseline


def era_adjust(value: float, year: int) -> float:
    """Scale a run-rate stat from `year` to baseline era."""
    factor = run_env_factor(year)
    if factor == 0:
        return np.nan
    return value / factor


def era_adjust_series(series: pd.Series, year_series: pd.Series) -> pd.Series:
    """Vectorized era adjustment across a DataFrame column."""
    factors = year_series.map(run_env_factor).fillna(1.0)
    return series / factors


def ball_era_flag(year: int) -> int:
    """Binary: 0 = juiced ball (pre-2019), 1 = dejuiced ball (2019+)."""
    return 1 if year >= BALL_ERA_BOUNDARY else 0


def compute_run_env_from_data(game_log: pd.DataFrame) -> dict[int, float]:
    """Compute actual runs-per-game from game log data and update internal table.

    Args:
        game_log: DataFrame with columns date, home_score, away_score.

    Returns:
        Dict mapping year -> avg runs per game (both teams).
    """
    if game_log.empty:
        return {}

    df = game_log.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["total_runs"] = df["home_score"] + df["away_score"]

    rpg = df.groupby("year")["total_runs"].mean().to_dict()
    _ERA_RUNS_PER_GAME.update(rpg)
    return rpg


def park_adjust(value: float, park_factor: float) -> float:
    """Adjust a rate stat for park effects.

    park_factor > 1 = hitter-friendly park; divide to neutralize.
    """
    if park_factor == 0:
        return np.nan
    return value / park_factor
