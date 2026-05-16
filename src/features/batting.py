"""
Season-level batting feature computation.

Produces one row per player-year-team with all batting features
needed for pregame prediction models.
"""
from __future__ import annotations

import pandas as pd

from src.data.loader import load_table
from src.metrics.advanced import enrich_batting_advanced
from src.features.people import enrich_with_people


def build_season_batting_features(
    start_year: int = 2006,
    end_year: int = 2024,
    min_pa: int = 100,
    lg_woba: float = 0.320,
    data_root=None,
) -> pd.DataFrame:
    """Load and enrich batting data across seasons.

    Returns DataFrame with one row per (playerID, yearID, teamID) with
    all traditional and advanced batting features.
    """
    raw = load_table("box-scores", start_year, end_year, data_root=data_root)
    if raw.empty:
        return pd.DataFrame()

    # Use most recent stint for players traded mid-season
    raw = (raw.sort_values(["playerID", "yearID", "stint"])
              .groupby(["playerID", "yearID"], as_index=False)
              .last())

    # Filter to qualified batters
    raw = raw[raw["PA"].fillna(0) >= min_pa].copy()

    # Add era flag
    raw["ball_era"] = (raw["yearID"] >= 2019).astype(int)

    # Advanced metrics (uses league-average wOBA; could be made year-specific)
    df = enrich_batting_advanced(raw, lg_woba=lg_woba)

    # Player metadata
    df = enrich_with_people(df)

    return df


def rolling_batting_features(
    df: pd.DataFrame,
    window: int = 15,
    player_col: str = "playerID",
    date_col: str = "date",
    stats: list[str] | None = None,
) -> pd.DataFrame:
    """Compute rolling means strictly before each row's date (no leakage).

    For use with game-log level data once available in Sprint 3+.
    """
    if stats is None:
        stats = ["AVG", "OBP", "SLG", "OPS", "ISO", "K_pct", "BB_pct", "wOBA"]

    df = df.sort_values([player_col, date_col]).copy()
    for stat in stats:
        if stat not in df.columns:
            continue
        col_name = f"{stat}_roll{window}"
        df[col_name] = (
            df.groupby(player_col)[stat]
              .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
        )
    return df
