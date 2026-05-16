"""
Scheduling and contextual features: back-to-back games, series position,
games density, days of rest.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def add_context_features(df: pd.DataFrame,
                          team_col: str = "team",
                          date_col: str = "date",
                          opponent_col: str = "opponent") -> pd.DataFrame:
    """Add scheduling context features to a team-game DataFrame.

    Input: one row per (team, game_date).
    Adds: is_back_to_back, games_last_7d, series_position, days_since_last_game.
    """
    if df.empty:
        return df

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values([team_col, date_col]).reset_index(drop=True)

    # Days since last game
    df["prev_date"] = df.groupby(team_col)[date_col].shift(1)
    df["days_since_last"] = (df[date_col] - df["prev_date"]).dt.days
    df["is_back_to_back"] = (df["days_since_last"] == 1).astype(int)

    # Games in last 7 days (rolling count, exclude current)
    df["days_into_season"] = df.groupby([team_col, df[date_col].dt.year])[date_col].transform(
        lambda x: (x - x.min()).dt.days
    )

    # Games last 7d: count rows in [date-7, date) per team
    def _games_in_window(group, days=7):
        result = []
        dates = group[date_col].values
        for i, d in enumerate(dates):
            cutoff = d - pd.Timedelta(days=days)
            count = ((dates[:i] >= cutoff) & (dates[:i] < d)).sum()
            result.append(count)
        return result

    counts = df.groupby(team_col, group_keys=False).apply(
        lambda g: pd.Series(_games_in_window(g), index=g.index)
    )
    df["games_last_7d"] = counts

    # Series position: which game in a consecutive series vs same opponent
    if opponent_col in df.columns:
        df["series_key"] = df[team_col] + "_" + df[opponent_col]
        df["prev_opponent"] = df.groupby(team_col)[opponent_col].shift(1)
        df["prev_date2"] = df.groupby(team_col)[date_col].shift(1)

        # Increment series counter when opponent changes or gap > 1 day
        new_series = (
            (df[opponent_col] != df["prev_opponent"]) |
            (df["days_since_last"].fillna(99) > 1)
        )
        df["series_id"] = new_series.groupby(df[team_col]).cumsum()
        df["series_position"] = df.groupby([team_col, "series_id"]).cumcount() + 1

        df.drop(columns=["series_key", "prev_opponent", "prev_date2", "series_id"],
                inplace=True)

    df.drop(columns=["prev_date"], inplace=True, errors="ignore")
    return df


def days_into_season(date_series: pd.Series, season_start: pd.Series) -> pd.Series:
    """Compute days elapsed since season start for each row."""
    return (pd.to_datetime(date_series) - pd.to_datetime(season_start)).dt.days
