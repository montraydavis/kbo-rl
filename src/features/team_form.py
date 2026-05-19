"""
Team-level rolling performance features computed from game-log data.

Requires game_log.csv from scripts/download_game_logs.py.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.data.loader import load_game_logs


def load_and_prepare_game_log(year: int, data_root=None) -> pd.DataFrame:
    """Load game log and expand to per-team-per-game rows."""
    gl = load_game_logs(year, data_root=data_root)
    if gl.empty:
        return pd.DataFrame()

    if not pd.api.types.is_datetime64_any_dtype(gl["date"]):
        gl["date"] = pd.to_datetime(gl["date"])

    # Expand: one row per team per game
    home = gl.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "runs_scored", "away_score": "runs_allowed",
    })[["game_id", "date", "team", "opponent", "runs_scored", "runs_allowed", "venue"]].copy()
    home["is_home"] = 1

    away = gl.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "runs_scored", "home_score": "runs_allowed",
    })[["game_id", "date", "team", "opponent", "runs_scored", "runs_allowed", "venue"]].copy()
    away["is_home"] = 0

    df = pd.concat([home, away], ignore_index=True)
    df["win"] = (df["runs_scored"] > df["runs_allowed"]).astype(int)
    df["run_diff"] = df["runs_scored"] - df["runs_allowed"]
    return df.sort_values(["team", "date"]).reset_index(drop=True)


def rolling_team_form(
    df: pd.DataFrame,
    win_window: int = 15,
    runs_window: int = 10,
) -> pd.DataFrame:
    """Add rolling team form features to a team-game DataFrame.

    All rolling computations use shift(1) to avoid leakage (exclude current game).
    """
    if df.empty:
        return df

    df = df.copy()
    grp = df.groupby("team")

    # Rolling win percentage
    df[f"win_pct_{win_window}g"] = grp["win"].transform(
        lambda x: x.shift(1).rolling(win_window, min_periods=3).mean()
    )

    # Rolling runs scored / allowed
    df[f"rs_{runs_window}g"] = grp["runs_scored"].transform(
        lambda x: x.shift(1).rolling(runs_window, min_periods=3).mean()
    )
    df[f"ra_{runs_window}g"] = grp["runs_allowed"].transform(
        lambda x: x.shift(1).rolling(runs_window, min_periods=3).mean()
    )

    # Pythagorean win percentage
    rs = df[f"rs_{runs_window}g"]
    ra = df[f"ra_{runs_window}g"]
    rs2 = rs ** 2
    ra2 = ra ** 2
    df[f"pythag_{runs_window}g"] = np.where(
        (rs2 + ra2) > 0, rs2 / (rs2 + ra2), np.nan
    )

    # Run differential
    df[f"run_diff_{runs_window}g"] = grp["run_diff"].transform(
        lambda x: x.shift(1).rolling(runs_window, min_periods=3).mean()
    )

    return df


def build_team_form_features(
    years: list[int],
    win_window: int = 15,
    runs_window: int = 10,
    data_root=None,
) -> pd.DataFrame:
    """Build team form features across multiple seasons."""
    frames = []
    for year in years:
        df = load_and_prepare_game_log(year, data_root=data_root)
        if df.empty:
            continue
        df = rolling_team_form(df, win_window=win_window, runs_window=runs_window)
        df["yearID"] = year
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
