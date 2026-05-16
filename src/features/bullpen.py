"""
Bullpen fatigue features from game-log and inning-level data.

Tracks recent usage (IP over last 3/7 days) and rolling ERA for bullpen arms.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def compute_bullpen_usage(
    pitcher_game_log: pd.DataFrame,
    team_col: str = "teamID",
    player_col: str = "playerID",
    date_col: str = "date",
    ip_col: str = "IP",
    is_starter_col: str = "is_starter",
) -> pd.DataFrame:
    """Compute rolling bullpen IP per team over 3-day and 7-day windows.

    Args:
        pitcher_game_log: One row per pitcher per game appearance.
            Must have teamID, playerID, date, IP, is_starter columns.

    Returns:
        Per-team-per-date DataFrame with bullpen_ip_3d, bullpen_ip_7d,
        bullpen_era_7d, closer_pitched_yesterday.
    """
    if pitcher_game_log.empty:
        return pd.DataFrame()

    df = pitcher_game_log.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Relievers only
    if is_starter_col in df.columns:
        relievers = df[df[is_starter_col] == 0].copy()
    else:
        relievers = df.copy()

    # Aggregate to team-date level
    team_daily = (
        relievers.groupby([team_col, date_col])
        .agg(
            bullpen_ip=pd.NamedAgg(column=ip_col, aggfunc="sum"),
            bullpen_er=pd.NamedAgg(column="ER", aggfunc="sum") if "ER" in relievers.columns else pd.NamedAgg(column=ip_col, aggfunc="count"),
        )
        .reset_index()
    )
    team_daily = team_daily.sort_values([team_col, date_col])

    def _rolling_ip(group, days):
        result = []
        dates = group[date_col].values
        ips = group["bullpen_ip"].values
        for i, d in enumerate(dates):
            cutoff = d - pd.Timedelta(days=days)
            mask = (dates < d) & (dates >= cutoff)
            result.append(ips[mask].sum())
        return result

    for window in [3, 7]:
        col = f"bullpen_ip_{window}d"
        vals = team_daily.groupby(team_col, group_keys=False).apply(
            lambda g: pd.Series(_rolling_ip(g, window), index=g.index)
        )
        team_daily[col] = vals

    # Bullpen ERA over 7-day window (ER * 9 / IP)
    if "bullpen_er" in team_daily.columns:
        ip_7 = team_daily["bullpen_ip_7d"]
        # Approximate: use same rolling window on ER
        def _rolling_er(group, days=7):
            result = []
            dates = group[date_col].values
            ers = group["bullpen_er"].values
            ips = group["bullpen_ip"].values
            for i, d in enumerate(dates):
                cutoff = d - pd.Timedelta(days=days)
                mask = (dates < d) & (dates >= cutoff)
                total_ip = ips[mask].sum()
                total_er = ers[mask].sum()
                result.append((total_er / total_ip * 9) if total_ip > 0 else np.nan)
            return result

        era_vals = team_daily.groupby(team_col, group_keys=False).apply(
            lambda g: pd.Series(_rolling_er(g), index=g.index)
        )
        team_daily["bullpen_era_7d"] = era_vals

    return team_daily


def closer_availability(
    pitcher_game_log: pd.DataFrame,
    team_col: str = "teamID",
    player_col: str = "playerID",
    date_col: str = "date",
    save_col: str = "SV",
) -> pd.DataFrame:
    """Return per-team-per-date flag: did the team's closer pitch yesterday?"""
    if pitcher_game_log.empty or save_col not in pitcher_game_log.columns:
        return pd.DataFrame()

    df = pitcher_game_log.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Identify closer as the player with the most saves for a team in a given season
    if "yearID" in df.columns:
        closers = (
            df.groupby([team_col, "yearID", player_col])[save_col]
            .sum()
            .reset_index()
            .sort_values([team_col, "yearID", save_col], ascending=[True, True, False])
            .groupby([team_col, "yearID"])
            .first()
            .reset_index()[[team_col, "yearID", player_col]]
            .rename(columns={player_col: "closer_id"})
        )
        df = df.merge(closers, on=[team_col, "yearID"], how="left")
    else:
        df["closer_id"] = None

    # Did the closer pitch yesterday?
    closer_pitched = (
        df[df[player_col] == df["closer_id"]]
        .groupby([team_col, date_col])
        .size()
        .reset_index(name="closer_pitched")
    )
    closer_pitched["closer_rested"] = (closer_pitched["closer_pitched"] == 0).astype(int)
    closer_pitched["next_date"] = closer_pitched[date_col] + pd.Timedelta(days=1)

    result = closer_pitched[[team_col, "next_date", "closer_rested"]].rename(
        columns={"next_date": date_col}
    )
    return result
