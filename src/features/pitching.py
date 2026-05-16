"""
Season-level pitching feature computation.

Produces one row per player-year-team with all pitching features
needed for pregame prediction models (starter and reliever profiles).
"""
from __future__ import annotations

import pandas as pd

from src.data.loader import load_table
from src.metrics.advanced import enrich_pitching_advanced, compute_fip_constant
from src.features.people import enrich_with_people


def build_season_pitching_features(
    start_year: int = 2006,
    end_year: int = 2024,
    min_ip: float = 20.0,
    data_root=None,
) -> pd.DataFrame:
    """Load and enrich pitching data across seasons.

    Returns DataFrame with one row per (playerID, yearID, teamID) with
    all traditional and advanced pitching features.
    """
    raw = load_table("pitching", start_year, end_year, data_root=data_root)
    if raw.empty:
        return pd.DataFrame()

    # Compute IP before filtering
    raw["IP_raw"] = raw["IPouts"] / 3.0

    # Use most recent stint for mid-season trades
    raw = (raw.sort_values(["playerID", "yearID", "stint"])
              .groupby(["playerID", "yearID"], as_index=False)
              .last())

    # Filter to qualified pitchers
    raw = raw[raw["IP_raw"].fillna(0) >= min_ip].copy()

    raw["ball_era"] = (raw["yearID"] >= 2019).astype(int)

    # Compute per-season FIP constant and apply
    frames = []
    for year, group in raw.groupby("yearID"):
        fip_c = compute_fip_constant(group)
        enriched = enrich_pitching_advanced(group.copy(), fip_constant=fip_c)
        frames.append(enriched)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = enrich_with_people(df)

    # Classify as starter vs reliever
    df["is_starter"] = (df["GS"] if "GS" in df.columns else 0 >= df["G"] * 0.5).astype(int)

    return df


def rolling_pitching_features(
    df: pd.DataFrame,
    window: int = 5,
    player_col: str = "playerID",
    date_col: str = "date",
    stats: list[str] | None = None,
) -> pd.DataFrame:
    """Compute rolling pitching stats strictly before each row's date."""
    if stats is None:
        stats = ["ERA", "FIP", "WHIP", "K_per_9", "BB_per_9", "BABIP_p"]

    df = df.sort_values([player_col, date_col]).copy()
    for stat in stats:
        if stat not in df.columns:
            continue
        col_name = f"{stat}_roll{window}"
        df[col_name] = (
            df.groupby(player_col)[stat]
              .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
        )
    return df


def compute_days_rest(starts_df: pd.DataFrame,
                      player_col: str = "playerID",
                      date_col: str = "date") -> pd.DataFrame:
    """Add days_rest column to a DataFrame of pitcher game appearances."""
    df = starts_df.sort_values([player_col, date_col]).copy()
    df["prev_date"] = df.groupby(player_col)[date_col].shift(1)
    df["days_rest"] = (
        pd.to_datetime(df[date_col]) - pd.to_datetime(df["prev_date"])
    ).dt.days
    return df
