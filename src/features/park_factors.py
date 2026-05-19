"""
Park factor computation and lookup for KBO venues.

Park factors are computed from game-log data (home vs. away run rates).
Directional priors are provided as fallback when data is insufficient.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# Directional priors (qualitative, from analyst community)
# >1.0 = hitter-friendly; <1.0 = pitcher-friendly; 1.0 = neutral
_PARK_PRIORS: dict[str, float] = {
    "Jamsil":    0.98,   # LG Twins / Doosan Bears — large, moderate
    "Gwangju":   1.08,   # Kia Tigers — smaller dimensions, hitter-friendly
    "Changwon":  1.02,   # NC Dinos — newer stadium, slight hitter tilt
    "Daejeon":   0.97,   # Hanwha Eagles — humid summers; pitcher-friendly
    "Incheon":   1.00,   # SSG Landers — neutral
    "Suwon":     1.00,   # KT Wiz
    "Busan":     1.03,   # Lotte Giants
    "Daegu":     1.01,   # Samsung Lions
    "Gochon":    1.00,   # Hanwha (newer)
}

# Map common abbreviations / alternate names to canonical venue
_VENUE_ALIASES: dict[str, str] = {
    "jamsil": "Jamsil",
    "gwangju": "Gwangju",
    "changwon": "Changwon",
    "nc park": "Changwon",
    "daejeon": "Daejeon",
    "hanwha life": "Daejeon",
    "incheon": "Incheon",
    "ssg": "Incheon",
    "suwon": "Suwon",
    "kt wiz park": "Suwon",
    "busan": "Busan",
    "sajik": "Busan",
    "daegu": "Daegu",
    "samsung": "Daegu",
}


def _normalize_venue(venue: str) -> str:
    return _VENUE_ALIASES.get(venue.lower(), venue)


def get_park_factor(venue: str, computed: dict[str, float] | None = None) -> float:
    """Return run park factor for a venue.

    Uses computed factors when available; falls back to priors.
    """
    canonical = _normalize_venue(venue)
    if computed and canonical in computed:
        return computed[canonical]
    return _PARK_PRIORS.get(canonical, 1.0)


def compute_park_factors(
    game_log: pd.DataFrame,
    stat: str = "runs",
    min_games: int = 20,
) -> dict[str, float]:
    """Compute park factors from game-log data.

    Method: compare average total runs in home games at each venue vs.
    the same teams' average total runs in away games.

    Args:
        game_log: DataFrame with columns: date, home_team, away_team,
                  home_score, away_score, venue.
        stat: 'runs' (only option currently; extendable to 'hr', 'hits').
        min_games: Minimum home games required before trusting the computed factor.

    Returns:
        Dict mapping venue → park factor.
    """
    if game_log.empty:
        return {}

    df = game_log.copy()
    df["total_runs"] = df["home_score"] + df["away_score"]

    park_factors = {}
    for venue, venue_games in df.groupby("venue"):
        if len(venue_games) < min_games:
            continue

        home_rate = venue_games["total_runs"].mean()

        # Away rate: same teams' games at other venues
        home_teams = venue_games["home_team"].unique()
        away_games = df[
            (df["home_team"].isin(home_teams) | df["away_team"].isin(home_teams)) &
            (df["venue"] != venue)
        ]
        if away_games.empty:
            continue

        away_rate = away_games["total_runs"].mean()
        if away_rate == 0:
            continue

        pf = home_rate / away_rate
        canonical = _normalize_venue(str(venue))
        park_factors[canonical] = round(pf, 4)

    return park_factors


def build_park_factor_table(
    game_logs: list[pd.DataFrame],
    years: list[int],
    blend_prior_weight: float = 0.3,
) -> pd.DataFrame:
    """Build a multi-year park factor table, blending computed with priors.

    Returns DataFrame with columns: venue, year, park_factor.
    """
    rows = []
    for year, gl in zip(years, game_logs):
        computed = compute_park_factors(gl)
        venues = set(list(_PARK_PRIORS.keys()) + list(computed.keys()))
        for venue in venues:
            prior = _PARK_PRIORS.get(venue, 1.0)
            data_pf = computed.get(venue, prior)
            # Blend: weight computed data more when we have it
            blended = blend_prior_weight * prior + (1 - blend_prior_weight) * data_pf
            rows.append({"venue": venue, "year": year, "park_factor": round(blended, 4)})

    return pd.DataFrame(rows)
