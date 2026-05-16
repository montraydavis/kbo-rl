"""
Advanced sabermetric metrics for KBO.

FIP constant and wOBA weights are KBO-specific and must be recalibrated
annually from league-average data. Defaults are post-2019 era estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.traditional import _safe_div, innings_pitched


# ---------------------------------------------------------------------------
# KBO-calibrated constants (post-2019 era)
# ---------------------------------------------------------------------------

KBO_FIP_CONSTANT = 3.10   # recalibrate with compute_fip_constant() each season
KBO_WOBA_SCALE = 1.157    # converts wOBA to run scale (approx, post-2019)

# Linear weights (post-2019 era; recalibrate annually)
WOBA_WEIGHTS = {
    "BB":  0.690,
    "HBP": 0.720,
    "1B":  0.880,
    "2B":  1.247,
    "3B":  1.578,
    "HR":  2.031,
}

# Positional adjustment for batting WAR (runs above average per 162 games)
POSITION_ADJUSTMENTS = {
    "C":   12.5,
    "SS":   7.5,
    "2B":   2.5,
    "3B":   2.5,
    "CF":   2.5,
    "RF":  -7.5,
    "LF":  -7.5,
    "1B": -12.5,
    "DH": -17.5,
}

KBO_RUNS_PER_WIN = 9.5   # approximate for post-2019 KBO scoring environment


# ---------------------------------------------------------------------------
# FIP
# ---------------------------------------------------------------------------

def fip(hr, bb, hbp, so, ip, fip_constant: float = KBO_FIP_CONSTANT):
    """Fielding Independent Pitching.

    Formula: ((13*HR) + (3*(BB+HBP)) - (2*SO)) / IP + FIP_constant
    """
    numerator = (13 * hr) + (3 * (bb + hbp)) - (2 * so)
    return _safe_div(numerator, ip) + fip_constant


def compute_fip_constant(season_pitching: pd.DataFrame) -> float:
    """Derive the FIP constant from a full season's pitching data.

    FIP_C = lgERA - raw_FIP_components
    """
    df = season_pitching.copy()
    df["IP"] = innings_pitched(df["IPouts"])
    total_ip = df["IP"].sum()
    if total_ip == 0:
        return KBO_FIP_CONSTANT

    raw = ((13 * df["HR"].sum()) + (3 * (df["BB"].sum() + df["HBP"].sum()))
           - (2 * df["SO"].sum())) / total_ip

    total_er = df["ER"].sum()
    lg_era = (total_er / total_ip) * 9
    return lg_era - raw


# ---------------------------------------------------------------------------
# wOBA
# ---------------------------------------------------------------------------

def woba(bb, hbp, singles, doubles, triples, hr, pa,
         weights: dict = WOBA_WEIGHTS):
    """Weighted On-Base Average."""
    numerator = (
        weights["BB"]  * bb  +
        weights["HBP"] * hbp +
        weights["1B"]  * singles +
        weights["2B"]  * doubles +
        weights["3B"]  * triples +
        weights["HR"]  * hr
    )
    return _safe_div(numerator, pa)


def wraa_per_pa(woba_val, lg_woba: float, woba_scale: float = KBO_WOBA_SCALE):
    """Weighted Runs Above Average per plate appearance."""
    return (woba_val - lg_woba) / woba_scale


def wrc_plus(woba_val, lg_woba: float, woba_scale: float,
             park_factor: float, lg_r_per_pa: float):
    """Park-and-era adjusted wRC+. League average = 100."""
    wraa_pa = wraa_per_pa(woba_val, lg_woba, woba_scale)
    wrc_per_pa = (wraa_pa + lg_r_per_pa) / park_factor
    return _safe_div(wrc_per_pa, lg_r_per_pa) * 100


# ---------------------------------------------------------------------------
# BABIP (pitcher-side; batting-side is in traditional.py)
# ---------------------------------------------------------------------------

def babip_pitching(h, hr, bfp, so):
    """(H - HR) / (BFP - SO - HR)"""
    return _safe_div(h - hr, bfp - so - hr)


# ---------------------------------------------------------------------------
# LOB%
# ---------------------------------------------------------------------------

def lob_pct(h, bb, hbp, hr, r):
    """Left-on-base percentage for pitchers. Sustainable range ~68–75%."""
    numerator = h + bb + hbp - r
    denominator = h + bb + hbp - (1.4 * hr)
    result = _safe_div(numerator, denominator)
    # Clip to [0, 1] to handle edge cases (e.g., all inherited runners score)
    if isinstance(result, pd.Series):
        return result.clip(0.0, 1.0)
    if result is np.nan or result is None:
        return np.nan
    return max(0.0, min(1.0, result))


# ---------------------------------------------------------------------------
# K-BB%
# ---------------------------------------------------------------------------

def k_minus_bb_pct(so, bb, bfp):
    """(SO - BB) / BFP — command/stuff combined indicator."""
    return _safe_div(so - bb, bfp)


# ---------------------------------------------------------------------------
# WAR proxies
# ---------------------------------------------------------------------------

def batting_war_proxy(wrc_plus_val: float, pa: float, position: str,
                      runs_per_win: float = KBO_RUNS_PER_WIN) -> float:
    """Approximate batting WAR using wRC+ and positional adjustment."""
    pos_adj = POSITION_ADJUSTMENTS.get(position.upper(), 0.0)
    runs_above_avg = (wrc_plus_val - 100) / 100 * (pa / 600) * 20
    total_runs = runs_above_avg + pos_adj
    return total_runs / runs_per_win


def pitching_war_proxy(fip_val: float, lg_fip: float, ip: float,
                       runs_per_win: float = KBO_RUNS_PER_WIN) -> float:
    """Approximate pitching WAR using FIP vs league FIP."""
    fip_runs_per_9 = lg_fip - fip_val
    runs_above_avg = (fip_runs_per_9 / 9) * ip
    return runs_above_avg / runs_per_win


# ---------------------------------------------------------------------------
# Leverage index proxy (no official KBO LI; simple situation heuristic)
# ---------------------------------------------------------------------------

def leverage_proxy(inning: int, score_diff: int, outs: int = 0) -> float:
    """Situation leverage: higher = more critical moment."""
    base = 1.0
    if inning >= 7:
        base *= 1.5
    if inning >= 9:
        base *= 1.3
    abs_diff = abs(score_diff)
    if abs_diff <= 1:
        base *= 2.0
    elif abs_diff <= 3:
        base *= 1.3
    elif abs_diff >= 6:
        base *= 0.5
    return base


# ---------------------------------------------------------------------------
# Season enrichment helpers
# ---------------------------------------------------------------------------

def enrich_pitching_advanced(df: pd.DataFrame,
                              fip_constant: float = KBO_FIP_CONSTANT) -> pd.DataFrame:
    """Add advanced pitching metrics to a pitching DataFrame."""
    from src.metrics.traditional import enrich_pitching
    df = enrich_pitching(df)   # adds IP, K_per_9, BB_per_9, etc.

    hbp = df["HBP"] if "HBP" in df.columns else 0
    df["FIP"] = fip(df["HR"], df["BB"], hbp, df["SO"], df["IP"], fip_constant)

    if "ERA" in df.columns:
        df["ERA_FIP_diff"] = df["ERA"] - df["FIP"]

    df["BABIP_p"] = babip_pitching(df["H"], df["HR"], df["BFP"], df["SO"])
    df["LOB_pct"] = lob_pct(df["H"], df["BB"], hbp, df["HR"], df["R"])
    df["K_BB_pct"] = k_minus_bb_pct(df["SO"], df["BB"], df["BFP"])

    return df


def enrich_batting_advanced(df: pd.DataFrame,
                             lg_woba: float = 0.320,
                             woba_scale: float = KBO_WOBA_SCALE,
                             park_factor: float = 1.0,
                             lg_r_per_pa: float = 0.115) -> pd.DataFrame:
    """Add advanced batting metrics to a batting DataFrame."""
    from src.metrics.traditional import enrich_batting
    df = enrich_batting(df)   # adds ISO, K_pct, BB_pct, etc.

    hbp = df["HBP"] if "HBP" in df.columns else 0
    doubles = df["2B"] if "2B" in df.columns else 0
    triples = df["3B"] if "3B" in df.columns else 0

    df["wOBA"] = woba(df["BB"], hbp, df["1B"], doubles, triples, df["HR"], df["PA"])
    df["wRAA_per_PA"] = wraa_per_pa(df["wOBA"], lg_woba, woba_scale)
    df["wRC_plus"] = wrc_plus(df["wOBA"], lg_woba, woba_scale, park_factor, lg_r_per_pa)

    return df
