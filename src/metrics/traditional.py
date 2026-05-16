"""
Traditional baseball metrics computable from KBO CSV columns.

All functions operate on scalars or pandas Series/arrays.
Guard against division by zero: return np.nan when denominator is 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(num, den):
    """Element-wise division; returns np.nan where den == 0."""
    if isinstance(num, pd.Series) or isinstance(den, pd.Series):
        num = pd.to_numeric(num, errors="coerce")
        # Ensure den is always a Series of the same length
        if not isinstance(den, pd.Series):
            den = pd.Series([den] * len(num), index=num.index if hasattr(num, 'index') else None)
        else:
            den = pd.to_numeric(den, errors="coerce")
        safe_den = den.where(den != 0, other=np.nan)
        return num / safe_den
    if den == 0:
        return np.nan
    return num / den


# ---------------------------------------------------------------------------
# Batting
# ---------------------------------------------------------------------------

def batting_avg(h, ab):
    """H / AB"""
    return _safe_div(h, ab)


def on_base_pct(h, bb, hbp, pa):
    """(H + BB + HBP) / PA"""
    return _safe_div(h + bb + hbp, pa)


def slugging_pct(tb, ab):
    """TB / AB"""
    return _safe_div(tb, ab)


def ops(obp, slg):
    """OBP + SLG"""
    return obp + slg


def isolated_power(slg, avg):
    """SLG - AVG"""
    return slg - avg


def babip_batting(h, hr, ab, so, sf=0):
    """(H - HR) / (AB - SO - HR + SF)"""
    num = h - hr
    den = ab - so - hr + sf
    return _safe_div(num, den)


def k_pct(so, pa):
    """SO / PA"""
    return _safe_div(so, pa)


def bb_pct(bb, pa):
    """BB / PA"""
    return _safe_div(bb, pa)


def sb_pct(sb, cs):
    """SB / (SB + CS)"""
    return _safe_div(sb, sb + cs)


def singles(h, doubles, triples, hr):
    """1B = H - 2B - 3B - HR"""
    return h - doubles - triples - hr


# ---------------------------------------------------------------------------
# Pitching
# ---------------------------------------------------------------------------

def innings_pitched(ipouts):
    """IPouts / 3 → fractional innings."""
    return _safe_div(ipouts, 3)


def era(er, ip):
    """(ER / IP) * 9"""
    return _safe_div(er, ip) * 9


def whip(h, bb, ip):
    """(H + BB) / IP"""
    return _safe_div(h + bb, ip)


def k_per_9(so, ip):
    """(SO / IP) * 9"""
    return _safe_div(so, ip) * 9


def bb_per_9(bb, ip):
    """(BB / IP) * 9"""
    return _safe_div(bb, ip) * 9


def hr_per_9(hr, ip):
    """(HR / IP) * 9"""
    return _safe_div(hr, ip) * 9


def k_bb_ratio(so, bb):
    """SO / BB"""
    return _safe_div(so, bb)


def babip_pitching(h, hr, bfp, so):
    """(H - HR) / (BFP - SO - HR)"""
    num = h - hr
    den = bfp - so - hr
    return _safe_div(num, den)


def win_pct(w, l):
    """W / (W + L)"""
    return _safe_div(w, w + l)


# ---------------------------------------------------------------------------
# Convenience: enrich a pitching DataFrame in place
# ---------------------------------------------------------------------------

def enrich_pitching(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived pitching columns to an existing pitching DataFrame."""
    df = df.copy()
    df["IP"] = innings_pitched(df["IPouts"])
    df["K_per_9"] = k_per_9(df["SO"], df["IP"])
    df["BB_per_9"] = bb_per_9(df["BB"], df["IP"])
    df["HR_per_9"] = hr_per_9(df["HR"], df["IP"])
    df["WHIP"] = whip(df["H"], df["BB"], df["IP"])
    df["K_BB"] = k_bb_ratio(df["SO"], df["BB"])
    df["BABIP_p"] = babip_pitching(df["H"], df["HR"], df["BFP"], df["SO"])
    if "W" in df.columns and "L" in df.columns:
        df["WPCT_calc"] = win_pct(df["W"], df["L"])
    return df


def enrich_batting(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived batting columns to an existing batting DataFrame."""
    df = df.copy()
    hbp = df["HBP"] if "HBP" in df.columns else 0
    sf = df["SF"] if "SF" in df.columns else 0

    if "OBP" not in df.columns or df["OBP"].isna().all():
        df["OBP"] = on_base_pct(df["H"], df["BB"], hbp, df["PA"])
    if "SLG" not in df.columns or df["SLG"].isna().all():
        df["SLG"] = slugging_pct(df["TB"], df["AB"])
    if "AVG" not in df.columns or df["AVG"].isna().all():
        df["AVG"] = batting_avg(df["H"], df["AB"])

    df["OPS"] = ops(df["OBP"], df["SLG"])
    df["ISO"] = isolated_power(df["SLG"], df["AVG"])
    df["K_pct"] = k_pct(df["SO"], df["PA"])
    df["BB_pct"] = bb_pct(df["BB"], df["PA"])
    df["SB_pct"] = sb_pct(df["SB"], df["CS"])

    doubles = df["2B"] if "2B" in df.columns else 0
    triples = df["3B"] if "3B" in df.columns else 0
    df["1B"] = singles(df["H"], doubles, triples, df["HR"])
    df["BABIP"] = babip_batting(df["H"], df["HR"], df["AB"], df["SO"], sf)

    return df
