"""
Feature sanity checks and bounds validation for KBO data tables.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# (column, lo, hi) — values outside [lo, hi] are flagged
STAT_BOUNDS: List[Tuple[str, float, float]] = [
    ("AVG",      0.000, 1.000),
    ("OBP",      0.000, 1.000),
    ("SLG",      0.000, 4.000),
    ("OPS",      0.000, 5.000),
    ("ISO",     -0.100, 1.000),
    ("ERA",      0.000, 30.00),
    ("FIP",      0.000, 15.00),
    ("WHIP",     0.000, 10.00),
    ("wOBA",     0.000, 1.000),
    ("wRC_plus", 0.000, 350.0),
    ("BABIP",    0.000, 1.000),
    ("LOB_pct",  0.000, 1.000),
    ("K_pct",    0.000, 1.000),
    ("BB_pct",   0.000, 1.000),
    ("K_per_9",  0.000, 27.00),
    ("BB_per_9", 0.000, 15.00),
    ("HR_per_9", 0.000, 15.00),
    ("FPCT",     0.000, 1.000),
    ("WPCT",     0.000, 1.000),
    ("age",      14.00, 55.00),
    ("IP",       0.000, 300.0),
    ("park_factor_runs", 0.500, 2.000),
]

# Columns that must never be negative
NON_NEGATIVE = ["G", "AB", "H", "HR", "BB", "SO", "PA", "TB", "RBI",
                "BFP", "IPouts", "SB", "CS", "R", "ER", "W", "L", "SV"]


def run_sanity_checks(df: pd.DataFrame, table_name: str, raise_on_error: bool = False) -> List[str]:
    """Validate statistical bounds in a KBO DataFrame.

    Args:
        df: DataFrame to check.
        table_name: Label for logging.
        raise_on_error: If True, raise ValueError on any issue.

    Returns:
        List of warning messages (empty = all clear).
    """
    warnings: List[str] = []

    for col, lo, hi in STAT_BOUNDS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        bad = series[(series < lo) | (series > hi)]
        if len(bad) > 0:
            msg = f"{table_name}.{col}: {len(bad)} values outside [{lo}, {hi}] (min={series.min():.4f}, max={series.max():.4f})"
            warnings.append(msg)
            logger.warning(msg)

    for col in NON_NEGATIVE:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        bad = series[series < 0]
        if len(bad) > 0:
            msg = f"{table_name}.{col}: {len(bad)} negative values"
            warnings.append(msg)
            logger.warning(msg)

    if raise_on_error and warnings:
        raise ValueError(f"Sanity check failed for {table_name}:\n" + "\n".join(warnings))

    return warnings


def check_missing_rate(df: pd.DataFrame, table_name: str, threshold: float = 0.20) -> List[str]:
    """Warn if any column has missing rate above threshold."""
    warnings = []
    for col in df.columns:
        rate = df[col].isna().mean()
        if rate > threshold:
            msg = f"{table_name}.{col}: {rate:.1%} missing"
            warnings.append(msg)
            logger.warning(msg)
    return warnings


def check_duplicate_records(df: pd.DataFrame, table_name: str,
                              key_cols: List[str]) -> List[str]:
    """Warn if key columns have duplicate rows."""
    available = [c for c in key_cols if c in df.columns]
    if not available:
        return []
    dupes = df.duplicated(subset=available, keep=False).sum()
    if dupes > 0:
        msg = f"{table_name}: {dupes} duplicate rows on {available}"
        logger.warning(msg)
        return [msg]
    return []


def full_validation(df: pd.DataFrame, table_name: str,
                    key_cols: List[str] | None = None) -> Dict:
    """Run all checks and return summary dict."""
    bounds_warnings = run_sanity_checks(df, table_name)
    missing_warnings = check_missing_rate(df, table_name)
    dupe_warnings = check_duplicate_records(df, table_name, key_cols or [])

    return {
        "table": table_name,
        "rows": len(df),
        "bounds_issues": len(bounds_warnings),
        "missing_issues": len(missing_warnings),
        "dupe_issues": len(dupe_warnings),
        "all_clear": len(bounds_warnings) + len(missing_warnings) + len(dupe_warnings) == 0,
        "messages": bounds_warnings + missing_warnings + dupe_warnings,
    }
