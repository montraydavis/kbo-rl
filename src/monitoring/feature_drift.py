"""
Population Stability Index (PSI) based feature drift detection.

Compares current feature distribution against a reference (training) distribution.
PSI > 0.25 → significant drift; alert.
PSI 0.10–0.25 → moderate drift; monitor.
PSI < 0.10 → stable.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PSI_THRESHOLDS = {"alert": 0.25, "warn": 0.10}

# Features to monitor for drift
MONITORED_FEATURES = [
    "ERA", "FIP", "WHIP", "K_pct", "BB_pct", "BABIP",
    "wOBA", "wRC_plus", "OPS", "ISO",
    "park_factor_runs", "age",
]


def compute_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Compute Population Stability Index between expected and actual distributions.

    Args:
        expected: Reference distribution (e.g. training data).
        actual: Current distribution.
        buckets: Number of quantile buckets.

    Returns:
        PSI value (float). NaN if computation impossible.
    """
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()

    if len(expected) < 10 or len(actual) < 10:
        return np.nan

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid log(0)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def detect_feature_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    alert_threshold: float = PSI_THRESHOLDS["alert"],
    warn_threshold: float = PSI_THRESHOLDS["warn"],
) -> Dict[str, dict]:
    """Compute PSI for each feature and flag drift.

    Returns:
        Dict mapping feature name → {psi, status, message}.
    """
    features = features or MONITORED_FEATURES
    results = {}

    for feat in features:
        if feat not in current_df.columns or feat not in reference_df.columns:
            continue

        psi = compute_psi(reference_df[feat], current_df[feat])
        if np.isnan(psi):
            status = "insufficient_data"
        elif psi >= alert_threshold:
            status = "alert"
        elif psi >= warn_threshold:
            status = "warn"
        else:
            status = "stable"

        if status == "alert":
            logger.warning("Feature drift ALERT: %s PSI=%.4f", feat, psi)
        elif status == "warn":
            logger.info("Feature drift warn: %s PSI=%.4f", feat, psi)

        results[feat] = {
            "psi": round(psi, 4) if not np.isnan(psi) else None,
            "status": status,
        }

    return results


def drift_report(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return a DataFrame summary of drift analysis."""
    results = detect_feature_drift(current_df, reference_df, features)
    rows = []
    for feat, info in results.items():
        rows.append({"feature": feat, "psi": info["psi"], "status": info["status"]})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")
