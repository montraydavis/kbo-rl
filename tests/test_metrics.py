"""
Unit tests for traditional metrics and era adjustment.
Uses known baseball values to verify formula correctness.
"""
import math
import pytest
import pandas as pd
import numpy as np

from src.metrics.traditional import (
    batting_avg, on_base_pct, slugging_pct, ops, isolated_power,
    babip_batting, k_pct, bb_pct, sb_pct, singles,
    innings_pitched, era, whip, k_per_9, bb_per_9, hr_per_9,
    k_bb_ratio, babip_pitching, win_pct,
    enrich_batting, enrich_pitching,
)
from src.metrics.era_adjust import (
    run_env_factor, era_adjust, ball_era_flag, park_adjust,
    BALL_ERA_BOUNDARY,
)


# ---------------------------------------------------------------------------
# Batting metrics
# ---------------------------------------------------------------------------

class TestBattingMetrics:
    def test_avg(self):
        assert abs(batting_avg(150, 500) - 0.300) < 1e-6

    def test_avg_zero_ab(self):
        assert math.isnan(batting_avg(0, 0))

    def test_obp(self):
        # (160 + 60 + 10) / 600 = 230/600 ≈ 0.3833
        result = on_base_pct(h=160, bb=60, hbp=10, pa=600)
        assert abs(result - (230 / 600)) < 1e-6

    def test_slg(self):
        # 750 TB / 500 AB = 1.500
        assert abs(slugging_pct(750, 500) - 1.500) < 1e-6

    def test_ops(self):
        assert abs(ops(0.370, 0.510) - 0.880) < 1e-6

    def test_iso(self):
        assert abs(isolated_power(0.480, 0.290) - 0.190) < 1e-6

    def test_babip(self):
        # (150 - 20) / (500 - 100 - 20) = 130/380 ≈ 0.342
        result = babip_batting(h=150, hr=20, ab=500, so=100)
        assert abs(result - (130 / 380)) < 1e-6

    def test_k_pct(self):
        assert abs(k_pct(120, 500) - 0.240) < 1e-6

    def test_bb_pct(self):
        assert abs(bb_pct(60, 500) - 0.120) < 1e-6

    def test_sb_pct(self):
        assert abs(sb_pct(30, 10) - 0.750) < 1e-6

    def test_sb_pct_zero(self):
        assert math.isnan(sb_pct(0, 0))

    def test_singles(self):
        # H=150, 2B=30, 3B=5, HR=20 → 1B=95
        assert singles(150, 30, 5, 20) == 95

    def test_series_avg(self):
        h = pd.Series([150, 180, 100])
        ab = pd.Series([500, 600, 400])
        result = batting_avg(h, ab)
        expected = pd.Series([0.3, 0.3, 0.25])
        pd.testing.assert_series_equal(result.reset_index(drop=True),
                                        expected.reset_index(drop=True),
                                        check_names=False)


# ---------------------------------------------------------------------------
# Pitching metrics
# ---------------------------------------------------------------------------

class TestPitchingMetrics:
    def test_ip(self):
        assert abs(innings_pitched(540) - 180.0) < 1e-6

    def test_ip_fractional(self):
        # 181 IPouts = 60.333... IP
        assert abs(innings_pitched(181) - (181 / 3)) < 1e-6

    def test_era(self):
        # 72 ER / 180 IP * 9 = 3.600
        assert abs(era(72, 180) - 3.600) < 1e-6

    def test_era_zero_ip(self):
        assert math.isnan(era(5, 0))

    def test_whip(self):
        # (180 + 60) / 180 = 1.333
        assert abs(whip(180, 60, 180) - (240 / 180)) < 1e-6

    def test_k_per_9(self):
        # 200 SO / 180 IP * 9 = 10.0
        assert abs(k_per_9(200, 180) - 10.0) < 1e-6

    def test_bb_per_9(self):
        assert abs(bb_per_9(60, 180) - 3.0) < 1e-6

    def test_hr_per_9(self):
        assert abs(hr_per_9(18, 180) - 0.9) < 1e-6

    def test_k_bb_ratio(self):
        assert abs(k_bb_ratio(200, 50) - 4.0) < 1e-6

    def test_babip_pitching(self):
        # (180 - 18) / (700 - 150 - 18) = 162 / 532 ≈ 0.3045
        result = babip_pitching(h=180, hr=18, bfp=700, so=150)
        assert abs(result - (162 / 532)) < 1e-6

    def test_win_pct(self):
        assert abs(win_pct(15, 10) - 0.600) < 1e-6

    def test_win_pct_zero(self):
        assert math.isnan(win_pct(0, 0))


# ---------------------------------------------------------------------------
# Enrichment helpers
# ---------------------------------------------------------------------------

class TestEnrichment:
    def _batting_df(self):
        return pd.DataFrame([{
            "playerID": "p1", "yearID": 2023, "teamID": "LG",
            "G": 130, "AB": 490, "R": 70, "H": 147,
            "2B": 28, "3B": 3, "HR": 18, "RBI": 75,
            "SB": 12, "CS": 5, "BB": 55, "SO": 95,
            "GIDP": 10, "PA": 560, "TB": 243,
            "AVG": 0.300, "OBP": 0.371, "SLG": 0.496,
        }])

    def _pitching_df(self):
        return pd.DataFrame([{
            "playerID": "p2", "yearID": 2023, "teamID": "KIA",
            "G": 28, "CG": 2, "SHO": 1, "W": 14, "L": 8,
            "SV": 0, "HLD": 0, "BFP": 720, "IPouts": 540,
            "H": 175, "HR": 18, "BB": 55, "HBP": 8, "SO": 185,
            "R": 72, "ER": 65, "ERA": 3.25,
        }])

    def test_enrich_batting_columns(self):
        df = enrich_batting(self._batting_df())
        for col in ["OPS", "ISO", "K_pct", "BB_pct", "SB_pct", "1B", "BABIP"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_enrich_pitching_columns(self):
        df = enrich_pitching(self._pitching_df())
        for col in ["IP", "K_per_9", "BB_per_9", "HR_per_9", "WHIP", "K_BB", "BABIP_p"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_enrich_pitching_ip_value(self):
        df = enrich_pitching(self._pitching_df())
        assert abs(df["IP"].iloc[0] - 180.0) < 1e-6

    def test_enrich_batting_iso(self):
        df = enrich_batting(self._batting_df())
        # SLG=0.496, AVG=0.300 → ISO=0.196
        assert abs(df["ISO"].iloc[0] - 0.196) < 1e-4


# ---------------------------------------------------------------------------
# Era adjustment
# ---------------------------------------------------------------------------

class TestEraAdjust:
    def test_ball_era_flag_pre(self):
        assert ball_era_flag(2018) == 0

    def test_ball_era_flag_post(self):
        assert ball_era_flag(2019) == 1
        assert ball_era_flag(2023) == 1

    def test_ball_era_boundary(self):
        assert BALL_ERA_BOUNDARY == 2019

    def test_run_env_factor_baseline(self):
        # 2022 is the baseline; factor should be 1.0
        assert abs(run_env_factor(2022) - 1.0) < 1e-6

    def test_run_env_factor_high_era(self):
        # 2017 was higher-scoring than 2022 → factor > 1
        assert run_env_factor(2017) > 1.0

    def test_run_env_factor_low_era(self):
        # 2020 was lower-scoring than 2022 → factor < 1
        assert run_env_factor(2020) < 1.0

    def test_era_adjust_scales_down(self):
        # ERA of 4.0 in high-scoring 2017 → adjusted down toward lower baseline
        adjusted = era_adjust(4.0, 2017)
        assert adjusted < 4.0

    def test_park_adjust(self):
        # ERA 4.0 in hitter-friendly park (factor=1.1) → 4.0/1.1 ≈ 3.636
        result = park_adjust(4.0, 1.1)
        assert abs(result - (4.0 / 1.1)) < 1e-6

    def test_park_adjust_zero_factor(self):
        assert math.isnan(park_adjust(4.0, 0.0))
