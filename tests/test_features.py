"""
Unit tests for advanced metrics and season-level feature enrichment.
"""
import math
import pytest
import pandas as pd
import numpy as np

from src.metrics.advanced import (
    fip, compute_fip_constant, woba, wraa_per_pa, wrc_plus,
    babip_pitching, lob_pct, k_minus_bb_pct,
    batting_war_proxy, pitching_war_proxy, leverage_proxy,
    enrich_pitching_advanced, enrich_batting_advanced,
    KBO_FIP_CONSTANT, WOBA_WEIGHTS,
)
from src.features.people import enrich_with_people


# ---------------------------------------------------------------------------
# FIP
# ---------------------------------------------------------------------------

class TestFIP:
    def test_fip_formula(self):
        # ((13*20) + (3*(50+8)) - (2*180)) / 180 + 3.10
        # = (260 + 174 - 360) / 180 + 3.10
        # = 74/180 + 3.10 = 0.4111 + 3.10 = 3.511
        result = fip(hr=20, bb=50, hbp=8, so=180, ip=180)
        assert abs(result - (74/180 + KBO_FIP_CONSTANT)) < 1e-4

    def test_fip_zero_ip(self):
        result = fip(hr=5, bb=10, hbp=2, so=30, ip=0)
        assert math.isnan(result)

    def test_fip_series(self):
        hr = pd.Series([10, 20])
        bb = pd.Series([30, 50])
        hbp = pd.Series([3, 8])
        so = pd.Series([120, 180])
        ip = pd.Series([100, 180])
        result = fip(hr, bb, hbp, so, ip)
        assert len(result) == 2
        assert not result.isna().any()

    def test_compute_fip_constant(self):
        df = pd.DataFrame([{
            "IPouts": 540, "HR": 18, "BB": 55, "HBP": 8,
            "SO": 185, "ER": 65, "R": 72,
        }])
        c = compute_fip_constant(df)
        # Should be close to KBO_FIP_CONSTANT range
        assert 2.5 < c < 4.5


# ---------------------------------------------------------------------------
# wOBA
# ---------------------------------------------------------------------------

class TestWOBA:
    def test_woba_basic(self):
        # Single realistic line: 60BB, 5HBP, 80 1B, 30 2B, 3 3B, 20 HR / 550 PA
        result = woba(bb=60, hbp=5, singles=80, doubles=30, triples=3, hr=20, pa=550)
        # Rough expected range: 0.300–0.380
        assert 0.280 < result < 0.420

    def test_woba_zero_pa(self):
        result = woba(bb=0, hbp=0, singles=0, doubles=0, triples=0, hr=0, pa=0)
        assert math.isnan(result)

    def test_wraa_per_pa_avg_player(self):
        # League-average player should have wRAA ≈ 0
        lg_woba = 0.320
        result = wraa_per_pa(0.320, lg_woba)
        assert abs(result) < 1e-9

    def test_wrc_plus_avg_player(self):
        result = wrc_plus(
            woba_val=0.320, lg_woba=0.320, woba_scale=1.157,
            park_factor=1.0, lg_r_per_pa=0.115,
        )
        assert abs(result - 100.0) < 0.01

    def test_wrc_plus_above_avg(self):
        result = wrc_plus(
            woba_val=0.400, lg_woba=0.320, woba_scale=1.157,
            park_factor=1.0, lg_r_per_pa=0.115,
        )
        assert result > 100


# ---------------------------------------------------------------------------
# LOB% and K-BB%
# ---------------------------------------------------------------------------

class TestOtherMetrics:
    def test_lob_pct_range(self):
        result = lob_pct(h=160, bb=55, hbp=8, hr=18, r=65)
        assert 0.0 <= result <= 1.0

    def test_lob_pct_clipped(self):
        # Extreme case should still be clipped to [0, 1]
        result = lob_pct(h=10, bb=5, hbp=0, hr=0, r=50)
        assert result == 0.0

    def test_k_bb_pct(self):
        result = k_minus_bb_pct(so=180, bb=50, bfp=700)
        assert abs(result - (130 / 700)) < 1e-6


# ---------------------------------------------------------------------------
# WAR proxies
# ---------------------------------------------------------------------------

class TestWAR:
    def test_batting_war_avg_player(self):
        # League-average at SS, 550 PA → ≈ positive due to positional value
        result = batting_war_proxy(wrc_plus_val=100, pa=550, position="SS")
        assert result > 0

    def test_batting_war_dh_negative(self):
        # Below-average DH should have negative WAR
        result = batting_war_proxy(wrc_plus_val=85, pa=500, position="DH")
        assert result < 0

    def test_pitching_war_better_than_league(self):
        # FIP below lgFIP → positive WAR
        result = pitching_war_proxy(fip_val=3.00, lg_fip=4.00, ip=180)
        assert result > 0

    def test_leverage_proxy_late_close(self):
        result = leverage_proxy(inning=9, score_diff=0)
        assert result > 2.0

    def test_leverage_proxy_blowout(self):
        result = leverage_proxy(inning=3, score_diff=8)
        assert result < 1.0


# ---------------------------------------------------------------------------
# Enrichment integration
# ---------------------------------------------------------------------------

class TestEnrichmentAdvanced:
    def _pitching_df(self):
        return pd.DataFrame([{
            "playerID": "p1", "yearID": 2023, "teamID": "LG",
            "G": 28, "CG": 2, "SHO": 1, "W": 14, "L": 8,
            "SV": 0, "HLD": 0, "BFP": 720, "IPouts": 540,
            "H": 175, "HR": 18, "BB": 55, "HBP": 8, "SO": 185,
            "R": 72, "ER": 65, "ERA": 3.25,
        }])

    def _batting_df(self):
        return pd.DataFrame([{
            "playerID": "p2", "yearID": 2023, "teamID": "KIA",
            "G": 130, "AB": 490, "R": 70, "H": 147,
            "2B": 28, "3B": 3, "HR": 18, "RBI": 75,
            "SB": 12, "CS": 5, "BB": 55, "SO": 95,
            "GIDP": 10, "PA": 560, "TB": 243,
            "AVG": 0.300, "OBP": 0.371, "SLG": 0.496,
        }])

    def test_advanced_pitching_has_fip(self):
        df = enrich_pitching_advanced(self._pitching_df())
        assert "FIP" in df.columns
        assert not df["FIP"].isna().all()

    def test_advanced_pitching_has_era_fip_diff(self):
        df = enrich_pitching_advanced(self._pitching_df())
        assert "ERA_FIP_diff" in df.columns

    def test_advanced_pitching_era_fip_reasonable(self):
        df = enrich_pitching_advanced(self._pitching_df())
        diff = df["ERA_FIP_diff"].iloc[0]
        assert -3.0 < diff < 3.0

    def test_advanced_batting_has_woba(self):
        df = enrich_batting_advanced(self._batting_df())
        assert "wOBA" in df.columns
        assert "wRC_plus" in df.columns

    def test_advanced_batting_wrc_plus_reasonable(self):
        df = enrich_batting_advanced(self._batting_df())
        wrc = df["wRC_plus"].iloc[0]
        assert 50 < wrc < 250


# ---------------------------------------------------------------------------
# People enrichment
# ---------------------------------------------------------------------------

class TestPeopleEnrichment:
    def _people_df(self):
        return pd.DataFrame([
            {"playerID": "p1", "birthYear": 1990, "birthMonth": 3, "birthDay": 15,
             "nationality": "KOR", "nameLast": "Kim", "nameFirst": "Jae",
             "nameGiven": "Kim Jae", "weight": 85.0, "height": 182.0,
             "bats": "R", "throws": "R", "position": "SP"},
            {"playerID": "p2", "birthYear": 1988, "birthMonth": 7, "birthDay": 22,
             "nationality": "USA", "nameLast": "Smith", "nameFirst": "John",
             "nameGiven": "John Smith", "weight": 95.0, "height": 188.0,
             "bats": "R", "throws": "R", "position": "1B"},
        ])

    def _stats_df(self):
        return pd.DataFrame([
            {"playerID": "p1", "yearID": 2023, "ERA": 3.25},
            {"playerID": "p2", "yearID": 2023, "AVG": 0.310},
        ])

    def test_adds_age(self):
        df = enrich_with_people(self._stats_df(), self._people_df())
        assert "age" in df.columns
        assert df[df.playerID == "p1"]["age"].iloc[0] == 33  # 2023 - 1990

    def test_adds_is_foreign(self):
        df = enrich_with_people(self._stats_df(), self._people_df())
        assert "is_foreign" in df.columns
        kor = df[df.playerID == "p1"]["is_foreign"].iloc[0]
        usa = df[df.playerID == "p2"]["is_foreign"].iloc[0]
        assert kor == 0
        assert usa == 1

    def test_unknown_player_no_crash(self):
        stats = pd.DataFrame([{"playerID": "unknown", "yearID": 2023}])
        df = enrich_with_people(stats, self._people_df())
        assert len(df) == 1
        assert "is_foreign" in df.columns
