"""
Tests for the RL environment and monitoring modules.
"""
import pytest
import numpy as np
import pandas as pd

from src.rl.env import KBOInningEnv, OBS_DIM, LIVE_FEATURE_KEYS
from src.monitoring.validation import run_sanity_checks, full_validation
from src.monitoring.feature_drift import compute_psi, detect_feature_drift


# ---------------------------------------------------------------------------
# KBOInningEnv
# ---------------------------------------------------------------------------

def _make_inning_data(game_id: str = "TEST_GAME_001", n_innings: int = 9) -> pd.DataFrame:
    rows = []
    for inning in range(1, n_innings + 1):
        for half in ["top", "bot"]:
            rows.append({
                "game_id": game_id,
                "inning": inning,
                "half": half,
                "team": "TEAM_A" if half == "top" else "TEAM_B",
                "runs": int(np.random.poisson(0.55)),
                "hits": 0, "errors": 0, "walks": 0,
            })
    return pd.DataFrame(rows)


class TestKBOInningEnv:
    def test_obs_shape(self):
        env = KBOInningEnv(year=2023)
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},) got {obs.shape}"

    def test_obs_dtype(self):
        env = KBOInningEnv(year=2023)
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        assert obs.dtype == np.float32

    def test_full_episode_completes(self):
        innings = _make_inning_data()
        env = KBOInningEnv(inning_data=innings, year=2023)
        result = env.reset(game_id="TEST_GAME_001")
        obs = result[0] if isinstance(result, tuple) else result

        rewards = []
        for _ in range(30):   # enough to exhaust any KBO game
            action = np.array([1.5], dtype=np.float32)
            result = env.step(action)
            if len(result) == 5:
                obs, r, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, r, done, info = result
            rewards.append(r)
            if done:
                break

        assert len(rewards) > 0, "Episode never started"
        assert all(r <= 0.0 for r in rewards), "Rewards should be non-positive (negative MAE)"

    def test_reward_is_negative_mae(self):
        """Predict exactly the right number of runs → reward should be 0."""
        rows = [{"game_id": "G1", "inning": 1, "half": "top",
                 "team": "T", "runs": 3, "hits": 0, "errors": 0, "walks": 0}]
        innings = pd.DataFrame(rows)
        env = KBOInningEnv(inning_data=innings, year=2023)
        env.reset(game_id="G1")

        action = np.array([3.0], dtype=np.float32)
        result = env.step(action)
        r = result[1]
        assert abs(r - 0.0) < 1e-6, f"Expected reward 0.0, got {r}"

    def test_synthetic_episode_without_data(self):
        """Env should function even without real inning data."""
        env = KBOInningEnv(year=2023, seed=0)
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        assert obs.shape == (OBS_DIM,)

    def test_feature_key_count(self):
        assert len(LIVE_FEATURE_KEYS) == OBS_DIM

    def test_max_innings_2025(self):
        env = KBOInningEnv(year=2025)
        assert env.max_innings == 11

    def test_max_innings_pre2025(self):
        env = KBOInningEnv(year=2023)
        assert env.max_innings == 12


# ---------------------------------------------------------------------------
# Monitoring: validation
# ---------------------------------------------------------------------------

class TestValidation:
    def _good_df(self):
        return pd.DataFrame([{
            "ERA": 3.50, "FIP": 3.20, "WHIP": 1.15,
            "K_pct": 0.25, "BB_pct": 0.08, "BABIP": 0.300,
            "G": 28, "AB": 0, "H": 0, "HR": 18, "BB": 55, "SO": 185,
            "PA": 600, "TB": 0, "RBI": 0, "BFP": 720, "IPouts": 540,
            "SB": 5, "CS": 2, "R": 72, "ER": 65, "W": 14, "L": 8,
            "SV": 0,
        }])

    def test_clean_data_no_warnings(self):
        warnings = run_sanity_checks(self._good_df(), "test")
        assert len(warnings) == 0

    def test_out_of_range_era_flagged(self):
        df = self._good_df()
        df["ERA"] = 50.0   # clearly wrong
        warnings = run_sanity_checks(df, "test")
        assert any("ERA" in w for w in warnings)

    def test_negative_hr_flagged(self):
        df = self._good_df()
        df["HR"] = -1
        warnings = run_sanity_checks(df, "test")
        assert any("HR" in w for w in warnings)

    def test_full_validation_returns_dict(self):
        result = full_validation(self._good_df(), "test_table",
                                  key_cols=["ERA"])
        assert "all_clear" in result
        assert result["rows"] == 1


# ---------------------------------------------------------------------------
# Monitoring: feature drift
# ---------------------------------------------------------------------------

class TestFeatureDrift:
    def test_psi_identical_distributions(self):
        rng = np.random.default_rng(0)
        data = pd.Series(rng.normal(3.5, 0.5, 500))
        psi = compute_psi(data, data.copy())
        assert psi < 0.05  # nearly 0

    def test_psi_different_distributions(self):
        rng = np.random.default_rng(0)
        ref = pd.Series(rng.normal(3.5, 0.5, 500))
        cur = pd.Series(rng.normal(5.0, 0.5, 500))   # mean shifted
        psi = compute_psi(ref, cur)
        assert psi > 0.10  # should flag drift

    def test_psi_too_small_series(self):
        tiny = pd.Series([1.0, 2.0, 3.0])
        psi = compute_psi(tiny, tiny)
        assert np.isnan(psi)

    def test_detect_drift_returns_dict(self):
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"ERA": rng.normal(3.5, 0.5, 200)})
        cur = pd.DataFrame({"ERA": rng.normal(5.0, 0.5, 200)})
        results = detect_feature_drift(cur, ref, features=["ERA"])
        assert "ERA" in results
        assert results["ERA"]["status"] in ("alert", "warn", "stable")
