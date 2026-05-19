"""
KBOInningEnv: Gymnasium environment for per-half-inning run prediction.

State space:  fixed-size float32 vector (LIVE_FEATURES).
Action space: Box(low=0, high=10, shape=(1,)) — predicted runs this half-inning.
Reward:       -abs(predicted_runs - actual_runs)  (negative MAE; always ≤ 0)
Episode:      One game; ends after all half-innings are exhausted.

KBO rules:
  - Max innings: 11 (2025+), 12 (pre-2025). Games can end in ties.
  - Half-innings per game: up to 22 (2025+) or 24 (pre-2025).

The environment is designed for offline training against historical game data.
Live inference simply calls reset(game_id=...) then step() once per half-inning
after the inning completes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Try to import gymnasium; fall back to gym for compatibility
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_API = "gymnasium"
except ImportError:
    try:
        import gym
        from gym import spaces
        _GYM_API = "gym"
    except ImportError:
        gym = None
        spaces = None
        _GYM_API = None


# ---------------------------------------------------------------------------
# Feature ordering (determines obs vector layout — must be stable)
# ---------------------------------------------------------------------------

PREGAME_FEATURE_KEYS = [
    "ball_era", "year", "park_factor_runs",
    # Home starter
    "hsp_era", "hsp_fip", "hsp_era_fip_diff", "hsp_k_per_9", "hsp_bb_per_9",
    "hsp_hr_per_9", "hsp_whip", "hsp_babip_p", "hsp_lob_pct", "hsp_k_bb_pct",
    "hsp_is_foreign", "hsp_age", "hsp_throws_r",
    # Away starter
    "asp_era", "asp_fip", "asp_era_fip_diff", "asp_k_per_9", "asp_bb_per_9",
    "asp_hr_per_9", "asp_whip", "asp_babip_p", "asp_lob_pct", "asp_k_bb_pct",
    "asp_is_foreign", "asp_age", "asp_throws_r",
    # Team form
    "home_win_pct_15g", "home_rs_10g", "home_ra_10g", "home_run_diff_10g", "home_pythag_10g",
    "away_win_pct_15g", "away_rs_10g", "away_ra_10g", "away_run_diff_10g", "away_pythag_10g",
]

LIVE_FEATURE_KEYS = PREGAME_FEATURE_KEYS + [
    "current_inning", "half", "home_score", "away_score", "score_diff",
    "hist_mean_runs_this_half_inning",
]

OBS_DIM = len(LIVE_FEATURE_KEYS)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class KBOInningEnv:
    """Per-half-inning KBO run prediction environment.

    Works with or without gymnasium installed (fallback to dict-based API).
    """

    def __init__(
        self,
        inning_data: Optional[pd.DataFrame] = None,
        pregame_data: Optional[pd.DataFrame] = None,
        year: int = 2023,
        max_innings: Optional[int] = None,
        data_root: Optional[Path] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            inning_data: DataFrame with columns: game_id, inning, half, team, runs.
                         If None, loads from data/kbo_innings/<year>/.
            pregame_data: DataFrame with pregame features per game_id.
                          If None, uses zeros (for testing without data).
            year: Season year; determines max_innings default.
            max_innings: Override max innings per game.
            data_root: Override default data directory.
            seed: Random seed for game sampling.
        """
        self.year = year
        self.max_innings = max_innings if max_innings is not None else (11 if year >= 2025 else 12)
        self.data_root = data_root
        self.rng = np.random.default_rng(seed)

        self._inning_data = inning_data
        self._pregame_data = pregame_data

        # Gymnasium-style spaces (only set if gymnasium/gym is available)
        if spaces is not None:
            self.observation_space = spaces.Box(
                low=-10.0, high=100.0, shape=(OBS_DIM,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=0.0, high=10.0, shape=(1,), dtype=np.float32
            )

        # Episode state
        self._game_id: Optional[str] = None
        self._game_innings: Optional[pd.DataFrame] = None
        self._half_inning_idx: int = 0
        self._half_innings: list[tuple] = []  # [(inning, half, team, runs), ...]
        self._home_score: int = 0
        self._away_score: int = 0
        self._pregame_vec: np.ndarray = np.zeros(len(PREGAME_FEATURE_KEYS), dtype=np.float32)
        # (inning, half) -> mean runs; built lazily from full inning dataset
        self._hist_mean_runs: dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, game_id: Optional[str] = None, seed=None, options=None):
        """Sample (or load) a game and return initial observation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        innings = self._get_inning_data()
        if innings.empty:
            logger.warning("No inning data available; using synthetic episode.")
            self._game_id = game_id or "synthetic"
            self._half_innings = self._synthetic_episode()
        else:
            if game_id:
                game_innings = innings[innings["game_id"] == game_id]
            else:
                game_ids = innings["game_id"].unique()
                chosen = self.rng.choice(game_ids)
                game_innings = innings[innings["game_id"] == chosen]
                game_id = str(chosen)

            self._game_id = game_id
            self._half_innings = self._parse_half_innings(game_innings)

        self._half_inning_idx = 0
        self._home_score = 0
        self._away_score = 0

        # Load pregame feature vector
        self._pregame_vec = self._get_pregame_vec(self._game_id)

        # Build historical mean runs lookup from full dataset on first reset
        if not self._hist_mean_runs:
            self._hist_mean_runs = self._build_hist_mean_runs()

        obs = self._get_observation()
        if _GYM_API == "gymnasium":
            return obs, {}
        return obs

    def step(self, action):
        """Advance one half-inning.

        Args:
            action: np.ndarray shape (1,) — predicted runs this half-inning.

        Returns:
            (obs, reward, terminated, truncated, info) for gymnasium,
            (obs, reward, done, info) for gym.
        """
        if self._half_inning_idx >= len(self._half_innings):
            obs = self._get_observation()
            if _GYM_API == "gymnasium":
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        inning, half, team, actual_runs = self._half_innings[self._half_inning_idx]
        predicted_runs = float(np.clip(action, 0, 10)[0] if hasattr(action, '__len__') else action)

        baseline_pred = self._hist_mean_runs.get((inning, half), 0.5)
        baseline_err = abs(baseline_pred - actual_runs)
        agent_err = abs(predicted_runs - actual_runs)
        reward = baseline_err - agent_err  # positive when agent beats baseline

        # Update score
        if half == "top":
            self._away_score += actual_runs
        else:
            self._home_score += actual_runs

        self._half_inning_idx += 1
        terminated = self._half_inning_idx >= len(self._half_innings)
        obs = self._get_observation()

        info = {
            "game_id": self._game_id,
            "inning": inning,
            "half": half,
            "actual_runs": actual_runs,
            "predicted_runs": predicted_runs,
        }

        if _GYM_API == "gymnasium":
            return obs, reward, terminated, False, info
        return obs, reward, terminated, info

    def render(self):
        if self._half_innings and self._half_inning_idx < len(self._half_innings):
            inning, half, team, _ = self._half_innings[self._half_inning_idx]
            print(f"Game: {self._game_id} | Inning {inning} {half} | "
                  f"Score: Home {self._home_score} - Away {self._away_score}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # Pregame features
        n_pre = len(PREGAME_FEATURE_KEYS)
        obs[:n_pre] = self._pregame_vec[:n_pre]

        # Normalize features to roughly [-3, 3] range
        # year: 2019-2024 → 0-1
        year_idx = PREGAME_FEATURE_KEYS.index("year")
        obs[year_idx] = (obs[year_idx] - 2019.0) / 6.0
        # rs_10g / ra_10g are in total-runs-over-10-games units (~50-100); scale to ~0-1
        for key in ("home_rs_10g", "home_ra_10g", "away_rs_10g", "away_ra_10g"):
            if key in PREGAME_FEATURE_KEYS:
                idx = PREGAME_FEATURE_KEYS.index(key)
                obs[idx] = obs[idx] / 100.0
        # run_diff_10g is in similar range; scale
        for key in ("home_run_diff_10g", "away_run_diff_10g"):
            if key in PREGAME_FEATURE_KEYS:
                idx = PREGAME_FEATURE_KEYS.index(key)
                obs[idx] = obs[idx] / 50.0
        # age: typical range 20-40 → center and scale
        for key in ("hsp_age", "asp_age"):
            if key in PREGAME_FEATURE_KEYS:
                idx = PREGAME_FEATURE_KEYS.index(key)
                obs[idx] = (obs[idx] - 28.0) / 5.0

        # Final clip to [-5, 5]
        obs[:n_pre] = np.clip(obs[:n_pre], -5.0, 5.0)

        # Live features
        live_offset = n_pre
        if self._half_inning_idx < len(self._half_innings):
            inning, half, _, _ = self._half_innings[self._half_inning_idx]
        else:
            inning, half = self.max_innings, "bot"

        live_vals = [
            float(inning),
            0.0 if half == "top" else 1.0,
            float(self._home_score),
            float(self._away_score),
            float(self._home_score - self._away_score),
            self._hist_mean_runs.get((inning, half), 0.5),
        ]
        for i, v in enumerate(live_vals):
            if live_offset + i < OBS_DIM:
                obs[live_offset + i] = v

        return obs

    def _build_hist_mean_runs(self) -> dict[tuple, float]:
        """Compute mean runs per (inning, half) from the full inning dataset."""
        innings = self._get_inning_data()
        if innings.empty or "inning" not in innings.columns or "half" not in innings.columns:
            return {}
        grouped = innings.groupby(["inning", "half"])["runs"].mean()
        return {(int(inning), str(half)): float(mean) for (inning, half), mean in grouped.items()}

    def _get_pregame_vec(self, game_id: str) -> np.ndarray:
        if self._pregame_data is not None and not self._pregame_data.empty:
            row = self._pregame_data[self._pregame_data["game_id"] == game_id]
            if not row.empty:
                vals = []
                for k in PREGAME_FEATURE_KEYS:
                    v = row.iloc[0].get(k, 0.0)
                    # Replace NaN/inf with 0; clip to reasonable range
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        v = 0.0
                    if not np.isfinite(v):
                        v = 0.0
                    vals.append(v)
                return np.array(vals, dtype=np.float32)
        return np.zeros(len(PREGAME_FEATURE_KEYS), dtype=np.float32)

    def _get_inning_data(self) -> pd.DataFrame:
        if self._inning_data is not None:
            return self._inning_data
        from src.data.loader import load_innings
        return load_innings(self.year, data_root=self.data_root)

    def _parse_half_innings(self, game_innings: pd.DataFrame) -> list[tuple]:
        """Convert inning DataFrame to ordered list of (inning, half, team, runs)."""
        result = []
        for _, row in game_innings.sort_values(["inning", "half"]).iterrows():
            result.append((
                int(row["inning"]),
                str(row["half"]),
                str(row["team"]),
                int(row["runs"]),
            ))
        return result

    def _synthetic_episode(self) -> list[tuple]:
        """Generate a synthetic 9-inning game for testing without data."""
        half_innings = []
        for inning in range(1, 10):
            for half in ["top", "bot"]:
                runs = int(self.rng.poisson(0.55))  # ~KBO average runs/half-inning
                half_innings.append((inning, half, "TEAM", runs))
        return half_innings
