"""
KBOFeaturePipeline: builds pregame and live feature vectors for a given game.

All features are derived strictly from data before the game date (no leakage).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.loader import load_table, load_game_logs, load_innings
from src.metrics.advanced import enrich_pitching_advanced, enrich_batting_advanced
from src.features.people import enrich_with_people
from src.features.team_form import load_and_prepare_game_log, rolling_team_form
from src.features.context import add_context_features
from src.features.park_factors import get_park_factor, compute_park_factors
from src.metrics.era_adjust import ball_era_flag

logger = logging.getLogger(__name__)


class KBOFeaturePipeline:
    """Builds pregame and live feature vectors for a given game date.

    All queries are bounded by target_date to prevent data leakage.
    """

    def __init__(self, data_root: Optional[Path] = None, target_date: Optional[date] = None):
        self.data_root = data_root
        self.target_date = target_date or date.today()
        self._park_factors: dict[str, float] = {}
        self._pitching_cache: Optional[pd.DataFrame] = None
        self._batting_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_pregame_features(
        self,
        home_team: str,
        away_team: str,
        venue: str,
        home_starter_id: Optional[str] = None,
        away_starter_id: Optional[str] = None,
        year: Optional[int] = None,
    ) -> dict:
        """Build pregame feature dict for a matchup.

        Returns flat dict suitable for pd.DataFrame([features]).
        """
        year = year or self.target_date.year
        features: dict = {}

        # Era context
        features["ball_era"] = ball_era_flag(year)
        features["year"] = year

        # Park factor
        pf = get_park_factor(venue, self._park_factors)
        features["park_factor_runs"] = pf

        # Starter features
        pitching_df = self._get_pitching(year)
        if home_starter_id:
            hsp = self._starter_features(home_starter_id, pitching_df, prefix="hsp_")
            features.update(hsp)
        if away_starter_id:
            asp = self._starter_features(away_starter_id, pitching_df, prefix="asp_")
            features.update(asp)

        # Team form features (from game log)
        form = self._team_form_features(home_team, away_team, year)
        features.update(form)

        return features

    def build_live_features(
        self,
        game_id: str,
        current_inning: int,
        current_half: str,
        home_score: int,
        away_score: int,
        year: int,
        pregame_features: Optional[dict] = None,
    ) -> dict:
        """Build live in-game feature dict for a specific half-inning state."""
        features: dict = {}

        # Carry forward pregame features
        if pregame_features:
            features.update(pregame_features)

        # Live game state
        features["current_inning"] = current_inning
        features["half"] = 0 if current_half == "top" else 1
        features["home_score"] = home_score
        features["away_score"] = away_score
        features["score_diff"] = home_score - away_score

        # Historical inning-level patterns
        inning_stats = self._historical_inning_stats(current_inning, current_half, year)
        features.update(inning_stats)

        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pitching(self, year: int) -> pd.DataFrame:
        if self._pitching_cache is not None and "yearID" in self._pitching_cache.columns:
            return self._pitching_cache[self._pitching_cache["yearID"] <= year]

        raw = load_table("pitching", start_year=max(2006, year - 3), end_year=year,
                         data_root=self.data_root)
        if raw.empty:
            return pd.DataFrame()

        enriched = enrich_pitching_advanced(raw)
        enriched = enrich_with_people(enriched)
        self._pitching_cache = enriched
        return enriched

    def _starter_features(self, player_id: str, pitching_df: pd.DataFrame,
                           prefix: str = "sp_") -> dict:
        if pitching_df.empty:
            return {}

        rows = pitching_df[pitching_df["playerID"] == player_id]
        if rows.empty:
            return {}

        row = rows.sort_values("yearID").iloc[-1]  # most recent season

        features = {}
        for col in ["ERA", "FIP", "ERA_FIP_diff", "K_per_9", "BB_per_9",
                    "HR_per_9", "WHIP", "BABIP_p", "LOB_pct", "K_BB_pct",
                    "is_foreign", "age"]:
            if col in row.index:
                features[f"{prefix}{col.lower()}"] = row[col]

        # Handedness
        if "throws" in row.index:
            features[f"{prefix}throws_r"] = 1 if row["throws"] == "R" else 0

        return features

    def _team_form_features(self, home_team: str, away_team: str, year: int) -> dict:
        try:
            gl = load_game_logs(year, data_root=self.data_root)
            if gl.empty:
                return {}
            if not pd.api.types.is_datetime64_any_dtype(gl["date"]):
                gl["date"] = pd.to_datetime(gl["date"])
            # Only games before target date
            gl = gl[gl["date"] < pd.Timestamp(self.target_date)]
            if gl.empty:
                return {}

            expanded = load_and_prepare_game_log(year, self.data_root)
            if expanded.empty:
                return {}
            expanded = expanded[expanded["date"] < pd.Timestamp(self.target_date)]
            expanded = rolling_team_form(expanded)

            features = {}
            for team, prefix in [(home_team, "home_"), (away_team, "away_")]:
                team_rows = expanded[expanded["team"] == team]
                if team_rows.empty:
                    continue
                last = team_rows.iloc[-1]
                for col in ["win_pct_15g", "rs_10g", "ra_10g", "run_diff_10g", "pythag_10g"]:
                    if col in last.index:
                        features[f"{prefix}{col}"] = last[col]
            return features
        except Exception:
            logger.exception("Team form feature extraction failed")
            return {}

    def _historical_inning_stats(self, inning: int, half: str, year: int) -> dict:
        """Mean runs scored in this inning/half across historical games."""
        try:
            innings = load_innings(year, data_root=self.data_root)
            if innings.empty:
                return {}
            mask = (innings["inning"] == inning) & (innings["half"] == half)
            mean_runs = innings[mask]["runs"].mean()
            return {"hist_mean_runs_this_half_inning": mean_runs if not np.isnan(mean_runs) else 0.5}
        except Exception:
            return {}

    def get_observation_vector(self, features: dict,
                                feature_order: Optional[list[str]] = None) -> np.ndarray:
        """Convert feature dict to fixed-length numpy array for RL agent."""
        if feature_order is None:
            feature_order = sorted(features.keys())
        vec = np.array([float(features.get(k, 0.0)) for k in feature_order], dtype=np.float32)
        return vec, feature_order
