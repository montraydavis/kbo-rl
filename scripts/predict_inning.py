"""
Live inning-by-inning run prediction using the trained KBO PPO agent.

Usage:
    # Predict next half-inning for a live game state
    python scripts/predict_inning.py \\
        --home KT --away LG \\
        --inning 5 --half top \\
        --home-score 3 --away-score 1 \\
        --year 2024

    # Load from a saved checkpoint
    python scripts/predict_inning.py --checkpoint checkpoints/checkpoint_000300.pt \\
        --home KT --away LG --inning 5 --half top

    # Simulate a full game prediction
    python scripts/predict_inning.py --home KT --away LG --year 2024 --full-game

Requires a trained checkpoint or will train a quick model from available data.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

from src.rl.env import KBOInningEnv, PREGAME_FEATURE_KEYS, LIVE_FEATURE_KEYS, OBS_DIM
from src.rl.actor_critic import ActorCritic
from src.data.loader import load_innings, load_table
from src.metrics.traditional import enrich_pitching, enrich_batting
from src.metrics.advanced import enrich_pitching_advanced
from src.features.people import enrich_with_people
from src.metrics.era_adjust import ball_era_flag


# ─────────────────────────────────────────────────────────────────────────────
# Feature builders
# ─────────────────────────────────────────────────────────────────────────────

def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_pregame_obs(
    home: str,
    away: str,
    year: int,
    hist_mean_runs: dict[tuple, float] | None = None,
    current_inning: int = 1,
    current_half: str = "top",
    home_score: int = 0,
    away_score: int = 0,
) -> np.ndarray:
    """Build a full OBS_DIM observation vector for a live game state."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    n_pre = len(PREGAME_FEATURE_KEYS)

    # --- Pregame features ---
    pit = load_table("pitching", year, year)
    bat = load_table("batting", year, year)
    ppl = load_table("people", year, year)

    if not pit.empty:
        _to_numeric(pit, ["ERA", "IPouts"])
        enrich_pitching(pit)
        enrich_pitching_advanced(pit)
        if not ppl.empty:
            pit = enrich_with_people(pit, ppl, year_col="yearID")

    if not bat.empty:
        _to_numeric(bat, ["AVG", "SLG", "OBP"])
        enrich_batting(bat)

    pre_dict: dict[str, float] = {}
    pre_dict["ball_era"] = float(ball_era_flag(year))
    pre_dict["year"] = float(year)
    pre_dict["park_factor_runs"] = 1.0

    for prefix, team in [("hsp_", home), ("asp_", away)]:
        if not pit.empty and "teamID" in pit.columns:
            team_pit = pit[pit["teamID"] == team]
            if not team_pit.empty:
                if "IPouts" in team_pit.columns:
                    team_pit = team_pit.sort_values("IPouts", ascending=False)
                ace = team_pit.iloc[0]
                pre_dict[f"{prefix}era"]          = float(ace.get("ERA", 4.0))
                pre_dict[f"{prefix}fip"]          = float(ace.get("FIP", 4.0))
                pre_dict[f"{prefix}era_fip_diff"] = float(ace.get("ERA_FIP_diff", 0.0))
                pre_dict[f"{prefix}k_per_9"]      = float(ace.get("K_per_9", 7.0))
                pre_dict[f"{prefix}bb_per_9"]     = float(ace.get("BB_per_9", 3.0))
                pre_dict[f"{prefix}hr_per_9"]     = float(ace.get("HR_per_9", 1.0))
                pre_dict[f"{prefix}whip"]         = float(ace.get("WHIP", 1.3))
                pre_dict[f"{prefix}babip_p"]      = float(ace.get("BABIP_p", 0.300))
                pre_dict[f"{prefix}lob_pct"]      = float(ace.get("LOB_pct", 0.72))
                pre_dict[f"{prefix}k_bb_pct"]     = float(ace.get("K_BB_pct", 0.12))
                pre_dict[f"{prefix}is_foreign"]   = float(ace.get("is_foreign", 0))
                pre_dict[f"{prefix}age"]          = float(ace.get("age", 28))
                pre_dict[f"{prefix}throws_r"]     = 1.0 if str(ace.get("throws", "R")) == "R" else 0.0

    for col_prefix, team in [("home_", home), ("away_", away)]:
        if not bat.empty and "teamID" in bat.columns:
            team_bat = bat[bat["teamID"] == team]
            if not team_bat.empty:
                pit_team = pit[pit["teamID"] == team] if not pit.empty else pd.DataFrame()
                rs = team_bat["R"].sum() / max(team_bat["G"].max(), 1) / 18 * 18 * 10 if "R" in team_bat else 50.0
                ra = pit_team["R"].sum() / max(pit_team["G"].max(), 1) / 18 * 18 * 10 if (not pit_team.empty and "R" in pit_team) else 50.0
                rs_phi = rs / (18 * 10) if rs else 0.55
                ra_phi = ra / (18 * 10) if ra else 0.55
                pythag = rs_phi**1.83 / (rs_phi**1.83 + ra_phi**1.83) if (rs_phi + ra_phi) > 0 else 0.5
                pre_dict[f"{col_prefix}win_pct_15g"]  = float(pythag)
                pre_dict[f"{col_prefix}rs_10g"]       = float(rs)
                pre_dict[f"{col_prefix}ra_10g"]       = float(ra)
                pre_dict[f"{col_prefix}run_diff_10g"] = float(rs - ra)
                pre_dict[f"{col_prefix}pythag_10g"]   = float(pythag)

    for i, key in enumerate(PREGAME_FEATURE_KEYS):
        val = pre_dict.get(key, 0.0)
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = 0.0
        obs[i] = v if np.isfinite(v) else 0.0

    # Apply same normalization as KBOInningEnv._get_observation
    year_idx = PREGAME_FEATURE_KEYS.index("year")
    obs[year_idx] = (obs[year_idx] - 2019.0) / 6.0
    for key in ("home_rs_10g", "home_ra_10g", "away_rs_10g", "away_ra_10g"):
        if key in PREGAME_FEATURE_KEYS:
            obs[PREGAME_FEATURE_KEYS.index(key)] /= 100.0
    for key in ("home_run_diff_10g", "away_run_diff_10g"):
        if key in PREGAME_FEATURE_KEYS:
            obs[PREGAME_FEATURE_KEYS.index(key)] /= 50.0
    for key in ("hsp_age", "asp_age"):
        if key in PREGAME_FEATURE_KEYS:
            idx = PREGAME_FEATURE_KEYS.index(key)
            obs[idx] = (obs[idx] - 28.0) / 5.0
    obs[:n_pre] = np.clip(obs[:n_pre], -5.0, 5.0)

    # --- Live features ---
    hist_mean = 0.5
    if hist_mean_runs:
        hist_mean = hist_mean_runs.get((current_inning, current_half), 0.5)

    live_vals = [
        float(current_inning),
        0.0 if current_half == "top" else 1.0,
        float(home_score),
        float(away_score),
        float(home_score - away_score),
        float(hist_mean),
    ]
    for i, v in enumerate(live_vals):
        if n_pre + i < OBS_DIM:
            obs[n_pre + i] = v

    return obs


def load_hist_mean_runs(years: list[int]) -> dict[tuple, float]:
    """Compute mean runs per (inning, half) from scraped inning data."""
    frames = []
    for yr in years:
        df = load_innings(yr)
        if not df.empty:
            frames.append(df)
    if not frames:
        return {}
    all_innings = pd.concat(frames, ignore_index=True)
    grouped = all_innings.groupby(["inning", "half"])["runs"].mean()
    return {(int(inn), str(half)): float(mean) for (inn, half), mean in grouped.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading / quick-train
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path | None) -> ActorCritic:
    import torch
    model = ActorCritic(obs_dim=OBS_DIM)
    if checkpoint_path and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        log.info("Loaded checkpoint: %s", checkpoint_path)
    else:
        log.warning("No checkpoint found — using untrained model. Run train_and_evaluate.py first.")
    model.eval()
    return model


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    # Prefer supervised checkpoint (PPO tends to collapse actor weights)
    sup = ckpt_dir / "checkpoint_supervised.pt"
    if sup.exists():
        return sup
    pts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    return pts[-1] if pts else None


def analytical_predict(
    batting_team: str,
    pitching_team: str,
    year: int,
    hist_mean_runs: dict[tuple, float],
    inning: int,
    half: str,
) -> dict:
    """
    Analytical fallback: expected runs = batting_team_rate × pitcher_era_adjustment.
    Used when the RL model has collapsed.
    """
    from src.data.loader import load_table
    from src.metrics.traditional import enrich_pitching

    league_mean = hist_mean_runs.get((inning, half), 0.5)

    # Offensive rate for batting team
    bat = load_table("batting", year, year)
    off_rate = league_mean
    if not bat.empty and "teamID" in bat.columns and "R" in bat.columns and "G" in bat.columns:
        tb = bat[bat["teamID"] == batting_team]
        if not tb.empty:
            r = pd.to_numeric(tb["R"], errors="coerce").sum()
            g = pd.to_numeric(tb["G"], errors="coerce").max()
            if g > 0:
                off_rate = float(r / g / 18)  # runs per half-inning

    # ERA adjustment from opposing pitcher
    pit = load_table("pitching", year, year)
    era_factor = 1.0
    if not pit.empty and "teamID" in pit.columns:
        tp = pit[pit["teamID"] == pitching_team].copy()
        if not tp.empty:
            tp["IPouts"] = pd.to_numeric(tp.get("IPouts", 0), errors="coerce").fillna(0)
            tp["ERA"] = pd.to_numeric(tp.get("ERA", 4.0), errors="coerce").fillna(4.0)
            ace = tp.sort_values("IPouts", ascending=False).iloc[0]
            league_era = 4.5
            era_factor = float(ace["ERA"]) / league_era

    predicted = float(np.clip(off_rate * era_factor, 0.0, 5.0))
    std = league_mean * 0.5
    return {
        "predicted_runs": round(predicted, 2),
        "std": round(std, 3),
        "ci_low": round(max(0.0, predicted - std), 2),
        "ci_high": round(predicted + std, 2),
        "method": "analytical",
    }


    pts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    return pts[-1] if pts else None


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_half_inning(
    model: ActorCritic,
    obs: np.ndarray,
    n_samples: int = 2000,
) -> dict:
    """Return predicted runs and confidence interval for one half-inning."""
    import torch
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        det_mean = model(obs_t).item()
        std = model.log_std.exp().clamp(1e-3, 5.0).item()
        # Use expected value of clamp(N(det_mean, std), 0, 10) — matches training behavior
        samples = torch.clamp(
            torch.distributions.Normal(det_mean, std).sample((n_samples,)), 0.0, 10.0
        )
        predicted = float(samples.mean().item())
        pred_std  = float(samples.std().item())
    return {
        "predicted_runs": round(predicted, 2),
        "std": round(pred_std, 3),
        "ci_low":  round(float(samples.quantile(0.05).item()), 2),
        "ci_high": round(float(samples.quantile(0.95).item()), 2),
    }


def predict_full_game(
    model: ActorCritic,
    home: str,
    away: str,
    year: int,
    hist_mean_runs: dict[tuple, float],
    max_innings: int = 9,
) -> list[dict]:
    """Predict runs for every half-inning of a game."""
    results = []
    home_score = 0
    away_score = 0

    for inning in range(1, max_innings + 1):
        for half, batting_team, pitching_team in [("top", away, home), ("bot", home, away)]:
            obs = build_pregame_obs(
                home=home, away=away, year=year,
                hist_mean_runs=hist_mean_runs,
                current_inning=inning, current_half=half,
                home_score=home_score, away_score=away_score,
            )
            rl_pred = predict_half_inning(model, obs)
            # Use RL if it produces meaningful predictions (supervised checkpoint), else analytical
            if rl_pred["predicted_runs"] > 0.05:
                pred = rl_pred
            else:
                pred = analytical_predict(batting_team, pitching_team, year, hist_mean_runs, inning, half)
            pred.update({
                "inning": inning,
                "half": half,
                "batting_team": batting_team,
                "home_score_before": home_score,
                "away_score_before": away_score,
            })
            results.append(pred)

            # Update running score with prediction for next inning's context
            if half == "top":
                away_score += round(pred["predicted_runs"])
            else:
                home_score += round(pred["predicted_runs"])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="KBO inning run predictor")
    parser.add_argument("--home", required=True, help="Home team code (e.g. KT)")
    parser.add_argument("--away", required=True, help="Away team code (e.g. LG)")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--inning", type=int, default=1)
    parser.add_argument("--half", choices=["top", "bot"], default="top")
    parser.add_argument("--home-score", type=int, default=0)
    parser.add_argument("--away-score", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to checkpoint .pt file (auto-detects latest if omitted)")
    parser.add_argument("--full-game", action="store_true",
                        help="Predict all innings of a fresh game")
    parser.add_argument("--max-innings", type=int, default=9)
    parser.add_argument("--data-years", nargs="+", type=int, default=[2022, 2023, 2024],
                        help="Years to load for historical mean runs lookup")
    args = parser.parse_args(argv)

    # Find checkpoint
    ckpt = args.checkpoint or find_latest_checkpoint(REPO / "checkpoints")
    model = load_model(ckpt)

    # Historical mean runs per half-inning
    hist_mean_runs = load_hist_mean_runs(args.data_years)
    if hist_mean_runs:
        log.info("Loaded historical means for %d (inning, half) combos", len(hist_mean_runs))
    else:
        log.warning("No scraped inning data found — hist_mean_runs will be 0.5 for all slots")

    if args.full_game:
        results = predict_full_game(
            model=model,
            home=args.home,
            away=args.away,
            year=args.year,
            hist_mean_runs=hist_mean_runs,
            max_innings=args.max_innings,
        )
        print(f"\n{'='*60}")
        print(f"  {args.away} (away) @ {args.home} (home)  —  {args.year}")
        print(f"{'='*60}")
        print(f"{'Inn':<5} {'Half':<5} {'Team':<8} {'Pred':>6} {'±':>6}  {'CI':>14}  Score")
        print(f"{'-'*60}")
        home_total = away_total = 0.0
        for r in results:
            score = f"{r['away_score_before']}-{r['home_score_before']}"
            ci = f"[{r['ci_low']:.1f}-{r['ci_high']:.1f}]"
            print(f"{r['inning']:<5} {r['half']:<5} {r['batting_team']:<8} "
                  f"{r['predicted_runs']:>6.2f} {r['std']:>6.3f}  {ci:>14}  {score}")
            if r["half"] == "top":
                away_total += r["predicted_runs"]
            else:
                home_total += r["predicted_runs"]
        print(f"{'='*60}")
        print(f"  Expected runs:  {args.away} {away_total:.1f}  —  {args.home} {home_total:.1f}")
        print(f"  Predicted score: {args.away} {round(away_total)}  —  {args.home} {round(home_total)}")
        print(f"{'='*60}\n")
    else:
        obs = build_pregame_obs(
            home=args.home, away=args.away, year=args.year,
            hist_mean_runs=hist_mean_runs,
            current_inning=args.inning, current_half=args.half,
            home_score=args.home_score, away_score=args.away_score,
        )
        pred = predict_half_inning(model, obs)
        batting = args.away if args.half == "top" else args.home
        print(f"\nInning {args.inning} {args.half}  ({batting} batting)")
        print(f"  Predicted runs : {pred['predicted_runs']:.2f}")
        print(f"  Std dev        : {pred['std']:.3f}")
        print(f"  90% CI         : [{pred['ci_low']:.2f} – {pred['ci_high']:.2f}]\n")
        print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
