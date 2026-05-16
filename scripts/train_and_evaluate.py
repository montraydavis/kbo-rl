"""
Train KBO PPO agent using real season-aggregate data from data/kbo/<year>/.

What "real data" means here:
  - Episode run distributions are parameterised directly from actual team R/G
    (summed player runs ÷ team games played) from box-scores.csv and pitching.csv.
  - Pregame observation vectors are built from real pitcher and batter season stats
    (ERA, FIP, K%, BB%, wOBA, wRC+, is_foreign, age, etc.) for every synthetic game.
  - The only thing that remains synthetic is the inning-by-inning sequence within
    each game, because that data is not in nk-datasets (it requires scraping
    koreabaseball.com — see scripts/download_inning_data.py).

Run:
    python scripts/train_and_evaluate.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.loader import load_table, load_innings
from src.metrics.traditional import enrich_pitching, enrich_batting
from src.metrics.advanced import enrich_pitching_advanced, enrich_batting_advanced, compute_fip_constant
from src.features.people import enrich_with_people
from src.metrics.era_adjust import ball_era_flag
from src.rl.env import KBOInningEnv, OBS_DIM, PREGAME_FEATURE_KEYS
from src.rl.actor_critic import ActorCritic
from src.rl.ppo import PPO, PPOConfig, RolloutStorage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TRAIN_YEARS = list(range(2019, 2023))
VAL_YEAR    = 2023
TEST_YEAR   = 2024
NUM_ENVS    = 256
NUM_UPDATES = 300
DEVICE      = "cpu"
SEED        = 42


# ═══════════════════════════════════════════════════════════════════════════
# Real-data helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def team_runs_per_half_inning(year: int) -> dict[str, float]:
    """
    Returns {teamID: mean_runs_per_half_inning} for batting teams, computed
    directly from box-scores.csv:
        team_runs_per_game = sum(R) / max(G)   per team
        runs_per_half_inning = runs_per_game / 18
    """
    bat = load_table("box-scores", start_year=year, end_year=year)
    if bat.empty:
        return {}
    bat = _to_numeric_cols(bat, ["R", "G"])
    team_r   = bat.groupby("teamID")["R"].sum()
    team_g   = bat.groupby("teamID")["G"].max()
    rpg      = team_r / team_g           # runs per game (team totals)
    return (rpg / 18.0).to_dict()        # per half-inning


def team_ra_per_half_inning(year: int) -> dict[str, float]:
    """
    Returns {teamID: mean_runs_allowed_per_half_inning} from pitching.csv:
        team_ra_per_game = sum(R) / max(G)
        ra_per_half_inning = ra_per_game / 18
    """
    pit = load_table("pitching", start_year=year, end_year=year)
    if pit.empty:
        return {}
    pit = _to_numeric_cols(pit, ["R", "G"])
    team_r = pit.groupby("teamID")["R"].sum()
    team_g = pit.groupby("teamID")["G"].max()
    rpg    = team_r / team_g
    return (rpg / 18.0).to_dict()


def build_pitcher_features(year: int) -> pd.DataFrame:
    """
    Build per-pitcher pregame feature dict from real pitching.csv.
    Returns DataFrame indexed by playerID with columns matching hsp_/asp_ keys.
    """
    pit = load_table("pitching", start_year=year, end_year=year)
    if pit.empty:
        return pd.DataFrame()
    pit = _to_numeric_cols(pit, ["ERA", "IPouts", "H", "HR", "BB", "HBP", "SO", "R", "ER", "BFP", "G"])
    fip_c = compute_fip_constant(pit)
    pit = enrich_pitching_advanced(pit, fip_constant=fip_c)
    pit = enrich_with_people(pit)
    pit["IP"] = pit["IPouts"] / 3.0
    # Only keep qualified starters (≥20 IP) — use most recent season per player
    pit = pit[pit["IP"].fillna(0) >= 20].copy()
    return pit


def build_batter_features(year: int) -> pd.DataFrame:
    """
    Build per-team offensive feature dict from real box-scores.csv.
    Returns DataFrame with one row per teamID.
    """
    bat = load_table("box-scores", start_year=year, end_year=year)
    if bat.empty:
        return pd.DataFrame()
    bat = _to_numeric_cols(bat, ["AVG", "OBP", "SLG", "AB", "H", "2B", "3B", "HR",
                                  "BB", "SO", "PA", "TB", "R", "G"])
    bat = enrich_batting_advanced(bat)
    bat = enrich_with_people(bat)
    # Team-level aggregates (mean of qualified batters ≥100 PA)
    qualified = bat[bat["PA"].fillna(0) >= 100]
    team_feats = qualified.groupby("teamID").agg(
        team_woba=("wOBA", "mean"),
        team_wrc_plus=("wRC_plus", "mean"),
        team_ops=("OPS", "mean"),
        team_k_pct=("K_pct", "mean"),
        team_bb_pct=("BB_pct", "mean"),
        team_iso=("ISO", "mean"),
        foreign_hitter_count=("is_foreign", "sum"),
    ).reset_index()
    return team_feats


def select_ace(pit_df: pd.DataFrame, teamID: str) -> pd.Series | None:
    """Pick the 'ace' starter for a team: highest IP among starters."""
    team_pit = pit_df[pit_df["teamID"] == teamID]
    if team_pit.empty:
        return None
    return team_pit.sort_values("IP", ascending=False).iloc[0]


def make_pregame_row(game_id: str, home: str, away: str, year: int,
                      pit_df: pd.DataFrame, bat_feats: pd.DataFrame) -> dict:
    """
    Build a pregame feature dict for one game using real pitcher and team stats.
    Fills in the PREGAME_FEATURE_KEYS layout used by KBOInningEnv.
    """
    row: dict = {"game_id": game_id}
    row["ball_era"] = float(ball_era_flag(year))
    row["year"]     = float(year)
    row["park_factor_runs"] = 1.0  # neutral until park data available

    # Starter features
    for prefix, team in [("hsp_", home), ("asp_", away)]:
        ace = select_ace(pit_df, team)
        if ace is not None:
            row[f"{prefix}era"]          = float(ace.get("ERA", 4.0))
            row[f"{prefix}fip"]          = float(ace.get("FIP", 4.0))
            row[f"{prefix}era_fip_diff"] = float(ace.get("ERA_FIP_diff", 0.0))
            row[f"{prefix}k_per_9"]      = float(ace.get("K_per_9", 7.0))
            row[f"{prefix}bb_per_9"]     = float(ace.get("BB_per_9", 3.0))
            row[f"{prefix}hr_per_9"]     = float(ace.get("HR_per_9", 1.0))
            row[f"{prefix}whip"]         = float(ace.get("WHIP", 1.3))
            row[f"{prefix}babip_p"]      = float(ace.get("BABIP_p", 0.300))
            row[f"{prefix}lob_pct"]      = float(ace.get("LOB_pct", 0.72))
            row[f"{prefix}k_bb_pct"]     = float(ace.get("K_BB_pct", 0.12))
            row[f"{prefix}is_foreign"]   = float(ace.get("is_foreign", 0))
            row[f"{prefix}age"]          = float(ace.get("age", 28))
            row[f"{prefix}throws_r"]     = 1.0 if str(ace.get("throws", "R")) == "R" else 0.0
        else:
            for k in ["era","fip","era_fip_diff","k_per_9","bb_per_9","hr_per_9",
                      "whip","babip_p","lob_pct","k_bb_pct","is_foreign","age","throws_r"]:
                row[f"{prefix}{k}"] = 0.0

    # Team offensive features
    team_rs  = team_runs_per_half_inning(year)   # {teamID: runs_per_half_inning}
    team_ra  = team_ra_per_half_inning(year)

    for col_prefix, team in [("home_", home), ("away_", away)]:
        team_row = bat_feats[bat_feats["teamID"] == team]
        rs_phi = team_rs.get(team, 0.55)   # runs scored per half-inning
        ra_phi = team_ra.get(team, 0.55)   # runs allowed per half-inning
        rs_10g = rs_phi * 18 * 10          # scaled to 10-game window
        ra_10g = ra_phi * 18 * 10
        run_diff = rs_10g - ra_10g
        # Pythagorean win% estimate (exp=1.83 is standard)
        pythag = rs_phi**1.83 / (rs_phi**1.83 + ra_phi**1.83) if (rs_phi + ra_phi) > 0 else 0.5

        if not team_row.empty:
            t = team_row.iloc[0]
            row[f"{col_prefix}win_pct_15g"]  = float(pythag)
            row[f"{col_prefix}rs_10g"]       = float(rs_10g)
            row[f"{col_prefix}ra_10g"]       = float(ra_10g)
            row[f"{col_prefix}run_diff_10g"] = float(run_diff)
            row[f"{col_prefix}pythag_10g"]   = float(pythag)
        else:
            for k in ["win_pct_15g","rs_10g","ra_10g","run_diff_10g","pythag_10g"]:
                row[f"{col_prefix}{k}"] = 0.0

    return row


# ═══════════════════════════════════════════════════════════════════════════
# Episode generator — Poisson λ from real R/G
# ═══════════════════════════════════════════════════════════════════════════

def build_real_innings(years: list[int], **_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (inning_df, pregame_df) using only real scraped inning data.

    Raises RuntimeError for any year that has no scraped data in
    data/kbo_innings/<year>/. No synthetic or Poisson-simulated data is used.
    """
    # Korean team name → English (matches nk-datasets pitching/batting CSVs)
    KO_TO_EN: dict[str, str] = {
        "롯데": "Lotte", "두산": "Doosan", "한화": "Hanwha", "삼성": "Samsung",
        "LG": "LG", "KT": "KT", "KIA": "KIA", "NC": "NC", "SSG": "SSG",
        "키움": "Kiwoom", "sk": "SSG", "SK": "SSG",
    }

    def _en(name: str) -> str:
        return KO_TO_EN.get(name, name)
    inning_rows: list[pd.DataFrame] = []
    pregame_rows: list[dict] = []

    for year in years:
        log.info("  Loading real scraped innings for %d …", year)
        real_innings = load_innings(year)
        if real_innings.empty:
            raise RuntimeError(
                f"No scraped inning data found for {year}. "
                f"Run: python scripts/download_inning_data.py --year {year}"
            )

        pit_df    = build_pitcher_features(year)
        bat_feats = build_batter_features(year)
        off_rates = team_runs_per_half_inning(year)
        def_rates = team_ra_per_half_inning(year)
        teams = sorted(set(list(off_rates.keys())) | set(list(def_rates.keys())))

        log.info("  %d games loaded for %d", real_innings["game_id"].nunique(), year)
        inning_rows.append(real_innings)

        for gid in real_innings["game_id"].unique():
            game_innings = real_innings[real_innings["game_id"] == gid]
            top_teams = game_innings[game_innings["half"] == "top"]["team"].unique()
            bot_teams = game_innings[game_innings["half"] == "bot"]["team"].unique()
            away = _en(str(top_teams[0])) if len(top_teams) else (teams[0] if teams else "UNK")
            home = _en(str(bot_teams[0])) if len(bot_teams) else (teams[1] if len(teams) > 1 else "UNK")
            pg = make_pregame_row(gid, home, away, year, pit_df, bat_feats)
            pregame_rows.append(pg)

    inning_df  = pd.concat(
        [r if isinstance(r, pd.DataFrame) else pd.DataFrame([r]) for r in inning_rows],
        ignore_index=True,
    ) if inning_rows else pd.DataFrame()
    pregame_df = pd.DataFrame(pregame_rows)
    return inning_df, pregame_df


# ═══════════════════════════════════════════════════════════════════════════
# Vectorised env wrapper
# ═══════════════════════════════════════════════════════════════════════════

class VecEnv:
    def __init__(self, num_envs: int, inning_data: pd.DataFrame,
                 pregame_data: pd.DataFrame, year: int, seed: int = 0):
        self.envs = [
            KBOInningEnv(inning_data=inning_data, pregame_data=pregame_data,
                         year=year, seed=seed + i)
            for i in range(num_envs)
        ]
        self.num_envs = num_envs

    def reset(self) -> np.ndarray:
        obs = []
        for env in self.envs:
            r = env.reset()
            obs.append(r[0] if isinstance(r, tuple) else r)
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        obs_list, rews, dones = [], [], []
        for env, act in zip(self.envs, actions):
            r = env.step(act)
            o, rew, done = (r[0], r[1], r[2] or r[3]) if len(r) == 5 else (r[0], r[1], r[2])
            obs_list.append(o)
            rews.append(rew)
            dones.append(float(done))
            if done:
                reset_r = env.reset()
                obs_list[-1] = reset_r[0] if isinstance(reset_r, tuple) else reset_r
        return np.stack(obs_list), np.array(rews, np.float32), np.array(dones, np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def pretrain_supervised(
    ac: ActorCritic,
    inning_data: pd.DataFrame,
    pregame_data: pd.DataFrame,
    year: int,
    n_epochs: int = 50,
    lr: float = 1e-3,
) -> ActorCritic:
    """Supervised regression warm-up: train actor to predict actual runs per half-inning."""
    import torch.optim as optim
    env = KBOInningEnv(inning_data=inning_data, pregame_data=pregame_data, year=year)
    obs_list, target_list = [], []

    # Collect (obs, actual_runs) pairs — obs is captured BEFORE stepping
    for gid in inning_data["game_id"].unique():
        obs = env.reset(game_id=gid)
        obs = obs[0] if isinstance(obs, tuple) else obs
        for _, _, _, actual_runs in env._half_innings:
            obs_list.append(obs.copy())
            target_list.append(float(actual_runs))
            result = env.step(np.array([actual_runs]))
            obs = result[0]
            obs = obs[0] if isinstance(obs, tuple) else obs

    if not obs_list:
        return ac

    obs_t = torch.as_tensor(np.stack(obs_list), dtype=torch.float32)
    tgt_t = torch.as_tensor(target_list, dtype=torch.float32).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(obs_t, tgt_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    # Phase 1: train only the output layer (fast convergence)
    out_layer = list(ac.actor.modules())[-1]
    for p in ac.actor.parameters():
        p.requires_grad_(False)
    out_layer.weight.requires_grad_(True)
    out_layer.bias.requires_grad_(True)
    optimizer = optim.Adam([out_layer.weight, out_layer.bias], lr=lr)

    ac.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for obs_b, tgt_b in loader:
            pred = nn.functional.softplus(ac.actor(obs_b))
            loss = nn.functional.huber_loss(pred, tgt_b, delta=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Keep bias in sane range
            with torch.no_grad():
                out_layer.bias.clamp_(-2.0, 4.0)
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(loader)
            with torch.no_grad():
                sample_preds = nn.functional.softplus(ac.actor(obs_t[:8])).squeeze().tolist()
            log.info("  supervised epoch %d/%d  loss=%.4f  sample_preds=%s",
                     epoch + 1, n_epochs, avg, [f'{p:.2f}' for p in sample_preds])

    # Phase 2: fine-tune all layers with lower LR
    for p in ac.actor.parameters():
        p.requires_grad_(True)
    optimizer2 = optim.Adam(ac.actor.parameters(), lr=lr * 0.1)
    for epoch in range(20):
        for obs_b, tgt_b in loader:
            pred = nn.functional.softplus(ac.actor(obs_b))
            loss = nn.functional.huber_loss(pred, tgt_b, delta=1.0)
            optimizer2.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), 0.5)
            optimizer2.step()
            with torch.no_grad():
                out_layer.bias.clamp_(-2.0, 4.0)

    with torch.no_grad():
        sample_preds = nn.functional.softplus(ac.actor(obs_t[:8])).squeeze().tolist()
    log.info("  supervised done. Final sample_preds=%s", [f'{p:.2f}' for p in sample_preds])
    ac.eval()
    return ac


def train(inning_data: pd.DataFrame, pregame_data: pd.DataFrame, year: int) -> ActorCritic:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    vec = VecEnv(NUM_ENVS, inning_data, pregame_data, year, seed=SEED)
    cfg = PPOConfig(num_steps_per_env=22, learning_rate=1e-4, num_epochs=5,
                   num_mini_batches=4, gamma=0.99, lam=0.95,
                   clip_param=0.2, entropy_coef=0.05)
    ac      = ActorCritic(obs_dim=OBS_DIM, actor_hidden=[256,128,64],
                          critic_hidden=[256,128,64]).to(DEVICE)

    # Supervised pre-training — save these weights as the inference model
    log.info("Supervised pre-training …")
    ac = pretrain_supervised(ac, inning_data, pregame_data, year, n_epochs=50)

    # Save supervised-only checkpoint for inference
    sup_ckpt = REPO / "checkpoints" / "checkpoint_supervised.pt"
    torch.save({"update": 0, "model_state": ac.state_dict()}, sup_ckpt)
    log.info("Supervised checkpoint saved → %s", sup_ckpt)

    ppo     = PPO(ac, cfg, DEVICE)
    storage = RolloutStorage(cfg.num_steps_per_env, NUM_ENVS, OBS_DIM, DEVICE)

    obs = vec.reset()
    t0  = time.time()
    log.info("Training %d updates × %d envs × %d steps/env …",
             NUM_UPDATES, NUM_ENVS, cfg.num_steps_per_env)

    for upd in range(1, NUM_UPDATES + 1):
        for _ in range(cfg.num_steps_per_env):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, log_prob = ac.act(obs_t)
                value = ac.get_value(obs_t)
            nobs, rews, dones = vec.step(action.numpy())
            storage.insert(obs, action.numpy(), rews, value, log_prob, dones)
            obs = nobs

        with torch.no_grad():
            last_val = ac.get_value(torch.as_tensor(obs, dtype=torch.float32))
        storage.compute_returns(last_val, cfg.gamma, cfg.lam)
        stats = ppo.update(storage)

        if upd % 50 == 0 or upd == 1:
            log.info("  update %3d/%d  π=%.4f  v=%.4f  H=%.4f  %.0fs",
                     upd, NUM_UPDATES, stats["policy_loss"],
                     stats["value_loss"], stats["entropy"], time.time() - t0)

    log.info("Training done in %.1fs", time.time() - t0)
    return ac


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(ac: ActorCritic, inning_data: pd.DataFrame,
             pregame_data: pd.DataFrame, year: int, label: str) -> dict:
    ac.eval()
    game_ids = inning_data["game_id"].unique()

    all_actual, all_pred = [], []
    inning_buckets: dict[int, tuple[list, list]] = {i: ([], []) for i in range(1, 13)}

    for gid in game_ids:
        env = KBOInningEnv(inning_data=inning_data, pregame_data=pregame_data,
                           year=year, seed=0)
        r   = env.reset(game_id=gid)
        obs = torch.as_tensor(r[0] if isinstance(r, tuple) else r,
                               dtype=torch.float32).unsqueeze(0)

        for _ in range(30):
            with torch.no_grad():
                pred = float(np.clip(ac(obs).squeeze(0).numpy()[0], 0, 10))
            r = env.step(np.array([pred]))
            obs_np, reward, done, info = (
                (r[0], r[1], r[2] or r[3], r[4]) if len(r) == 5
                else (r[0], r[1], r[2], r[3])
            )
            actual = info["actual_runs"]
            inn    = info["inning"]
            all_actual.append(actual)
            all_pred.append(pred)
            inning_buckets[inn][0].append(actual)
            inning_buckets[inn][1].append(pred)
            obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            if done:
                break

    actual_arr = np.array(all_actual)
    pred_arr   = np.array(all_pred)
    errors     = np.abs(actual_arr - pred_arr)
    lg_mean    = float(actual_arr.mean())

    per_inning = {
        inn: round(float(np.abs(np.array(a) - np.array(p)).mean()), 4)
        for inn, (a, p) in inning_buckets.items()
        if a
    }

    return {
        "label":                    label,
        "n_games":                  len(game_ids),
        "n_half_innings":           len(actual_arr),
        "league_mean":              round(lg_mean, 4),
        "league_mean_baseline_mae": round(float(np.abs(actual_arr - lg_mean).mean()), 4),
        "mae":                      round(float(errors.mean()), 4),
        "rmse":                     round(float(np.sqrt((errors**2).mean())), 4),
        "within_1_run_pct":         round(float((errors <= 1.0).mean() * 100), 1),
        "exact_pct":                round(float((errors < 0.5).mean() * 100), 1),
        "mae_vs_baseline":          round(float(errors.mean()) -
                                          float(np.abs(actual_arr - lg_mean).mean()), 4),
        "per_inning_mae":           per_inning,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _print_result(label: str, pre: dict, post: dict) -> None:
    print(f"\n  {label}")
    print(f"    Games evaluated          : {post['n_games']}")
    print(f"    Half-innings             : {post['n_half_innings']}")
    print(f"    Actual league mean runs  : {post['league_mean']:.4f} / half-inning")
    print(f"    League-mean baseline MAE : {post['league_mean_baseline_mae']:.4f}")
    print(f"    Pre-train  MAE           : {pre['mae']:.4f}")
    print(f"    Post-train MAE           : {post['mae']:.4f}  (RMSE {post['rmse']:.4f})")
    delta = pre["mae"] - post["mae"]
    print(f"    MAE improvement          : {delta:+.4f}  ({'better' if delta > 0 else 'worse'})")
    print(f"    Within ±1 run            : {post['within_1_run_pct']:.1f}%")
    print(f"    'Exact'  (±0.5 run)      : {post['exact_pct']:.1f}%")
    print(f"    vs baseline              : {post['mae_vs_baseline']:+.4f}")
    print(f"    Per-inning MAE           :", end="")
    for inn in range(1, 10):
        if inn in post["per_inning_mae"]:
            print(f"  inn{inn}={post['per_inning_mae'][inn]:.3f}", end="")
    print()


def main():
    log.info("=== Building episode data from real KBO season CSVs ===")
    log.info("  data source : data/kbo/<year>/box-scores.csv + pitching.csv + people.csv")
    log.info("  λ source    : actual team R/G from R column ÷ max(G)")
    log.info("  obs source  : real ERA/FIP/K%%/BB%%/wOBA/age/is_foreign per player")

    log.info("Loading train years %s …", TRAIN_YEARS)
    train_inn, train_pg = build_real_innings(TRAIN_YEARS, games_per_team_year=72, seed=SEED)

    log.info("Loading val year %d …", VAL_YEAR)
    val_inn, val_pg     = build_real_innings([VAL_YEAR], games_per_team_year=72, seed=SEED+1)

    log.info("Loading test year %d …", TEST_YEAR)
    test_inn, test_pg   = build_real_innings([TEST_YEAR], games_per_team_year=72, seed=SEED+2)

    log.info("Train half-innings: %d  val: %d  test: %d",
             len(train_inn), len(val_inn), len(test_inn))
    log.info("Train pregame rows: %d  val: %d  test: %d",
             len(train_pg), len(val_pg), len(test_pg))

    # Baseline (untrained)
    log.info("=== Baseline evaluation (random-init policy) ===")
    ac0      = ActorCritic(obs_dim=OBS_DIM)
    pre_val  = evaluate(ac0, val_inn,  val_pg,  VAL_YEAR,  "pre-train val")
    pre_test = evaluate(ac0, test_inn, test_pg, TEST_YEAR, "pre-train test")

    # Train
    log.info("=== Training ===")
    ac = train(train_inn, train_pg, year=TRAIN_YEARS[-1])

    # Evaluate
    log.info("=== Post-training evaluation ===")
    post_val  = evaluate(ac, val_inn,  val_pg,  VAL_YEAR,  "post-train val")
    post_test = evaluate(ac, test_inn, test_pg, TEST_YEAR, "post-train test")

    print("\n" + "="*66)
    print("  KBO Inning Prediction — Accuracy Report")
    print("  (episodes parameterised from real team R/G; obs = real player stats)")
    print("="*66)
    _print_result("Validation (2023)", pre_val,  post_val)
    _print_result("Test       (2024)", pre_test, post_test)
    print("="*66)

    out = REPO / "checkpoints" / "accuracy_report.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"pre_val": pre_val, "post_val": post_val,
                   "pre_test": pre_test, "post_test": post_test}, f, indent=2)
    log.info("Results saved → %s", out)

    ckpt_path = REPO / "checkpoints" / f"checkpoint_{NUM_UPDATES:06d}.pt"
    torch.save({"update": NUM_UPDATES, "model_state": ac.state_dict()}, ckpt_path)
    log.info("Checkpoint saved → %s", ckpt_path)


if __name__ == "__main__":
    main()
