"""
Training orchestrator for KBO PPO agent.

Collects rollouts from vectorized KBOInningEnv instances, runs PPO updates,
and saves checkpoints every N updates.

Usage:
    python -m src.rl.runner --train-years 2019 2022 --val-year 2023 --num-envs 500
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _try_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class VecKBOEnv:
    """Minimal vectorized wrapper for KBOInningEnv.

    Runs N independent environments synchronously (no multiprocessing).
    For large num_envs, replace with multiprocessing or async workers.
    """

    def __init__(self, num_envs: int, year: int, inning_data=None,
                 pregame_data=None, data_root=None, seed=None):
        from src.rl.env import KBOInningEnv
        self.envs = [
            KBOInningEnv(
                inning_data=inning_data,
                pregame_data=pregame_data,
                year=year,
                data_root=data_root,
                seed=None if seed is None else seed + i,
            )
            for i in range(num_envs)
        ]
        self.num_envs = num_envs

    def reset(self):
        obs_list = []
        for env in self.envs:
            result = env.reset()
            obs = result[0] if isinstance(result, tuple) else result
            obs_list.append(obs)
        return np.stack(obs_list)

    def step(self, actions):
        obs_list, rewards, dones = [], [], []
        for env, action in zip(self.envs, actions):
            result = env.step(action)
            if len(result) == 5:
                obs, r, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                obs, r, done, _ = result
            obs_list.append(obs)
            rewards.append(r)
            dones.append(float(done))

        # Auto-reset on done
        for i, (env, done) in enumerate(zip(self.envs, dones)):
            if done:
                result = env.reset()
                obs_list[i] = result[0] if isinstance(result, tuple) else result

        return np.stack(obs_list), np.array(rewards), np.array(dones)


def collect_rollout(vec_env, actor_critic, storage, torch, device):
    """Fill storage with one rollout from the vectorized env."""
    obs = vec_env.reset()

    for step in range(storage.num_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, log_prob = actor_critic.act(obs_t)
            value = actor_critic.get_value(obs_t)

        actions_np = action.cpu().numpy()
        next_obs, rewards, dones = vec_env.step(actions_np)

        storage.insert(obs, actions_np, rewards, value, log_prob, dones)
        obs = next_obs

    # Bootstrap value for last obs
    with torch.no_grad():
        last_value = actor_critic.get_value(
            torch.as_tensor(obs, dtype=torch.float32, device=device)
        )
    return last_value


def evaluate(vec_env, actor_critic, torch, device, n_episodes: int = 50) -> dict:
    """Run deterministic evaluation and return mean reward per episode."""
    episode_rewards = []
    obs = vec_env.reset()
    ep_rew = np.zeros(vec_env.num_envs)
    steps_done = 0

    while len(episode_rewards) < n_episodes:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = actor_critic(obs_t)
        action_np = action.cpu().numpy().clip(0, 10)
        obs, rewards, dones = vec_env.step(action_np)
        ep_rew += rewards
        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(ep_rew[i])
                ep_rew[i] = 0.0
        steps_done += 1
        if steps_done > 2000:
            break

    mean_rew = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    return {"mean_episode_reward": mean_rew, "n_episodes": len(episode_rewards)}


def train(
    train_years: list[int],
    val_year: int,
    num_envs: int = 500,
    num_updates: int = 1000,
    checkpoint_every: int = 100,
    output_dir: Path = Path("checkpoints"),
    data_root: Optional[Path] = None,
    device: str = "cpu",
    seed: int = 42,
):
    torch = _try_torch()
    if torch is None:
        raise ImportError("torch is required. pip install torch")

    from src.rl.actor_critic import ActorCritic
    from src.rl.ppo import PPO, PPOConfig, RolloutStorage
    from src.rl.env import OBS_DIM
    from src.data.loader import load_innings, load_table
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load inning data for training years
    train_innings = pd.concat(
        [load_innings(y, data_root=data_root) for y in train_years],
        ignore_index=True,
    ) if train_years else pd.DataFrame()

    val_innings = load_innings(val_year, data_root=data_root)

    logger.info("Train innings: %d games", len(train_innings["game_id"].unique()) if not train_innings.empty else 0)
    logger.info("Val innings:   %d games", len(val_innings["game_id"].unique()) if not val_innings.empty else 0)

    # Build pregame feature lookup keyed by game_id from season-aggregate stats
    def _build_pregame(years: list[int]) -> pd.DataFrame | None:
        try:
            from src.metrics.traditional import enrich_pitching, enrich_batting
            from src.metrics.advanced import enrich_pitching_advanced, enrich_batting_advanced
            from src.features.people import enrich_with_people
            frames = []
            for yr in years:
                pit = load_table("pitching", yr, yr, data_root=data_root)
                bat = load_table("batting", yr, yr, data_root=data_root)
                ppl = load_table("people", yr, yr, data_root=data_root)
                if pit.empty or bat.empty:
                    continue
                for col in ["ERA", "AVG", "SLG", "OBP"]:
                    if col in pit.columns:
                        pit[col] = pd.to_numeric(pit[col], errors="coerce")
                    if col in bat.columns:
                        bat[col] = pd.to_numeric(bat[col], errors="coerce")
                enrich_pitching(pit)
                enrich_batting(bat)
                enrich_pitching_advanced(pit)
                if not ppl.empty:
                    pit = enrich_with_people(pit, ppl, year_col="yearID")
                # Team-level batting aggregates
                bat_team = bat.groupby("teamID").agg(
                    woba=("woba", "mean"), ops=("ops", "mean"),
                    k_pct=("k_pct", "mean"), bb_pct=("bb_pct", "mean"),
                ).reset_index()
                # Pitcher-level: keep top-IP starter per team
                pit["ip"] = pit["IPouts"] / 3.0
                pit_starters = pit.sort_values("ip", ascending=False).groupby("teamID").first().reset_index()
                merged = pit_starters.merge(bat_team, on="teamID", how="left", suffixes=("_p", "_b"))
                merged["year"] = yr
                frames.append(merged)
            return pd.concat(frames, ignore_index=True) if frames else None
        except Exception as e:
            logger.warning("Could not build pregame data: %s", e)
            return None

    train_pregame = _build_pregame(train_years)
    val_pregame = _build_pregame([val_year])
    logger.info("Train pregame rows: %s", len(train_pregame) if train_pregame is not None else 0)
    logger.info("Val pregame rows:   %s", len(val_pregame) if val_pregame is not None else 0)

    # Build envs
    train_env = VecKBOEnv(num_envs, year=train_years[-1] if train_years else 2022,
                          inning_data=train_innings if not train_innings.empty else None,
                          pregame_data=train_pregame,
                          data_root=data_root, seed=seed)
    val_env = VecKBOEnv(min(num_envs, 50), year=val_year,
                        inning_data=val_innings if not val_innings.empty else None,
                        pregame_data=val_pregame,
                        data_root=data_root, seed=seed + 9999)

    # Model
    actor_critic = ActorCritic(obs_dim=OBS_DIM).to(device)
    cfg = PPOConfig(num_steps_per_env=22)
    ppo = PPO(actor_critic, cfg, device)
    storage = RolloutStorage(cfg.num_steps_per_env, num_envs, OBS_DIM, device)

    metrics_log = []
    t0 = time.time()

    for update in range(1, num_updates + 1):
        last_value = collect_rollout(train_env, actor_critic, storage, torch, device)
        storage.compute_returns(last_value, cfg.gamma, cfg.lam)
        update_stats = ppo.update(storage)

        if update % 10 == 0:
            elapsed = time.time() - t0
            logger.info(
                "Update %d/%d | policy_loss=%.4f | value_loss=%.4f | entropy=%.4f | t=%.1fs",
                update, num_updates,
                update_stats["policy_loss"], update_stats["value_loss"],
                update_stats["entropy"], elapsed,
            )

        if update % checkpoint_every == 0:
            val_stats = evaluate(val_env, actor_critic, torch, device)
            logger.info("Val @ update %d: mean_ep_reward=%.4f", update, val_stats["mean_episode_reward"])
            update_stats.update(val_stats)

            ckpt_path = output_dir / f"checkpoint_{update:06d}.pt"
            torch.save({
                "update": update,
                "model_state": actor_critic.state_dict(),
                "optimizer_state": ppo.optimizer.state_dict(),
                "metrics": update_stats,
            }, ckpt_path)
            logger.info("Saved checkpoint → %s", ckpt_path)

        metrics_log.append({"update": update, **update_stats})

    # Save final metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    return actor_critic


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train KBO PPO agent")
    parser.add_argument("--train-years", nargs="+", type=int, default=[2019, 2020, 2021, 2022])
    parser.add_argument("--val-year", type=int, default=2023)
    parser.add_argument("--num-envs", type=int, default=500)
    parser.add_argument("--num-updates", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/run_001"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    train(
        train_years=args.train_years,
        val_year=args.val_year,
        num_envs=args.num_envs,
        num_updates=args.num_updates,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output,
        data_root=args.data_root,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
