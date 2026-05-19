"""
PPO (Proximal Policy Optimization) update loop for KBO inning prediction.

Adapted from rl-finance / rsl_rl:
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Value function loss with optional clipping
- Entropy bonus for exploration
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None


@dataclass
class PPOConfig:
    # Policy update
    clip_param: float = 0.2
    entropy_coef: float = 0.05
    value_loss_coef: float = 0.5
    max_grad_norm: float = 1.0

    # GAE
    gamma: float = 0.99
    lam: float = 0.95      # GAE lambda

    # Optimization
    learning_rate: float = 1e-4
    num_epochs: int = 5    # PPO update epochs per rollout
    num_mini_batches: int = 4

    # Rollout
    num_steps_per_env: int = 22   # max half-innings per KBO game episode


if _TORCH_AVAILABLE:
    class RolloutStorage:
        """Stores one rollout's worth of (obs, action, reward, value, log_prob)."""

        def __init__(self, num_steps: int, num_envs: int, obs_dim: int, device: str = "cpu"):
            self.num_steps = num_steps
            self.num_envs = num_envs
            self.device = device

            self.obs      = torch.zeros(num_steps, num_envs, obs_dim).to(device)
            self.actions  = torch.zeros(num_steps, num_envs, 1).to(device)
            self.rewards  = torch.zeros(num_steps, num_envs).to(device)
            self.values   = torch.zeros(num_steps + 1, num_envs).to(device)
            self.log_probs = torch.zeros(num_steps, num_envs).to(device)
            self.dones    = torch.zeros(num_steps, num_envs).to(device)

            self.step = 0

        def insert(self, obs, action, reward, value, log_prob, done):
            t = self.step
            self.obs[t].copy_(torch.as_tensor(obs, dtype=torch.float32))
            self.actions[t].copy_(torch.as_tensor(action, dtype=torch.float32))
            self.rewards[t].copy_(torch.as_tensor(reward, dtype=torch.float32))
            self.values[t].copy_(value.detach())
            self.log_probs[t].copy_(log_prob.detach())
            self.dones[t].copy_(torch.as_tensor(done, dtype=torch.float32))
            self.step = (self.step + 1) % self.num_steps

        def compute_returns(self, last_value: "torch.Tensor", gamma: float, lam: float):
            """Compute GAE advantages and returns in-place."""
            self.values[-1].copy_(last_value.detach())
            advantages = torch.zeros_like(self.rewards)
            gae = torch.zeros(self.num_envs, device=self.device)

            for t in reversed(range(self.num_steps)):
                not_done = 1.0 - self.dones[t]
                delta = self.rewards[t] + gamma * self.values[t + 1] * not_done - self.values[t]
                gae = delta + gamma * lam * not_done * gae
                advantages[t] = gae

            self.returns = advantages + self.values[:-1]
            self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        def mini_batch_generator(self, num_mini_batches: int):
            """Yield randomized mini-batches over the rollout."""
            batch_size = self.num_steps * self.num_envs
            mini_batch_size = batch_size // num_mini_batches
            indices = torch.randperm(batch_size, device=self.device)

            obs_flat      = self.obs.reshape(batch_size, -1)
            actions_flat  = self.actions.reshape(batch_size, -1)
            log_probs_flat = self.log_probs.reshape(batch_size)
            returns_flat  = self.returns.reshape(batch_size)
            adv_flat      = self.advantages.reshape(batch_size)

            for start in range(0, batch_size, mini_batch_size):
                idx = indices[start: start + mini_batch_size]
                yield (
                    obs_flat[idx],
                    actions_flat[idx],
                    log_probs_flat[idx],
                    returns_flat[idx],
                    adv_flat[idx],
                )


    class PPO:
        """PPO trainer for KBOInningEnv."""

        def __init__(self, actor_critic, config: PPOConfig = None, device: str = "cpu"):
            if not _TORCH_AVAILABLE:
                raise ImportError("torch is required. pip install torch")
            self.ac = actor_critic
            self.cfg = config or PPOConfig()
            self.device = device
            self.optimizer = optim.Adam(actor_critic.parameters(), lr=self.cfg.learning_rate)

        def update(self, storage: "RolloutStorage"):
            """Run PPO update epochs on the collected rollout."""
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            n_updates = 0

            for _ in range(self.cfg.num_epochs):
                for obs, actions, old_log_probs, returns, advantages in \
                        storage.mini_batch_generator(self.cfg.num_mini_batches):

                    log_probs, entropy, values = self.ac.evaluate(obs, actions)

                    # Policy loss (clipped surrogate)
                    ratio = (log_probs - old_log_probs).exp()
                    surr1 = ratio * advantages
                    surr2 = ratio.clamp(1 - self.cfg.clip_param,
                                        1 + self.cfg.clip_param) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = nn.functional.mse_loss(values, returns)

                    loss = (policy_loss
                            + self.cfg.value_loss_coef * value_loss
                            - self.cfg.entropy_coef * entropy.mean())

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()

                    # Clamp actor output bias to prevent softplus collapse
                    with torch.no_grad():
                        out_layer = list(self.ac.actor.modules())[-1]
                        if hasattr(out_layer, 'bias') and out_layer.bias is not None:
                            out_layer.bias.clamp_(-2.0, 4.0)

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    n_updates += 1

            return {
                "policy_loss": total_policy_loss / max(n_updates, 1),
                "value_loss":  total_value_loss / max(n_updates, 1),
                "entropy":     total_entropy / max(n_updates, 1),
            }

else:
    class RolloutStorage:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required. pip install torch")

    class PPO:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required. pip install torch")
