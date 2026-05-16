"""
MLP Actor-Critic network for KBO per-inning run prediction.

Adapted from rl-finance / rsl_rl architecture:
- Shared trunk option or separate actor/critic MLPs
- Asymmetric actor-critic: actor sees LIVE_FEATURES only;
  critic sees LIVE_FEATURES + optional privileged features (e.g. actual outcome)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None

from src.rl.env import OBS_DIM


def _mlp(input_dim: int, hidden_dims: list[int], output_dim: int,
          activation=None) -> "nn.Sequential":
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for the RL agent. pip install torch")
    if activation is None:
        activation = nn.ELU

    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), activation()]
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


if _TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """Asymmetric actor-critic for KBO inning prediction.

        Actor:  obs (OBS_DIM) → action mean (1,)
        Critic: obs + privileged_obs → value (1,)

        Action is sampled from N(mean, std); std is a learned parameter.
        """

        def __init__(
            self,
            obs_dim: int = OBS_DIM,
            privileged_obs_dim: int = 0,
            actor_hidden: list[int] = None,
            critic_hidden: list[int] = None,
            action_dim: int = 1,
            init_noise_std: float = 1.0,
        ):
            super().__init__()
            actor_hidden = actor_hidden or [256, 128, 64]
            critic_hidden = critic_hidden or [256, 128, 64]

            self.actor = _mlp(obs_dim, actor_hidden, action_dim)
            critic_input = obs_dim + privileged_obs_dim
            self.critic = _mlp(critic_input, critic_hidden, 1)

            # Learned log-std (clipped during forward)
            self.log_std = nn.Parameter(
                torch.ones(action_dim) * np.log(init_noise_std)
            )

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
            # Output layer: small weights, bias set so softplus(bias) ≈ 0.6 (league mean)
            # softplus(x) = 0.6 → x ≈ 0.375
            out_layer = list(self.actor.modules())[-1]
            nn.init.orthogonal_(out_layer.weight, gain=0.01)
            nn.init.constant_(out_layer.bias, 0.375)

        def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
            """Return action mean (deterministic, non-negative via softplus)."""
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            return nn.functional.softplus(self.actor(obs))

        def act(self, obs: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Sample action and return (action, log_prob)."""
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            mean = nn.functional.softplus(self.actor(obs))
            std = self.log_std.exp().clamp(1e-3, 5.0)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = action.clamp(0.0, 10.0)
            log_prob = dist.log_prob(action).sum(-1)
            return action, log_prob

        def evaluate(
            self,
            obs: "torch.Tensor",
            actions: "torch.Tensor",
            privileged_obs: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """Return (log_prob, entropy, value) for PPO update."""
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            mean = nn.functional.softplus(self.actor(obs))
            mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=0.0)
            log_std = torch.nan_to_num(self.log_std, nan=0.0).clamp(-4.0, 2.0)
            std = log_std.exp().clamp(1e-3, 5.0)
            dist = torch.distributions.Normal(mean, std)

            log_prob = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)

            critic_input = obs if privileged_obs is None else torch.cat([obs, privileged_obs], dim=-1)
            value = self.critic(critic_input).squeeze(-1)

            return log_prob, entropy, value

        def get_value(
            self,
            obs: "torch.Tensor",
            privileged_obs: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            critic_input = obs if privileged_obs is None else torch.cat([obs, privileged_obs], dim=-1)
            return self.critic(critic_input).squeeze(-1)

else:
    class ActorCritic:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for ActorCritic. pip install torch")
